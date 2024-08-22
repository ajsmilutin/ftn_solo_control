#include <ftn_solo_control/estimators.h>
// Common includes
#include <Eigen/Dense>
#include <ftn_solo_control/types/common.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <math.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// ftn solo includes
#include <ftn_solo_control/utils/conversions.h>
#include <ftn_solo_control/utils/utils.h>

namespace ftn_solo_control {

namespace {
static rclcpp::Node::SharedPtr estimator_node;
static rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    publisher;

static rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr
    pose_publisher;

visualization_msgs::msg::Marker GetPointMarker(ConstRefVector3d position,
                                               int id) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "world";
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.color.r = 1;
  marker.color.a = marker.color.g = 0.5;

  marker.pose.position = ToPoint(position);
  marker.scale.x = marker.scale.y = marker.scale.z = 0.01;
  marker.ns = "contact";
  marker.id = id;
  return marker;
}

void PublishMarkers(
    const std::unordered_map<size_t, pinocchio::SE3> &poses,
    const std::unordered_map<size_t, pinocchio::SE3> &touching_poses,
    const pinocchio::Data &data) {
  visualization_msgs::msg::MarkerArray all_markers;
  int i = 0;
  geometry_msgs::msg::PoseArray poses_msg;
  poses_msg.header.frame_id = "world";
  for (const auto &position : poses) {
    all_markers.markers.push_back(
        GetPointMarker(position.second.translation(), 300 + (++i)));
    const auto &touching_pose = touching_poses.at(position.first);
    poses_msg.poses.push_back(ToPose(pinocchio::SE3(
        touching_pose.rotation(),
        data.oMf.at(position.first).translation() + touching_pose.translation())));
  }
  if (pose_publisher) {

    pose_publisher->publish(poses_msg);
  }
  if (publisher) {
    publisher->publish(all_markers);
  }
}

} // namespace
void InitEstimatorPublisher() {
  estimator_node = std::make_shared<rclcpp::Node>("estimator_node");
  publisher =
      estimator_node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "estimator_markers", 10);
  pose_publisher =
      estimator_node->create_publisher<geometry_msgs::msg::PoseArray>(
          "estimator_poses", 10);
}

FixedPointsEstimator::FixedPointsEstimator(double dt,
                                           const pinocchio::Model &model,
                                           pinocchio::Data &data,
                                           const std::vector<size_t> &indexes)
    : dt_(dt), model_(model), data_(data), indexes_(indexes) {
  num_joints_ = model_.nv - 6;
  estimated_q_ = Eigen::VectorXd::Zero(model_.nq);
  estimated_qv_ = Eigen::VectorXd::Zero(model_.nv);
  rclcpp::Node node = rclcpp::Node("my_node");
}

void FixedPointsEstimator::SetData(ConstRefVectorXd q, ConstRefVectorXd qv,
                                   const SensorData &sensors) {
  estimated_q_.segment<3>(3) = sensors.imu_data.attitude.segment<3>(1);
  estimated_q_(6) = sensors.imu_data.attitude(0);
  estimated_q_.tail(num_joints_) = q;
  estimated_qv_.tail(num_joints_) = qv;
}

void FixedPointsEstimator::Init(ConstRefVectorXd q, ConstRefVectorXd qv,
                                const SensorData &sensors) {
  estimated_q_(2) = 0.0;
  SetData(q, qv, sensors);
  pinocchio::framesForwardKinematics(model_, data_, estimated_q_);
  Eigen::Vector3d mean;
  for (size_t i = 0; i < indexes_.size(); ++i) {
    const auto pose =
        GetTouchingPose(model_, data_, indexes_.at(i), Eigen::Vector3d::UnitZ());
    poses_.emplace(indexes_.at(i),
                   pinocchio::SE3(Eigen::Matrix3d::Identity(),
                                  data_.oMf.at(indexes_.at(i)).translation() +
                                      pose.translation()));
    mean += poses_.at(indexes_.at(i)).translation();
  }
  mean = mean / static_cast<double>(indexes_.size());
  estimated_q_(2) = -mean(2);
  for (auto &position : poses_) {
    position.second.translation()(2) -= mean(2);
  }
  Eigen::Vector3d x = (poses_.at(indexes_.at(0)).translation() +
                       poses_.at(indexes_.at(1)).translation()) -
                      (poses_.at(indexes_.at(2)).translation() +
                       poses_.at(indexes_.at(3)).translation());
  Eigen::Vector3d y = (poses_.at(indexes_.at(0)).translation() +
                       poses_.at(indexes_.at(2)).translation()) -
                      (poses_.at(indexes_.at(1)).translation() +
                       poses_.at(indexes_.at(3)).translation());
  Eigen::Matrix3d rot;
  rot.col(2) = x.cross(y).normalized();
  rot.col(0) = y.cross(rot.col(2)).normalized();
  rot.col(1) = rot.col(2).cross(rot.col(1));
  estimated_q_.segment<4>(3) = Eigen::Quaterniond(rot).inverse().coeffs();
}

bool FixedPointsEstimator::UnFix(size_t index) {
  auto it = poses_.find(index);
  if (it == poses_.end()) {
    return false;
  } else {
    poses_.erase(it);
    return true;
  }
}

void FixedPointsEstimator::SetFixed(size_t frame_index,
                                    ConstRefMatrix3d new_orientation) {
  pinocchio::SE3 new_pose(new_orientation,
                          data_.oMf.at(frame_index).translation());
  const auto pose =
      GetTouchingPose(model_, data_, frame_index, new_orientation.col(2));
  new_pose.translation() += pose.translation();
  new_pose.translation().z() -= 0.000;
  poses_[frame_index] = new_pose;
}

void FixedPointsEstimator::Estimate(double t, ConstRefVectorXd q,
                                    ConstRefVectorXd qv,
                                    const SensorData &sensors) {
  SetData(q, qv, sensors);
  double grad = 1;
  size_t num_constraints = poses_.size() * 3;
  constraint_ = Eigen::MatrixXd::Zero(num_constraints, model_.nv);
  acceleration_ = velocity_ = Eigen::VectorXd::Zero(num_constraints);
  Eigen::VectorXd errors = Eigen::VectorXd::Ones(num_constraints);
  Eigen::VectorXd step = Eigen::VectorXd::Zero(model_.nv);
  pinocchio::framesForwardKinematics(model_, data_, estimated_q_);
  pinocchio::computeJointJacobians(model_, data_, estimated_q_);
  std::unordered_map<size_t, pinocchio::SE3> placements;
  std::unordered_map<size_t, pinocchio::SE3> touching_poses;
  while (grad > 1e-6 && errors.norm() > 0.5e-5) {
    size_t i = 0;
    for (const auto &position : poses_) {
      touching_poses[position.first] = GetTouchingPose(
          model_, data_, position.first, position.second.rotation().col(2));
      errors.segment<3>(i * 3) = position.second.translation() -
                                 (data_.oMf.at(position.first).translation() +
                                  touching_poses.at(position.first).translation());
      const auto &frame = model_.frames.at(position.first);
      placements[position.first] = GetTouchingPlacement(
          model_, data_, position.first, touching_poses.at(position.first));
      constraint_.middleRows<3>(i * 3) =
          pinocchio::getFrameJacobian(
              model_, data_, frame.parentJoint, placements.at(position.first),
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
              .topRows<3>();
      ++i;
    }
    step.head<6>() = constraint_.leftCols<6>().householderQr().solve(errors);
    grad = step.head<6>().norm();
    pinocchio::integrate(model_, estimated_q_, 0.75 * step, estimated_q_);
    pinocchio::framesForwardKinematics(model_, data_, estimated_q_);
    pinocchio::computeJointJacobians(model_, data_, estimated_q_);
  }
  PublishMarkers(poses_, touching_poses, data_);
  EstimateVelocities(sensors);
  UpdateInternals(touching_poses, placements);
}

void FixedPointsEstimator::EstimateVelocities(const SensorData &sensors) {
  const double alpha = 10;
  size_t num_constraints = poses_.size() * 3;
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(num_constraints + 3, model_.nv);
  J.block<3, 3>(0, 3) =
      alpha * Eigen::Quaterniond(estimated_q_[6], estimated_q_[3],
                                 estimated_q_[4], estimated_q_[5])
                  .matrix()
                  .transpose();
  J.bottomRows(num_constraints) = constraint_;

  Eigen::VectorXd b(J.rows());
  b.head<3>() = alpha * sensors.imu_data.angular_velocity;
  b.tail(num_constraints) =
      -J.bottomRightCorner(num_constraints, model_.nv - 6) *
      estimated_qv_.tail(num_joints_);
  estimated_qv_.head<6>() = J.leftCols<6>().householderQr().solve(b);
}

void FixedPointsEstimator::UpdateInternals(
    const std::unordered_map<size_t, pinocchio::SE3> &touching_poses,
    const std::unordered_map<size_t, pinocchio::SE3> &placements) {
  pinocchio::forwardKinematics(model_, data_, estimated_q_, estimated_qv_);
  pinocchio::updateFramePlacements(model_, data_);
  size_t i = 0;
  for (auto &position : poses_) {
    const auto &frame = model_.frames.at(position.first);
    const Eigen::Vector3d center_velocity =
        pinocchio::getFrameVelocity(
            model_, data_, position.first,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
            .linear();
    const double vel =
        center_velocity.dot(touching_poses.at(position.first).rotation().col(0));
    position.second.translation() +=
        dt_ * vel * touching_poses.at(position.first).rotation().col(0);
    velocity_.segment<3>(3 * i) =
        pinocchio::getFrameVelocity(
            model_, data_, frame.parentJoint, placements.at(position.first),
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
            .linear();
    acceleration_.segment<3>(3 * i) =
        pinocchio::getFrameAcceleration(
            model_, data_, frame.parentJoint, placements.at(position.first),
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
            .linear();
    ++i;
  }
}

} // namespace ftn_solo_control
