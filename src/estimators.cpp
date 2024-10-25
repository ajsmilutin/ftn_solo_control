#include <ftn_solo_control/estimators.h>
// Common includes
#include <Eigen/Dense>
#include <ftn_solo_control/types/common.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <math.h>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_broadcaster.h>
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
static rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
    joint_state_publisher;
static std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

static size_t index = 0;
constexpr size_t publish_on = 12;

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

void PublishMarkers(const std::map<size_t, pinocchio::SE3> &poses,
                    const std::map<size_t, pinocchio::SE3> &touching_poses_,
                    const pinocchio::Data &data) {
  if ((++index) % 50 != publish_on) {
    return;
  }
  visualization_msgs::msg::MarkerArray all_markers;
  int i = 0;
  geometry_msgs::msg::PoseArray poses_msg;
  poses_msg.header.frame_id = "world";
  for (const auto &position : poses) {
    all_markers.markers.push_back(
        GetPointMarker(position.second.translation(), 300 + (++i)));
    const auto &touching_pose = touching_poses_.at(position.first);
    poses_msg.poses.push_back(ToPose(pinocchio::SE3(
        touching_pose.rotation(), data.oMf.at(position.first).translation() +
                                      touching_pose.translation())));
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
  joint_state_publisher =
      estimator_node->create_publisher<sensor_msgs::msg::JointState>(
          "joint_states", 10);
  tf_broadcaster =
      std::make_unique<tf2_ros::TransformBroadcaster>(estimator_node);
}

FixedPointsEstimator::FixedPointsEstimator(double dt,
                                           const pinocchio::Model &model,
                                           pinocchio::Data &data,
                                           const std::vector<size_t> &indexes)
    : dt_(dt), model_(model), data_(data), indexes_(indexes) {
  initialized_ = false;
  num_joints_ = model_.nv - 6;
  estimated_q_ = Eigen::VectorXd::Zero(model_.nq);
  estimated_qv_ = Eigen::VectorXd::Zero(model_.nv);
}

FixedPointsEstimator::FixedPointsEstimator(
    const ftn_solo_control::FixedPointsEstimator &other)
    : t_(other.t_), dt_(other.dt_),
      sensor_angular_velocity_(other.sensor_angular_velocity_),
      num_joints_(other.num_joints_), model_(other.model_), data_(other.data_),
      indexes_(other.indexes_), poses_(other.poses_),
      touching_poses_(other.touching_poses_), indexes_map_(other.indexes_map_),
      initialized_(other.Initialized()), estimated_q_(other.estimated_q_),
      estimated_qv_(other.estimated_qv_), constraint_(other.constraint_),
      eef_positions_(other.eef_positions_), velocity_(other.velocity_),
      acceleration_(other.acceleration_) {}

void FixedPointsEstimator::SetData(double t, ConstRefVectorXd q,
                                   ConstRefVectorXd qv,
                                   const SensorData &sensors) {
  t_ = t;
  Eigen::Quaterniond new_orientation(
      sensors.imu_data.attitude[0], sensors.imu_data.attitude[1],
      sensors.imu_data.attitude[2], sensors.imu_data.attitude[3]);
  measured_orientation_ = (initial_orientation_ * new_orientation).coeffs();
  estimated_q_.tail(num_joints_) = q;
  estimated_qv_.tail(num_joints_) = qv;
  sensor_angular_velocity_ = sensors.imu_data.angular_velocity;
}

void FixedPointsEstimator::Init(double t, ConstRefVectorXd q,
                                ConstRefVectorXd qv,
                                const SensorData &sensors) {
  const Eigen::Vector3d x_axis =
      Eigen::Quaterniond(
          sensors.imu_data.attitude[0], sensors.imu_data.attitude[1],
          sensors.imu_data.attitude[2], sensors.imu_data.attitude[3])
          .matrix()
          .col(0);
  const double phi = std::atan2(x_axis(1), x_axis(0));
  initial_orientation_ =
      Eigen::Quaterniond(Eigen::AngleAxisd(-phi, Eigen::Vector3d::UnitZ()));
  estimated_q_(2) = 0.0;
  estimated_q_(6) = 1.0;
  SetData(t, q, qv, sensors);
  thread_ = std::thread(&FixedPointsEstimator::InitAndEstimate, this);
}

void FixedPointsEstimator::InitAndEstimate() {
  pinocchio::framesForwardKinematics(model_, data_, estimated_q_);
  Eigen::Vector3d mean;
  for (size_t i = 0; i < indexes_.size(); ++i) {
    const auto pose = GetTouchingPose(model_, data_, indexes_.at(i),
                                      Eigen::Vector3d::UnitZ());
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
  UpdateIndexes();
  EstimateInternal();
  initialized_ = true;
}

void FixedPointsEstimator::UpdateIndexes() {
  indexes_.clear();
  for (const auto pose : poses_) {
    indexes_.push_back(pose.first);
  }
}

bool FixedPointsEstimator::UnFix(size_t index) {
  auto it = poses_.find(index);
  if (it == poses_.end()) {
    return false;
  } else {
    poses_.erase(it);
    UpdateIndexes();
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
  poses_[frame_index] = new_pose;
  UpdateIndexes();
}

void FixedPointsEstimator::Estimate(double t, ConstRefVectorXd q,
                                    ConstRefVectorXd qv,
                                    const SensorData &sensors) {
  std::chrono::time_point<std::chrono::system_clock> last =
      std::chrono::system_clock::now();
  SetData(t, q, qv, sensors);
  EstimateInternal();
}

void FixedPointsEstimator::EstimateInternal() {
  double grad = 1;
  size_t num_constraints = poses_.size() * 3;
  constraint_ = Eigen::MatrixXd::Zero(num_constraints, model_.nv);
  eef_positions_ = acceleration_ = velocity_ =
      Eigen::VectorXd::Zero(num_constraints);
  Eigen::VectorXd errors = Eigen::VectorXd::Ones(num_constraints);
  Eigen::VectorXd step = Eigen::VectorXd::Zero(model_.nv);
  pinocchio::framesForwardKinematics(model_, data_, estimated_q_);
  pinocchio::computeJointJacobians(model_, data_, estimated_q_);
  std::map<size_t, pinocchio::SE3> placements;
  touching_poses_ = {};

  Eigen::Quaterniond q0;
  Eigen::MatrixXd c2 = constraint_;
  while (grad > 1e-6 && errors.norm() / num_constraints > 1e-4) {
    size_t i = 0;
    GetConstraintJacobian(model_, data_, poses_, constraint_, &touching_poses_,
                          &placements);
    for (const auto &position : poses_) {
      errors.segment<3>(i * 3) =
          position.second.translation() -
          (data_.oMf.at(position.first).translation() +
           touching_poses_.at(position.first).translation());
      ++i;
    }
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(num_constraints + 2, 6);
    q0 = Eigen::Quaterniond(estimated_q_[6], estimated_q_[3], estimated_q_[4],
                            estimated_q_[5]);
    double alpha = 0.05;
    Eigen::Vector3d quat_err =
        alpha *
        (q0.toRotationMatrix() *
         pinocchio::log3(
             (q0.inverse() * measured_orientation_).toRotationMatrix()));
    J.block<2, 2>(0, 3) = alpha * Eigen::MatrixXd::Identity(2, 2);
    J.bottomRows(num_constraints) = constraint_.leftCols<6>() / num_constraints;
    Eigen::VectorXd vec(num_constraints + 2);
    vec.head<2>() = quat_err.head<2>();
    vec.tail(errors.size()) = errors / num_constraints;
    step.head<6>() = J.householderQr().solve(vec);
    grad = step.head<6>().norm();
    pinocchio::integrate(model_, estimated_q_, 0.5 * step, estimated_q_);
    pinocchio::framesForwardKinematics(model_, data_, estimated_q_);
    pinocchio::computeJointJacobians(model_, data_, estimated_q_);
  }
  GetConstraintJacobian(model_, data_, poses_, constraint_, &touching_poses_);
  PublishMarkers(poses_, touching_poses_, data_);
  EstimateVelocities();
  UpdateInternals(touching_poses_, placements);
}

void FixedPointsEstimator::PublishState(size_t seconds,
                                        size_t nanoseconds) const {
  sensor_msgs::msg::JointState joint_state;
  joint_state.header.stamp.sec = seconds;
  joint_state.header.stamp.nanosec = nanoseconds;
  size_t num_joints = estimated_q_.size() - 7;
  joint_state.position.resize(num_joints);
  joint_state.velocity.resize(num_joints);
  joint_state.effort.resize(num_joints);
  const auto &joint_names = model_.names;
  joint_state.name = std::vector<std::string>(joint_names.end() - num_joints,
                                              joint_names.end());
  Eigen::VectorXd::Map(&joint_state.position[0], num_joints) =
      estimated_q_.tail(num_joints);
  Eigen::VectorXd::Map(&joint_state.velocity[0], num_joints) =
      estimated_qv_.tail(num_joints);
  Eigen::VectorXd::Map(&joint_state.effort[0], num_joints) = effort_;
  joint_state_publisher->publish(joint_state);

  auto world_T_base = geometry_msgs::msg::TransformStamped();
  world_T_base.header.stamp = joint_state.header.stamp;
  world_T_base.header.frame_id = "world";
  world_T_base.child_frame_id = "base_link";
  world_T_base.transform.translation.x = estimated_q_(0);
  world_T_base.transform.translation.y = estimated_q_(1);
  world_T_base.transform.translation.z = estimated_q_(2);
  world_T_base.transform.rotation.x = estimated_q_(3);
  world_T_base.transform.rotation.y = estimated_q_(4);
  world_T_base.transform.rotation.z = estimated_q_(5);
  world_T_base.transform.rotation.w = estimated_q_(6);
  tf_broadcaster->sendTransform(world_T_base);
}

void FixedPointsEstimator::EstimateVelocities() {
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
  b.head<3>() = alpha * sensor_angular_velocity_;
  b.tail(num_constraints) =
      -J.bottomRightCorner(num_constraints, model_.nv - 6) *
      estimated_qv_.tail(num_joints_);
  estimated_qv_.head<6>() = J.leftCols<6>().householderQr().solve(b);
}

void FixedPointsEstimator::UpdateInternals(
    const std::map<size_t, pinocchio::SE3> &touching_poses_,
    const std::map<size_t, pinocchio::SE3> &placements) {
  pinocchio::forwardKinematics(model_, data_, estimated_q_, estimated_qv_,
                               0 * estimated_qv_);
  pinocchio::updateFramePlacements(model_, data_);
  pinocchio::centerOfMass(model_, data_, estimated_q_, estimated_qv_);
  pinocchio::crba(model_, data_, estimated_q_);
  pinocchio::nonLinearEffects(model_, data_, estimated_q_, estimated_qv_);

  size_t i = 0;
  for (auto &position : poses_) {
    const auto &frame = model_.frames.at(position.first);
    const Eigen::Vector3d center_velocity =
        pinocchio::getFrameVelocity(
            model_, data_, position.first,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
            .linear();
    const double vel = center_velocity.dot(
        touching_poses_.at(position.first).rotation().col(0));
    eef_positions_.segment<3>(3 * i) =
        (data_.oMf.at(position.first).translation() +
         touching_poses_.at(position.first).translation());
    position.second.translation() +=
        dt_ * vel * touching_poses_.at(position.first).rotation().col(0);
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

FrictionConeMap FixedPointsEstimator::GetFrictionCones(double mu,
                                                       size_t num_sides) {
  FrictionConeMap result;
  for (const auto &pose : poses_) {
    result.emplace(pose.first, FrictionCone(mu, num_sides, pose.second));
  }
  return result;
}

FrictionCone
FixedPointsEstimator::CreateFrictionCone(size_t frame_index,
                                         ConstRefMatrix3d new_orientation,
                                         double mu, size_t num_sides) const {
  pinocchio::SE3 new_pose(new_orientation,
                          data_.oMf.at(frame_index).translation());
  const auto pose =
      GetTouchingPose(model_, data_, frame_index, new_orientation.col(2));
  new_pose.translation() += pose.translation();
  return FrictionCone(mu, num_sides, new_pose);
}

} // namespace ftn_solo_control
