#include <ftn_solo_control/utils/trajectory_planner.h>
// Common includes
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <yaml-cpp/yaml.h>

#ifndef SKIP_PUBLISH_MARKERS
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// ftn_solo_control includes
#include <ftn_solo_control/utils/conversions.h>
#include <ftn_solo_control/utils/visualization_utils.h>
#endif

// ftn_solo_control includes
#include <ftn_solo_control/motions/com_motion.h>
#include <ftn_solo_control/motions/eef_position_motion.h>
#include <ftn_solo_control/motions/eef_rotation_motion.h>
#include <ftn_solo_control/motions/motion.h>
#include <ftn_solo_control/motions/solver.h>
#include <ftn_solo_control/utils/config_utils.h>
#include <ftn_solo_control/utils/wcm.h>

namespace ftn_solo_control {

namespace {

#ifndef SKIP_PUBLISH_MARKERS
static rclcpp::Node::SharedPtr trajectory_planner_node;
static rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    marker_publisher;
static rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
    joint_state_publisher;
static std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

void PublishIK(double t, ConstRefVectorXd q,
               std::vector<std::string> &joint_names) {
  sensor_msgs::msg::JointState joint_state;
  joint_state.header.stamp.sec = std::floor(t);
  joint_state.header.stamp.nanosec = std::floor((t - std::floor(t)) * 1e9);
  size_t num_joints = q.size() - 7;
  joint_state.position.resize(num_joints);
  joint_state.name = std::vector<std::string>(joint_names.end() - num_joints,
                                              joint_names.end());
  Eigen::VectorXd::Map(&joint_state.position[0], num_joints) =
      q.tail(num_joints);
  joint_state_publisher->publish(joint_state);

  auto world_T_base = geometry_msgs::msg::TransformStamped();
  world_T_base.header.stamp = joint_state.header.stamp;
  world_T_base.header.frame_id = "world";
  world_T_base.child_frame_id = "ik/base_link";
  world_T_base.transform.translation.x = q(0);
  world_T_base.transform.translation.y = q(1);
  world_T_base.transform.translation.z = q(2);
  world_T_base.transform.rotation.x = q(3);
  world_T_base.transform.rotation.y = q(4);
  world_T_base.transform.rotation.z = q(5);
  world_T_base.transform.rotation.w = q(6);
  tf_broadcaster->sendTransform(world_T_base);
}

void PublishWCM(const ConvexHull2D &wcm, ConstRefVector2d com,
                const std_msgs::msg::ColorRGBA &color,
                const std::string ns = "", double height = 0.001) {
  if (wcm.points_.size() == 0) {
    return;
  }
  visualization_msgs::msg::MarkerArray markers;
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "world";
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.color = color;
  marker.scale = ToVector(Eigen::Vector3d::Constant(0.005));
  marker.id = 215;
  marker.ns = "boundary";
  if (!ns.empty()) {
    marker.ns = ns + "_" + marker.ns;
  }
  marker.pose = ToPose(pinocchio::SE3::Identity());
  for (const auto &point : wcm.points_) {
    marker.points.push_back(
        ToPoint(Eigen::Vector3d(point.x(), point.y(), height)));
  }
  marker.points.push_back(ToPoint(Eigen::Vector3d(
      wcm.points_.front().x(), wcm.points_.front().y(), height)));
  markers.markers.push_back(marker);

  marker.type = visualization_msgs::msg::Marker::SPHERE;
  ++marker.id;
  marker.pose.position = ToPoint(Eigen::Vector3d(com[0], com[1], height));
  marker.scale = ToVector(Eigen::Vector3d::Constant(0.02));
  marker.color = MakeColor(1.0, 1.0, 0.0);
  marker.ns = "com";
  if (!ns.empty()) {
    marker.ns = ns + "_ " + marker.ns;
  }
  markers.markers.push_back(marker);
  marker_publisher->publish(markers);
}

#endif

} // namespace

void InitTrajectoryPlannerPublisher() {
#ifndef SKIP_PUBLISH_MARKERS
  trajectory_planner_node =
      std::make_shared<rclcpp::Node>("trajectory_planner_node");
  marker_publisher =
      trajectory_planner_node
          ->create_publisher<visualization_msgs::msg::MarkerArray>("markers",
                                                                   10);
  joint_state_publisher =
      trajectory_planner_node->create_publisher<sensor_msgs::msg::JointState>(
          "ik/joint_states", 10);
  tf_broadcaster =
      std::make_unique<tf2_ros::TransformBroadcaster>(trajectory_planner_node);
#endif
}

TrajectoryPlanner::TrajectoryPlanner(const pinocchio::Model &model,
                                     size_t base_index,
                                     const pinocchio::SE3 &origin,
                                     const std::string &config)
    : model_(model), data_(pinocchio::Data(model_)), base_index_(base_index),
      origin_(origin), computation_started_(false), computation_done_(false),
      update_started_(false), update_done_(false) {
  YAML::Node config_node = YAML::Load(config);
  auto tmp_config = config_node["COM"];
  if (tmp_config) {
    READ_DOUBLE_CONFIG(tmp_config, Kp, config_.COM.Kp);
    READ_DOUBLE_CONFIG(tmp_config, Kd, config_.COM.Kd);
  }
  tmp_config = config_node["base_linear"];
  if (tmp_config) {
    READ_DOUBLE_CONFIG(tmp_config, Kp, config_.base_linear.Kp);
    READ_DOUBLE_CONFIG(tmp_config, Kd, config_.base_linear.Kd);
  }
  tmp_config = config_node["base_angular"];
  if (tmp_config) {
    READ_DOUBLE_CONFIG(tmp_config, Kp, config_.base_angular.Kp);
    READ_DOUBLE_CONFIG(tmp_config, Kd, config_.base_angular.Kd);
  }
}

TrajectoryPlanner::TrajectoryPlanner(const TrajectoryPlanner &other)
    : motions_(other.motions_), model_(other.model_), data_(other.data_),
      base_index_(other.base_index_), origin_(other.origin_),
      config_(other.config_) {}

TrajectoryPlanner::~TrajectoryPlanner() {
  if (thread_.joinable()) {
    thread_.join();
  }
}
void TrajectoryPlanner::StartComputation(
    double t, const FrictionConeMap &friction_cones,
    const FrictionConeMap &next_friction_cones, ConstRefVectorXd q,
    double duration, double torso_height, double max_torque) {
  q_ = q;
  thread_ =
      std::thread(&TrajectoryPlanner::DoComputation, this, t, friction_cones,
                  next_friction_cones, duration, torso_height, max_torque);
}

void TrajectoryPlanner::DoComputation(double t, FrictionConeMap friction_cones,
                                      FrictionConeMap next_friction_cones,
                                      double duration, double torso_height,
                                      double max_torque) {
  computation_started_ = true;
  pinocchio::framesForwardKinematics(model_, data_, q_);
  pinocchio::computeJointJacobians(model_, data_, q_);
  pinocchio::computeGeneralizedGravity(model_, data_, q_);
  pinocchio::centerOfMass(model_, data_, q_, false);
  const Eigen::Vector3d com_pos = data_.com[0];
  const pinocchio::SE3 base_pose = data_.oMf[base_index_];
  auto next_wcm =
      GetProjectedWCMWithTorque(model_, data_, next_friction_cones, max_torque);
  com_xy_ = ComputeCoMPos(next_wcm, origin_.translation().head<2>());

#ifndef SKIP_PUBLISH_MARKERS
  PublishWCM(next_wcm, com_pos.head<2>(), MakeColor(0.0, 1.0, 1.0, 0.5), "wcm");
#endif
  com_trajectory_ = boost::make_shared<PieceWiseLinearPosition>();
  Eigen::VectorXd tmp_pos = com_pos.head<2>();
  com_trajectory_->AddPoint(tmp_pos, 0);
  com_trajectory_->AddPoint(com_xy_, duration);
  boost::shared_ptr<COMMotion> com_motion = boost::make_shared<COMMotion>(
      (Vector3b() << true, true, false).finished(), pinocchio::SE3::Identity(),
      config_.COM.Kp, config_.COM.Kd);
  com_motion->SetTrajectory(com_trajectory_);

  base_trajectory_ = boost::make_shared<PieceWiseLinearPosition>();
  base_trajectory_->AddPoint(origin_.actInv(base_pose).translation().tail<1>(),
                             0);
  tmp_pos = Eigen::VectorXd::Constant(1, torso_height);
  base_trajectory_->AddPoint(tmp_pos, duration);
  boost::shared_ptr<EEFPositionMotion> base_linear_motion =
      boost::make_shared<EEFPositionMotion>(
          base_index_, (Vector3b() << false, false, true).finished(), origin_,
          config_.base_linear.Kp, config_.base_linear.Kd);
  base_linear_motion->SetTrajectory(base_trajectory_);
  base_linear_motion->SetPriority(1, 1.0);
  rotation_trajectory_ = boost::make_shared<PieceWiseLinearRotation>();
  auto tmp_orientation = base_pose.rotation();
  rotation_trajectory_->AddPoint(tmp_orientation, 0);
  tmp_orientation = origin_.rotation();
  rotation_trajectory_->AddPoint(tmp_orientation, duration);
  boost::shared_ptr<EEFRotationMotion> base_angular_motion =
      boost::make_shared<EEFRotationMotion>(
          base_index_, config_.base_angular.Kp, config_.base_angular.Kd);
  base_angular_motion->SetTrajectory(rotation_trajectory_);
  base_angular_motion->SetPriority(1, 0.5);

  std::vector<boost::shared_ptr<Motion>> motions = {
      com_motion, base_linear_motion, base_angular_motion};
  Eigen::Vector3d new_com;
  while (true) {
    GetEndOfMotionPrioritized(model_, data_, friction_cones, motions, q_, true);
    pinocchio::computeGeneralizedGravity(model_, data_, q_);
    next_wcm = GetProjectedWCMWithTorque(model_, data_, next_friction_cones,
                                         max_torque);
    new_com = data_.com[0];
    new_com(2) = 1;
    if (((next_wcm.Equations() * new_com).array() > (0.025 - 1e-4)).all()) {
      break;
    }
    com_xy_ = ComputeCoMPos(next_wcm, origin_.translation().head<2>());
    com_trajectory_->PopPoint();
    com_trajectory_->AddPoint(com_xy_, duration);
  }
#ifndef SKIP_PUBLISH_MARKERS
  PublishIK(t, q_, model_.names);
  PublishWCM(next_wcm, new_com.head<2>(), MakeColor(1.0, 0.0, 0.5, 0.5), "wct");
#endif

  constexpr const size_t num_phis = 32;
  ConvexHull2D wcm_with_acceleration_bound;

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  for (size_t i = 0; i < num_phis; ++i) {
    Eigen::Vector3d acom = 2 * Eigen::Vector3d(cos(i * 2.0 * M_PI / num_phis),
                                               sin(i * 2.0 * M_PI / num_phis), 0.0);
    const auto wcm = GetProjectedWCMWithTorque(
        model_, data_, next_friction_cones, max_torque, acom);
    if (i == 0) {
      wcm_with_acceleration_bound = wcm;
    } else {
      const auto tmp = Intersect(wcm_with_acceleration_bound, wcm);
      wcm_with_acceleration_bound = tmp;
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "TTTTTTTTTTTTTTTTTTTime taken: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " milliseconds" << std::endl;

  PublishWCM(wcm_with_acceleration_bound, com_pos.head<2>(),
             MakeColor(1.0, 1.0, 0.0, 1.0), "wcm_with_acceleration_bound",
             0.005);

  ConvexHull2D wcm_with_hull_constraint;
  start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < num_phis; ++i) {
    Eigen::Vector3d acom = 2 * Eigen::Vector3d(cos(i * 2.0 * M_PI / num_phis),
                                               sin(i * 2.0 * M_PI / num_phis), 0.0);
    if (i == 0) {
      wcm_with_hull_constraint = GetProjectedWCMWithTorque(
          model_, data_, next_friction_cones, max_torque, acom);
    } else {
      wcm_with_hull_constraint =
          GetProjectedWCMWithTorque(model_, data_, next_friction_cones,
                                    max_torque, acom, wcm_with_hull_constraint);
    }
  }


  end = std::chrono::steady_clock::now();
  std::cout << "TTTTTTTTTTTTTTTime taken: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " milliseconds" << std::endl;

  PublishWCM(wcm_with_hull_constraint, com_pos.head<2>(),
             MakeColor(1.0, 0.0, 1.0, 1.0), "wcm_with_hull_constraint", 0.0075);

  // rclcpp::sleep_for(std::chrono::milliseconds(50));
  com_trajectory_->PopPoint();
  com_trajectory_->AddPoint(data_.com[0].head<2>(), duration);
  const auto &base_pose_goal = data_.oMf[base_index_];
  base_trajectory_->PopPoint();
  base_trajectory_->AddPoint(
      origin_.actInv(base_pose_goal).translation().tail<1>(), duration);
  rotation_trajectory_->PopPoint();
  tmp_orientation = base_pose_goal.rotation();
  rotation_trajectory_->AddPoint(tmp_orientation, duration);
  motions_ = {com_motion, base_linear_motion, base_angular_motion};
  computation_done_ = true;
}

Eigen::Vector2d TrajectoryPlanner::ComputeCoMPos(const ConvexHull2D &wcm,
                                                 ConstRefVector2d com_pos) {
  proxsuite::proxqp::dense::QP<double> qp(
      2, 0, wcm.Equations().rows(), false,
      proxsuite::proxqp::HessianType::Diagonal);
  qp.init(Eigen::Matrix2d::Identity(), -com_pos, proxsuite::nullopt,
          proxsuite::nullopt, wcm.Equations().leftCols<2>(),
          Eigen::VectorXd::Constant(qp.model.n_in, 0.05) -
              wcm.Equations().rightCols<1>(),
          Eigen::VectorXd::Constant(qp.model.n_in, 1e10));
  qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START;
  qp.solve(wcm.Centroid(), proxsuite::nullopt, proxsuite::nullopt);
  return qp.results.x;
}

void TrajectoryPlanner::UpdateEEFTrajectory(
    double t, const FrictionConeMap &friction_cones,
    std::vector<boost::shared_ptr<Motion>> eef_motions,
    std::vector<boost::shared_ptr<Motion>> joint_motions, ConstRefVectorXd q,
    double duration) {
  q_ = q;
  if (thread_.joinable()) {
    thread_.join();
  }
  thread_ = std::thread(&TrajectoryPlanner::DoUpdate, this, t, friction_cones,
                        eef_motions, joint_motions, duration);
}

void TrajectoryPlanner::DoUpdate(
    double t, FrictionConeMap friction_cones,
    std::vector<boost::shared_ptr<Motion>> eef_motions,
    std::vector<boost::shared_ptr<Motion>> joint_motions, double duration) {
  update_started_ = true;
  pinocchio::framesForwardKinematics(model_, data_, q_);
  const Eigen::Vector3d com_pos = data_.com[0];
  const pinocchio::SE3 base_pose = data_.oMf[base_index_];
  motions_.insert(motions_.end(), eef_motions.begin(), eef_motions.end());
  motions_.insert(motions_.end(), joint_motions.begin(), joint_motions.end());
  GetEndOfMotionPrioritized(model_, data_, friction_cones, motions_, q_, true);
#ifndef SKIP_PUBLISH_MARKERS
  PublishIK(t, q_, model_.names);
#endif
  com_trajectory_->Reset();
  Eigen::VectorXd tmp_pos = com_pos.head<2>();
  com_trajectory_->AddPoint(tmp_pos, 0);
  com_trajectory_->AddPoint(data_.com[0].head<2>(), duration);
  const pinocchio::SE3 base_pose_goal = data_.oMf[base_index_];
  base_trajectory_->Reset();
  base_trajectory_->AddPoint(origin_.actInv(base_pose).translation().tail<1>(),
                             0);
  base_trajectory_->AddPoint(
      origin_.actInv(base_pose_goal).translation().tail<1>(), duration);
  auto tmp_orientation = base_pose.rotation();
  rotation_trajectory_->Reset();
  rotation_trajectory_->AddPoint(tmp_orientation, 0);
  tmp_orientation = base_pose_goal.rotation();
  rotation_trajectory_->AddPoint(tmp_orientation, duration);
  update_done_ = true;
}

} // namespace ftn_solo_control
