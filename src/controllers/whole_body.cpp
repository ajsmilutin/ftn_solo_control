#include <ftn_solo_control/controllers/whole_body.h>
// Common includes
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <yaml-cpp/yaml.h>
// FTN solo includes
#include <ftn_solo_control/utils/config_utils.h>
#include <ftn_solo_control/utils/conversions.h>
#include <ftn_solo_control/utils/utils.h>

namespace ftn_solo_control {
namespace {
size_t TotalSides(const FrictionConeMap &friction_cones) {
  size_t total = 0;
  for (const auto &cone : friction_cones) {
    total += cone.second.GetNumSides();
  }
  return total;
}

static rclcpp::Node::SharedPtr whole_body_node;
static rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    publisher;

static size_t index = 0;
constexpr size_t publish_on = 15;

static std::unordered_set<size_t> all_eefs;
} // namespace

void InitWholeBodyPublisher() {
  whole_body_node = std::make_shared<rclcpp::Node>("whole_body_node");
  publisher =
      whole_body_node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "force_markers", 10);
}

WholeBodyController::WholeBodyController(const FixedPointsEstimator &estimator,
                                         const FrictionConeMap &friction_cones,
                                         double max_torque,
                                         const std::string &config)
    : max_torque_(max_torque),
      qp_(estimator.NumJoints() + estimator.NumDoF() +
              estimator.NumContacts() * 3,
          estimator.NumDoF() + estimator.NumContacts() * 3,
          estimator.NumJoints() + TotalSides(friction_cones),
          false, proxsuite::proxqp::HessianType::Dense) {
  // Parse YAML configuration
  YAML::Node config_node = YAML::Load(config);
  READ_DOUBLE_CONFIG(config_node, lambda_tangential, config_.lambda_tangential);
  READ_DOUBLE_CONFIG(config_node, lambda_kd, config_.lambda_kd);
  READ_DOUBLE_CONFIG(config_node, lambda_torque, config_.lambda_torque);
  READ_DOUBLE_CONFIG(config_node, smooth, config_.smooth);
  READ_DOUBLE_CONFIG(config_node, kd, config_.kd);
  READ_DOUBLE_CONFIG(config_node, force_margin, config_.force_margin);
  config_.B = GetVectorFromConfig(estimator.NumJoints(), config_node, "B");
  config_.Fv = GetVectorFromConfig(estimator.NumJoints(), config_node, "Fv");
  config_.sigma =
      GetVectorFromConfig(estimator.NumJoints(), config_node, "sigma");

  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(qp_.model.n_in, qp_.model.dim);
  size_t start_row = 0;
  size_t start_col = estimator.NumDoF() + estimator.NumJoints();
  Eigen::VectorXd d = config_.force_margin * Eigen::VectorXd::Ones(qp_.model.n_in);
  H_ = 0.001 * Eigen::MatrixXd::Identity(qp_.model.dim, qp_.model.dim);
  for (const auto &cone : friction_cones) {
    eefs_.push_back(cone.first);
    forces_.emplace(cone.first, Eigen::Vector3d::Zero());
    tangential_.emplace(
        cone.first,
        cone.second.GetPose().rotation().leftCols<2>() *
            cone.second.GetPose().rotation().leftCols<2>().transpose());
    normal_.emplace(
        cone.first,
        cone.second.GetPose().rotation().rightCols<1>() *
            cone.second.GetPose().rotation().rightCols<1>().transpose());
    C.block(start_row, start_col, cone.second.GetNumSides(), 3) =
        cone.second.primal_.face_;
    start_row += cone.second.GetNumSides();
    // Penalize tangential forces
    H_.block<3, 3>(start_col, start_col) =
        config_.lambda_tangential * tangential_.at(cone.first);
    start_col += 3;
  }
  C.block(start_row, estimator.NumDoF(), estimator.NumJoints(),
          estimator.NumJoints()) =
      Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());
  d.tail(estimator.NumJoints()) =
      -max_torque_ * Eigen::VectorXd::Ones(estimator.NumJoints());
  Eigen::VectorXd u = 1e5 * Eigen::VectorXd::Ones(qp_.model.n_in);
  u.tail(estimator.NumJoints()) =
      max_torque_ * Eigen::VectorXd::Ones(estimator.NumJoints());
  qp_.init(proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt,
           proxsuite::nullopt, C, d, u);
}

Eigen::VectorXd WholeBodyController::Compute(
    double t, const pinocchio::Model &model, pinocchio::Data &data,
    FixedPointsEstimator &estimator,
    const std::vector<boost::shared_ptr<Motion>> &motions,
    ConstRefVectorXd old_torque, const double alpha, size_t new_contact,
    size_t ending_contact) {
  size_t motions_dim = GetMotionsDim(motions);
  Eigen::MatrixXd motions_jacobian =
      Eigen::MatrixXd(motions_dim, estimator.NumDoF());
  GetMotionsJacobian(model, data, estimator.estimated_q_,
                     estimator.estimated_qv_, motions, motions_jacobian);
  Eigen::VectorXd motions_ades = Eigen::VectorXd(motions_dim);
  size_t start_row = 0;
  for (const auto &motion : motions) {
    motions_ades.segment(start_row, motion->dim_) =
        motion->GetDesiredAcceleration(t, model, data, estimator.estimated_q_,
                                       estimator.estimated_qv_) -
        motion->GetAcceleration(model, data, estimator.estimated_q_,
                                estimator.estimated_qv_);
    start_row += motion->dim_;
  }

  size_t start_col = estimator.NumDoF() + estimator.NumJoints();
  const double eps_start = 1 - 1 / (1 + exp(-(alpha - 0.2) / 0.03));
  const double eps_end = 1 / (1 + exp(-(alpha - 0.8) / 0.03));
  for (const auto &eef : eefs_) {
    if (new_contact == eef) {
      H_.block<3, 3>(start_col, start_col) =
          config_.lambda_tangential * tangential_.at(eef) +
          config_.lambda_tangential * eps_start * normal_.at(eef);
    } else if (ending_contact == eef) {
      H_.block<3, 3>(start_col, start_col) =
          config_.lambda_tangential * tangential_.at(eef) +
          config_.lambda_tangential * eps_end * normal_.at(eef);
    }
    start_col += 3;
  }

  H_.topLeftCorner(estimator.NumDoF(), estimator.NumDoF()) =
      motions_jacobian.transpose() * motions_jacobian;
  H_.block(6, 6, estimator.NumJoints(), estimator.NumJoints()) +=
      config_.lambda_kd *
      Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());
  H_.block(estimator.NumDoF(), estimator.NumDoF(), estimator.NumJoints(),
           estimator.NumJoints()) =
      config_.lambda_torque *
      Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());
  Eigen::VectorXd g = Eigen::VectorXd::Zero(qp_.model.dim);
  g.head(estimator.NumDoF()) = -motions_jacobian.transpose() * motions_ades;
  g.segment(6, estimator.NumJoints()) +=
      config_.lambda_kd * config_.kd *
      estimator.velocity_.tail(estimator.NumJoints());
  g.segment(estimator.NumDoF(), estimator.NumJoints()) =
      -config_.lambda_torque * old_torque;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(qp_.model.n_eq, qp_.model.dim);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(qp_.model.n_eq);
  // Multi-body dynamics constraint
  A.topLeftCorner(estimator.NumDoF(), estimator.NumDoF()) = data.M;
  A.block(6, estimator.NumDoF(), estimator.NumJoints(), estimator.NumJoints()) =
      -Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());
  A.block(0, estimator.NumJoints() + estimator.NumDoF(), estimator.NumDoF(),
          estimator.NumContacts() * 3) = -estimator.constraint_.transpose();
  b.head(estimator.NumDoF()) = -data.nle;
  // Friction compensation
  Eigen::VectorXd qd_des = estimator.estimated_qv_.tail(estimator.NumJoints());
  Eigen::VectorXd friction =
      config_.B.array() * qd_des.array() +
      config_.Fv.array() * (qd_des.array() / config_.sigma.array()).atan() /
          M_PI_2;
  b.segment(6, estimator.NumJoints()) -= friction;
  // Constraint
  A.block(estimator.NumDoF(), 0, 3 * estimator.NumContacts(),
          estimator.NumDoF()) = estimator.constraint_;
  b.tail(3 * estimator.NumContacts()) = -estimator.acceleration_;
  qp_.update(H_, g, A, b, proxsuite::nullopt, proxsuite::nullopt,
             proxsuite::nullopt);
  qp_.solve();
  qp_.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  size_t start_force = estimator.NumDoF() + estimator.NumJoints();
  for (size_t i = 0; i < eefs_.size(); ++i) {
    forces_[eefs_.at(i)] = qp_.results.x.segment<3>(start_force + i * 3);
  }
  PublishForceMarker(estimator);
  estimator.SetEffort(
      qp_.results.x.segment(estimator.NumDoF(), estimator.NumJoints()));
  return config_.smooth *
             qp_.results.x.segment(estimator.NumDoF(), estimator.NumJoints()) +
         (1 - config_.smooth) * old_torque;
}

void WholeBodyController::PublishForceMarker(
    const FixedPointsEstimator &estimator) {
  if ((++index) % 50 != publish_on) {
    return;
  }
  visualization_msgs::msg::MarkerArray all_markers;
  visualization_msgs::msg::Marker single_arrow;
  single_arrow.header.frame_id = "world";
  single_arrow.ns = "force";
  single_arrow.action = visualization_msgs::msg::Marker::DELETEALL;
  all_markers.markers.push_back(single_arrow);
  single_arrow.action = visualization_msgs::msg::Marker::ADD;
  single_arrow.type = visualization_msgs::msg::Marker::ARROW;
  single_arrow.color.r = single_arrow.color.a = 1;
  single_arrow.color.g = 0.5;
  single_arrow.scale.x = 0.015;
  single_arrow.scale.y = 0.025;
  single_arrow.scale.z = 0.05;
  all_eefs.insert(eefs_.begin(), eefs_.end());
  for (const auto id : all_eefs) {
    single_arrow.id = id;
    single_arrow.ns = "force" + std::to_string(id);
    single_arrow.points.clear();
    const auto it = std::find(eefs_.begin(), eefs_.end(), id);
    if (it != eefs_.end()) {
      size_t i = std::distance(eefs_.begin(), it);
      single_arrow.points.push_back(
          ToPoint(estimator.eef_positions_.segment<3>(3 * i)));
      single_arrow.points.push_back(ToPoint(
          estimator.eef_positions_.segment<3>(3 * i) + 0.025 * GetForce(id)));
    } else {
      single_arrow.points.push_back(ToPoint(Eigen::Vector3d::Zero()));
      single_arrow.points.push_back(ToPoint(Eigen::Vector3d::Zero()));
    }
    all_markers.markers.push_back(single_arrow);
  }
  publisher->publish(all_markers);
}

} // namespace ftn_solo_control