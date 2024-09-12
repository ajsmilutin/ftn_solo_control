#include <ftn_solo_control/controllers/whole_body.h>
// Common includes
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// FTN solo includes
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

} // namespace

void InintWholeBodyPublisher() {
  whole_body_node = std::make_shared<rclcpp::Node>("whole_body_node");
  publisher =
      whole_body_node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "force_markers", 10);
}

WholeBodyController::WholeBodyController(const FixedPointsEstimator &estimator,
                                         const FrictionConeMap &friction_cones,
                                         double max_torque)
    : max_torque_(max_torque),
      qp_(estimator.NumJoints() + estimator.NumDoF() +
              estimator.NumContacts() * 3,
          estimator.NumDoF() + estimator.NumContacts() * 3,
          estimator.NumJoints() + TotalSides(friction_cones) +
              estimator.NumContacts(),
          false, proxsuite::proxqp::HessianType::Dense) {
  size_t total_sides = TotalSides(friction_cones);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(qp_.model.n_in, qp_.model.dim);
  size_t start_row = 0;
  size_t start_col = estimator.NumDoF() + estimator.NumJoints();
  d_ = 0.1 * Eigen::VectorXd::Ones(qp_.model.n_in);
  H_ = 0.001 * Eigen::MatrixXd::Identity(qp_.model.dim, qp_.model.dim);
  double lambda_tangential = 0.1;
  std::cout<<"CCCCCCCCCCCC";
  for (const auto &cone : friction_cones) {
    eefs_.push_back(cone.first);
    std::cout<<" "<<cone.first<< std::endl;
    forces_.emplace(cone.first, Eigen::Vector3d::Zero());
    C.block(start_row, start_col, cone.second.GetNumSides(), 3) =
        cone.second.primal_.face_;
    start_row += cone.second.GetNumSides();
    C.block<1, 3>(start_row, start_col) =
        cone.second.GetPose().rotation().col(2).transpose();
    eef_to_row_[cone.first] = start_row;
    H_.block<3, 3>(start_col, start_col) =
        lambda_tangential *
        cone.second.GetPose().rotation().bottomRows<2>().transpose() *
        cone.second.GetPose().rotation().bottomRows<2>();
    ++start_row;
    start_col += 3;
  }
  std::cout<<std::endl;
  std::cout<<"ESTIMATOR "<< std::endl;
  for (const auto pt: estimator.GetContactPoints()){
    std::cout<<" "<< pt;
  }
  std::cout<<std::endl;

  C.block(start_row, estimator.NumDoF(), estimator.NumJoints(),
          estimator.NumJoints()) =
      Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());
  d_.tail(estimator.NumJoints()) =
      -max_torque_ * Eigen::VectorXd::Ones(estimator.NumJoints());
  Eigen::VectorXd u = 1e20 * Eigen::VectorXd::Ones(qp_.model.n_in);
  u.tail(estimator.NumJoints()) =
      max_torque_ * Eigen::VectorXd::Ones(estimator.NumJoints());

  qp_.init(proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt,
           proxsuite::nullopt, C, d_, u);
}

Eigen::VectorXd WholeBodyController::Compute(
    double t, const pinocchio::Model &model, pinocchio::Data &data,
    const FixedPointsEstimator &estimator,
    const std::vector<boost::shared_ptr<Motion>> &motions,
    ConstRefVectorXd old_torque, ConstRefVectorXi index,
    ConstRefVectorXd limits) {
  size_t motions_dim = GetMotionsDim(motions);

  Eigen::MatrixXd motions_jacobian =
      Eigen::MatrixXd(motions_dim, estimator.NumDoF());
  GetMotionsJacobian(model, data, estimator.estimated_q_,
                     estimator.estimated_qv_, motions, motions_jacobian);
  Eigen::VectorXd motions_ades = Eigen::VectorXd(motions_dim);
  size_t start_row = 0;
  for (const auto &motion : motions) {
    motions_ades.segment(start_row, motion->dim_) =
        motion->GetDesiredAcceleration(t, model, data) -
        motion->GetAcceleration(model, data);
    start_row += motion->dim_;
  }

  const auto points = estimator.GetContactPoints();
  for (const auto eef : eefs_) {
    for (size_t i = 0; i < index.size(); ++i) {
      if (eef == index[i]) {
        d_[eef_to_row_.at(eef)] = limits[i];
      }
    }
  }

  double lambda_kd = 0.0;
  double Kd = 1;

  H_.topLeftCorner(estimator.NumDoF(), estimator.NumDoF()) =
      motions_jacobian.transpose() * motions_jacobian;
  H_.block(6, 6, estimator.NumJoints(), estimator.NumJoints()) +=
      lambda_kd *
      Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());
  double lambda_torque = 5;
  H_.block(estimator.NumDoF(), estimator.NumDoF(), estimator.NumJoints(),
           estimator.NumJoints()) =
      lambda_torque *
      Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());

  Eigen::VectorXd g = Eigen::VectorXd::Zero(qp_.model.dim);
  g.head(estimator.NumDoF()) = -motions_jacobian.transpose() * motions_ades;
  g.segment(6, estimator.NumJoints()) +=
      lambda_kd * Kd * estimator.velocity_.tail(estimator.NumJoints());
  g.segment(estimator.NumDoF(), estimator.NumJoints()) =
      -lambda_torque * old_torque;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(qp_.model.n_eq, qp_.model.dim);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(qp_.model.n_eq);
  // Multi-body dynamics constraint
  A.topLeftCorner(estimator.NumDoF(), estimator.NumDoF()) = data.M;
  A.block(6, estimator.NumDoF(), estimator.NumJoints(), estimator.NumJoints()) =
      -Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());
  A.block(0, estimator.NumJoints() + estimator.NumDoF(), estimator.NumDoF(),
          estimator.NumContacts() * 3) = -estimator.constraint_.transpose();
  b.head(estimator.NumDoF()) = -data.nle;
  // Constraint
  A.block(estimator.NumDoF(), 0, 3 * estimator.NumContacts(),
          estimator.NumDoF()) = estimator.constraint_;
  b.tail(3 * estimator.NumContacts()) = -estimator.acceleration_;
  qp_.update(H_, g, A, b, proxsuite::nullopt, d_,
             proxsuite::nullopt);
  qp_.solve();
  qp_.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  size_t start_force = estimator.NumDoF() + estimator.NumJoints();
  for (size_t i = 0; i < eefs_.size(); ++i) {
    forces_[eefs_.at(i)] = qp_.results.x.segment<3>(start_force + i * 3);
    std::cout<<" FFFF "<<  i <<" eeef"<< eefs_.at(i)<< " FFF "<< forces_.at(eefs_.at(i))<< std::endl;
  }
  PublishForceMarker(estimator);
  if (qp_.results.info.status !=
      proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
    std::cout << "AAAAAAAAAAAAAAAAAAAAAAA" << std::endl;
    char c;
    std::cin >> c;
    return old_torque;
  }
  return qp_.results.x.segment(estimator.NumDoF(), estimator.NumJoints());
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
  single_arrow.scale.x = 0.02;
  single_arrow.scale.y = 0.04;
  single_arrow.scale.z = 0.04;
  for (size_t i = 0; i < eefs_.size(); ++i) {
    single_arrow.ns = "force" + std::to_string(eefs_.at(i));
    single_arrow.id = eefs_.at(i);
    single_arrow.points.clear();
    single_arrow.points.push_back(
        ToPoint(estimator.eef_positions_.segment<3>(3 * i)));
    single_arrow.points.push_back(
        ToPoint(estimator.eef_positions_.segment<3>(3 * i) +
                0.01 * GetForce(eefs_.at(i))));
    all_markers.markers.push_back(single_arrow);
  }
  publisher->publish(all_markers);
}

} // namespace ftn_solo_control