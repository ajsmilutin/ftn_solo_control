#include <ftn_solo_control/utils/wcm.h>
// Common includes
#include <proxsuite/proxqp/dense/dense.hpp>
// ftn_solo_control includes
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/utils/utils.h>

namespace ftn_solo_control {

namespace {

double Dist(ConstRefVector2d u, ConstRefVector2d v) {
  return u(0) * v(1) - u(1) * v(0);
}

void ExpandSurface(std::vector<Eigen::Vector2d> &result, ConstRefVectorXd pt_0,
                   ConstRefVectorXd pt_1,
                   proxsuite::proxqp::dense::QP<double> &qp) {
  Eigen::Vector2d line = (pt_1.tail<2>() - pt_0.tail<2>()).normalized();
  Eigen::Vector2d normal = (Eigen::Vector2d() << line(1), -line(0)).finished();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(pt_0.size());
  g.tail<2>() = -normal;
  qp.update(proxsuite::nullopt, g, proxsuite::nullopt, proxsuite::nullopt,
            proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
  qp.solve((pt_0 + pt_1) * 0.5, Eigen::VectorXd::Zero(qp.model.n_eq),
           Eigen::VectorXd::Zero(qp.model.n_in));
  Eigen::VectorXd pt_mid = qp.results.x;
  if (Dist(pt_mid.tail<2>() - pt_0.tail<2>(), line) > 0.000125) {
    ExpandSurface(result, pt_0, pt_mid, qp);
    ExpandSurface(result, pt_mid, pt_1, qp);
  } else {
    result.push_back(pt_0.tail<2>());
  }
}

} // namespace

ConvexHull2D GetProjectedWCM(const FrictionConeMap &friction_cones,
                             const Eigen::MatrixXd &torque_constraint,
                             const Eigen::VectorXd &lb,
                             const Eigen::VectorXd &ub,
                             const Eigen::Vector3d &acom,
                             const ConvexHull2D &hull_constraint) {

  const size_t total_sides = GetTotalSides(friction_cones);
  const size_t num_force = 3 * friction_cones.size();
  const size_t num_torque = torque_constraint.rows();
  size_t num_hull_constraints = 2;
  if (hull_constraint.points_.size() > 0) {
    num_hull_constraints = hull_constraint.points_.size();
  }
  proxsuite::proxqp::dense::QP<double> qp(
      num_force + 2, 6, total_sides + num_hull_constraints + num_torque, false,
      proxsuite::proxqp::HessianType::Zero);
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, num_force + 2);
  size_t start_row = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (const auto &cone : friction_cones) {
    A.block<3, 3>(0, start_row) = Eigen::MatrixXd::Identity(3, 3);
    A.block<3, 3>(3, start_row) = CrossMatrix(cone.second.GetPosition());
    position += cone.second.GetPosition();
    start_row += 3;
  }
  position /= friction_cones.size();
  Eigen::VectorXd b = (Eigen::VectorXd(6) << 0, 0, 1, 0, 0, 0).finished();
  b.head<3>() += acom;
  A.bottomRightCorner<3, 2>() = CrossMatrix(b.head<3>()).leftCols<2>();

  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(
      total_sides + num_torque + num_hull_constraints, num_force + 2);
  start_row = 0;
  size_t i = 0;
  for (const auto &cone : friction_cones) {
    C.block(start_row, 3 * i, cone.second.GetNumSides(), 3) =
        cone.second.primal_.face_;
    ++i;
    start_row += cone.second.GetNumSides();
  }
  C.bottomLeftCorner(num_torque, num_torque) = torque_constraint;
  Eigen::VectorXd d =
      Eigen::VectorXd::Zero(total_sides + num_torque + num_hull_constraints);
  d.tail(num_torque) = lb;
  Eigen::VectorXd u = Eigen::VectorXd::Constant(
      total_sides + num_torque + num_hull_constraints, 1e10);
  u.tail(num_torque) = ub;
  if (hull_constraint.points_.size() > 0) {
    C.block(total_sides, num_force, num_hull_constraints, 2) =
        hull_constraint.Equations().leftCols<2>();
    d.segment(total_sides, num_hull_constraints) =
        -hull_constraint.Equations().rightCols<1>();
  } else {
    C.block<2, 2>(total_sides, num_force) = Eigen::Matrix2d::Identity();
    u.segment<2>(total_sides) =
        position.head<2>() + 10 * Eigen::Vector2d::Ones();
    d.segment<2>(total_sides) =
        position.head<2>() - 10 * Eigen::Vector2d::Ones();
  }

  Eigen::VectorXd g = Eigen::VectorXd::Zero(num_force + 2);
  g(num_force) = -1;
  qp.init(Eigen::MatrixXd::Zero(num_force + 2, num_force + 2), g, A, b, C, d,
          u);
  qp.settings.eps_abs = 1e-6;
  qp.solve();
  qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  Eigen::VectorXd result_0 = qp.results.x;
  g(num_force) = 1;
  qp.update(proxsuite::nullopt, g, proxsuite::nullopt, proxsuite::nullopt,
            proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
  qp.solve(qp.results.x, proxsuite::nullopt, proxsuite::nullopt);
  Eigen::VectorXd result_1 = qp.results.x;
  qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START;
  std::vector<Eigen::Vector2d> result;
  ExpandSurface(result, result_0, result_1, qp);
  ExpandSurface(result, result_1, result_0, qp);
  return ConvexHull2D(result);
}

ConvexHull2D GetProjectedWCMWithTorque(const pinocchio::Model &model,
                                       pinocchio::Data &data,
                                       const FrictionConeMap &friction_cones,
                                       double torque_limit,
                                       const Eigen::Vector3d &acom,
                                       const ConvexHull2D &hull_constraint) {
  size_t num_forces = friction_cones.size();
  Eigen::MatrixXd torque_constraint =
      Eigen::MatrixXd::Zero(3 * num_forces, 3 * num_forces);
  size_t start = 0;
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(3 * num_forces, -torque_limit);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(3 * num_forces, torque_limit);
  for (const auto &cone : friction_cones) {
    size_t joint = model.frames.at(cone.first).parentJoint;
    // Remove floating base
    joint = joint - 1;
    torque_constraint.block<3, 3>(start, start) =
        -GetContactJacobian(model, data, cone.first, cone.second.GetPose())
             .middleCols<3>(joint - 3 + 6)
             .transpose();
    lb.segment<3>(start) -= data.g.segment<3>(joint - 3 + 6);
    ub.segment<3>(start) -= data.g.segment<3>(joint - 3 + 6);
    start += 3;
  }
  const double mg = data.mass[0] * 9.81;
  return GetProjectedWCM(friction_cones, torque_constraint, lb / mg, ub / mg,
                         acom / 9.81, hull_constraint);
}

} // namespace ftn_solo_control