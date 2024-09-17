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

void ExpandSurface(std::vector<Eigen::Vector2d> &result, ConstRefVector2d pt_0,
                   ConstRefVector2d pt_1,
                   proxsuite::proxqp::dense::QP<double> &qp, RefVectorXd g) {
  Eigen::Vector2d line = (pt_1 - pt_0).normalized();
  Eigen::Vector2d normal = (Eigen::Vector2d() << line(1), -line(0)).finished();
  g.tail<2>() = -normal;
  qp.update(proxsuite::nullopt, g, proxsuite::nullopt, proxsuite::nullopt,
            proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
  qp.solve();
  Eigen::Vector2d pt_mid = qp.results.x.tail<2>();
  if (Dist(pt_mid - pt_0, line) > 0.0025) {
    ExpandSurface(result, pt_0, pt_mid, qp, g);
    ExpandSurface(result, pt_mid, pt_1, qp, g);
  } else {
    if (result.empty()) {
      result.push_back(pt_0);
    } else if ((pt_0 - result.back()).norm() > 1e-4) {
      result.push_back(pt_0);
    }
  }
}

} // namespace

ConvexHull2D GetProjectedWCM(const FrictionConeMap &friction_cones,
                             const Eigen::MatrixXd &torque_constraint,
                             const Eigen::VectorXd &lb,
                             const Eigen::VectorXd &ub) {
  std::vector<Eigen::Vector2d> result;
  const size_t total_sides = GetTotalSides(friction_cones);
  const size_t num_force = 3 * friction_cones.size();
  const size_t num_torque = torque_constraint.rows();
  proxsuite::proxqp::dense::QP<double> qp(num_force + 2, 6,
                                          total_sides + 2 + num_torque, false,
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
  A(4, num_force) = 1;
  A(3, num_force + 1) = -1;
  Eigen::VectorXd b = (Eigen::VectorXd(6) << 0, 0, 1, 0, 0, 0).finished();
  Eigen::MatrixXd C =
      Eigen::MatrixXd::Zero(total_sides + num_torque + 2, num_force + 2);
  start_row = 0;
  size_t i = 0;
  for (const auto &cone : friction_cones) {
    C.block(start_row, 3 * i, cone.second.GetNumSides(), 3) =
        cone.second.primal_.face_;
    ++i;
    start_row += cone.second.GetNumSides();
  }
  C.block<2, 2>(total_sides, num_force) = Eigen::Matrix2d::Identity();
  C.bottomLeftCorner(num_torque, num_torque) = torque_constraint;
  Eigen::VectorXd d = Eigen::VectorXd::Zero(total_sides + num_torque + 2);
  d.segment<2>(total_sides) = position.head<2>() - Eigen::Vector2d::Ones();
  d.tail(num_torque) = lb;
  Eigen::VectorXd u =
      Eigen::VectorXd::Constant(total_sides + num_torque + 2, 1e10);
  u.segment<2>(total_sides) = position.head<2>() + Eigen::Vector2d::Ones();
  u.tail(num_torque) = ub;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(num_force + 2);
  g(num_force) = -1;
  qp.settings.eps_abs = 1e-6;
  qp.init(Eigen::MatrixXd::Zero(num_force + 2, num_force + 2), g, A, b, C, d,
          u);
  qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  qp.solve();
  qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  Eigen::Vector2d pt_0 = qp.results.x.tail<2>();
  g(num_force) = 1;
  qp.update(proxsuite::nullopt, g, proxsuite::nullopt, proxsuite::nullopt,
            proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
  qp.solve(qp.results.x, proxsuite::nullopt, proxsuite::nullopt);
  Eigen::Vector2d pt_1 = qp.results.x.tail<2>();
  ExpandSurface(result, pt_0, pt_1, qp, g);
  ExpandSurface(result, pt_1, pt_0, qp, g);
  return ConvexHull2D(result);
}

ConvexHull2D GetProjectedWCMWithTorque(const pinocchio::Model &model,
                                       pinocchio::Data &data,
                                       const FrictionConeMap &friction_cones,
                                       double torque_limit) {
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
  return GetProjectedWCM(friction_cones, torque_constraint, lb / mg, ub / mg);
}

} // namespace ftn_solo_control