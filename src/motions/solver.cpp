#include <ftn_solo_control/motions/solver.h>
// Common includes
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
// FTN solo control includes
#include <ftn_solo_control/utils/utils.h>

namespace ftn_solo_control {

bool GetEndOfMotion(const pinocchio::Model &model, pinocchio::Data &data,
                    const FrictionConeMap &friction_cones,
                    const std::vector<boost::shared_ptr<Motion>> &motions,
                    RefVectorXd q) {
  size_t num_constraints = friction_cones.size() * 3;
  Eigen::MatrixXd constraint_jacobian =
      Eigen::MatrixXd::Zero(num_constraints, model.nv);
  size_t motions_dim = GetMotionsDim(motions);
  Eigen::MatrixXd motions_jacobian =
      Eigen::MatrixXd::Zero(motions_dim, model.nv);
  Eigen::VectorXd motions_ades = Eigen::VectorXd(motions_dim);
  bool solved = false;
  proxsuite::proxqp::dense::QP<double> qp(model.nv, num_constraints, 0);
  size_t iteration = 0;
  Eigen::VectorXd qv = 0 * q;
  while (!solved && (++iteration) <= 1000) {
    pinocchio::framesForwardKinematics(model, data, q);
    pinocchio::computeJointJacobians(model, data, q);
    GetConstraintJacobian(model, data, friction_cones, constraint_jacobian);
    GetMotionsJacobian(model, data, qv, 0 * q, motions, motions_jacobian);
    size_t start_row = 0;
    for (const auto &motion : motions) {
      motions_ades.segment(start_row, motion->dim_) =
          motion->GetPositionErrorToEnd(model, data, q, qv);
      start_row += motion->dim_;
    }
    qp.init(motions_jacobian.transpose() * motions_jacobian,
            -motions_jacobian.transpose() * motions_ades, constraint_jacobian,
            Eigen::VectorXd::Zero(num_constraints), proxsuite::nullopt,
            proxsuite::nullopt, proxsuite::nullopt);
    qp.solve();
    pinocchio::integrate(model, q, 0.1 * qp.results.x, q);
    solved = motions_ades.norm() < 1e-4 || (qp.results.x.norm() < 1e-4);
  }
  pinocchio::framesForwardKinematics(model, data, q);
  pinocchio::computeJointJacobians(model, data, q);
  return solved;
}

bool GetEndOfMotionPrioritized(
    const pinocchio::Model &model, pinocchio::Data &data,
    const FrictionConeMap &friction_cones,
    const std::vector<boost::shared_ptr<Motion>> &motions, RefVectorXd q,
    bool has_joint_limits) {
  std::map<size_t, std::vector<boost::shared_ptr<Motion>>> priority_sets;
  std::map<size_t, size_t> priority_dims;
  for (size_t i = 0; i < motions.size(); ++i) {
    const size_t priority = motions.at(i)->priority_;
    if (priority_sets.count(priority) == 0) {
      priority_sets.emplace(priority, std::vector<boost::shared_ptr<Motion>>());
      priority_dims.emplace(priority, 0);
    }
    priority_sets.at(priority).push_back(motions.at(i));
    priority_dims.at(priority) += motions.at(i)->dim_;
  }
  size_t num_constraints = friction_cones.size() * 3;
  std::map<size_t, Eigen::VectorXd> motions_ades;
  std::map<size_t, proxsuite::proxqp::dense::QP<double>> qps;
  size_t prev_dim = 0;
  for (const auto &dim : priority_dims) {
    motions_ades.emplace(dim.first, Eigen::VectorXd::Zero(dim.second));
    qps.emplace(dim.first,
                proxsuite::proxqp::dense::QP<double>(
                    model.nv, num_constraints + prev_dim, 0, has_joint_limits));
    prev_dim += dim.second;
  }
  Eigen::MatrixXd constraint_jacobian =
      Eigen::MatrixXd::Zero(num_constraints + prev_dim, model.nv);
  Eigen::VectorXd constraint_value =
      Eigen::VectorXd::Zero(num_constraints + prev_dim);
  bool solved = false;
  size_t iteration = 0;
  const double alpha = 0.1;
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(model.nv, 1e10);
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(model.nv, -1e10);
  const double kLimit = M_PI / 6;
  for (size_t index : {6, 9, 12, 15}) {
    ub(index) = 1.0;
    lb(index) = -1.0;
  }
  for (size_t index : {7, 10, 13, 16}) {
    ub(index) = 3.5;
    lb(index) = -1.5;
  }
  
  for (size_t index : {8, 11, 14, 17}) {
    ub(index) = -0.8;
    lb(index) = -2.7;
  }
  // for (size_t index : {8, 11, 14, 17}) {
  //   if (q(index + 1) < 0) {
  //     ub(index) = -kLimit;
  //   } else {
  //     lb(index) = kLimit;
  //   }
  // }

  while (!solved && (++iteration) <= 1000) {
    pinocchio::framesForwardKinematics(model, data, q);
    pinocchio::centerOfMass(model, data, q, true);
    pinocchio::computeJointJacobians(model, data, q);
    GetConstraintJacobian(model, data, friction_cones, constraint_jacobian);
    size_t prev_dim = 0;
    Eigen::VectorXd qv;
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(q.size());
    solved = true;
    for (const auto &ps : priority_sets) {
      size_t dim = priority_dims.at(ps.first);
      auto &ades = motions_ades.at(ps.first);
      auto &qp = qps.at(ps.first);
      // Solve per priority
      GetMotionsJacobian(
          model, data, q, 0 * q, ps.second,
          constraint_jacobian.middleRows(num_constraints + prev_dim, dim),
          true);
      size_t start_row = 0;
      for (const auto &motion : ps.second) {
        ades.segment(start_row, motion->dim_) =
            motion->GetWeight() *
            motion->GetPositionErrorToEnd(model, data, q, zero);
        start_row += motion->dim_;
      }
      proxsuite::optional<Eigen::VectorXd> ubx = proxsuite::nullopt;
      proxsuite::optional<Eigen::VectorXd> lbx = proxsuite::nullopt;
      if (has_joint_limits) {
        ubx = (ub - q.tail(model.nv)) / alpha;
        lbx = (lb - q.tail(model.nv)) / alpha;
      }
      qp.init(
          constraint_jacobian.middleRows(num_constraints + prev_dim, dim)
                  .transpose() *
              constraint_jacobian.middleRows(num_constraints + prev_dim, dim),
          -constraint_jacobian.middleRows(num_constraints + prev_dim, dim)
                  .transpose() *
              ades,
          constraint_jacobian.topRows(num_constraints + prev_dim),
          constraint_value.head(num_constraints + prev_dim), proxsuite::nullopt,
          proxsuite::nullopt, proxsuite::nullopt, lbx, ubx);
      qp.solve();
      qv = qp.results.x;
      constraint_value.segment(num_constraints + prev_dim, dim) =
          constraint_jacobian.middleRows(num_constraints + prev_dim, dim) * qv;
      solved = solved && (ades.norm() < 1e-4 || qv.norm() < 1e-4);
      prev_dim += dim;
    }
    pinocchio::integrate(model, q, alpha * qv, q);
  }
  pinocchio::framesForwardKinematics(model, data, q);
  pinocchio::centerOfMass(model, data, q, true);
  pinocchio::computeJointJacobians(model, data, q);
  return solved;
}

} // namespace ftn_solo_control