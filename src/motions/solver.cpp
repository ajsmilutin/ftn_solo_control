#include <ftn_solo_control/motions/solver.h>
// Common includes
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
  while (!solved && (++iteration) <= 1000) {
    pinocchio::framesForwardKinematics(model, data, q);
    pinocchio::computeJointJacobians(model, data, q);
    GetConstraintJacobian(model, data, friction_cones, constraint_jacobian);
    GetMotionsJacobian(model, data, q, 0 * q, motions, motions_jacobian);
    size_t start_row = 0;
    for (const auto &motion : motions) {
      motions_ades.segment(start_row, motion->dim_) =
          motion->GetPositionErrorToEnd(model, data);
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

} // namespace ftn_solo_control