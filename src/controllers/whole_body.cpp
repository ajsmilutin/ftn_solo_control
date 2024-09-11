#include <ftn_solo_control/controllers/whole_body.h>
// FTN solo includes
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

} // namespace

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
  Eigen::VectorXd d = Eigen::VectorXd::Zero(qp_.model.n_in);
  for (const auto &cone : friction_cones) {
    C.block(start_row, start_col, cone.second.GetNumSides(), 3) =
        cone.second.primal_.face_;
    start_row += cone.second.GetNumSides();
    C.block<1, 3>(start_row, start_col) =
        cone.second.GetPose().rotation().col(2).transpose();
    d(start_row) = 0.5;
    ++start_row;
    start_col += 3;
  }
  C.block(start_row, estimator.NumDoF(), estimator.NumJoints(),
          estimator.NumJoints()) =
      Eigen::MatrixXd::Identity(estimator.NumJoints(), estimator.NumJoints());
  d.tail(estimator.NumJoints()) =
      -max_torque_ * Eigen::VectorXd::Ones(estimator.NumJoints());
  Eigen::VectorXd u = 1e20 * Eigen::VectorXd::Ones(qp_.model.n_in);
  u.tail(estimator.NumJoints()) =
      max_torque_ * Eigen::VectorXd::Ones(estimator.NumJoints());
  qp_.init(proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt,
           proxsuite::nullopt, C, d, u);
}

Eigen::VectorXd WholeBodyController::Compute(
    double t, const pinocchio::Model &model, pinocchio::Data &data,
    const FixedPointsEstimator &estimator,
    const std::vector<boost::shared_ptr<Motion>> &motions) {
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
  Eigen::MatrixXd H =
      0.001 * Eigen::MatrixXd::Identity(qp_.model.dim, qp_.model.dim);
  H.topLeftCorner(estimator.NumDoF(), estimator.NumDoF()) =
      motions_jacobian.transpose() * motions_jacobian;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(qp_.model.dim);
  g.head(estimator.NumDoF()) = -motions_jacobian.transpose() * motions_ades;
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
  qp_.update(H, g, A, b, proxsuite::nullopt, proxsuite::nullopt,
             proxsuite::nullopt);
  qp_.solve();
  qp_.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  return qp_.results.x.segment(estimator.NumDoF(), estimator.NumJoints());
}

} // namespace ftn_solo_control