#pragma once
// Common includes
#include <proxsuite/proxqp/dense/dense.hpp>
// FTN Solo control includes
#include <ftn_solo_control/estimators.h>
#include <ftn_solo_control/motions/motion.h>

namespace ftn_solo_control {

class COMMotionWrapper;

class WholeBodyController {
public:
  WholeBodyController(const FixedPointsEstimator &estimator,
                      const FrictionConeMap &friction_cones, double max_torque);

  Eigen::VectorXd
  Compute(double t, const pinocchio::Model &model, pinocchio::Data &data,
          const FixedPointsEstimator &estimator,
          const std::vector<boost::shared_ptr<Motion>> &motions);

protected:
  double max_torque_;
  proxsuite::proxqp::dense::QP<double> qp_;
};
} // namespace ftn_solo_control
