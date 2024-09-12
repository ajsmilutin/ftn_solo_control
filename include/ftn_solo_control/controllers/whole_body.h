#pragma once
// Common includes
#include <boost/optional.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
// FTN Solo control includes
#include <ftn_solo_control/estimators.h>
#include <ftn_solo_control/motions/motion.h>

namespace ftn_solo_control {

class COMMotionWrapper;

void InintWholeBodyPublisher();
class WholeBodyController {
public:
  WholeBodyController(const FixedPointsEstimator &estimator,
                      const FrictionConeMap &friction_cones, double max_torque);

  Eigen::VectorXd Compute(double t, const pinocchio::Model &model,
                          pinocchio::Data &data,
                          const FixedPointsEstimator &estimator,
                          const std::vector<boost::shared_ptr<Motion>> &motions,
                          ConstRefVectorXd old_torque,
                          ConstRefVectorXi index = VectorXi(0),
                          ConstRefVectorXd limits = VectorXd(0));

  inline Eigen::VectorXd
  ComputeWithoutLimits(double t, const pinocchio::Model &model,
                       pinocchio::Data &data,
                       const FixedPointsEstimator &estimator,
                       const std::vector<boost::shared_ptr<Motion>> &motions,
                       ConstRefVectorXd old_torque) {
    return Compute(t, model, data, estimator, motions, old_torque);
  }

  ConstRefVector3d GetForce(size_t eef) const { return forces_.at(eef); }

  void PublishForceMarker(const FixedPointsEstimator &estimator);

protected:
  double max_torque_;
  Eigen::MatrixXd H_;
  Eigen::VectorXd d_;
  proxsuite::proxqp::dense::QP<double> qp_;
  std::vector<size_t> eefs_;
  std::unordered_map<size_t, Eigen::Vector3d> forces_;
  std::unordered_map<size_t, size_t> eef_to_row_;
};
} // namespace ftn_solo_control
