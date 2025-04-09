#pragma once
// Common includes
#include <boost/optional.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
// FTN Solo control includes
#include <ftn_solo_control/estimators.h>
#include <ftn_solo_control/motions/motion.h>

namespace ftn_solo_control {

class COMMotionWrapper;

void InitWholeBodyPublisher();
class WholeBodyController {
public:
  WholeBodyController(const FixedPointsEstimator &estimator,
                      const FrictionConeMap &friction_cones, double max_torque,
                      const std::string &config);

  Eigen::VectorXd Compute(double t, const pinocchio::Model &model,
                          pinocchio::Data &data,
                          FixedPointsEstimator &estimator,
                          const std::vector<boost::shared_ptr<Motion>> &motions,
                          ConstRefVectorXd old_torque, const double alpha = 0,
                          size_t new_contacts = 0,
                          size_t ending_contacts = 0);

  ConstRefVector3d GetForce(size_t eef) const { return forces_.at(eef); }

  void PublishForceMarker(const FixedPointsEstimator &estimator);

protected:
  struct Config {
    double lambda_tangential = 0.01;
    double lambda_kd = 0.0;
    double lambda_torque = 0.03;
    double smooth = 0.8;
    Eigen::VectorXd B;       // Vector for configuration
    Eigen::VectorXd Fv;      // Vector for configuration
    Eigen::VectorXd sigma;   // Vector for configuration

    friend std::ostream &operator<<(std::ostream &os, const Config &config) {
      os << "Config {"
         << "\n  lambda_tangential: " << config.lambda_tangential
         << "\n  lambda_kd: " << config.lambda_kd
         << "\n  lambda_torque: " << config.lambda_torque
         << "\n  smooth: " << config.smooth
         << "\n  B: " << config.B.transpose()
         << "\n  Fv: " << config.Fv.transpose()
         << "\n  sigma: " << config.sigma.transpose()
         << "\n}";
      return os;
    }
  };

  Config config_; // Member variable for configuration
  double max_torque_; // Member variable for maximum torque
  Eigen::MatrixXd H_;
  proxsuite::proxqp::dense::QP<double> qp_;
  std::vector<size_t> eefs_;
  std::map<size_t, Eigen::Vector3d> forces_;
  std::map<size_t, Eigen::Matrix3d> tangential_;
  std::map<size_t, Eigen::Matrix3d> normal_;
};
} // namespace ftn_solo_control
