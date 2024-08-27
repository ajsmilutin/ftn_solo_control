#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
// FTN includes
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/types/friction_cone.h>
#include <ftn_solo_control/types/sensors.h>

namespace ftn_solo_control {

void InitEstimatorPublisher();

class FixedPointsEstimator {
public:
  FixedPointsEstimator(double dt, const pinocchio::Model &model,
                       pinocchio::Data &data,
                       const std::vector<size_t> &indexes_);
  void Init(ConstRefVectorXd q, ConstRefVectorXd qv, const SensorData &sensors);
  void SetFixed(size_t frame_index, ConstRefMatrix3d new_orientation);
  bool UnFix(size_t frame_index);

  void Estimate(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
                const SensorData &sensors);

  std::map<size_t, FrictionCone> GetFrictionCones(double mu = 1,
                                                  size_t num_sides = 4);

  Eigen::VectorXd estimated_q_;
  Eigen::VectorXd estimated_qv_;
  Eigen::MatrixXd constraint_;
  Eigen::VectorXd velocity_;
  Eigen::VectorXd acceleration_;

protected:
  void SetData(ConstRefVectorXd q, ConstRefVectorXd qv,
               const SensorData &sensors);

  void EstimateVelocities(const SensorData &sensors);
  void UpdateInternals(
      const std::unordered_map<size_t, pinocchio::SE3> &touching_poses,
      const std::unordered_map<size_t, pinocchio::SE3> &placements);
  double dt_;
  size_t num_joints_;
  const pinocchio::Model &model_;
  pinocchio::Data &data_;
  std::vector<size_t> indexes_;
  std::unordered_map<size_t, pinocchio::SE3> poses_;
  std::unordered_map<size_t, pinocchio::SE3> touching_poses_;
  std::unordered_map<size_t, size_t> indexes_map_;
};

} // namespace ftn_solo_control