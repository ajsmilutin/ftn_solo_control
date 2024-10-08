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
  // copy constructor
  FixedPointsEstimator(const ftn_solo_control::FixedPointsEstimator &other);
  void Init(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
            const SensorData &sensors);
  void SetFixed(size_t frame_index, ConstRefMatrix3d new_orientation);
  bool UnFix(size_t frame_index);

  void Estimate(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
                const SensorData &sensors);

  FrictionCone CreateFrictionCone(size_t frame_index, ConstRefMatrix3d new_orientation,
                          double mu, size_t num_sides) const;

  FrictionConeMap GetFrictionCones(double mu = 1, size_t num_sides = 4);

  inline size_t NumJoints() const { return num_joints_; }
  inline size_t NumDoF() const { return model_.nv; }
  inline size_t NumContacts() const { return poses_.size(); }
  inline bool Initialized() const { return initialized_; }
  Eigen::VectorXd estimated_q_;
  Eigen::VectorXd estimated_qv_;
  Eigen::MatrixXd constraint_;
  Eigen::VectorXd eef_positions_;
  Eigen::VectorXd velocity_;
  Eigen::VectorXd acceleration_;

  inline const std::vector<size_t> &GetContactPoints() const {
    return indexes_;
  }

protected:
  void SetData(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
               const SensorData &sensors);

  void EstimateVelocities();
  void UpdateInternals(const std::map<size_t, pinocchio::SE3> &touching_poses,
                       const std::map<size_t, pinocchio::SE3> &placements);
  void UpdateIndexes();
  void InitAndEstimate();
  void EstimateInternal();

  double t_;
  double dt_;
  Eigen::Vector3d sensor_angular_velocity_;
  size_t num_joints_;
  const pinocchio::Model &model_;
  pinocchio::Data &data_;
  std::vector<size_t> indexes_;
  std::map<size_t, pinocchio::SE3> poses_;
  std::map<size_t, pinocchio::SE3> touching_poses_;
  std::map<size_t, size_t> indexes_map_;
  std::atomic<bool> initialized_;
  std::thread thread_;
};

} // namespace ftn_solo_control