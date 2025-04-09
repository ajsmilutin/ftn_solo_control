#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
// FTN includes
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/types/friction_cone.h>
#include <ftn_solo_control/types/sensors.h>
#include <ftn_solo_control/iir/Butterworth.h>

namespace ftn_solo_control {

void InitEstimatorPublisher();

void ExposeEstimators();

class BaseEstimator {
public:
  BaseEstimator(double dt, const pinocchio::Model &model,
                pinocchio::Data &data);
  BaseEstimator(const BaseEstimator &other);

  virtual void Init(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
                    const SensorData &sensors) {};
  virtual void Estimate(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
                        const SensorData &sensors) {};
  virtual void PublishState(size_t seconds, size_t nanoseconds) const;

  inline void SetEffort(ConstRefVectorXd effort) { effort_ = effort; }

  Eigen::VectorXd estimated_q_;
  Eigen::VectorXd estimated_qv_;


  inline size_t NumJoints() const { return num_joints_; }
  inline size_t NumDoF() const { return model_.nv; }
  inline bool Initialized() const { return initialized_; }  

protected:
  double dt_;
  double t_;
  const pinocchio::Model &model_;
  pinocchio::Data &data_;
  size_t num_joints_;
  size_t base_index_;
  Eigen::VectorXd effort_;
  std::atomic<bool> initialized_;
};

class FixedRobotEstimator : public BaseEstimator {
public:
  FixedRobotEstimator(
      double dt, const pinocchio::Model &model, pinocchio::Data &data,
      bool fixed_orientation = false,
      ConstRefVector3d position = Eigen::Vector3d::Zero(),
      ConstRefMatrix3d orientation = Eigen::Matrix3d::Identity());
  // copy constructor
  FixedRobotEstimator(const ftn_solo_control::FixedRobotEstimator &other);
  void Init(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
            const SensorData &sensors) override;
  void Estimate(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
                const SensorData &sensors) override;
protected:
  Iir::Butterworth::LowPass<6, Iir::DirectFormII<Eigen::Matrix<double, 12, 1>>>
      qv_filter_;                

private:
  bool fixed_orientation_;
  Eigen::Vector3d position_;
  Eigen::Quaterniond orientation_;
};

class FixedPointsEstimator : public BaseEstimator {
public:
  FixedPointsEstimator(double dt, const pinocchio::Model &model,
                       pinocchio::Data &data,
                       const std::vector<size_t> &indexes_);
  // copy constructor
  FixedPointsEstimator(const ftn_solo_control::FixedPointsEstimator &other);
  void Init(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
            const SensorData &sensors) override;
  void SetFixed(size_t frame_index, ConstRefMatrix3d new_orientation);
  bool UnFix(size_t frame_index);

  void Estimate(double t, ConstRefVectorXd q, ConstRefVectorXd qv,
                const SensorData &sensors) override;

  FrictionCone CreateFrictionCone(size_t frame_index, ConstRefMatrix3d new_orientation,
                          double mu, size_t num_sides) const;

  FrictionConeMap GetFrictionCones(double mu = 1, size_t num_sides = 4);
  inline size_t NumContacts() const { return poses_.size(); }
  Eigen::MatrixXd constraint_;
  Eigen::VectorXd eef_positions_;
  Eigen::VectorXd velocity_;
  Eigen::VectorXd acceleration_;

  inline const std::vector<size_t> &GetContactPoints() const {
    return indexes_;
  }

protected:
  void SetData(double t, ConstRefVectorXd q, const VectorXd qv,
               const SensorData &sensors);

  void EstimateVelocities();
  void UpdateInternals(const std::map<size_t, pinocchio::SE3> &touching_poses,
                       const std::map<size_t, pinocchio::SE3> &placements);
  void UpdateIndexes();
  void InitAndEstimate();
  void EstimateInternal();

  Eigen::Vector3d sensor_angular_velocity_;
  std::vector<size_t> indexes_;
  std::map<size_t, pinocchio::SE3> poses_;
  std::map<size_t, pinocchio::SE3> touching_poses_;
  std::map<size_t, size_t> indexes_map_;
  std::thread thread_;
  Eigen::Quaterniond initial_orientation_;
  Eigen::Quaterniond measured_orientation_;
  Iir::Butterworth::LowPass<6, Iir::DirectFormII<Eigen::Matrix<double, 12, 1>>>
      qv_filter_;
};

} // namespace ftn_solo_control