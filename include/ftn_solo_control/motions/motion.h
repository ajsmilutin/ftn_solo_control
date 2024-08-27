#pragma once

// STL includes
#include <iostream>
#include <memory>
// Common includes
#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
// ftn_solo_control
#include <ftn_solo_control/trajectories/trajectory.h>
namespace ftn_solo_control {
class Motion {
public:
  Motion(double Kp = 100, double Kd = 50) : Kp_(Kp), Kd_(Kd) {}

  virtual Eigen::VectorXd
  GetDesiredAcceleration(double t, const pinocchio::Model &model,
                         pinocchio::Data &data) const = 0;
  size_t dim_;

protected:
  double Kp_;
  double Kd_;
};

template <class Trajectory> class MotionWithTrajectory : public Motion {
public:
  MotionWithTrajectory(double Kp = 100, double Kd = 50)
      : Motion(Kp, Kd), trajectory_(nullptr) {}

  virtual Eigen::VectorXd
  GetPositionError(const Trajectory::PositionTypeRef pos,
                   const pinocchio::Model &model,
                   pinocchio::Data &data) const = 0;

  virtual Eigen::VectorXd
  GetVelocityError(const Trajectory::VelocityTypeRef pos,
                   const pinocchio::Model &model,
                   pinocchio::Data &data) const = 0;

  virtual Eigen::MatrixXd GetJacobian(const pinocchio::Model &model,
                                      pinocchio::Data &data, ConstRefVectorXd q,
                                      ConstRefVectorXd qv) const = 0;

  Eigen::VectorXd GetDesiredAcceleration(double t,
                                         const pinocchio::Model &model,
                                         pinocchio::Data &data) const {
    auto pos = trajectory_->ZeroPosition();
    auto vel = trajectory_->ZeroVelocity();
    auto acc = trajectory_->ZeroVelocity();
    trajectory_->Get(t, pos, vel, acc);

    return acc + Kp_ * GetPositionError(pos, model, data) +
           Kd_ * GetVelocityError(vel, model, data);
  }

  virtual Eigen::VectorXd GetAcceleration(const pinocchio::Model &model,
                                          pinocchio::Data &data) const = 0;

  void SetTrajectory(const std::shared_ptr<Trajectory> trajectory) {
    trajectory_ = trajectory;
  }

  std::shared_ptr<Trajectory> trajectory_;
};

} // namespace ftn_solo_control
