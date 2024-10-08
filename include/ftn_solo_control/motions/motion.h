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
  Motion(double Kp = 100, double Kd = 50)
      : priority_(0), Kp_(Kp), Kd_(Kd), weight_(1.0) {}

  void SetPriority(size_t priority, double weight) {
    priority_ = priority;
    weight_ = weight;
  }

  virtual Eigen::VectorXd GetDesiredAcceleration(double t,
                                                 const pinocchio::Model &model,
                                                 pinocchio::Data &data,
                                                 ConstRefVectorXd q,
                                                 ConstRefVectorXd qv) const {
    return Eigen::VectorXd::Zero(dim_);
  }

  virtual Eigen::MatrixXd GetJacobian(const pinocchio::Model &model,
                                      pinocchio::Data &data, ConstRefVectorXd q,
                                      ConstRefVectorXd qv) const {
    return Eigen::MatrixXd::Zero(dim_, model.nv);
  }

  virtual Eigen::VectorXd GetAcceleration(const pinocchio::Model &model,
                                          pinocchio::Data &data,
                                          ConstRefVectorXd q,
                                          ConstRefVectorXd qv) const {
    return Eigen::MatrixXd::Zero(dim_, model.nv);
  };

  virtual Eigen::VectorXd GetPositionErrorToEnd(const pinocchio::Model &model,
                                                pinocchio::Data &data,
                                                ConstRefVectorXd q,
                                                ConstRefVectorXd qv) const {
    return Eigen::MatrixXd::Zero(dim_, model.nv);
  }

  virtual void SetStart(double t) {}

  virtual bool Finished() const { return false; }

  size_t dim_;
  size_t priority_;

  inline double GetWeight() const { return weight_; }

protected:
  double Kp_;
  double Kd_;
  double weight_;
};

template <class Trajectory> class MotionWithTrajectory : public Motion {
public:
  MotionWithTrajectory(ConstRefVector3b selected, double Kp = 100,
                       double Kd = 50)
      : Motion(Kp, Kd), trajectory_(nullptr) {
    dim_ = selected.count();
    indexes_ = Eigen::VectorXi(dim_);
    size_t j = 0;
    for (size_t i = 0; i < selected.size(); ++i) {
      if (selected[i]) {
        indexes_[j] = i;
        ++j;
      }
    }
  }

  MotionWithTrajectory(ConstRefVectorXi indexes, double Kp = 100,
                       double Kd = 50)
      : Motion(Kp, Kd), trajectory_(nullptr), indexes_(indexes) {
    dim_ = indexes_.size();
  }

  virtual Eigen::VectorXd
  GetPositionError(const Trajectory::PositionTypeRef pos,
                   const pinocchio::Model &model, pinocchio::Data &data,
                   ConstRefVectorXd q, ConstRefVectorXd qv) const = 0;

  virtual Eigen::VectorXd
  GetVelocityError(const Trajectory::VelocityTypeRef pos,
                   const pinocchio::Model &model, pinocchio::Data &data,
                   ConstRefVectorXd q, ConstRefVectorXd qv) const = 0;

  Eigen::VectorXd GetDesiredAcceleration(double t,
                                         const pinocchio::Model &model,
                                         pinocchio::Data &data,
                                         ConstRefVectorXd q,
                                         ConstRefVectorXd qv) const {
    auto pos = trajectory_->ZeroPosition();
    auto vel = trajectory_->ZeroVelocity();
    auto acc = trajectory_->ZeroVelocity();
    trajectory_->Get(t, pos, vel, acc);
    return acc + Kp_ * GetPositionError(pos, model, data, q, qv) +
           Kd_ * GetVelocityError(vel, model, data, q, qv);
  }

  void SetTrajectory(const boost::shared_ptr<Trajectory> trajectory) {
    trajectory_ = trajectory;
  }

  Eigen::VectorXd GetPositionErrorToEnd(const pinocchio::Model &model,
                                        pinocchio::Data &data,
                                        ConstRefVectorXd q,
                                        ConstRefVectorXd qv) const {
    auto final_position = trajectory_->FinalPosition();
    return GetPositionError(final_position, model, data, q, qv);
  }

  boost::shared_ptr<Trajectory> trajectory_;

  void SetStart(double t) override { trajectory_->SetStart(t); }

  bool Finished() const override { return trajectory_->finished_; }

protected:
  VectorXi indexes_;
};

} // namespace ftn_solo_control
