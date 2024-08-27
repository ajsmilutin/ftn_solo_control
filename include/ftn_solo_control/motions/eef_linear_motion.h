#pragma once

#include <ftn_solo_control/motions/motion.h>
// common includes
#include <pinocchio/algorithm/frames.hpp>
// ftn_solo_control
#include <ftn_solo_control/trajectories/trajectory.h>
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {
class EEFLinearMotion
    : public MotionWithTrajectory<Trajectory<Eigen::VectorXd, RefVectorXd,
                                             Eigen::VectorXd, RefVectorXd>> {
public:
  EEFLinearMotion(
      size_t eef_index, ConstRefVector3b selected,
      const pinocchio::SE3 &origin_pose = pinocchio::SE3::Identity(),
      double Kp = 100, double Kd = 50)
      : MotionWithTrajectory(Kp, Kd), eef_index_(eef_index),
        origin_(origin_pose) {
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

  Eigen::VectorXd GetPositionError(const RefVectorXd pos,
                                   const pinocchio::Model &model,
                                   pinocchio::Data &data) const override;
  virtual Eigen::VectorXd
  GetVelocityError(const RefVectorXd vel, const pinocchio::Model &model,
                   pinocchio::Data &data) const override;

  Eigen::MatrixXd GetJacobian(const pinocchio::Model &model,
                              pinocchio::Data &data, ConstRefVectorXd q,
                              ConstRefVectorXd qv) const override;

  Eigen::VectorXd GetAcceleration(const pinocchio::Model &model,
                                  pinocchio::Data &data) const override;

protected:
  size_t eef_index_;
  VectorXi indexes_;
  const pinocchio::SE3 origin_;
};

} // namespace ftn_solo_control
