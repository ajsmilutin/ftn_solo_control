#pragma once

#include <ftn_solo_control/motions/motion.h>
// common includes
#include <pinocchio/algorithm/frames.hpp>
// ftn_solo_control
#include <ftn_solo_control/trajectories/trajectory.h>
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {
class EEFPositionMotion
    : public MotionWithTrajectory<Trajectory<Eigen::VectorXd, RefVectorXd,
                                             Eigen::VectorXd, RefVectorXd>> {
public:
  EEFPositionMotion(
      size_t eef_index, ConstRefVector3b selected,
      const pinocchio::SE3 &origin_pose = pinocchio::SE3::Identity(),
      double Kp = 100, double Kd = 50);

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
  const pinocchio::SE3 origin_;
};

} // namespace ftn_solo_control
