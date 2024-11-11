#pragma once

#include <ftn_solo_control/motions/motion.h>
// ftn_solo_control
#include <ftn_solo_control/trajectories/trajectory.h>
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {
class JointMotion
    : public MotionWithTrajectory<Trajectory<Eigen::VectorXd, RefVectorXd,
                                             Eigen::VectorXd, RefVectorXd>> {
public:
  JointMotion(ConstRefVectorXi joints, double Kp = 100, double Kd = 50);

  Eigen::VectorXd GetPositionError(const RefVectorXd pos,
                                   const pinocchio::Model &model,
                                   pinocchio::Data &data, ConstRefVectorXd q,
                                   ConstRefVectorXd qv) const override;
  virtual Eigen::VectorXd GetVelocityError(const RefVectorXd vel,
                                           const pinocchio::Model &model,
                                           pinocchio::Data &data,
                                           ConstRefVectorXd q,
                                           ConstRefVectorXd qv) const override;

  Eigen::MatrixXd GetJacobian(const pinocchio::Model &model,
                              pinocchio::Data &data, ConstRefVectorXd q,
                              ConstRefVectorXd qv) const override;

  Eigen::VectorXd GetAcceleration(const pinocchio::Model &model,
                                  pinocchio::Data &data, ConstRefVectorXd q,
                                  ConstRefVectorXd qv) const override;

protected:
  VectorXi indexes_6_;
  VectorXi indexes_7_;
};

} // namespace ftn_solo_control
