#include <ftn_solo_control/motions/com_motion.h>
// Common includes
#include <pinocchio/algorithm/center-of-mass.hpp>

namespace ftn_solo_control {

COMMotion::COMMotion(ConstRefVector3b selected,
                     const pinocchio::SE3 &origin_pose, double Kp, double Kd)
    : MotionWithTrajectory(selected, Kp, Kd), origin_(origin_pose) {}

Eigen::VectorXd COMMotion::GetPositionError(const RefVectorXd pos,
                                            const pinocchio::Model &model,
                                            pinocchio::Data &data,
                                            ConstRefVectorXd q,
                                            ConstRefVectorXd qv) const {
  return pos - origin_.actInv(data.com[0])(indexes_);
}

Eigen::VectorXd COMMotion::GetVelocityError(const RefVectorXd vel,
                                            const pinocchio::Model &model,
                                            pinocchio::Data &data,
                                            ConstRefVectorXd q,
                                            ConstRefVectorXd qv) const {

  return vel - (origin_.rotation().transpose() * data.vcom[0])(indexes_);
}

Eigen::MatrixXd COMMotion::GetJacobian(const pinocchio::Model &model,
                                       pinocchio::Data &data,
                                       ConstRefVectorXd q,
                                       ConstRefVectorXd qv) const {
  return (origin_.rotation().transpose() *
          pinocchio::jacobianCenterOfMass(model, data, q, false))(
      indexes_, Eigen::placeholders::all);
}

Eigen::VectorXd COMMotion::GetAcceleration(const pinocchio::Model &model,
                                           pinocchio::Data &data,
                                           ConstRefVectorXd q,
                                           ConstRefVectorXd qv) const {
  return (origin_.rotation().transpose() * data.acom[0])(indexes_);
}
} // namespace ftn_solo_control