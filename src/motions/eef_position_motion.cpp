#include <ftn_solo_control/motions/eef_position_motion.h>

namespace ftn_solo_control {

EEFPositionMotion::EEFPositionMotion(size_t eef_index, ConstRefVector3b selected,
                                 const pinocchio::SE3 &origin_pose, double Kp,
                                 double Kd)
    : MotionWithTrajectory(selected, Kp, Kd), eef_index_(eef_index),
      origin_(origin_pose) {}

Eigen::VectorXd EEFPositionMotion::GetPositionError(const RefVectorXd pos,
                                                  const pinocchio::Model &model,
                                                  pinocchio::Data &data) const {
  return pos - origin_.actInv(data.oMf[eef_index_].translation())(indexes_);
}

Eigen::VectorXd EEFPositionMotion::GetVelocityError(const RefVectorXd vel,
                                                  const pinocchio::Model &model,
                                                  pinocchio::Data &data) const {

  return vel - (origin_.rotation().transpose() *
                (pinocchio::getFrameVelocity(
                     model, data, eef_index_,
                     pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED))
                    .linear())(indexes_);
}

Eigen::MatrixXd EEFPositionMotion::GetJacobian(const pinocchio::Model &model,
                                             pinocchio::Data &data,
                                             ConstRefVectorXd q,
                                             ConstRefVectorXd qv) const {

  return (origin_.rotation().transpose() *
          pinocchio::getFrameJacobian(
              model, data, eef_index_,
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
              .topRows<3>())(indexes_, Eigen::placeholders::all);
}

Eigen::VectorXd EEFPositionMotion::GetAcceleration(const pinocchio::Model &model,
                                                 pinocchio::Data &data) const {
  return (origin_.rotation().transpose() *
          pinocchio::getFrameAcceleration(
              model, data, eef_index_,
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
              .linear())(indexes_);
}
} // namespace ftn_solo_control