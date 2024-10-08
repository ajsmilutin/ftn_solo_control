#include <ftn_solo_control/motions/eef_rotation_motion.h>

namespace ftn_solo_control {

EEFRotationMotion::EEFRotationMotion(size_t eef_index, double Kp, double Kd)
    : MotionWithTrajectory(Vector3b::Constant(true), Kp, Kd),
      eef_index_(eef_index) {}

Eigen::VectorXd EEFRotationMotion::GetPositionError(
    const RefMatrix3d rot, const pinocchio::Model &model, pinocchio::Data &data,
    ConstRefVectorXd q, ConstRefVectorXd qv) const {
  const auto &ori = data.oMf[eef_index_].rotation();
  const auto err = ori * pinocchio::log3(ori.transpose() * rot);
  return ori * pinocchio::log3(ori.transpose() * rot);
}

Eigen::VectorXd EEFRotationMotion::GetVelocityError(
    const RefVectorXd vel, const pinocchio::Model &model, pinocchio::Data &data,
    ConstRefVectorXd q, ConstRefVectorXd qv) const {

  return vel - pinocchio::getFrameVelocity(
                   model, data, eef_index_,
                   pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
                   .angular();
}

Eigen::MatrixXd EEFRotationMotion::GetJacobian(const pinocchio::Model &model,
                                               pinocchio::Data &data,
                                               ConstRefVectorXd q,
                                               ConstRefVectorXd qv) const {

  return pinocchio::getFrameJacobian(
             model, data, eef_index_,
             pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
      .bottomRows<3>();
}

Eigen::VectorXd
EEFRotationMotion::GetAcceleration(const pinocchio::Model &model,
                                   pinocchio::Data &data, ConstRefVectorXd q,
                                   ConstRefVectorXd qv) const {
  return pinocchio::getFrameAcceleration(
             model, data, eef_index_,
             pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
      .angular();
}
} // namespace ftn_solo_control