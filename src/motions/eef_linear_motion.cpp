#include <ftn_solo_control/motions/eef_linear_motion.h>

namespace ftn_solo_control {

Eigen::VectorXd EEFLinearMotion::GetPositionError(const RefVectorXd pos,
                                                  const pinocchio::Model &model,
                                                  pinocchio::Data &data) const {
  return pos - origin_.actInv(data.oMf[eef_index_].translation())(indexes_);
}

Eigen::VectorXd EEFLinearMotion::GetVelocityError(const RefVectorXd vel,
                                                  const pinocchio::Model &model,
                                                  pinocchio::Data &data) const {

  return vel - (origin_.rotation().transpose() *
                (pinocchio::getFrameVelocity(
                     model, data, eef_index_,
                     pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED))
                    .linear())(indexes_);
}

Eigen::MatrixXd EEFLinearMotion::GetJacobian(const pinocchio::Model &model,
                                             pinocchio::Data &data,
                                             ConstRefVectorXd q,
                                             ConstRefVectorXd qv) const {

  return (origin_.rotation().transpose() *
          pinocchio::getFrameJacobian(
              model, data, eef_index_,
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
              .topRows<3>())(indexes_, Eigen::placeholders::all);
}

Eigen::VectorXd EEFLinearMotion::GetAcceleration(const pinocchio::Model &model,
                                                 pinocchio::Data &data) const {
  return (origin_.rotation().transpose() *
          pinocchio::getFrameAcceleration(
              model, data, eef_index_,
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
              .linear())(indexes_);
}
} // namespace ftn_solo_control