#include <ftn_solo_control/motions/joint_motion.h>

namespace ftn_solo_control {

JointMotion::JointMotion(ConstRefVectorXi joints, double Kp, double Kd)
    : MotionWithTrajectory(joints, Kp, Kd) {
  indexes_6_ = indexes_ + Eigen::VectorXi::Constant(indexes_.size(), 6);
  indexes_7_ = indexes_6_ + Eigen::VectorXi::Ones(indexes_.size());
}

Eigen::VectorXd JointMotion::GetPositionError(const RefVectorXd pos,
                                              const pinocchio::Model &model,
                                              pinocchio::Data &data,
                                              ConstRefVectorXd q,
                                              ConstRefVectorXd qv) const {
  return pos - q(indexes_7_);
}

Eigen::VectorXd JointMotion::GetVelocityError(const RefVectorXd vel,
                                              const pinocchio::Model &model,
                                              pinocchio::Data &data,
                                              ConstRefVectorXd q,
                                              ConstRefVectorXd qv) const {

  return vel - qv(indexes_6_);
}

Eigen::MatrixXd JointMotion::GetJacobian(const pinocchio::Model &model,
                                         pinocchio::Data &data,
                                         ConstRefVectorXd q,
                                         ConstRefVectorXd qv) const {
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(dim_, model.nv);
  for (size_t i = 0; i < dim_; ++i) {
    jacobian(i, indexes_6_(i)) = 1;
  }
  return jacobian;
}

Eigen::VectorXd JointMotion::GetAcceleration(const pinocchio::Model &model,
                                             pinocchio::Data &data,
                                             ConstRefVectorXd q,
                                             ConstRefVectorXd qv) const {
  return Eigen::VectorXd::Zero(dim_);
}

} // namespace ftn_solo_control