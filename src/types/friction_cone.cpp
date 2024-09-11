#include <ftn_solo_control/types/friction_cone.h>

#include <iostream>

#include <Eigen/Dense>
#include <math.h>

namespace ftn_solo_control {
size_t FrictionCone::total_cones_ = 0;

SimpleConvexCone::SimpleConvexCone(size_t num_sides, ConstRefVector3d vector,
                                   double start_angle,
                                   ConstRefVector3d translation)
    : num_sides_(num_sides), translation_(translation) {
  face_ = Eigen::MatrixXd(num_sides, 3);
  span_ = Eigen::MatrixXd(3, num_sides);
  double angle = 2 * M_PI / num_sides;
  Eigen::Quaterniond q_start(
      Eigen::AngleAxisd(start_angle, Eigen::Vector3d::UnitZ()));
  Eigen::Quaterniond q_initial(
      Eigen::AngleAxisd(angle / 2, Eigen::Vector3d::UnitZ()));
  Eigen::Quaterniond q_relative(
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
  span_.col(0) = q_initial * q_start * vector.normalized();
  for (size_t i = 1; i < num_sides_; ++i) {
    span_.col(i) = q_relative * span_.col(i - 1);
  }
  face_.row(0) =
      (span_.block<3, 1>(0, 0).cross(span_.block<3, 1>(0, 1))).normalized();
  for (size_t i = 1; i < num_sides_; ++i) {
    face_.row(i) = q_relative * face_.row(i - 1);
  }
}

void SimpleConvexCone::Rotate(ConstRefMatrixXd rot) {
  span_ = rot * span_;
  face_ = face_ * rot.transpose();
}

FrictionCone::FrictionCone(double mu, size_t num_sides,
                           const pinocchio::SE3 &pose)
    : mu_(mu), num_sides_(num_sides), cone_num_(total_cones_++) {
  pose_ = pose;
  Eigen::Vector3d vector(mu, 0, 1);
  primal_ = SimpleConvexCone(num_sides_, vector, 0, pose.translation());
  dual_ = SimpleConvexCone(num_sides_, primal_.face_.row(0), M_PI / num_sides_,
                           pose.translation());
  primal_.Rotate(pose.rotation().matrix());
  dual_.Rotate(pose.rotation().matrix());
}

size_t GetTotalSides(const FrictionConeMap &friction_cones) {
  size_t result = 0;
  for (const auto &cone : friction_cones) {
    result += cone.second.GetNumSides();
  }
  return result;
}

} // namespace ftn_solo_control
