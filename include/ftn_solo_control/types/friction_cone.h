#pragma once
// Stl includes
#include <string>
// Common includes
#include <Eigen/Dense>
#include <pinocchio/spatial/se3.hpp>
// FTN includes
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

class SimpleConvexCone {
public:
  SimpleConvexCone() {}
  SimpleConvexCone(size_t num_sides, ConstRefVector3d vector,
                   double start_angle, ConstRefVector3d translation);

  void Rotate(ConstRefMatrixXd rot);
  inline size_t GetNumSides() const { return num_sides_; }

  Eigen::MatrixXd face_;
  Eigen::MatrixXd span_;

protected:
  size_t num_sides_;
  Eigen::Vector3d translation_;
};

class FrictionCone {
public:
  FrictionCone(double mu, size_t num_sides, const pinocchio::SE3 &pose);
  SimpleConvexCone primal_;
  SimpleConvexCone dual_;
  inline size_t GetNum() const { return cone_num_; }
  inline ConstRefVector3d GetPosition() const { return pose_.translation(); }

protected:
  static size_t total_cones_;
  double mu_;
  size_t num_sides_;
  size_t cone_num_;
  pinocchio::SE3 pose_;
};

} // namespace ftn_solo_control
