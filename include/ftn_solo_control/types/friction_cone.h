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
  FrictionCone(double mu = 0.8, size_t num_sides = 4,
               const pinocchio::SE3 &pose = pinocchio::SE3::Identity());
  SimpleConvexCone primal_;
  SimpleConvexCone dual_;
  inline size_t GetNum() const { return cone_num_; }
  inline Eigen::Vector3d GetPosition() const { return pose_.translation(); }
  inline pinocchio::SE3 GetPose() const { return pose_; }
  inline size_t GetNumSides() const { return num_sides_; }

protected:
  static size_t total_cones_;
  double mu_;
  size_t num_sides_;
  size_t cone_num_;
  pinocchio::SE3 pose_;
};

typedef std::map<size_t, FrictionCone> FrictionConeMap;

size_t GetTotalSides(const FrictionConeMap &friction_cones);

} // namespace ftn_solo_control
