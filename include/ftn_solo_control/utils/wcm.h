#pragma once
// Common includes
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
// ftn_solo_includes
#include <ftn_solo_control/types/convex_hull_2d.h>
#include <ftn_solo_control/types/friction_cone.h>

namespace ftn_solo_control {

ConvexHull2D GetProjectedWCM(
    const FrictionConeMap &friction_cones,
    const Eigen::MatrixXd &torque_constraint = Eigen::MatrixXd(0, 0),
    const Eigen::VectorXd &lb = Eigen::VectorXd(0),
    const Eigen::VectorXd &ub = Eigen::VectorXd(0),
    const Eigen::Vector3d &acom = Eigen::Vector3d::Zero(),
    const ConvexHull2D &hull_constraint = ConvexHull2D());

ConvexHull2D
GetProjectedWCMWithTorque(const pinocchio::Model &model, pinocchio::Data &data,
                          const FrictionConeMap &friction_cones,
                          double torque_limit,
                          const Eigen::Vector3d &acom = Eigen::Vector3d::Zero(),
                          const ConvexHull2D &hull_constraint = ConvexHull2D());
} // namespace ftn_solo_control
