#pragma once
// Stl includes
// Common includes
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <pinocchio/spatial/se3.hpp>
// ftn_solo_control includes
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

geometry_msgs::msg::Point ToPoint(ConstRefVector3d vector);
geometry_msgs::msg::Pose ToPose(const pinocchio::SE3 &pose);
geometry_msgs::msg::Vector3 ToVector(ConstRefVector3d vector);

geometry_msgs::msg::Quaternion ToQuaternion(ConstRefMatrix3d rotation);
geometry_msgs::msg::Quaternion
ToQuaternion(const Eigen::Quaterniond &quaternion);

} // namespace ftn_solo_control
