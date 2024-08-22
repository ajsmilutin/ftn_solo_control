#include <ftn_solo_control/utils/conversions.h>
// Common includes
#include <geometry_msgs/msg/point.hpp>
#include <pinocchio/spatial/se3.hpp>
// ftn_solo_control includes
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

geometry_msgs::msg::Point ToPoint(ConstRefVector3d vector) {
  geometry_msgs::msg::Point point;
  point.x = vector.x();
  point.y = vector.y();
  point.z = vector.z();
  return point;
}

geometry_msgs::msg::Pose ToPose(const pinocchio::SE3 &pose) {
  geometry_msgs::msg::Pose pose_msg;
  pose_msg.position = ToPoint(pose.translation());
  pose_msg.orientation = ToQuaternion(pose.rotation());
  return pose_msg;
}

geometry_msgs::msg::Quaternion ToQuaternion(ConstRefMatrix3d rotation) {
  return ToQuaternion(Eigen::Quaterniond(rotation));
}

geometry_msgs::msg::Quaternion
ToQuaternion(const Eigen::Quaterniond &quaternion) {
  geometry_msgs::msg::Quaternion q;
  q.x = quaternion.x();
  q.y = quaternion.y();
  q.z = quaternion.z();
  q.w = quaternion.w();
  return q;
}

} // namespace ftn_solo_control