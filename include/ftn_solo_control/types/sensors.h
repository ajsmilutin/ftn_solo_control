#pragma once

#include <Eigen/Dense>
// FTN includes
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

struct ImuData {
  ImuData() {}
  Eigen::Vector3d angular_velocity;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d magnetometer;
  Eigen::Vector4d attitude;
};

struct SensorData {
  SensorData() {}
  ImuData imu_data;
  Vector4b touch;
};

} // namespace ftn_solo_control