#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>
// FTN solo includes
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

pinocchio::SE3 GetTouchingPose(const pinocchio::Model &model,
                                const pinocchio::Data &data, size_t frame_index,
                                ConstRefVector3d normal);

pinocchio::SE3 GetTouchingPlacement(const pinocchio::Model &model,
                                     const pinocchio::Data &data,
                                     size_t frame_index,
                                     const pinocchio::SE3 &pose);

} // namespace ftn_solo_control