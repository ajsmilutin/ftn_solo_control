#include <ftn_solo_control/utils/utils.h>

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>
// FTN solo includes
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

pinocchio::SE3 GetTouchingPose(const pinocchio::Model &model,
                               const pinocchio::Data &data, size_t frame_index,
                               ConstRefVector3d normal) {
  pinocchio::SE3 result;
  const auto &pose = data.oMf.at(frame_index);
  result.rotation() = pose.rotation();
  result.rotation().col(0) =
      result.rotation().col(1).cross(normal).normalized();
  result.rotation().col(2) =
      result.rotation().col(0).cross(result.rotation().col(1));
  result.translation() = -kRadius * result.rotation().col(2);
  return result;
}

pinocchio::SE3 GetTouchingPlacement(const pinocchio::Model &model,
                                    const pinocchio::Data &data,
                                    size_t frame_index,
                                    const pinocchio::SE3 &pose) {
  return model.frames.at(frame_index).placement *
         pinocchio::SE3(data.oMf.at(frame_index).rotation(),
                        Eigen::Vector3d::Zero())
             .inverse() *
         pose;
}

} // namespace ftn_solo_control