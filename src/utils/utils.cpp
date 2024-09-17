#include <ftn_solo_control/utils/utils.h>

#include <Eigen/Dense>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>
// FTN solo includes
#include <ftn_solo_control/motions/motion.h>
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/types/friction_cone.h>

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

void GetConstraintJacobian(
    const pinocchio::Model &model, pinocchio::Data &data,
    const FrictionConeMap &friction_cones, RefMatrixXd constraint,
    std::map<size_t, pinocchio::SE3> *touching_poses,
    std::map<size_t, pinocchio::SE3> *placements) {

  std::map<size_t, pinocchio::SE3> poses;
  std::transform(friction_cones.cbegin(), friction_cones.cend(),
                 std::inserter(poses, poses.end()),
                 [](const std::pair<size_t, FrictionCone> &friction_cone) {
                   return std::make_pair(friction_cone.first,
                                         friction_cone.second.GetPose());
                 });
  GetConstraintJacobian(model, data, poses, constraint, touching_poses,
                        placements);
}

Eigen::MatrixXd GetContactJacobian(const pinocchio::Model &model,
                                   pinocchio::Data &data, size_t eef_index,
                                   const FrictionCone &friction_cone,
                                   pinocchio::SE3 *touching_pose,
                                   pinocchio::SE3 *placement) {
  return GetContactJacobian(model, data, eef_index, friction_cone.GetPose(),
                            touching_pose, placement);
}

Eigen::MatrixXd GetContactJacobian(const pinocchio::Model &model,
                                   pinocchio::Data &data, size_t eef_index,
                                   const pinocchio::SE3 &pose,
                                   pinocchio::SE3 *touching_pose_ptr,
                                   pinocchio::SE3 *placement_ptr) {
  pinocchio::SE3 tmp_pose, tmp_placement;
  tmp_pose = GetTouchingPose(model, data, eef_index, pose.rotation().col(2));
  const auto &frame = model.frames.at(eef_index);
  tmp_placement = GetTouchingPlacement(model, data, eef_index, tmp_pose);
  if (touching_pose_ptr) {
    *touching_pose_ptr = tmp_pose;
  }
  if (placement_ptr) {
    *placement_ptr = tmp_placement;
  }
  return pinocchio::getFrameJacobian(
             model, data, frame.parentJoint, tmp_placement,
             pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
      .topRows<3>();
}

void GetConstraintJacobian(
    const pinocchio::Model &model, pinocchio::Data &data,
    const std::map<size_t, pinocchio::SE3> &poses,
    RefMatrixXd constraint,
    std::map<size_t, pinocchio::SE3> *touching_poses,
    std::map<size_t, pinocchio::SE3> *placements) {
  size_t i = 0;
  size_t num_constraints = poses.size() * 3;
  pinocchio::SE3 *touching_pose_ptr = nullptr;
  pinocchio::SE3 *placement_ptr = nullptr;
  for (const auto &position : poses) {
    if (touching_poses) {
      touching_pose_ptr = &((*touching_poses)[position.first]);
    }
    if (placements) {
      placement_ptr = &((*placements)[position.first]);
    }
    constraint.middleRows<3>(i * 3) =
        GetContactJacobian(model, data, position.first, position.second,
                           touching_pose_ptr, placement_ptr);
    ++i;
  }
  constraint;
}

size_t GetMotionsDim(const std::vector<boost::shared_ptr<Motion>> &motions) {
  size_t motions_dim = 0;
  for (const auto &motion : motions) {
    motions_dim += motion->dim_;
  }
  return motions_dim;
}

void GetMotionsJacobian(const pinocchio::Model &model, pinocchio::Data &data,
                        ConstRefVectorXd q, ConstRefVectorXd qv,
                        const std::vector<boost::shared_ptr<Motion>> &motions,
                        RefMatrixXd jacobian) {
  size_t start_row = 0;
  for (const auto &motion : motions) {
    jacobian.middleRows(start_row, motion->dim_) =
        motion->GetJacobian(model, data, q, qv);
    start_row += motion->dim_;
  }
}

Eigen::Matrix3d CrossMatrix(ConstRefVector3d pos) {
  return (Eigen::Matrix3d() << 0, -pos[2], pos[1], pos[2], 0, -pos[0], -pos[1],
          pos[0], 0)
      .finished();
}

} // namespace ftn_solo_control