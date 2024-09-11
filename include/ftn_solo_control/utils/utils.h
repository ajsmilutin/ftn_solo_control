#pragma once

#include <Eigen/Dense>
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
                               ConstRefVector3d normal);

pinocchio::SE3 GetTouchingPlacement(const pinocchio::Model &model,
                                    const pinocchio::Data &data,
                                    size_t frame_index,
                                    const pinocchio::SE3 &pose);

Eigen::MatrixXd GetContactJacobian(const pinocchio::Model &model,
                                   pinocchio::Data &data, size_t eef_index,
                                   const FrictionCone &friction_cone,
                                   pinocchio::SE3 *touching_pose = nullptr,
                                   pinocchio::SE3 *placement = nullptr);

Eigen::MatrixXd GetContactJacobian(const pinocchio::Model &model,
                                   pinocchio::Data &data, size_t eef_index,
                                   const pinocchio::SE3 &pose,
                                   pinocchio::SE3 *touching_pose_ptr = nullptr,
                                   pinocchio::SE3 *placement_ptr = nullptr);

void GetConstraintJacobian(
    const pinocchio::Model &model, pinocchio::Data &data,
    const FrictionConeMap &friction_cones, RefMatrixXd constraint,
    std::unordered_map<size_t, pinocchio::SE3> *touching_poses = nullptr,
    std::unordered_map<size_t, pinocchio::SE3> *placements = nullptr);

void GetConstraintJacobian(
    const pinocchio::Model &model, pinocchio::Data &data,
    const std::unordered_map<size_t, pinocchio::SE3> &poses,
    RefMatrixXd constraint,
    std::unordered_map<size_t, pinocchio::SE3> *touching_poses = nullptr,
    std::unordered_map<size_t, pinocchio::SE3> *placements = nullptr);

size_t GetMotionsDim(const std::vector<boost::shared_ptr<Motion>> &motions);

void GetMotionsJacobian(const pinocchio::Model &model, pinocchio::Data &data,
                        ConstRefVectorXd q, ConstRefVectorXd qv,
                        const std::vector<boost::shared_ptr<Motion>> &motions,
                        RefMatrixXd jacobian);

Eigen::Matrix3d CrossMatrix(ConstRefVector3d pos);

} // namespace ftn_solo_control