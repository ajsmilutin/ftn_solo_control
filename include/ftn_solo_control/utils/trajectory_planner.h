#pragma once
// Common includes
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
// ftn_solo_control includes
#include <ftn_solo_control/motions/motion.h>
#include <ftn_solo_control/trajectories/piecewise_linear.h>
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/types/convex_hull_2d.h>
#include <ftn_solo_control/types/friction_cone.h>

namespace ftn_solo_control {

void InitTrajectoryPlannerPublisher();

class TrajectoryPlanner {
public:
  TrajectoryPlanner(const pinocchio::Model &model, size_t base_index,
                    const pinocchio::SE3 &origin);

  TrajectoryPlanner(const TrajectoryPlanner &other);

  ~TrajectoryPlanner();

  void StartComputation(double t, const FrictionConeMap &friction_cones,
                        const FrictionConeMap &next_friction_cones,
                        ConstRefVectorXd q, double duration,
                        double torso_height, double max_torque);

  void UpdateEEFTrajectory(double t, const FrictionConeMap &friction_cones,
                           std::vector<boost::shared_ptr<Motion>> eef_motions,
                           std::vector<boost::shared_ptr<Motion>> joint_motions,
                           ConstRefVectorXd q, double duration);
  Eigen::Vector2d com_xy_;
  Eigen::VectorXd q_;
  inline std::vector<boost::shared_ptr<Motion>> Motions() const {
    return motions_;
  }
  inline bool ComputationStarted() const { return computation_started_; }
  inline bool ComputationDone() const { return computation_done_; }
  inline bool UpdateStarted() const { return update_started_; }
  inline bool UpdateDone() const { return update_done_; }

private:
  void DoComputation(double t, FrictionConeMap friction_cones,
                     FrictionConeMap next_friction_cones, double duration,
                     double torso_height, double max_torque);

  void DoUpdate(double t, FrictionConeMap friction_cones,
                std::vector<boost::shared_ptr<Motion>> eef_motions,
                std::vector<boost::shared_ptr<Motion>> joint_motions,
                double duration);

  typedef PiecewiseLinearTrajectory<Eigen::VectorXd, RefVectorXd>
      PieceWiseLinearPosition;
  typedef PiecewiseLinearTrajectory<Eigen::Matrix3d, RefMatrix3d>
      PieceWiseLinearRotation;

  Eigen::Vector2d ComputeCoMPos(const ConvexHull2D &wcm,
                                ConstRefVector2d com_pos,
                                pinocchio::SE3 origin);
  boost::shared_ptr<PieceWiseLinearPosition> com_trajectory_;
  boost::shared_ptr<PieceWiseLinearPosition> base_trajectory_;
  boost::shared_ptr<PieceWiseLinearRotation> rotation_trajectory_;
  std::vector<boost::shared_ptr<Motion>> motions_;
  pinocchio::Model model_;
  pinocchio::Data data_;
  size_t base_index_;
  pinocchio::SE3 origin_;
  std::atomic<bool> computation_started_;
  std::atomic<bool> computation_done_;
  std::atomic<bool> update_started_;
  std::atomic<bool> update_done_;
  std::thread thread_;
};

} // namespace ftn_solo_control
