#pragma once

#include <ftn_solo_control/trajectories/trajectory.h>
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/types/piecewise_poly_path.h>

namespace ftn_solo_control {

class SplineTrajectory : public Trajectory<Eigen::VectorXd, RefVectorXd,
                                           Eigen::VectorXd, RefVectorXd> {
public:
  SplineTrajectory(bool follow_through)
      : Trajectory(), follow_through_(follow_through) {}
  void AddPoint(const RefVectorXd point, double t) override;

  Eigen::VectorXd ZeroPosition() const override;
  Eigen::VectorXd ZeroVelocity() const override;

protected:
  void StartPoint(double trel, RefVectorXd pos, RefVectorXd val,
                  RefVectorXd acc) const override;

  void EndPoint(double trel, RefVectorXd pos, RefVectorXd val,
                RefVectorXd acc) const override;

  void Interpolated(double t, RefVectorXd pos, RefVectorXd val,
                    RefVectorXd acc) const override;

  bool follow_through_;
  PiecewisePolyPath cubic_spline_;
};

} // namespace ftn_solo_control
