#pragma once

#include <ftn_solo_control/trajectories/trajectory.h>
// Common includes
#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>
// ftn_solo_control
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

template <class PosType, class PosTypeRef>
class PiecewiseLinearTrajectory
    : public Trajectory<PosType, PosTypeRef, Eigen::VectorXd, RefVectorXd> {

public:
  void CloseLoop(double t);

  PosType ZeroPosition() const override;
  Eigen::VectorXd ZeroVelocity() const override;

protected:
  void StartPoint(double trel, PosTypeRef pos, RefVectorXd val,
                  RefVectorXd acc) const override;

  void EndPoint(double trel, PosTypeRef pos, RefVectorXd val,
                RefVectorXd acc) const override;

  void Interpolated(double t, PosTypeRef pos, RefVectorXd val,
                    RefVectorXd acc) const override;
};

} // namespace  ftn_solo_control
