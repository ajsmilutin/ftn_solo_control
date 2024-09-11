#include <ftn_solo_control/trajectories/spline.h>
// ftn_solo_control
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

void SplineTrajectory::AddPoint(const RefVectorXd pos, double t) {
  Trajectory::AddPoint(pos, t);
  BoundaryCondFull bc_type;
  bc_type[0] = BoundaryCond(BoundaryCond::Clamped, ZeroVelocity());
  bc_type[1] = BoundaryCond(BoundaryCond::Clamped, ZeroVelocity());
  if (follow_through_) {
    bc_type[1] = BoundaryCond(BoundaryCond::Natural, ZeroVelocity());
  }
  Eigen::VectorXd times = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      this->times_.data(), this->times_.size());
  if (this->points_.size() >= 2) {
    cubic_spline_ =
        PiecewisePolyPath::CubicSpline(this->points_, times, bc_type);
  }
}

void SplineTrajectory::StartPoint(double t, RefVectorXd pos, RefVectorXd vel,
                                  RefVectorXd acc) const {
  pos = this->points_.front();
  vel = ZeroVelocity();
  acc = ZeroVelocity();
}

void SplineTrajectory::EndPoint(double t, RefVectorXd pos, RefVectorXd vel,
                                RefVectorXd acc) const {
  if (follow_through_) {
    vel = cubic_spline_.EvalSingle(times_.back(), 1);
    pos = points_.back() + vel * (t-times_.back());
    acc = ZeroVelocity();
  } else {
    pos = this->points_.back();
    vel = ZeroVelocity();
    acc = ZeroVelocity();
  }
}

void SplineTrajectory::Interpolated(double t, RefVectorXd pos, RefVectorXd vel,
                                    RefVectorXd acc) const {
  pos = cubic_spline_.EvalSingle(t, 0);
  vel = cubic_spline_.EvalSingle(t, 1);
  acc = cubic_spline_.EvalSingle(t, 2);
}

Eigen::VectorXd SplineTrajectory::ZeroPosition() const {
  if (points_.size() > 0) {
    return 0 * points_.back();
  } else {
    return Eigen::Vector3d::Zero();
  }
}

Eigen::VectorXd SplineTrajectory::ZeroVelocity() const {
  if (points_.size() > 0) {
    return 0 * points_.back();
  } else {
    return Eigen::Vector3d::Zero();
  }
}
} // namespace ftn_solo_control
