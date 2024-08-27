#include <ftn_solo_control/trajectories/piecewise_linear.h>
// common includes
#include <pinocchio/algorithm/frames.hpp>
// ftn_solo_control
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

namespace {
std::array<double, 3> Poly5(double t, double t_start, double t_end) {
  std::array<double, 3> result;
  double delta_t = t_end - t_start;
  double tau = (t - t_start) / delta_t;
  result[0] = (((6 * tau - 15) * tau) + 10) * tau * tau * tau;
  result[1] = (((30 * tau - 60) * tau) + 30) * tau * tau / delta_t;
  result[2] = (((120 * tau - 180) * tau) + 60) * tau / delta_t / delta_t;
  return result;
}

void ComputeInterpolation(const std::array<double, 3> s,
                          ConstRefVectorXd p_start, ConstRefVectorXd p_end,
                          RefVectorXd pos, RefVectorXd vel, RefVectorXd acc) {
  Eigen::VectorXd direction = p_end - p_start;
  pos = p_start + s[0] * direction;
  vel = s[1] * direction;
  acc = s[2] * direction;
}

void ComputeInterpolation(const std::array<double, 3> s,
                          ConstRefMatrix3d q_start, ConstRefMatrix3d q_end,
                          RefMatrix3d pos, RefVectorXd vel, RefVectorXd acc) {
  Eigen::Vector3d rotation_vector =
      pinocchio::log3(q_start.transpose() * q_end);
  pos = q_start * pinocchio::exp3(s[0] * rotation_vector);
  vel = s[1] * q_start * rotation_vector;
  acc = s[2] * q_start * rotation_vector;
}

} // namespace

template <class PosType, class PosTypeRef>
void PiecewiseLinearTrajectory<PosType, PosTypeRef>::CloseLoop(double t) {
  this->loop_ = true;
  this->AddPoint(this->points_.front(), t);
}

template <class PosType, class PosTypeRef>
void PiecewiseLinearTrajectory<PosType, PosTypeRef>::StartPoint(
    double trel, PosTypeRef pos, RefVectorXd val, RefVectorXd acc) const {
  pos = this->points_.front();
  val = ZeroVelocity();
  acc = ZeroVelocity();
}

template <class PosType, class PosTypeRef>
void PiecewiseLinearTrajectory<PosType, PosTypeRef>::EndPoint(
    double trel, PosTypeRef pos, RefVectorXd val, RefVectorXd acc) const {
  pos = this->points_.back();
  val = ZeroVelocity();
  acc = ZeroVelocity();
}

template <class PosType, class PosTypeRef>
void PiecewiseLinearTrajectory<PosType, PosTypeRef>::Interpolated(
    double t, PosTypeRef pos, RefVectorXd val, RefVectorXd acc) const {
  size_t segment = 0;
  while (t > this->times_.at(segment + 1)) {
    ++segment;
  }
  const auto s =
      Poly5(t, this->times_.at(segment), this->times_.at(segment + 1));
  ComputeInterpolation(s, this->points_.at(segment),
                       this->points_.at(segment + 1), pos, val, acc);
}

template <>
Eigen::VectorXd
PiecewiseLinearTrajectory<Eigen::VectorXd, RefVectorXd>::ZeroPosition() const {
  if (points_.size() > 0) {
    return 0 * points_.back();
  } else {
    return Eigen::Vector3d::Zero();
  }
}

template <>
Eigen::VectorXd
PiecewiseLinearTrajectory<Eigen::VectorXd, RefVectorXd>::ZeroVelocity() const {
  if (points_.size() > 0) {
    return 0 * points_.back();
  } else {
    return Eigen::Vector3d::Zero();
  }
}

template <>
Eigen::Matrix3d
PiecewiseLinearTrajectory<Eigen::Matrix3d, RefMatrix3d>::ZeroPosition() const {
  return Eigen::Matrix3d::Identity();
}

template <>
Eigen::VectorXd
PiecewiseLinearTrajectory<Eigen::Matrix3d, RefMatrix3d>::ZeroVelocity() const {
  return Eigen::VectorXd::Zero(3);
}

template class PiecewiseLinearTrajectory<Eigen::VectorXd, RefVectorXd>;
template class PiecewiseLinearTrajectory<Eigen::Matrix3d, RefMatrix3d>;
} // namespace ftn_solo_control
