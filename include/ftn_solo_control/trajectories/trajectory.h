#pragma once
// Stl includes
#include <cmath>
#include <iostream>
#include <vector>
// ftn_solo_control
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

template <class PosType, class PosTypeRef, class VelType, class VelTypeRef>
class Trajectory {
public:
  typedef PosType PositionType;
  typedef PosTypeRef PositionTypeRef;
  typedef VelType VelocityType;
  typedef VelTypeRef VelocityTypeRef;
  Trajectory()
      : finished_(false), loop_(false), points_({}), times_({}),
        start_time_(-1), end_time_(-1) {}
  void SetStart(double t) {
    start_time_ = t;
    if (times_.size() > 0) {
      end_time_ = start_time_ + times_.back();
    }
  }

  virtual void AddPoint(const PosTypeRef point, double t) {
    times_.push_back(t);
    points_.push_back(point);
  }

  virtual void Get(double t, PosTypeRef pos, VelTypeRef vel, VelTypeRef acc) {
    double start = 0;
    if (start_time_ > 0) {
      start = start_time_;
    }
    double trel = t - start;
    if (trel < 0) {
      StartPoint(trel, pos, vel, acc);
      return;
    }
    if (loop_) {
      trel = std::fmod(trel, times_.back());
    }
    if (trel >= times_.back()) {
      finished_ = true;
      EndPoint(trel, pos, vel, acc);
      return;
    }
    Interpolated(trel, pos, vel, acc);
  }

  bool finished_;

  inline double Duration() {
    if (times_.size() > 0) {
      return times_.back();
    } else {
      return 0;
    }
  }

  inline double StartTime() { return start_time_; }
  inline double EndTime() { return end_time_; }

  virtual PosType ZeroPosition() const = 0;
  virtual VelType ZeroVelocity() const = 0;

protected:
  virtual void StartPoint(double t, PosTypeRef pos, VelTypeRef val,
                          VelTypeRef acc) const = 0;
  virtual void EndPoint(double t, PosTypeRef pos, VelTypeRef al,
                        VelTypeRef acc) const = 0;
  virtual void Interpolated(double t, PosTypeRef pos, VelTypeRef val,
                            VelTypeRef acc) const = 0;

protected:
  bool loop_;
  std::vector<PosType, Eigen::aligned_allocator<PosType>> points_;
  std::vector<double> times_;
  double start_time_;
  double end_time_;
};
} // namespace ftn_solo_control