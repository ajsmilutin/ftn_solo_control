#include <ftn_solo_control/motions/bindings.h>
// FTN solo includes
#include <ftn_solo_control/motions/com_motion.h>
#include <ftn_solo_control/motions/eef_position_motion.h>
#include <ftn_solo_control/motions/eef_rotation_motion.h>
#include <ftn_solo_control/motions/joint_motion.h>
#include <ftn_solo_control/motions/solver.h>
#include <ftn_solo_control/trajectories/piecewise_linear.h>
#include <ftn_solo_control/trajectories/spline.h>

// Boost Python
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <eigenpy/eigen-from-python.hpp>
#include <eigenpy/eigenpy.hpp>

namespace ftn_solo_control {

namespace {

template <class PosType, class PosTypeRef>
class PieceWiseLinearWrapper
    : public PiecewiseLinearTrajectory<PosType, PosTypeRef> {
public:
  boost::python::tuple GetValues(double t) {
    PosType pos = this->ZeroPosition();
    Eigen::VectorXd vel = this->ZeroVelocity();
    Eigen::VectorXd acc = this->ZeroVelocity();
    this->Get(t, pos, vel, acc);
    return boost::python::make_tuple(pos, vel, acc);
  }
};

class SplineTrajectoryWrapper : public SplineTrajectory {
public:
  SplineTrajectoryWrapper(bool follow_through = false)
      : SplineTrajectory(follow_through) {}
  boost::python::tuple GetValues(double t) {
    Eigen::VectorXd pos = this->ZeroPosition();
    Eigen::VectorXd vel = this->ZeroVelocity();
    Eigen::VectorXd acc = this->ZeroVelocity();
    this->Get(t, pos, vel, acc);
    return boost::python::make_tuple(pos, vel, acc);
  }
};

typedef PieceWiseLinearWrapper<Eigen::VectorXd, RefVectorXd>
    PieceWiseLinearPositionWrapper;
typedef PieceWiseLinearWrapper<Eigen::Matrix3d, RefMatrix3d>
    PieceWiseLinearRotationWrapper;

typedef PieceWiseLinearWrapper<Eigen::Matrix3d, RefMatrix3d>
    PieceWiseLinearRotationWrapper;

namespace bp = boost::python;

} // namespace

void ExposeMotionsAndTrajectories() {
  class EEFPositionMotionWrapper : public EEFPositionMotion {
  public:
    using EEFPositionMotion::EEFPositionMotion;
    void SetTrajectoryLinearPosition(
        const boost::shared_ptr<PieceWiseLinearPositionWrapper> &trajectory) {
      EEFPositionMotion::SetTrajectory(trajectory);
    }
    void SetTrajectorySpline(
        const boost::shared_ptr<SplineTrajectoryWrapper> &trajectory) {
      EEFPositionMotion::SetTrajectory(trajectory);
    }
  };

  class EEFRotationMotionWrapper : public EEFRotationMotion {
  public:
    using EEFRotationMotion::EEFRotationMotion;
    void SetTrajectoryLinearRotation(
        const boost::shared_ptr<PieceWiseLinearRotationWrapper> &trajectory) {
      EEFRotationMotion::SetTrajectory(trajectory);
    }
  };

  class COMMotionWrapper : public COMMotion {
  public:
    using COMMotion::COMMotion;
    void SetTrajectoryLinearPosition(
        const boost::shared_ptr<PieceWiseLinearPositionWrapper> &trajectory) {
      COMMotion::SetTrajectory(trajectory);
    }
    void SetTrajectorySpline(
        const boost::shared_ptr<SplineTrajectoryWrapper> &trajectory) {
      COMMotion::SetTrajectory(trajectory);
    }
  };

  class JointMotionWrapper : public JointMotion {
  public:
    using JointMotion::JointMotion;
    void SetTrajectorySpline(
        const boost::shared_ptr<SplineTrajectoryWrapper> &trajectory) {
      JointMotion::SetTrajectory(trajectory);
    }
  };
  bp::class_<PieceWiseLinearPositionWrapper>("PieceWiseLinearPosition",
                                             bp::init<>())
      .def("add", &PieceWiseLinearPositionWrapper::AddPoint)
      .def("close_loop", &PieceWiseLinearPositionWrapper::CloseLoop)
      .def("get", &PieceWiseLinearPositionWrapper::GetValues)
      .def("set_start", &PieceWiseLinearPositionWrapper::SetStart)
      .def_readonly("finished", &PieceWiseLinearPositionWrapper::finished_)
      .def("duration", &PieceWiseLinearPositionWrapper::Duration)
      .def("start_time", &PieceWiseLinearPositionWrapper::StartTime)
      .def("end_time", &PieceWiseLinearPositionWrapper::EndTime);

  bp::class_<PieceWiseLinearRotationWrapper>("PieceWiseLinearRotation",
                                             bp::init<>())
      .def("add", &PieceWiseLinearRotationWrapper::AddPoint)
      .def("close_loop", &PieceWiseLinearPositionWrapper::CloseLoop)
      .def("get", &PieceWiseLinearRotationWrapper::GetValues)
      .def("set_start", &PieceWiseLinearRotationWrapper::SetStart)
      .def_readonly("finished", &PieceWiseLinearRotationWrapper::finished_)
      .def("duration", &PieceWiseLinearRotationWrapper::Duration)
      .def("start_time", &PieceWiseLinearRotationWrapper::StartTime)
      .def("end_time", &PieceWiseLinearRotationWrapper::EndTime);

  bp::class_<SplineTrajectoryWrapper>("SplineTrajectory", bp::init<bool>())
      .def("add", &SplineTrajectoryWrapper::AddPoint)
      .def("get", &SplineTrajectoryWrapper::GetValues)
      .def("set_start", &SplineTrajectoryWrapper::SetStart)
      .def_readonly("finished", &SplineTrajectoryWrapper::finished_)
      .def("duration", &SplineTrajectoryWrapper::Duration)
      .def("start_time", &SplineTrajectoryWrapper::StartTime)
      .def("end_time", &SplineTrajectoryWrapper::EndTime);

  bp::class_<Motion>("motion", bp::init<double, double>())
      .def("set_priority", &Motion::SetPriority)
      .def("finished", &Motion::Finished)
      .def("set_start", &Motion::SetStart)
      .def("get_alpha", &Motion::GetAlpha);

  bp::class_<EEFPositionMotionWrapper, bp::bases<Motion>>(
      "EEFPositionMotion", bp::init<size_t, ConstRefVector3b,
                                    const pinocchio::SE3 &, double, double>())
      .def("set_trajectory",
           &EEFPositionMotionWrapper::SetTrajectoryLinearPosition)
      .def("set_trajectory", &EEFPositionMotionWrapper::SetTrajectorySpline)
      .def("get_jacobian", &EEFPositionMotionWrapper::GetJacobian)
      .def("get_desired_acceleration",
           &EEFPositionMotionWrapper::GetDesiredAcceleration)
      .def("get_acceleration", &EEFPositionMotionWrapper::GetAcceleration)
      .def_readonly("dim", &EEFPositionMotionWrapper::dim_)
      .def_readonly("trajectory", &EEFPositionMotionWrapper::trajectory_);

  bp::class_<EEFRotationMotionWrapper, bp::bases<Motion>>(
      "EEFRotationMotion", bp::init<size_t, double, double>())
      .def("set_trajectory",
           &EEFRotationMotionWrapper::SetTrajectoryLinearRotation)
      .def("get_jacobian", &EEFRotationMotionWrapper::GetJacobian)
      .def("get_desired_acceleration",
           &EEFRotationMotionWrapper::GetDesiredAcceleration)
      .def("get_acceleration", &EEFRotationMotionWrapper::GetAcceleration)
      .def_readonly("dim", &EEFRotationMotionWrapper::dim_)
      .def_readonly("trajectory", &EEFRotationMotionWrapper::trajectory_);

  bp::class_<COMMotionWrapper, bp::bases<Motion>>(
      "COMMotion",
      bp::init<ConstRefVector3b, const pinocchio::SE3 &, double, double>())
      .def("set_trajectory", &COMMotionWrapper::SetTrajectoryLinearPosition)
      .def("set_trajectory", &COMMotionWrapper::SetTrajectorySpline)
      .def("get_jacobian", &COMMotionWrapper::GetJacobian)
      .def("get_desired_acceleration",
           &COMMotionWrapper::GetDesiredAcceleration)
      .def("get_acceleration", &COMMotionWrapper::GetAcceleration)
      .def_readonly("dim", &COMMotionWrapper::dim_)
      .def_readonly("trajectory", &COMMotionWrapper::trajectory_);

  bp::class_<JointMotionWrapper, bp::bases<Motion>>(
      "JointMotion", bp::init<ConstRefVectorXi, double, double>())
      .def("set_trajectory", &JointMotionWrapper::SetTrajectorySpline)
      .def("get_jacobian", &JointMotionWrapper::GetJacobian)
      .def("get_desired_acceleration",
           &JointMotionWrapper::GetDesiredAcceleration)
      .def("get_acceleration", &JointMotionWrapper::GetAcceleration)
      .def_readonly("dim", &JointMotionWrapper::dim_)
      .def_readonly("trajectory", &JointMotionWrapper::trajectory_);

  bp::class_<std::vector<boost::shared_ptr<Motion>>>("MotionsVector")
      .def(bp::vector_indexing_suite<std::vector<boost::shared_ptr<Motion>>,
                                     true>());

  bp::register_ptr_to_python<boost::shared_ptr<Motion>>();
  bp::register_ptr_to_python<boost::shared_ptr<EEFPositionMotionWrapper>>();
  bp::register_ptr_to_python<boost::shared_ptr<EEFRotationMotionWrapper>>();
  bp::register_ptr_to_python<boost::shared_ptr<COMMotionWrapper>>();

  bp::def("get_end_of_motion", &GetEndOfMotion,
          (bp::arg("model"), bp::arg("data"), bp::arg("friction_cones"),
           bp::arg("motions"), bp::arg("q")),
          "Computes robot pose at the end of motion");

  bp::def("get_end_of_motion_prioritized", &GetEndOfMotionPrioritized,
          (bp::arg("model"), bp::arg("data"), bp::arg("friction_cones"),
           bp::arg("motions"), bp::arg("q")),
          "Computes robot pose at the end of motion");
}

} // namespace ftn_solo_control