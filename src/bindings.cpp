
#include <rclcpp/rclcpp.hpp>
// FTN solo includes
#include <ftn_solo_control/controllers/whole_body.h>
#include <ftn_solo_control/estimators.h>
#include <ftn_solo_control/motions/com_motion.h>
#include <ftn_solo_control/motions/eef_position_motion.h>
#include <ftn_solo_control/motions/eef_rotation_motion.h>
#include <ftn_solo_control/motions/joint_motion.h>
#include <ftn_solo_control/motions/solver.h>
#include <ftn_solo_control/trajectories/piecewise_linear.h>
#include <ftn_solo_control/trajectories/spline.h>
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/types/convex_hull_2d.h>
#include <ftn_solo_control/types/friction_cone.h>
#include <ftn_solo_control/types/sensors.h>
#include <ftn_solo_control/utils/trajectory_planner.h>
#include <ftn_solo_control/utils/utils.h>
#include <ftn_solo_control/utils/visualization_utils.h>
#include <ftn_solo_control/utils/wcm.h>

// Boost Python
#include <boost/python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <eigenpy/eigen-from-python.hpp>
#include <eigenpy/eigenpy.hpp>

using namespace ftn_solo_control;

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

Eigen::MatrixXd get_contact_jacobian_proxy(const pinocchio::Model &model,
                                           pinocchio::Data &data,
                                           size_t eef_index,
                                           const FrictionCone &friction_cone) {
  return GetContactJacobian(model, data, eef_index, friction_cone.GetPose());
}

typedef PieceWiseLinearWrapper<Eigen::VectorXd, RefVectorXd>
    PieceWiseLinearPositionWrapper;
typedef PieceWiseLinearWrapper<Eigen::Matrix3d, RefMatrix3d>
    PieceWiseLinearRotationWrapper;

typedef PieceWiseLinearWrapper<Eigen::Matrix3d, RefMatrix3d>
    PieceWiseLinearRotationWrapper;

ConvexHull2D get_projected_wcm_proxy(const FrictionConeMap &friction_cones) {
  return GetProjectedWCM(friction_cones);
}

} // namespace

BOOST_PYTHON_MODULE(libftn_solo_control_py) {
  namespace bp = boost::python;

  Py_Initialize();
  int wargc = 0;
  wchar_t **wargv;
  Py_GetArgcArgv(&wargc, &wargv);
  const char **argv = new const char *[wargc];
  for (int i = 0; i < wargc; ++i) {
    std::wstring wchar_in_ws(wargv[i]);
    std::string wchar_in_str(wchar_in_ws.begin(), wchar_in_ws.end());
    argv[i] = wchar_in_str.c_str();
  }
  rclcpp::init(wargc, argv);
  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<VectorXi>();
  eigenpy::enableEigenPySpecific<VectorXl>();
  eigenpy::enableEigenPySpecific<VectorXb>();
  eigenpy::enableEigenPySpecific<Eigen::VectorXd>();

  InitVisualizationPublisher();
  InitEstimatorPublisher();
  InitWholeBodyPublisher();
  InitTrajectoryPlannerPublisher();

  bp::class_<ImuData>("ImuData", bp::init())
      .def_readwrite("angular_velocity", &ImuData::angular_velocity)
      .def_readwrite("linear_acceleration", &ImuData::linear_acceleration)
      .def_readwrite("magnetometer", &ImuData::magnetometer)
      .def_readwrite("attitude", &ImuData::attitude);
  bp::class_<SensorData>("SensorData", bp::init())
      .def_readwrite("imu_data", &SensorData::imu_data)
      .def_readwrite("touch", &SensorData::touch);

  bp::class_<FrictionCone>("FrictionCone",
                           bp::init<double, size_t, pinocchio::SE3>())
      .def_readonly("primal", &FrictionCone::primal_)
      .def_readonly("dual", &FrictionCone::dual_)
      .def("get_position", &FrictionCone::GetPosition)
      .def("get_num_sides", &FrictionCone::GetNumSides);

  bp::class_<SimpleConvexCone>(
      "SimpleConvexCone",
      bp::init<double, ConstRefVector3d, double, ConstRefVector3d>())
      .def_readonly("face", &SimpleConvexCone::face_)
      .def_readonly("span", &SimpleConvexCone::span_);

  bp::class_<FrictionConeMap>("FrictionConeMap")
      .def(bp::map_indexing_suite<FrictionConeMap, true>());

  bp::class_<FixedPointsEstimator>(
      "FixedPointsEstimator",
      bp::init<double, pinocchio::Model &, pinocchio::Data &,
               const std::vector<size_t> &>())
      .def("init", &FixedPointsEstimator::Init)
      .def("initialized", &FixedPointsEstimator::Initialized)
      .def("estimate", &FixedPointsEstimator::Estimate)
      .def("un_fix", &FixedPointsEstimator::UnFix)
      .def("set_fixed", &FixedPointsEstimator::SetFixed)
      .def("get_friction_cones", &FixedPointsEstimator::GetFrictionCones,
           (bp::arg("mu") = 1.0, bp::arg("num_sides") = 4))
      .def("create_friction_cone", &FixedPointsEstimator::CreateFrictionCone)
      .def_readonly("estimated_q", &FixedPointsEstimator::estimated_q_)
      .def_readonly("estimated_qv", &FixedPointsEstimator::estimated_qv_)
      .def_readonly("constraint", &FixedPointsEstimator::constraint_)
      .def_readonly("acceleration", &FixedPointsEstimator::acceleration_)
      .def_readonly("velocity", &FixedPointsEstimator::velocity_);

  bp::def("get_touching_pose", &GetTouchingPose,
          (bp::arg("model"), bp::arg("data"), bp::arg("frame_index"),
           bp::arg("normal")),
          "Computes touching pose for a given eef");
  bp::def("get_touching_placement", &GetTouchingPlacement,
          (bp::arg("model"), bp::arg("data"), bp::arg("frame_index"),
           bp::arg("pose")),
          "Computes touching frame placement for a given eef");

  bp::def("publish_cone_marker", &PublishConeMarker,
          (bp::arg("cone"), bp::arg("ns_format_string") = "%1%_%2%",
           bp::arg("show_dual") = true, bp::arg("size") = 0.15),
          "Publishes the friction cone");

  bp::class_<PieceWiseLinearPositionWrapper>("PieceWiseLinearPosition",
                                             bp::init<>())
      .def("add", &PieceWiseLinearPositionWrapper::AddPoint)
      .def("get", &PieceWiseLinearPositionWrapper::GetValues)
      .def("set_start", &PieceWiseLinearPositionWrapper::SetStart)
      .def_readonly("finished", &PieceWiseLinearPositionWrapper::finished_)
      .def("duration", &PieceWiseLinearPositionWrapper::Duration)
      .def("start_time", &PieceWiseLinearPositionWrapper::StartTime)
      .def("end_time", &PieceWiseLinearPositionWrapper::EndTime);

  bp::class_<PieceWiseLinearRotationWrapper>("PieceWiseLinearRotation",
                                             bp::init<>())
      .def("add", &PieceWiseLinearRotationWrapper::AddPoint)
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

  bp::class_<Motion>("motion", bp::init<double, double>())
      .def("set_priority", &Motion::SetPriority)
      .def("finished", &Motion::Finished)
      .def("set_start", &Motion::SetStart);
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

  bp::class_<WholeBodyController>(
      "WholeBodyController",
      bp::init<const FixedPointsEstimator &, const FrictionConeMap &, double>())
      .def("compute", &WholeBodyController::Compute)
      .def("get_force", &WholeBodyController::GetForce);

  bp::def("get_end_of_motion", &GetEndOfMotion,
          (bp::arg("model"), bp::arg("data"), bp::arg("friction_cones"),
           bp::arg("motions"), bp::arg("q")),
          "Computes robot pose at the end of motion");

  bp::def("get_end_of_motion_prioritized", &GetEndOfMotionPrioritized,
          (bp::arg("model"), bp::arg("data"), bp::arg("friction_cones"),
           bp::arg("motions"), bp::arg("q")),
          "Computes robot pose at the end of motion");

  bp::def("get_contact_jacobian", &get_contact_jacobian_proxy,
          (bp::arg("model"), bp::arg("data"), bp::arg("eef_index"),
           bp::arg("friction_cone")),
          "Computes contact jacobian for eef");

  bp::class_<std::vector<Eigen::Vector2d>>("PointsVector")
      .def(bp::vector_indexing_suite<std::vector<Eigen::Vector2d>, true>());

  bp::class_<ConvexHull2D>("ConvexHull2D",
                           bp::init<std::vector<Eigen::Vector2d>>())
      .def_readonly("points", &ConvexHull2D::points_)
      .def("equations", &ConvexHull2D::Equations)
      .def("area", &ConvexHull2D::Area);
  bp::def("intersect", &Intersect,
          (bp::arg("convex_hull_1"), bp::arg("convex_hull_2")),
          "Intersects two convex hulls");

  bp::def("get_projected_wcm", &get_projected_wcm_proxy,
          (bp::arg("friction_cones")), "Computes projected WCM");

  bp::def("get_projected_wcm_with_torque", &GetProjectedWCMWithTorque,
          (bp::arg("friction_cones")), "Computes projected WCM");

  bp::class_<TrajectoryPlanner>(
      "TrajectoryPlanner",
      bp::init<const pinocchio::Model &, size_t, const pinocchio::SE3 &>())
      .def("start_computation", &TrajectoryPlanner::StartComputation)
      .def("update_eef_trajectory", &TrajectoryPlanner::UpdateEEFTrajectory)
      .def_readonly("com_xy", &TrajectoryPlanner::com_xy_)
      .def_readonly("q", &TrajectoryPlanner::q_)
      .def("motions", &TrajectoryPlanner::Motions)
      .def("computation_started", &TrajectoryPlanner::ComputationStarted)
      .def("computation_done", &TrajectoryPlanner::ComputationDone)
      .def("update_started", &TrajectoryPlanner::UpdateStarted)
      .def("update_done", &TrajectoryPlanner::UpdateDone);
}