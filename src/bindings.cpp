
#include <rclcpp/rclcpp.hpp>
// FTN solo includes
#include <ftn_solo_control/controllers/whole_body.h>
#include <ftn_solo_control/estimators.h>
#include <ftn_solo_control/motions/bindings.h>
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
Eigen::MatrixXd get_contact_jacobian_proxy(const pinocchio::Model &model,
                                           pinocchio::Data &data,
                                           size_t eef_index,
                                           const FrictionCone &friction_cone) {
  return GetContactJacobian(model, data, eef_index, friction_cone.GetPose());
}

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
  ExposeEstimators();
  InitWholeBodyPublisher();
  InitTrajectoryPlannerPublisher();
  ExposeMotionsAndTrajectories();

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

  bp::class_<WholeBodyController>(
      "WholeBodyController",
      bp::init<const FixedPointsEstimator &, const FrictionConeMap &, double,
               const std::string &>())
      .def("compute", &WholeBodyController::Compute)
      .def("get_force", &WholeBodyController::GetForce); 

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