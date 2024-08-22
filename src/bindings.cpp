#include <ftn_solo_control/estimators.h>
#include <ftn_solo_control/types/friction_cone.h>
#include <ftn_solo_control/types/sensors.h>
#include <ftn_solo_control/utils/visualization_utils.h>

#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/utils/utils.h>
#include <rclcpp/rclcpp.hpp>

BOOST_PYTHON_MODULE(libftn_solo_control_py) {
  namespace bp = boost::python;
  using namespace ftn_solo_control;
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
      .def("get_position", &FrictionCone::GetPosition);
  bp::class_<SimpleConvexCone>(
      "SimpleConvexCone",
      bp::init<double, ConstRefVector3d, double, ConstRefVector3d>())
      .def_readonly("face", &SimpleConvexCone::face_)
      .def_readonly("span", &SimpleConvexCone::span_);

  bp::class_<FixedPointsEstimator>(
      "FixedPointsEstimator",
      bp::init<double, pinocchio::Model &, pinocchio::Data &,
               const std::vector<size_t> &>())
      .def("init", &FixedPointsEstimator::Init)
      .def("estimate", &FixedPointsEstimator::Estimate)
      .def("un_fix", &FixedPointsEstimator::UnFix)
      .def("set_fixed", &FixedPointsEstimator::SetFixed)
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
}