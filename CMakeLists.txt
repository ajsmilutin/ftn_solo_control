cmake_minimum_required(VERSION 3.8)
project(ftn_solo_control)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pinocchio REQUIRED)
find_package(PythonLibs ${PY_VERSION} REQUIRED)
find_package(Boost 1.74.0 REQUIRED COMPONENTS system python)
find_package(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")
find_package(eigenpy REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(proxsuite REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

include(cmake/python.cmake)
findpython()

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${eigenpy_INCLUDE_DIRS})
include_directories(${visualization_msgs_INCLUDE_DIRS})
include_directories(${proxsuite_INCLUDE_DIRS})

#add_compile_definitions(SKIP_PUBLISH_MARKERS)
# create the lib
add_library(ftn_solo_control_py SHARED 
  src/controllers/whole_body.cpp
  src/motions/com_motion.cpp
  src/motions/eef_rotation_motion.cpp
  src/motions/eef_position_motion.cpp
  src/motions/joint_motion.cpp
  src/motions/solver.cpp
  src/trajectories/piecewise_linear.cpp
  src/trajectories/spline.cpp
  src/types/convex_hull_2d.cpp
  src/types/geometric_path.cpp
  src/types/piecewise_poly_path.cpp
  src/types/friction_cone.cpp
  src/utils/wcm.cpp
  src/utils/conversions.cpp
  src/utils/visualization_utils.cpp
  src/utils/trajectory_planner.cpp
  src/utils/utils.cpp
  src/estimators.cpp
  src/bindings.cpp)

target_link_libraries(ftn_solo_control_py
                      ${Boost_LIBRARIES}
                      proxsuite::proxsuite-vectorized)
# target_compile_options(ftn_solo_control_py PUBLIC "-march=native")
ament_target_dependencies(ftn_solo_control_py rclcpp PythonLibs Eigen3 eigenpy visualization_msgs geometry_msgs tf2 tf2_ros)
target_include_directories(ftn_solo_control_py SYSTEM PRIVATE ${PYTHON_INCLUDE_DIR})


# Suppress prefix "lib" because Python does not allow this prefix
# set_target_properties(ftn_solo_control_py PROPERTIES PREFIX "")
set_target_properties(ftn_solo_control_py PROPERTIES INSTALL_RPATH
                                             "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS ftn_solo_control_py  DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
install(PROGRAMS __init__.py DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})

MESSAGE(WARNING "File  ${PYTHON_SITELIB} does not exist.")
#install(FILES __init__.py  DESTINATION "${PYTHON_INSTALL_PATH}")



ament_package()
