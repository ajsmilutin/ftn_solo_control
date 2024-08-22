#include <ftn_solo_control/utils/visualization_utils.h>
// Common includes
#include <boost/format.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// ftn_solo_control includes
#include <ftn_solo_control/types/friction_cone.h>
#include <ftn_solo_control/utils/conversions.h>

namespace ftn_solo_control {

namespace {

static rclcpp::Node::SharedPtr visualization_node;
static rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    publisher;

visualization_msgs::msg::Marker
GetSimpleConeMarkers(const SimpleConvexCone &cone,
                     const geometry_msgs::msg::Point &position,
                     const std::string ns, const std_msgs::msg::ColorRGBA color,
                     double size = 0.15) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "world";
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
  marker.color = color;
  marker.pose.position = position;
  marker.scale.x = marker.scale.y = marker.scale.z = 1;
  marker.ns = ns;
  marker.id = 200;
  geometry_msgs::msg::Point zero;
  zero.x = zero.y = zero.z = 0;
  for (size_t i = 0; i < cone.GetNumSides(); ++i) {
    marker.points.push_back(zero);
    marker.points.push_back(ToPoint(size * cone.span_.col(i)));
    marker.points.push_back(
        ToPoint(size * cone.span_.col((i + 1) % cone.GetNumSides())));
  }
  return marker;
}

} // namespace

void InitVisualizationPublisher() {
  visualization_node = std::make_shared<rclcpp::Node>("visualization_node");
  publisher = visualization_node
                  ->create_publisher<visualization_msgs::msg::MarkerArray>(
                      "visualizaton_markers", 10);
}

void PublishConeMarker(const FrictionCone &cone,
                       const std::string &ns_format_string, bool show_dual,
                       double size) {
  visualization_msgs::msg::MarkerArray all_markers;
  std_msgs::msg::ColorRGBA color;
  color.r = 0.8;
  color.g = 0;
  color.b = 1;
  color.a = 0.5;
  all_markers.markers.push_back(std::move(GetSimpleConeMarkers(
      cone.primal_, ToPoint(cone.GetPosition()),
      (boost::format(ns_format_string) % "primal" % cone.GetNum()).str(), color,
      size)));
  if (show_dual) {
    color.r = 0.2;
    color.g = 0.9;
    color.b = 1;
    color.a = 0.5;
    all_markers.markers.push_back(std::move(GetSimpleConeMarkers(
        cone.dual_, ToPoint(cone.GetPosition()),
        (boost::format(ns_format_string) % "dual" % cone.GetNum()).str(), color,
        size)));
  }
  publisher->publish(all_markers);
}

} // namespace ftn_solo_control
