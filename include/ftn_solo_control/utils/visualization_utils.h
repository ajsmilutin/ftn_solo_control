#pragma once
// Stl includes
#include <string>
// Common includes
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker_array.h>
// ftn_solo_control includes
#include <ftn_solo_control/types/friction_cone.h>

namespace ftn_solo_control {

void InitVisualizationPublisher();
void PublishConeMarker(const FrictionCone &cone,
                       const std::string &ns_format_string = "%1%_%2%",
                       bool show_dual = true, double size = 0.15);

std_msgs::msg::ColorRGBA MakeColor(double r = 1.0, double g = 0.0,
                                   double b = 0.0, double a = 1.0);

std_msgs::msg::ColorRGBA RandomColor();

} // namespace ftn_solo_control
