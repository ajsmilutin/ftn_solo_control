#pragma once
// Stl includes
#include <string>
// Common includes
#include <visualization_msgs/msg/marker_array.h>
// ftn_solo_control includes
#include <ftn_solo_control/types/friction_cone.h>

namespace ftn_solo_control {

void InitVisualizationPublisher();
void PublishConeMarker(const FrictionCone &cone,
                       const std::string &ns_format_string = "%1%_%2%",
                       bool show_dual = true, double size = 0.15);

} // namespace ftn_solo_control
