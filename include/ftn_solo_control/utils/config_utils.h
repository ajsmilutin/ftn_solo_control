#pragma once
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

namespace ftn_solo_control {

Eigen::VectorXd GetVectorFromConfig(const size_t length,
                                    const YAML::Node &config,
                                    const std::string &key,
                                    const double default_value = 0);

#define READ_DOUBLE_CONFIG(config_node, key, target)                           \
  if (config_node[#key]) {                                                     \
    target = config_node[#key].as<double>();                                   \
  }

} // namespace ftn_solo_control