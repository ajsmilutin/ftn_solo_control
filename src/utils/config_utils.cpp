#include <ftn_solo_control/utils/config_utils.h>

namespace ftn_solo_control {

Eigen::VectorXd GetVectorFromConfig(const size_t length,
                                    const YAML::Node &config,
                                    const std::string &key,
                                    const double default_value) {
  if (config[key]) {
    const auto &cfg = config[key];
    if (cfg.IsScalar()) {
      return Eigen::VectorXd::Constant(length, cfg.as<double>());
    } else {
      return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
          cfg.as<std::vector<double>>().data(), length);
    }
  } else {
    return Eigen::VectorXd::Constant(length, default_value);
  }
}

} // namespace ftn_solo_control