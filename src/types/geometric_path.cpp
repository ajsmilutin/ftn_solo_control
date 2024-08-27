#include <ftn_solo_control/types/geometric_path.h>
// STL includes
#include <iterator>
#include <list>
#include <vector>
namespace ftn_solo_control {
namespace {
constexpr double kNearlyZero = 1e-8;
}

Vectors GeometricPath::Eval(ConstRefVectorXd positions,
                                                 int order) const {
  Vectors outputs;
  outputs.resize(positions.size());
  for (size_t i = 0; i < positions.size(); i++) {
    outputs[i] = EvalSingle(positions(i), order);
  }
  return outputs;
};

Eigen::VectorXd
GeometricPath::ProposeGridpoints(double max_err_threshold, int max_iteration,
                                 double max_seg_length, int min_nb_points,
                                 ConstRefVectorXd initial_gridpoints) const {
  std::list<double> gridpoints;
  // gridpoints.reserve(std::max(initial_gridpoints.size(), min_nb_points));
  const Bound I = PathInterval();
  if (initial_gridpoints.size() == 0) {
    gridpoints.push_front(I[1]);
    gridpoints.push_front(I[0]);
  } else {
    if (initial_gridpoints.size() == 1)
      throw std::invalid_argument(
          "initial_gridpoints should be empty or have at least 2 elements");
    int N = initial_gridpoints.size() - 1;
    for (int i : {0, 1}) {
      if (std::abs(I[i] - initial_gridpoints[i * N]) > kNearlyZero) {
        std::ostringstream oss;
        oss << "initial_gridpoints[" << i * N << "] must be " << I[i]
            << " and not " << initial_gridpoints[i * N];
        throw std::invalid_argument(oss.str());
      }
    }
    if ((initial_gridpoints.tail(N).array() <=
         initial_gridpoints.head(N).array())
            .any())
      throw std::invalid_argument(
          "initial_gridpoints should be monotonically increasing.");
    for (int i = N; i >= 0; --i)
      gridpoints.push_front(initial_gridpoints[i]);
  }

  // Add points according to error threshold
  for (int iter = 0; iter < max_iteration; iter++) {
    bool add_new_points = false;
    for (auto point = gridpoints.begin();
         std::next(point, 1) != gridpoints.end(); point++) {

      auto next = std::next(point, 1);

      double p_mid = 0.5 * (*point + *next);
      auto dist = (*next - *point);

      if (dist > max_seg_length) {
        gridpoints.emplace(next, p_mid);
        add_new_points = true;
        continue;
      }

      // maximum interpolation error
      auto max_err =
          (0.5 * EvalSingle(p_mid, 2) * dist * dist).cwiseAbs().maxCoeff();
      if (max_err > max_err_threshold) {
        gridpoints.emplace(next, p_mid);
        add_new_points = true;
        continue;
      }
    }

    if (!add_new_points)
      break;
  }

  // Add points according to smallest number of points
  while (gridpoints.size() < min_nb_points) {
    for (auto point = gridpoints.begin();
         std::next(point, 1) != gridpoints.end(); std::advance(point, 2)) {
      auto next = std::next(point, 1);
      double p_mid = 0.5 * (*point + *next);
      gridpoints.emplace(next, p_mid);
    }
  }

  // Return the Eigen vector
  std::vector<double> result(gridpoints.begin(), gridpoints.end());
  return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(result.data(),
                                                       result.size());
}
} // namespace ftn_solo_control
