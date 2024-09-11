#pragma once
// STL includes
#include <vector>
// Common includes
#include <Eigen/Dense>
#include <boost/optional.hpp>
// ftn_solo_control includes
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

class ConvexHull2D {
public:
  ConvexHull2D(std::vector<Eigen::Vector2d> points)
      : points_(std::move(points)), equations_(boost::none){};

  std::vector<Eigen::Vector2d> points_;

  ConstRefMatrixXd Equations();

  double Area();

protected:
  boost::optional<Eigen::MatrixXd> equations_;
};

ConvexHull2D Intersect(const ConvexHull2D &convex_hull_1,
                       const ConvexHull2D &convex_hull_2);

} // namespace ftn_solo_control
