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
  ConvexHull2D(std::vector<Eigen::Vector2d> points);

  std::vector<Eigen::Vector2d> points_;

  ConstRefMatrixXd Equations() const {return equations_;};
  ConstRefVector2d Centroid() const { return centroid_; }
  double Area() const {return area_;};

protected:
  Eigen::MatrixXd equations_;
  Eigen::Vector2d centroid_;
  double area_;
};

ConvexHull2D Intersect(const ConvexHull2D &convex_hull_1,
                       const ConvexHull2D &convex_hull_2);

} // namespace ftn_solo_control
