#include <ftn_solo_control/types/convex_hull_2d.h>

#include <iostream>
namespace ftn_solo_control {
namespace {
Eigen::Vector2d ComputeIntersection(ConstRefVector2d p0,
                                    ConstRefVector2d direction_0,
                                    ConstRefVector2d p1,
                                    ConstRefVector2d direction_1) {

  const double determinant =
      direction_0.x() * direction_1.y() - direction_0.y() * direction_1.x();
  const double s0 =
      -1 / determinant *
      Eigen::Vector2d(direction_1.y(), -direction_1.x()).dot(p0 - p1);
  return p0 + s0 * direction_0;
}

inline double ComputeTriangleArea(ConstRefVector2d p0, ConstRefVector2d p1,
                                  ConstRefVector2d p2) {
  return std::abs(0.5 *
                  (p0.x() * (p1.y() - p2.y()) + p1.x() * (p2.y() - p0.y()) +
                   p2.x() * (p0.y() - p1.y())));
}
const double kEpsilon = 1e-6;

} // namespace

ConvexHull2D::ConvexHull2D(std::vector<Eigen::Vector2d> points)
    : points_(std::move(points)) {
  area_ = 0;
  centroid_ = Eigen::Vector2d::Zero();
  for (size_t i = 2; i < points_.size(); ++i) {
    double triangle_area =
        ComputeTriangleArea(points_[0], points_[i - 1], points_[i]);
    centroid_ += triangle_area * (points_[0] + points_[i - 1] + points_[i]);
    area_ += triangle_area;
  }
  centroid_ = centroid_ / area_ / 3;
  equations_ = Eigen::MatrixXd(points_.size(), 3);
  for (size_t i = 0; i < points_.size(); ++i) {
    size_t start = i;
    size_t end = (i + 1) % points_.size();
    Eigen::Vector2d direction =
        (points_.at(end) - points_.at(start)).normalized();
    equations_.row(i) =
        Eigen::Vector3d(-direction(1), direction(0),
                        -points_.at(start)(1) * direction(0) +
                            direction(1) * points_.at(start)(0));
  }
}

ConvexHull2D Intersect(const ConvexHull2D &convex_hull_1,
                       const ConvexHull2D &convex_hull_2) {
  const auto &points_1 = convex_hull_1.points_;
  std::vector<Eigen::Vector2d> points_2 = convex_hull_2.points_;
  for (size_t i = 0; i < points_1.size() && points_2.size() > 0; ++i) {
    std::vector<Eigen::Vector2d> new_points;
    const Eigen::Vector2d direction =
        (points_1.at((i + 1) % points_1.size()) - points_1.at(i)).normalized();
    const Eigen::Vector2d normal(-direction(1), direction(0));
    const double offset = -normal.dot(points_1.at(i));
    bool prev_point_in = points_2.front().dot(normal) + offset > kEpsilon;
    for (size_t j = 0; j < points_2.size(); ++j) {
      const auto &current_point = points_2.at((j + 1) % points_2.size());
      const auto &prev_point = points_2.at(j);

      const bool current_point_in =
          current_point.dot(normal) + offset > kEpsilon;
      Eigen::Vector2d intersection = ComputeIntersection(
          prev_point, (current_point - prev_point).normalized(), points_1.at(i),
          direction);
      if (current_point_in) {
        if (!prev_point_in &&
            (intersection - current_point).norm() > kEpsilon) {
          new_points.push_back(intersection);
        }
        new_points.push_back(current_point);
      } else if (prev_point_in &&
                 (intersection - prev_point).norm() > kEpsilon) {
        new_points.push_back(intersection);
      }
      prev_point_in = current_point_in;
    }
    points_2 = std::move(new_points);
  }
  std::vector<Eigen::Vector2d> clean_points;
  clean_points.push_back(points_2.front());
  for (const auto &pt : points_2) {
    if ((pt - clean_points.back()).norm() > 1e-3 &&
        (pt - clean_points.front()).norm() > 1e-3) {
      clean_points.push_back(pt);
    }
  }
  return ConvexHull2D(clean_points);
}

} // namespace ftn_solo_control