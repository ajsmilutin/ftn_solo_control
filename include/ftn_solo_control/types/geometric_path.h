#pragma once
// STD includes
#include <cstddef>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>
// Common includes
#include <Eigen/Dense>
// ftn_solo_control
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

typedef Eigen::Vector2d Bound;

/**
 * \brief Abstract interface for geometric paths.
 */
class GeometricPath {
public:
  GeometricPath() = default;

  /**
   * Constructor of GeometricPath on vector spaces.
   */
  GeometricPath(int n_dof) : config_size_(n_dof), dof_(n_dof) {}

  /**
   * Constructor of GeometricPath on non-vector spaces.
   */
  GeometricPath(int config_size, int n_dof)
      : config_size_(config_size), dof_(n_dof) {}

  /**
   * \brief Evaluate the path at given position.
   */
  virtual Eigen::VectorXd EvalSingle(double, int order = 0) const = 0;

  /**
   * \brief Evaluate the path at given positions (vector).
   *
   * Default implementation: Evaluation each point one-by-one.
   */
  virtual Vectors Eval(ConstRefVectorXd positions,
                                            int order = 0) const;

  /**
     \brief Generate gridpoints that sufficiently cover the given path.

     This function operates in multiple passes through the geometric
     path from the start to the end point. In each pass, for each
     segment, the maximum interpolation error is estimated using the
     following equation:

        err_{est} = 0.5 * \mathrm{max}(\mathrm{abs}(p'' * d_{segment} ^ 2))

     Here `p''` is the second derivative of the path and d_segment is
     the length of the segment. If the estimated error `err_{test}` is
     greater than the given threshold `max_err_threshold` then the
     segment is divided in two half.

     Intuitively, at positions with higher curvature, there must be
     more points in order to improve approximation
     quality. Theoretically toppra performs the best when the proposed
     gridpoint is optimally distributed.

     @param max_err_threshold Maximum worstcase error thrshold allowable.
     @param max_iteration Maximum number of iterations.
     @param max_seg_length All segments length should be smaller than this value.
     @param min_nb_points Minimum number of points.
     @param initial_gridpoints Initial gridpoints to start the algorithm from. If
            not provided, the path interval is used. If not empty, it must start
            and end with the path interval limits.
     @return The proposed gridpoints.

   */
  Eigen::VectorXd ProposeGridpoints(
      double max_err_threshold = 1e-4, int max_iteration = 100,
      double max_seg_length = 0.05, int min_nb_points = 100,
      ConstRefVectorXd initial_gridpoints = Eigen::VectorXd()) const;

  /**
   * \brief Dimension of the configuration space
   */
  size_t ConfigSize() const { return config_size_; }

  /**
   * \return the number of degrees-of-freedom of the path.
   */
  size_t Dof() const { return dof_; }

  /**
   * \brief Starting and ending path positions.
   */
  virtual Bound PathInterval() const = 0;

  virtual ~GeometricPath() {}

protected:
  size_t config_size_;
  size_t dof_;
};

} // namespace ftn_solo_control
