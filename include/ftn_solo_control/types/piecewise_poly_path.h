#pragma once
// STL includes
#include <vector>
// FTN solo control
#include <ftn_solo_control/types/common.h>
#include <ftn_solo_control/types/geometric_path.h>

namespace ftn_solo_control {

struct BoundaryCond {
  BoundaryCond() = default;

  /**
   * @brief Construct a new BoundaryCond object with a manually specified
   * derivative.
   *
   * @param order Order of the specified derivative.
   * @param values Vector of values. Must have the same size as the path.
   */
  BoundaryCond(int order, const std::vector<double> &values);

  BoundaryCond(int order, ConstRefVectorXd values);

  /**
   * @brief Construct a new Boundary Cond object with well-known boundary
   * condition.
   *
   * @param bc_type Possible values: not-a-knot, clamped, natural and manual.
   */
  BoundaryCond(std::string bc_type);

  enum Type { NotAKnot, Clamped, Natural, Manual };
  Type bc_type = NotAKnot;
  int order = 0;
  Eigen::VectorXd values;
};

using BoundaryCondFull = std::array<BoundaryCond, 2>;

/**
 * \brief Piecewise polynomial geometric path.
 *
 * An implementation of a piecewise polynomial geometric path.
 *
 * The coefficient vector has shape (N, P, D), where N is the number
 * segments. For each segment, the i-th row (P index) denotes the
 * power, while the j-th column is the degree of freedom. In
 * particular,
 *
 *  coeff(0) * dt ^ 3 + coeff(1) * dt ^ 2 + coeff(2) * dt + coeff(3)
 *
 *
 */
class PiecewisePolyPath : public GeometricPath {
public:
  PiecewisePolyPath() = default;

  /**
   * \brief Construct new piecewise polynomial.
   *
   * See class docstring for details.
   *
   * @param coefficients Polynomial coefficients.
   * @param breakpoints Vector of breakpoints.
   */
  PiecewisePolyPath(const Matrices &coefficients,
                    const std::vector<double> &breakpoints);

  /**
   * /brief Evaluate the path at given position.
   */
  Eigen::VectorXd EvalSingle(double, int order = 0) const;

  /**
   * /brief Evaluate the path at given positions (vector).
   */
  Vectors Eval(ConstRefVectorXd, int order = 0) const;

  /**
   * Return the starting and ending path positions.
   */
  Bound PathInterval() const;

  /**
   * @brief Construct a piecewise Cubic Hermite polynomial.
   *
   * See https://en.wikipedia.org/wiki/Cubic_Hermite_spline for a good
   * description of this interplation scheme.
   *
   * This function is implemented based on scipy.interpolate.CubicHermiteSpline.
   *
   * Note that path generates by this function is not guaranteed to have
   * continuous acceleration, or the path second-order derivative.
   *
   * @param positions Robot joints corresponding to the given times. Must have
   * the same size as times.
   * @param velocities Robot joint velocities.
   * @param times Path positions or times. This is the independent variable.
   * @return PiecewisePolyPath
   */
  static PiecewisePolyPath
  CubicHermiteSpline(const Vectors &positions,
                     const Vectors &velocities,
                     const std::vector<double> times);

  /**
   * @brief Construct a cubic spline.
   *
   * Interpolate the given joint positions with a spline that is twice
   * continuously differentiable. This means the position, velocity and
   * acceleration are guaranteed to be continous but not jerk.
   *
   * This method is modelled after scipy.interpolate.CubicSpline.
   *
   * @param positions Robot joints corresponding to the given times.
   * @param times Path positions or times. This is the independent variable.
   * @param bc_type Boundary condition. Currently on fixed boundary condition.
   * @return PiecewisePolyPath
   */
  static PiecewisePolyPath
  CubicSpline(const Vectors &positions,
              ConstRefVectorXd times, BoundaryCondFull bc_type);

private:
  /**
   * @brief Calculate coefficients for Hermite spline.
   *
   * @param positions See the constructor.
   * @param velocities
   * @param times
   */
  void InitAsHermite(const Vectors &positions,
                     const Vectors &velocities,
                     const std::vector<double> times);

protected:
  static void computeCubicSplineCoefficients(
      const Vectors &positions, ConstRefVectorXd times,
      const BoundaryCondFull &bc_type, RefMatrixXd coefficients);

  void Reset();
  size_t FindSegmentIndex(double pos) const;
  void CheckInputArgs();
  void ComputeDerivativesCoefficients();
  ConstRefMatrixXd GetCoefficient(size_t seg_index, int order) const;
  Matrices coefficients_, coefficients_1_,
      coefficients_2_;
  std::vector<double> breakpoints_;
  int m_degree;
};

} // namespace ftn_solo_control