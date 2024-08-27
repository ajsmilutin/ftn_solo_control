#include <ftn_solo_control/types/piecewise_poly_path.h>
// STL includes
#include <array>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
// Common includes
#include <Eigen/Dense>
// ftn_solo_control
#include <ftn_solo_control/types/common.h>

namespace ftn_solo_control {

BoundaryCond::BoundaryCond(int order, const std::vector<double> &values_in)
    : order(order), bc_type(Manual) {
  values.resize(values_in.size());
  for (std::size_t i = 0; i < values.size(); i++)
    values(i) = values_in[i];
}

BoundaryCond::BoundaryCond(int order, ConstRefVectorXd values)
    : order(order), values(values), bc_type(Manual){};

BoundaryCond::BoundaryCond(std::string type) {
  if (type == "not-a-knot" || type == "notaknot") {
    bc_type = Type::NotAKnot;
  } else if (type == "clamped") {
    bc_type = Type::Clamped;
  } else if (type == "natural") {
    bc_type = Type::Natural;
  } else
    throw std::runtime_error("Unknown bc type" + type);
}

namespace _cubic_spline {
/**
 * @brief Check, potentially modify input.
 *
 * @param positions
 * @param times
 * @param bc_type
 */
void CheckInputArgs(const Vectors &positions, ConstRefVectorXd times,
                    BoundaryCondFull &bc_type) {
  if (positions.size() != times.rows()) {
    throw std::runtime_error(
        "The length of 'positions' doesn't match the length of 'times'.");
  }
  if (times.rows() < 2) {
    throw std::runtime_error("'times' must contain at least 2 elements.");
  }
  for (size_t i = 1; i < positions.size(); i++) {
    if (positions[i].rows() != positions[i - 1].rows()) {
      throw std::runtime_error(
          "The number of elements in each position has to be equal.");
    }
  }
  Eigen::VectorXd dtimes(times.rows() - 1);
  for (auto i = 1u; i < times.rows(); i++) {
    dtimes(i - 1) = times(i) - times(i - 1);
    if (dtimes(i - 1) <= 0) {
      throw std::runtime_error(
          "'times' must be a strictly increasing sequence.");
    }
  }

  // Validate boundary conditions
  size_t expected_deriv_size = positions[0].size();
  for (BoundaryCond &bc : bc_type) {
    if (bc.bc_type == BoundaryCond::Clamped) {
      bc.order = 1;
      bc.values.resize(expected_deriv_size);
      bc.values.setZero();
    } else if (bc.bc_type == BoundaryCond::Natural) {
      bc.order = 2;
      bc.values.resize(expected_deriv_size);
      bc.values.setZero();
    } else if (bc.bc_type == BoundaryCond::NotAKnot) {
      throw std::runtime_error(
          "Boundary condition NotAKnot is not implemented");
    } else if (bc.order != 1 && bc.order != 2) {
      throw std::runtime_error(
          "The specified derivative order must be 1 or 2.");
    } else if (bc.values.size() != expected_deriv_size) {
      throw std::runtime_error("`deriv_value` size " +
                               std::to_string(bc.values.size()) +
                               " is not the expected one " +
                               std::to_string(expected_deriv_size) + ".");
    }
  }
}

/**
 * @brief Compute the piecewise poly coefficients.
 *
 * @param positions
 * @param times
 * @param bc_type
 * @param coefficients
 */
void computePPolyCoefficients(const Vectors &positions, ConstRefVectorXd times,
                              const BoundaryCondFull &bc_type,
                              Matrices &coefficients) {
  // h(i) = t(i+1) - t(i)
  Eigen::VectorXd h(times.rows() - 1);
  for (auto i = 0u; i < h.rows(); i++) {
    h(i) = times(i + 1) - times(i);
  }

  // Construct the tri-diagonal matrix A based on spline continuity criteria
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(times.rows(), times.rows());
  for (auto i = 1u; i < A.rows() - 1; i++) {
    A.row(i).segment(i - 1, 3) << h(i - 1), 2 * (h(i - 1) + h(i)), h(i);
  }

  // Construct B based on spline continuity criteria
  Vectors B(positions.at(0).rows());
  for (auto i = 0u; i < B.size(); i++) {
    B[i].resize(times.rows());
    for (auto j = 1u; j < A.rows() - 1; j++) {
      B[i](j) = 3 * (positions[j + 1](i) - positions[j](i)) / h(j) -
                3 * (positions[j](i) - positions[j - 1](i)) / h(j - 1);
    }
  }

  // Insert boundary conditions to A and B
  if (bc_type[0].order == 1) {
    A.row(0).segment(0, 2) << 2 * h(0), h(0);
    for (auto i = 0u; i < B.size(); i++) {
      B[i](0) = 3 * (positions[1](i) - positions[0](i)) / h(0) -
                3 * bc_type[0].values(i);
    }
  } else if (bc_type[0].order == 2) {
    A(0, 0) = 2;
    for (auto i = 0u; i < B.size(); i++) {
      B[i](0) = bc_type[0].values(i);
    }
  }

  if (bc_type[1].order == 1) {
    A.row(A.rows() - 1).segment(A.cols() - 2, 2) << h(h.rows() - 1),
        2 * h(h.rows() - 1);
    for (auto i = 0u; i < B.size(); i++) {
      B[i](B[i].rows() - 1) =
          3 * bc_type[1].values(i) - 3 *
                                         (positions[positions.size() - 1](i) -
                                          positions[positions.size() - 2](i)) /
                                         h(h.rows() - 1);
    }
  } else if (bc_type[1].order == 2) {
    A(A.rows() - 1, A.cols() - 1) = 2;
    for (auto i = 0u; i < B.size(); i++) {
      B[i](B[i].rows() - 1) = bc_type[1].values(i);
    }
  }

  // Solve AX = B
  Vectors X(positions[0].rows());
  for (auto i = 0u; i < X.size(); i++) {
    X[i].resize(times.rows());
    X[i] = A.colPivHouseholderQr().solve(B[i]);
  }

  // Insert spline coefficients
  for (auto i = 0u; i < coefficients.size(); i++) {
    coefficients[i].resize(4, positions[0].rows());
    for (auto j = 0u; j < coefficients[i].cols(); j++) {
      coefficients[i](0, j) = (X[j](i + 1) - X[j](i)) / (3 * h(i));
      coefficients[i](1, j) = X[j](i);
      coefficients[i](2, j) = (positions[i + 1](j) - positions[i](j)) / h(i) -
                              h(i) / 3 * (2 * X[j](i) + X[j](i + 1));
      coefficients[i](3, j) = positions[i](j);
    }
  }
}
} // namespace _cubic_spline

PiecewisePolyPath PiecewisePolyPath::CubicSpline(const Vectors &positions,
                                                 ConstRefVectorXd times,
                                                 BoundaryCondFull bc_type) {
  Matrices coefficients(times.size() - 1);
  _cubic_spline::CheckInputArgs(positions, times, bc_type);
  _cubic_spline::computePPolyCoefficients(positions, times, bc_type,
                                          coefficients);
  std::vector<double> breakpoints(times.data(), times.data() + times.size());
  return PiecewisePolyPath(coefficients, breakpoints);
}

Eigen::MatrixXd differentiateCoefficients(ConstRefMatrixXd coefficients) {
  Eigen::MatrixXd deriv(coefficients.rows(), coefficients.cols());
  deriv.setZero();
  for (Eigen::Index i = 1; i < coefficients.rows(); i++) {
    deriv.row(i) = coefficients.row(i - 1) * (coefficients.rows() - i);
  }
  return deriv;
}

PiecewisePolyPath::PiecewisePolyPath(const Matrices &coefficients,
                                     const std::vector<double> &breakpoints)
    : GeometricPath(int(coefficients[0].cols())), coefficients_(coefficients),
      breakpoints_(std::move(breakpoints)),
      m_degree(int(coefficients[0].rows()) - 1) {

  CheckInputArgs();
  ComputeDerivativesCoefficients();
}
Bound PiecewisePolyPath::PathInterval() const {
  Bound v;
  v << breakpoints_.front(), breakpoints_.back();
  return v;
};

Eigen::VectorXd PiecewisePolyPath::EvalSingle(double pos, int order) const {
  assert(order < 3 && order >= 0);
  Eigen::VectorXd v(dof_);
  v.setZero();
  size_t seg_index = FindSegmentIndex(pos);
  auto coeff = GetCoefficient(seg_index, order);
  for (int power = 0; power < m_degree + 1; power++) {
    v +=
        coeff.row(power) * pow(pos - breakpoints_[seg_index], m_degree - power);
  }
  return v;
}

// Not the most efficient implementation. Coefficients are
// recomputed. Should be refactored.
Vectors PiecewisePolyPath::Eval(ConstRefVectorXd positions, int order) const {
  assert(order < 3 && order >= 0);
  Vectors outputs;
  outputs.resize(positions.size());
  for (auto i = 0u; i < positions.size(); i++) {
    outputs[i] = EvalSingle(positions(i), order);
  }
  return outputs;
}

size_t PiecewisePolyPath::FindSegmentIndex(double pos) const {
  size_t idx = 0;
  if (pos > breakpoints_.back()) {
    idx = coefficients_.size() - 1;
  } else if (pos < breakpoints_[0]) {
    idx = 0;
  } else {
    auto it = std::upper_bound(breakpoints_.begin(), breakpoints_.end(), pos);
    idx = std::distance(breakpoints_.begin(), it) - 1;
    if (idx < 0)
      return 0;
  }
  return std::min<size_t>(idx, coefficients_.size() - 1);
}

void PiecewisePolyPath::CheckInputArgs() {
  assert(coefficients_[0].cols() == dof_);
  assert(coefficients_[0].rows() == (m_degree + 1));
  if ((1 + coefficients_.size()) != breakpoints_.size()) {
    throw std::runtime_error(
        "Number of breakpoints must equals number of segments plus 1.");
  }
  for (size_t seg_index = 0; seg_index < coefficients_.size(); seg_index++) {
    if (breakpoints_[seg_index] >= breakpoints_[seg_index + 1]) {
      throw std::runtime_error("Require strictly increasing breakpoints");
    }
  }
}

void PiecewisePolyPath::ComputeDerivativesCoefficients() {
  coefficients_1_.reserve(coefficients_.size());
  coefficients_2_.reserve(coefficients_.size());
  for (size_t seg_index = 0; seg_index < coefficients_.size(); seg_index++) {
    coefficients_1_.push_back(
        differentiateCoefficients(coefficients_[seg_index]));
    coefficients_2_.push_back(
        differentiateCoefficients(coefficients_1_[seg_index]));
  }
}

ConstRefMatrixXd PiecewisePolyPath::GetCoefficient(size_t seg_index,
                                                   int order) const {
  switch (order) {
  case 0:
    return coefficients_.at(seg_index);
  case 1:
    return coefficients_1_.at(seg_index);
  case 2:
    return coefficients_2_.at(seg_index);
  default:
    break;
  }

  return coefficients_2_.at(seg_index);
}

void PiecewisePolyPath::Reset() {
  breakpoints_.clear();
  coefficients_.clear();
  coefficients_1_.clear();
  coefficients_2_.clear();
}

void PiecewisePolyPath::InitAsHermite(const Vectors &positions,
                                      const Vectors &velocities,
                                      const std::vector<double> times) {
  Reset();
  assert(positions.size() == times.size());
  assert(velocities.size() == times.size());
  config_size_ = dof_ = int(positions[0].size());
  m_degree = 3; // cubic spline
  breakpoints_ = times;
  for (std::size_t i = 0; i < times.size() - 1; i++) {
    Eigen::MatrixXd c(4, dof_);
    auto dt = times[i + 1] - times[i];
    assert(dt > 0);
    // ... after some derivations
    c.row(3) = positions.at(i);
    c.row(2) = velocities.at(i);
    c.row(0) =
        (velocities.at(i + 1).transpose() * dt -
         2 * positions.at(i + 1).transpose() + c.row(2) * dt + 2 * c.row(3)) /
        pow(dt, 3);
    c.row(1) =
        (velocities.at(i + 1).transpose() - c.row(2) - 3 * c.row(0) * dt * dt) /
        (2 * dt);
    coefficients_.push_back(c);
  }
  CheckInputArgs();
  ComputeDerivativesCoefficients();
}

PiecewisePolyPath
PiecewisePolyPath::CubicHermiteSpline(const Vectors &positions,
                                      const Vectors &velocities,
                                      const std::vector<double> times) {
  PiecewisePolyPath path;
  path.InitAsHermite(positions, velocities, times);
  return path;
}

} // namespace ftn_solo_control
