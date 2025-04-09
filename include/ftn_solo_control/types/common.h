/**
 * @file common.hpp
 * @author Julian Viereck (jviereck@tuebingen.mpg.de)
 * license License BSD-3-Clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft.
 * @date 2020-11-27
 *
 * @brief Common definitions used across the library.
 */
#pragma once
#include <Eigen/Dense>

namespace ftn_solo_control {
typedef Eigen::VectorXd VectorXd;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXi;
typedef Eigen::Matrix<long, Eigen::Dynamic, 1> VectorXl;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;
typedef Eigen::Matrix<bool, 3, 1> Vector3b;
typedef Eigen::Matrix<bool, 4, 1> Vector4b;
typedef Eigen::Matrix<int, 4, 1> Vector4i;

typedef Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 1>> RefVectorXi;
typedef Eigen::Ref<Eigen::Matrix<long, Eigen::Dynamic, 1>> RefVectorXl;
typedef Eigen::Ref<Eigen::Matrix<bool, Eigen::Dynamic, 1>> RefVectorXb;
typedef Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 1>> RefVectorXd;
typedef Eigen::Ref<Eigen::Vector2d> RefVector2d;
typedef Eigen::Ref<Eigen::Vector3d> RefVector3d;
typedef Eigen::Ref<Eigen::Vector4d> RefVector4d;
typedef Eigen::Ref<Eigen::Matrix3d> RefMatrix3d;
typedef Eigen::Ref<Eigen::MatrixXd> RefMatrixXd;

typedef const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, 1>>
    ConstRefVectorXi;
typedef const Eigen::Ref<const Eigen::Matrix<long, Eigen::Dynamic, 1>>
    ConstRefVectorXl;
typedef const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic, 1>>
    ConstRefVectorXb;
typedef const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 1>>
    ConstRefVectorXd;
typedef const Eigen::Ref<const Eigen::Vector2d> ConstRefVector2d;
typedef const Eigen::Ref<const Eigen::Vector3d> ConstRefVector3d;
typedef const Eigen::Ref<const Eigen::Vector4d> ConstRefVector4d;
typedef const Eigen::Ref<const Vector3b> ConstRefVector3b;
typedef const Eigen::Ref<const Eigen::VectorXi> ConstRefVectorXi;
typedef const Eigen::Ref<const Eigen::MatrixXd> ConstRefMatrixXd;
typedef const Eigen::Ref<const Eigen::Matrix3d> ConstRefMatrix3d;

/// Vector of Vector
typedef std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>>
    Vectors;
/// Vector of Matrix
typedef std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
    Matrices;

// const double kRadius = 0.016;
const double kRadius = 0.02;
} // namespace ftn_solo_control
