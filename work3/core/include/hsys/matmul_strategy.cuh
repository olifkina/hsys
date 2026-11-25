#pragma once

#include <type_traits>
#include "hsys/matrix.cuh"

namespace hsys {

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
class IMatMulStrategy {
public:
    virtual ~IMatMulStrategy() = default;
    virtual Matrix<AtomT> multiply(const Matrix<AtomT>& lhs,
                                   const Matrix<AtomT>& rhs) const = 0;
};

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
class NaiveMatMulStrategy final : public IMatMulStrategy<AtomT> {
public:
    Matrix<AtomT> multiply(const Matrix<AtomT>& lhs,
                           const Matrix<AtomT>& rhs) const override;
};

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
class ShmemMatMulStrategy final : public IMatMulStrategy<AtomT> {
public:
    Matrix<AtomT> multiply(const Matrix<AtomT>& lhs,
                           const Matrix<AtomT>& rhs) const override;
};

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
Matrix<AtomT> matmul(const Matrix<AtomT>& lhs,
                     const Matrix<AtomT>& rhs,
                     const IMatMulStrategy<AtomT>& strategy);

} // namespace hsys
