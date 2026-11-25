#pragma once

#include <cuda_fp16.h>
#include "hsys/matrix.cuh"

namespace hsys {

class IHalfMatMulStrategy {
public:
    virtual ~IHalfMatMulStrategy() = default;
    virtual Matrix<__half> multiply(const Matrix<__half>& lhs,
                                    const Matrix<__half>& rhs) const = 0;
};

class ShmemHalfMatMulStrategy final : public IHalfMatMulStrategy {
public:
    Matrix<__half> multiply(const Matrix<__half>& lhs,
                            const Matrix<__half>& rhs) const override;
};

class WmmaHalfMatMulStrategy final : public IHalfMatMulStrategy {
public:
    Matrix<__half> multiply(const Matrix<__half>& lhs,
                            const Matrix<__half>& rhs) const override;
};

Matrix<__half> matmul_half(const Matrix<__half>& lhs,
                           const Matrix<__half>& rhs,
                           const IHalfMatMulStrategy& strategy);

} // namespace hsys
