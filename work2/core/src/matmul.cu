#include "hsys/matrix.cuh"
#include "hsys/cuda_check.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace hsys {

template <typename AtomT>
__global__ void kernel_matmul_naive(MatrixView<AtomT> a,
                                    MatrixView<AtomT> b,
                                    MatrixView<AtomT> c) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row < c.nrows() && col < c.ncols()) {
        AtomT sum = AtomT{0};
        for (std::size_t p = 0; p < a.ncols(); ++p) {
            sum += a(row, p) * b(p, col);
        }
        c(row, col) = sum;
    }
}

template <typename AtomT>
requires (std::is_arithmetic_v<AtomT> || std::is_same_v<AtomT, __half>)
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs) {
    if (lhs.ncols() != rhs.nrows()) {
        throw std::invalid_argument("Incompatible matrix dimensions for operator*");
    }

    const std::size_t m = lhs.nrows();
    const std::size_t n = rhs.ncols();
    Matrix<AtomT> result(m, n);

    if (m > 0 && n > 0) {
        constexpr unsigned int block_dim = 16;
        const dim3 block(block_dim, block_dim);
        const dim3 grid(
            static_cast<unsigned int>((n + block.x - 1) / block.x),
            static_cast<unsigned int>((m + block.y - 1) / block.y)
        );
        kernel_matmul_naive<AtomT><<<grid, block>>>(lhs.view(), rhs.view(), result.view());
        HSYS_CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

template Matrix<float> operator*(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator*(const Matrix<double>&, const Matrix<double>&);
template Matrix<int> operator*(const Matrix<int>&, const Matrix<int>&);

} // namespace hsys
