#include "hsys/matmul_strategy.cuh"
#include "hsys/cuda_check.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace hsys {

constexpr unsigned int TILE_SIZE = 16;

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
__global__ void kernel_matmul_shmem(MatrixView<AtomT> a,
                                    MatrixView<AtomT> b,
                                    MatrixView<AtomT> c) {
    __shared__ AtomT tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ AtomT tile_b[TILE_SIZE][TILE_SIZE];

    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * TILE_SIZE + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * TILE_SIZE + threadIdx.x;
    const std::size_t k_dim = a.ncols();
    const std::size_t num_tiles = (k_dim + TILE_SIZE - 1) / TILE_SIZE;

    AtomT sum = AtomT{0};

    for (std::size_t t = 0; t < num_tiles; ++t) {
        const std::size_t a_col = t * TILE_SIZE + threadIdx.x;
        const std::size_t b_row = t * TILE_SIZE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < a.nrows() && a_col < k_dim) ? a(row, a_col) : AtomT{0};
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k_dim && col < b.ncols()) ? b(b_row, col) : AtomT{0};

        __syncthreads();

        for (unsigned int i = 0; i < TILE_SIZE; ++i) {
            sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < c.nrows() && col < c.ncols()) {
        c(row, col) = sum;
    }
}

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
Matrix<AtomT> NaiveMatMulStrategy<AtomT>::multiply(const Matrix<AtomT>& lhs,
                                                   const Matrix<AtomT>& rhs) const {
    if (lhs.ncols() != rhs.nrows()) {
        throw std::invalid_argument("Incompatible matrix dimensions for matmul");
    }

    const std::size_t m = lhs.nrows();
    const std::size_t n = rhs.ncols();
    Matrix<AtomT> result(m, n);

    if (m > 0 && n > 0) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid(
            static_cast<unsigned int>((n + block.x - 1) / block.x),
            static_cast<unsigned int>((m + block.y - 1) / block.y)
        );
        kernel_matmul_naive<AtomT><<<grid, block>>>(lhs.view(), rhs.view(), result.view());
        HSYS_CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
Matrix<AtomT> ShmemMatMulStrategy<AtomT>::multiply(const Matrix<AtomT>& lhs,
                                                   const Matrix<AtomT>& rhs) const {
    if (lhs.ncols() != rhs.nrows()) {
        throw std::invalid_argument("Incompatible matrix dimensions for matmul");
    }

    const std::size_t m = lhs.nrows();
    const std::size_t n = rhs.ncols();
    Matrix<AtomT> result(m, n);

    if (m > 0 && n > 0) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid(
            static_cast<unsigned int>((n + block.x - 1) / block.x),
            static_cast<unsigned int>((m + block.y - 1) / block.y)
        );
        kernel_matmul_shmem<AtomT><<<grid, block>>>(lhs.view(), rhs.view(), result.view());
        HSYS_CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
Matrix<AtomT> matmul(const Matrix<AtomT>& lhs,
                     const Matrix<AtomT>& rhs,
                     const IMatMulStrategy<AtomT>& strategy) {
    return strategy.multiply(lhs, rhs);
}

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs) {
    static const ShmemMatMulStrategy<AtomT> default_strategy{};
    return matmul(lhs, rhs, default_strategy);
}

template class NaiveMatMulStrategy<float>;
template class NaiveMatMulStrategy<double>;
template class NaiveMatMulStrategy<int>;

template class ShmemMatMulStrategy<float>;
template class ShmemMatMulStrategy<double>;
template class ShmemMatMulStrategy<int>;

template Matrix<float> matmul(const Matrix<float>&, const Matrix<float>&, const IMatMulStrategy<float>&);
template Matrix<double> matmul(const Matrix<double>&, const Matrix<double>&, const IMatMulStrategy<double>&);
template Matrix<int> matmul(const Matrix<int>&, const Matrix<int>&, const IMatMulStrategy<int>&);

template Matrix<float> operator*(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator*(const Matrix<double>&, const Matrix<double>&);
template Matrix<int> operator*(const Matrix<int>&, const Matrix<int>&);

} // namespace hsys
