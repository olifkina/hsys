#include "hsys/matmul_wmma.cuh"
#include "hsys/cuda_check.cuh"

#include <cuda_runtime.h>
#include <mma.h>
#include <stdexcept>

namespace hsys {

using namespace nvcuda;

constexpr unsigned int TILE_SIZE = 16;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void kernel_matmul_shmem_half(MatrixView<__half> a,
                                         MatrixView<__half> b,
                                         MatrixView<__half> c) {
    __shared__ __half tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ __half tile_b[TILE_SIZE][TILE_SIZE];

    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * TILE_SIZE + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * TILE_SIZE + threadIdx.x;
    const std::size_t k_dim = a.ncols();
    const std::size_t num_tiles = (k_dim + TILE_SIZE - 1) / TILE_SIZE;

    float sum = 0.0f;

    for (std::size_t t = 0; t < num_tiles; ++t) {
        const std::size_t a_col = t * TILE_SIZE + threadIdx.x;
        const std::size_t b_row = t * TILE_SIZE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < a.nrows() && a_col < k_dim) ? a(row, a_col) : __float2half(0.0f);
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k_dim && col < b.ncols()) ? b(b_row, col) : __float2half(0.0f);

        __syncthreads();

        for (unsigned int i = 0; i < TILE_SIZE; ++i) {
            sum += __half2float(tile_a[threadIdx.y][i]) * __half2float(tile_b[i][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < c.nrows() && col < c.ncols()) {
        c(row, col) = __float2half(sum);
    }
}

__global__ void kernel_matmul_wmma(MatrixView<__half> a,
                                   MatrixView<__half> b,
                                   MatrixView<__half> c) {
    const std::size_t warp_row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t warp_col = static_cast<std::size_t>(blockIdx.x);

    const std::size_t c_row = warp_row * WMMA_M;
    const std::size_t c_col = warp_col * WMMA_N;

    if (c_row >= c.nrows() || c_col >= c.ncols()) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    const std::size_t k_dim = a.ncols();
    const std::size_t lda = a.ncols();
    const std::size_t ldb = b.ncols();
    const std::size_t ldc = c.ncols();

    for (std::size_t k = 0; k < k_dim; k += WMMA_K) {
        const half* a_ptr = &a(c_row, k);
        const half* b_ptr = &b(k, c_col);

        wmma::load_matrix_sync(a_frag, a_ptr, static_cast<unsigned int>(lda));
        wmma::load_matrix_sync(b_frag, b_ptr, static_cast<unsigned int>(ldb));
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    half* c_ptr = &c(c_row, c_col);
    wmma::store_matrix_sync(c_ptr, c_frag, static_cast<unsigned int>(ldc), wmma::mem_row_major);
}

Matrix<__half> ShmemHalfMatMulStrategy::multiply(const Matrix<__half>& lhs,
                                                 const Matrix<__half>& rhs) const {
    if (lhs.ncols() != rhs.nrows()) {
        throw std::invalid_argument("Incompatible matrix dimensions for matmul");
    }

    const std::size_t m = lhs.nrows();
    const std::size_t n = rhs.ncols();
    Matrix<__half> result(m, n);

    if (m > 0 && n > 0) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid(
            static_cast<unsigned int>((n + block.x - 1) / block.x),
            static_cast<unsigned int>((m + block.y - 1) / block.y)
        );
        kernel_matmul_shmem_half<<<grid, block>>>(lhs.view(), rhs.view(), result.view());
        HSYS_CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

Matrix<__half> WmmaHalfMatMulStrategy::multiply(const Matrix<__half>& lhs,
                                                const Matrix<__half>& rhs) const {
    if (lhs.ncols() != rhs.nrows()) {
        throw std::invalid_argument("Incompatible matrix dimensions for WMMA matmul");
    }

    const std::size_t m = lhs.nrows();
    const std::size_t k = lhs.ncols();
    const std::size_t n = rhs.ncols();

    if (m % WMMA_M != 0 || k % WMMA_K != 0 || n % WMMA_N != 0) {
        throw std::invalid_argument("WMMA requires matrix dimensions to be multiples of 16");
    }

    Matrix<__half> result(m, n);

    if (m > 0 && n > 0) {
        constexpr unsigned int warps_per_block = 4;
        const dim3 block(32, warps_per_block);
        const dim3 grid(
            static_cast<unsigned int>(n / WMMA_N),
            static_cast<unsigned int>((m / WMMA_M + warps_per_block - 1) / warps_per_block)
        );
        kernel_matmul_wmma<<<grid, block>>>(lhs.view(), rhs.view(), result.view());
        HSYS_CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

Matrix<__half> matmul_half(const Matrix<__half>& lhs,
                           const Matrix<__half>& rhs,
                           const IHalfMatMulStrategy& strategy) {
    return strategy.multiply(lhs, rhs);
}

template <>
Matrix<__half> operator*<__half>(const Matrix<__half>& lhs, const Matrix<__half>& rhs) {
    static const WmmaHalfMatMulStrategy default_strategy{};
    return matmul_half(lhs, rhs, default_strategy);
}

} // namespace hsys
