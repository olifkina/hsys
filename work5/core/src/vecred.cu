#include "hsys/vecred.cuh"
#include "hsys/cuda_check.cuh"

#include <cuda_runtime.h>
#include <algorithm>

namespace hsys {

constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int MAX_GRID_SIZE = 1024;

template <typename AtomT>
__global__ void kernel_vecred_nobr(VectorView<AtomT> vec, AtomT* out) {
    __shared__ AtomT sdata[BLOCK_SIZE];

    const unsigned int tid = threadIdx.x;
    const std::size_t n = vec.size();
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    AtomT local_sum = AtomT{0};
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
         idx < n;
         idx += stride) {
        local_sum += vec[idx];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

template <typename AtomT>
__device__ __forceinline__ AtomT warp_reduce_sum(AtomT val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

template <typename AtomT>
__global__ void kernel_vecred_br(VectorView<AtomT> vec, AtomT* out) {
    __shared__ AtomT warp_sums[BLOCK_SIZE / 32];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp_id = tid >> 5;
    const std::size_t n = vec.size();
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    AtomT val = AtomT{0};
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
         idx < n;
         idx += stride) {
        val += vec[idx];
    }

    val = warp_reduce_sum(val);

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        constexpr unsigned int num_warps = BLOCK_SIZE / 32;
        val = (lane < num_warps) ? warp_sums[lane] : AtomT{0};
        val = warp_reduce_sum(val);
        if (lane == 0) {
            atomicAdd(out, val);
        }
    }
}

static unsigned int calc_grid_size(std::size_t n) {
    if (n == 0) return 0;
    const std::size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    return static_cast<unsigned int>(std::min<std::size_t>(blocks, MAX_GRID_SIZE));
}

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
void vecred_nobr_device(VectorView<AtomT> vec, AtomT* d_out) {
    HSYS_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(AtomT)));
    const unsigned int grid = calc_grid_size(vec.size());
    if (grid > 0) {
        kernel_vecred_nobr<AtomT><<<grid, BLOCK_SIZE>>>(vec, d_out);
        HSYS_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
void vecred_br_device(VectorView<AtomT> vec, AtomT* d_out) {
    HSYS_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(AtomT)));
    const unsigned int grid = calc_grid_size(vec.size());
    if (grid > 0) {
        kernel_vecred_br<AtomT><<<grid, BLOCK_SIZE>>>(vec, d_out);
        HSYS_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
AtomT vecred_nobr(const Vector<AtomT>& vec) {
    AtomT* d_out = nullptr;
    HSYS_CUDA_CHECK(cudaMalloc(&d_out, sizeof(AtomT)));
    vecred_nobr_device(vec.view(), d_out);
    AtomT result = AtomT{0};
    HSYS_CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(AtomT), cudaMemcpyDeviceToHost));
    HSYS_CUDA_CHECK(cudaFree(d_out));
    return result;
}

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
AtomT vecred_br(const Vector<AtomT>& vec) {
    AtomT* d_out = nullptr;
    HSYS_CUDA_CHECK(cudaMalloc(&d_out, sizeof(AtomT)));
    vecred_br_device(vec.view(), d_out);
    AtomT result = AtomT{0};
    HSYS_CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(AtomT), cudaMemcpyDeviceToHost));
    HSYS_CUDA_CHECK(cudaFree(d_out));
    return result;
}

template float vecred_nobr<float>(const Vector<float>&);
template double vecred_nobr<double>(const Vector<double>&);
template float vecred_br<float>(const Vector<float>&);
template double vecred_br<double>(const Vector<double>&);

template void vecred_nobr_device<float>(VectorView<float>, float*);
template void vecred_nobr_device<double>(VectorView<double>, double*);
template void vecred_br_device<float>(VectorView<float>, float*);
template void vecred_br_device<double>(VectorView<double>, double*);

} // namespace hsys
