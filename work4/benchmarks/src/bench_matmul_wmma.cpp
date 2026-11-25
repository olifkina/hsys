#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#include "hsys/matmul_wmma.cuh"
#include "hsys/cuda_check.cuh"

static void run_half_benchmark(benchmark::State& state,
                               const hsys::IHalfMatMulStrategy& strategy) {
    const auto n = static_cast<std::size_t>(state.range(0));

    Eigen::MatrixXf a_f = Eigen::MatrixXf::Random(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    Eigen::MatrixXf b_f = Eigen::MatrixXf::Random(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    std::vector<__half> a_buf(n * n);
    std::vector<__half> b_buf(n * n);
    for (std::size_t i = 0; i < n * n; ++i) {
        a_buf[i] = __float2half(a_f.data()[i]);
        b_buf[i] = __float2half(b_f.data()[i]);
    }

    hsys::Matrix<__half> a_gpu(n, n);
    hsys::Matrix<__half> b_gpu(n, n);
    a_gpu.data().copy_from_host(a_buf.data());
    b_gpu.data().copy_from_host(b_buf.data());

    cudaEvent_t start, stop;
    HSYS_CUDA_CHECK(cudaEventCreate(&start));
    HSYS_CUDA_CHECK(cudaEventCreate(&stop));

    {
        auto warm = hsys::matmul_half(a_gpu, b_gpu, strategy);
        HSYS_CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (auto _ : state) {
        HSYS_CUDA_CHECK(cudaEventRecord(start));
        auto c_gpu = hsys::matmul_half(a_gpu, b_gpu, strategy);
        HSYS_CUDA_CHECK(cudaEventRecord(stop));
        HSYS_CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        HSYS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(static_cast<double>(ms) / 1000.0);
        benchmark::DoNotOptimize(c_gpu);
    }

    HSYS_CUDA_CHECK(cudaEventDestroy(start));
    HSYS_CUDA_CHECK(cudaEventDestroy(stop));
}

static void BM_CudaMatMulShmem(benchmark::State& state) {
    static const hsys::ShmemHalfMatMulStrategy strategy{};
    run_half_benchmark(state, strategy);
}

static void BM_CudaMatMulWmma(benchmark::State& state) {
    static const hsys::WmmaHalfMatMulStrategy strategy{};
    run_half_benchmark(state, strategy);
}

BENCHMARK(BM_CudaMatMulShmem)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CudaMatMulWmma)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
