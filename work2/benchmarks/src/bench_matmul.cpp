#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "hsys/matrix.cuh"
#include "hsys/cuda_check.cuh"

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static void BM_CudaMatMulNaive(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    RowMajorMatrixXf a_host = RowMajorMatrixXf::Random(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    RowMajorMatrixXf b_host = RowMajorMatrixXf::Random(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    hsys::Matrix<float> a_gpu(n, n);
    hsys::Matrix<float> b_gpu(n, n);
    a_gpu.data().copy_from_host(a_host.data());
    b_gpu.data().copy_from_host(b_host.data());

    cudaEvent_t start, stop;
    HSYS_CUDA_CHECK(cudaEventCreate(&start));
    HSYS_CUDA_CHECK(cudaEventCreate(&stop));

    // Прогрев
    {
        auto warm = a_gpu * b_gpu;
        HSYS_CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (auto _ : state) {
        HSYS_CUDA_CHECK(cudaEventRecord(start));
        auto c_gpu = a_gpu * b_gpu;
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

static void BM_EigenMatMul(benchmark::State& state) {
    const auto n = static_cast<Eigen::Index>(state.range(0));
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf c(n, n);

    for (auto _ : state) {
        c.noalias() = a * b;
        benchmark::DoNotOptimize(c.data());
    }
}

BENCHMARK(BM_CudaMatMulNaive)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_EigenMatMul)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->Unit(benchmark::kMillisecond);
