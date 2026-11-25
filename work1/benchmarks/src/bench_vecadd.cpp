#include "hsys/vector.cuh"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstddef>

static void BM_CudaVecAdd(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Eigen::VectorXf host_a = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
    Eigen::VectorXf host_b = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));

    hsys::Vector<float> a(n);
    hsys::Vector<float> b(n);
    a.data().copy_from_host(host_a.data());
    b.data().copy_from_host(host_b.data());

    // Прогрев GPU
    {
        auto warmup = a + b;
        HSYS_CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    HSYS_CUDA_CHECK(cudaEventCreate(&start));
    HSYS_CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state) {
        HSYS_CUDA_CHECK(cudaEventRecord(start));
        auto c = a + b;
        HSYS_CUDA_CHECK(cudaEventRecord(stop));
        HSYS_CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HSYS_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        state.SetIterationTime(static_cast<double>(elapsed_ms) * 1e-3);
        benchmark::DoNotOptimize(c.data().data());
    }

    HSYS_CUDA_CHECK(cudaEventDestroy(start));
    HSYS_CUDA_CHECK(cudaEventDestroy(stop));
}

static void BM_EigenVecAdd(benchmark::State& state) {
    const Eigen::Index n = static_cast<Eigen::Index>(state.range(0));

    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);

    for (auto _ : state) {
        Eigen::VectorXf c = a + b;
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_CudaVecAdd)
    ->RangeMultiplier(8)
    ->Range(8, 16777216) // 8^1 .. 8^8
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_EigenVecAdd)
    ->RangeMultiplier(8)
    ->Range(8, 16777216) // 8^1 .. 8^8
    ->Unit(benchmark::kMillisecond);
