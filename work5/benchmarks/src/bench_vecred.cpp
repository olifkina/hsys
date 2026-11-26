#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "hsys/vecred.cuh"
#include "hsys/cuda_check.cuh"

template <bool UseBroadcast>
static void BM_CudaVecRed(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));

    hsys::Vector<float> v_gpu(n);
    // Инициализируем память на устройстве без выделения гигантского буфера в RAM
    HSYS_CUDA_CHECK(cudaMemset(v_gpu.data().data(), 0, n * sizeof(float)));

    float* d_out = nullptr;
    HSYS_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    cudaEvent_t start, stop;
    HSYS_CUDA_CHECK(cudaEventCreate(&start));
    HSYS_CUDA_CHECK(cudaEventCreate(&stop));

    // Прогрев
    if constexpr (UseBroadcast) {
        hsys::vecred_br_device(v_gpu.view(), d_out);
    } else {
        hsys::vecred_nobr_device(v_gpu.view(), d_out);
    }
    HSYS_CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        HSYS_CUDA_CHECK(cudaEventRecord(start));
        if constexpr (UseBroadcast) {
            hsys::vecred_br_device(v_gpu.view(), d_out);
        } else {
            hsys::vecred_nobr_device(v_gpu.view(), d_out);
        }
        HSYS_CUDA_CHECK(cudaEventRecord(stop));
        HSYS_CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        HSYS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(static_cast<double>(ms) / 1000.0);
    }

    HSYS_CUDA_CHECK(cudaEventDestroy(start));
    HSYS_CUDA_CHECK(cudaEventDestroy(stop));
    HSYS_CUDA_CHECK(cudaFree(d_out));
}

static void BM_CudaVecRedNoBr(benchmark::State& state) {
    BM_CudaVecRed<false>(state);
}

static void BM_CudaVecRedBr(benchmark::State& state) {
    BM_CudaVecRed<true>(state);
}

BENCHMARK(BM_CudaVecRedNoBr)
    ->RangeMultiplier(2)
    ->Range(8, 8LL << 27)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_CudaVecRedBr)
    ->RangeMultiplier(2)
    ->Range(8, 8LL << 27)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
