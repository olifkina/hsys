#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace hsys {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            " -> " + cudaGetErrorString(err));
    }
}

} // namespace hsys

#define HSYS_CUDA_CHECK(call) ::hsys::cuda_check((call), __FILE__, __LINE__)
