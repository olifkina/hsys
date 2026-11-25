#pragma once

#include <cuda_fp16.h>

#include "hsys/cuda_check.cuh"

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>

namespace hsys {

template <typename AtomT>
requires (std::is_arithmetic_v<AtomT> || std::is_same_v<AtomT, __half>)
class Data {
private:
    std::size_t size_{0};
    AtomT* data_{nullptr};

public:
    explicit Data(std::size_t size) : size_(size) {
        if (size_ > 0) {
            HSYS_CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(AtomT)));
        }
    }

    Data(const Data& other) : size_(other.size_) {
        if (size_ > 0) {
            HSYS_CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(AtomT)));
            HSYS_CUDA_CHECK(cudaMemcpy(
                data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice));
        }
    }

    Data(Data&& other) noexcept
        : size_(std::exchange(other.size_, 0)),
          data_(std::exchange(other.data_, nullptr)) {}

    Data& operator=(const Data& other) {
        if (this != &other) {
            Data tmp(other);
            std::swap(size_, tmp.size_);
            std::swap(data_, tmp.data_);
        }
        return *this;
    }

    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            size_ = std::exchange(other.size_, 0);
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    ~Data() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    AtomT* data() noexcept { return data_; }
    const AtomT* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    void copy_to_host(AtomT* host_ptr) const {
        if (size_ > 0) {
            HSYS_CUDA_CHECK(cudaMemcpy(
                host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost));
        }
    }

    void copy_from_host(const AtomT* host_ptr) {
        if (size_ > 0) {
            HSYS_CUDA_CHECK(cudaMemcpy(
                data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice));
        }
    }
};

} // namespace hsys
