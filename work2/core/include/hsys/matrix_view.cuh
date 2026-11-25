#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <type_traits>

namespace hsys {

template <typename AtomT>
requires (std::is_arithmetic_v<AtomT> || std::is_same_v<AtomT, __half>)
class MatrixView {
private:
    AtomT* data_{nullptr};
    std::size_t nrows_{0};
    std::size_t ncols_{0};

public:
    MatrixView() = default;

    __host__ __device__ MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols) noexcept
        : data_(data), nrows_(nrows), ncols_(ncols) {}

    __host__ __device__ std::size_t size() const noexcept { return nrows_ * ncols_; }
    __host__ __device__ std::size_t nrows() const noexcept { return nrows_; }
    __host__ __device__ std::size_t ncols() const noexcept { return ncols_; }

    __host__ __device__ AtomT& operator[](std::size_t n) noexcept { return data_[n]; }
    __host__ __device__ const AtomT& operator[](std::size_t n) const noexcept { return data_[n]; }

    __host__ __device__ AtomT& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * ncols_ + j];
    }
    __host__ __device__ const AtomT& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * ncols_ + j];
    }
};

static_assert(std::is_trivially_copyable_v<MatrixView<float>>,
              "MatrixView must be trivially copyable");

} // namespace hsys
