#include "hsys/vector.cuh"

#include <stdexcept>

namespace hsys {

template <typename AtomT>
__global__ void kernel_vecadd(VectorView<AtomT> a,
                              VectorView<AtomT> b,
                              VectorView<AtomT> c) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < c.size()) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vector sizes must match for operator+");
    }

    const std::size_t n = lhs.size();
    Vector<AtomT> result(n);

    if (n > 0) {
        constexpr unsigned int block_size = 256;
        const unsigned int grid_size =
            static_cast<unsigned int>((n + block_size - 1) / block_size);
        kernel_vecadd<AtomT><<<grid_size, block_size>>>(
            lhs.view(), rhs.view(), result.view());
        HSYS_CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

template class Data<float>;
template class Data<double>;
template class VectorView<float>;
template class VectorView<double>;
template class Vector<float>;
template class Vector<double>;

template Vector<float> operator+(const Vector<float>&, const Vector<float>&);
template Vector<double> operator+(const Vector<double>&, const Vector<double>&);

} // namespace hsys
