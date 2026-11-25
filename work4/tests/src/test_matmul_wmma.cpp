#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <tuple>
#include <vector>

#include "hsys/matmul_wmma.cuh"
#include "hsys/cuda_check.cuh"

using RowMajorMatrixXh = Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

TEST(MatMulWmmaBasicTest, DimensionMismatchThrows) {
    hsys::Matrix<__half> a(16, 32);
    hsys::Matrix<__half> b(16, 16);
    EXPECT_THROW(auto c = a * b, std::invalid_argument);
}

class MatMulWmmaParamTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(MatMulWmmaParamTest, MatchesEigen) {
    const auto [m, k, n] = GetParam();

    RowMajorMatrixXf a_f = RowMajorMatrixXf::Random(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
    RowMajorMatrixXf b_f = RowMajorMatrixXf::Random(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));

    RowMajorMatrixXh a_h = a_f.cast<Eigen::half>();
    RowMajorMatrixXh b_h = b_f.cast<Eigen::half>();
    RowMajorMatrixXh expected_h = a_h * b_h;
    Eigen::MatrixXf expected = expected_h.cast<float>();

    std::vector<__half> a_buf(m * k);
    std::vector<__half> b_buf(k * n);
    for (std::size_t i = 0; i < m * k; ++i) a_buf[i] = __float2half(static_cast<float>(a_h.data()[i]));
    for (std::size_t i = 0; i < k * n; ++i) b_buf[i] = __float2half(static_cast<float>(b_h.data()[i]));

    hsys::Matrix<__half> a_gpu(m, k);
    hsys::Matrix<__half> b_gpu(k, n);
    a_gpu.data().copy_from_host(a_buf.data());
    b_gpu.data().copy_from_host(b_buf.data());

    hsys::Matrix<__half> c_gpu = a_gpu * b_gpu;
    HSYS_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> c_buf(m * n);
    c_gpu.data().copy_to_host(c_buf.data());

    RowMajorMatrixXf actual_rm(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < m * n; ++i) actual_rm.data()[i] = __half2float(c_buf[i]);
    Eigen::MatrixXf actual = actual_rm;

    EXPECT_TRUE(actual.isApprox(expected, 1e-2f))
        << "Mismatch for m=" << m << ", k=" << k << ", n=" << n;
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    MatMulWmmaParamTest,
    ::testing::Combine(
        ::testing::Values(16, 32, 64, 128, 256, 512),
        ::testing::Values(16, 32, 64, 128, 256, 512),
        ::testing::Values(16, 32, 64, 128, 256, 512)
    )
);
