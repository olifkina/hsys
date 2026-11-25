#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>

#include "hsys/matrix.cuh"
#include "hsys/cuda_check.cuh"

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

TEST(MatrixBasicTest, ViewIsTriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable_v<hsys::MatrixView<float>>);
    EXPECT_TRUE(std::is_trivially_copyable_v<hsys::MatrixView<double>>);
}

TEST(MatrixBasicTest, DimensionMismatchThrows) {
    hsys::Matrix<float> a(3, 4);
    hsys::Matrix<float> b(5, 2);
    EXPECT_THROW(auto c = a * b, std::invalid_argument);
}

class MatMulParamTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(MatMulParamTest, MatchesEigen) {
    const auto [m, k, n] = GetParam();

    Eigen::MatrixXf a_eigen = Eigen::MatrixXf::Random(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
    Eigen::MatrixXf b_eigen = Eigen::MatrixXf::Random(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
    Eigen::MatrixXf expected = a_eigen * b_eigen;

    RowMajorMatrixXf a_rm = a_eigen;
    RowMajorMatrixXf b_rm = b_eigen;

    hsys::Matrix<float> a_gpu(m, k);
    hsys::Matrix<float> b_gpu(k, n);
    a_gpu.data().copy_from_host(a_rm.data());
    b_gpu.data().copy_from_host(b_rm.data());

    hsys::Matrix<float> c_gpu = a_gpu * b_gpu;
    HSYS_CUDA_CHECK(cudaDeviceSynchronize());

    RowMajorMatrixXf c_rm(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
    c_gpu.data().copy_to_host(c_rm.data());

    Eigen::MatrixXf actual = c_rm;
    EXPECT_TRUE(actual.isApprox(expected, 1e-5f))
        << "Mismatch for m=" << m << ", k=" << k << ", n=" << n;
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    MatMulParamTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512)
    )
);
