#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cstddef>

#include "hsys/vecred.cuh"

class VecRedParamTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VecRedParamTest, NoBroadcastMatchesEigen) {
    const std::size_t n = GetParam();
    Eigen::VectorXf v = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
    const float expected = v.sum();

    hsys::Vector<float> v_gpu(n);
    v_gpu.data().copy_from_host(v.data());

    const float actual = hsys::vecred_nobr(v_gpu);
    EXPECT_NEAR(actual, expected, 1e-4f) << "Mismatch (nobr) for n=" << n;
}

TEST_P(VecRedParamTest, BroadcastMatchesEigen) {
    const std::size_t n = GetParam();
    Eigen::VectorXf v = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
    const float expected = v.sum();

    hsys::Vector<float> v_gpu(n);
    v_gpu.data().copy_from_host(v.data());

    const float actual = hsys::vecred_br(v_gpu);
    EXPECT_NEAR(actual, expected, 1e-4f) << "Mismatch (br) for n=" << n;
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    VecRedParamTest,
    ::testing::Values(1, 2, 3, 127, 129, 512, 541, 1037)
);
