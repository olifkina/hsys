#include "hsys/vector.cuh"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

TEST(VectorViewTest, IsTriviallyCopyable) {
    EXPECT_TRUE((std::is_trivially_copyable_v<hsys::VectorView<float>>));
    EXPECT_TRUE((std::is_trivially_copyable_v<hsys::VectorView<double>>));
}

TEST(DataTest, CopyAndMoveSemantics) {
    std::vector<float> host_in = {1.0f, 2.0f, 3.0f, 4.0f};
    hsys::Data<float> d1(host_in.size());
    d1.copy_from_host(host_in.data());

    hsys::Data<float> d2(d1);
    std::vector<float> host_out(host_in.size(), 0.0f);
    d2.copy_to_host(host_out.data());
    EXPECT_EQ(host_out, host_in);

    hsys::Data<float> d3(std::move(d2));
    EXPECT_EQ(d2.size(), 0u);
    EXPECT_EQ(d2.data(), nullptr);
    std::fill(host_out.begin(), host_out.end(), 0.0f);
    d3.copy_to_host(host_out.data());
    EXPECT_EQ(host_out, host_in);
}

TEST(VectorAddTest, MismatchedSizesThrows) {
    hsys::Vector<float> a(10);
    hsys::Vector<float> b(11);
    EXPECT_THROW((void)(a + b), std::invalid_argument);
}

class VecAddParamTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VecAddParamTest, MatchesEigen) {
    const std::size_t n = GetParam();

    Eigen::VectorXf ea = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
    Eigen::VectorXf eb = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
    Eigen::VectorXf expected = ea + eb;

    hsys::Vector<float> a(n);
    hsys::Vector<float> b(n);
    a.data().copy_from_host(ea.data());
    b.data().copy_from_host(eb.data());

    hsys::Vector<float> c = a + b;
    HSYS_CUDA_CHECK(cudaDeviceSynchronize());

    Eigen::VectorXf actual(static_cast<Eigen::Index>(n));
    c.data().copy_to_host(actual.data());

    EXPECT_TRUE(actual.isApprox(expected, 1e-6f));
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    VecAddParamTest,
    ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029)
);
