add_executable(work4_tests
    src/test_matmul_wmma.cpp
)

target_link_libraries(work4_tests PRIVATE
    work4_core
    GTest::gtest_main
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(work4_tests)
