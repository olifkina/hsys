add_executable(work2_tests
    src/test_matmul.cpp
)

target_link_libraries(work2_tests PRIVATE
    work2_core
    GTest::gtest_main
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(work2_tests)
