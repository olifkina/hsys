add_executable(work3_tests
    src/test_matmul_shmem.cpp
)

target_link_libraries(work3_tests PRIVATE
    work3_core
    GTest::gtest_main
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(work3_tests)
