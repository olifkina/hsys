add_executable(work5_tests
    src/test_vecred.cpp
)

target_link_libraries(work5_tests PRIVATE
    work5_core
    GTest::gtest_main
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(work5_tests)
