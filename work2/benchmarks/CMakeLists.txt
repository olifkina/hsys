add_executable(work2_benchmarks
    src/bench_matmul.cpp
)

target_link_libraries(work2_benchmarks PRIVATE
    work2_core
    benchmark::benchmark_main
    Eigen3::Eigen
)
