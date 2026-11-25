add_executable(work3_benchmarks
    src/bench_matmul_shmem.cpp
)

target_link_libraries(work3_benchmarks PRIVATE
    work3_core
    benchmark::benchmark_main
    Eigen3::Eigen
)
