add_library(work3_core STATIC
    src/matmul_shmem.cu
)

target_include_directories(work3_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/work2/core/include
)

target_link_libraries(work3_core PUBLIC work1_core)

set_target_properties(work3_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
