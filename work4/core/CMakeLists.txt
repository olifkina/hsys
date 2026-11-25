add_library(work4_core STATIC
    src/matmul_wmma.cu
)

target_include_directories(work4_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/work2/core/include
)

target_link_libraries(work4_core PUBLIC work1_core)

set_target_properties(work4_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)
