add_library(work2_core STATIC
    src/matmul.cu
)

target_include_directories(work2_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(work2_core PUBLIC work1_core)

set_target_properties(work2_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
