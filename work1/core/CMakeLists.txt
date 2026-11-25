add_library(work1_core STATIC
    src/vecadd.cu
)

target_include_directories(work1_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(work1_core
    PUBLIC
        CUDA::cudart
)

set_target_properties(work1_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
