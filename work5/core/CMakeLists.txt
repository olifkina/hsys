add_library(work5_core STATIC
    src/vecred.cu
)

target_include_directories(work5_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(work5_core PUBLIC work1_core)

set_target_properties(work5_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
