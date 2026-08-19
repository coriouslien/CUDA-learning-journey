cmake_minimum_required(VERSION 3.18)
project(SgemmProtfolio LANGUAGES CXX CUDA)

# Tell CMake to generate the file VS Code needs for IntelliSense!
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

set(SOURCES
    src/main.cpp
    src/warmup.cu
    src/sgemm_regs_tiling_runner.cu
    #kernels/03_sgemm_register_tiling.cu
    #kernels/03_sgemm_register_tiling_float4.cu
    kernels/03_sgemm_register_tiling_swizzing.cu
    
)


add_executable(sgemm_benchmarker ${SOURCES})


# CUDA general compile flags for all *.cu files
target_compile_options(sgemm_benchmarker PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info> # line info for ncu profile
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>                  # optimization level
    $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)


# ── ptxas flags ONLY for kernel files ────────────────────────────────────────
set(KERNEL_SOURCES
    #kernels/03_sgemm_register_tiling.cu
    #kernels/03_sgemm_register_tiling_float4.cu
    kernels/03_sgemm_register_tiling_swizzing.cu
)

set_source_files_properties(${KERNEL_SOURCES}
    PROPERTIES COMPILE_OPTIONS
    "-Xptxas=-v;-Xptxas=-warn-spills"
)

target_link_libraries(sgemm_benchmarker
    PRIVATE
    OpenMP::OpenMP_CXX
    cublas
)

set_source_files_properties(${KERNEL_SOURCES}
    PROPERTIES COMPILE_OPTIONS
    "-Xptxas=-v;-Xptxas=-warn-spills"
)

