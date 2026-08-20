cmake_minimum_required(VERSION 3.18)
project(SgemmPortfolio LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cu
    src/warmup.cu
    src/slab_allocator.cu
)

add_executable(slab_allocator ${SOURCES})

target_compile_options(slab_allocator PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

set_source_files_properties(src/slab_allocator.cu
    PROPERTIES COMPILE_OPTIONS
    "-Xptxas=-v;-Xptxas=-warn-spills"
)

target_link_libraries(slab_allocator PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    OpenMP::OpenMP_CXX
)

