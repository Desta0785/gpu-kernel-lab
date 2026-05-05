cmake_minimum_required(VERSION 3.20)
project(gpu-kernel-lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_HIP "Build HIP benchmark targets" OFF)

add_executable(bench_vectorAdd bench/vectorAdd.cu)
target_compile_options(bench_vectorAdd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

add_executable(bench_reduce_sum_benchmark bench/reduce_sum_benchmark.cu)
target_compile_options(bench_reduce_sum_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

add_executable(bench_transpose_benchmark bench/transpose_benchmark.cu)
target_compile_options(bench_transpose_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

add_executable(bench_softmax_benchmark bench/softmax_benchmark.cu)
target_compile_options(bench_softmax_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

if(BUILD_HIP)
    enable_language(HIP)

    add_executable(bench_transpose_hip hip/transpose_benchmark.hip.cpp)
    set_source_files_properties(
        hip/transpose_benchmark.hip.cpp
        PROPERTIES LANGUAGE HIP
    )
    target_compile_options(bench_transpose_hip PRIVATE -O3)
endif()
