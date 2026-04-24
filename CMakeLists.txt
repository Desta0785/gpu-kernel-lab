cmake_minimum_required(VERSION 3.20)
project(gpu-kernel-lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(bench_vectorAdd bench/vectorAdd.cu)
target_compile_options(bench_vectorAdd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)