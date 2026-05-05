# HIP Port Notes

## Goal

This document describes the initial CUDA-to-HIP portability work for this repository.

The first HIP-readiness target is:

```text
hip/transpose_benchmark.hip.cpp
```

It is based on:

```text
bench/transpose_benchmark.cu
```

## Why Transpose First

Matrix transpose is a good first HIP port target because it uses common CUDA/HIP programming concepts:

- global memory
- shared memory
- thread/block indexing
- device memory allocation
- device-host memory copies
- event-based timing

It does not require CUDA-specific warp intrinsics, so the CUDA-to-HIP mapping is straightforward.

## CUDA to HIP API Mapping

| CUDA | HIP |
|---|---|
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` |
| `cudaError_t` | `hipError_t` |
| `cudaSuccess` | `hipSuccess` |
| `cudaGetErrorString` | `hipGetErrorString` |
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` |
| `cudaEvent_t` | `hipEvent_t` |
| `cudaEventCreate` | `hipEventCreate` |
| `cudaEventRecord` | `hipEventRecord` |
| `cudaEventSynchronize` | `hipEventSynchronize` |
| `cudaEventElapsedTime` | `hipEventElapsedTime` |
| `cudaEventDestroy` | `hipEventDestroy` |
| `cudaGetLastError` | `hipGetLastError` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |

## Kernel Syntax

These CUDA kernel concepts are also supported in HIP:

```cpp
__global__
__shared__
__syncthreads()
threadIdx
blockIdx
blockDim
gridDim
dim3
```

For kernel launches, this HIP port uses:

```cpp
hipLaunchKernelGGL(...)
```

instead of CUDA triple-chevron launch syntax.

## Build

HIP build is optional and disabled by default.

Expected CMake flow on a ROCm/HIP system:

```bash
cmake -S . -B build-hip -DBUILD_HIP=ON
cmake --build build-hip --target bench_transpose_hip -j
```

## Validation Status

The CUDA version has been validated on the current NVIDIA development machine.

The HIP version is intended for ROCm systems. Since the current development machine is NVIDIA-only, HIP runtime validation should be performed on AMD hardware.

Expected validation command on a ROCm machine:

```bash
./build-hip/bench_transpose_hip
```

Expected result:

```text
All HIP transpose tests passed.
```

## Next Steps

- Validate correctness on AMD GPU hardware.
- Profile the HIP version with ROCm profiling tools.
- Compare CUDA Nsight Compute results with ROCm profiler metrics.
- Port reduction or softmax kernels next.
