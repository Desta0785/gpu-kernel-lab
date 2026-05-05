# CUDA Kernel Optimization Suite

This repository contains small CUDA benchmarks focused on GPU kernel performance, correctness, and profiling.

The goal is to build a compact portfolio project that demonstrates:

- C++/CUDA implementation skills
- baseline-first performance engineering
- correctness testing against CPU references
- incremental kernel optimization
- benchmark reporting and case-study writing

## Benchmarks

| Benchmark | File | CMake Target | Implementations |
|---|---|---|---|
| Reduce Sum | `bench/reduce_sum_benchmark.cu` | `bench_reduce_sum_benchmark` | shared-memory baseline, grid-stride + warp-shuffle optimized |
| Matrix Transpose | `bench/transpose_benchmark.cu` | `bench_transpose_benchmark` | naive global-memory, tiled shared-memory |
| Row-wise Softmax | `bench/softmax_benchmark.cu` | `bench_softmax_benchmark` | shared-memory baseline, warp reduction, warp + `float4` |

Optional HIP readiness target:

| Benchmark | File | CMake Target | Notes |
|---|---|---|---|
| Matrix Transpose HIP | `hip/transpose_benchmark.hip.cpp` | `bench_transpose_hip` | disabled by default; requires ROCm/HIP toolchain |

## Build

Requirements:

- CMake 3.20+
- C++17 compiler
- CUDA toolkit
- NVIDIA GPU + compatible driver

Build all benchmarks:

```bash
cmake -S . -B build
cmake --build build -j
```

Build one target:

```bash
cmake --build build --target bench_reduce_sum_benchmark -j
cmake --build build --target bench_transpose_benchmark -j
cmake --build build --target bench_softmax_benchmark -j
```

Build the optional HIP target on a ROCm/HIP system:

```bash
cmake -S . -B build-hip -DBUILD_HIP=ON
cmake --build build-hip --target bench_transpose_hip -j
```

## Run

```bash
./build/bench_reduce_sum_benchmark
./build/bench_transpose_benchmark
./build/bench_softmax_benchmark
```

Each benchmark prints:

- correctness status
- timing in milliseconds
- approximate effective bandwidth
- speedup for optimized versions

## Case Studies

| Case Study | Description |
|---|---|
| [`case_studies/reduce_sum_case_study.md`](case_studies/reduce_sum_case_study.md) | Reduction baseline, grid-stride accumulation, warp shuffle reduction |
| [`case_studies/transpose_case_study.md`](case_studies/transpose_case_study.md) | Naive transpose vs tiled shared-memory transpose |
| [`case_studies/softmax_case_study.md`](case_studies/softmax_case_study.md) | Row-wise softmax baseline, warp reduction, `float4` vectorization |

HIP port notes:

- [`docs/hip_port.md`](docs/hip_port.md)

## Profiling

Suggested Nsight Compute commands:

```bash
mkdir -p reports

PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:reduce_sum_baseline_stage1_kernel \
  --force-overwrite -o reports/ncu/reduce_sum_baseline \
  ./build/bench_reduce_sum_benchmark

PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:transpose_tiled_kernel \
  --force-overwrite -o reports/ncu/transpose_tiled \
  ./build/bench_transpose_benchmark

PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:softmax_warp_float4_kernel \
  --force-overwrite -o reports/ncu/softmax_float4 \
  ./build/bench_softmax_benchmark
```

Metrics to inspect:

- memory throughput
- achieved occupancy
- warp stall reasons
- shared-memory usage
- shared-memory bank conflicts
- instruction throughput

## Current Status

- [x] Reduction baseline + correctness
- [x] Reduction optimized benchmark
- [x] Reduction case study template
- [x] Transpose naive + tiled benchmark
- [x] Transpose case study template
- [x] Softmax baseline + correctness
- [x] Softmax warp reduction optimization
- [x] Softmax `float4` optimization
- [x] Softmax case study with measured results
- [x] Add HIP readiness notes
- [x] Add Nsight Compute exported metrics for every case study

## Notes

The benchmark bandwidth numbers are approximate effective bandwidths. They are useful for comparing versions inside this repository, but final performance claims should be backed by profiler metrics.
