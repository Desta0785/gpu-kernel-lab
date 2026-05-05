# Reduce Sum Optimization Case Study

## Problem

Compute the sum of a 1D `float` array.

The benchmark compares:

- a baseline shared-memory block reduction
- an optimized grid-stride + warp-shuffle reduction

The final reduction of block partial sums is completed on the CPU. This keeps the first benchmark focused on stage-1 GPU reduction performance and correctness.

## Correctness

Reference implementation:

- CPU sum with `double` accumulation

Checks:

- relative error <= `1e-5`, or
- absolute error <= `1e-3`

Test sizes include small edge cases and large inputs:

```text
1, 17, 255, 256, 257, 1023, 1024, 1025, 2^20, 2^24, 2^26
```

Command:

```bash
cmake -S . -B build
cmake --build build --target bench_reduce_sum_benchmark -j
./build/bench_reduce_sum_benchmark
```

## Baseline

The baseline kernel maps one CUDA block to a contiguous chunk of `2 * blockDim.x` elements.

Each block:

1. loads up to two elements per thread
2. writes each thread's local sum into shared memory
3. performs a shared-memory tree reduction
4. writes one partial sum to global memory

Expected bottlenecks:

- multiple `__syncthreads()` calls
- shared-memory traffic at every reduction step
- many partial sums for large inputs

## Optimization #1: Grid-stride accumulation

The optimized kernel limits the number of blocks and lets each block process a grid-stride chunk of the input.

Why this helps:

- reduces the number of partial sums written to global memory
- improves work per block for large inputs
- reduces CPU-side final accumulation cost

In this benchmark:

```cpp
optimized_grid = min(baseline_grid, 1024)
```

## Optimization #2: Warp-level reduction

Inside each block, the optimized kernel uses warp shuffle instructions:

```cpp
__shfl_down_sync(...)
```

Each warp first reduces its own values using register-to-register lane exchange. Then lane 0 of each warp writes one value to shared memory, and the first warp reduces those warp-level partial sums.

Why this helps:

- fewer shared-memory reads/writes
- fewer block-wide synchronization points
- better use of warp-level execution for reductions

## Results

Measured on local GPU environment:

```text
Reduce sum benchmark and correctness

N=1
  baseline   grid=     1  time_ms=  0.0079  GB/s=  0.0005  abs_err=0.0000  rel_err=0.0000
  optimized  grid=     1  time_ms=  0.0073  GB/s=  0.0005  speedup=1.0816x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=17
  baseline   grid=     1  time_ms=  0.0231  GB/s=  0.0029  abs_err=0.0000  rel_err=0.0000
  optimized  grid=     1  time_ms=  0.0209  GB/s=  0.0033  speedup=1.1056x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=255
  baseline   grid=     1  time_ms=  0.0102  GB/s=  0.1002  abs_err=0.0000  rel_err=0.0000
  optimized  grid=     1  time_ms=  0.0096  GB/s=  0.1060  speedup=1.0577x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=256
  baseline   grid=     1  time_ms=  0.0058  GB/s=  0.1762  abs_err=0.0000  rel_err=0.0000
  optimized  grid=     1  time_ms=  0.0061  GB/s=  0.1670  speedup=0.9480x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=257
  baseline   grid=     1  time_ms=  0.0174  GB/s=  0.0591  abs_err=0.0000  rel_err=0.0000
  optimized  grid=     1  time_ms=  0.0108  GB/s=  0.0955  speedup=1.6166x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=1023
  baseline   grid=     2  time_ms=  0.0093  GB/s=  0.4420  abs_err=0.0000  rel_err=0.0000
  optimized  grid=     2  time_ms=  0.0099  GB/s=  0.4120  speedup=0.9320x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=1024
  baseline   grid=     2  time_ms=  0.0066  GB/s=  0.6192  abs_err=0.0000  rel_err=0.0000
  optimized  grid=     2  time_ms=  0.0061  GB/s=  0.6734  speedup=1.0875x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=1025
  baseline   grid=     3  time_ms=  0.0059  GB/s=  0.6988  abs_err=0.0000  rel_err=0.0000
  optimized  grid=     3  time_ms=  0.0090  GB/s=  0.4571  speedup=0.6541x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=1048576
  baseline   grid=  2048  time_ms=  0.0084  GB/s=498.3094  abs_err=0.0000  rel_err=0.0000
  optimized  grid=  1024  time_ms=  0.0118  GB/s=354.1211  speedup=0.7106x  abs_err=0.0000  rel_err=0.0000
  correctness: PASS

N=16777216
  baseline   grid= 32768  time_ms=  0.0824  GB/s=814.1118  abs_err=0.0002  rel_err=0.0000
  optimized  grid=  1024  time_ms=  0.0260  GB/s=2580.1576  speedup=3.1693x  abs_err=0.0004  rel_err=0.0000
  correctness: PASS

N=67108864
  baseline   grid=131072  time_ms=  0.4432  GB/s=605.7019  abs_err=0.0008  rel_err=0.0000
  optimized  grid=  1024  time_ms=  0.4476  GB/s=599.6860  speedup=0.9901x  abs_err=0.0018  rel_err=0.0000
  correctness: PASS

All reduce sum tests passed.
```

| N | Baseline ms | Optimized ms | Speedup | Baseline GB/s | Optimized GB/s | Correctness |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0079 | 0.0073 | 1.0816x | 0.0005 | 0.0005 | PASS |
| 17 | 0.0231 | 0.0209 | 1.1056x | 0.0029 | 0.0033 | PASS |
| 255 | 0.0102 | 0.0096 | 1.0577x | 0.1002 | 0.1060 | PASS |
| 256 | 0.0058 | 0.0061 | 0.9480x | 0.1762 | 0.1670 | PASS |
| 257 | 0.0174 | 0.0108 | 1.6166x | 0.0591 | 0.0955 | PASS |
| 1023 | 0.0093 | 0.0099 | 0.9320x | 0.4420 | 0.4120 | PASS |
| 1024 | 0.0066 | 0.0061 | 1.0875x | 0.6192 | 0.6734 | PASS |
| 1025 | 0.0059 | 0.0090 | 0.6541x | 0.6988 | 0.4571 | PASS |
| 2^20 | 0.0084 | 0.0118 | 0.7106x | 498.3094 | 354.1211 | PASS |
| 2^24 | 0.0824 | 0.0260 | 3.1693x | 814.1118 | 2580.1576 | PASS |
| 2^26 | 0.4432 | 0.4476 | 0.9901x | 605.7019 | 599.6860 | PASS |

## Result Analysis

- Correctness passed for all tested sizes.
- For tiny inputs, timing is dominated by kernel launch and synchronization overhead, so bandwidth numbers are not very meaningful.
- The optimized kernel gives the best result at `2^24` elements, reaching `3.1693x` speedup and about `2580 GB/s` effective bandwidth.
- The optimized kernel is not universally faster. At `2^20` and `2^26`, the baseline is similar or faster. This suggests the optimal grid size and reduction strategy are input-size dependent.
- Next experiment: sweep `kMaxOptimizedGrid` and compare `512`, `1024`, `2048`, and `4096` blocks to find a better grid-size policy.

## Nsight Compute Metrics to Collect

Profiler commands used:

```bash
PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:reduce_sum_baseline_stage1_kernel \
  --force-overwrite -o reports/ncu/reduce_sum_baseline \
  ./build/bench_reduce_sum_benchmark

PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:reduce_sum_grid_stride_stage1_kernel \
  --force-overwrite -o reports/ncu/reduce_sum_optimized \
  ./build/bench_reduce_sum_benchmark
```

Generated local reports:

```text
reports/ncu/reduce_sum_baseline.ncu-rep
reports/ncu/reduce_sum_optimized.ncu-rep
```

## Profiler Evidence

Nsight Compute was run with `PROFILE_MODE=1`, which profiles a representative `N = 2^24` case with one benchmark iteration.

| Kernel | Duration | Memory Throughput | Compute Throughput | Achieved Occupancy | Grid Size | Registers / Thread | Dynamic Shared Memory / Block |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline shared-memory reduction | 104.67 us | 680.20 GB/s | 77.32% | 87.90% | 32768 | 16 | 1.02 KB |
| Grid-stride + warp-shuffle reduction | 110.37 us | 669.64 GB/s | 5.82% | 92.00% | 1024 | 29 | 32 B |

### Interpretation

- Both kernels are strongly memory-throughput oriented.
- The optimized kernel greatly reduces dynamic shared-memory usage per block: `1.02 KB` to `32 B`.
- The optimized kernel also reduces the number of launched blocks from `32768` to `1024`, which reduces the number of global partial sums.
- In this Nsight single-launch profile, the optimized kernel has similar memory throughput but slightly longer duration. This reinforces that the optimization is input-size and launch-policy dependent.
- The benchmark timing still shows a strong win at `2^24` in the normal benchmark run, so the next step is to profile multiple grid-size policies such as `512`, `1024`, `2048`, and `4096` blocks.

## Takeaways

1. A simple shared-memory reduction is a good correctness baseline, but it pays synchronization and shared-memory overhead.
2. Grid-stride accumulation reduces the number of partial sums and improves scalability for larger inputs.
3. Warp shuffle reduction reduces shared-memory traffic by doing most of the reduction through warp-level register exchange.
