# Matrix Transpose Optimization Case Study

## Problem

Compute matrix transpose:

```text
output[col, row] = input[row, col]
```

The benchmark compares:

- a naive global-memory transpose
- a tiled shared-memory transpose

## Correctness

Reference implementation:

- CPU transpose

Checks:

- every output element must match the CPU reference within `1e-5`

Command:

```bash
cmake -S . -B build
cmake --build build --target bench_transpose_benchmark -j
./build/bench_transpose_benchmark
```

Current benchmark shapes:

```text
1024 x 1024
2048 x 2048
4096 x 4096
4096 x 3072
```

## Baseline: Naive Transpose

The naive kernel maps one thread to one matrix element:

```cpp
output[col * rows + row] = input[row * cols + col];
```

Expected bottleneck:

- reads from input are coalesced
- writes to output are strided and poorly coalesced

This makes the naive transpose memory-inefficient, especially for large matrices.

## Optimization: Tiled Shared-memory Transpose

The optimized kernel uses a `32 x 32` tile stored in shared memory:

```cpp
__shared__ float tile[32][33];
```

Each block:

1. loads a tile from global memory into shared memory
2. synchronizes the block
3. writes the transposed tile back to global memory

Why this helps:

- global input loads are coalesced
- global output stores become coalesced after tile reordering
- shared memory is used to convert strided global writes into coalesced global writes

## Optimization Detail: Bank-conflict Padding

The shared memory tile is declared as:

```cpp
tile[32][33]
```

The extra column avoids many shared-memory bank conflicts when reading the tile in transposed order.

Without padding:

```cpp
tile[32][32]
```

threads in a warp can access addresses that map to the same shared-memory bank during the transpose phase.

## Results

Measured on local GPU environment:

```text
rows=1024 cols=1024 naive_ms=0.0445926 naive_GBps=188.116 tiled_ms=0.0078336 tiled_GBps=1070.85 speedup=5.69248x
rows=2048 cols=2048 naive_ms=0.127631 naive_GBps=262.902 tiled_ms=0.0120938 tiled_GBps=2774.52 speedup=10.5535x
rows=4096 cols=4096 naive_ms=0.504746 naive_GBps=265.911 tiled_ms=0.23559 tiled_GBps=569.709 speedup=2.14248x
rows=4096 cols=3072 naive_ms=0.383878 naive_GBps=262.227 tiled_ms=0.17706 tiled_GBps=568.526 speedup=2.16807x
All transpose tests passed.
```

| Shape | Naive ms | Naive GB/s | Tiled ms | Tiled GB/s | Speedup | Correctness |
|---|---:|---:|---:|---:|---:|---|
| 1024 x 1024 | 0.0446 | 188.116 | 0.0078 | 1070.85 | 5.6925x | PASS |
| 2048 x 2048 | 0.1276 | 262.902 | 0.0121 | 2774.52 | 10.5535x | PASS |
| 4096 x 4096 | 0.5047 | 265.911 | 0.2356 | 569.709 | 2.1425x | PASS |
| 4096 x 3072 | 0.3839 | 262.227 | 0.1771 | 568.526 | 2.1681x | PASS |

## Result Analysis

- Correctness passed for all tested shapes.
- The tiled kernel is faster for every tested shape.
- The largest speedup appears at `2048 x 2048`, where tiled transpose reaches `10.5535x` over the naive kernel.
- For larger shapes such as `4096 x 4096`, the tiled kernel is still faster, but speedup drops to around `2.14x`. This should be investigated with profiler metrics such as memory throughput, occupancy, and stall reasons.
- The result confirms the main transpose optimization idea: using shared-memory tiling improves global memory coalescing for the write path.

## Nsight Compute Metrics to Collect

Profiler commands used:

```bash
PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:transpose_naive_kernel \
  --force-overwrite -o reports/ncu/transpose_naive \
  ./build/bench_transpose_benchmark

PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:transpose_tiled_kernel \
  --force-overwrite -o reports/ncu/transpose_tiled \
  ./build/bench_transpose_benchmark
```

Generated local reports:

```text
reports/ncu/transpose_naive.ncu-rep
reports/ncu/transpose_tiled.ncu-rep
```

## Profiler Evidence

Nsight Compute was run with `PROFILE_MODE=1`, which profiles a representative `2048 x 2048` case with one benchmark iteration.

| Kernel | Duration | Memory Throughput | DRAM Throughput | Compute Throughput | Achieved Occupancy | Grid Size | Registers / Thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive transpose | 139.62 us | 135.64 GB/s | 18.86% | 4.09% | 76.15% | 16384 | 16 |
| Tiled shared-memory transpose | 27.07 us | 642.74 GB/s | 89.47% | 23.88% | 91.30% | 4096 | 24 |

### Profiler Notes

Nsight Compute reported the following warning for the naive kernel:

```text
The memory access pattern for global stores to L1TEX might not be optimal.
On average, only 4.0 of the 32 bytes transmitted per sector are utilized by each thread.
```

It also reported:

```text
This workload has uncoalesced global accesses resulting in 3,670,016 excessive sectors.
```

### Interpretation

- The naive kernel is limited by uncoalesced global stores.
- The tiled kernel improves memory throughput from `135.64 GB/s` to `642.74 GB/s`.
- DRAM throughput rises from `18.86%` to `89.47%`, confirming that tiling makes much better use of memory bandwidth.
- The tiled kernel also improves achieved occupancy from `76.15%` to `91.30%`.
- This profiler evidence directly supports the case-study explanation: shared-memory tiling converts a strided global write pattern into a coalesced write pattern.

## Takeaways

1. Naive transpose is limited by strided global memory stores.
2. Tiling through shared memory changes the memory access pattern so both reads and writes can be coalesced.
3. Padding shared memory tiles is a simple but important technique to reduce bank conflicts.
