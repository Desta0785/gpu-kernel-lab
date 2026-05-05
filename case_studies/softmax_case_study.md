# Row-wise Softmax Optimization Case Study

## Problem

Compute row-wise softmax for a matrix of shape `[rows, cols]`.

For each row:

```text
output[i] = exp(input[i] - row_max) / sum(exp(input[j] - row_max))
```

Subtracting `row_max` improves numerical stability and avoids overflow.

## Correctness

Reference implementation:

- CPU row-wise softmax
- double-precision accumulation

Checks:

- maximum absolute error <= `1e-4`
- every output row sums to approximately `1.0`
- no `NaN` or `Inf`

Command:

```bash
cmake -S . -B build
cmake --build build --target bench_softmax_benchmark -j
./build/bench_softmax_benchmark
```

## Baseline

One CUDA block handles one row.

The baseline uses shared memory reductions:

1. reduce row maximum
2. reduce sum of `exp(x - row_max)`
3. write normalized output

Expected bottlenecks:

- multiple block-wide `__syncthreads()`
- shared memory traffic
- repeated `expf` computation

## Optimization #1: Warp-level reduction

One warp handles one row.

Instead of shared memory reduction, the kernel uses:

```cpp
__shfl_down_sync(...)
```

Why it can be faster:

- removes block-wide synchronization
- reduces shared memory traffic
- maps well to moderate row sizes such as 256, 512, 1024, and 2048 columns

## Optimization #2: Vectorized `float4` load/store

The second optimized kernel keeps the one-warp-per-row mapping but processes four contiguous floats at a time.

Requirement:

```text
cols % 4 == 0
```

Why it can be faster:

- fewer loop iterations
- fewer memory instructions
- better memory instruction efficiency when data is aligned

## Results

Measured on local GPU environment:

```text
Row-wise softmax benchmark and correctness
tolerance: max_abs_error <= 1e-4, max_row_sum_error <= 1e-4

shape rows=1 cols=16 elements=16
  baseline       0.0122 ms    0.0210 GB/s  max_err=0.0000  row_sum_err=0.0000
  warp           0.0082 ms    0.0313 GB/s  speedup=1.4951x  max_err=0.0000  row_sum_err=0.0000
  warp_float4    0.0069 ms    0.0371 GB/s  speedup=1.7708x  max_err=0.0000  row_sum_err=0.0000
  correctness: PASS

shape rows=4 cols=128 elements=512
  baseline       0.0075 ms    1.0870 GB/s  max_err=0.0000  row_sum_err=0.0000
  warp           0.0070 ms    1.1713 GB/s  speedup=1.0776x  max_err=0.0000  row_sum_err=0.0000
  warp_float4    0.0061 ms    1.3404 GB/s  speedup=1.2332x  max_err=0.0000  row_sum_err=0.0000
  correctness: PASS

shape rows=128 cols=256 elements=32768
  baseline       0.0060 ms   86.7797 GB/s  max_err=0.0000  row_sum_err=0.0000
  warp           0.0066 ms   78.8906 GB/s  speedup=0.9091x  max_err=0.0000  row_sum_err=0.0000
  warp_float4    0.0056 ms   92.9483 GB/s  speedup=1.0711x  max_err=0.0000  row_sum_err=0.0000
  correctness: PASS

shape rows=1024 cols=512 elements=524288
  baseline       0.0090 ms  931.9682 GB/s  max_err=0.0000  row_sum_err=0.0000
  warp           0.0072 ms  1166.0172 GB/s  speedup=1.2511x  max_err=0.0000  row_sum_err=0.0000
  warp_float4    0.0217 ms  387.1112 GB/s  speedup=0.4154x  max_err=0.0000  row_sum_err=0.0000
  correctness: PASS

shape rows=4096 cols=1024 elements=4194304
  baseline       0.0418 ms  1604.1858 GB/s  max_err=0.0000  row_sum_err=0.0000
  warp           0.0537 ms  1250.6871 GB/s  speedup=0.7796x  max_err=0.0000  row_sum_err=0.0000
  warp_float4    0.0290 ms  2315.7596 GB/s  speedup=1.4436x  max_err=0.0000  row_sum_err=0.0000
  correctness: PASS

shape rows=4096 cols=2048 elements=8388608
  baseline       0.0496 ms  2706.2354 GB/s  max_err=0.0000  row_sum_err=0.0000
  warp           0.0689 ms  1948.5432 GB/s  speedup=0.7200x  max_err=0.0000  row_sum_err=0.0000
  warp_float4    0.0562 ms  2387.4681 GB/s  speedup=0.8822x  max_err=0.0000  row_sum_err=0.0000
  correctness: PASS

All softmax tests passed.
```

| Shape | Baseline ms | Warp ms | Warp speedup | Float4 ms | Float4 speedup |
|---|---:|---:|---:|---:|---:|
| 1 x 16 | 0.0122 | 0.0082 | 1.4951x | 0.0069 | 1.7708x |
| 4 x 128 | 0.0075 | 0.0070 | 1.0776x | 0.0061 | 1.2332x |
| 128 x 256 | 0.0060 | 0.0066 | 0.9091x | 0.0056 | 1.0711x |
| 1024 x 512 | 0.0090 | 0.0072 | 1.2511x | 0.0217 | 0.4154x |
| 4096 x 1024 | 0.0418 | 0.0537 | 0.7796x | 0.0290 | 1.4436x |
| 4096 x 2048 | 0.0496 | 0.0689 | 0.7200x | 0.0562 | 0.8822x |

## Result Analysis

- Correctness passed for all tested shapes.
- Warp-level reduction is not universally faster. It improves some smaller and medium shapes, but it is slower for larger rows such as `4096 x 1024` and `4096 x 2048`.
- The `float4` version gives the best result for `4096 x 1024`, reaching `1.4436x` speedup over baseline.
- The `float4` version is slower for `1024 x 512` and `4096 x 2048`, which suggests that vectorization alone is not enough; occupancy, instruction mix, register pressure, and memory behavior should be checked with Nsight Compute.
- The baseline is already strong for large rows because one block per row provides more threads per row than the one-warp-per-row version.

## Nsight Compute Metrics to Collect

Profiler commands used:

```bash
PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:softmax_baseline_kernel \
  --force-overwrite -o reports/ncu/softmax_baseline \
  ./build/bench_softmax_benchmark

PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:softmax_warp_kernel \
  --force-overwrite -o reports/ncu/softmax_warp \
  ./build/bench_softmax_benchmark

PROFILE_MODE=1 ncu --set full --target-processes all --launch-count 1 \
  --kernel-name regex:softmax_warp_float4_kernel \
  --force-overwrite -o reports/ncu/softmax_float4 \
  ./build/bench_softmax_benchmark
```

Generated local reports:

```text
reports/ncu/softmax_baseline.ncu-rep
reports/ncu/softmax_warp.ncu-rep
reports/ncu/softmax_float4.ncu-rep
```

## Profiler Evidence

Nsight Compute was run with `PROFILE_MODE=1`, which profiles a representative `4096 x 1024` case with one benchmark iteration.

| Kernel | Duration | Memory Throughput | Compute Throughput | Achieved Occupancy | Grid Size | Registers / Thread | Dynamic Shared Memory / Block |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline block-level softmax | 38.62 us | 457.94 GB/s | 65.62% | 92.72% | 4096 | 22 | 1.02 KB |
| Warp-level softmax | 44.13 us | 381.34 GB/s | 17.83% | 93.95% | 512 | 38 | 0 B |
| Warp + `float4` softmax | 41.89 us | 428.78 GB/s | 15.85% | 93.09% | 512 | 38 | 0 B |

### Interpretation

- The warp-level kernels remove dynamic shared-memory usage for row reductions, but they also use more registers per thread: `22` to `38`.
- The baseline block-level version has higher compute throughput for this shape because one row is processed by a full block instead of a single warp.
- The `float4` version improves over the plain warp version in this profile: `44.13 us` to `41.89 us`, and `381.34 GB/s` to `428.78 GB/s`.
- The profiler evidence supports the benchmark observation that softmax optimization is shape-dependent. Warp-level reductions reduce synchronization and shared-memory usage, but they do not always outperform a strong block-level baseline for larger rows.

## Takeaways

1. Softmax requires both reductions and elementwise math, so reduction overhead matters.
2. Warp-level reductions can reduce synchronization overhead for moderate row sizes.
3. `float4` vectorization is useful when alignment and shape constraints are satisfied.
