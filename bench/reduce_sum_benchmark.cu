#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << "\n";
        std::exit(1);
    }
}

double sum_cpu_ref(const float* input, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(input[i]);
    }
    return sum;
}

// Baseline:
// Each block reduces 2 * blockDim.x contiguous elements and writes one partial sum.
// The host copies partial sums back and finishes the final accumulation on CPU.
__global__ void reduce_sum_baseline_stage1_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_sums,
    int n)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 2;
    int i0 = base + tid;
    int i1 = base + tid + blockDim.x;

    float value = 0.0f;
    if (i0 < n) value += input[i0];
    if (i1 < n) value += input[i1];

    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

// Optimized:
// Each block processes a grid-stride chunk, uses warp shuffle reduction inside
// each warp, and uses shared memory only to combine warp-level partial sums.
__global__ void reduce_sum_grid_stride_stage1_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_sums,
    int n)
{
    extern __shared__ float warp_sums[];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    float value = 0.0f;
    int grid_stride = gridDim.x * blockDim.x * 2;

    for (int i = blockIdx.x * blockDim.x * 2 + tid; i < n; i += grid_stride) {
        value += input[i];
        int j = i + blockDim.x;
        if (j < n) {
            value += input[j];
        }
    }

    value = warp_reduce_sum(value);

    if (lane == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = block_sum;
    }
}

struct RunResult {
    double gpu_sum = 0.0;
    double abs_error = 0.0;
    double rel_error = 0.0;
    float ms = 0.0f;
    int grid = 0;
};

template <typename KernelLauncher>
RunResult run_reduce_case(
    KernelLauncher launch_kernel,
    const std::vector<float>& h_input,
    double cpu_sum,
    float* d_input,
    float* d_partial,
    int n,
    int grid,
    int iterations)
{
    launch_kernel(d_input, d_partial, n, grid);
    cuda_check(cudaGetLastError(), "warmup kernel launch");
    cuda_check(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
        launch_kernel(d_input, d_partial, n, grid);
    }
    cuda_check(cudaGetLastError(), "benchmark kernel launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float total_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::vector<float> h_partial(grid);
    cuda_check(cudaMemcpy(
                   h_partial.data(),
                   d_partial,
                   static_cast<size_t>(grid) * sizeof(float),
                   cudaMemcpyDeviceToHost),
               "cudaMemcpy partial D2H");

    RunResult result;
    result.grid = grid;
    result.ms = total_ms / static_cast<float>(iterations);
    for (float partial : h_partial) {
        result.gpu_sum += static_cast<double>(partial);
    }

    result.abs_error = std::abs(result.gpu_sum - cpu_sum);
    result.rel_error = result.abs_error / (std::abs(cpu_sum) + 1e-12);
    return result;
}

bool passed(const RunResult& result) {
    return result.rel_error <= 1e-5 || result.abs_error <= 1e-3;
}

double bandwidth_gbps(int n, float ms) {
    double gb = static_cast<double>(n) * sizeof(float) / 1e9;
    return gb / (static_cast<double>(ms) / 1000.0);
}

void run_case(int n) {
    constexpr int kBlockSize = 256;
    constexpr int kMaxOptimizedGrid = 1024;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_input(n);
    for (float& value : h_input) {
        value = dist(rng);
    }

    double cpu_sum = sum_cpu_ref(h_input.data(), n);

    float* d_input = nullptr;
    float* d_partial = nullptr;
    size_t input_bytes = static_cast<size_t>(n) * sizeof(float);

    int baseline_grid = (n + kBlockSize * 2 - 1) / (kBlockSize * 2);
    int optimized_grid = std::min(baseline_grid, kMaxOptimizedGrid);
    int max_partial_count = std::max(baseline_grid, optimized_grid);

    cuda_check(cudaMalloc(&d_input, input_bytes), "cudaMalloc(d_input)");
    cuda_check(cudaMalloc(&d_partial, static_cast<size_t>(max_partial_count) * sizeof(float)),
               "cudaMalloc(d_partial)");
    cuda_check(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy input H2D");

    const bool profile_mode = std::getenv("PROFILE_MODE") != nullptr;
    int iterations = profile_mode ? 1 : ((n < (1 << 20)) ? 100 : 30);

    auto launch_baseline = [](const float* input, float* partial, int n_value, int grid) {
        size_t shared_bytes = kBlockSize * sizeof(float);
        reduce_sum_baseline_stage1_kernel<<<grid, kBlockSize, shared_bytes>>>(
            input, partial, n_value);
    };

    auto launch_optimized = [](const float* input, float* partial, int n_value, int grid) {
        int num_warps = (kBlockSize + 31) / 32;
        size_t shared_bytes = static_cast<size_t>(num_warps) * sizeof(float);
        reduce_sum_grid_stride_stage1_kernel<<<grid, kBlockSize, shared_bytes>>>(
            input, partial, n_value);
    };

    RunResult baseline = run_reduce_case(
        launch_baseline, h_input, cpu_sum, d_input, d_partial, n, baseline_grid, iterations);
    RunResult optimized = run_reduce_case(
        launch_optimized, h_input, cpu_sum, d_input, d_partial, n, optimized_grid, iterations);

    bool all_passed = passed(baseline) && passed(optimized);

    std::cout << "N=" << n << "\n";
    std::cout << "  baseline   grid=" << std::setw(6) << baseline.grid
              << "  time_ms=" << std::setw(8) << baseline.ms
              << "  GB/s=" << std::setw(8) << bandwidth_gbps(n, baseline.ms)
              << "  abs_err=" << baseline.abs_error
              << "  rel_err=" << baseline.rel_error << "\n";
    std::cout << "  optimized  grid=" << std::setw(6) << optimized.grid
              << "  time_ms=" << std::setw(8) << optimized.ms
              << "  GB/s=" << std::setw(8) << bandwidth_gbps(n, optimized.ms)
              << "  speedup=" << baseline.ms / optimized.ms << "x"
              << "  abs_err=" << optimized.abs_error
              << "  rel_err=" << optimized.rel_error << "\n";
    std::cout << "  correctness: " << (all_passed ? "PASS" : "FAIL") << "\n\n";

    cuda_check(cudaFree(d_input), "cudaFree(d_input)");
    cuda_check(cudaFree(d_partial), "cudaFree(d_partial)");

    if (!all_passed) {
        std::exit(1);
    }
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Reduce sum benchmark and correctness\n\n";

    const bool profile_mode = std::getenv("PROFILE_MODE") != nullptr;
    const std::vector<int> test_sizes = profile_mode ? std::vector<int>{1 << 24} : std::vector<int>{
        1,
        17,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        1 << 20,
        1 << 24,
        1 << 26,
    };

    for (int n : test_sizes) {
        run_case(n);
    }

    std::cout << "All reduce sum tests passed.\n";
    return 0;
}
