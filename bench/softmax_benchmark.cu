#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

static void cuda_check(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << "\n";
        std::exit(1);
    }
}

__device__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

// Day 6 baseline:
// One CUDA block handles one row. Threads in the block cooperate through shared
// memory reductions to compute row max and row sum.
__global__ void softmax_baseline_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_in[col]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = shared[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(row_in[col] - row_max);
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = shared[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        row_out[col] = expf(row_in[col] - row_max) / row_sum;
    }
}

// Day 7 optimization #1:
// One warp handles one row. Warp shuffle reductions avoid shared-memory traffic
// and block-wide __syncthreads().
__global__ void softmax_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float local_max = -FLT_MAX;
    for (int col = lane; col < cols; col += 32) {
        local_max = fmaxf(local_max, row_in[col]);
    }

    float row_max = warp_reduce_max(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    float local_sum = 0.0f;
    for (int col = lane; col < cols; col += 32) {
        local_sum += expf(row_in[col] - row_max);
    }

    float row_sum = warp_reduce_sum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    for (int col = lane; col < cols; col += 32) {
        row_out[col] = expf(row_in[col] - row_max) / row_sum;
    }
}

// Day 7 optimization #2:
// Same one-warp-per-row design, but each memory instruction processes float4.
// This version requires cols % 4 == 0.
__global__ void softmax_warp_float4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    const float4* row_in4 = reinterpret_cast<const float4*>(row_in);
    float4* row_out4 = reinterpret_cast<float4*>(row_out);
    int cols4 = cols / 4;

    float local_max = -FLT_MAX;
    for (int i = lane; i < cols4; i += 32) {
        float4 v = row_in4[i];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }

    float row_max = warp_reduce_max(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    float local_sum = 0.0f;
    for (int i = lane; i < cols4; i += 32) {
        float4 v = row_in4[i];
        local_sum += expf(v.x - row_max);
        local_sum += expf(v.y - row_max);
        local_sum += expf(v.z - row_max);
        local_sum += expf(v.w - row_max);
    }

    float row_sum = warp_reduce_sum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    for (int i = lane; i < cols4; i += 32) {
        float4 v = row_in4[i];
        float4 out;
        out.x = expf(v.x - row_max) / row_sum;
        out.y = expf(v.y - row_max) / row_sum;
        out.z = expf(v.z - row_max) / row_sum;
        out.w = expf(v.w - row_max) / row_sum;
        row_out4[i] = out;
    }
}

static void softmax_cpu_ref(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; ++row) {
        double row_max = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(row_max, static_cast<double>(input[row * cols + col]));
        }

        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            row_sum += std::exp(static_cast<double>(input[row * cols + col]) - row_max);
        }

        for (int col = 0; col < cols; ++col) {
            output[row * cols + col] = static_cast<float>(
                std::exp(static_cast<double>(input[row * cols + col]) - row_max) / row_sum);
        }
    }
}

struct CheckResult {
    float max_abs_error = 0.0f;
    double max_row_sum_error = 0.0;
    bool passed = true;
};

static CheckResult check_softmax(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    int rows,
    int cols)
{
    CheckResult result;

    for (int i = 0; i < rows * cols; ++i) {
        if (!std::isfinite(actual[i])) {
            result.passed = false;
        }
        result.max_abs_error = std::max(result.max_abs_error, std::abs(actual[i] - expected[i]));
    }

    for (int row = 0; row < rows; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            row_sum += static_cast<double>(actual[row * cols + col]);
        }
        result.max_row_sum_error = std::max(result.max_row_sum_error, std::abs(row_sum - 1.0));
    }

    if (result.max_abs_error > 1e-4f || result.max_row_sum_error > 1e-4) {
        result.passed = false;
    }

    return result;
}

static float time_baseline(float* d_output, const float* d_input, int rows, int cols, int iterations) {
    constexpr int kBlockSize = 256;
    size_t shared_bytes = kBlockSize * sizeof(float);

    softmax_baseline_kernel<<<rows, kBlockSize, shared_bytes>>>(d_input, d_output, rows, cols);
    cuda_check(cudaGetLastError(), "softmax_baseline warmup launch");
    cuda_check(cudaDeviceSynchronize(), "softmax_baseline warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");

    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        softmax_baseline_kernel<<<rows, kBlockSize, shared_bytes>>>(d_input, d_output, rows, cols);
    }
    cuda_check(cudaGetLastError(), "softmax_baseline launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");

    return ms / iterations;
}

static float time_warp(float* d_output, const float* d_input, int rows, int cols, int iterations) {
    constexpr int kThreads = 256;
    constexpr int kWarpsPerBlock = kThreads / 32;
    int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;

    softmax_warp_kernel<<<blocks, kThreads>>>(d_input, d_output, rows, cols);
    cuda_check(cudaGetLastError(), "softmax_warp warmup launch");
    cuda_check(cudaDeviceSynchronize(), "softmax_warp warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");

    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        softmax_warp_kernel<<<blocks, kThreads>>>(d_input, d_output, rows, cols);
    }
    cuda_check(cudaGetLastError(), "softmax_warp launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");

    return ms / iterations;
}

static float time_warp_float4(float* d_output, const float* d_input, int rows, int cols, int iterations) {
    constexpr int kThreads = 256;
    constexpr int kWarpsPerBlock = kThreads / 32;
    int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;

    softmax_warp_float4_kernel<<<blocks, kThreads>>>(d_input, d_output, rows, cols);
    cuda_check(cudaGetLastError(), "softmax_warp_float4 warmup launch");
    cuda_check(cudaDeviceSynchronize(), "softmax_warp_float4 warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");

    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        softmax_warp_float4_kernel<<<blocks, kThreads>>>(d_input, d_output, rows, cols);
    }
    cuda_check(cudaGetLastError(), "softmax_warp_float4 launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");

    return ms / iterations;
}

static double effective_gbps(int rows, int cols, float ms) {
    // Approximation: softmax reads input for max, reads input for sum, then reads
    // input and writes output. This is not a perfect roofline model, but it is a
    // useful first benchmark metric.
    double bytes = static_cast<double>(rows) * cols * sizeof(float) * 4.0;
    return bytes / (static_cast<double>(ms) / 1000.0) / 1e9;
}

int main() {
    const bool profile_mode = std::getenv("PROFILE_MODE") != nullptr;
    const std::vector<std::pair<int, int>> shapes = profile_mode ? std::vector<std::pair<int, int>>{
        {4096, 1024},
    } : std::vector<std::pair<int, int>>{
        {1, 16},
        {4, 128},
        {128, 256},
        {1024, 512},
        {4096, 1024},
        {4096, 2048},
    };

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Row-wise softmax benchmark and correctness\n";
    std::cout << "tolerance: max_abs_error <= 1e-4, max_row_sum_error <= 1e-4\n\n";

    for (auto [rows, cols] : shapes) {
        int n = rows * cols;
        size_t bytes = static_cast<size_t>(n) * sizeof(float);
        std::vector<float> h_input(n);
        std::vector<float> h_expected(n);
        std::vector<float> h_output(n);

        for (float& value : h_input) {
            value = dist(rng);
        }
        softmax_cpu_ref(h_input, h_expected, rows, cols);

        float* d_input = nullptr;
        float* d_output = nullptr;
        cuda_check(cudaMalloc(&d_input, bytes), "cudaMalloc d_input");
        cuda_check(cudaMalloc(&d_output, bytes), "cudaMalloc d_output");
        cuda_check(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

        int iterations = profile_mode ? 1 : ((n <= 1024 * 1024) ? 100 : 30);

        float baseline_ms = time_baseline(d_output, d_input, rows, cols, iterations);
        cuda_check(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "baseline D2H");
        CheckResult baseline_check = check_softmax(h_output, h_expected, rows, cols);

        float warp_ms = time_warp(d_output, d_input, rows, cols, iterations);
        cuda_check(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "warp D2H");
        CheckResult warp_check = check_softmax(h_output, h_expected, rows, cols);

        float float4_ms = -1.0f;
        CheckResult float4_check;
        bool run_float4 = (cols % 4 == 0);
        if (run_float4) {
            float4_ms = time_warp_float4(d_output, d_input, rows, cols, iterations);
            cuda_check(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "float4 D2H");
            float4_check = check_softmax(h_output, h_expected, rows, cols);
        }

        bool passed = baseline_check.passed && warp_check.passed && (!run_float4 || float4_check.passed);

        std::cout << "shape rows=" << rows << " cols=" << cols
                  << " elements=" << n << "\n";
        std::cout << "  baseline     " << std::setw(8) << baseline_ms << " ms"
                  << "  " << std::setw(8) << effective_gbps(rows, cols, baseline_ms) << " GB/s"
                  << "  max_err=" << baseline_check.max_abs_error
                  << "  row_sum_err=" << baseline_check.max_row_sum_error
                  << "\n";
        std::cout << "  warp         " << std::setw(8) << warp_ms << " ms"
                  << "  " << std::setw(8) << effective_gbps(rows, cols, warp_ms) << " GB/s"
                  << "  speedup=" << baseline_ms / warp_ms << "x"
                  << "  max_err=" << warp_check.max_abs_error
                  << "  row_sum_err=" << warp_check.max_row_sum_error
                  << "\n";
        if (run_float4) {
            std::cout << "  warp_float4  " << std::setw(8) << float4_ms << " ms"
                      << "  " << std::setw(8) << effective_gbps(rows, cols, float4_ms) << " GB/s"
                      << "  speedup=" << baseline_ms / float4_ms << "x"
                      << "  max_err=" << float4_check.max_abs_error
                      << "  row_sum_err=" << float4_check.max_row_sum_error
                      << "\n";
        }
        std::cout << "  correctness: " << (passed ? "PASS" : "FAIL") << "\n\n";

        cuda_check(cudaFree(d_input), "cudaFree d_input");
        cuda_check(cudaFree(d_output), "cudaFree d_output");

        if (!passed) {
            std::cerr << "Softmax correctness failed.\n";
            return 1;
        }
    }

    std::cout << "All softmax tests passed.\n";
    return 0;
}
