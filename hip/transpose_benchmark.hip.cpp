#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

void hip_check(hipError_t error, const char* message) {
    if (error != hipSuccess) {
        std::cerr << message << ": " << hipGetErrorString(error) << "\n";
        std::exit(1);
    }
}

void transpose_cpu_ref(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows,
    int cols)
{
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            output[col * rows + row] = input[row * cols + col];
        }
    }
}

__global__ void transpose_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

__global__ void transpose_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    __shared__ float tile[kTileDim][kTileDim + 1];

    int input_col = blockIdx.x * kTileDim + threadIdx.x;
    int input_row = blockIdx.y * kTileDim + threadIdx.y;

    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row = input_row + j;
        if (row < rows && input_col < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * cols + input_col];
        }
    }

    __syncthreads();

    int output_col = blockIdx.y * kTileDim + threadIdx.x;
    int output_row = blockIdx.x * kTileDim + threadIdx.y;

    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row = output_row + j;
        if (row < cols && output_col < rows) {
            output[row * rows + output_col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

float benchmark_kernel(
    const std::string& name,
    const float* d_input,
    float* d_output,
    int rows,
    int cols,
    int iterations)
{
    hipEvent_t start{};
    hipEvent_t stop{};
    hip_check(hipEventCreate(&start), "hipEventCreate(start)");
    hip_check(hipEventCreate(&stop), "hipEventCreate(stop)");

    const dim3 tiled_block(kTileDim, kBlockRows);
    const dim3 tiled_grid(
        (cols + kTileDim - 1) / kTileDim,
        (rows + kTileDim - 1) / kTileDim);

    const dim3 naive_block(kTileDim, kBlockRows);
    const dim3 naive_grid(
        (cols + naive_block.x - 1) / naive_block.x,
        (rows + naive_block.y - 1) / naive_block.y);

    if (name == "naive") {
        hipLaunchKernelGGL(
            transpose_naive_kernel,
            naive_grid,
            naive_block,
            0,
            0,
            d_input,
            d_output,
            rows,
            cols);
    } else {
        hipLaunchKernelGGL(
            transpose_tiled_kernel,
            tiled_grid,
            tiled_block,
            0,
            0,
            d_input,
            d_output,
            rows,
            cols);
    }
    hip_check(hipGetLastError(), "warmup kernel launch");
    hip_check(hipDeviceSynchronize(), "warmup hipDeviceSynchronize");

    hip_check(hipEventRecord(start), "hipEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
        if (name == "naive") {
            hipLaunchKernelGGL(
                transpose_naive_kernel,
                naive_grid,
                naive_block,
                0,
                0,
                d_input,
                d_output,
                rows,
                cols);
        } else {
            hipLaunchKernelGGL(
                transpose_tiled_kernel,
                tiled_grid,
                tiled_block,
                0,
                0,
                d_input,
                d_output,
                rows,
                cols);
        }
    }
    hip_check(hipGetLastError(), "benchmark kernel launch");
    hip_check(hipEventRecord(stop), "hipEventRecord(stop)");
    hip_check(hipEventSynchronize(stop), "hipEventSynchronize(stop)");

    float total_ms = 0.0f;
    hip_check(hipEventElapsedTime(&total_ms, start, stop), "hipEventElapsedTime");

    hip_check(hipEventDestroy(start), "hipEventDestroy(start)");
    hip_check(hipEventDestroy(stop), "hipEventDestroy(stop)");

    return total_ms / static_cast<float>(iterations);
}

bool check_correctness(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance = 1e-5f)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::abs(expected[i] - actual[i]);
        if (diff > tolerance) {
            std::cerr << "Correctness failed at index " << i
                      << ": expected=" << expected[i]
                      << " actual=" << actual[i]
                      << " diff=" << diff << "\n";
            return false;
        }
    }
    return true;
}

void run_case(int rows, int cols) {
    const size_t num_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t num_bytes = num_elements * sizeof(float);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_input(num_elements);
    std::vector<float> h_expected(num_elements);
    std::vector<float> h_actual(num_elements);

    for (float& value : h_input) {
        value = dist(rng);
    }

    transpose_cpu_ref(h_input, h_expected, rows, cols);

    float* d_input = nullptr;
    float* d_output = nullptr;
    hip_check(hipMalloc(&d_input, num_bytes), "hipMalloc(d_input)");
    hip_check(hipMalloc(&d_output, num_bytes), "hipMalloc(d_output)");
    hip_check(
        hipMemcpy(d_input, h_input.data(), num_bytes, hipMemcpyHostToDevice),
        "hipMemcpy(input H2D)");

    const int iterations = 100;

    const float naive_ms = benchmark_kernel("naive", d_input, d_output, rows, cols, iterations);
    hip_check(
        hipMemcpy(h_actual.data(), d_output, num_bytes, hipMemcpyDeviceToHost),
        "hipMemcpy(naive D2H)");
    if (!check_correctness(h_expected, h_actual)) {
        std::exit(1);
    }

    const float tiled_ms = benchmark_kernel("tiled", d_input, d_output, rows, cols, iterations);
    hip_check(
        hipMemcpy(h_actual.data(), d_output, num_bytes, hipMemcpyDeviceToHost),
        "hipMemcpy(tiled D2H)");
    if (!check_correctness(h_expected, h_actual)) {
        std::exit(1);
    }

    const double gb = 2.0 * static_cast<double>(num_bytes) / 1e9;
    const double naive_gbps = gb / (static_cast<double>(naive_ms) / 1000.0);
    const double tiled_gbps = gb / (static_cast<double>(tiled_ms) / 1000.0);
    const double speedup = static_cast<double>(naive_ms) / static_cast<double>(tiled_ms);

    std::cout << "rows=" << rows
              << " cols=" << cols
              << " naive_ms=" << naive_ms
              << " naive_GBps=" << naive_gbps
              << " tiled_ms=" << tiled_ms
              << " tiled_GBps=" << tiled_gbps
              << " speedup=" << speedup << "x\n";

    hip_check(hipFree(d_input), "hipFree(d_input)");
    hip_check(hipFree(d_output), "hipFree(d_output)");
}

} // namespace

int main() {
    std::vector<std::pair<int, int>> test_cases = {
        {1024, 1024},
        {2048, 2048},
        {4096, 4096},
        {4096, 3072},
    };

    for (const auto& [rows, cols] : test_cases) {
        run_case(rows, cols);
    }

    std::cout << "All HIP transpose tests passed.\n";
    return 0;
}
