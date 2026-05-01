#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

void cuda_check(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << "\n";
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
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    const dim3 block(kTileDim, kBlockRows);
    const dim3 grid(
        (cols + kTileDim - 1) / kTileDim,
        (rows + kTileDim - 1) / kTileDim);

    if (name == "naive") {
        transpose_naive_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    } else {
        transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    }
    cuda_check(cudaGetLastError(), "warmup kernel launch");
    cuda_check(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize");

    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
        if (name == "naive") {
            transpose_naive_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
        } else {
            transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
        }
    }
    cuda_check(cudaGetLastError(), "benchmark kernel launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float total_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");

    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

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
    cuda_check(cudaMalloc(&d_input, num_bytes), "cudaMalloc(d_input)");
    cuda_check(cudaMalloc(&d_output, num_bytes), "cudaMalloc(d_output)");
    cuda_check(cudaMemcpy(d_input, h_input.data(), num_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(input H2D)");

    const int iterations = 100;

    const float naive_ms = benchmark_kernel("naive", d_input, d_output, rows, cols, iterations);
    cuda_check(cudaMemcpy(h_actual.data(), d_output, num_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(naive D2H)");
    if (!check_correctness(h_expected, h_actual)) {
        std::exit(1);
    }

    const float tiled_ms = benchmark_kernel("tiled", d_input, d_output, rows, cols, iterations);
    cuda_check(cudaMemcpy(h_actual.data(), d_output, num_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(tiled D2H)");
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

    cuda_check(cudaFree(d_input), "cudaFree(d_input)");
    cuda_check(cudaFree(d_output), "cudaFree(d_output)");
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

    std::cout << "All transpose tests passed.\n";
    return 0;
}
