#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

__global__ void vectorAdd (const float* A ,const float* B, float* C, int vectorlength) {
    int workIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (workIndex < vectorlength) {
         C[workIndex] = A[workIndex] + B[workIndex];
    }
}

static void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ":" << cudaGetErrorString(e) << "\n"; 
        std::exit(1);
    }
}

int main() {
    const int n = 1 << 24;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cuda_check(cudaMalloc(&d_a, bytes), "cudaMalloc(d_a)");
    cuda_check(cudaMalloc(&d_b, bytes), "cudaMalloc(d_b)");
    cuda_check(cudaMalloc(&d_c, bytes), "cudaMalloc(d_c)");

    cuda_check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(h_a -> d_a)");
    cuda_check(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(h_b -> d_b)");

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t start{};
    cudaEvent_t end{}; 
    
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(&start)");
    cuda_check(cudaEventCreate(&end), "cudaEventCreate(&end)");

    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");

    vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaEventRecord(end), "cudaEventRecord(end)");
    cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");

    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, end), "cudaEventElapsedTime");
    cuda_check(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(d_c -> h_c)");

    std::cout << "vecAdd time(ms): " << ms << "\n";
    std::cout << "h_c[0] : " << h_c[0] << "(expect 3)\n";

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}