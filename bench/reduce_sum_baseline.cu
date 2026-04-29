#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>



static void cuda_check (cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ":" << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }

}

// CPU reference: use double for accuracy
double sum_cpu_ref(const float* x, int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += (double)x[i];
    return s;
}

// stage-1 reduction: each block reduces its assigned chunk of data and writes the partial sum to out_partial[blockIdx.x].
__global__ void reduction_sum_stage1 (const float* __restrict__ in, float* __restrict__ out_partial, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int base = blockIdx.x * (blockDim.x * 2);
    int i0 = tid + base;
    int i1 = tid + base + blockDim.x;

    float v = 0.0f;
    if (i0 < n) v += in[i0];
    if (i1 < n) v += in[i1];

    sdata[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out_partial[blockIdx.x] = sdata[0];
}

int main() {
    // Try multiple sizes including non-multiples of block size
    std::vector<int> testNs = {1, 17, 255, 256, 257, 1023, 1024, 1025, 1<<20};

    const int block = 256; // power-of-two
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int N : testNs) {
        std::vector<float> hx(N);
        for (auto& v : hx) v = dist(rng);

        double cpu = sum_cpu_ref(hx.data(), N);

        // allocate device input
        float *d_in = nullptr, *d_partial = nullptr;
        cuda_check(cudaMalloc(&d_in, (size_t)N * sizeof(float)), "cudaMalloc d_in");
        cuda_check(cudaMemcpy(d_in, hx.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "H2D copy");

        // grid: each block reduces 2*block elements
        int grid = (N + (block * 2 - 1)) / (block * 2);
        cuda_check(cudaMalloc(&d_partial, (size_t)grid * sizeof(float)), "cudaMalloc d_partial");

        // optional timing
        cudaEvent_t start, stop;
        cuda_check(cudaEventCreate(&start), "event create start");
        cuda_check(cudaEventCreate(&stop), "event create stop");

        size_t smem_bytes = (size_t)block * sizeof(float);

        cuda_check(cudaEventRecord(start), "event record start");
        reduction_sum_stage1<<<grid, block, smem_bytes>>>(d_in, d_partial, N);
        cuda_check(cudaGetLastError(), "kernel launch");
        cuda_check(cudaEventRecord(stop), "event record stop");
        cuda_check(cudaEventSynchronize(stop), "event sync stop");

        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

        // copy partial sums back and finalize on CPU
        std::vector<float> hpartial(grid);
        cuda_check(cudaMemcpy(hpartial.data(), d_partial, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost), "D2H partial");

        double gpu = 0.0;
        for (float p : hpartial) gpu += (double)p;

        double abs_err = std::abs(gpu - cpu);
        double rel_err = abs_err / (std::abs(cpu) + 1e-12);

        std::cout << "N=" << N
                  << " grid=" << grid
                  << " cpu=" << cpu
                  << " gpu=" << gpu
                  << " abs_err=" << abs_err
                  << " rel_err=" << rel_err
                  << " time_ms=" << ms
                  << "\n";

        // tolerance: float reduction order differs; allow small error
        if (rel_err > 1e-5 && abs_err > 1e-3) {
            std::cerr << "FAILED correctness at N=" << N << "\n";
            return 1;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_in);
        cudaFree(d_partial);
    }

    std::cout << "All tests passed.\n";
    return 0;
}
