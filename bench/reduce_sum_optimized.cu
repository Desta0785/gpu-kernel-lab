#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <algorithm>



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

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// stage-1 reduction: each block accumulates a grid-stride chunk of data and writes one partial sum.
__global__ void reduce_sum_grid_stride_stage1_kernel (const float* __restrict__ in, float* __restrict__ out_partial, int n) {
    extern __shared__ float warp_sums[];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    float v = 0.0f;

    int grid_stride = gridDim.x * blockDim.x * 2;

    for (int i = blockIdx.x * blockDim.x * 2 + tid; i < n; i += grid_stride) {
        v += in[i];
        
        int j = i + blockDim.x;
        if (j < n) {
            v += in[j];
        }
    }


    v = warp_reduce_sum(v);

    if (lane == 0) {
        warp_sums[warp_id] = v;
    }

    __syncthreads();

    float block_sum = 0.0f;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (warp_id == 0) {
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    if (tid == 0) out_partial[blockIdx.x] = block_sum;
}

int main() {
    // Try multiple sizes including non-multiples of block size
    std::vector<int> testNs = {
        1, 17, 255, 256, 257, 1023, 1024, 1025,
        1 << 20,
        1 << 24,
        1 << 26
    };

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
        const int max_grid = 1024;
        int grid = std::min((N + block * 2 - 1) / (block * 2), max_grid);
        cuda_check(cudaMalloc(&d_partial, (size_t)grid * sizeof(float)), "cudaMalloc d_partial");

        // optional timing
        cudaEvent_t start, stop;
        cuda_check(cudaEventCreate(&start), "event create start");
        cuda_check(cudaEventCreate(&stop), "event create stop");

        int num_warps = (block + 31) / 32;
        size_t smem_bytes = (size_t)num_warps * sizeof(float);

        cuda_check(cudaEventRecord(start), "event record start");
        reduce_sum_grid_stride_stage1_kernel<<<grid, block, smem_bytes>>>(d_in, d_partial, N);
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
        double gb = static_cast<double>(N) * sizeof(float) / 1e9;
        double bandwidth_gbps = gb / (ms / 1000.0);

        std::cout << "N=" << N
                  << " grid=" << grid
                  << " cpu=" << cpu
                  << " gpu=" << gpu
                  << " abs_err=" << abs_err
                  << " rel_err=" << rel_err
                  << " time_ms=" << ms
                  << " bandwidth_GBps=" << bandwidth_gbps
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
