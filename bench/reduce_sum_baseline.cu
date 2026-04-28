#include <cuda_runtime.h>
#include <iostream>


static void check_cuda (cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ":" << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }

}

// stage-1 reduction: each block reduces its assigned chunk of data and writes the partial sum to out_partial[blockIdx.x].
__global__ void reduction_sum_stage1 (const float* __restrict__ in, float* __restrict__ out_partial, int N) {
    extern __shared__ float smem[]; // size = blockDim.x * sizeof(float)
    int tid = threadIdx.x;

    // each block handles 2*blockDim.x elements (classic baseline)
    int base = blockIdx.x * (blockDim.x * 2);
    int i0 = base + tid;
    int i1 = base + tid + blockDim.x;

    float v = 0.0f;
    if (i0 < N) v += in[i0];
    if (i1 < N) v += in[i1];

    smem[tid] = v;
    __syncthreads();

    // shared memory tree reduction
    // assume blockDim.x is power of 2
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }

    if (tid == 0) out_partial[blockIdx.x] = smem[0];

}