#include <cuda_runtime.h>
#include <iostream>


static void check_cuda (cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ":" << cudaGetErrorString(e) << "/n";
        std::exit(1);
    }

}

// stage-1 reduction: each block reduces its assigned chunk of data and writes the partial sum to out_partial[blockIdx.x].
__global__ void reduction_sum_stage1 (const float* __restrict__ in, float* __restrict__ out_partial, int N) {

}