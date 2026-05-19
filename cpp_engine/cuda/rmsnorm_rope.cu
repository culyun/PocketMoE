#include "cuda_ops.hpp"

#include <cuda_runtime.h>

namespace dsv4 {
namespace {

__device__ __forceinline__ float bf16_to_float(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    float out;
    __builtin_memcpy(&out, &value, sizeof(out));
    return out;
}

__global__ void rmsnorm_bf16_gamma_kernel(
    const float* __restrict__ x,
    const uint16_t* __restrict__ gamma,
    float* __restrict__ y,
    int cols,
    float eps) {
    extern __shared__ float scratch[];
    const int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float v = x[c];
        sum_sq += v * v;
    }
    scratch[tid] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(cols) + eps);

    for (int c = tid; c < cols; c += blockDim.x) {
        y[c] = x[c] * inv * bf16_to_float(gamma[c]);
    }
}

}  // namespace

bool rmsnorm_bf16_gamma_cuda(
    const float* d_x,
    const uint16_t* d_gamma_bf16,
    float* d_y,
    int cols,
    float eps,
    void* stream) {
    if (d_x == nullptr || d_gamma_bf16 == nullptr || d_y == nullptr) return false;
    if (cols <= 0) return false;
    const int threads = 256;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    rmsnorm_bf16_gamma_kernel<<<1, threads, threads * sizeof(float), cuda_stream>>>(d_x, d_gamma_bf16, d_y, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
