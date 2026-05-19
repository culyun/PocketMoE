#include "cuda_ops.hpp"

#include <cuda_runtime.h>

namespace dsv4 {
namespace {

__device__ __forceinline__ float fp4_e2m1_value(uint8_t code) {
    constexpr float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return table[code & 0x0f];
}

__device__ __forceinline__ float e8m0_value(uint8_t code) {
    return exp2f(static_cast<float>(static_cast<int>(code) - 127));
}

__global__ void fp4_e2m1_e8m0_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int packed_cols,
    int scale_cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const uint8_t packed = weight[static_cast<size_t>(row) * packed_cols + col / 2];
        const uint8_t code = (col & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0f);
        const float s = e8m0_value(scale[static_cast<size_t>(row) * scale_cols + col / 32]);
        sum += fp4_e2m1_value(code) * s * x[col];
    }

    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) y[row] = scratch[0];
}

}  // namespace

bool fp4_e2m1_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream) {
    if (d_x == nullptr || d_weight == nullptr || d_scale == nullptr || d_y == nullptr) return false;
    if (rows <= 0 || cols <= 0 || (cols % 32) != 0) return false;
    const int threads = 256;
    const int packed_cols = cols / 2;
    const int scale_cols = cols / 32;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp4_e2m1_e8m0_matvec_kernel<<<rows, threads, threads * sizeof(float), cuda_stream>>>(
        d_x, d_weight, d_scale, d_y, rows, cols, packed_cols, scale_cols);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
