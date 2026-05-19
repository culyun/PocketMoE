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

__global__ void bf16_row_to_float_kernel(const uint16_t* matrix, float* y, int row, int cols) {
    const uint16_t* src = matrix + static_cast<size_t>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = bf16_to_float(src[c]);
}

__global__ void bf16_matvec_kernel(const float* x, const uint16_t* w, float* y, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += bf16_to_float(w[static_cast<size_t>(row) * cols + c]) * x[c];
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

bool bf16_row_to_float_cuda(const uint16_t* d_matrix_bf16, float* d_y, int row, int cols, void* stream) {
    if (d_matrix_bf16 == nullptr || d_y == nullptr || row < 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_row_to_float_kernel<<<1, 256, 0, cuda_stream>>>(d_matrix_bf16, d_y, row, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_matvec_cuda(const float* d_x, const uint16_t* d_w_bf16, float* d_y, int rows, int cols, void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_matvec_kernel<<<rows, 256, 256 * sizeof(float), cuda_stream>>>(d_x, d_w_bf16, d_y, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
