#include "cuda_ops.hpp"

#include <cuda_runtime.h>

namespace dsv4 {

bool cuda_runtime_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace dsv4
