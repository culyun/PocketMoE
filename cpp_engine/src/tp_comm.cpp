#include "tp_comm.hpp"

#ifdef DSV4_HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#endif

namespace dsv4 {

bool nccl_available() {
#ifdef DSV4_HAVE_NCCL
    return true;
#else
    return false;
#endif
}

#ifdef DSV4_HAVE_NCCL
namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

void check_nccl(ncclResult_t err, const char* what) {
    if (err != ncclSuccess) throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(err));
}

ncclUniqueId load_or_create_id(int rank, const char* path) {
    ncclUniqueId id;
    if (rank == 0) {
        check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to write NCCL id file");
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        if (!out) throw std::runtime_error("failed to write NCCL id bytes");
        return id;
    }
    for (int attempt = 0; attempt < 300; ++attempt) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (in.gcount() == static_cast<std::streamsize>(sizeof(id))) return id;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("timed out waiting for NCCL id file");
}

}  // namespace

void run_nccl_float_sum_smoke(int world, int rank, int device, const char* id_path, float value) {
    if (world <= 0 || rank < 0 || rank >= world) throw std::runtime_error("invalid NCCL world/rank");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    ncclUniqueId id = load_or_create_id(rank, id_path);
    ncclComm_t comm;
    check_nccl(ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    float* d_value = nullptr;
    float* d_sum = nullptr;
    check_cuda(cudaMalloc(&d_value, sizeof(float)), "cudaMalloc nccl value");
    check_cuda(cudaMalloc(&d_sum, sizeof(float)), "cudaMalloc nccl sum");
    check_cuda(cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice), "copy nccl input");
    check_nccl(ncclAllReduce(d_value, d_sum, 1, ncclFloat, ncclSum, comm, nullptr), "ncclAllReduce");
    check_cuda(cudaDeviceSynchronize(), "sync nccl smoke");
    float sum = 0.0f;
    check_cuda(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost), "copy nccl sum");
    std::cout << "nccl_smoke rank=" << rank << " value=" << value << " sum=" << sum << "\n";
    cudaFree(d_value);
    cudaFree(d_sum);
    ncclCommDestroy(comm);
}
#endif

}  // namespace dsv4
