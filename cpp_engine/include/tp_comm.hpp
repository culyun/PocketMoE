#pragma once

namespace dsv4 {

bool nccl_available();

#ifdef DSV4_HAVE_NCCL
void run_nccl_float_sum_smoke(int world, int rank, int device, const char* id_path, float value);
#endif

}  // namespace dsv4
