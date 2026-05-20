#pragma once

namespace dsv4 {

struct TpTopResult {
    int token = 0;
    float logit = 0.0f;
};

bool nccl_available();

#ifdef DSV4_HAVE_NCCL
void run_nccl_float_sum_smoke(int world, int rank, int device, const char* id_path, float value);
TpTopResult nccl_global_top1(int world, int rank, int device, const char* id_path, int local_token, float local_logit);
#endif

}  // namespace dsv4
