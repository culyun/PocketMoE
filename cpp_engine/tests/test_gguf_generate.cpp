// Phase 3 step: end-to-end greedy decode smoke. Loads the model once and runs
// a per-step forward over [seed_tokens] + [max_new_tokens] positions, with
// KV cache sized to the full sequence. Validates that:
//   - the run completes without OOM/throw,
//   - each step produces a valid in-range token,
//   - per-step wall time is reported for a coarse decode-tps signal.
//
// TP mode (optional, all ranks must agree):
//   --tp-world W --tp-rank R --nccl-id-path PATH [--device D]
//
// Positional args are: <model.gguf> [max_new_tokens] [seed1 seed2 ...]

#include "dsv4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    dsv4::ForwardSmokeOptions opts;
    std::string ckpt;
    int max_new = 4;
    std::vector<int> seeds;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if (a == "--tp-world") opts.tp_world = std::stoi(need("--tp-world"));
        else if (a == "--tp-rank") opts.tp_rank = std::stoi(need("--tp-rank"));
        else if (a == "--device") opts.device = std::stoi(need("--device"));
        else if (a == "--nccl-id-path") opts.nccl_id_path = need("--nccl-id-path");
        else positional.push_back(a);
    }
    if (positional.empty()) {
        std::cerr << "usage: test_gguf_generate <model.gguf> [max_new_tokens] [seed1 seed2 ...] "
                     "[--tp-world W --tp-rank R --nccl-id-path PATH --device D]\n";
        return 2;
    }
    ckpt = positional[0];
    if (positional.size() >= 2) max_new = std::atoi(positional[1].c_str());
    for (size_t i = 2; i < positional.size(); ++i) seeds.push_back(std::atoi(positional[i].c_str()));
    if (seeds.empty()) seeds = {1234};

    try {
        auto r = dsv4::run_gguf_generate_smoke(ckpt, seeds, max_new, opts);
        const bool is_root = (opts.tp_rank == 0);
        if (is_root) {
            std::printf("n_layers=%d dim=%d vocab=%d prompt_tokens=%d decode_tokens=%d\n",
                        r.n_layers, r.dim, r.vocab, r.prompt_tokens, r.decode_tokens);
            std::printf("load_seconds   = %.3f\n", r.load_seconds);
            std::printf("forward_seconds= %.3f (%d positions; %.1f ms/step)\n",
                        r.forward_seconds,
                        static_cast<int>(r.top_logits.size()),
                        r.top_logits.empty() ? 0.0
                                             : 1000.0 * r.forward_seconds /
                                                   static_cast<double>(r.top_logits.size()));
            if (r.decode_tokens > 0) {
                const double tps = static_cast<double>(r.decode_tokens) / r.forward_seconds;
                std::printf("decode_tps     = %.2f (decode_tokens / total_forward)\n", tps);
            }
            std::printf("seeds          = [");
            for (size_t i = 0; i < seeds.size(); ++i) {
                std::printf("%s%d", i ? ", " : "", seeds[i]);
            }
            std::printf("]\n");
            std::printf("generated      = [");
            for (size_t i = 0; i < r.generated_tokens.size(); ++i) {
                std::printf("%s%d", i ? ", " : "", r.generated_tokens[i]);
            }
            std::printf("]\n");
            std::printf("top_logits[first..last] = ");
            for (size_t i = 0; i < r.top_logits.size(); ++i) {
                std::printf("%s%.3f", i ? ", " : "", r.top_logits[i]);
            }
            std::printf("\n");
        }

        if (static_cast<int>(r.generated_tokens.size()) != max_new) {
            std::cerr << "[FAIL] generated_tokens size mismatch\n"; return 1;
        }
        for (int t : r.generated_tokens) {
            if (t < 0 || t >= r.vocab) {
                std::cerr << "[FAIL] generated token out of vocab\n"; return 1;
            }
        }
        if (is_root) std::cout << "[PASS] gguf greedy decode smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
