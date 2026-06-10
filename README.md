# PocketMoE（口袋 MoE）

[English](README.md) | [中文](README_CN.md)

PocketMoE is a **MoE-only low-bit and heterogeneous inference engine for consumer GPUs**, targeting local deployment of MoE models up to roughly the 300B-parameter class on home/workstation hardware.

The name means bringing hundred-billion-parameter MoE models that normally belong to data-center clusters into the user's “pocket” consumer GPUs through compression, quantized expert kernels, and CPU/GPU heterogeneous scheduling.

The project started as a DeepSeek-V4-on-4×RTX-2080-Ti engineering effort. DeepSeek-V4 remains the first validated backend and performance baseline, but the repository is now being organized as a broader MoE engine.

## Positioning

PocketMoE focuses on MoE models and consumer/workstation GPUs such as:

- RTX 2080 Ti, including 22 GiB modded cards;
- RTX 3090;
- RTX 4090;
- similar PCIe consumer GPU boxes without data-center memory capacity or NVLink.

The core question is simple:

> How can we run large MoE models on cards that were never meant to host them?

PocketMoE answers that with two execution modes.

### 1. Low-bit all-device mode

If a low-bit MoE checkpoint fits in aggregate GPU memory, keep the model on device as much as possible:

- dense / attention weights on GPU;
- router / gate weights on GPU;
- routed experts on GPU;
- decode without active-expert H2D copies;
- prefill through grouped resident MoE kernels;
- raw quantized blocks consumed directly by kernels instead of fp32 expansion.

This is the preferred direction for GGUF IQ1/IQ2/Q2/Q3-style MoE checkpoints that can fit across multiple consumer GPUs.

### 2. Heterogeneous routed-expert mode

If the checkpoint does not fit on device, keep routed experts in CPU pinned/NUMA memory and move only the active experts needed for the current token or prefill chunk:

- dense/router computation stays on GPU where possible;
- routed experts live on CPU memory;
- decode stages top-k active experts to GPU;
- prefill overlaps expert H2D with GPU compute;
- hot expert cache and layer-local prefetch can reduce repeated PCIe traffic.

This mode is for higher-bit MoE checkpoints or low-bit checkpoints that exceed the practical GPU memory budget after KV cache and workspace are reserved.

## Non-goals

PocketMoE is intentionally **not** a general dense-model serving framework.

Dense layers, attention, tokenization, and OpenAI-compatible serving are supported because MoE models need them, but the project does not try to compete with vLLM, SGLang, or llama.cpp as a general dense LLM runtime. The core focus is:

- routed expert quantization;
- routed expert placement;
- active expert dispatch;
- prefill/decode MoE kernels;
- CPU/GPU heterogeneous scheduling on PCIe consumer hardware.

## Current backend: DeepSeek-V4 / DSV4

The first validated backend is DeepSeek-V4-Flash on a 4×RTX 2080 Ti machine. This backend includes custom handling for:

- DeepSeek-V4 MLA, sparse attention, and C4 indexer paths;
- DeepSeek-style hash routing and routed experts;
- FP4 / FP8-style checkpoint formats on Turing GPUs without native FP4/FP8 tensor cores;
- GGUF Q2/IQ2/IQ1 routed expert block paths;
- CPU/GPU active expert staging;
- PyTorch runtime and native C++/CUDA engine paths.

Existing scripts and benchmark numbers in this repository are currently DeepSeek-V4 backend results unless explicitly stated otherwise.

## Next backend: MiniMax-M2.7

The next target is MiniMax-M2.7 GGUF. The downloaded `UD-IQ1_M` bundle is a 3-shard GGUF checkpoint with:

- `general.architecture = minimax-m2`;
- 62 layers;
- hidden size 3072;
- context length 196608;
- 48 attention heads and 8 KV heads;
- 256 routed experts, top-k 8;
- routed experts stored as `iq2_xxs`;
- attention projections stored as `q5_k`;
- embedding/head stored as `q4_k`;
- router/norm/bias tensors stored as `f32`.

MiniMax-M2.7 support starts with:

- sharded GGUF inspection;
- architecture/spec parsing;
- tensor schema validation;
- quant capability reporting;
- all-device versus heterogeneous placement planning.

Full MiniMax generation is deferred until MiniMax GQA runtime, q4_k/q5_k kernels, and an all-`iq2_xxs` MoE path are implemented.

## Hardware baseline

The original benchmark machine is:

- GPU: 4× NVIDIA GeForce RTX 2080 Ti, 22 GiB each, Turing architecture.
- CPU: dual-socket Intel Xeon E5-2696 v4 @ 2.20 GHz.
- CPU topology: 88 logical CPUs, 2 sockets, 22 cores/socket, 2 threads/core.
- System memory: 1 TiB.
- Runtime mode: `torchrun --nproc-per-node 4`, one rank per GPU.

Important constraints:

- RTX 2080 Ti has no native BF16, FP8, or FP4 tensor cores.
- The machine has no NVLink; GPU-GPU communication goes through PCIe/host bridges.
- PCIe Gen3 x16 bandwidth makes routed-expert staging and overlap critical.
- Single-request latency and decode TPS are the primary optimization targets.

## DeepSeek-V4 backend performance snapshot

### PyTorch FP4 heterogeneous path

Representative FP4 results on the 4×RTX 2080 Ti machine:

| Scenario | Prompt / prefill tokens | Decode tokens | Prefill | Decode TPS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Maximum validated context | 65,536 | 2 | 257.45s (~255 tok/s) | n/a | OpenAI path, content check returned `OK`. |
| Long prompt decode | 2,148 | 63 | 6.69s (~321 tok/s after warmup) | 3.49 tok/s mean | FP4 resident OpenAI path, 3 fresh runs: 3.464/3.500/3.507 tok/s. |
| Short prompt decode | 29 | 127 | ~1.7-3.2s observed | 3.16 tok/s mean | Short prefill timing is noisy on this machine. |

### GGUF Q2/IQ2 heterogeneous path

The GGUF IQ2_XXS/Q2_K path uses 4-GPU TP, grouped GPU prefill, active-expert decode, slot caching, async all-reduce/fused finalize, and quantized-block expert staging. Routed GGUF expert weights remain CPU-resident and are staged to GPU as quantized blocks rather than expanded fp32 weights.

Representative OpenAI-compatible benchmark, `CASE=all REPEAT=2`, 4×RTX 2080 Ti:

| Case | Request | Prompt tokens | Service decode tokens | Prefill | Decode TPS | Wall time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_short` | 1 | 5 | 7 | 3.17s (1.58 tok/s) | 2.94 | 5.64s | Cold decode cache. |
| `short_short` | 2 | 5 | 7 | 2.46s (2.03 tok/s) | 4.83 | 3.98s | Warm slot/cache path. |
| `long_short` | 2 | 2,148 | 7 | 9.94s (216.17 tok/s) | 4.44 | 11.66s | Warm prefill staging. |
| `long_long` | 2 | 2,148 | 63 | 10.08s (213.05 tok/s) | 3.75 | 27.18s | Longer decode remains harder. |

Current conclusion: prefill is close to the best result reached by the current architecture on this hardware. Decode remains limited by active-expert cache misses/H2D copies, TP all-reduce/finalize, and long-context attention cost.

### C++ engine

The native C++/CUDA engine under `cpp_engine/` removes Python/PyTorch per-step overhead for the DeepSeek-V4 backend. Rank 0 embeds an OpenAI-compatible HTTP server and ranks 1-3 are NCCL workers.

Validated FP4 results on the same 4×RTX 2080 Ti machine:

| Scenario | Prompt tokens | Prefill | Prefill TPS | Decode TPS | Peak GPU/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short prompt | 2,101 | ~7.6s | ~275 tok/s | ~3.7 tok/s | ~7 GiB |
| 32K context | 32,768 | ~82s | ~402 tok/s | n/a | ~11.2 GiB |
| 64K context | 65,536 | ~164s | ~401 tok/s | ~3.7 tok/s | ~14.5 GiB |

At 64K, prefill is roughly 1.6× faster than the PyTorch FP4 path. Decode TPS is similar because the bottleneck is PCIe expert staging rather than Python overhead.

## Build

Python extensions:

```bash
python -m pip install -r requirements.txt
python setup.py build_ext
```

C++ engine:

```bash
cmake -S cpp_engine -B build/cpp_engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_engine -j
```

The current C++ binary is still named:

```text
build/cpp_engine/dsv4_cpp_engine
```

The name will be generalized after the PocketMoE model-spec layer stabilizes.

## Run the existing DeepSeek-V4 backend

### PyTorch OpenAI-compatible server

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
bash scripts/run_openai_server.sh
```

Useful overrides:

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
HOST=127.0.0.1 \
PORT=8000 \
MASTER_PORT=29920 \
NPROC_PER_NODE=4 \
bash scripts/run_openai_server.sh
```

### GGUF Q2/IQ2 path

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash-tokenizer \
bash scripts/run_gguf_q2_layer_pp.sh
```

### C++ server

```bash
CKPT=/path/to/DeepSeek-V4-Flash \
PORT=8000 \
MAX_CONTEXT=8192 \
PYTHON=/path/to/python \
bash scripts/run_cpp_serve_tp4.sh
```

## Inspect GGUF checkpoints

PocketMoE adds sharded GGUF inspection for MoE model onboarding.

DeepSeek-V4 legacy inspect:

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/deepseek-v4.gguf \
  --summary \
  --validate-ds4-q2
```

MiniMax-M2.7 spec/capability inspect:

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M \
  --architecture auto \
  --spec-summary \
  --validate-spec \
  --capability-report \
  --placement-report
```

MiniMax generation is not enabled yet; the inspect path reports it as deferred.

## Roadmap

- [x] DeepSeek-V4 backend on 4×RTX 2080 Ti.
- [x] FP4 and GGUF Q2/IQ2/IQ1 routed expert paths for DeepSeek-V4.
- [ ] MoE model spec and architecture registry.
- [ ] Sharded GGUF bundle inspection.
- [ ] MiniMax-M2.7 tensor map validation and capability report.
- [ ] q4_k/q5_k payload/kernels for MiniMax dense and attention weights.
- [ ] MiniMax GQA attention runtime.
- [ ] all-`iq2_xxs` MiniMax MoE resident path.
- [ ] General high-bit heterogeneous routed expert policy for future MoE models.

## Known limitations

- The current generation runtime is still DeepSeek-V4-specific.
- MiniMax-M2.7 support initially covers inspect/spec/validation/capability reporting only.
- q4_k/q5_k execution kernels for MiniMax are not implemented yet.
- Dense-only models are not a project target.
- Performance is hardware-, NUMA-, and PCIe-topology-sensitive.

## License

This repository's code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

Permitted uses include personal use, academic research, education, non-commercial benchmarking, and non-commercial deployment.

Commercial use is not permitted without separate written permission from the copyright holder, including but not limited to:

- selling hosted inference services based on this code;
- selling packaged deployments or appliances;
- selling hardware/software bundles using this code;
- using this code in paid consulting deliverables or commercial products.

Model weights, tokenizer files, and third-party dependencies are governed by their respective licenses. This repository's code license does not grant any additional rights to DeepSeek, MiniMax, or other third-party model assets.

## Acknowledgement

PocketMoE builds on the PyTorch, CUDA, safetensors, GGUF, transformers, and NCCL ecosystems. The runtime organization, quantized expert paths, CPU/GPU scheduling, and performance scripts in this repository are engineering work for local MoE inference on consumer GPUs.
