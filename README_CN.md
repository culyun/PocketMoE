# PocketMoE（口袋 MoE）

[English](README.md) | [中文](README_CN.md)

PocketMoE 是一个面向消费级 GPU 的 **MoE-only 低 bit + 异构推理引擎**，目标是在家用/工作站硬件上做好 300B 以下 MoE 模型的本地化部署。

“口袋 MoE”的含义是：把原本属于数据中心集群的千亿级 MoE 模型，通过低 bit 压缩、量化 expert kernel 和 CPU/GPU 异构调度，放进用户的“口袋”（消费级显卡）里。

本项目最初来自“在 4×RTX 2080 Ti 上运行 DeepSeek-V4”的工程实践。DeepSeek-V4 仍然是第一个已验证 backend 和性能基线，但仓库现在会按更通用的 MoE 推理引擎方向组织。

## 项目定位

PocketMoE 专注于 MoE 模型，以及以下消费级/工作站 GPU：

- RTX 2080 Ti，包括 22 GiB 魔改卡；
- RTX 3090；
- RTX 4090；
- 类似的 PCIe 消费级多卡机器，没有数据中心级显存容量，也没有 NVLink。

项目要回答的问题很直接：

> 怎么在原本放不下这些模型的消费级显卡上运行大 MoE？

PocketMoE 的答案是两种执行模式。

### 1. 低 bit all-device 模式

如果低 bit MoE checkpoint 能放进多卡总显存，就尽量让模型常驻 device：

- dense / attention 权重在 GPU；
- router / gate 权重在 GPU；
- routed experts 在 GPU；
- decode 不走 active-expert H2D；
- prefill 使用 grouped resident MoE kernel；
- 热路径直接消费 raw quantized blocks，不把权重展开成 fp32。

这条路线适合 GGUF IQ1/IQ2/Q2/Q3 等能跨多张消费级显卡放下的 MoE checkpoint。

### 2. 异构 routed-expert 模式

如果 checkpoint 放不进 device，就把 routed experts 放在 CPU pinned/NUMA 内存中，只搬当前 token 或 prefill chunk 真正激活的 experts：

- dense/router 计算尽量留在 GPU；
- routed experts 常驻 CPU 内存；
- decode 按 top-k active experts staging 到 GPU；
- prefill 把 expert H2D 和 GPU compute overlap；
- hot expert cache 和 layer-local prefetch 用于减少重复 PCIe 流量。

这条路线适合高 bit MoE checkpoint，或者虽然是低 bit 但在预留 KV cache/workspace 后仍然放不下的模型。

## 非目标

PocketMoE 不是通用 dense 模型 serving 框架。

Dense layer、attention、tokenizer、OpenAI 兼容服务会被支持，是因为 MoE 模型需要它们；但本项目不打算和 vLLM、SGLang、llama.cpp 在通用 dense LLM runtime 上正面竞争。核心关注点是：

- routed expert 量化；
- routed expert 放置策略；
- active expert dispatch；
- prefill/decode MoE kernel；
- PCIe 消费级硬件上的 CPU/GPU 异构调度。

## 当前 backend：DeepSeek-V4 / DSV4

第一个已验证 backend 是 4×RTX 2080 Ti 上的 DeepSeek-V4-Flash。这个 backend 包含：

- DeepSeek-V4 MLA、sparse attention、C4 indexer 路径；
- DeepSeek 风格 hash routing 和 routed experts；
- Turing GPU 上没有原生 FP4/FP8 Tensor Core 时的 FP4/FP8 风格 checkpoint 处理；
- GGUF Q2/IQ2/IQ1 routed expert block 路径；
- CPU/GPU active expert staging；
- PyTorch runtime 和原生 C++/CUDA engine 两套路径。

仓库里现有脚本和 benchmark 数字，除非特别说明，都是 DeepSeek-V4 backend 的结果。

## 下一个 backend：MiniMax-M2.7

下一个目标是 MiniMax-M2.7 GGUF。已经下载的 `UD-IQ1_M` bundle 是 3 分片 GGUF checkpoint，结构为：

- `general.architecture = minimax-m2`；
- 62 层；
- hidden size 3072；
- context length 196608；
- 48 个 attention heads，8 个 KV heads；
- 256 个 routed experts，top-k 8；
- routed experts 是 `iq2_xxs`；
- attention projection 是 `q5_k`；
- embedding/head 是 `q4_k`；
- router/norm/bias 是 `f32`。

MiniMax-M2.7 支持会从以下能力开始：

- sharded GGUF inspect；
- architecture/spec 解析；
- tensor schema validation；
- quant capability report；
- all-device 与异构 placement planning。

MiniMax 的完整 generation 暂不启用；需要等 MiniMax GQA runtime、q4_k/q5_k kernel、以及 all-`iq2_xxs` MoE 路径实现后再打开。

## 硬件基线

原始 benchmark 机器为：

- GPU：4× NVIDIA GeForce RTX 2080 Ti，每张 22 GiB，Turing 架构。
- CPU：双路 Intel Xeon E5-2696 v4 @ 2.20 GHz。
- CPU 拓扑：88 个逻辑 CPU，2 socket，每 socket 22 core，每 core 2 thread。
- 系统内存：1 TiB。
- 运行方式：`torchrun --nproc-per-node 4`，每张 GPU 一个 rank。

重要限制：

- RTX 2080 Ti 没有原生 BF16、FP8、FP4 Tensor Core。
- 机器没有 NVLink；GPU-GPU 通信需要经过 PCIe/host bridge。
- PCIe Gen3 x16 带宽使 routed-expert staging 和 overlap 成为核心问题。
- 单请求 latency 和 decode TPS 是主要优化目标。

## DeepSeek-V4 backend 性能快照

### PyTorch FP4 异构路径

4×RTX 2080 Ti 上的代表性 FP4 结果：

| 场景 | Prompt / prefill tokens | Decode tokens | Prefill | Decode TPS | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| 已验证最大上下文 | 65,536 | 2 | 257.45s（约 255 tok/s） | n/a | OpenAI 路径，内容校验返回 `OK`。 |
| 长 prompt decode | 2,148 | 63 | 6.69s（warmup 后约 321 tok/s） | 3.49 tok/s mean | FP4 resident OpenAI 路径，3 次 fresh run：3.464/3.500/3.507 tok/s。 |
| 短 prompt decode | 29 | 127 | 实测约 1.7-3.2s | 3.16 tok/s mean | 短 prompt prefill timing 在这台机器上噪声较大。 |

### GGUF Q2/IQ2 异构路径

GGUF IQ2_XXS/Q2_K 路径使用 4-GPU TP、grouped GPU prefill、active-expert decode、slot cache、async all-reduce/fused finalize，以及量化 block expert staging。Routed GGUF expert 权重常驻 CPU，并以量化 block 形式 staging 到 GPU，不在热路径中展开成 fp32。

OpenAI 兼容 benchmark，`CASE=all REPEAT=2`，4×RTX 2080 Ti：

| Case | 请求 | Prompt tokens | 服务端 decode tokens | Prefill | Decode TPS | Wall time | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `short_short` | 1 | 5 | 7 | 3.17s（1.58 tok/s） | 2.94 | 5.64s | 冷 decode cache。 |
| `short_short` | 2 | 5 | 7 | 2.46s（2.03 tok/s） | 4.83 | 3.98s | warm slot/cache 路径。 |
| `long_short` | 2 | 2,148 | 7 | 9.94s（216.17 tok/s） | 4.44 | 11.66s | warm prefill staging。 |
| `long_long` | 2 | 2,148 | 63 | 10.08s（213.05 tok/s） | 3.75 | 27.18s | 较长 decode 仍然更难。 |

当前结论：在这套硬件和 checkpoint 上，prefill 已接近当前架构能达到的最好结果。Decode 仍主要受 active-expert cache miss/H2D copy、TP all-reduce/finalize 和长上下文 attention 成本限制。

### C++ engine

`cpp_engine/` 下的原生 C++/CUDA engine 用于减少 DeepSeek-V4 backend 的 Python/PyTorch per-step overhead。Rank 0 内置 OpenAI 兼容 HTTP 服务，rank 1-3 是 NCCL workers。

同一台 4×RTX 2080 Ti 机器上验证过的 FP4 结果：

| 场景 | Prompt tokens | Prefill | Prefill TPS | Decode TPS | 单卡峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短 prompt | 2,101 | ~7.6s | ~275 tok/s | ~3.7 tok/s | ~7 GiB |
| 32K 上下文 | 32,768 | ~82s | ~402 tok/s | n/a | ~11.2 GiB |
| 64K 上下文 | 65,536 | ~164s | ~401 tok/s | ~3.7 tok/s | ~14.5 GiB |

64K prefill 大约比 PyTorch FP4 路径快 1.6×。Decode TPS 接近，因为瓶颈主要是 PCIe expert staging，而不是 Python overhead。

## 构建

Python extension：

```bash
python -m pip install -r requirements.txt
python setup.py build_ext
```

C++ engine：

```bash
cmake -S cpp_engine -B build/cpp_engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_engine -j
```

当前 C++ binary 仍叫：

```text
build/cpp_engine/dsv4_cpp_engine
```

等 PocketMoE model-spec 层稳定后会再统一命名。

## 运行现有 DeepSeek-V4 backend

### PyTorch OpenAI 兼容服务

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
bash scripts/run_openai_server.sh
```

常用覆盖项：

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
HOST=127.0.0.1 \
PORT=8000 \
MASTER_PORT=29920 \
NPROC_PER_NODE=4 \
bash scripts/run_openai_server.sh
```

### GGUF Q2/IQ2 路径

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

## Inspect GGUF checkpoint

PocketMoE 会为 MoE model onboarding 增加 sharded GGUF inspect 能力。

DeepSeek-V4 legacy inspect：

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/deepseek-v4.gguf \
  --summary \
  --validate-ds4-q2
```

MiniMax-M2.7 spec/capability inspect：

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M \
  --architecture auto \
  --spec-summary \
  --validate-spec \
  --capability-report \
  --placement-report
```

MiniMax generation 还未启用；inspect 路径会把它标记为 deferred。

## Roadmap

- [x] 4×RTX 2080 Ti 上的 DeepSeek-V4 backend。
- [x] DeepSeek-V4 FP4 和 GGUF Q2/IQ2/IQ1 routed expert 路径。
- [ ] MoE model spec 和 architecture registry。
- [ ] Sharded GGUF bundle inspect。
- [ ] MiniMax-M2.7 tensor map validation 和 capability report。
- [ ] MiniMax dense/attention 权重的 q4_k/q5_k payload/kernel。
- [ ] MiniMax GQA attention runtime。
- [ ] all-`iq2_xxs` MiniMax MoE resident path。
- [ ] 面向未来 MoE 模型的高 bit 异构 routed expert 策略。

## 已知限制

- 当前 generation runtime 仍然是 DeepSeek-V4-specific。
- MiniMax-M2.7 初期只支持 inspect/spec/validation/capability report。
- MiniMax 的 q4_k/q5_k 执行 kernel 尚未实现。
- Dense-only 模型不是本项目目标。
- 性能对硬件、NUMA、PCIe 拓扑非常敏感。

## License

This repository's code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

Permitted uses include personal use, academic research, education, non-commercial benchmarking, and non-commercial deployment.

Commercial use is not permitted without separate written permission from the copyright holder, including but not limited to:

- selling hosted inference services based on this code;
- selling packaged deployments or appliances;
- selling hardware/software bundles using this code;
- using this code in paid consulting deliverables or commercial products.

Model weights, tokenizer files, and third-party dependencies are governed by their respective licenses. This repository's code license does not grant any additional rights to DeepSeek, MiniMax, or other third-party model assets.

## 致谢

PocketMoE 基于 PyTorch、CUDA、safetensors、GGUF、transformers、NCCL 等生态。仓库中的 runtime 组织、量化 expert 路径、CPU/GPU 调度和性能脚本，是面向消费级 GPU 的本地 MoE 推理工程实践。
