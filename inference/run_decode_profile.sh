#!/usr/bin/env bash
# Capture a single decode step torch.profiler trace from rank 0.
set -eo pipefail
ROOT="/mnt/data1/dsv4_inference/inference"
TORCHRUN="/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun"
CKPT="${CKPT_PATH:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash-w8a8}"
CFG="$ROOT/config_w8a8.json"
LONG="/tmp/dsv4_long_input_single.txt"
OUTDIR="${OUTDIR:-/tmp/dsv4_decode_profile}"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

env \
  DEEPSEEK_DECODE_PROFILE_DIR="$OUTDIR" \
  DEEPSEEK_DECODE_PROFILE_STEP=2 \
  DEEPSEEK_DECODE_PROFILE_RANK=0 \
  DEEPSEEK_PD_PHASE_AUTO_SELECT=1 \
  DEEPSEEK_GPU_PREFILL_MOE=1 \
  DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM=1 \
  DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN=1 \
  DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS=3 \
  DEEPSEEK_GPU_PREFILL_MOE_ARENA=1 \
  DEEPSEEK_INT8_IMPL=cuda_ext \
  DEEPSEEK_MOE_ASYNC_ALLREDUCE=1 \
  DEEPSEEK_SHARED_EXPERT_INT8=1 \
  DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1 \
  DEEPSEEK_FUSED_C4_INDEXER_CUDA=1 \
  DEEPSEEK_HC_PRE_CUDA=1 \
  DEEPSEEK_PD_DECODE_OMP_THREADS=8 \
  DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD=1 \
  DEEPSEEK_CPU_TOPK_PERSISTENT=1 \
  DEEPSEEK_PD_PREFILL_WQ_A_INT8=1 \
  DEEPSEEK_PD_PREFILL_WQ_B_INT8=1 \
  DEEPSEEK_PD_PREFILL_WKV_INT8=1 \
  DEEPSEEK_PD_PREFILL_WO_A_INT8=1 \
  DEEPSEEK_PD_PREFILL_WO_B_INT8=1 \
  DEEPSEEK_PD_PREFILL_INDEXER_WQ_B_INT8=1 \
  DEEPSEEK_PD_DECODE_WQ_A_INT8=1 \
  DEEPSEEK_PD_DECODE_WQ_B_INT8=1 \
  DEEPSEEK_PD_DECODE_WKV_INT8=1 \
  DEEPSEEK_PD_DECODE_WO_A_INT8=1 \
  DEEPSEEK_PD_DECODE_WO_B_INT8=1 \
  DEEPSEEK_PD_DECODE_INDEXER_WQ_B_INT8=1 \
  PYTHONPATH="$ROOT" \
  "$TORCHRUN" \
    --master-port 29915 \
    --nproc-per-node 4 \
    "$ROOT/generate.py" \
    --ckpt-path "$CKPT" \
    --config "$CFG" \
    --input-file "$LONG" \
    --max-new-tokens 6 \
    --temperature 0 \
    --routed-experts-device cpu \
    --pd-mode scheduler \
    >"$OUTDIR/run.log" 2>&1

echo "=== timing ==="
grep -E "generate time:|prefill time:|^Completion:" "$OUTDIR/run.log" || true
echo "=== summary (top 25 by CUDA time) ==="
ls "$OUTDIR"/*.summary.txt 2>/dev/null | while read f; do
  echo "-- $f --"
  cat "$f"
done
