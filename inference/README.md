# Inference code for DeepSeek models

Convert the original DeepSeek-V4-Flash checkpoint to INT8 W8A8 routed-expert weights while preserving the original 46-file safetensors layout and `model.safetensors.index.json` mapping.

```bash
python convert.py \
  --hf-ckpt-path /mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash \
  --save-path /mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash-w8a8
```

Run batch inference from file:

```bash
torchrun --nproc-per-node 4 generate.py \
  --ckpt-path /mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash-w8a8 \
  --config config.json \
  --input-file smoke_input.txt \
  --routed-experts-device cpu
```

Run interactive chat:

```bash
torchrun --nproc-per-node 4 generate.py \
  --ckpt-path /mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash-w8a8 \
  --config config.json \
  --interactive \
  --routed-experts-device cpu
```
