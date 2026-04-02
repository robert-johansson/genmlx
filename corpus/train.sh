#!/bin/bash
# Fine-tune Qwen3-0.6B on ClojureScript corpus via LoRA
#
# Conservative settings to avoid memory issues on 32GB M4:
#   - LoRA on 8 layers (not all 28)
#   - Batch size 1
#   - Max seq length 512
#   - Gradient checkpointing ON
#
# Memory estimate: ~2-3GB for model + ~1GB for LoRA/optimizer = ~4GB total
#
# After training, fuse adapters:
#   python3 -m mlx_lm fuse \
#     --model ~/.cache/models/qwen3-0.6b-mlx-bf16 \
#     --adapter-path corpus/adapters \
#     --save-path ~/.cache/models/qwen3-0.6b-cljs

python3 -m mlx_lm lora \
  --model ~/.cache/models/qwen3-0.6b-mlx-bf16 \
  --train \
  --data corpus/ \
  --fine-tune-type lora \
  --mask-prompt \
  --batch-size 1 \
  --iters 24000 \
  --learning-rate 1e-5 \
  --num-layers 8 \
  --max-seq-length 512 \
  --steps-per-report 100 \
  --steps-per-eval 250 \
  --val-batches 25 \
  --save-every 500 \
  --adapter-path corpus/adapters \
  --grad-checkpoint
