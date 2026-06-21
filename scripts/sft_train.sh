#!/bin/bash
# LoRA SFT for the cljs-coder student (genmlx-o8w9, loop step 3/5).
#
# Trains a LoRA adapter on the qwen3.5 student over the oracle-validated, instruct/chat
# corpus prepared by scripts/sft_prep.cljs, then fuses it into a standalone model the
# GRPO step (genmlx-2ctu) and the GenMLX owned-forward (@genmlx/core) can reload.
#
# This is INSTRUCT SFT, not FIM: --mask-prompt trains the loss on the assistant turn only.
# The prior cljs fine-tunes failed STRUCTURALLY (qwen2 arch + FIM objective) — this path fixes both.
#
# <think> CONSISTENCY (genmlx-o8w9 review): mlx-lm renders SFT training rows with the chat
# template DEFAULT (it exposes no enable_thinking knob), while distill_teacher.py GENERATES with
# enable_thinking=False (a CLOSED EMPTY <think></think>). For qwen3.5-0.8b the template default IS
# closed-think, so train render == inference render and the student learns to emit code directly.
# For qwen3.5-4b the template default is OPEN-think, so the two diverge. The CHECK_THINK preflight
# below ABORTS on any such mismatch, so a student can never silently train in the wrong <think>
# context. (Fix for a mismatched model: pre-render a {text} corpus or override its template default.)
#
# 32GB-SAFE CONFIG (defaults below; override via env):
#   STUDENT      qwen3.5-0.8b-mlx-bf16  (the loop workhorse; = what GRPO loads)
#                qwen3.5-4b-mlx-bf16    (the ceiling-raiser; also fits 32GB under LoRA)
#   rank/scale   scripts/sft_lora.yaml  (rank 16, scale 20, dropout 0.05)
#   NUM_LAYERS   8   (last N of 24; -1 = all. 8 is a small-corpus default; raise with scale)
#   ITERS        200 (smoke. A meaningful run after genmlx-7473 scales the corpus: ~600-1500)
#   BATCH        1   (tiny corpus; raise to 2-4 once the corpus is larger)
#   LR           1e-5
#   MAXSEQ       2048 (system+user+assistant fit easily; completions are short forms)
#
# USAGE
#   # 1. build the data dir first:
#   bun run --bun nbb scripts/sft_prep.cljs --corpus <distill_sft.jsonl> --out $TMPDIR/genmlx-sft
#   # 2. train + fuse:
#   bash scripts/sft_train.sh
#   # override anything:
#   STUDENT=qwen3.5-4b-mlx-bf16 ITERS=800 NUM_LAYERS=-1 bash scripts/sft_train.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS="$HOME/.cache/models"
TMP="${TMPDIR:-/tmp}"

STUDENT="${STUDENT:-qwen3.5-0.8b-mlx-bf16}"
DATA="${DATA:-$TMP/genmlx-sft}"
ADAPTER="${ADAPTER:-$MODELS/${STUDENT}-cljs-sft-lora}"
FUSED="${FUSED:-$MODELS/${STUDENT}-cljs-sft-fused}"
NUM_LAYERS="${NUM_LAYERS:-8}"
ITERS="${ITERS:-200}"
BATCH="${BATCH:-1}"
LR="${LR:-1e-5}"
MAXSEQ="${MAXSEQ:-2048}"
FUSE="${FUSE:-1}"
LOG="${LOG:-$DATA/train.log}"

MODEL="$MODELS/$STUDENT"
[ -d "$MODEL" ] || { echo "ERROR: student model not found: $MODEL" >&2; exit 1; }
[ -f "$DATA/train.jsonl" ] || { echo "ERROR: $DATA/train.jsonl missing — run sft_prep.cljs first" >&2; exit 1; }
[ -f "$DATA/valid.jsonl" ] || { echo "ERROR: $DATA/valid.jsonl missing — run sft_prep.cljs first" >&2; exit 1; }

echo "Student:  $MODEL"
echo "Data:     $DATA ($(wc -l < "$DATA/train.jsonl" | tr -d ' ') train, $(wc -l < "$DATA/valid.jsonl" | tr -d ' ') valid)"
echo "Adapter:  $ADAPTER"
echo "Config:   rank=16 (sft_lora.yaml), layers=$NUM_LAYERS, iters=$ITERS, batch=$BATCH, lr=$LR, maxseq=$MAXSEQ, mask-prompt"
echo "Log:      $LOG"
echo ""

# <think> train/inference consistency preflight (genmlx-o8w9). mlx-lm trains with the chat
# template DEFAULT; distill_teacher.py infers with enable_thinking=False. If they disagree
# for this model, abort rather than train a student in a different <think> context than it
# will see at inference. Disable with CHECK_THINK=0 (e.g. once a {text} corpus is pre-rendered).
if [ "${CHECK_THINK:-1}" = "1" ]; then
  echo "Preflight: chat-template default vs enable_thinking=False ..."
  if ! python3 - "$MODEL" <<'PYCHK'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
user = [{"role": "user", "content": "x"}]
# A = mlx-lm's prompt-mask boundary render: default template, add_generation_prompt=True
# (this is the context the model is CONDITIONED on during --mask-prompt training).
A = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
# B = distill_teacher.py inference render: enable_thinking=False, add_generation_prompt=True.
try:
    B = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False, enable_thinking=False)
except TypeError:
    B = A  # template ignores the flag -> A IS the no-think render
if A == B:
    print("  OK: training mask-boundary render == enable_thinking=False inference render")
    sys.exit(0)
print("  MISMATCH: this model's DEFAULT generation-prompt differs from enable_thinking=False.")
print("    train boundary:", repr(A[-30:]))
print("    inference     :", repr(B[-30:]))
print("    mlx-lm --mask-prompt conditions training on the DEFAULT prompt; distill_teacher.py")
print("    infers with enable_thinking=False, so the student trains in a different <think> context")
print("    (e.g. qwen3.5-4b's default leaves <think> OPEN; it would learn to emit the closing")
print("    </think> that inference already supplies). Use a student whose default matches")
print("    (qwen3.5-0.8b), pre-render a {text} corpus with the closed-think prefix baked in, or")
print("    override this model's chat_template default.")
sys.exit(1)
PYCHK
  then
    echo "ABORTING: train/inference <think> render mismatch for $STUDENT (set CHECK_THINK=0 to override)"
    exit 9
  fi
fi

mkdir -p "$ADAPTER" "$(dirname "$LOG")"

python3 -m mlx_lm lora \
  -c "$DIR/sft_lora.yaml" \
  --model "$MODEL" \
  --data "$DATA" \
  --train \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers "$NUM_LAYERS" \
  --batch-size "$BATCH" \
  --iters "$ITERS" \
  --learning-rate "$LR" \
  --max-seq-length "$MAXSEQ" \
  --steps-per-report 20 \
  --steps-per-eval 50 \
  --val-batches -1 \
  --save-every 100 \
  --adapter-path "$ADAPTER" 2>&1 | tee "$LOG"

echo ""
echo "LoRA adapter written -> $ADAPTER"

if [ "$FUSE" = "1" ]; then
  echo "Fusing adapter into a standalone model -> $FUSED"
  python3 -m mlx_lm fuse \
    --model "$MODEL" \
    --adapter-path "$ADAPTER" \
    --save-path "$FUSED"
  # mlx_lm.fuse may shard large models; merge shards so the GenMLX owned-forward loader
  # (@genmlx/core) sees a single model.safetensors. No-op for a single-shard small model.
  if ls "$FUSED"/model-*.safetensors >/dev/null 2>&1; then
    echo "  merging safetensors shards -> model.safetensors"
    python3 -c "
import mlx.core as mx, os, glob
p='$FUSED'
shards=sorted(glob.glob(os.path.join(p,'model-*.safetensors')))
t={}
for s in shards: t.update(mx.load(s))
mx.save_safetensors(os.path.join(p,'model.safetensors'), t)
print('  merged', len(shards), 'shards')
"
  fi
  echo "Fused student -> $FUSED  (reloadable by GenMLX / the GRPO step)"
fi

echo ""
echo "DONE. Next: grade baseline vs SFT on the held-out eval tasks:"
echo "  bun run --bun nbb scripts/sft_prep.cljs --export-eval-tasks $DATA/eval_tasks.jsonl"
echo "  python3 scripts/distill_teacher.py --model $STUDENT               --tasks $DATA/eval_tasks.jsonl --out $DATA/eval_baseline.jsonl --greedy-first --n 5"
echo "  python3 scripts/distill_teacher.py --model $STUDENT --adapter $ADAPTER --tasks $DATA/eval_tasks.jsonl --out $DATA/eval_sft.jsonl      --greedy-first --n 5"
echo "  bun run --bun nbb scripts/sft_eval.cljs --baseline $DATA/eval_baseline.jsonl --sft $DATA/eval_sft.jsonl --out $DATA --k 4"
