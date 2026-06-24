#!/usr/bin/env python3
"""Convert a native GenMLX `saveModel` directory into an mlx-lm-loadable model.

GenMLX's native `GrpoTrainingEngine`/`Qwen35Model.saveModel` writes weights in the
engine's *internal* layout (see mlx-node `models/qwen3_5/persistence.rs`):

  - bare keys, no `model.language_model.` prefix
  - FUSED GatedDeltaNet projections `in_proj_qkvz`, `in_proj_ba`
  - lowercase `a_log`
  - `embedding.weight` / `final_norm.weight` (vs HF `embed_tokens` / `norm`)

Python mlx-lm (`models/qwen3_5.py`) expects the HF layout the *base* checkpoint
uses: the `model.language_model.` prefix, SPLIT projections
`in_proj_qkv`/`in_proj_z` and `in_proj_b`/`in_proj_a`, and capital `A_log`. This
script applies the exact inverse of `persistence.rs::sanitize_weights` +
`merge_split_projections`, so a GRPO-trained model can be served by
`scripts/llm_server.py` for in-loop eval (bean genmlx-boh1).

The forward (load) mapping being inverted, from persistence.rs:
  in_proj_qkv ++ in_proj_z      --concat axis0-->  in_proj_qkvz   (q,k,v | z)
  in_proj_b   ++ in_proj_a      --concat axis0-->  in_proj_ba     (b | a)
  embed_tokens.* -> embedding.* ; norm.weight -> final_norm.weight
  A_log -> a_log ; conv1d transposed to [ch, k, 1] (kept as-is here: mlx-lm's
  sanitize passes [..., 1] through untouched, matching the internal layout).

Split points come from config: with key_dim = num_k_heads*key_head_dim and
value_dim = num_v_heads*value_head_dim,
  qkvz [key_dim*2 + value_dim*2] -> qkv [key_dim*2 + value_dim] | z [value_dim]
  ba   [num_v_heads*2]           -> b   [num_v_heads]           | a [num_v_heads]

Config + tokenizer are copied from --base (the original mlx-lm-loadable
checkpoint of the SAME architecture); the native dir's own config.json is a
minimal engine config, not an HF one.

Usage:
  python3 scripts/convert_native_to_mlxlm.py <native_dir> --base <base_model_dir> -o <out_dir>
"""
import argparse
import json
import os
import shutil
import sys

import mlx.core as mx

PREFIX = "model.language_model."
LINEAR_ATTN = ".linear_attn."


def _dims_from_config(cfg):
    """Return (qkv_split, b_split) from an HF/native config (handles text_config nesting)."""
    tc = cfg.get("text_config", cfg)
    num_k = tc["linear_num_key_heads"]
    num_v = tc["linear_num_value_heads"]
    kd = tc["linear_key_head_dim"]
    vd = tc["linear_value_head_dim"]
    key_dim = num_k * kd
    value_dim = num_v * vd
    qkv_split = key_dim * 2 + value_dim  # rows of in_proj_qkv (q,k,v); z is the rest
    b_split = num_v                      # rows of in_proj_b; a is the rest
    return qkv_split, b_split


def _split_fused(prefix_key, w, split, lo_name, hi_name):
    """Split a fused [out, in] (or [out, ...]) projection along axis 0 into two."""
    if w.shape[0] <= split:
        raise ValueError(
            f"{prefix_key}: fused dim {w.shape[0]} <= split {split}; "
            f"config dims do not match this checkpoint"
        )
    return {
        prefix_key + lo_name: w[:split],
        prefix_key + hi_name: w[split:],
    }


def _rename_text_key(k):
    """Inverse of persistence.rs sanitize renames (bare key -> HF-on-disk key, no prefix)."""
    if k.startswith("embedding."):
        k = "embed_tokens." + k[len("embedding."):]
    elif k == "final_norm.weight":
        k = "norm.weight"
    if k.endswith(LINEAR_ATTN + "a_log"):
        k = k[: -len("a_log")] + "A_log"
    return k


def _find_native_weights(native_dir):
    cand = os.path.join(native_dir, "weights.safetensors")
    if os.path.isfile(cand):
        return cand
    sts = [f for f in os.listdir(native_dir) if f.endswith(".safetensors")]
    if len(sts) == 1:
        return os.path.join(native_dir, sts[0])
    raise FileNotFoundError(
        f"no unambiguous *.safetensors in {native_dir} (found {sts})"
    )


def convert(native_dir, base_dir, out_dir):
    cfg_path = os.path.join(base_dir, "config.json")
    cfg = json.load(open(cfg_path))
    qkv_split, b_split = _dims_from_config(cfg)

    weights_file = _find_native_weights(native_dir)
    weights = mx.load(weights_file)

    out = {}
    n_split = 0
    for k, w in sorted(weights.items()):
        if k.endswith(LINEAR_ATTN + "in_proj_qkvz.weight"):
            p = PREFIX + k[: -len("in_proj_qkvz.weight")]
            out.update(_split_fused(p, w, qkv_split, "in_proj_qkv.weight", "in_proj_z.weight"))
            n_split += 1
        elif k.endswith(LINEAR_ATTN + "in_proj_ba.weight"):
            p = PREFIX + k[: -len("in_proj_ba.weight")]
            out.update(_split_fused(p, w, b_split, "in_proj_b.weight", "in_proj_a.weight"))
            n_split += 1
        else:
            out[PREFIX + _rename_text_key(k)] = w

    os.makedirs(out_dir, exist_ok=True)
    mx.save_safetensors(
        os.path.join(out_dir, "model.safetensors"), out, metadata={"format": "mlx"}
    )

    # Copy config + tokenizer/aux files from the base (HF) checkpoint; skip the
    # base's own weights and shard index (we wrote a single-file safetensors).
    skip_ext = (".safetensors",)
    skip_names = {"model.safetensors.index.json"}
    copied = []
    for fn in os.listdir(base_dir):
        if fn in skip_names or fn.endswith(skip_ext):
            continue
        src = os.path.join(base_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
            copied.append(fn)

    print(f"  native weights:   {len(weights)} keys ({weights_file})")
    print(f"  mlx-lm weights:   {len(out)} keys  ({n_split * 2} from {n_split} fused splits)")
    print(f"  copied from base: {sorted(copied)}")
    print(f"  wrote:            {out_dir}/model.safetensors")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("native_dir", help="native GenMLX saveModel dir (weights.safetensors + config.json)")
    ap.add_argument("--base", required=True, help="base mlx-lm checkpoint of the SAME arch (for config + tokenizer)")
    ap.add_argument("-o", "--out", required=True, help="output dir (mlx-lm-loadable)")
    args = ap.parse_args()

    for d, what in [(args.native_dir, "native_dir"), (args.base, "base")]:
        if not os.path.isdir(d):
            print(f"error: {what} is not a directory: {d}", file=sys.stderr)
            sys.exit(2)

    convert(args.native_dir, args.base, args.out)
    print("done.")


if __name__ == "__main__":
    main()
