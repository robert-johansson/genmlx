#!/usr/bin/env python3
"""Reconstitute a frozen-experts GRPO-trained MoE save into a loadable checkpoint.

`Qwen35MoeModel.save_model_sync` (mlx-node `models/qwen3_5_moe/model.rs`) is
dense/bf16-only: on a 4-bit checkpoint trained with FROZEN packed experts
(genmlx-n32r) it skips the `switch_mlp` expert tensors with a warn, so the
trained save holds only the non-expert weights, in the engine's *internal*
layout (see the dense twin `scripts/convert_native_to_mlxlm.py`):

  - bare keys, no `language_model.model.` prefix
  - FUSED GatedDeltaNet projections `in_proj_qkvz` / `in_proj_ba` (CONTIGUOUS
    q,k,v|z and b|a — NOT the interleaved 80B layout)
  - lowercase `a_log`
  - `embedding.weight` / `final_norm.weight` (vs HF `embed_tokens` / `norm`)
  - `visual.*` vision keys (vs HF `vision_tower.*`)

This script merges such a save with its SOURCE snapshot (the checkpoint the
training run loaded): start from ALL source tensors, overwrite every mapped
trained tensor as dense bf16, delete the `.scales`/`.biases` companions of
modules that became dense, and keep the packed `switch_mlp` experts and the
`vision_tower.*` tensors unchanged. `config.json` is copied from the SOURCE
unchanged: per-tensor `.scales` presence decides quantized-vs-dense at load,
stale per-module overrides for now-dense modules are inert, and the top-level
quantization block is REQUIRED for the packed experts.

The fused GDN split points come from the SOURCE config (text_config nesting):
with key_dim = linear_num_key_heads*linear_key_head_dim and value_dim =
linear_num_value_heads*linear_value_head_dim,

  qkvz [2*key_dim + 2*value_dim, in] -> qkv [2*key_dim + value_dim] | z [value_dim]
  ba   [2*num_v_heads, in]           -> b   [num_v_heads]           | a [num_v_heads]

A save from an UNFROZEN (dense-experts) run also reconstitutes: its dense 3-D
`switch_mlp` tensors map by the plain rename and displace the packed source
experts + their quant companions.

Output: ONE `model.safetensors` in <out_dir> (~25 GB for the 35B; no index
json — both loaders prefer the single file) plus the source's config/tokenizer/
preprocessor files.

Usage:
  /home/robert/code/mlx/.venv/bin/python scripts/reconstitute_moe_checkpoint.py \
      <native_save_dir> --base <source_snapshot_dir> -o <out_dir> [--plan]
  /home/robert/code/mlx/.venv/bin/python scripts/reconstitute_moe_checkpoint.py \
      --self-test <source_snapshot_dir>

  --plan       dry run: read only names/shapes/dtypes (source index +
               safetensors headers, trained weights.mlx sidecar or header),
               print the full rename mapping, split plans, deletions and kept
               counts, and run all assertions. No tensor data is read.
  --self-test  synthesize the expected trained-save manifest from the source
               index (inverting this script's own mapping for the non-expert,
               non-vision modules) and feed it through --plan. Validates the
               mapping without needing a trained save on disk.
"""
import argparse
import json
import os
import shutil
import struct
import sys

import mlx.core as mx

# Never touch the GPU (one-GPU-process discipline on shared training hosts).
mx.set_default_device(mx.cpu)

PREFIX = "language_model.model."
LM_HEAD = "language_model.lm_head.weight"
LINEAR_ATTN = ".linear_attn."
QUANT_COMPANIONS = (".scales", ".biases")

# mlx-node DType enum (crates/mlx-core/src/array/mod.rs) -> safetensors tag,
# for decoding the weights.mlx sidecar's integer dtype field.
MLX_NODE_DTYPES = {0: "F32", 1: "I32", 2: "F16", 3: "BF16", 4: "U32", 5: "U8", 6: "I8"}


class PlanError(Exception):
    pass


class UnmappableKey(Exception):
    pass


# ---------------------------------------------------------------------------
# Source / trained-save readers (names, shapes, dtypes only — no tensor data)
# ---------------------------------------------------------------------------

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


def _read_safetensors_header(path):
    """Parse a safetensors header (8-byte LE length + JSON) -> {name: (dtype, shape)}."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {
        k: (v["dtype"], tuple(v["shape"]))
        for k, v in header.items()
        if k != "__metadata__"
    }


def _load_source(base_dir):
    """Return (index {name->shard}, info {name->(dtype, shape)}, config dict)."""
    cfg = json.load(open(os.path.join(base_dir, "config.json")))
    index = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    info = {}
    for shard in sorted(set(weight_map.values())):
        info.update(_read_safetensors_header(os.path.join(base_dir, shard)))
    missing = set(weight_map) - set(info)
    if missing:
        raise PlanError(f"{len(missing)} index entries missing from shard headers: "
                        f"{sorted(missing)[:5]}...")
    return weight_map, info, cfg


def _load_manifest(native_dir):
    """Trained-save manifest {name: (dtype|None, shape)} from the safetensors
    header when present, else the weights.mlx JSON sidecar. Reads no tensor data."""
    st = os.path.join(native_dir, "weights.safetensors")
    if os.path.isfile(st):
        return _read_safetensors_header(st), "weights.safetensors header"
    sidecar = os.path.join(native_dir, "weights.mlx")
    if os.path.isfile(sidecar):
        meta = json.load(open(sidecar))["weights"]
        return (
            {k: (MLX_NODE_DTYPES.get(v["dtype"]), tuple(v["shape"])) for k, v in meta.items()},
            "weights.mlx sidecar",
        )
    raise FileNotFoundError(f"neither weights.safetensors nor weights.mlx in {native_dir}")


# ---------------------------------------------------------------------------
# Name mapping (inverse of qwen3_5_moe save_model_sync's internal layout)
# ---------------------------------------------------------------------------

def _rename_plain(k):
    """Bare engine key -> HF-on-disk key. Raises UnmappableKey for aliens."""
    if k == "embedding.weight":
        return PREFIX + "embed_tokens.weight"
    if k == "final_norm.weight":
        return PREFIX + "norm.weight"
    if k == "lm_head.weight":
        return LM_HEAD
    if k.startswith("layers."):
        if k.endswith(LINEAR_ATTN + "a_log"):
            k = k[: -len("a_log")] + "A_log"
        return PREFIX + k
    raise UnmappableKey(k)


def _quant_bits(cfg, produced_name):
    """Quantization bits for a source module (per-module override, else default)."""
    q = cfg.get("quantization", cfg.get("quantization_config", {})) or {}
    module = produced_name[: -len(".weight")] if produced_name.endswith(".weight") else produced_name
    override = q.get(module)
    if isinstance(override, dict) and "bits" in override:
        return override["bits"]
    return q.get("bits", 4)


def _dense_shape(src_info, cfg, name):
    """Shape a dense replacement for source tensor `name` must have.

    Quantized modules store a U32-packed weight [..., in/(32/bits)]; the dense
    bf16 tensor unpacks the last axis. Unquantized tensors match exactly.
    """
    dtype, shape = src_info[name]
    scales = (name[: -len(".weight")] + ".scales") if name.endswith(".weight") else None
    if scales and scales in src_info and dtype == "U32":
        per_u32 = 32 // _quant_bits(cfg, name)
        return shape[:-1] + (shape[-1] * per_u32,)
    return shape


# ---------------------------------------------------------------------------
# Plan: the full merge, computed from names/shapes/dtypes only
# ---------------------------------------------------------------------------

def build_plan(manifest, src_info, cfg):
    """Compute the merge plan and run every assertion dry.

    Returns a dict with renames, expected shapes, deletions, kept counts,
    warnings and errors. Reads no tensor data.
    """
    qkv_split, b_split = _dims_from_config(cfg)
    errors, warnings = [], []
    renames = []          # (trained_key, [(src_name, (lo, hi) | None)])
    produced = {}         # src_name -> trained_key
    expected = {}         # src_name -> required dense shape
    visual_skipped = []
    n_split = 0

    for k in sorted(manifest):
        if k.startswith("visual."):
            visual_skipped.append(k)
            continue
        parts = None
        if k.endswith(LINEAR_ATTN + "in_proj_qkvz.weight"):
            base = PREFIX + k[: -len("in_proj_qkvz.weight")]
            lo_name, hi_name = base + "in_proj_qkv.weight", base + "in_proj_z.weight"
            split = qkv_split
        elif k.endswith(LINEAR_ATTN + "in_proj_ba.weight"):
            base = PREFIX + k[: -len("in_proj_ba.weight")]
            lo_name, hi_name = base + "in_proj_b.weight", base + "in_proj_a.weight"
            split = b_split
        else:
            try:
                parts = [(_rename_plain(k), None)]
            except UnmappableKey:
                errors.append(f"unmappable trained key: {k}")
                continue
        if parts is None:  # fused split
            if lo_name not in src_info or hi_name not in src_info:
                errors.append(f"{k}: split targets missing from source index "
                              f"({lo_name}, {hi_name})")
                continue
            lo_rows = _dense_shape(src_info, cfg, lo_name)[0]
            hi_rows = _dense_shape(src_info, cfg, hi_name)[0]
            if lo_rows != split:
                errors.append(f"{k}: config split {split} != source {lo_name} "
                              f"rows {lo_rows}")
            fused_rows = manifest[k][1][0]
            if fused_rows != lo_rows + hi_rows:
                errors.append(f"{k}: fused rows {fused_rows} != "
                              f"{lo_rows}+{hi_rows} (source qkv|z / b|a rows)")
            parts = [(lo_name, (0, lo_rows)), (hi_name, (lo_rows, lo_rows + hi_rows))]
            n_split += 1

        tdtype, tshape = manifest[k]
        if tdtype is not None and tdtype != "BF16":
            errors.append(f"{k}: dtype {tdtype}, save_model_sync is dense/bf16-only")
        for name, sl in parts:
            if name in produced:
                errors.append(f"{name}: produced twice ({produced[name]} and {k})")
                continue
            if name not in src_info:
                errors.append(f"{k} -> {name}: not in source index")
                continue
            produced[name] = k
            want = _dense_shape(src_info, cfg, name)
            expected[name] = want
            have = tshape if sl is None else (sl[1] - sl[0],) + tuple(tshape[1:])
            if tuple(have) != tuple(want):
                errors.append(f"{k} -> {name}: shape {tuple(have)} != required "
                              f"dense shape {tuple(want)}")
        renames.append((k, parts))

    if visual_skipped:
        warnings.append(
            f"trained save contains {len(visual_skipped)} visual.* tensors "
            f"(e.g. {visual_skipped[0]}) — SKIPPING them and keeping the source's "
            f"vision_tower.* unchanged (vision frozen during text-only training; "
            f"bit-equality with the save's visual.* is NOT asserted)")

    # Companions of now-dense modules. By construction a companion is deleted
    # iff it exists in the source (i.e. iff the module was quantized there).
    deletions = set()
    for name in produced:
        if name.endswith(".weight"):
            base = name[: -len(".weight")]
            for c in QUANT_COMPANIONS:
                if base + c in src_info:
                    deletions.add(base + c)

    # Every remaining .scales must be a switch_mlp expert with a U32 .weight.
    for name in sorted(src_info):
        if not name.endswith(".scales") or name in deletions:
            continue
        if ".mlp.switch_mlp." not in name:
            errors.append(f"non-expert quantized module survives the merge: {name} "
                          f"(its .weight was not replaced by the trained save)")
            continue
        sibling = name[: -len(".scales")] + ".weight"
        if sibling not in src_info or src_info[sibling][0] != "U32":
            errors.append(f"{name}: packed sibling {sibling} missing or not U32 "
                          f"({src_info.get(sibling, ('absent',))[0]})")

    kept = [n for n in src_info if n not in produced and n not in deletions]
    kept_experts = sum(1 for n in kept if ".mlp.switch_mlp." in n)
    kept_vision = sum(1 for n in kept if n.startswith("vision_tower."))
    kept_other = [n for n in kept if ".mlp.switch_mlp." not in n
                  and not n.startswith("vision_tower.")]
    if kept_other:
        warnings.append(f"{len(kept_other)} non-expert, non-vision source tensors "
                        f"not replaced by the trained save (e.g. {kept_other[0]}) — "
                        f"kept from source")

    return {
        "renames": renames,
        "produced": produced,
        "expected": expected,
        "deletions": sorted(deletions),
        "visual_skipped": visual_skipped,
        "splits": {"qkv_split": qkv_split, "b_split": b_split, "n_split": n_split},
        "counts": {
            "trained": len(manifest),
            "mapped": len(renames),
            "produced": len(produced),
            "source": len(src_info),
            "kept_experts": kept_experts,
            "kept_vision": kept_vision,
            "kept_other": len(kept_other),
            "output": len(src_info) - len(deletions),
        },
        "errors": errors,
        "warnings": warnings,
    }


def print_plan(plan, manifest_src, full_mapping):
    c, s = plan["counts"], plan["splits"]
    print(f"  trained manifest: {c['trained']} tensors ({manifest_src})")
    print(f"  source:           {c['source']} tensors")
    print(f"  splits:           qkv_split={s['qkv_split']}, b_split={s['b_split']} "
          f"({s['n_split']} fused tensors split)")
    if full_mapping:
        print(f"  rename mapping ({c['mapped']} trained -> {c['produced']} source):")
        for k, parts in plan["renames"]:
            if len(parts) == 1:
                print(f"    {k} -> {parts[0][0]}")
            else:
                segs = " + ".join(f"{n} [{lo}:{hi})" for n, (lo, hi) in parts)
                print(f"    {k} -> {segs}")
        print(f"  deletions ({len(plan['deletions'])} quantization companions of "
              f"now-dense modules):")
        for name in plan["deletions"]:
            print(f"    - {name}")
    else:
        print(f"  rename mapping:   {c['mapped']} trained -> {c['produced']} source")
        print(f"  deletions:        {len(plan['deletions'])} quantization companions")
    print(f"  kept from source: {c['kept_experts']} packed experts, "
          f"{c['kept_vision']} vision, {c['kept_other']} other")
    print(f"  output:           {c['output']} tensors -> model.safetensors")
    for w in plan["warnings"]:
        print(f"  WARNING: {w}")
    for e in plan["errors"]:
        print(f"  ERROR: {e}")
    n_checks = c["produced"] + len(plan["deletions"]) + s["n_split"]
    if not plan["errors"]:
        print(f"  assertions:       all passed "
              f"({c['produced']} name+shape, {s['n_split']} split-boundary, "
              f"remaining-.scales sweep)")
    else:
        print(f"  assertions:       {len(plan['errors'])} FAILED (of ~{n_checks})")


# ---------------------------------------------------------------------------
# Merge (the only code path that reads tensor data)
# ---------------------------------------------------------------------------

def merge(native_dir, base_dir, out_dir, plan, weight_map):
    weights = mx.load(os.path.join(native_dir, "weights.safetensors"))

    produced_arrays = {}
    for k, parts in plan["renames"]:
        w = weights[k]
        for name, sl in parts:
            arr = w if sl is None else w[sl[0]:sl[1]]
            if tuple(arr.shape) != tuple(plan["expected"][name]):
                raise PlanError(f"{k} -> {name}: loaded shape {tuple(arr.shape)} != "
                                f"planned {tuple(plan['expected'][name])}")
            produced_arrays[name] = arr

    deletions = set(plan["deletions"])
    out, consumed = {}, set()
    for shard in sorted(set(weight_map.values())):
        for name, arr in mx.load(os.path.join(base_dir, shard)).items():
            if name in deletions:
                continue
            if name in produced_arrays:
                out[name] = produced_arrays[name]
                consumed.add(name)
            else:
                out[name] = arr

    unconsumed = set(produced_arrays) - consumed
    if unconsumed:
        raise PlanError(f"{len(unconsumed)} trained tensors not consumed: "
                        f"{sorted(unconsumed)[:5]}...")
    if len(out) != plan["counts"]["output"]:
        raise PlanError(f"output has {len(out)} tensors, planned "
                        f"{plan['counts']['output']}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.safetensors")
    mx.save_safetensors(out_path, out, metadata={"format": "mlx"})
    print(f"  wrote:            {out_path} ({len(out)} tensors)")

    # Aux files come from the SOURCE unchanged (config.json especially: its
    # quantization block is required for the packed experts; stale overrides
    # for now-dense modules are inert because .scales presence decides at load).
    skip = {"model.safetensors.index.json", "README.md"}
    copied = []
    for fn in sorted(os.listdir(base_dir)):
        src = os.path.join(base_dir, fn)
        if fn in skip or fn.endswith(".safetensors") or not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(out_dir, fn))
        copied.append(fn)
    print(f"  aux files (from source): {', '.join(copied)}")


# ---------------------------------------------------------------------------
# Self-test: synthesize the expected trained-save manifest from the source
# ---------------------------------------------------------------------------

def synthesize_manifest(src_info, cfg):
    """Invert the trained->HF mapping over the source index: the exact name+shape
    set a complete frozen-experts save_model_sync save of this checkpoint has."""
    manifest = {}
    fused = {}  # bare fused name -> [rows_lo, rows_hi, tail_dims]
    for name in sorted(src_info):
        if name.startswith("vision_tower.") or ".mlp.switch_mlp." in name:
            continue
        if name.endswith(QUANT_COMPANIONS):
            continue
        shape = _dense_shape(src_info, cfg, name)
        if name == PREFIX + "embed_tokens.weight":
            bare = "embedding.weight"
        elif name == PREFIX + "norm.weight":
            bare = "final_norm.weight"
        elif name == LM_HEAD:
            bare = "lm_head.weight"
        else:
            assert name.startswith(PREFIX), name
            bare = name[len(PREFIX):]
            if bare.endswith(LINEAR_ATTN + "A_log"):
                bare = bare[: -len("A_log")] + "a_log"
        m = None
        for lo_suf, hi_suf, fused_suf in (
            ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_qkvz.weight"),
            ("in_proj_b.weight", "in_proj_a.weight", "in_proj_ba.weight"),
        ):
            if bare.endswith(LINEAR_ATTN + lo_suf):
                m = (bare[: -len(lo_suf)] + fused_suf, 0, shape)
            elif bare.endswith(LINEAR_ATTN + hi_suf):
                m = (bare[: -len(hi_suf)] + fused_suf, 1, shape)
        if m is None:
            manifest[bare] = ("BF16", shape)
        else:
            fname, half, shape = m
            fused.setdefault(fname, [None, None])[half] = shape
    for fname, (lo, hi) in fused.items():
        assert lo is not None and hi is not None and lo[1:] == hi[1:], (fname, lo, hi)
        manifest[fname] = ("BF16", (lo[0] + hi[0],) + lo[1:])
    return manifest


def self_test(base_dir):
    weight_map, src_info, cfg = _load_source(base_dir)
    manifest = synthesize_manifest(src_info, cfg)
    plan = build_plan(manifest, src_info, cfg)
    print_plan(plan, "SYNTHESIZED from source index", full_mapping=False)

    unmappable = [e for e in plan["errors"] if "unmappable" in e]
    unexpected = [e for e in plan["errors"] if "not in source index" in e]
    print(f"  unconsumed trained tensors: {len(unmappable)}")
    print(f"  unexpected produced names:  {len(unexpected)}")

    # The produced set must be exactly the replaceable source set.
    expected_replaced = {
        n for n in src_info
        if not n.startswith("vision_tower.") and ".mlp.switch_mlp." not in n
        and not n.endswith(QUANT_COMPANIONS)
    }
    ok = True
    if set(plan["produced"]) != expected_replaced:
        d1 = sorted(expected_replaced - set(plan["produced"]))[:5]
        d2 = sorted(set(plan["produced"]) - expected_replaced)[:5]
        print(f"  FAIL: produced != replaceable source set (missing {d1}, extra {d2})")
        ok = False
    if plan["errors"]:
        ok = False

    # Negative checks: visual.* keys skip with a warning; alien keys error.
    with_visual = dict(manifest)
    with_visual["visual.patch_embed.proj.weight"] = ("BF16", (1152, 3, 16, 16))
    p2 = build_plan(with_visual, src_info, cfg)
    visual_ok = (not p2["errors"] and len(p2["visual_skipped"]) == 1
                 and any("visual" in w for w in p2["warnings"]))
    print(f"  negative check (visual.* -> skip+warn, no error): "
          f"{'pass' if visual_ok else 'FAIL'}")
    with_alien = dict(manifest)
    with_alien["mtp.fc.weight"] = ("BF16", (2048, 4096))
    p3 = build_plan(with_alien, src_info, cfg)
    alien_ok = any("unmappable" in e for e in p3["errors"])
    print(f"  negative check (alien key -> error):               "
          f"{'pass' if alien_ok else 'FAIL'}")

    ok = ok and visual_ok and alien_ok
    print(f"self-test: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("native_dir", nargs="?",
                    help="native GenMLX save dir (weights.safetensors + weights.mlx)")
    ap.add_argument("--base", help="SOURCE snapshot the training run loaded "
                    "(shards + index + config + tokenizer)")
    ap.add_argument("-o", "--out", help="output checkpoint dir")
    ap.add_argument("--plan", action="store_true",
                    help="dry run on names/shapes/dtypes only; write nothing")
    ap.add_argument("--self-test", metavar="SOURCE_SNAPSHOT",
                    help="validate the mapping against a source snapshot with a "
                    "synthesized trained-save manifest (no trained save needed)")
    args = ap.parse_args()

    if args.self_test:
        if not os.path.isdir(args.self_test):
            print(f"error: not a directory: {args.self_test}", file=sys.stderr)
            sys.exit(2)
        sys.exit(0 if self_test(args.self_test) else 1)

    if not (args.native_dir and args.base and args.out):
        ap.error("native_dir, --base and -o are required (unless --self-test)")
    for d, what in [(args.native_dir, "native_dir"), (args.base, "base")]:
        if not os.path.isdir(d):
            print(f"error: {what} is not a directory: {d}", file=sys.stderr)
            sys.exit(2)

    weight_map, src_info, cfg = _load_source(args.base)
    manifest, manifest_src = _load_manifest(args.native_dir)
    plan = build_plan(manifest, src_info, cfg)
    print_plan(plan, manifest_src, full_mapping=True)
    if plan["errors"]:
        print(f"aborting: {len(plan['errors'])} plan errors", file=sys.stderr)
        sys.exit(1)
    if args.plan:
        print("plan only — nothing written.")
        return
    merge(args.native_dir, args.base, args.out, plan, weight_map)
    print("done.")


if __name__ == "__main__":
    main()
