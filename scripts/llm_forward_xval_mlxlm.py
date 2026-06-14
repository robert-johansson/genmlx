#!/usr/bin/env python3
"""Cross-validate the GenMLX LLM forward golden pins against Python mlx-lm.

The f6ov golden gate (``test/genmlx/llm_forward_golden_test.cljs``) pins GenMLX's
forward output -- argmax + top-5 token ids + logprobs -- for the trusted Qwen3.5
models. That test is *self-referential*: it pins the CURRENT GenMLX forward
against itself, so a systematically-wrong-but-self-consistent forward would still
pass it (the silent-blow-up failure mode f6ov warns about).

This script closes that gap. It loads the SAME models through Python mlx-lm -- a
fully independent forward implementation -- runs the SAME prompt, and asserts the
GenMLX golden pins reproduce under mlx-lm. It is the external "reference oracle"
leg the f6ov gate is still missing.

It is GATED: it skips cleanly (exit 0) if mlx-lm is not importable or a model dir
is absent, so it never breaks a machine without the oracle or the checkpoints.

Run:
    python3 scripts/llm_forward_xval_mlxlm.py
    # or fully isolated, without touching any venv (uv is installed):
    uv run --with mlx-lm python3 scripts/llm_forward_xval_mlxlm.py

Env:
    GENMLX_MODEL_ROOT   model directory (default: ~/.cache/models)

Exit codes:
    0  every present-model check passed, or everything skipped
    1  a present model's forward disagreed with the golden pins

SOURCE OF TRUTH for the pins below is ``test/genmlx/llm_forward_golden_test.cljs``
(the ``golden`` vector). If that test changes, re-sync GOLDEN here -- a mismatch
between the two files is itself the drift signal this cross-check exists to catch.
"""

import os
import sys

# --- Golden pins, copied verbatim from test/genmlx/llm_forward_golden_test.cljs ---
PROMPT = "The capital of France is"
EXPECTED_PROMPT_IDS = [760, 6511, 314, 9338, 369]
GOLDEN = [
    {
        "name": "qwen3.5-0.8b", "dir": "qwen3.5-0.8b-mlx-bf16",
        "argmax": 11751, "argmax_decoded": " Paris",
        "top5": [(11751, -2.171875), (279, -2.234375), (7172, -2.609375),
                 (25, -2.984375), (198, -2.984375)],
    },
    {
        "name": "qwen3.5-4b", "dir": "qwen3.5-4b-mlx-bf16",
        "argmax": 11751, "argmax_decoded": " Paris",
        "top5": [(11751, -0.601563), (7172, -2.843750), (264, -3.031250),
                 (3750, -3.468750), (279, -3.593750)],
    },
]

# Cross-IMPLEMENTATION tolerance. The CLJS test uses 0.01 (same bit-reproducible
# build vs itself). mlx-lm is a DIFFERENT forward, so individual logprobs differ
# by bf16 cross-kernel noise -- measured up to ~0.17 nat on 0.8b. 0.25 tolerates
# that band while still catching the "off by whole nats" garbage-forward failure
# mode (the pre-fix bug f6ov's gate was built to detect was off by whole nats).
LOGPROB_TOL = 0.25

MODEL_ROOT = os.environ.get(
    "GENMLX_MODEL_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "models")
)


def main() -> int:
    try:
        from mlx_lm import load
        import mlx.core as mx
    except ImportError as exc:
        print(f"SKIP: mlx-lm / mlx not importable ({exc}). "
              f"Run via `uv run --with mlx-lm python3 {sys.argv[0]}` "
              f"or `pip install mlx-lm`.")
        return 0

    npass = nfail = nskip = 0

    def ok(label):
        nonlocal npass
        npass += 1
        print(f"  PASS: {label}")

    def bad(label):
        nonlocal nfail
        nfail += 1
        print(f"  FAIL: {label}")

    for g in GOLDEN:
        path = os.path.join(MODEL_ROOT, g["dir"])
        if not os.path.isdir(path):
            print(f"\n== {g['name']} -- SKIP (absent: {path}) ==")
            nskip += 1
            continue

        print(f"\n== {g['name']} ==  prompt={PROMPT!r}")
        model, tok = load(path)

        # tokenizer stability (HARD) -- if this drifts the rest is meaningless
        ids = list(tok.encode(PROMPT))
        if ids == EXPECTED_PROMPT_IDS:
            ok(f"tokenizer prompt-ids stable {ids}")
        else:
            bad(f"tokenizer prompt-ids: expected {EXPECTED_PROMPT_IDS} got {ids}")
            continue

        # forward -> last-position logprobs (independent of GenMLX's forward)
        logits = model(mx.array([ids]))[0, -1]
        mx.eval(logits)
        logprobs = logits - mx.logsumexp(logits)

        order = [int(i) for i in mx.argsort(logits)[-5:].tolist()][::-1]
        amax = order[0]

        # argmax id -- the f6ov gate's core invariant (HARD)
        if amax == g["argmax"]:
            ok(f"argmax token id == {g['argmax']}")
        else:
            bad(f"argmax id: golden {g['argmax']} got {amax} ({tok.decode([amax])!r})")

        # argmax decodes to the expected word (sanity)
        dec = tok.decode([amax])
        if dec == g["argmax_decoded"]:
            ok(f"argmax decodes to {g['argmax_decoded']!r}")
        else:
            bad(f"argmax decode: golden {g['argmax_decoded']!r} got {dec!r}")

        # top-5 ids as a SET (rank order can tie/shuffle within bf16 noise)
        golden_ids = [tid for tid, _ in g["top5"]]
        if set(order) == set(golden_ids):
            ok(f"top-5 ids match (set) {sorted(golden_ids)}")
        else:
            bad(f"top-5 ids: golden {sorted(golden_ids)} got {sorted(order)}")

        # per-id logprob within the cross-impl bf16 band. Looks up mlx-lm's
        # logprob AT each golden id, so it is robust even if mlx-lm's own top-5
        # tail orders differently.
        worst = 0.0
        for tid, glp in g["top5"]:
            alp = float(logprobs[tid])
            d = abs(alp - glp)
            worst = max(worst, d)
            tag = f"logprob id={tid} golden={glp:.4f} mlx-lm={alp:.4f} (d={d:.4f})"
            if d <= LOGPROB_TOL:
                ok(tag)
            else:
                bad(tag + f" > tol {LOGPROB_TOL}")
        print(f"    [info] worst top-5 logprob d vs golden = {worst:.4f} "
              f"(bf16 cross-kernel band; tol {LOGPROB_TOL})")

    print(f"\n=== forward-xval (mlx-lm oracle): {npass} PASS, {nfail} FAIL, "
          f"{nskip} model(s) skipped ===")
    if npass == 0 and nfail == 0:
        print("    (nothing verified -- oracle present but no trusted model on disk)")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
