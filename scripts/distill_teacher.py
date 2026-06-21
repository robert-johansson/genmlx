#!/usr/bin/env python3
"""Teacher generation for the cljs-coder distillation loop (genmlx-j0d6, step 2).

Reads task prompts (exported by `distill_filter.cljs --export-tasks`), runs a teacher
LLM N times per prompt via mlx-lm, and writes raw_candidates.jsonl for the GenMLX
oracle-filter (step 3). This is the ONLY step that touches the big model; it is a
decoupled batch job whose only interface to GenMLX is the candidates file.

ENGINE / MACHINE NOTES
  - mlx-lm is METAL-ONLY (Apple Silicon). On the Jetson Thor (CUDA) use vLLM /
    transformers instead — the candidates file is engine-agnostic.
  - The PRODUCTION teacher is Qwen3-Coder-Next-4bit (~45GB) — it needs a 96GB Mac;
    it will NOT load on a 32GB box. For a LOCAL smoke on a 32GB Mac mini, the default
    here is a small coder model (qwen25-coder-3b-4bit, ~2GB) that produces a realistic
    valid/invalid mix to exercise the filter. Swap with --model for the real teacher.

USAGE
  python scripts/distill_teacher.py --tasks <tasks.jsonl> --out <raw_candidates.jsonl> \\
      [--model <dir-or-hf-id>] [--n 8] [--max-tokens 512] [--temp 0.8] [--limit K]
"""

import argparse
import json
import os
import sys
import time


def resolve_model(spec: str) -> str:
    """A local ~/.cache/models/<name> dir if it exists, else the spec verbatim
    (an HF id / mlx-community name mlx-lm will download)."""
    local = os.path.expanduser(os.path.join("~/.cache/models", spec))
    return local if os.path.isdir(local) else spec


def build_prompt(tokenizer, system_prompt, user_prompt):
    """Render a chat prompt; Qwen3 reasoning is disabled when supported so the model
    emits the code form directly. Falls back to a plain template if needed."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="tasks.jsonl from --export-tasks")
    ap.add_argument("--out", required=True, help="raw_candidates.jsonl to write")
    ap.add_argument("--model", default="qwen25-coder-3b-4bit",
                    help="local ~/.cache/models/<name> dir or an HF/mlx-community id")
    ap.add_argument("--n", type=int, default=8, help="samples per prompt")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first K tasks (0 = all) — for a quick smoke")
    args = ap.parse_args()

    # Import here so --help works without mlx installed.
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    model_path = resolve_model(args.model)
    print(f"Loading teacher: {model_path}", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    n_written = 0
    with open(args.out, "w") as out:
        for ti, task in enumerate(tasks):
            prompt = build_prompt(tokenizer, task.get("system_prompt"), task["prompt"])
            print(f"[{ti + 1}/{len(tasks)}] {task['task_id']} "
                  f"({task.get('kind')}) x{args.n}", flush=True)
            for s in range(args.n):
                text = generate(model, tokenizer, prompt=prompt,
                                max_tokens=args.max_tokens, sampler=sampler,
                                verbose=False)
                out.write(json.dumps({
                    "task_id": task["task_id"],
                    "kind": task.get("kind"),
                    "sample_idx": s,
                    "raw_text": text,
                }) + "\n")
                out.flush()
                n_written += 1

    print(f"\nWrote {n_written} candidates -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
