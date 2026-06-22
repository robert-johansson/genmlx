#!/usr/bin/env python3
"""Persistent mlx-lm generation worker for the REPL-synthesis LOOP (genmlx-0yv7 /
genmlx-wpua). The policy LLM lives OUT-OF-PROCESS, exactly as the GenMLX oracle spine
(codegen.eval + msa-score) is native-free: the synthesis loop in `genmlx.world.synth` /
`genmlx.world.search` never loads a model — it shells out to THIS worker for proposals.

WHY A LONG-RUNNING SERVER (not the per-call batch in distill_teacher.py): the loop is
ADAPTIVE — every prompt is conditioned on the previous step's check feedback, so the
proposals cannot be pre-generated. The model must stay resident and answer one
request per propose() call. This holds the model in memory once (~19GB for the 35B-A3B
teacher on the 32GB mini) and serves K samples per HTTP round-trip.

PROMPT RENDERING is byte-identical to distill_teacher.build_prompt (chat template,
enable_thinking=False so Qwen3 emits the form directly), so the in-loop proposer and
the offline distill/SFT pipeline render the same way.

PROTOCOL (single-threaded http.server; the CLJS side curls it synchronously):
  POST /generate  {prompt, system?, n?, temperature?, max_tokens?, top_p?, seed?}
                  -> {completions: [str,...], n, gen_time_s, prompt_tokens,
                      completion_tokens, model}
  GET  /health    -> {status:"ready", model}

USAGE
  python3 scripts/llm_server.py --model qwen3.5-0.8b-cljs-sft600 --port 8765
  python3 scripts/llm_server.py --model Qwen3.6-35B-A3B-4bit   --port 8765
Prints "READY <model>" to stdout once the weights are loaded (the launcher polls it).
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def resolve_model(spec):
    """A local ~/.cache/models/<name> dir if it exists, else the spec verbatim."""
    local = os.path.expanduser(os.path.join("~/.cache/models", spec))
    return local if os.path.isdir(local) else spec


def build_prompt(tokenizer, system_prompt, user_prompt):
    """Render a chat prompt with Qwen3 reasoning disabled (emit the code form
    directly). Identical to scripts/distill_teacher.py."""
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
    ap.add_argument("--model", required=True,
                    help="local ~/.cache/models/<name> dir or an HF/mlx-community id")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model_path = resolve_model(args.model)
    adapter = resolve_model(args.adapter) if args.adapter else None
    print(f"Loading model: {model_path}"
          + (f"  (+LoRA {adapter})" if adapter else ""), flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path, adapter_path=adapter)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    def generate_n(prompt_text, system, n, temperature, max_tokens, top_p, seed):
        full = build_prompt(tokenizer, system, prompt_text)
        prompt_tokens = len(tokenizer.encode(full))
        sampler = make_sampler(temp=temperature, top_p=top_p)
        greedy = make_sampler(temp=0.0)
        outs, comp_tokens = [], 0
        t = time.time()
        for i in range(n):
            # Seed per sample for reproducibility; sample 0 of a >1 batch is greedy
            # (argmax) so a single deterministic anchor is always present, the rest
            # sampled for diversity (the stochastic-proposer regime Phase 2 needs).
            mx.random.seed(seed * 100003 + i)
            use_greedy = (n > 1 and i == 0) or temperature <= 0.0
            text = generate(model, tokenizer, prompt=full, max_tokens=max_tokens,
                            sampler=(greedy if use_greedy else sampler), verbose=False)
            outs.append(text)
            comp_tokens += len(tokenizer.encode(text))
        return {"completions": outs, "n": n, "gen_time_s": round(time.time() - t, 3),
                "prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
                "model": os.path.basename(model_path)}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ready",
                                 "model": os.path.basename(model_path)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                res = generate_n(
                    req["prompt"], req.get("system"),
                    int(req.get("n", 1)),
                    float(req.get("temperature", args.temp)),
                    int(req.get("max_tokens", args.max_tokens)),
                    float(req.get("top_p", args.top_p)),
                    int(req.get("seed", 1)))
                self._send(200, res)
            except Exception as e:  # noqa: BLE001 — report, never crash the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"READY {os.path.basename(model_path)} @ http://{args.host}:{args.port}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
