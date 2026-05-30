# GenMLX — Distinctive-Features Demo Suite

Eight self-contained, runnable demos, each isolating one thing that makes GenMLX
distinctive. Every demo prints narrated output ending in `=== done ===` (or
`Done.`). The numbers below were observed on a real run — re-running reproduces
them up to PRNG variation.

```bash
# from the repo root
examples/distinctive/run_all.sh            # whole suite
examples/distinctive/run_all.sh core       # only the pure-GenMLX demos (no weights)
examples/distinctive/run_all.sh 06         # a single demo by number
# or directly:
bun run --bun nbb examples/distinctive/01_model_is_a_value.cljs
```

A harmless Bun/JSC `SIGTRAP` may print *after* a demo's output on teardown
(dev/TODO.md INFRA-3) — it is not a failure; the `=== done ===` line precedes it.

## The demos

| # | File | Distinctive feature | Headline observed result | Weights |
|---|------|--------------------|--------------------------|---------|
| 01 | `01_model_is_a_value.cljs` | **Homoiconicity-as-substrate** — one `gen` source, never rewritten | Same source is run, analyzed (`:L1-M3`), conditioned, vectorized (`[1000]` from one body run, Var≈4), and a static variant auto-detects conjugacy (`:L1-M2`) | — |
| 02 | `02_compilation_ladder_equivalence.cljs` | **Verified handler↔compiled equivalence** | Same idiom → 3 tiers (`L1-M3`/`L1-M4`/`L1-M2`); compiled vs handler `\|Δ\| = 0.000000000000`; 200-trial sweep, 0 failures | — |
| 03 | `03_llm_is_a_distribution.cljs` | **An LLM is an actual distribution** | Identical `p/simulate`/`p/generate` on coin, Gaussian, LLM; constraint log-weights `-0.6931` / `-2.0439` / `-12.6875` | qwen3-0.6b |
| 04 | `04_grammar_conditioning.cljs` | **Code/text as a grammar-conditioned random variable** | regex→DFA; invalid tokens masked to `-inf`; constrained `simulate` → "587"/"yes"; non-LLM die constrained to "3" → `[3 3 3]` | qwen3-0.6b |
| 05 | `05_program_synthesis.cljs` | **ClojureScript writing ClojureScript** (propose→SCI-eval→GFI-score) | 4B writes 3 models; log-ML ranks them (tighter prior wins, ≈ reference); posterior μ ≈ 2.95–3.09 vs truth 3 | qwen3.5-4b |
| 06 | `06_auto_analytical.cljs` | **Exact conjugacy auto-detected from source** | Analytical log-ML matches hand-derived MVN to `3e-7`; IS converges to it; posterior mean matches closed form to `0.01` | — |
| 07 | `07_shape_vectorization.cljs` | **Vectorization by shape/broadcasting, not vmap** | `[10000]` leaves from one body run; E/Var match prior; **~2400× speedup** vs scalar loop | — |
| 08 | `08_value_semantics_gpu.cljs` | **Lazy MLX graph is a value; `eval!` the sole dispatch** | 50-layer graph build 1.6 ms vs eval 76 ms (≈46×); trace `:score` is an `MxArray` until `mx/item` forces it | — |

## Notes on the LLM demos

Models live under `~/.cache/models`. Demos 03/04 use the small `qwen3-0.6b-mlx-bf16`
(loads in ~1–2 s — plenty for "the API is uniform"). Demo 05 uses
`qwen3.5-4b-mlx-bf16` because program synthesis needs a model that writes coherent
probabilistic programs; bump `MODEL-NAME` to `Qwen3.6-35B-A3B-4bit` for the strongest
synthesis.

**Thinking models:** Qwen3.5 is a *thinking* model. Generation in demo 05 uses
`llm/generate-text-raw`, which injects a closed empty `<think>\n\n</think>\n\n`
block (only when the detected `:type` is in `#{:qwen3 :qwen3_5 :qwen3_5_moe}`) so the
model skips its reasoning phase and answers directly. Verified: type `:qwen3_5`
matches, and no `<think>`/`</think>` tokens are emitted. The `msa` ChatSession path
(`synthesize-and-rank`) does **not** skip thinking and can hang for minutes on a
thinking model — that is why 05 drives generation via `generate-text-raw` + the
`msa` parse/assemble/eval/score helpers directly.

## How each maps to the "what is distinctive" discussion

- 01 → homoiconicity-as-substrate (the root: a model is data run/analyzed/compiled/conditioned/vectorized/synthesized)
- 02 → the compilation ladder with a verified equivalence invariant
- 03 → LLMs as first-class generative functions
- 04 + 05 → code as a grammar-conditioned random variable, and program synthesis as inference
- 06 → auto-analytical inference from static source analysis
- 07 → shape-based vectorization
- 08 → value-semantics carried through the GPU (the substrate that lets all of the above stay purely functional)
