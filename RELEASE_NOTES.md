# GenMLX v0 — pinned research release

**2026-07-18.** The first pinned research release: the GenMLX tree plus its
fork SHAs, frozen as a reproducible unit. Every claim below is measured — the
run that produced it is either in-tree (`docs/cuda-test-triage.md`,
`test/`, `bench/`) or named in the issue tracker (`.beans/`).

## Pins

| repo | commit | branch |
|---|---|---|
| genmlx | this tag | `main` |
| mlx-node (submodule) | `6aabbd9` | `thor/d58a-up-sync` (upstream `up/main 5602f12` fully merged) |
| mlx (nested submodule) | `102a90cb` | `thor/mlx-sync-d58a` (14 fork commits rebased onto upstream `de7e34290`) |
| malli (submodule) | `a74e3b45` | upstream `metosin/malli` (no fork) |
| instaparse / test.check | `afc70aa` / `48d8694` | nbb-compat forks |

Runtime: Bun + nbb `1.4.208`. `@genmlx/core` surface: 226 function exports,
pinned by the membrane coverage matrix (`docs/membrane-coverage.md`).

## Platform

Validated end-to-end on **Linux/CUDA** (Jetson AGX Thor, aarch64, 128GB
unified memory). **Metal (Apple Silicon) is untested at this exact pin** —
the tree is the same one developed on Metal, but the release-gate suites ran
on CUDA only. Known CUDA platform warts (exit-teardown aborts after green
tests; GPU run discipline) are documented in the README and
`docs/thor-gpu-discipline.md`.

## Suite state at the pin (Thor/CUDA, `TEST_TIME_SCALE=8`)

- Tiers: **fast 190/190 · medium 144/149 · slow 70/77** (d58a option-a
  validation), with every not-pass triaged in `docs/cuda-test-triage.md`.
  After this release's sweep the honest residual is **sbc_test** (slow-tier
  timeout at the scaled cap) plus **4 single-assert statistical MCMC band
  flakes** (non-deterministic, pass on reruns; the sound across-seed fix is
  specced in `genmlx-5hhd`). Everything else is green, cleanly skipped
  (absent optional checkpoints), or platform-gated with negative-contract
  assertions.
- True-binary contract battery (post-rebuild gate, all green): exact
  120/120, gradient_fd 83, score_gradient 48, clip_contract 7,
  membrane_coverage 32 (incl. the addon-linkage-freshness guard),
  llm_forward_parity 7/7, llm_branch_tokens 20/20, llm_cljs_forward_qwen35
  5/5, qmm_determinism, 80B qwen3_next native 13/13, llm_branched 8/8,
  llm_pi_provider 48/48.
- GFI algebraic laws: 84 laws in `gfi.cljs`, exercised by gfi_laws_test
  (196 assertions) and the property suite.

## Headline capabilities (measured)

**35B MoE as a probabilistic substrate.** Qwen3.6-35B-A3B (4-bit and 8-bit)
runs bit-deterministically on CUDA — 30/30 identical prefill+decode after
the qmm_sm80 race fix (guarded by qmm_determinism_test). Text forward,
KV-branching, and particle inference verified on-device; token-SMC lanes
give 2.53x over scalar; the owned (CLJS-graph) prefill runs at 5.0 ms/tok
via the fused GDN scan — faster than the native path. GRPO training works
against the 4-bit checkpoint via frozen experts (world_train 35/35 +
world_train_reward green at the pin); the T2 bake-off measured tokens-per-
solution 432 (base) / 1558 (arm B) / 1081 (sharpened arm C), GRPO −31%.

**Vision on CUDA.** qwen3.5 VLM image turns work on non-Metal via the flat
vision cores (dense and MoE), including owned VLM prefill into the branch
ledger at the pi-provider seam: an image turn renders marker tokens, the
model names the fixture's color at temp 0, follow-up text turns
delta-prefill over the image-conditioned branch byte-identically
(llm_pi_provider P9).

**Grammar-constrained tool calls, three levels deep.** The qwen3_xml
tool-call dialect compiles to a DFA over the declared toolset — malformed
calls are unrepresentable at sampling time, scalar and K-lane-batched alike
(llm_toolcall_test). Per-argument constraints ride the tool parameter
schema: a JSON-Schema `pattern` regex (the P10 gate: hot-temperature
control emits off-pattern coordinates, the constrained arm cannot), and —
new in this release — `x-genmlx-grammar: "cljs"`, the reader-level leg: the
argument is exactly one complete, delimiter-opened ClojureScript form,
enforced byte-granularly by edamame through a hybrid DFA+reader masker.
Acceptance evidence (`bench/cljs_grammar_evidence.cljs`, 0.8b, temp 1.0,
guarded): **220 turns → 118 constrained emissions, 118/118 exactly one
complete form, zero unparseable, zero retries** (76 turns hit the token
budget mid-block — the documented sampling-artifact class, no call
produced). Contract suite 61/61 model-free, including a real-tokenizer
replay pinning the eos-as-symbol exclusion (special tokens decode to
reader-valid symbol text and must never be admitted mid-form). See
`genmlx-3g0t`.

**Best-of-K with a verifier seam.** K candidates decode in one batched
owned forward (K=8 ≈ 1.33x one scalar step); the verifier callback receives
all K (text + parsed tool calls) and picks by winner-index or scores, with
throw/timeout fallback (llm_pi_provider P8/P11).

**The compilation ladder, L0–L4.** Level-0 certification 68/68; L1 schema
266/266, compiled simulate 85/85, partial prefix 92/92, combinator fusion
92/92; L3/L3.5 analytical elimination exact to the float32 floor; L4 single
fused graph with a measured 9.2x compiled-Adam speedup; gen_clj_compat
356/356 and genjax_compat 73/73.

## Scope notes

- The de-fork/packaging epics (`0yjj`/`nldo`/`00l3`) and agent-integration
  Levels 3–5 (`llj0`) are explicitly out of this release.
- Best-of-K does not yet compose with per-argument grammar or thinking
  (typed errors, not silent degradation): `genmlx-x7oj`, `genmlx-0tqv`.
- Fresh-clone story: `bun install` resolves `@genmlx/core` as a workspace
  symlink; the README Quick Start is the setup path (verified against a
  fresh clone on this host at the pin).
