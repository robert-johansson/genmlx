# HANDOFF — GenMLX-native particle engine (Route A + Route B) — autonomous run

**Audience: a fresh Claude Code session after a machine reboot, with NO memory of the prior
conversation.** Everything you need is here, in beans, in git, and in `~/genmlx-loop-artifacts/`.
`/tmp` was wiped by the reboot — do NOT look for anything under `/tmp/genmlx-loop`.

Bean: **genmlx-qsoa** (high priority, parent milestone genmlx-8587). Related: genmlx-2ctu (GRPO),
genmlx-7473 (generator, done), genmlx-8587 (milestone). Prior results: `docs/cljs-coder-loop-results.md`.

---

## 0. YOUR MISSION (run autonomously, ~several hours OK)

Build the **mlx-node-native batched particle engine** and ship a **demo** where the local
**0.8B GenMLX-cljs specialist generates a probabilistic program by sampling K particles in one
batched GPU pass and selecting the best by the GenMLX oracle (Bayesian model evidence)** — and
show it is **(a) better-fitting and (b) faster than a single shot from the 35B teacher**
(Qwen3.6-35B-A3B). Attempt **Route A** (reliable, de-risked) first, then **Route B** (the
GFI-native version, a stretch). Be ruthlessly honest in the final report: if a CLEAN benchmark
shows we do NOT beat the teacher on programs, say so plainly (see §11 go/no-go).

Scope the demo to **probabilistic-program / model synthesis** (GenMLX's actual domain). That is
where the 0.8B+particles+oracle beats the teacher. Do NOT scope it to general cljs functions —
there the 35B teacher wins on raw capability and best-of-K cannot close the gap (evidence below).

---

## 1. GOAL, SUCCESS CRITERIA, FALLBACK

**Demo "win" = both of these, measured cleanly, on probabilistic-program tasks:**
1. **Better fit:** the best-of-K-selected program's model evidence (log p(data|program)) ≥ the
   teacher's single-shot program's evidence (higher = better). Tonight's non-clean number:
   **0.8B best-of-16 = −4.10 vs teacher single = −4.83** (mean over the 11 held-out program tasks).
2. **Faster:** wall-clock to generate+score K batched particles (model preloaded) < wall-clock
   for one teacher completion. Tonight's per-use numbers: 0.8B ~1.08 s/completion (~53 tok/s)
   vs teacher ~9–10 s/completion (~6–11 tok/s); native batched decode of 16 seqs took 13.5 s
   INCLUDING engine construction + prefill (not a clean number — you must measure cleanly).

**Fallback (when to recommend relying on qwen-3.6 instead):** only if, after a clean Route-A
benchmark, the 0.8B best-of-K does NOT beat the teacher on programs in BOTH fit and speed. The
prior (non-clean) evidence says we DO win, so the expected outcome is "ship the particle engine,"
not "fall back." General-cljs coding is a different story (we lose there) — but that is not this demo.

---

## 2. HONEST FEASIBILITY (assessed 2026-06-22, before the reboot)

- **Route A (native `generateBatch` → oracle best-of-K → benchmark): HIGH feasibility.** The
  native batched decoder was probed standalone and is STABLE (16 seqs, 13.5 s, no wedge). Clear,
  well-scoped. This is the reliable deliverable and very likely lands autonomously.
- **Route B (batch-dim the owned cljs forward + masked-EOS LLM-GF so `vsimulate` runs K
  particles): STRETCH / risky.** It is deep surgery on the value-level forward (esp. the qwen3.5
  hybrid GatedDeltaNet cache) and must match the golden oracle bit-for-bit. Attempt it, gate it
  on correctness, and FALL BACK to Route A for the demo if it can't be made correct reliably.
- **"Better than teacher": TRUE on probabilistic programs, FALSE on general functions.** Build
  the demo on programs.

---

## 3. CURRENT STATE (all committed to git on branch `feat/cljs-task-gen-7473`)

Commits (newest first): `010d99a` GRPO wedge doc · `7d80513` GRPO script + results doc ·
`e5dc40d` pipeline fixes (sandbox --gen, sft_eval arity) · `52d5a3c` the 183-task generator.
(If on `main` after a merge, `git log --oneline | grep cljs-coder` to find them.)

Built & shipped this loop:
- **183-task generator** `src/genmlx/world/distill_gen.cljs` (+ `distill_gen_extra.cljs`): 44
  conjugate programs (all exact-scored) + 139 functions; 143 train / 40 eval split. Test
  `test/genmlx/distill_gen_test.cljs` 18/18.
- **Distill→SFT pipeline:** teacher (Qwen3.6-35B-A3B) → 512-row oracle-validated corpus → LoRA
  SFT of qwen3.5-0.8b → **two fused models** (see §4). Held-out lift (oracle-graded, leakage-free):
  function pass@1 0.172→0.345 (SFT-300) / 0.276 (SFT-600); function pass@4 0.103→0.310/0.414.
- **GRPO** `scripts/grpo_sharpen.cljs`: mechanism validated but the native TRAINING step
  **wedged** (~3.5 h stall in `generateBatchForTraining`/weight-update — NOT the decode). Bean
  genmlx-2ctu.
- Results writeup: `docs/cljs-coder-loop-results.md`.

---

## 4. WHERE EVERYTHING IS (post-reboot)

**Models** (`~/.cache/models/`, survive reboot):
- `qwen3.5-0.8b-cljs-sft600` — **the particle/demo model** (best pass@4 0.575, most headroom).
- `qwen3.5-0.8b-cljs-sft300` — best greedy pass@1 0.525.
- `qwen3.5-0.8b-mlx-bf16` — the base 0.8B (what the GRPO/native path loads via `Qwen35Model.load`).
- `Qwen3.6-35B-A3B-4bit` — the teacher (the baseline to beat). 19 GB; peaks ~20 GB on this 32 GB box.

**Preserved artifacts** (`~/genmlx-loop-artifacts/`, copied out of /tmp before reboot):
- `corpus/distill_sft.jsonl` (512 SFT rows), `corpus/stats.json`, `corpus/verdicts.jsonl`
- `gen_tasks.edn` (183 tasks; single-line EDN — regenerate with `gen_tasks.cljs --write-edn` if needed)
- `train_prompts.jsonl` (143), `eval_prompts.jsonl` (40), `reference_candidates.jsonl` (143)
- `teacher_candidates.jsonl` (858 — EXPENSIVE to regenerate; ~40 min of 35B)
- `eval/baseline.jsonl`, `eval/sft_300.jsonl`, `eval/sft_600.jsonl` (n=5), `eval/teacher_eval.jsonl`
  (40 greedy), `eval/sft600_k16.jsonl` (640 = K=16 particles), `eval/cmp{300,600}/eval_report.json`
- `adapter/` (LoRA checkpoints 100–700), `sft-data/` (train/valid)
- `compare.cljs`, `demo.cljs` (best-of-K scorer), `probe.cljs` (the stable-decode probe) — reference impls

**Out-of-tree nbb runner** (to run a scratch script against the repo): from the repo root,
`ln -sfn $(pwd)/node_modules <dir>/node_modules` then
`bunx nbb@1.4.208 -cp "src:test:examples:test.check:malli/src:instaparse/src" <dir>/script.cljs`.

---

## 5. VERIFIED ARCHITECTURE STATE (workflow w182a5x7r, 3 cross-checked readers; do NOT re-derive)

Batched `vsimulate` of K LLM particles in one native pass is **NOT wired** as an inference path.
Three layers:

- **Layer 1 — pure-cljs vectorized sampling: DONE.** Categorical batched sampler `dist.cljs:586`
  (Gumbel-max `[n,k]→[N]`); per-particle `[N,K]` log-prob via take-along-axis `dist.cljs:558`
  (exactly the LLM case — each particle a different history); dist-agnostic batched handler
  `handler.cljs:241+`; `vsimulate`/`vgenerate` `[N]` axis `dynamic.cljs:1342`.
- **Layer 2 — the LLM-GF body: BLOCKS batching.** `make-llm-gf` `llm/core.cljs:60-65` calls
  `(mx/item tok)` per token for the EOS check + host control flow (`loop/recur`, EOS `if`) on the
  scalar. Docstring `core.cljs:43-45`: "Not compatible with vsimulate/vgenerate." Needs a
  `[K]`-shaped rewrite with a per-particle **active-mask** instead of `mx/item`.
- **Layer 3 — native forward + KV cache: batch-1 on the GFI path.** native `forward` `[1,seq,vocab]`
  / `forwardWithCache` `[1,1,vocab]` (`index.d.ts:1558-1567`); `ids->input` `[1 N]`
  `backend.cljs:299-311`; `forward-step` single scalar token `backend.cljs:391-406`; owned cljs
  forward `qwen35_forward.cljs:271` `[1 T hidden]`, recurrent/conv cache `[1 Hv Dv Dk]`
  `qwen35_forward.cljs:187`; `CljsForwardModel` cache atom `{:cache :offset}` no batch dim
  `backend.cljs:94`. (qwen3 vanilla forward `qwen3_forward.cljs` is `[seq]`/`[seq hidden]`.)
- **BUT a true native batched decoder EXISTS:** `generateBatch(prompts, groupSize, config) →
  BatchGenerationResult` ("sequential prefill + batched decode", `index.d.ts:2095`, `:411-425`),
  reachable via the GRPO engine (`world/train.cljs:395` `train/generate-batch`). **Probed
  standalone = STABLE.** Tonight's wedge is the *training* step (`generateBatchForTraining` +
  autograd), NOT the decode.

The oracle (use, do not rebuild): `genmlx.llm.msa-score/score-model*` → `{:log-ml :method}`
(exact analytical marginal for conjugate programs, IS otherwise); `genmlx.world.distill/evaluate-candidate`
(full gate ladder); `genmlx.codegen.eval/valid-cljs?` + `eval-fn` (edamame reader + SCI — this
is "use SCI to test valid cljs", and it works); `genmlx.world.train-reward/extract-program`
(isolates the FIRST form, dropping the native engine's no-EOS `\n!\n!` filler — IMPORTANT, the
native decode does not stop at EOS, so always isolate the first form before scoring).

Conditioning substrates (all compose via `dispatch/with-handler` middleware): token-DFA
`genmlx.llm.grammar`, byte-DFA `genmlx.llm.bytes`, reader-as-grammar `genmlx.llm.codegen`,
Instaparse grammar (`genmlx.llm.msa` knowledge mode; instaparse is on the nbb classpath).

---

## 6. ROUTE A — build plan (reliable, de-risked)

Goal: a standalone **best-of-K inference primitive** (NOT via training) + oracle selection + a
clean benchmark + the demo.

1. **Best-of-K generation primitive.** Add `genmlx.llm.particles` (or extend `llm/backend`):
   `(best-of-k-generate model-dir prompts {:k K :max-tokens N ...}) → {prompt → [k completions]}`.
   Implementation: reuse `genmlx.world.train` — `(.load (.-Qwen35Model gcore) model-dir)` then
   `with-trainer` with `{:group-size K :enable-thinking false :max-completion-length N
   :lm-head-chunk-size 2 :forward-chunk-size 4 :temperature 0.9}` and `train/generate-batch`
   (the STABLE native batched decode). Construct the engine ONCE, generate for all demo prompts,
   dispose via `with-trainer`. (This is exactly what `~/genmlx-loop-artifacts/probe.cljs` did and
   it returned cleanly.) Use the model `qwen3.5-0.8b-cljs-sft600`.
2. **Oracle scoring + selection.** For each completion: `reward/extract-program` (drop the
   no-EOS filler) → `score/score-model* gf observations` → evidence. Select argmax-evidence per
   task = best-of-K-by-evidence. (For function tasks, `distill/evaluate-candidate` + any-pass.)
3. **Clean benchmark** (model PRELOADED, identical max-tokens):
   - batched K-particle gen+score wall-clock (the primitive)
   - vs teacher single-shot wall-clock (load `Qwen3.6-35B-A3B-4bit`, 1 completion/task)
   - vs (optional) sequential K (for the batching speedup factor)
   - quality: best-of-K evidence vs teacher evidence on the 11 held-out PROGRAM tasks
     (`eval_prompts.jsonl`, kind=program), using `~/genmlx-loop-artifacts/` data where useful.
4. **The demo script** `scripts/particle_demo.cljs`: given a data-explanation task, print the
   teacher's single program + its evidence + time, then the 0.8B best-of-K program + its evidence
   + time, and the verdict (better fit, faster). Make it pretty.

Risk: Route A uses the GRPO engine purely for its batched decode. The probe proved the *decode*
is stable; do NOT call `train-step!`/`generateBatchForTraining` (that is what wedged). If the
engine construction itself is heavy, construct once and batch many prompts.

---

## 7. ROUTE B — build plan (GFI-native, stretch)

Goal: a `[K]`-particle LLM-GF that `vsimulate` runs in one native pass — fully in the GFI, so it
composes with grammar/SCI/instaparse conditioning and SMC.

1. **Batch-dim the owned cljs forward** (`src/genmlx/llm/qwen3_forward.cljs`,
   `qwen35_forward.cljs`): make the leading dim a variable `B`/`K` instead of literal 1.
   - qwen3 (vanilla attn): `[B,T,hidden]`, KV cache `[B,heads,T,dim]`.
   - qwen3.5 hybrid (GatedDeltaNet): the recurrent state `[B,Hv,Dv,Dk]` + conv window
     `[B,K-1,conv-dim]` — batch the leading dim. **This is the delicate part.**
   - Update `CljsForwardModel` cache to carry a batch dim; `ids->input` to `[B,T]`.
2. **Rewrite `make-llm-gf`** (`llm/core.cljs`): drop `(mx/item tok)`; keep tokens `[K]`-shaped;
   use a per-particle **active-mask** (`[K]` bool) to stop appending after each particle's EOS;
   no host scalar control flow. Generate to a fixed max or until the mask is all-false.
3. **GOLDEN-ORACLE GATE (mandatory):** a `[K]`-batched forward MUST equal single-stream outputs.
   Test: (a) K identical prompts → all K logits == the single-stream logits; (b) K different
   prompts → each == its own single-stream run. Reuse the f6ov golden-oracle test infra. Do NOT
   proceed to vsimulate until this passes.
4. **`vsimulate` the LLM-GF at `[K]`** → K token streams in one pass → oracle-score → select.
   Then demonstrate **in-generation conditioning** (grammar/reader) composed on top.

If the hybrid-cache batching can't be made correct reliably in this run, STOP Route B, document
exactly where it broke, and ship the Route-A demo. Route B correctness > Route B completeness.

---

## 8. THE DEMO (what to actually show)

A reproducible head-to-head on probabilistic-program synthesis:
- Pick 3–6 held-out PROGRAM tasks (e.g. from `eval_prompts.jsonl` kind=program: gaussian-mean /
  beta-bernoulli / gamma-poisson instances) and/or a fresh data-explanation task.
- For each: teacher single shot (program + evidence + time) vs 0.8B best-of-K (batched particles
  + evidence-selection: program + evidence + time).
- Headline table: **evidence (fit) and wall-clock, 0.8B-best-of-K vs teacher.** Plus show the
  selected 0.8B program is well-formed, adapted, idiomatic GenMLX cljs.
- State the honest scope: this wins on program/model synthesis (GenMLX's domain); general cljs is
  the teacher's.

**ALSO probe ADVANCED models (required — user ask).** Beyond the simple single-latent conjugate
programs, hand-craft 2–4 harder, oracle-scoreable GenMLX models and run the SAME best-of-K-vs-teacher
comparison: e.g. linear regression (coupled slope+intercept+noise — L3 joint linear-Gaussian
eliminator scores it exactly), a hierarchical/group-means model, a small mixture (GMM), and/or a
Kalman/HMM chain (L3.5). The 0.8B was SFT'd ONLY on simple single-latent programs, so expect
whole-program best-of-K to DEGRADE or fail on these — that is the POINT: measure where the
whole-program approach breaks as model complexity rises. Report the hit-rate curve vs complexity.
This is the empirical signal for whether advanced models need a stepwise/FIM strategy (§12), and
whether the teacher (or a FIM coder) is needed there. Be honest about the breakdown point.

---

## 9. REPRODUCTION (if an artifact is missing/stale)

```bash
cd ~/code/genmlx
# task set + prompts (fast, no GPU):
bun run --bun nbb scripts/gen_tasks.cljs --validate          # 183/183 grounded, 44/44 programs exact
bun run --bun nbb scripts/gen_tasks.cljs --export-eval  /tmp/eval_prompts.jsonl
bun run --bun nbb scripts/gen_tasks.cljs --export-train /tmp/train_prompts.jsonl
bun run --bun nbb scripts/gen_tasks.cljs --write-edn    /tmp/gen_tasks.edn
# corpus (EXPENSIVE, ~1h GPU; only if you must re-derive — else use ~/genmlx-loop-artifacts/corpus):
python3 scripts/distill_teacher.py --model Qwen3.6-35B-A3B-4bit --tasks /tmp/train_prompts.jsonl --out /tmp/teacher.jsonl --n 6
cat /tmp/teacher.jsonl scripts/... ; bun run --bun nbb scripts/distill_filter.cljs --gen --candidates ... --out ... --top-k 4
# eval grading uses the oracle sandbox:
bun run --bun nbb scripts/sft_eval.cljs --gen --baseline <b.jsonl> --sft <s.jsonl> --out <dir> --k 4
```

---

## 10. GOTCHAS (load-bearing)

- **Metal wedge is real.** The native GRPO *training* step wedged ~3.5 h tonight (uninterruptible-ish;
  reboot clears it). For long native jobs: run in the background, watch a flushed progress file, and
  set a hard timeout. NEVER run two GPU/Metal jobs concurrently. The batched *decode* (Route A) is
  stable; the *training* path is the suspect.
- **mlx-node-native ≠ Python mlx-lm.** "Native" = cljs/nbb → `@genmlx/core` (the Rust/NAPI addon) →
  Metal, one process, shared `MxArray` graph with the oracle. Python `mlx-lm` is a foreign process
  (file interface only) — it was scaffolding for the teacher (35B MoE has a deferred native SIGTRAP)
  and SFT (LoRA). The demo must be **mlx-node-native** (Route A's `generate-batch` is).
- **The native decode does not stop at EOS** — completions are `(form)` + `\n!\n!…` filler. ALWAYS
  isolate the first complete form (`reward/extract-program`, or `ce/parse-form` + `pr-str` to
  preserve names) before scoring. mlx-lm stops at EOS; the native engine does not.
- **Programs use a lenient "covering-model" gate** — the meaningful program metric is EVIDENCE
  (log-ML), not pass/fail. Rank/compare by evidence.
- 32 GB box: don't co-resident the 35B teacher (~20 GB) with training. Run benchmark phases
  sequentially: teacher pass, then student pass.
- Run tests with `bun run --bun nbb test/...` (never node-nbb). Verify `distill_gen_test` 18/18,
  `sft_test` 49/49, `distill_sandbox_test` 5/5 after any change to those modules.

---

## 11. GO / NO-GO (be honest in the final report)

After the clean Route-A benchmark on PROGRAM tasks:
- **If 0.8B best-of-K beats the teacher on BOTH fit (evidence) and speed → SHIP the particle
  engine; this is the win.** (Expected, per tonight's −4.10 vs −4.83 and ~9× cheaper/use.)
- **If it does NOT** (teacher's evidence ≥ ours, or batched particles are not faster) → say so
  plainly, and recommend planning around the qwen-3.6 teacher for this use case. Do not spin a loss
  as a win. Capture the numbers either way in `docs/cljs-coder-loop-results.md` and on genmlx-qsoa.

Suggested order: Route A primitive → clean benchmark → go/no-go call → demo script (incl. the
ADVANCED-model probe in §8) → (if green and time remains) Route B forward-batching with the
golden-oracle gate → update results doc + beans → write up §12 findings.

---

## 12. ADVANCED MODELS & the FIM / stepwise question (research direction — do NOT need to fully build this run)

User's framing (2026-06-22): simple programs are demo'd whole-program; **advanced GenMLX models
(hierarchical, coupled regression, mixtures, Kalman/HMM, combinator-structured) likely need a
STEPWISE construction strategy, and FIM (fill-in-the-middle) may be a good mechanism — possibly
requiring a FIM-capable qwen *coder* model.** This run's job is to PROBE and DOCUMENT, not to build
the full stepwise engine.

Why stepwise (the real lever): whole-program best-of-K hit-rate COLLAPSES as models grow (search
space explodes; a small model rarely emits a long correct program). Stepwise construction with
PER-STEP oracle scoring rescues it — model evidence on a PARTIAL model is still a real number, so
it gives intermediate reward to prune the search (SMC over construction steps, not whole programs).
GenMLX uniquely supplies that intermediate reward + the composition machinery (GFI update/regenerate/
edit/CompositeEdit; combinators Map/Unfold/Scan for repeated structure).

FIM's role + the resolved tension: FIM is a natural per-step mechanism — fix the interface
(prefix `(fn [trace] (let [` + suffix `] {observations}))`), fill the body (middle) — and it
composes with particles+oracle (K fills per hole → score → keep best). We DROPPED FIM earlier
(prior cljs fine-tunes failed), BUT that was FIM-for-WHOLE-generation on qwen2 (non-owned-forward).
KEY: the inference particle+oracle loop does NOT need owned-forward/differentiability (that was only
for GRPO weight updates) — so a FIM-capable qwen CODER can be used as a PURE PROPOSAL, scored by the
GenMLX oracle, sidestepping exactly what broke before. Trade-off: FIM-coder capability vs the cheap
local 0.8B. (Caveat: qwen2.5-coder is FIM-capable but qwen2 arch; check whether a qwen3-family /
Qwen3-Coder model offers FIM tokens. FIM tokens: `<|fim_prefix|> <|fim_suffix|> <|fim_middle|>`.)

Nuance: stepwise is the STRATEGY; FIM is ONE mechanism. Alternatives to validate empirically:
(a) instruct-driven stepwise ("here is the partial model, add the next latent" — no FIM tokens),
(b) grammar/SMC-guided structured generation fully inside the GFI (reader-as-grammar / instaparse +
incremental oracle scoring — no FIM model at all, composes natively). Pick by experiment.

Deliverable for THIS run re advanced models: the §8 advanced-model probe (the hit-rate-vs-complexity
curve) + a written note in `docs/cljs-coder-loop-results.md` on where whole-program breaks and which
stepwise mechanism looks most promising. Captured as a DRAFT bean for the full stepwise/FIM engine.
