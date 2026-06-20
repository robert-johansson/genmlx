# GenMLX ↔ mlx-node: Strategic Synthesis & Cutting-Edge Roadmap

> **Tracking anchor:** milestone [`genmlx-nv1t`] — *GenMLX × mlx-node deep integration*.
> Sub-beans: analysis epic `genmlx-jjqg`; Phase 0 `genmlx-zftr`; Phase 1 `genmlx-ugkv`;
> Phase 2 `genmlx-911t`; Phase 3 `genmlx-p1nz`; Phase 4 `genmlx-tuoo`; fork-minimization `genmlx-nldo`.
>
> **Provenance:** produced 2026-06-20 by a 6-analyst workflow over the GenMLX + mlx-node
> source trees (recorded under bean `genmlx-jjqg`), with the fork facts re-verified by hand
> (§1). Sections 2–7 are the synthesis as written; §1 and §8 carry the corrected fork facts.

## 1. Current setup (Q2) — the precise mental model

GenMLX vendors **two nested forks, and BOTH are add-only**. Three layers; two carry small, surgical, purely-additive modifications:

```
ml-explore/mlx (C++)         <- FORKED, add-only. robert-johansson/mlx, vendored as a
  pinned at 49503b65            nested submodule at mlx-node/crates/mlx-sys/mlx.
  9 commits / +1771 / -0       9 commits ON TOP of upstream merge-base 2e6632e5:
                               Cholesky JVP/VJP, Inverse JVP/VJP, lgamma, digamma,
                               bessel_i0e/i1e, searchsorted, num_resources/resource_limit
                               (Metal buffer-count), Metal pipeline release, CommandEncoder
                               compat. NOTE: `git status` is clean only because the patches
                               are COMMITTED into the pin -- check commit authorship.
        ^ vendored fork
mlx-node (Rust/NAPI + TS)    <- FORKED, add-only. robert-johansson/mlx-node.
  vs fork-point 35559218       TRUE delta vs the 2026-05-29 "update to latest mlx-node"
  (2026-05-29):                rebase point = +6417 / -1497, ZERO deleted files. Additions
  +6417 / -1497                isolated in NEW files (genmlx.rs, transforms.rs, memory_napi.rs,
                               mlx_{linalg,random,transforms}.cpp) + keyed PRNG +213 in
                               array/random.rs; wired by ONE line `pub mod genmlx;` at lib.rs:32.
        ^ file: dependency (built from the submodule, package.json:14-15, NOT npm)
GenMLX (ClojureScript)       <- consumes @mlx-node/core + @mlx-node/lm.
```

> **Correction note.** An earlier pass mis-reported MLX as "pristine-pinned" and the mlx-node
> fork as "+9.5k/-78.9k, mostly pruning." Both were artifacts: the first from reading `git status`
> (which only shows *uncommitted* edits) instead of the pinned commit's authorship; the second from
> diffing against *latest* `upstream/main`, which conflates upstream tree-drift as fake deletions.
> Diffed against the actual fork-point merge-base, **both forks are add-only.**

**What the mlx-node fork adds.**
- **`crates/mlx-core/src/genmlx.rs`** -- fork-only file, **85 `#[napi]` exports**. ~68 are plain re-exports of stock MLX ops (`exp`, `matmul`, `cholesky`, `eigh`, `svd`, `einsum`, `take_along_axis`, ...); their only "GenMLX-ness" is the `Either<&MxArray,f64>` / `Vec<f64>` coercion that keeps `mlx.cljs` thin. (genmlx.rs:50-504)
- **Keyed/splittable PRNG** -- `randomKey`/`randomSplit`/`keyNormal`/`keyCategorical`/... The mlx-sys FFI layer already exposes `mlx_random_*_key` (lib.rs:587-620), i.e. **MLX C++ has native keyed random** -- our addition is the *binding layer* (mlx-sys C++ shim + mlx-core wrapper + NAPI), NOT an MLX-core patch. This is the GFI's single entropy-injection point (`rng/fresh-key`).
- **`transforms.rs`** (+361) -- `vmap` / `compileFn` over MLX's C transforms via JS-callback FFI.
- **`autograd.rs`** NAPI exports (+141) -- `valueAndGrad` / `computeGradients` taking a JS loss closure.
- **f6ov fused-NN exports** -- `rmsNorm`, `rope`, `scaledDotProductAttention` (explicit-mask), `silu`, `loadSafetensors` -- GenMLX's owned LLM forward.

**What the MLX fork adds (and why it can't trivially leave MLX).** The 9 commits register C++ primitives + autograd (vjp/jvp) rules *inside* MLX's primitive system. lgamma/digamma/bessel power gamma/beta/Dirichlet log-prob gradients; Cholesky/Inverse JVP-VJP power differentiable MVN/Kalman. These are upstreamable to ml-explore/mlx (or reimplementable via MLX `custom_vjp`) -- the path to a zero-MLX-fork future (see §8).

**The membrane.** `src/genmlx/mlx.cljs` does `(defonce c (js/require "@mlx-node/core"))` once and binds ops as direct property references (`(def matmul (.-matmul c))`, mlx.cljs:261). The coverage matrix (`docs/membrane-coverage.md`, `membrane_coverage_test.cljs`) machine-pins **212 exports = 164 wrapped ⊎ 48 intentionally omitted**. The 48 omissions are where the entire untapped training/serving stack sits.

---

## 2. Unresolved issues (Q1) — distilled bean table, grouped by theme

| Theme | Beans | Status / impact |
|---|---|---|
| **Upstream model/loader gaps** (block real checkpoints) | `o94r` sharded-safetensors (any HF checkpoint >~3GB fails `loadModel`), `k199` port `qwen3_moe`, `gvmy` `qwen3_next` + CUDA/Jetson retarget (~2-4 person-months, off critical path) | These are genuine *feature* gaps in the fork's loader — they gate which models you can load at all. `o94r` is the highest-leverage: it blocks every dense Qwen3/3.5/Gemma ≥3GB. |
| **Native forward crashes** (mitigated, fallback-only) | `7siy` Qwen3.5 prefill null-ptr (M2 Max stream bug — passes on this M4; **mitigated**, owned forward is default), `s8fi` `qwen3_5_moe` 256-expert `gather_mm` SIGTRAP (guarded off), `52lf`/`uteq` unguarded LLM-path shims + verify | All on the *borrowed* `.forward` fallback path. f6ov made the GenMLX-owned forward the default, so these are no longer user-reachable by default. Real fixes are upstream Rust; need M2 Max / 19GB repro this box can't do. |
| **Membrane / memory ergonomics** | `1pi9` dispose-trace no-op (downgraded high→normal: crash gone, ergonomics remain), `eulz` lazy-load membrane (DEFER — broad churn, no consumer), `z9li` out-of-tree `-cp` breaks `setCacheLimit` | `5413` (edit decouple) **COMPLETED**. The crash cluster was one root cause (Metal buffer-COUNT, not bytes) — fixed; only control/ergonomics linger. |
| **Packaging** | `w837` pin/drop `@mlx-node/core` optionalDependency (stale-prebuilt trap) | In-repo runtime guard (`91b3`) already ships; remaining fix is in the submodule repo, out of scope here. |
| **Ergonomics features** | `wxvk` MxArray `IEquiv`/`IComparable` for scalars | Nice-to-have, has a v1.0 spec, not critical. |

**Key lessons from closed beans (Report 6):** a stricter NAPI binary (v0.31.2) turns latent *CLJS shaping* bugs into hard SIGTRAPs — suspect the membrane, not the binary (`d30o`, `yo6y`). Gitignored `dist/` and stale `.node` prebuilts repeatedly masked source regressions (`mwm4`, `d30o`). Test runs can wedge in uninterruptible Metal U-state — reboot-only; always `bun run --bun nbb`, never parallel GPU.

**Net:** no open bean blocks the *deep-integration* roadmap below except where noted (`o94r` gates anything loading large checkpoints for training). The training stack is unwired by *choice*, not by bug.

---

## 3. Long-term viability + stock-mlx-node verdict (Q3)

**Position: Keep a minimal fork (Path C) now; pursue upstreaming buckets (a)+(b) as the aspirational endgame (Path A). Pure stock (Path B) is infeasible.**

The hard blocker is the **keyed/splittable PRNG**. Upstream `array/random.rs` has *zero* `key_*` functions and only stateful global-RNG ops. GenMLX's entire entropy model — `rng/fresh-key`, split-key-per-sample, reproducibility — is built on it, and a counter-based splittable PRNG **cannot be composed from stock ops in CLJS** without losing reproducibility and performance; it must be native (Report 1 §4). So "off-the-shelf mlx-node only" is not achievable without this one ~213-line native addition.

The three realistic options and their tradeoffs:

- **Path A — Upstream genmlx.rs + key-PRNG + transforms.** Cleanest endgame. The ~68 plain-reexport ops are trivially acceptable; the keyed PRNG is a legitimate JAX/GenJAX-parity contribution. **Technically high feasibility, politically uncertain** — mlx-node is an LLM inference/training framework, not a PPL substrate, and the `Either<&MxArray,f64>` convention conflicts with their `BigInt64Array` style (index.d.cts:3570). **Lost: nothing.**
- **Path B — Reimplement genmlx.rs exports in CLJS over stock ops.** The 68 math/linalg ops survive (stock exposes the `MxArray` instance methods), and `valueAndGrad`/`vmap`/`compileFn` survive *if* the autograd-export commits upstream. **But the keyed PRNG cannot be reconstructed, native fused LLM forward (f6ov) is lost, and native value_and_grad NAPI exposure is lost.** **Not viable.**
- **Path C — Minimal fork (RECOMMENDED, ≈ current state).** Reduce the fork to its irreducible native surface: key-PRNG (~213 lines) + transforms.rs (~361) + autograd NAPI exports (~140) + fused-NN (~70) ≈ **under 800 lines of genuinely fork-only capability**. The other ~600 lines of genmlx.rs are convenience re-exports — cheap to keep, droppable if upstream ever exposes equivalents. **Lost: nothing; cost = a rebase burden** (already visible in the post-MLX-v0.31.2 C-API-drift adaptation commit `f095cfa`).

**Recommendation:** Stay on Path C. The fork's irreducible core is small and stable. **But begin Path A opportunistically** by upstreaming the pure *bugfixes* immediately (§4) — those shrink your rebase burden regardless of whether the feature exports are ever accepted, and they're unambiguous wins for all mlx-node consumers.

---

## 4. Upstream contribution plan (Q4) — ranked PR candidates

These fork commits touch **upstream-shared code** (not GenMLX-specific glue) and fix real bugs — strongest acceptance odds, and each one upstreamed is one less rebase conflict:

**Tier 1 — Robustness/correctness bugfixes (PR these now):**
1. **FFI exception-guard family** (`4debd39` +310, `f92c9c1` +102, `8f4c8f6` +55) — MLX C++ exceptions (Metal buffer-limit, shape errors) currently **abort the Node process** instead of becoming catchable JS errors. A correctness bug for *any* consumer. Strongest candidate. (genmlx-5ucd)
2. **Autograd error-surfacing** (`a7e5f04` +8/−3) — `compute_gradients` returning generic "returned 0" instead of the recorded MLX message (autograd.rs:279-288). Small, pairs with #1.
3. **Allocation-throw guarding** (`2d57cc7` +214) — catch MLX OOM/alloc throws → catchable errors. Same class as #1.
4. **`copy_to_buffer`: single eval instead of double eval** (`81f43c0` −51/+17) — a genuine perf bug, benefits everyone.
5. **Compile multi-output callback fix + remove eval inside autograd** (`0b15830`) — correctness fix in shared compile/autograd plumbing.
6. **`stop_gradient` on PRNG results to fix RandomBits VJP** (`e9b6804`/`0bb9bef`) — a real MLX-autograd-vs-RNG interaction bug; valuable upstream though GenMLX-motivated.
7. **Qwen2 attention bias + non-fused path** (`2007ffe`) and **qwen2 model_type for Qwen2.5-Coder** (`3c9d7ee`) — straightforward model-support fixes, directly on upstream's mission.

**Tier 2 — Feature exports (offer as opt-in, lower odds):** the keyed/splittable PRNG family (`502565c`) and the `Either`-coercing math/linalg module-level exports (`5e3513e`, `ed16c60`). Frame as JAX/GenJAX parity.

**Do NOT PR:** pruning/rebase commits, the `genmlx.rs` ergonomic shims (`scalar`/`fromFloat32`), `loadSafetensors`, and the `Either`/`Vec<f64>` convention — these encode the CLJS-membrane design, not upstream-general need.

**Also worth filing as upstream issues/PRs (from beans):** `o94r` sharded-safetensors loader (the native impl already supports it — just lift the index.json pre-check into the shared loader), and `7siy` Fix A (`Stream::new(Gpu)`→`Stream::default(Gpu)` at 6 model.rs sites) once an M2 Max can verify.

---

## 5. Capability surface & the GRPO answer (Q5)

**Have we started GRPO? No. Not at all.** Verified by grep over `src/`: **zero** references to `@mlx-node/trl`, `GrpoTrainingEngine`, `SftTrainingEngine`, `train_step`, `applyGradients`, `OutputStore`, or any native optimizer. The only "training" matches in GenMLX are doc comments. GenMLX consumes mlx-node through exactly **two** packages (`@mlx-node/core` + `@mlx-node/lm`); `trl`, `server`, `vlm`, `privacy`, `cli` are built in the submodule but **never imported**. (Reports 2 §7, 3 §3, 5 §3)

**The untapped stack — built, NAPI-exposed, tested, dormant** (Report 2):

- **GRPO training engine** (`grpo/engine.rs`, `GRPOTrainingEngine`, 1500+ LOC) — a faithful, numerically-hardened Rust port of HuggingFace TRL's `grpo_trainer.py`. Full pipeline `generate→reward→advantage→clipped-surrogate→AdamW`, exposed as `#[napi]` with async `train_step_auto`. The algorithm is complete: group-relative advantages (zero-mean per prompt group, `advantages.rs:33`), the PPO clipped surrogate with token/sequence importance sampling and grpo/bnpo/dr_grpo/dapo aggregation (`loss.rs:120-339`), bf16-native autograd with vocab-chunked `efficient_selective_log_softmax` to avoid the `[B,T,151936]` ~1.2GB tensor (`autograd.rs`). **One gap: KL-to-reference under autograd is rejected (`beta>0` rejected at `autograd.rs:76`)** — so it runs KL-free unless you drive `grpo_loss` directly.
- **SFT engine** (`sft/engine.rs`) — cross-entropy on completions, next-token shift, `ignore_index=-100`, same AdamW machinery. Mature, NAPI-exposed.
- **Native optimizers** (`optimizers/`) — stateful, *resumable* AdamW/Adam/SGD/RMSprop with moment save/load to safetensors, global-norm grad clip, LR schedulers. (GenMLX's three CLJS optimizers in `nn.cljs`/`learning.cljs` are *functional* and parallel — a deliberate value-semantics choice that *conflicts* with native in-place mutation; see §6.)
- **Reward registry + JS callback** (`grpo/rewards.rs`) — `NativeRewardRegistry` built-ins plus a **`ThreadsafeFunction<String, Promise<Vec<f64>>>`** reward callback: completions go out as JSON, a `Promise<number[]>` of rewards comes back (`engine.rs:905-914`). **This is the single most important seam in the whole system** (§6).
- **Tool-call/thinking parsing** (`tools/mod.rs`, 7 napi) — structured tool-call + reasoning-span extraction. High agents relevance.
- **PagedAttention** (`mlx-paged-attn` crate) — vLLM-style block KV alloc, COW, prefix cache, continuous batching. Relevant for high-throughput batched LLM-as-GF.
- **Persistence** (`mlx-db` + `OutputStore`/`ResponseStore`) — SQLite training-run/replay-buffer substrate.
- **Server** (`packages/server`) — OpenAI `/v1/responses` + Anthropic `/v1/messages` compatible.
- **TRL JS drivers** (`packages/trl`) — `GRPOTrainer`/`SFTTrainer` with `trainStep(prompts, rewards)` taking **externally-computed rewards** (`grpo-trainer.ts:38,1313`), dataset loaders, loggers.

---

## 6. THE CUTTING-EDGE ROADMAP (Q6) — the centerpiece

The strategic insight unifying everything: **GenMLX produces, for free, exactly the scalars that policy-gradient RL wants as rewards.** The LLM is already a generative function (each token a trace site `:tN ~ Categorical(logits)`, `core.cljs:26-67`). For any generated program, GenMLX already computes `p/generate` marginal log-ML (exact for conjugates via L3, IS otherwise), SCI run-or-throw verification, and `inspect/inspect` compilation tier. A GRPO reward is just `(completion) → scalar`. **These GFI quantities *are* that function.** No existing PPL closes the loop between *Bayesian inference quality* and *LLM weight updates* — that intersection is the novel contribution.

The new surface this needs is **one third world-membrane face — `genmlx.world.train`** — sibling to `genmlx.mlx` (compute), `genmlx.world.net` (network), `genmlx.world.proc` (scheduler). It binds the native `GrpoTrainingEngine`/`SftTrainingEngine` behind a try/finally mutable-state quarantine (the same discipline as the KV-cache atom in `llm/backend.cljs:34`). **Training is RL's `eval!`-equivalent — the sole training side effect; everything above it stays pure** (reward = pure `trace→scalar`, rollout population = pure steppable, allocation policy = pure GF). (Report 5)

### Phase 0 (PREREQUISITE) — `genmlx.world.train` membrane face + sharded loader
- **Unlocks:** the binding seam for everything below.
- **Lands:** new Layer-0 `world/train.cljs`; new coverage-matrix entries moving `GrpoTrainingEngine`/`SftTrainingEngine`/`NativeRewardRegistry`/`Gradients`/`OutputStore` from `:training-orchestration` omission → wrapped (bean genmlx-706r names this exactly, `membrane-coverage.md:84-86`).
- **New surface:** the mutable-state quarantine; add `@mlx-node/trl` to `package.json` (currently only core+lm).
- **Gated by:** **`o94r` (sharded safetensors)** — training needs to load real checkpoints, most of which are >3GB and sharded. This is the one open bean that genuinely blocks the roadmap. Fix it first.

### Phase 1 — GRPO/SFT fine-tuning where the REWARD is a GFI quantity ★ HIGHEST LEVERAGE
- **Capability:** the self-training-coder/self-improving-modeler flywheel. The LLM-GF generates a probabilistic program → SCI-eval → score by `p/generate` log-ML (+ bonus when `inspect` reports L3/L4 compilation, − complexity penalty, − λ·KL-to-base) → GRPO updates the policy weights.
- **Lands:** Layer 9 `genmlx.llm.trl` (or `control.self_train`). The reward is a pure CLJS `(fn [trace] → scalar)` built from **already-existing scorers**: `msa/score-model` exact-or-IS log-evidence (`msa.cljs:495-508`) and `codegen/verify-transition-fn` `:accuracy` + `score-structure` idiomaticity (`codegen.cljs:304-336,455-468`) — both already return `number[]`-shaped values. Registered into `NativeRewardRegistry`; native engine owns rollout-group + advantage + autograd + weight update.
- **New surface:** a thin CLJS bridge from the `ThreadsafeFunction` reward callback to the GFI scorers. **No new Rust, no new autograd plumbing** — `compute_advantages`/`grpo_loss` take rewards as inputs (Report 4 §1).
- **Why cutting-edge:** This is the first system where *Bayesian model evidence* (exact marginal likelihood) is the RL reward signal for an LLM. It operationalizes "the LLM learns to write better probabilistic programs, judged by how well those programs explain data" — wake-phase of a Helmholtz/wake-sleep loop at the level of *program space*, not parameter space. The precedent already exists in miniature: `agents/differentiable.cljs` differentiates through a planner into a policy, and `inference/adev.cljs:98` implements the REINFORCE surrogate `cost + stop_gradient(cost−baseline)·reinforce-lp` — *the same estimator family GRPO uses* (group-mean as baseline). GRPO is the scaled, weight-updating version of a pattern already in the stack. (Report 5 Seam A, Seam C)
- **Honest caveat:** the one autograd gap (KL-to-ref `beta>0` rejected) means initial runs are KL-free unless you drive `grpo_loss` directly. Wiring ref-model logprobs through autograd is a Phase-1.5 follow-up — and a clean upstream contribution.

### Phase 2 — Steppable/budgeted inference as the RL rollout substrate ★ DEEPEST / MOST DISTINCTIVE
- **Capability:** GRPO is structurally *sample-a-group → score-each → group-relative-advantage → update*. GenMLX **already owns** "sample a population and score each by downstream value" — that is `control/synth_steppable.cljs` (a candidate-program pool, each scored by exact/IS evidence) driven by `world/proc.cljs::with-deadline`. The mapping is near-isomorphic: GRPO rollout group ↔ the `:pool`; per-rollout reward ↔ `score-candidate` log-ML; advantage ↔ group-normalize over `:log-ml`.
- **The novel piece:** the **VOC controller becomes a resource-rational rollout allocator.** Instead of a fixed GRPO group size × fixed scoring depth, the metareasoner (`control/meta_mdp.cljs::controlled-steppable-k`, already built with per-action metered trial-advance + EU-vector softmax) decides *how many completions to sample × how deeply to score each × when to stop the group*, under the honest GPU-cost meter (`inference/cost.cljs`, reads `mx/read-cost-counters`). This is value-of-computation over *training compute*, not just inference compute.
- **Lands:** Layer 7/control — the substrate (`steppable.cljs`/`synth_steppable.cljs`) is **already built and contract-pinned** (`control/CONTRACTS.md`); the reward stays *downstream* (`decision_value.cljs::assert-downstream!` enforces "never reward the sampler's own ESS/log-ML diagnostic").
- **Why cutting-edge:** No RL trainer adaptively allocates rollout-group compute by value-of-information. `rrps-design.md §8` names the citable unification — the twisted-SMC twist function (Zhao 2024) ≈ the soft-RL value function ≈ what the VOC controller estimates over the synthesis loop. This is the genuinely new intersection of *anytime/bounded-rational metareasoning* with *RLHF*.
- **Honest caveat:** host-serial, not GPU-parallel across distinct programs — distinct candidates have distinct trace addresses and can't share a batch axis; the single Metal GPU time-slices (`rrps-design.md §2`). The win is *adaptive allocation*, not throughput — which is precisely why the control axis exists.

### Phase 3 — Agents/control axis as RL policies + amortized/variational inference with native optimizers
- **Capability (3a):** an agent *is* a GF and its policy is `Categorical(softmax(α·EU))` (`agents/agent.cljs:189`). With the native engine, agent/control policies become RL-trainable at scale — close the inverse-planning loop (`agents/differentiable.cljs::recover-params` already does Adam through the planner) into a full GRPO loop for LLM-backed agent policies.
- **Capability (3b):** amortized/variational inference (`inference/amortized.cljs` encoder, `inference/vi.cljs`) trained with the **native stateful, resumable AdamW** instead of the three parallel CLJS optimizers — checkpointable recognition networks, moment save/load.
- **Lands:** Layer 6/8, driven through `world.train`.
- **Tension to resolve:** GenMLX's CLJS optimizers are *functional* (immutable params threaded, or audited atom); native optimizers mutate in place. They **cannot share param maps** — native training must stay quarantined behind `world.train`, a *parallel* path, not a composition at the gradient level (Report 3 §4). Pick native for full-model LLM weight updates; keep CLJS-functional for small tensor policies and GFI-score gradients.

### Phase 4 — Scale: PagedAttention + model_thread for batched LLM-as-GF, then server
- **Capability:** GenMLX's KV cache is a single per-model atom; `core.cljs:43-45` and `vision.cljs:100-104` explicitly forbid concurrent execution on one model. PagedAttention's `layer_kv_pool` + `model_thread`'s batched dispatch (the same machinery `train_step_auto` uses for group generation) would enable **batched multi-particle/multi-prompt LLM-GF execution** — directly relieving `msa/importance-sample`'s serial `p/generate` loop (`msa.cljs:558-587`).
- **Lands:** a new backend forward variant driving `model_thread` batched prefill, returning `[N,vocab]` logits to feed the existing shape-based batched handler. The owned forward's value-level structure (genmlx.rs ops, not opaque structs) makes a `[N]`-batched forward tractable.
- **Then `@mlx-node/server`** — expose grammar-constrained/GFI-scored generation behind `/v1/messages` (composes with the existing `world.net` membrane). **Lowest leverage** — pure deployment plumbing, no GFI-semantics gain.
- **Why later:** larger lift, and the host-serial caveat from Phase 2 limits the gain across *distinct* programs; the win is batched *identical-structure* particles.

**Leverage ranking:** Phase 1 (GFI-reward GRPO — near-zero impedance, all pieces exist) ≫ Phase 2 (VOC rollout allocator — the distinctive research contribution) > Phase 3 (agent-RL + native-optimizer VI) > Phase 4 (scale/serving). **Dependency:** Phase 0 (esp. `o94r`) gates all; Phases 1–2 are the cutting edge and can proceed in parallel once Phase 0 lands.

---

## 7. Recommended next 3 concrete steps

1. **Upstream the Tier-1 robustness bugfixes now (§4 #1–#4).** The FFI exception-guard family + autograd error-surfacing + `copy_to_buffer` single-eval are unambiguous wins for all mlx-node consumers, shrink your rebase burden permanently, and require no decision about the PPL-specific surface. This is the cheapest, highest-certainty action and validates the upstream relationship before you bet the roadmap on it.

2. **Fix `o94r` (sharded safetensors loader) — the one open bean gating the roadmap.** The native impl already supports sharded loads; lift the `model.safetensors.index.json` → shard-list pre-check into the shared loader. Without this, training/fine-tuning can't load real checkpoints (most are >3GB). File it upstream too.

3. **Build the Phase-0 `genmlx.world.train` membrane face + a Phase-1 spike.** Add `@mlx-node/trl` to `package.json`; bind `GrpoTrainingEngine` behind the try/finally mutable-state quarantine; wire a single reward callback to the *already-existing* `msa/score-model` log-ML scorer (`msa.cljs:495-508`). A one-prompt `train_step_auto` round-trip with a GFI-log-ML reward proves the entire thesis (inference-quality → LLM weight update) end-to-end with no new Rust and no new autograd code. Per the project's milestone protocol, **write the `world.train` spec first and stop for review** before implementing — this is a new membrane face and a new mutable boundary, exactly the kind of surface that warrants spec-first approval.

**Relevant files:** `mlx-node/crates/mlx-core/src/genmlx.rs`, `.../array/random.rs`, `.../transforms.rs`, `.../autograd.rs`, `.../grpo/{engine,loss,advantages,autograd,rewards}.rs`, `.../sft/{engine,autograd,loss}.rs`; `mlx-node/packages/trl/src/trainers/grpo-trainer.ts`; GenMLX `src/genmlx/mlx.cljs`, `src/genmlx/mlx/random.cljs`, `src/genmlx/llm/{core,backend,codegen,msa}.cljs`, `src/genmlx/control/{meta_mdp,synth_steppable,decision_value}.cljs`, `src/genmlx/inference/{steppable,cost,adev,amortized}.cljs`, `src/genmlx/agents/{agent,differentiable}.cljs`, `src/genmlx/world/proc.cljs`; `docs/membrane-coverage.md`, `docs/rrps-{design,dreaming}.md`; beans `genmlx-706r` (the binding seam), `genmlx-o94r` (roadmap gate), `genmlx-7siy`/`s8fi` (mitigated forward crashes).


---

## 8. Fork-minimization track (added 2026-06-20) — bean `genmlx-nldo`

Goal: ride as close to **stock mlx-node** as possible while keeping the PPL-specific code surgical and ours. Both forks are already add-only (§1); the strategy makes them *removable*:

**Layer 1 — MLX fork -> upstream it to zero.** The 9 add-only commits are individually general. lgamma/digamma/bessel/searchsorted are trivially upstreamable; Cholesky + Inverse JVP/VJP are desirable autograd coverage; num_resources/resource_limit + Metal-pipeline-release are reasonable. If accepted, the MLX fork disappears (back to pinning stock MLX). Fallback: reimplement the grads via MLX `custom_vjp` inside our own crate.

**Layer 2 — the separate-crate idea: build `@genmlx/core` as a SUPERSET addon.** A new NAPI crate that depends on **stock `mlx-core`** and *adds* the genmlx surface (the 85 exports + keyed PRNG + transforms + autograd exports), emitting a **single** `.node` addon. Then **mlx-node becomes an unforked dependency** and all "being a PPL" Rust lives in one crate we own. Feasibility is high — the keyed PRNG only needs MLX's native keyed random (already FFI-exposed), and the genmlx modules are near free-standing (`pub mod genmlx;`, one line).

> **Hard constraint to validate first (the de-risking spike):** a *single MLX runtime in a single addon*. Two separate addons each static-linking MLX would give each its own NAPI `MxArray` type registry and its own Metal memory pool, so an array from `@mlx-node/core` could not be used by `@genmlx/core` — breaking the membrane. Therefore `@genmlx/core` must be a **superset addon** (depends on `mlx-core` as a crate, emits one combined addon), *not* a second addon beside it. The spike: confirm napi-rs collects `#[napi]` symbols across the crate boundary into one module init. This decides **superset-addon vs. keep-the-minimal-fork**.

**Layer 3 — upstream the pure bugfixes now regardless** (FFI exception guards `4debd39`/`f92c9c1`/`8f4c8f6`, autograd error-surfacing `a7e5f04`, alloc-throw guard `2d57cc7`, `copy_to_buffer` single-eval `81f43c0`). These shrink rebase burden no matter which path wins and are wins for every mlx-node consumer (see §4).

**Target end-state:** stock MLX + stock mlx-node + one `@genmlx/core` addon we own.
