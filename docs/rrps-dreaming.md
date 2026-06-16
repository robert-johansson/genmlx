# Dreaming Extends RRPS — Thor + Qwen3-Coder-Next + a Self-Improving Homoiconic Synthesizer

**Status:** research design / mostly unbuilt. Companion to [`rrps-design.md`](./rrps-design.md)
(which specifies the resource-rational program-synthesis **wake** phase). This document covers
(a) the **Jetson AGX Thor 128GB + Qwen3-Coder-Next + modified-mlx-node** substrate and
(b) how DreamCoder-style **Dreaming** (wake → sleep-abstraction → sleep-dream) takes
resource-rational program synthesis (RRPS) to the next level.

**Provenance:** 17-agent research + design + adversarial-critique workflow (run `w178w7s6m`,
2026-06-16; ~1.7M tokens). External facts sourced to upstream (NVIDIA datasheets, ml-explore/mlx
PRs/issues, the Qwen3-Coder-Next HF card + tokenizer, DreamCoder/Stitch/LILO/twisted-SMC papers);
GenMLX claims cited to source `file:line` and verified against `de0aca2`/`858ed05`.

> **Thesis.** GenMLX becomes a self-hosting *"lisp machine that dreams in its own language,"*
> where **one homoiconic GFI value-algebra** closes the entire learn-to-synthesize loop on **one
> coherent-memory box**: a resident code LLM proposes ClojureScript `(gen …)` programs, GenMLX
> SCI-evals them in-process into real generative functions, scores them on the same GPU by **exact
> marginal evidence**, a value-of-computation controller allocates compute resource-rationally
> (**wake**), a library learner compresses the homoiconic corpus into reusable GF abstractions
> (**abstract-sleep**), and a recognition model is trained on exact-evidence-labeled fantasies
> sampled from the growing library (**dream-sleep**) — with the *same* controller folded over the
> wake/abstract/dream budget itself.

---

## Part I — The substrate

### I.1 Jetson AGX Thor 128GB: a coherent unified-memory CUDA machine

| Property | Value (sourced) |
|---|---|
| GPU | Blackwell, **2560 CUDA cores + 96 5th-gen Tensor Cores** (Transformer Engine, MIG), ~1.57 GHz |
| CPU | **14-core Arm Neoverse-V3AE**, ~2.6 GHz — same SoC as the GPU |
| Memory | **128 GB LPDDR5X**, 256-bit bus, **273 GB/s** |
| Unified memory | **YES — one physical pool; CUDA 13.0 full UVM coherence** (GPU reads host page tables; zero `cudaMemcpy` for mmap/malloc buffers) |
| Compute capability | **`sm_110`** (CUDA 13.0; was `sm_101` in 12.8/12.9) |
| Platform | JetPack 7, Ubuntu 24.04, Linux 6.8, CUDA 13.0; 40–130 W; $3,499 dev kit |

**The load-bearing fact (verdict: the user's intuition is RIGHT, and is the strongest single
fact in the whole vision).** Thor's CUDA-13 hardware-coherent UVM over pageable host memory is
**architecturally near-identical to Apple Silicon** — one coherent physical pool, real zero-copy
CPU↔GPU. This is the *literal* design rationale of MLX's CUDA backend (Awni Hannun: *"MLX is
designed around Apple silicon — unified memory architecture — no need to move arrays around from
CPU to GPU memory"*; the Apple-sponsored PR #1983 pitch: *"develop on Mac, deploy on NVIDIA"*).
A CUDA-retargeted mlx-node therefore **inherits GenMLX's Layer-B purity** ("the lazy graph is a
value; `mx/eval!` is the sole side effect") instead of fighting a discrete-GPU page-migration model.

**Honest calibrations.**
- **Bandwidth, not throughput.** 273 GB/s is **M-Pro-class**, not M-Ultra (~800) or HBM (multi-TB/s).
  LLM decode is bandwidth-bound → Thor wins on **capacity + zero-copy + power**, not speed.
- **`sm_110` is the first wall.** It is **not** in MLX's enumerated `CUDA_ARCHITECTURES`
  (90/100/121) → needs a gencode patch. No published MLX-on-Thor benchmark exists.

### I.2 MLX CUDA backend: real fit, beta gaps

MLX ships a CUDA backend (`pip install 'mlx[cuda]'`, merged from PRs #1983/#2075). It uses
`cudaMallocManaged` (managed/unified memory) behind a `supports_managed_memory()` capability check
with tiered pools — exactly the regime that pays off on a fully-coherent device like Thor.

**Beta gaps that bite this project (sourced, ml-explore/mlx #2422):** the CUDA backend currently
**lacks quantized matmul, the MoE gather-matmul, FFT, and LAPACK**, and **crashes on missing ops
with no CPU fallback**. The **MoE gather gap directly blocks running a 512-expert MoE on MLX-CUDA**
until it lands upstream — a hard dependency, not a nuisance. Every Cholesky/QR/SVD in the MVN/Kalman
L3.5 paths must be explicitly routed to `mx.cpu`.

### I.3 Qwen3-Coder-Next: feasible, a great MoE-on-unified-memory fit, "a bit" is optimistic

From the HF card + the Qwen3-Next base card: **`qwen3_next` (NOT plain `qwen3_moe`)** —
**80B total / 3B active**, MoE with **512 experts (10 active + 1 shared)**, hybrid 48-layer schedule
`12 × (3×(Gated DeltaNet → MoE) → 1×(Gated Attention → MoE))`, hidden 2048, **FIM-native** (verified
sentinels `fim_prefix 151659 / fim_middle 151660 / fim_suffix 151661 / fim_pad`), **256K** native
context. The MLX 4-bit conversion is **~44.8 GB** — fits Thor's 128GB **with vast headroom** for KV
cache + the host program pool + LoRA adapters. A huge-total / tiny-active MoE is the *ideal*
unified-memory workload: hold all 512 experts resident, compute only ~3B per token.

**Feasibility (verdict: feasible, rides existing kernels, materially more than "a bit").**
- ✅ The math is proven: mlx-lm already runs Qwen3-Next (`mlx-community/Qwen3-Next-80B-A3B-…-4bit`).
- ✅ The native scaffold exists in mlx-node: `AttentionType::{Linear,Full}`, `sparse_moe` with
  `SwitchGLU` + `gather_qmm`. So porting `qwen3_next` is **registration + config-shape +
  weight-key-mapping**, *not* new kernels.
- ❌ **Immediate blocker:** `SUPPORTED_MODEL_TYPES` lacks `'qwen3_next'` (verified gap). **Bean
  `k199` is mislabeled** — it targets plain `qwen3_moe`; a *new* `qwen3_next` bean is needed.
- ⚠️ The fp32 Gated-DeltaNet recurrent-state **precision landmine** (ollama #15865), the
  fused-vs-split `in_proj_qkvz` projection layout, the **sharded safetensors loader (bean `o94r`)**,
  and golden-oracle parity make it a **multi-bean port — weeks, not days**, on top of the CUDA retarget.

**Gated DeltaNet** carries a **recurrent matrix-valued state** (a scan, not stateless softmax
attention). This *composes* with GenMLX's "each token is a trace site" DynamicGF wrapping — the
host-coupled per-token decode loop (`core.cljs:61`) already serializes generation, so threading the
recurrent state fits the existing loop, just heavier per step.

**Bottom line for the substrate:** the full CUDA-retarget + `qwen3_next` port + wrapped MoE-LoRA is a
**~2–4 person-month systems track on unsupported hardware** (sm_110 unenumerated; MLX-CUDA beta
missing MoE-gather; linalg-to-CPU audit). **Memory is not the blocker; op-coverage is.** This track
should be **decoupled from the scientific claim** (below), which is demonstrable on Apple Silicon today.

---

## Part II — How the substrate realizes RRPS

On **one coherent box**, the loop runs with zero host↔device copies:

```
Qwen3-Coder-Next  --(GPU logits)-->  (gen …) ClojureScript form        [proposal]
        |  msa.cljs:104 code->source-form  →  dynamic.cljs:1210 make-gen-fn
        v          (schema.cljs + conjugacy.cljs classify cost/quality CLASS at ZERO GPU)
   real DynamicGF  --(same GPU)-->  score-exact (msa.cljs:437) = exact marginal log p(obs)  [reward]
        |                            (7 conjugate families, cross-checked ~5e-7 vs nn-marginal-closed)
        v
   VOC controller  allocates  #proposals × scoring-depth × stop                              [wake]
```

Proposal logits (GPU), SCI eval (host), and exact-evidence scoring (GPU) **all share one memory
pool**. See [`rrps-design.md`](./rrps-design.md) for the full wake-phase architecture, the
demonstration experiment, and the honest P0 gate.

**Why "ClojureScript generating ClojureScript" is load-bearing, not decorative** (verdict: the
genuine architectural distinctiveness). A synthesized program is simultaneously **data**
(`schema.cljs` walks the `gen.cljc:17-29` quoted form) and **code** (SCI evals it) — so
proposer-output, fantasies, abstractions, scorer, and controller are **all GFI values in one
algebra**. A discovered abstraction is a GF with the **full GFI**, giving **compositional inference
(update / regenerate / MCMC / SMC) *through* learned library pieces** — which **LILO / ReGAL cannot
do** (frozen LLM over *deterministic* programs scored by execution-correctness). And GenMLX **owns
its LLM forward as a GF**, so proposer `q` and exact-evidence target `p` live in the *same* algebra —
which **Loula / genlm / LLaMPPL cannot express** (external LM bolted onto an external grammar).
*Caveat:* referees discount "nice Lisp" — lead with the measured net-utility frontier; homoiconicity
is the one-paragraph affordance that makes the closure *real*, not the headline.

---

## Part III — Dreaming extends RRPS (the creative core)

**The framing that makes it click.** RRPS makes the **wake** phase resource-rational, but its
ceiling is fixed by the two things a VOC controller *cannot change*: **the hypothesis space and the
proposer**. Dreaming removes exactly those two ceilings.

### III.1 Abstract-sleep grows the hypothesis space

Run an anti-unifier / **Stitch**-style top-down MDL compressor (Bowers, POPL 2023 — 3–4 orders
faster than DreamCoder's version-space learner) over the schema-walked quoted `(gen …)` forms
(`schema.cljs:760 walk-form` already gives the structural view). Because programs **are** ClojureScript
data, refactoring is an s-expression rewrite riding `rewrite.cljs`'s existing
`detect-motif → transform → emit` engine (`IRewriteRule:27`, `apply-rewrites:213`). The discovered
subtree becomes a **new combinator record or `(gen …)` GF** — and since `combinators.cljs`
`map:435` / `unfold:802` / `switch:1080` / `mask:1284` are **already higher-order GFs** implementing
the full GFI, the abstraction drops into the same dispatcher / conjugacy / exact-scoring / compilation
machinery **with zero special-casing**.

**The payoff is precise:** a new library primitive **moves the "arrival time of the correct structure"
earlier in the proposer stream** — and that arrival time is RRPS's documented **make-or-break HARD
axis** — so the controller spends fewer proposals to find the right model and **the whole net-utility
frontier shifts outward.**

> **Real unbuilt research (the critiques' load-bearing demand):** anti-unification over **effectful**
> `(gen …)` forms must respect **trace-address renaming** — you cannot merge two subtrees with
> conflicting addresses without an address-rename scheme, and SCI/effect semantics must be preserved.
> Stitch's guarantees and speedups are for **pure λ-calculus**; transfer to effectful gen-forms is
> unverified. This is genuine research, not "`schema.cljs` already does it."

### III.2 Dream-sleep grows the proposer (fantasy + replay)

**Fantasy.** Sample programs from the growing library prior (sampling-from-library = `p/simulate`
over a gen-fn-over-programs — *free* for any GF), **SCI-eval them to synthesize their own datasets**,
and **exact-score** via the conjugacy oracle **without any external data**. The resulting
`(synthetic-data → program)` pairs are self-generated and **labeled with the true exact
marginal-evidence weight** (the true conjugate posterior).

> **This is the one genuine theoretical upgrade over DreamCoder.** DreamCoder dreams against a 0/1
> task-solved likelihood; GenMLX dreams against **exact marginal evidence**, so the recognition target
> is the **true Bayesian posterior over program structure** — *amortized Bayesian structure learning*,
> not MAP. Crucially, this **breaks the twist-learning chicken-and-egg**: you get true posterior
> targets **without already having solved inference**.

**Replay.** Also retrain on real accepted **wake** programs (anti-forgetting; mitigates
fantasy-distribution-shift / dreaming-on-your-own-errors).

**What gets trained — split by scale (the honest version):**
- **(a) The in-tree recognition net.** `amortized.cljs:87 make-elbo-loss` + `:154 train-proposal` +
  `nn.cljs:282 value-and-grad` + Adam, with gradients flowing **through the handler path** so `q`
  trains against the **same exact oracle**; deployed via `amortized.cljs:208
  neural-importance-sampling`. **Gap:** `amortized.cljs` has **only gaussian/log-normal
  continuous-latent posterior families** (verified) — a **new categorical/sequence posterior family
  over program STRUCTURE** must be built. This is the cleanest dreaming slice (no 80B port, no CUDA).
- **(b) The big proposer.** LoRA via the **native `SftTrainingEngine` / `GrpoTrainingEngine`**
  (`fromQwen35Moe` verified present in `@mlx-node/core`, **zero cljs references** — unwrapped, not
  missing), targeting `SwitchLinear` experts (stock mlx-lm LoRA **no-ops** MoE experts — issue #571,
  ~0.022 % trainable; third-party `mlx-tune` shows the fix). Framed strictly as an **hours-class
  OFFLINE** sleep phase, never interactive.

**FIM is the targeting engine.** Coder-Next's FIM sentinels are verified on disk; **Lisp is the
ideal FIM target** (the suffix supplies the closing parens); `grammar.cljs` / `bytes.cljs` logit
masks compose **unchanged** on the same per-token categorical trace site. FIM also makes the
controller's deferred **`:refine`** action real: a cheap, high-local-information,
structure-preserving edit (swap one distribution, finish one binding) reusing verified surrounding
structure — *not* whole-program regeneration. (See bean `706r`.)

### III.3 The deepest move — the unification (`q = ψ = VOC`)

[`rrps-design.md §8`](./rrps-design.md), sourced to **Zhao et al. 2024 (twisted SMC), Def 3.1**:

> The DreamCoder **recognition model `q(program | data)`**, the twisted-SMC **twist `ψ_t`**
> (= expected future potential = the value function), and the resource-rational **VOC controller's
> value-of-computation** are **three views of ONE amortized value over program-prefixes.**

Therefore **Dreaming is the principled way to *learn* the value the controller currently estimates
myopically.** It turns the controller's per-instance value-of-continuing into an **amortized
expected-future-evidence function** — which **de-myopicizes** it, curing the Hay-Russell
under-exploration that makes RRPS "give up on HARD" instances (stop before the late-arriving correct
structure is discovered).

### III.4 Resource-rational sleep over wake/abstract/dream

Fold the **same** VOC controller over the **wake / abstract / dream budget itself** — *when is it
worth dreaming (training the proposer) vs waking (solving real tasks) vs abstracting (growing the
library)?* This **resource-rational sleep on top of resource-rational wake** is the synthesis that
**neither DreamCoder (fixed schedule) nor RRPS (frozen proposer/grammar) has alone.** It is the
honest answer to *"how does Dreaming take RRPS to the next level."*

---

## Part IV — The sharp caveat (all three critics converged)

**The regime where dreaming is SOUND is nearly disjoint from the regime where it PAYS.**

- **Sound** on the **conjugate slice** (exact targets) — but that slice is small and near-enumerable:
  `program.cljs` already enumerates-and-scores it, and a strong coder already knows the idioms, so
  **dreaming amortizes almost nothing there.**
- **Pays** in a **rich non-conjugate space** where the correct structure arrives late and is
  expensive to rediscover — but there you fall back to **noisy IS evidence** and the chicken-and-egg
  returns.

Bridging the two — extending the exact oracle beyond 7 conjugate families, or making dreaming robust
to noisy evidence targets — is **the core open research problem** this vision opens. The demonstrable
dreaming slice must be a carefully chosen **semi-conjugate** space where the *learned* proposer beats
the *frozen* one **with CIs**.

**Dominant risk (blunt):** Dreaming **multiplies a win that does not yet exist.** The `gdtq` anytime
bench is currently **mean-only / no-CI**. The RRPS **P0 heterogeneity gate + a `CI-lo > 0` wake win
must land first** (Apple Silicon, today) — otherwise dreaming multiplies zero-or-negative.

---

## Part V — Demonstrable vs aspirational

| Tier | What | Where |
|---|---|---|
| **Demonstrable now** | Wake end-to-end (`msa` → `score-exact` → rank) + `program.cljs` enumeration; the exact oracle (7 families, ~5e-7); the abstraction **algebra** (`combinators` + `schema` walker + `rewrite` engine — the substrate, *not* a built learner); the recognition-net training stack (`nn`/`amortized` ELBO + autograd through the handler path, continuous-latent only); the control substrate (`rfal`/`i0s4`/`nrkq` [myopic `[:continue :stop]`, **not** K-action] / `gdtq`); FIM as prompt framing over the per-token categorical DynamicGF | **Apple Silicon, blessed small models, no Thor** |
| **Needs Thor + Coder-Next** | CUDA-retargeted mlx-node (`build.rs` CUDA define + framework gating + `sm_110` gencode + paged-attn decision + missing-op/linalg→CPU audit); the `qwen3_next` port (registration + config-shape + weight-key-mapping over the `qwen3_5_moe` scaffold + fp32 DeltaNet-state validation + golden parity); the wrapped native `SftTrainingEngine` for big-proposer LoRA | **~2–4 person-month systems track, decoupled from the science** |
| **Multi-year research** | The integrated RRPS loop landing a `CI-lo > 0` win (a tie is the modal outcome); the homoiconic MDL library learner with semantics-preserving anti-unification over effectful forms + address-renaming; grammar-grows-from-library feedback; the program-structure recognition model + fantasy/replay; the contrastively-**learned** amortized value head serving proposer + SMC twist + VOC over a meta-scheduled wake/abstract/dream loop | the frontier |

---

## Part VI — Novelty claim

> The defensible contribution is the **composition nobody has shipped:** an LLM proposer + a **learned
> library of reusable probabilistic-model motifs that are full generative functions** (supporting
> compositional inference — update/regenerate/MCMC/SMC **through** abstractions, not just generation)
> + **exact marginal-evidence scoring** (7 conjugate families) as **both** the synthesis reward **and**
> the recognition/twist-training target + an **owned-LLM forward** so proposer `q` and exact target `p`
> live in **one GFI algebra**. The exact-evidence labeling breaks the twist-learning chicken-and-egg
> on the conjugate slice (true posterior targets without already solving inference).

**Non-novel, correctly cited, never claimed as contribution:** the VOC mechanism (Russell-Wefald /
Callaway 2018), wake-sleep (DreamCoder / Ellis 2021), twist = value (Zhao 2024 Def 3.1), STaR / RFT,
Stitch / LILO compression, FIM-as-framing (Bavarian 2022). The grand unification (*one dreamed value
serving proposer + SMC twist + VOC over a meta-scheduled wake/abstract/dream loop*) is a compelling
**framing built on unbuilt organs** — it earns the discussion section only **after** the narrow
"GF-library-with-compositional-inference" slice lands empirically.

---

## Part VII — Roadmap (two decoupled tracks)

**Systems track (does NOT gate the science):**
- **S0 — CUDA-retarget mlx-node.** Add `MLX_BUILD_CUDA` gated on `target_os == linux` next to
  `build.rs:200`'s `MLX_BUILD_METAL`; gate Foundation/Accelerate (`build.rs:311-312`) behind
  `target_os == macos`; point `mlx-sys` at a CUDA MLX rev; add `sm_110` to `CUDA_ARCHITECTURES`;
  decide `mlx-paged-attn` (disable → MLX-native attention, or CUDA reimpl); audit-and-route every
  Cholesky/QR/SVD to `mx.cpu`. **Gate:** a token decodes through the owned forward on Thor.
  *Depends:* Thor hardware + MLX-CUDA building on `sm_110` (first wall) + the MoE-gather op landing
  upstream in MLX-CUDA.
- **S1 — Port `qwen3_next`.** Register in `model-loader.ts SUPPORTED_MODEL_TYPES`; port over the
  existing `qwen3_5_moe` native scaffold (config-shape + weight-key-mapping + fused-vs-split
  projection); validate fp32 Gated-DeltaNet recurrent state; golden-oracle parity vs Python mlx-lm.
  *Depends:* S0 (or Apple-Silicon-native first); beans `o94r` (sharded loader) + a **new `qwen3_next`
  bean** (`k199` is mislabeled).

**Science track (Apple Silicon today):**
- **P0 — RRPS wake: heterogeneity proof + exact oracle (the HARD GATE).** See `rrps-design.md`. If it
  fails, STOP — title-B unreachable.
- **P1–P4 — RRPS wake: integrated loop + genuine K-action controller + rigor.** Gate: a `CI-lo > 0`
  net-utility frontier-dominance win, OR report mean-only honestly and floor at title-A.
- **A1 — Abstract: homoiconic library learner (the new organ, gated behind a P4 win).** Anti-unifier/MDL
  compressor over quoted `(gen …)` forms with an address-renaming scheme; growing-library structure;
  discovered subtree → new combinator/GF. Demo: compression + inference **through** a discovered
  primitive on hierarchical conjugate models. *No LLM fine-tuning, no Thor required.*
- **A2 — Abstract: grammar grows from library.** Add each abstraction as a weighted production to the
  proposer grammar (fragment grammar; weight ∝ corpus usage), so the prior moves toward what compressed
  well and the correct-structure arrival time moves earlier.
- **D1 — Dream: recognition-net fantasy/replay (the cleanest dreaming slice).** Build the
  categorical/sequence posterior family `amortized.cljs` lacks; train the **small in-tree net** on
  library-sampled, exact-evidence-labeled fantasies + replays on a **semi-conjugate** space; show the
  learned proposer beats the frozen stream **with CIs**. Isolates *"does exact-evidence dreaming improve
  a proposer?"* without the 80B port or CUDA retarget.
- **D2 — Dream: big-proposer LoRA + meta-control over wake/abstract/dream (aspirational).** Wrap the
  native `SftTrainingEngine` targeting `SwitchLinear` experts; LoRA Coder-Next on FIM-framed
  exact-evidence-verified fantasies **offline**; fold the VOC controller over the wake/abstract/dream
  budget with the dreamed value head de-myopicizing the controller. *Depends:* S1 + D1 + the native MoE
  loader; on-device LoRA of the active MoE is unproven.

**Strategic insight:** `D1` proves the dreaming **science** on small models on the Mac you already
have. **Thor + Coder-Next make it *scale and self-host* — they are not on the critical path to the
first publishable dreaming result.**

---

## Part VIII — Honest verdicts on the originating intuitions

- **Thor unified-memory fit — RIGHT, the strongest single fact.** CUDA-13 hardware-coherent UVM is the
  one NVIDIA class where MLX's no-copy value-semantics are *real* zero-copy, not a leaky page-migration
  approximation. Sound architecture; **unproven-on-the-actual-box performance** (273 GB/s is M-Pro-class;
  `sm_110` not yet in MLX's arch list).
- **Coder-Next feasibility — RIGHT that it's feasible; WRONG that it's "a bit."** It's `qwen3_next`
  (hybrid DeltaNet + 512-expert MoE), strictly more than `qwen3_moe`; the native scaffold exists so it's
  a registration/config/weight-mapping job (not new kernels), but with the fp32-state landmine, sharded
  loader, MoE-gather-on-CUDA gap, and golden parity it's **weeks-not-days, a multi-bean port.**
- **ClojureScript generating ClojureScript — RIGHT and load-bearing, not decorative.** It's the genuine
  architectural distinctiveness (compositional inference through learned GF abstractions; `q` and `p` in
  one algebra). *But* it's the enabling affordance, not the headline.
- **Dreaming = next level — RIGHT in principle, with one sharp caveat.** It grows the hypothesis space
  *and* the proposer *and* de-myopicizes the controller, and exact-evidence labeling is a real upgrade
  over DreamCoder's 0/1. *But* the sound regime (conjugate) and the high-payoff regime (rich
  non-conjugate) are nearly disjoint, and it **multiplies a win that must first be measured.**

---

## Part IX — Critical dependencies & risks

**HW/SW dependencies:** MLX-CUDA building on Thor `sm_110` (CUDA 13.0) + the **MoE-gather op landing
in the CUDA backend** (currently unsupported — blocks the 512-expert MoE); mlx-node CUDA retarget
(build.rs + framework gating + paged-attn decision + linalg→CPU); `qwen3_next` registration + port +
fp32-state correctness + sharded loader (`o94r`) + golden parity; wrapped native MoE-LoRA targeting
`SwitchLinear` experts; a FIM prompt assembler over the per-token categorical DynamicGF; a
categorical/sequence posterior family over program structure in `amortized.cljs`.

**Risks (in priority order):**
1. **The underlying RRPS win does not yet exist** — dreaming multiplies a currently-zero quantity; the
   P0 gate + SBC must pass first, and a tie is the modal outcome.
2. **Exact oracle is narrow (7 families)** — sound-regime and pays-regime nearly disjoint; scope to
   conjugate/semi-conjugate structure-search.
3. **Anti-unification/MDL over effectful `(gen …)` forms is unbuilt research** — trace-address conflicts,
   effect-semantics preservation, library bloat; Stitch's guarantees are for pure λ-calculus.
4. **The K-action wake/abstract/dream meta-controller is design, not code** — `meta_mdp.cljs` is verified
   `[:continue :stop]` myopic; genuine K-action VOC is real work (per-action metered trial-advance,
   heterogeneous-unit cost commensuration).
5. **The MLX-on-Thor + qwen3_next + MoE-LoRA stack is ~2–4 person-months on unsupported hardware** before
   a token decodes — decouple it from the scientific claim.
6. **"Self-improving proposer" collapses to "self-improving small recognition net biasing a frozen 80B"**
   unless the native `SftTrainingEngine` is wrapped and validated for MoE experts. **Mode-collapse risk:**
   GRPO on a single sharp-optimum exact-evidence reward concentrates the proposer onto a few templates —
   the opposite of the library diversity dreaming should grow; dreaming-on-own-errors from a too-narrow
   library prior trains a recognizer that confidently reproduces blind spots.
7. **The `q = ψ = VOC` unification is only a *result* if the twist is actually LEARNED** (contrastive /
   value regression). With a myopic VOC surrogate it is a paragraph of insight, not a working shared
   value head. An honest paper either learns it or says it uses a myopic surrogate.

---

## References

**Hardware / runtime**
- NVIDIA, *Introducing Jetson Thor* — developer.nvidia.com/blog/introducing-nvidia-jetson-thor-…
  (Blackwell 2560 CUDA / 96 Tensor cores; 14-core Neoverse-V3AE; 128GB 256-bit LPDDR5X 273 GB/s; CUDA 13.0).
- SCAN/NVIDIA datasheet DS-11945-001 — "transparent, unified memory model"; CUDA 13.0 full UVM coherence.
- NVIDIA, *CUDA for Tegra* appnote — Jetson integrated-GPU zero-copy unified memory.
- ml-explore/mlx **PR #1983 / #2075** (CUDA backend, Apple-sponsored); **#2422** (CUDA backend status:
  beta; unsupported = quantized matmul, **MoE gather**, FFT, LAPACK); `unified_memory.html`.
- Awni Hannun, x.com/awnihannun/status/1948878861795819662 (CUDA backend rationale).

**Model**
- Qwen, *Qwen3-Coder-Next* model card + `tokenizer_config.json` (80B/3B, 512 experts, hybrid DeltaNet+MoE,
  FIM sentinels) — huggingface.co/Qwen/Qwen3-Coder-Next.
- Qwen, *Qwen3-Next-80B-A3B-Instruct* card (explicit hybrid layout); `mlx-community/Qwen3-Next-…-4bit`
  (reference MLX port); `lmstudio-community/Qwen3-Coder-Next-MLX-4bit` (44.8 GB).
- mlx-lm `LORA.md`; mlx-lm **#571** (LoRA no-ops MoE experts); `ARahim3/mlx-tune` (SwitchLinear LoRA).

**Synthesis / dreaming / inference**
- Ellis et al., *DreamCoder* — arXiv:2006.08381 (PLDI 2021 / Phil Trans R Soc A 2023); wake / sleep-abstraction
  / sleep-dreaming; replays vs fantasies.
- Bowers et al., *Top-Down Synthesis for Library Learning (Stitch)* — POPL 2023, arXiv:2211.16605.
- Grand et al., *LILO* — ICLR 2024, arXiv:2310.19791. Wong et al., *LAPS* — ICML 2021, arXiv:2106.11053.
  Stengel-Eskin et al., *ReGAL* — ICML 2024, arXiv:2401.16467.
- Zhao, Brekelmans, Makhzani, Grosse, *Probabilistic Inference in Language Models via Twisted SMC* —
  ICML 2024, arXiv:2404.17546 (Def 3.1; optimal twist = value function).
- Callaway et al., *Learning to Select Computations* — UAI 2018, arXiv:1711.06892 (VOC as a value function).
- Saad et al., *Bayesian Synthesis of Probabilistic Programs* — POPL 2019, arXiv:1907.06249.
- Bavarian et al., *Efficient Training of LMs to Fill in the Middle (FIM)* — arXiv:2207.14255.

*Companion: [`rrps-design.md`](./rrps-design.md) (the resource-rational wake phase + the P0 gate).
Provenance: workflow run `w178w7s6m` (2026-06-16), 17 agents, ~1.7M tokens.*
