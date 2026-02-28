# GenMLX Long-Term Roadmap: Closing the Gap with Gen.jl and GenJAX

> A phased plan to bring GenMLX to full feature parity with Gen.jl and GenJAX,
> while remaining 100% idiomatic ClojureScript -- purely functional, data-driven,
> and GPU-accelerated on Apple Silicon via MLX.
>
> *Updated 2026-02-28. Reflects ADEV, VIMCO, CustomGradientGF, amortized inference,
> NN integration, adaptive HMC, loop compilation, batch PRNG optimization.*
>
> **Note:** Performance numbers in this document have not been independently
> verified with a rigorous cross-platform benchmark. A proper comparison
> against Gen.jl and GenJAX is needed before trusting any speedup claims.

---

## Current State of GenMLX

GenMLX is a ~14,000-line probabilistic programming system implementing the
**Generative Function Interface (GFI)** in ClojureScript on Node.js (nbb) with
MLX for GPU acceleration. It provides:

| Category | What exists today |
|---|---|
| **GFI protocols (8)** | `simulate`, `generate`, `assess`, `update`, `regenerate`, `propose`, `project`, `update-with-diffs` |
| **DSL** | `gen` macro, `dyn/trace`, `dyn/splice`, `dyn/param` (Dynamic DSL only) |
| **Distributions (27)** | gaussian, uniform, bernoulli, beta, gamma, exponential, categorical, poisson, laplace, student-t, log-normal, multivariate-normal, dirichlet, delta, cauchy, inv-gamma, geometric, neg-binomial, binomial, discrete-uniform, truncated-normal, mixture, half-normal, wishart, inv-wishart, von-mises, piecewise-uniform |
| **Combinators (9)** | Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap/MapRetval (dimap) |
| **Inference (20+)** | IS, Importance Resampling, MH, Custom Proposal MH, MALA, HMC (with adaptive step-size + mass matrix), NUTS (with adaptive step-size + mass matrix), Enumerative Gibbs, Elliptical Slice Sampling, Involutive MCMC (with Jacobian), MAP, SMC, Conditional SMC, SMCP3, ADVI, Programmable VI (ELBO/IWELBO/PWake/QWake), VIMCO, ADEV, Wake-Sleep |
| **Inference composition** | chain, repeat-kernel, seed, cycle-kernels, mix-kernels, run-kernel |
| **Loop compilation** | Entire MCMC chains compiled into single Metal dispatches |
| **Edit interface** | IEdit protocol, ConstraintEdit, SelectionEdit, ProposalEdit with backward requests |
| **Incremental computation** | Argdiffs/retdiffs, diff-aware update on Map/Unfold/Scan/Switch combinators |
| **Trainable parameters** | `dyn/param`, functional parameter store, SGD/Adam optimizers, `train`, `make-param-loss-fn` |
| **Choice gradients** | Per-choice and score gradients via MLX autograd |
| **Custom gradient GFs** | `CustomGradientGF`, `IHasArgumentGrads` protocol |
| **Neural network integration** | `nn->gen-fn`, `NeuralNetGF`, amortized inference via trained neural proposals |
| **ADEV** | Reparameterization + REINFORCE strategies, vectorized GPU execution, compiled optimization loops, baseline variance reduction |
| **Vectorized inference** | `vsimulate`, `vgenerate`, `vupdate`, `vregenerate`, vectorized IS/SMC, vectorized switch |
| **Diagnostics** | ESS, R-hat, summary statistics |
| **MLX integration** | Full autograd (`grad`, `value-and-grad`, `jvp`, `vjp`), `compile-fn` (JIT to Metal), `vmap`, `tidy`, functional PRNG |
| **Data structures** | Hierarchical choice maps, immutable traces, `VectorizedTrace`, composable selections |
| **Distribution constructors** | `defdist` macro, `defdist-transform` (derived dists via transforms), `mixture` constructor, `map->dist` bridge |
| **Validation** | Parameter validation on all distributions, `validate-gen-fn` (address uniqueness, score finiteness), GFI contract verification (11 contracts, 575 checks) |
| **Trace Kernel DSL** | Compositional kernel construction |

---

## Gap Analysis

The tables below identify every significant capability present in Gen.jl or
GenJAX that GenMLX does not yet have, organized by subsystem.

### 1. Distributions

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Normal / Gaussian | `normal` | `normal` | `gaussian` | -- |
| Uniform continuous | `uniform` | `uniform` | `uniform` | -- |
| Bernoulli | `bernoulli` | `bernoulli`/`flip` | `bernoulli` | -- |
| Beta | `beta` | `beta` | `beta-dist` | -- |
| Gamma | `gamma` | -- | `gamma-dist` | -- |
| Exponential | `exponential` | -- | `exponential` | -- |
| Categorical | `categorical` | `categorical` | `categorical` | -- |
| Poisson | `poisson` | -- | `poisson` | -- |
| Laplace | `laplace` | -- | `laplace` | -- |
| Student-t | -- | -- | `student-t` | -- |
| Log-Normal | -- | -- | `log-normal` | -- |
| Multivariate Normal | `mvnormal` | `mv_normal` | `multivariate-normal` | -- |
| Dirichlet | `dirichlet` | -- | `dirichlet` | -- |
| Delta | -- | -- | `delta` | -- |
| Cauchy | `cauchy` | -- | `cauchy` | -- |
| Inverse Gamma | `inv_gamma` | -- | `inv-gamma` | -- |
| Geometric | `geometric` | -- | `geometric` | -- |
| Negative Binomial | `neg_binom` | -- | `neg-binomial` | -- |
| Binomial | `binom` | -- | `binomial` | -- |
| Discrete Uniform | `uniform_discrete` | -- | `discrete-uniform` | -- |
| Truncated Normal | -- | `truncated_normal` | `truncated-normal` | -- |
| Mixture | `HomogeneousMixture` | -- | `mixture` constructor | -- |
| `@dist` constructor | `@dist` transform DSL | -- | `defdist-transform` macro | -- |
| Piecewise Uniform | `piecewise_uniform` | -- | `piecewise-uniform` | -- |
| Beta-Uniform | `beta_uniform` | -- | `beta-uniform-mixture` | -- |
| Broadcasted Normal | `broadcasted_normal` | -- | `broadcasted-normal` | -- |
| Wishart / Inv-Wishart | via Distributions.jl | -- | `wishart`, `inv-wishart` | -- |
| External dist compat | GenDistributions.jl | ExactDensity / Distribution base | `map->dist` bridge | -- |
| **Product distribution** | `product` | -- | -- | **GAP** |
| **Directional stats** | -- | -- | -- | **GAP** |

### 2. Inference Algorithms

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Importance Sampling | `importance_sampling` | ImportanceK | `importance-sampling` | -- |
| Importance Resampling | `importance_resampling` | -- | `importance-resampling` | -- |
| MH (selection) | `mh(trace, sel)` | MH | `mh` | -- |
| MH (custom proposal GF) | `mh(trace, proposal, args)` | Custom proposals | `mh-custom` | -- |
| Involutive MCMC | `mh(trace, proposal, involution)` | Involutive MCMC | `involutive-mh` (with Jacobian) | -- |
| Enumerative Gibbs | via enumeration | Enumerative Gibbs | `gibbs` | -- |
| MALA | `mala` | -- | `mala` | -- |
| HMC | `hmc` | HMC | `hmc` (+ adaptive step-size, mass matrix) | -- |
| NUTS | -- | -- | `nuts` (+ adaptive step-size, mass matrix) | -- (GenMLX ahead) |
| SMC / Particle Filter | `initialize_particle_filter` | Bootstrap PF | `smc` | -- |
| Conditional SMC | -- | cSMC | `csmc` | -- |
| SMCP3 | GenSMCP3.jl | First-class SMCP3 | `smcp3` | -- |
| ADVI | -- | -- | `vi` | -- (GenMLX ahead) |
| Programmable VI | -- | PLDI 2024 module | `programmable-vi` (ELBO/IWELBO/PWake/QWake) | -- |
| VIMCO | `black_box_vimco!` | -- | `vimco` | -- |
| ADEV | -- | ADEV (Jaxpr + modular_vmap) | `adev-optimize` (shape-based batching) | Partial (see notes) |
| Wake-Sleep | `lecture!` | -- | `wake-sleep` (with auto-discovery) | -- |
| Inference composition | -- | Kernel combinators | `chain`, `repeat-kernel`, `seed`, `cycle-kernels`, `mix-kernels` | -- |
| Elliptical Slice Sampling | `elliptic_slice` | -- | `elliptical-slice` | -- |
| MAP Optimization | `map_optimize` | -- | `map-optimize` + `vectorized-map-optimize` | -- (GenMLX ahead) |
| **Enumerative / grid inference** | `enumerative` | -- | -- | **GAP** |

**ADEV notes:** GenMLX has ADEV with reparameterization and REINFORCE strategies,
vectorized GPU execution, compiled optimization, and baseline variance reduction.
GenJAX's ADEV is more sophisticated: enumeration strategies, measure-valued
derivatives, forward-mode `Dual` numbers, and `ADEVPrimitive` extensibility.

### 3. Combinators & Model Composition

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Map | `Map` | `Map` | `map-combinator` (full GFI + diffs) | -- |
| Unfold | `Unfold` | `Unfold` | `unfold-combinator` (full GFI) | -- |
| Switch / Cond | `Switch` | `Cond` | `switch-combinator` (full GFI) | -- |
| Scan | -- | `Scan` (lax.scan) | `scan-combinator` (full GFI) | -- |
| Mask | -- | Masking combinator | `mask-combinator` | -- |
| Mix | -- | -- | `mix-combinator` (full GFI) | -- (GenMLX ahead) |
| Contramap / Dimap | -- | -- | `contramap-gf`, `map-retval`, `dimap` (full GFI) | -- (GenMLX ahead) |
| Vectorized Cond | -- | Vectorized stochastic branching | `vectorized-switch` (mx/where masking) | -- |
| Recurse | `Recurse` | -- | `recurse-combinator` (full GFI) | -- |

### 4. Gradient & Differentiable Programming

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| `grad(f)` | -- | `jax.grad` | `mx/grad` | -- |
| `value-and-grad` | -- | `jax.value_and_grad` | `mx/value-and-grad` | -- |
| `jvp` / `vjp` | -- | native JAX | `mx/jvp`, `mx/vjp` | -- |
| `stop-gradient` | -- | `jax.lax.stop_gradient` | `mx/stop-gradient` | -- |
| `choice_gradients` | Per-choice gradients from trace | -- | `choice-gradients`, `score-gradient` | -- |
| Param gradient accumulation | `accumulate_param_gradients!` | -- | `make-param-loss-fn` (via MLX autograd) | -- |
| Custom gradient GFs | `CustomGradientGF` | -- | `CustomGradientGF` + `IHasArgumentGrads` | -- |

### 5. Trainable Parameters & Learning

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| `@param` declarations | Trainable params in GFs | -- | `dyn/param` + `*param-store*` | -- |
| Parameter store | `init_param!`, `get_params`, `set_param!` | -- | `make-param-store`, `get-param`, `set-param` | -- |
| Parameter update configs | `ParamUpdate`, `FixedStepGradientDescent`, Flux.ADAM | -- | `sgd-step`, `adam-step` | -- |
| `train!` | Supervised training loop | -- | `train` (generic loop with pluggable optimizer) | -- |
| Wake-Sleep learning | `lecture!`, `lecture_batched!` | -- | `wake-sleep` (with auto-discovery) | -- |
| Amortized inference | Via trained proposals | Neural proposals via Equinox | `nn->gen-fn`, neural amortized proposals | -- |

### 6. Trace Operations & Incremental Computation

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| `simulate` | yes | yes | yes | -- |
| `generate` | yes | yes | yes | -- |
| `assess` | yes | yes | yes | -- |
| `update` | yes | `edit` | yes | -- |
| `regenerate` | yes | -- | yes | -- |
| `propose` | `propose(gf, args)` | -- | `propose` on DynamicGF + Distribution | -- |
| Argdiffs / Retdiffs | `NoChange`, `UnknownChange`, `VectorDiff` | `Retdiff` change propagation | `no-change`, `unknown-change`, `value-change`, `vector-diff`, `map-diff` | -- |
| Diff-aware update | Static DSL skips unchanged | Retdiff-based | `update-with-diffs` on Map (per-element), Unfold, Scan, Switch | -- |
| EditRequest types | -- | Parametric edit interface | `ConstraintEdit`, `SelectionEdit`, `ProposalEdit` + backward requests | -- |
| `project` | `project(trace, sel)` | -- | `project` via project-handler | -- |
| **Static DSL incremental** | Static DSL skips unchanged subgraphs | XLA-compiled updates | -- | **GAP** |

### 7. Static DSL / Compilation

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| **Static computation graph** | `@gen (static)` compiles DAG | All GFs are JAX-traceable | -- | **GAP** |
| **Incremental trace update** | Skips unchanged subgraphs | XLA-compiled updates | -- | **GAP** |
| **Static choice maps** | `StaticChoiceMap` | JAX Pytree CMs | -- | **GAP** |
| JIT-compiled inference loops | -- | `jax.jit` over full loops | `mx/compile-fn` loop compilation | -- |

### 8. Vectorization & Hardware Acceleration

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| GPU acceleration | CPU only | GPU/TPU via XLA | GPU via MLX (Apple Silicon) | -- |
| `compile-fn` / JIT | -- | `jax.jit` | `mx/compile-fn` | -- |
| Vectorized traces | -- | First-class trace vmap | `VectorizedTrace` (shape-based) | -- |
| Vectorized particle inference | -- | Vectorized particle inference | `vsimulate`/`vgenerate`/`vupdate`/`vregenerate` | -- |
| Vectorized stochastic branching | -- | Cond via masking | `vectorized-switch` + `mask-combinator` | -- |
| **Formal vectorization correctness** | -- | lambda_GEN proof (POPL '26) | -- | **GAP** |

### 9. Ecosystem & Tooling

| Feature | Gen.jl | GenJAX | GenMLX | Gap? |
|---|---|---|---|---|
| Neural network integration | GenFlux.jl (Flux), GenTF.jl | Equinox, Flax | `nn->gen-fn`, `NeuralNetGF` | -- |
| **LLM-as-GF** | GenGPT3.jl | -- | -- | **GAP** |
| **Visualization** | -- | GenStudio | -- | **GAP** |
| **Auto-genify** | Genify.jl (auto-convert Julia fns) | -- | -- | **GAP** |
| **Stochastic probabilities** | GenSP.jl | -- | -- | **GAP** |
| Trace Kernel DSL | GenTraceKernelDSL.jl | -- | Compositional kernel algebra | -- |
| External dist compat | GenDistributions.jl | ExactDensity API | `map->dist` bridge | -- |
| Input validation | -- | Static tracing-time typecheck | Parameter validation + `validate-gen-fn` | Partial |
| Error messages | -- | -- | Helpful error messages for common mistakes | -- |
| **Trace serialization** | -- | -- | -- | **GAP** |

---

## Roadmap

The roadmap is organized into **7 phases**, roughly ordered by
(a) foundational dependencies, (b) impact on user modeling power, and
(c) feasibility. Each phase is independently valuable and shippable.

### Design Principles (apply to every phase)

1. **Purely functional first.** State lives in persistent data structures passed
   through function arguments. No atoms, no mutation, no side effects in the
   modeling layer. The only mutable boundary is the handler's `volatile!`
   (already isolated).

2. **Data-driven, open for extension.** Prefer multimethods and records over
   closed hierarchies. New distributions, combinators, and inference algorithms
   should be addable without modifying core code.

3. **MLX arrays end-to-end.** Values stay as MLX arrays from sampling through
   scoring through gradient computation. Only extract to JS numbers at
   inference boundaries.

4. **Lazy graph + explicit eval.** Leverage MLX's lazy evaluation. Mark
   materialization points explicitly with `mx/eval!` to bound memory.

5. **ClojureScript idioms.** Protocols, multimethods, `defrecord`, transducers,
   persistent data structures, structural sharing, spec/Malli for validation.

---

### Phase 1: Distribution Library Completeness ✅ COMPLETE

**Goal:** Match Gen.jl's full distribution library and add distribution
composition primitives.

**Status:** 27 distributions total. All have native `dist-sample-n` batch
sampling (including rejection-sampling dists: beta, gamma, dirichlet, student-t).
`defdist-transform` macro, `mixture` constructor, and `map->dist` bridge implemented.
Beta-uniform mixture and broadcasted normal added.

---

### Phase 2: Custom Proposals & Advanced MCMC ✅ COMPLETE

**Goal:** Enable user-written proposal generative functions for MH, add
Gibbs sampling, and advanced MCMC.

**Status:** All sub-items implemented and tested. Includes custom proposal MH,
elliptical slice sampling, MAP optimization, and Gibbs sampling with enumeration.

---

### Phase 3: Involutive MCMC & Reversible Jump ✅ CORE COMPLETE

**Goal:** Implement Gen.jl's involutive MCMC framework.

**Status:** `involutive-mh-step` and `involutive-mh` implemented with full
Jacobian support. Reversible jump helpers remain.

---

### Phase 4: Trainable Parameters & Learning ✅ COMPLETE

**Goal:** Add trainable parameters, gradient infrastructure, and learning
algorithms.

**Status:** All sub-items implemented. Parameter store, choice/score gradients,
SGD/Adam optimizers, training loop, and wake-sleep learning.

---

### Phase 5: Incremental Computation & Performance ✅ CORE COMPLETE

**Goal:** Add change-propagation hints (argdiffs/retdiffs) so `update` can
skip unchanged computation.

**Status:** Diff types, `IUpdateWithDiffs` protocol, and combinator
implementations all complete. Static analysis pass remains aspirational.

---

### Phase 6: SMCP3 & Programmable Sequential Inference ✅ COMPLETE

**Goal:** Implement SMCP3 and the edit interface.

**Status:** All sub-items implemented. Edit requests, inference kernel
composition, and SMCP3 inference.

---

### Phase 7: Vectorized Inference & Advanced Compilation ✅ MOSTLY COMPLETE

**Goal:** Bring GenJAX's key innovation -- compositional vectorization of
traces and inference -- to GenMLX via shape-based batching.

**Status:** Shape-based batching for vectorized traces and inference operations.
Loop compilation for MCMC chains. Batch PRNG optimization for compiled-mh.
No `splice` in batched mode. No compiled SMC sweeps.

---

### Phase 8: Ecosystem & Interop (Partially Complete)

**Goal:** Build out the ecosystem with neural network integration,
visualization, and interop.

**Status:** NN integration (`nn->gen-fn`, `NeuralNetGF`) and amortized inference
(neural proposals, ELBO training) are done. ADEV with vectorized GPU execution
and compiled optimization is done. Visualization, LLM-as-GF, and auto-genify
remain.

---

## Summary: Phase Status

```
Phase 1: Distributions          ✅ COMPLETE (27 distributions + constructors + bridge)
Phase 2: Custom Proposals       ✅ COMPLETE (ESS, MAP, Gibbs all done)
Phase 3: Involutive MCMC        ✅ CORE COMPLETE (RJ helpers remaining)
Phase 4: Parameters & Learning  ✅ COMPLETE
Phase 5: Incremental Compute    ✅ CORE COMPLETE (static analysis remaining)
Phase 6: SMCP3                  ✅ COMPLETE
Phase 7: Vectorized Inference   ✅ MOSTLY COMPLETE (no splice in batched mode, no compiled SMC)
Phase 8: Ecosystem              🟡 PARTIALLY COMPLETE (NN + amortized + ADEV done; viz, LLM, auto-genify remain)
```

**Remaining gaps (by priority):**
1. PJAX staging infrastructure / static DSL (significant architectural effort)
2. ADEV enumeration + measure-valued derivative strategies
3. Ecosystem: visualization, LLM-as-GF, auto-genify (open-ended)
4. Product distribution, trace serialization, enumerative inference

---

## Design Philosophy: GenMLX vs Gen.jl vs GenJAX

| Aspect | Gen.jl | GenJAX | GenMLX |
|---|---|---|---|
| **Language** | Julia (imperative + functional) | Python + JAX (functional transforms) | ClojureScript (purely functional) |
| **Mutation** | Mutable traces, parameter store | JAX Pytrees (immutable-ish) | Persistent data structures, no mutation |
| **Dispatch** | Julia multiple dispatch | Python class hierarchy | Clojure multimethods + protocols |
| **Extension** | Subtype `GenerativeFunction` | Subclass / Pytree | Implement protocols, add multimethods |
| **Compilation** | Julia JIT | XLA (GPU/TPU) | MLX Metal (Apple Silicon GPU) |
| **Vectorization** | Manual | `jax.vmap` (first-class) | Shape-based batching (dispatch amortization) |
| **PRNG** | Global RNG | JAX splittable keys | Splittable keys (functional) |
| **Memory** | Julia GC | XLA managed | MLX unified memory (zero-copy CPU/GPU) |

GenMLX's unique advantage: **unified memory architecture** on Apple Silicon
means CPU control flow and GPU numerics share the same memory with zero
transfer cost. This is ideal for the "thin CPU control + fat GPU compute"
pattern of MCMC inference.

---

## What GenMLX Already Does Better

- **NUTS sampler** -- neither Gen.jl nor GenJAX ship a built-in NUTS (with adaptive step-size + mass matrix)
- **ADVI** -- Gen.jl's `black_box_vi!` is more general but less turnkey
- **Adaptive HMC** -- dual averaging step-size tuning + diagonal mass matrix (Hoffman & Gelman 2014)
- **Vectorized MAP** -- N random restarts optimized simultaneously
- **Loop-compiled MCMC** -- entire chains as single Metal dispatches
- **GFI contract verification** -- 11 measure-theoretic contracts, tested on 13 canonical models
- **Purely functional handler design** -- cleaner than Gen.jl's mutable traces
- **`defdist` macro** -- more ergonomic than Gen.jl's manual `Distribution` subtyping
- **MLX unified memory** -- zero-copy CPU/GPU is architecturally superior to PCIe transfer
- **Vectorized inference without `vmap`** -- shape-based batching amortizes Metal
  dispatch across N particles, simpler than GenJAX's JAX vmap
- **~14,000 lines** -- dramatically smaller than Gen.jl (~20k+) or GenJAX (~10k+),
  making the system auditable and hackable

---

*This roadmap is a living document. Phases should be re-prioritized based on
user needs and upstream MLX capabilities.*
