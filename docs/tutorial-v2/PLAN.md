# GenMLX Tutorial — Master Plan

## Design Principles

- **One running example** (Bayesian linear regression) that grows across chapters
- **Show, then explain** — every concept starts with runnable code before theory
- **Honest about impurity** — show the volatile!, explain why it's there
- **No prerequisites** beyond basic Clojure — probability taught inline
- **Each chapter self-contained** but builds on previous chapters
- **Frame in PP terms** — no web framework analogies; explain the pure-core / managed-shell idea through models, handlers, and inference

## Test Strategy

Each chapter has a companion test file: `test/genmlx/tutorial/chNN_test.cljs`
- Every code listing in the chapter has a corresponding test
- Tests verify: correct output, no errors, expected types/values
- Run all: `for f in test/genmlx/tutorial/ch*_test.cljs; do bun run --bun nbb "$f"; done`

## Coverage Checklist (from old tutorial audit)

The new tutorial must cover everything the old tutorial covered.

### Core API Coverage
- [ ] `gen` macro
- [ ] `trace` with keyword addresses
- [ ] `splice` for sub-model composition
- [ ] `param` for trainable parameters
- [ ] `p/simulate` — forward sampling
- [ ] `p/generate` — constrained execution
- [ ] `p/update` — incremental update
- [ ] `p/regenerate` — resample selected addresses
- [ ] `p/assess` — score fully-specified choices
- [ ] `p/project` — score selected addresses

### Data Structures
- [ ] `cm/choicemap` constructor
- [ ] `cm/from-map`, `cm/to-map` conversion
- [ ] `cm/get-submap`, `cm/get-value`, `cm/get-choice`
- [ ] `cm/set-value`, `cm/set-choice`
- [ ] `cm/addresses` enumeration
- [ ] `cm/merge-cm` merging
- [ ] Nested/hierarchical choice maps
- [ ] `Value` and `Node` records
- [ ] `Trace` record fields (gen-fn, args, choices, retval, score)
- [ ] `sel/select`, `sel/all`, `sel/none`
- [ ] `sel/complement-sel`, `sel/hierarchical`

### Distributions (all 27)
- [ ] Continuous: gaussian, uniform, beta, gamma, exponential, laplace, student-t, log-normal, cauchy, inv-gamma, truncated-normal, half-normal, logistic, pareto, gumbel, von-mises
- [ ] Discrete: bernoulli, categorical, poisson, geometric, binomial, neg-binomial
- [ ] Multivariate: dirichlet, multivariate-normal, wishart
- [ ] Special: delta, mixture, product

### Inference Algorithms
- [ ] `importance/importance-sampling`
- [ ] `importance/importance-resampling`
- [ ] `mcmc/mh` basic Metropolis-Hastings
- [ ] `mcmc/compiled-mh`
- [ ] `mcmc/mala` — Langevin
- [ ] `mcmc/hmc` — Hamiltonian MC
- [ ] `mcmc/nuts` — No-U-Turn Sampler
- [ ] Adaptive HMC (step size, mass matrix)
- [ ] `mcmc/gibbs` with schedule
- [ ] `mcmc/elliptical-slice`
- [ ] `mcmc/map-optimize` — MAP estimation
- [ ] `smc/smc` with rejuvenation
- [ ] `smc/csmc` — conditional SMC
- [ ] `smcp3/smcp3` — probabilistic program proposals
- [ ] `vi/vi` — variational inference
- [ ] `vi/vi-from-model` convenience
- [ ] `vi/compiled-vi-from-model`
- [ ] `vi/programmable-vi` with objectives (ELBO, IWELBO, VIMCO)

### Kernel Composition
- [ ] `kern/mh-kernel`
- [ ] `kern/random-walk`
- [ ] `kern/prior`
- [ ] `kern/proposal` (symmetric and asymmetric)
- [ ] `kern/gibbs`
- [ ] `kern/chain`
- [ ] `kern/repeat-kernel`
- [ ] `kern/cycle-kernels`
- [ ] `kern/mix-kernels`
- [ ] `kern/seed`
- [ ] `kern/run-kernel` with burn, thin, callback
- [ ] `kern/with-reversal`, `kern/symmetric-kernel`

### Combinators
- [ ] `comb/map-combinator`
- [ ] `comb/unfold-combinator`
- [ ] `comb/switch-combinator`
- [ ] `comb/scan-combinator`
- [ ] `comb/mix-combinator`
- [ ] `comb/recurse`
- [ ] `comb/contramap-gf`, `comb/dimap`
- [ ] `vmap/vmap-gf`, `vmap/repeat-gf`

### Vectorized Inference
- [ ] `dyn/vsimulate`
- [ ] `dyn/vgenerate`
- [ ] `dyn/vupdate`, `dyn/vregenerate`
- [ ] `VectorizedTrace` structure
- [ ] `vec/vtrace-ess`, `vec/vtrace-log-ml-estimate`
- [ ] `vec/resample-vtrace`
- [ ] `vec/systematic-resample-indices` (CPU and GPU)
- [ ] `vec/merge-vtraces-by-mask`

### Gradients and Learning
- [ ] `grad/choice-gradients`
- [ ] `grad/score-gradient`
- [ ] `adev/adev-gradient`, `adev/vadev-gradient`
- [ ] `adev/adev-optimize`, `adev/compiled-adev-optimize`
- [ ] `learn/make-param-store`
- [ ] `learn/sgd-step`, `learn/adam-init`, `learn/adam-step`
- [ ] `learn/train`
- [ ] `learn/wake-sleep`
- [ ] `learn/simulate-with-params`

### Neural Integration
- [ ] `nn/sequential`, `nn/linear`, activation functions
- [ ] `nn/nn->gen-fn`
- [ ] `cg/custom-gradient-gf`
- [ ] `amort/train-proposal!`, `amort/neural-importance-sampling`

### Extensions and Verification
- [ ] `defdist` macro (sample, log-prob, reparam, support)
- [ ] `dc/map->dist` for quick custom distributions
- [ ] `contracts/verify-gfi-contracts` (11 contracts)
- [ ] `verify/validate-gen-fn` static validation

### Diagnostics
- [ ] `diag/ess`, `diag/r-hat`
- [ ] `diag/sample-mean`, `diag/sample-std`, `diag/sample-quantiles`
- [ ] `diag/summarize`

### Compilation Ladder (conceptual)
- [ ] Schema extraction explanation
- [ ] Compiled simulate/generate
- [ ] Vectorized inference as Level 0
- [ ] Auto-analytical elimination
- [ ] The `fit` API

---

## Chapter Plans

### Introduction
**File:** `src/introduction.md`
**Test:** none (no code)

- [ ] What is GenMLX? (one paragraph)
- [ ] The principle: your model is pure, the framework manages state
- [ ] What you'll build: Bayesian linear regression from scratch to GPU
- [ ] Prerequisites: macOS + Apple Silicon, Bun, npm install
- [ ] How to run examples: `bun run --bun nbb example.cljs`
- [ ] 30-second demo: coin flip model, simulate, print trace

---

### Chapter 1: Your First Model
**File:** `src/ch01-first-model.md`
**Test:** `test/genmlx/tutorial/ch01_test.cljs`

- [ ] The `gen` macro — what it does (makes a generative function)
- [ ] `trace` — declaring a random choice at an address
- [ ] Distributions: `dist/gaussian`, `dist/bernoulli`, `dist/beta`
- [ ] `p/simulate` — run the model forward, get a trace
- [ ] Inspecting traces: `:choices`, `:retval`, `:score`
- [ ] MLX arrays: `mx/scalar`, `mx/item`, `mx/add`, `mx/multiply`
- [ ] Running example: coin model (Beta prior on bias, Bernoulli flips)
- [ ] Running example: linear regression (slope, intercept, noisy observations)
- [ ] Run simulate 5 times, show different traces
- [ ] What the score means: log p(all choices)
- [ ] Distribution catalog overview (list the 27, show 5-6 in examples)

---

### Chapter 2: Conditioning — The Core Trick
**File:** `src/ch02-conditioning.md`
**Test:** `test/genmlx/tutorial/ch02_test.cljs`

- [ ] `p/generate` with observations — constrain some addresses
- [ ] Building choice maps: `cm/choicemap`
- [ ] The weight: marginal log-likelihood contribution
- [ ] Same model, different handler interpretation (the "what vs how")
- [ ] Importance sampling by hand: generate N times, normalize weights
- [ ] Estimating the posterior mean on slope
- [ ] `importance/importance-sampling` — the built-in version
- [ ] Log marginal likelihood estimation
- [ ] Prior vs posterior comparison (visual/numerical)
- [ ] Nested choice maps for splice models

---

### Chapter 3: How It Works — The Handler Loop
**File:** `src/ch03-handler-loop.md`
**Test:** `test/genmlx/tutorial/ch03_test.cljs`

- [ ] The handler is a pure function: `(state, addr, dist) -> (value, state')`
- [ ] Show `simulate-transition` code (from handler.cljs)
- [ ] Show `generate-transition` code — the only difference is constraint check
- [ ] Handler state: `:key`, `:choices`, `:score`, `:weight`, `:constraints`
- [ ] The `volatile!` boundary: where mutation lives and only there
- [ ] `run-handler` creates closures, passes runtime to model body
- [ ] The `gen` macro: 8 lines, injects runtime, binds trace/splice/param
- [ ] The pure core / managed shell philosophy
- [ ] Honest about impurity: the volatile! is real, the GPU layer is effectful
- [ ] Why purity matters: composition, shape polymorphism, compilation
- [ ] The analogy: the volatile! is to GenMLX what a single state cell is to any functional architecture — isolated, managed, invisible to the user

---

### Chapter 4: Choice Maps and Traces
**File:** `src/ch04-data-structures.md`
**Test:** `test/genmlx/tutorial/ch04_test.cljs`

- [ ] `Value` and `Node` records — the two types
- [ ] `cm/choicemap` constructor with keyword-value pairs
- [ ] `cm/from-map` / `cm/to-map` for nested structures
- [ ] `cm/get-submap`, `cm/get-value`, `cm/get-choice`
- [ ] `cm/set-value`, `cm/set-choice`, `cm/merge-cm`
- [ ] `cm/addresses` — list all leaf paths
- [ ] `cm/stack-choicemaps` / `cm/unstack-choicemap` for batching
- [ ] Trace record: `{:gen-fn :args :choices :retval :score}`
- [ ] Trace metadata: splice-scores, element-scores
- [ ] Selections: `sel/select`, `sel/all`, `sel/none`
- [ ] `sel/complement-sel`, `sel/hierarchical`
- [ ] Selections as a Boolean algebra

---

### Chapter 5: Updating and Regenerating
**File:** `src/ch05-update-regenerate.md`
**Test:** `test/genmlx/tutorial/ch05_test.cljs`

- [ ] `p/update` — change choices, get new trace + weight + discard
- [ ] Weight semantics: log(new_score / old_score)
- [ ] The discard: what was replaced
- [ ] `p/regenerate` — resample selected addresses
- [ ] Using selections to target specific addresses
- [ ] `p/assess` — score fully-specified choices
- [ ] `p/project` — score a selection
- [ ] Build MH by hand: regenerate, compute weight, accept/reject
- [ ] The accept-reject criterion: log(u) < log(w)
- [ ] Run MH chain for 500 steps on linear regression
- [ ] Burn-in and thinning concepts

---

### Chapter 6: Composition — Splice and Combinators
**File:** `src/ch06-composition.md`
**Test:** `test/genmlx/tutorial/ch06_test.cljs`

- [ ] `splice` for calling sub-models at namespaced addresses
- [ ] Hierarchical choice maps from splice
- [ ] `comb/map-combinator` — independent repetition
- [ ] `comb/unfold-combinator` — sequential state (HMM example)
- [ ] `comb/switch-combinator` — conditional branching
- [ ] `comb/scan-combinator` — fold with carry
- [ ] `comb/mix-combinator` — mixture models
- [ ] `comb/recurse` — recursive models (tree example)
- [ ] `comb/contramap-gf`, `comb/dimap` — argument/return transforms
- [ ] `vmap/vmap-gf`, `vmap/repeat-gf` — batched repetition
- [ ] Addressing conventions per combinator
- [ ] Composing combinators: Map of Unfold for multiple time series

---

### Chapter 7: The Inference Toolkit
**File:** `src/ch07-inference.md`
**Test:** `test/genmlx/tutorial/ch07_test.cljs`

- [ ] Importance sampling and resampling
- [ ] Kernel composition: `chain`, `repeat`, `cycle`, `mix`, `seed`
- [ ] `kern/mh-kernel`, `kern/random-walk`, `kern/prior`, `kern/gibbs`
- [ ] `kern/proposal` — symmetric and asymmetric
- [ ] `kern/run-kernel` with burn, thin, callback
- [ ] Reversal propagation: `with-reversal`, `symmetric-kernel`
- [ ] The edit interface: ConstraintEdit, SelectionEdit, ProposalEdit
- [ ] Gradient-based MCMC: MALA, HMC, NUTS
- [ ] Adaptive HMC: step size, mass matrix
- [ ] SMC: particles, resampling, rejuvenation, cSMC
- [ ] SMCP3: probabilistic program proposals
- [ ] Variational inference: ADVI, IWELBO, VIMCO, programmable VI
- [ ] Gibbs sampling, elliptical slice sampling, MAP
- [ ] Diagnostics: ESS, R-hat, summary statistics
- [ ] "When to use what" guidance table

---

### Chapter 8: Going Fast — Vectorization and Compilation
**File:** `src/ch08-vectorization.md`
**Test:** `test/genmlx/tutorial/ch08_test.cljs`

- [ ] Shape polymorphism: why handlers that don't inspect shapes work at any shape
- [ ] `dyn/vsimulate` and `dyn/vgenerate`
- [ ] `VectorizedTrace` structure
- [ ] ESS and log-ML from vectorized traces
- [ ] GPU resampling: systematic, stratified, residual
- [ ] Batched update and regenerate
- [ ] Merge by mask for per-particle accept/reject
- [ ] Performance comparison: scalar vs vectorized
- [ ] Schema extraction: what `gen` knows about your model
- [ ] Compiled simulate/generate for static models
- [ ] Partial compilation for dynamic models
- [ ] Auto-analytical elimination overview
- [ ] The `fit` API
- [ ] Limitations: no splice in batched mode

---

### Chapter 9: Gradients, Learning, and Neural Models
**File:** `src/ch09-learning.md`
**Test:** `test/genmlx/tutorial/ch09_test.cljs`

- [ ] Choice gradients: `grad/choice-gradients`
- [ ] Score gradients: `grad/score-gradient`
- [ ] ADEV: automatic gradient selection (reparam vs REINFORCE)
- [ ] Vectorized and compiled ADEV
- [ ] Parameter stores: `learn/make-param-store`, `param` in models
- [ ] Optimizers: SGD, Adam
- [ ] Training loop: `learn/train`
- [ ] Wake-sleep learning
- [ ] MLX neural networks: `nn/sequential`, `nn/linear`, activations
- [ ] `nn/nn->gen-fn` — wrap networks as generative functions
- [ ] Custom gradient GFs: `cg/custom-gradient-gf`
- [ ] Amortized inference: train proposal, neural IS
- [ ] End-to-end differentiable probabilistic programming

---

### Chapter 10: Extensions and Verification
**File:** `src/ch10-extensions.md`
**Test:** `test/genmlx/tutorial/ch10_test.cljs`

- [ ] `defdist` macro: sample, log-prob, reparam, support clauses
- [ ] Batch sampling with `defmethod dc/dist-sample-n*`
- [ ] `dc/map->dist` for quick custom distributions
- [ ] Transformed distributions
- [ ] GFI contracts: all 11, `contracts/verify-gfi-contracts`
- [ ] Static validation: `verify/validate-gen-fn`
- [ ] The compilation ladder: conceptual overview of Levels 0-4
- [ ] The FP/PP correspondence table
- [ ] Where to go next: the paper, the source, Level 5

---

## Verification Protocol

For each chapter:
1. **Write** the Markdown content
2. **Write** the companion test file
3. **Run** the test: `bun run --bun nbb test/genmlx/tutorial/chNN_test.cljs`
4. **Verify** all PASS, zero FAIL
5. **Cross-check** code in Markdown matches code in test file
6. **Check off** all items in the chapter plan above
7. **Build** the book: `cd docs/tutorial-v2 && mdbook build`
8. **Visual check** in browser
