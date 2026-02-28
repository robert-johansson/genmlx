# Paper 1: The GenMLX System Paper

**Title:** "GenMLX: Purely Functional Probabilistic Programming on Apple Silicon via MLX"

**Target venue:** [ACM Transactions on Probabilistic Machine Learning (TOPML)](https://dl.acm.org/journal/topml) — regular submission

**Status:** Near-complete draft exists in `paper/genmlx.md` (760 lines, 9 sections).

---

## Venue Fit

TOPML's scope includes "design and implementation of probabilistic programming
languages", Bayesian modelling and inference, variational inference, and Monte
Carlo sampling. GenMLX is a probabilistic programming system with all of these.

TOPML had a [Special Issue on Probabilistic Programming](https://dl.acm.org/pb-assets/static_journal_pages/topml/pdf/TOPML_CfP_SI_Probabilistic_Programming-1704743488800.pdf)
(guest editors: Katoen, Rainforth, Yang; published Sept 2025 as Vol 1 Issue 3).
That deadline has passed, but the topic area is clearly in scope for regular
submissions. The special issue establishes precedent for PPL systems papers.

**TOPML format requirements:**
- Double-anonymous review (remove author names, affiliations, acknowledgements)
- ACM LaTeX template (single-column)
- Gold Open Access
- No strict page limit (journal format — aim for 25-35 pages)
- Quarterly publication (March, June, September, December)
- Submit via https://mc.manuscriptcentral.com/topml

---

## Assessment of Existing Draft

`paper/genmlx.md` is a strong, comprehensive system paper covering all 7
architectural layers. For TOPML, it needs:

1. **Double-anonymous formatting** — remove any self-identifying references
2. **Quantitative evaluation section** — the biggest gap
3. **More depth on inference algorithms** — TOPML readers care about this
4. **Formal foundations section** — connects to Paper 2, distinguishes from
   other PPL systems papers
5. **Corrected claims** (from verification):
   - Source LOC: ~10,800 (not ~17,000 as in CLAUDE.md; paper already says ~10,400)
   - Distributions: 22 via defdist + 5 via defn = 27 total (correct, but clarify)
   - Property tests: 153 (not 162)

---

## Planned Changes

### 1. Rewrite Abstract for TOPML Audience

TOPML readers are probabilistic ML researchers, not PL researchers. Lead with
the inference capabilities and the Apple Silicon/MLX angle, not the language
design.

**Proposed thesis:**
> We present GenMLX, a probabilistic programming system that implements the
> Generative Function Interface on Apple Silicon via MLX, providing 29 inference
> algorithms (9 MCMC variants including adaptive HMC/NUTS, 4 SMC variants,
> programmable VI with 5 objectives, and ADEV gradient estimation), 27
> distributions, and 10 model combinators in ~10,800 lines of ClojureScript.
> Shape-based vectorization achieves significant particle-method speedup without
> program transformation. Loop compilation fuses multi-step MCMC chains into
> single Metal GPU dispatches. A formal specification in quasi-Borel spaces
> establishes correctness of all GFI operations, connected to 11 executable
> runtime contracts.

**Proposed contributions (4):**
1. A complete implementation of the GFI on Apple Silicon with 29 inference
   algorithms, matching and extending GenJAX's repertoire
2. Shape-based vectorization via broadcasting (correctness proof included)
3. Loop-compiled MCMC chains on Metal
4. Formal specification with 15 theorems and 11 runtime contracts

### 2. Expand Inference Algorithms Section

TOPML readers want inference depth. The current Section 2.7 lists algorithms
but doesn't explain the design decisions. Expand to cover:

- **Why 9 MCMC variants?** Different posterior geometries need different samplers.
  Show the selection logic: unimodal smooth → HMC/NUTS, multimodal → MH mixture,
  discrete → Gibbs, Gaussian prior → elliptical slice.
- **Adaptive tuning:** Dual averaging for step-size (Algorithm 4, Hoffman & Gelman),
  Welford online variance for mass matrix. Explain warmup/sampling phase split.
- **SMCP3 design:** How probabilistic program proposals integrate via the edit
  interface. This is the most novel inference contribution relative to non-Gen PPLs.
- **Programmable VI:** 5 objectives × 2 estimators. Explain when each combination
  is appropriate.

### 3. Add Evaluation Section (NEW — critical for TOPML)

Journal papers need thorough evaluation. TOPML is not a workshop.

**Proposed evaluation (5 experiments):**

| Experiment | What it shows | Data source |
|-----------|--------------|-------------|
| Vectorization speedup vs N | Broadcasting works, scales with particles | vectorized_benchmark.cljs |
| Loop compilation speedup vs K | Metal fusion amortizes dispatch overhead | compiled_benchmark.cljs |
| Inference convergence | Correct posteriors on conjugate models | conjugate posterior tests |
| GFI contract verification | 11 contracts × 13 models = 575 checks, 0 failures | contract_verification_test.cljs |
| Cross-implementation compatibility | 165 Gen.clj + 73 GenJAX tests pass | compat test suites |

**Additional evaluation for TOPML:**
- **Posterior quality comparison:** Run the same model (e.g., 8-schools) with
  GenMLX NUTS vs Stan NUTS vs NumPyro NUTS. Compare ESS/sec, R-hat.
- **Scaling with model complexity:** Score evaluation time vs number of trace
  sites (S=5, 10, 50, 100). Show lazy graph fusion benefit.

### 4. Add Formal Foundations Section (NEW)

This distinguishes GenMLX from every other PPL paper at TOPML. No other PPL
submission will have 6,652 lines of formal proofs.

**Content (2-3 pages):**
- QBS denotational semantics for all GFI operations
- Handler soundness theorem (execution = denotation)
- Broadcasting correctness via logical relations
- The 11 runtime contracts and their formal backing
- The verification ladder: semantics → proofs → contracts → property tests →
  compatibility tests

Bridge to Paper 2 for the full treatment.

### 5. Compress Architecture Section

Journal format allows more space, but the current architecture section is still
too catalog-like. Restructure around *design decisions*:

- **Why handlers?** Compare to Gen.jl's approach (separate implementations per
  operation) vs GenMLX's unified handler (one mutable cell, pure transitions)
- **Why open multimethods for distributions?** Compare to class hierarchies
- **Why persistent data structures?** The purity argument in 1 paragraph

### 6. Anonymization

For double-anonymous review:
- Remove "GenMLX" from title? Or keep it (established in prior work)?
- Remove GitHub URL
- Refer to "our system" instead of "GenMLX" where possible
- Anonymize any self-citations

---

## Paper Outline (target: 30-35 pages, TOPML journal format)

```
1. Introduction (3 pages)
   1.1 Probabilistic Programming on Consumer Hardware
   1.2 The Generative Function Interface
   1.3 Contributions

2. System Architecture (5 pages)
   2.1 MLX Foundation: Unified Memory and Lazy Evaluation
   2.2 Handler Architecture: Pure State Transitions
   2.3 Dynamic DSL and Model Definition
   2.4 Data-Driven Distributions (27 types)
   2.5 Combinators (10 types)
   2.6 Edit Interface and Incremental Computation

3. Inference Algorithms (5 pages)
   3.1 MCMC: From MH to NUTS
       - Standard MH, custom-proposal MH, enumerative Gibbs
       - Gradient-based: MALA, HMC (with fused leapfrog), NUTS
       - Special: elliptical slice, involutive MCMC, MAP
   3.2 Adaptive Tuning
       - Dual averaging for step-size
       - Welford online variance for mass matrix
   3.3 Loop Compilation
       - Fusing K-step chains into single Metal dispatches
       - Pre-generated randomness strategy
   3.4 Particle Methods: IS, SMC, SMCP3
       - SMCP3: probabilistic program proposals via edit interface
   3.5 Variational Inference
       - ADVI, programmable VI (5 objectives, 2 estimators)
       - ADEV gradient estimation
   3.6 Amortized Inference and Wake-Sleep

4. Vectorized Inference via Shape Broadcasting (3 pages)
   4.1 The Insight: Shapes, Not Transforms
   4.2 Implementation: dist-sample-n, Batched Handlers, VectorizedTrace
   4.3 Correctness: Logical Relations Argument
   4.4 Vectorized Switch for Discrete Structure
   4.5 Limitations

5. Formal Foundations (3 pages)
   5.1 Denotational Semantics in Quasi-Borel Spaces
   5.2 Core Theorems
       - Handler soundness
       - Weight correctness (generate, update)
       - Broadcasting correctness
       - Combinator compositionality
       - Edit/backward duality
   5.3 Runtime Contracts
       - 11 contracts, 13 canonical models, 575 checks
   5.4 The Verification Ladder

6. Evaluation (5 pages)
   6.1 Experimental Setup
   6.2 Vectorization Speedup
   6.3 Loop Compilation Speedup
   6.4 Inference Convergence on Conjugate Models
   6.5 Cross-Implementation Compatibility
   6.6 Contract Verification Results
   6.7 Comparison with Related Systems (Gen.jl, GenJAX, Stan, NumPyro)

7. Related Work (2 pages)
   7.1 The Gen Ecosystem (Gen.jl, GenJAX)
   7.2 Other PPL Systems (Stan, Pyro, NumPyro, Turing.jl, Edward)
   7.3 Functional PPL Design (Church, Anglican, Hakaru, WebPPL)
   7.4 Formal Methods for PPLs
   7.5 GPU-Accelerated Inference

8. Conclusion (1 page)

Appendix A: Distribution Catalog (27 distributions with signatures)
Appendix B: Combinator Descriptions (10 combinators)
Appendix C: Complete Evaluation Data
```

---

## Relationship to Other Papers

- **Paper 2 (Formal):** Section 5 here is a 3-page summary; Paper 2 is the full
  30-page treatment. Both submitted to TOPML — they complement rather than overlap.
  Paper 1 is "here's the system and why it works"; Paper 2 is "here's the proof
  that it's correct."
- **Papers 3-5:** Vectorization, compilation, and language design are integrated
  into this paper's sections. If accepted, the standalone papers become unnecessary
  for TOPML (but could still target other venues).

**Submit Paper 1 first.** Paper 2 can reference it (or be submitted simultaneously
as a companion paper).

---

## POPL 2026 Gap (Our Opportunity)

The Becker et al. POPL 2026 paper (GenJAX/λ_GEN) formalizes:
- λ_GEN calculus: types, terms, QBS denotational semantics
- Two program transformations only: simulate{−} and assess{−}
- Proposition 3.1: correctness of simulate and assess
- vmap_n as source-to-source transformation with logical relations proof

The POPL 2026 paper does **NOT** formalize:
- generate, update, regenerate, project, propose, edit
- Handler-based execution (they use JAX tracing, not handlers)
- Combinators beyond scan (no Map, Unfold, Switch, Mask, Mix, Recurse)
- Diff-aware incremental computation
- Broadcasting as an alternative to vmap

This gap is the paper's opportunity. State it clearly in Section 1.

---

## Required Benchmarks (from TOPML expectations)

**Minimum 3 benchmark models:**

1. **Hidden Markov Model (HMM)** — standard PP benchmark. Compare sequential IS,
   vectorized IS, SMC. Report: log-marginal-likelihood accuracy, wall-clock time,
   ESS. Exercises Unfold + generate + importance weighting.

2. **Bayesian Linear/Polynomial Regression** — full inference pipeline. Compare
   MH, HMC, NUTS, VI. Report: posterior accuracy (compare to analytic solution),
   convergence speed, samples/second.

3. **Gaussian Mixture Model** — exercises Mix/Switch combinator + discrete variables.
   Compare Gibbs, involutive MCMC (ProposalEdit). Report: mixing time, ESS,
   correctness of component assignment.

**Minimum 2 system comparisons:**
- Gen.jl (Julia) — the reference GFI implementation (same hardware, CPU)
- At least one of: GenJAX, Pyro, NumPyro, Turing.jl

**Required ablation:** Vectorized vs sequential speedup curve (N = 1, 10, 100, 1000).

**Methodology:** Mean ± std over minimum 10 runs. Paired tests for system
comparisons. Report both time and accuracy.

---

## What Must Be Created

| Item | Effort | Priority |
|------|--------|----------|
| Benchmark suite (Models 1-3 with timing harness) | Large | Critical |
| Gen.jl comparison (install, implement same models, run) | Large | Critical |
| Figures: architecture diagram, broadcasting dataflow, speedup plots, convergence curves | Medium | Critical |
| Vectorization scaling study (N=1 to N=1000) | Medium | Critical |
| Hardware profiling (GPU vs CPU time, memory) | Small | Important |
| Inference code examples (5-line snippets for §6) | Small | Important |
| LaTeX manuscript in TOPML template | Medium | Required |

## What Already Exists

| Item | Source |
|------|--------|
| System implementation | src/genmlx/ (~10,800 lines) |
| Compatibility tests | test/genmlx/*_compat_test.cljs |
| Vectorization speedup numbers | test/genmlx/vectorized_benchmark.cljs |
| GPU benchmarks (exploratory) | test/genmlx/gpu_benchmark.cljs |
| Formal proofs (appendix material) | formal/ (17 files, 6,652 lines) |
| Architecture description | ARCHITECTURE.md, CLAUDE.md |
| Model examples | README.md, test files |
| Distribution catalog | src/genmlx/dist.cljs |

---

## Gap Analysis

| Requirement | Status | Gap |
|---|---|---|
| Contribution statement | Implicit in draft | Sharpen for abstract; state POPL gap |
| Evaluation section | Speedup numbers exist, no formal benchmarks | **Major gap**: need proper benchmark suite |
| Related work | Scattered references | **Major gap**: need 2-3 page structured section |
| Formal foundations (§5) | formal/ complete | Condense to 3 pages for this paper |
| Anonymization | Not started | Straightforward |
| Reproducibility | Code exists, no artifact | Document reproduction steps, PRNG seeds |

---

## Reviewer Objections to Anticipate

1. **"ClojureScript is niche — who will use this?"**
   Response: The contribution is the broadcasting approach and the formal
   foundation, not the language choice. The technique applies to any framework
   with element-wise operations.

2. **"No comparison with GenJAX on NVIDIA hardware."**
   Response: Acknowledge this limitation. Compare with Gen.jl on the same
   hardware. The point is "competitive on consumer hardware without CUDA."

3. **"Float32 only — can you do real statistics?"**
   Response: Float32 is sufficient for most Bayesian inference. Stan uses
   Float64 but acknowledges Float32 is often adequate. Report any numerical
   issues encountered.

4. **"The formal proofs are in an appendix — is this really a theory contribution?"**
   Response: Primary contribution is the system and broadcasting approach. Formal
   proofs provide confidence. For full treatment, see companion Paper 2.

5. **"Only 27 distributions — Pyro has hundreds."**
   Response: The `defdist` macro makes adding distributions trivial (10 lines).
   27 covers the standard set for Bayesian modeling. Open multimethod design
   means users extend without modifying core.

---

## Effort Estimate

- Anonymize and reformat for ACM template: 1 day
- Rewrite abstract + introduction for TOPML audience: 1 day
- Expand inference algorithms section: 1.5 days
- Write formal foundations section: 1 day
- Run benchmarks and write evaluation: 3-4 days
- Compress + restructure architecture: 1 day
- Related work: 0.5 days
- Polish: 1 day

**Total: ~10-11 days**
