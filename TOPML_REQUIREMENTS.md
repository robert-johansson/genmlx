# Requirements for TOPML Submission

> What must be in the paper. If every item below is addressed, I believe
> the paper is publishable at ACM TOPML. If any Tier 1 item is missing,
> it is not.

---

## Venue Profile

**ACM Transactions on Probabilistic Machine Learning (TOPML)**
- 20–30 pages, single-column, Gold Open Access ($1,300 APC for ACM members)
- Values: bridging theory and practice, reproducibility, interdisciplinary
- Editorial board: strong Bayesian computation + statistical ML theory
- Had a special issue on probabilistic programming (guest editors:
  Katoen/RWTH, Rainforth/Oxford, Yang/KAIST) — PP is explicitly in scope
- Inaugural issue Feb 2025; Vol 1 No 3 includes PP-related papers

**The POPL 2026 paper (Becker et al.) formalizes:**
- λ_GEN calculus: types, terms, QBS denotational semantics
- Two program transformations only: simulate{−} and assess{−}
- Proposition 3.1: correctness of simulate and assess
- vmap_n as source-to-source transformation with logical relations proof
- Theorem 3.3: vmap_n correctness; Corollary 3.4: commutativity
- Stochastic branching via cond/select

**The POPL 2026 paper does NOT formalize:**
- generate, update, regenerate, project, propose, edit
- Handler-based execution (their implementation uses JAX tracing, not handlers)
- Combinators beyond scan (no Map, Unfold, Switch, Mask, Mix, Recurse)
- Diff-aware incremental computation
- Broadcasting as an alternative to vmap

This gap is the paper's opportunity.

---

## Tier 1: Required for Publication

Every item below must be fully addressed. A missing Tier 1 item is
grounds for rejection.

### 1.1 Clear Contribution Statement

The paper must state exactly what is new beyond the POPL 2026 paper.
The contribution is NOT "we ported Gen to MLX" — that is a systems
contribution insufficient for TOPML. The contribution must be:

**The first complete formalization of the Generative Function Interface.**

The POPL paper formalizes 2 of 8 GFI operations. This paper formalizes
all 8: simulate, assess, generate, update, regenerate, project, propose,
edit. It also formalizes the handler-based execution model, proves
combinator compositionality, proves broadcasting correctness (as an
alternative to vmap), and proves edit/backward duality for MCMC.

This must be stated in the abstract and introduction without hedging.

### 1.2 Formal Results: Three Headline Theorems

The paper needs at minimum three self-contained, precisely-stated
theorems with complete proofs. "Complete" means: all hypotheses stated,
all proof steps justified by cited lemmas or direct computation, no
"by inspection" or "analogous to the other case."

**Theorem A (GFI Correctness).** For each GFI operation
op ∈ {generate, update, regenerate, project, propose, edit}, and for
any closed well-typed term t : G_γ η with denotation (μ, f) = ⟦t⟧:

The program transformation op{t} correctly implements the intended
semantic operation — specifically:

- generate: weight = marginal density of observations under μ
- update: weight = score(new trace) − score(old trace)
- regenerate: weight = log MH acceptance ratio for the resample move
- project: weight = log-density at selected addresses
- propose: equivalent to simulate (returns score as weight)
- edit: weight satisfies the edit-specific contract (constraint/
  selection/proposal)

Proof must be by structural induction on t, with explicit base case
(trace(k, d)) and inductive case (do_G{x ← t₁; t₂}), for each
operation. The generate and update cases must handle all three
sub-cases (constrained, keep-old, fresh-sample) with explicit
density computations.

**Theorem B (Broadcasting Correctness).** For a generative function
g : G_γ η using only broadcastable distributions, and for
N ∈ ℕ:

⟦m{g_N}⟧ ≡ zip_R ∘ (⟦m{g}⟧)^{⊗N}

for m ∈ {simulate, generate, update, regenerate}, where g_N denotes
batched execution with N particles.

The proof must:
1. Define the logical relations R_τ ⊆ ⟦τ⟧^N × ⟦τ[N]⟧ precisely
2. State and prove four lemmas:
   - Batch sampling independence (dist-sample-n)
   - Log-prob broadcasting (element-wise density)
   - Score accumulation broadcasting
   - Handler shape-agnosticism (formal definition required)
3. State the preconditions on the model body: all operations used
   must commute with broadcasting (no mx/item, mx/shape on values,
   no control flow branching on array values)
4. Explicitly list which distributions are excluded and why (wishart,
   inv-wishart use mx/ndim/mx/shape on values)
5. Handle the splice case (batched sub-GF execution)

**Theorem C (Edit/Backward Duality).** For each EditRequest type:

- ConstraintEdit: backward edit recovers original trace; weight negates
- SelectionEdit: regenerate weight satisfies MH detailed balance
- ProposalEdit: forward-backward swap satisfies detailed balance

The ProposalEdit proof must:
1. Define the proposal consistency conditions precisely (support
   coverage, backward support matching, bijection condition)
2. Derive the MH acceptance ratio from the weight formula
   w_upd + w_b − w_f
3. Show that the backward weight is the negative of the forward weight
   (under consistency)
4. State when consistency fails and what happens (rejection, not
   incorrectness)

### 1.3 Formal Definitions: Complete Calculus and Semantics

The paper must contain:

**Calculus (1–2 pages).** Full type grammar and term grammar for λ_MLX
extending λ_GEN. Must include:
- Handler state types H(σ, τ) with explicit state schemas
- EditRequest types
- Typing rules for all new terms (update, regenerate, edit, splice, fix)
- Each typing rule must be a displayed inference rule, not prose

**Denotational semantics (1–2 pages).** QBS interpretation of all types.
Must include:
- Statement that ⟦G_γ η⟧ = 𝒫_≪(⟦γ⟧) × (⟦γ⟧ → ⟦η⟧) forms a QBS
  (cite Heunen et al. 2017 for the probability monad on QBS)
- Handler transition semantics for all 5 modes as explicit state
  transitions in H(σ, τ)
- Splice semantics showing address scoping

**Program transformations (2–3 pages).** All 8 GFI operations as
source-to-source transformations. Must include type transformation
and term transformation rules for generate, update, regenerate,
project, propose, edit — not just simulate and assess.

### 1.4 Combinator Compositionality

**Theorem D (not a headline but required).** If g satisfies the GFI
contract, then C(g) satisfies the GFI contract for each combinator
C ∈ {Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap, Dimap}.

This can go in an appendix, but the statement must be in the main
paper body. The proof for Map and Unfold must be in the main paper.
Mix and Recurse proofs can be in the appendix but must exist.

Specifically:
- Map: prove score additivity over independent elements
- Unfold: prove chain-rule factorization with carry state, explain
  why the stock measure is still a product despite sequential dependence
- Mix: handle same-component and different-component update cases
  with explicit weight formulas
- Recurse: state the well-foundedness condition; prove by induction on
  recursion depth

### 1.5 Experimental Evaluation

TOPML requires empirical validation. The experiments must demonstrate
that the formal guarantees hold in practice and that the system is
competitive.

**Required benchmarks (minimum 3):**

1. **Hidden Markov Model (HMM)** — standard PP benchmark. Compare:
   sequential IS, vectorized IS, SMC. Report: log-marginal-likelihood
   accuracy, wall-clock time, ESS. This exercises Unfold + generate +
   importance weighting.

2. **Bayesian Linear/Polynomial Regression** — demonstrates full
   inference pipeline. Compare: MH, HMC, NUTS, variational inference.
   Report: posterior accuracy (compare to analytic solution where
   available), convergence speed, samples/second.

3. **Gaussian Mixture Model or Switching State-Space Model** — exercises
   Mix or Switch combinator + discrete variables. Compare: Gibbs,
   involutive MCMC (ProposalEdit). Report: mixing time, effective
   sample size, correctness of component assignment.

**Required comparisons (minimum 2 systems):**

- Gen.jl (Julia) — the reference implementation of GFI
- At least one of: GenJAX (JAX), Pyro (PyTorch), NumPyro (JAX), Turing.jl

For each comparison: same model, same inference algorithm, same number
of samples/particles. Report wall-clock time and statistical accuracy
(e.g., KL divergence to ground truth or analytic posterior).

**Required ablation:**

- Vectorized vs sequential: show the N-particle speedup curve (N = 1,
  10, 100, 1000). Must be on the same model, same machine, reporting
  time per effective sample.

**Hardware and reproducibility:**

- All experiments on Apple Silicon (M1/M2/M3/M4 — specify which)
- Report: GPU memory usage, whether results are deterministic given
  PRNG seed
- Code must be available (GitHub link or supplementary archive)

### 1.6 Related Work

Must discuss and precisely differentiate from:

1. **Becker et al. 2026 (GenJAX/λ_GEN, POPL)** — the paper we extend.
   State exactly which results are inherited, which are new. Do not
   re-prove Prop 3.1 or Theorem 3.3 — cite them.

2. **Cusumano-Towner et al. 2019 (Gen.jl)** — the GFI was introduced
   here. Our formalization is of their interface. Acknowledge this.

3. **Ścibior et al. 2018 (Denotational validation of higher-order
   Bayesian inference)** — QBS for probabilistic programming. Our
   semantics builds on their framework.

4. **Heunen et al. 2017 (A convenient category for higher-order
   probability theory)** — the QBS paper. We use their results for
   function spaces and probability monad.

5. **Lew et al. 2023 (ADEV)** — automatic differentiation of expected
   values. Relevant to our gradient infrastructure.

6. **Bingham et al. 2019 (Pyro)** — effect-handler-based PPL. Compare
   our handler architecture to Pyro's Poutine system.

7. **Ge et al. 2018 (Turing.jl)** — Julia PPL with different design
   (compiler-based vs handler-based).

8. **Involutive MCMC literature** — Neklyudov et al. 2020,
   Cusumano-Towner 2020. Our ProposalEdit generalizes this.

9. **Incremental computation in PP** — Wingate et al. 2011 (lightweight
   implementations), Mansinghka et al. 2014 (Venture). Our diff-aware
   update is related.

Must be 2–3 pages. Each comparison must state what the other work does
and does NOT do, and what we do differently. No strawman comparisons.

### 1.7 Reproducibility

- All code in a public repository with a README that reproduces every
  figure and table
- Specify exact software versions: Node.js, nbb, @frost-beta/mlx,
  macOS version
- Include PRNG seeds for all stochastic experiments
- Include raw data for all reported numbers (not just summaries)

---

## Tier 2: Strongly Recommended

Missing Tier 2 items will not cause rejection but will weaken the paper
significantly. Reviewers will note their absence.

### 2.1 Handler Soundness Theorem

**Theorem.** For each handler mode m, `run-handler(m-transition, σ₀, body)`
equals the denotation m{body}(σ₀). Proof by induction on the number
of trace operations.

This theorem connects the implementation (volatile!-based handler
dispatch) to the denotational semantics (state monad). Without it,
there is a gap between "we specified the semantics" and "our code
implements the semantics."

Must include: volatile! invisibility argument (state threading
equivalence), splice soundness (nested run-handler isolation for both
scalar and batched modes).

### 2.2 Diff-Aware Update Correctness

**Theorem.** For MapCombinator with VectorDiff(S), update-with-diffs
produces the same result as full update but only re-executes elements
in S ∪ C. For Unfold/Scan with prefix skipping, same but skipping
prefix steps.

This is the efficiency story: MCMC with O(changed) cost instead of
O(model size). Important for practical relevance but not a formal
contribution per se (it's an optimization, not a new semantic result).

### 2.3 Metatheorem (Unifying Statement)

A single theorem that subsumes Theorems A–D:

**Theorem (Full GFI Correctness).** For any well-typed λ_MLX program
composed of generative functions and combinators, executing under the
handler system produces traces, scores, and weights that equal the
denotational semantics. Broadcasting preserves this correspondence for
N independent particles.

This would be the paper's "crown jewel" — but it requires Theorems A–D
to all be proven first, and then a composition argument. If the
individual theorems are strong, this metatheorem can be stated and
its proof sketched.

### 2.4 Worked Example (End-to-End)

Take a concrete model (e.g., a 3-site Bayesian regression) and walk
through:
1. The λ_MLX typing derivation
2. The generate{−} transformation applied to it
3. The handler execution trace (state transitions step by step)
4. The broadcasting lift to N=4 particles
5. The weights computed, showing they match the semantic specification

This is 2–3 pages but dramatically improves readability. Reviewers
who cannot follow the formal development will follow the example.

### 2.5 QBS Construction Verification

For each type constructor used (products, function spaces, probability
measures), verify that the QBS axioms are satisfied. Cite specific
results from Heunen et al. 2017 and Ścibior et al. 2018.

Specifically:
- ⟦G_γ η⟧ = 𝒫_≪(⟦γ⟧) × (⟦γ⟧ → ⟦η⟧) must be shown to be a QBS
- The handler type ⟦H(σ,τ)⟧ = ⟦σ⟧ → ⟦τ⟧ × ⟦σ⟧ must be a QBS morphism
  space (not just a set of functions)
- The probability monad on QBS (from Heunen et al.) must be cited for
  the do_P and do_G sequencing

### 2.6 Numerical Stability Discussion

A paragraph or two addressing:
- All densities are computed in log-space (no underflow)
- mx/add for score accumulation (not mx/multiply)
- logsumexp for mixture weights
- What happens at measure-zero constraints (log(0) = -∞, weight = -∞,
  MH always rejects — this is correct behavior, not a bug)

---

## Tier 3: Nice to Have

These strengthen the paper but their absence will not be noticed by
most reviewers.

### 3.1 Comparison to Pyro's Effect Handler System

Our handler architecture (pure transitions + volatile! wrapper) is
similar to Pyro's Poutine system but with a key difference: our
transitions are pure functions, enabling formal reasoning. A 1-page
comparison would position us well.

### 3.2 Automatic Differentiation Integration

The paper can mention gradient computation (choice gradients, score
gradients, ELBO gradients) as demonstrating that the formal framework
extends to differentiable inference. No formal treatment needed — just
show it works.

### 3.3 Extension to Continuous-Time Models

Mention as future work: the handler architecture could support
continuous-time processes (SDEs) if dist-sample-n is extended to
SDE solvers. This connects to the broader TOPML audience.

### 3.4 Performance on M-Series Chips

If benchmarks are run on M1, M2, M3, and M4, showing scaling across
Apple Silicon generations, this is interesting to the TOPML audience
(practical hardware story).

---

## What We Already Have (Gap Analysis)

| Requirement | Status | Gap |
|---|---|---|
| **1.1** Contribution statement | Implicit in LAMBDA_MLX.md | Needs to be sharpened for abstract |
| **1.2** Theorem A (GFI correctness) | Proof sketches in proofs/correctness.md | Needs: all hypotheses explicit, generate/update/regenerate/project/edit all proved, no "analogous" shortcuts |
| **1.3** Calculus + semantics | calculus.md + semantics.md complete | Needs: displayed inference rules (not prose), QBS construction citations |
| **1.2** Theorem B (broadcasting) | proofs/broadcasting.md with categories | Needs: model-body precondition stated, full inductive proof, splice case |
| **1.2** Theorem C (edit duality) | proofs/edit-duality.md with consistency conditions | Needs: full derivation of ProposalEdit weight, not just sketch |
| **1.4** Combinators | proofs/combinators.md | Needs: Map and Unfold expanded to full proofs in main paper |
| **1.5** Experiments | Speedup numbers exist, no formal benchmarks | **Major gap**: need proper benchmark suite with comparisons |
| **1.6** Related work | Scattered references | **Major gap**: need 2-3 page structured section |
| **1.7** Reproducibility | Code exists, no artifact | Need: documented reproduction steps |
| **2.1** Handler soundness | proofs/handler-soundness.md | Exists, needs polish |
| **2.2** Diff-aware update | proofs/diff-update.md | Exists, needs polish |
| **2.3** Metatheorem | Does not exist | Would be new work |
| **2.4** Worked example | Does not exist | Would be new work |
| **2.5** QBS verification | §1.5 of semantics.md | Needs expansion with citations |

**Largest gaps:**
1. Experimental evaluation (nothing publishable exists)
2. Related work section (not written)
3. Proofs need to be upgraded from semi-formal to fully rigorous
4. Model-body precondition for broadcasting never stated

---

## Suggested Paper Structure

```
1. Introduction                              (2 pages)
   - Contribution statement
   - Comparison to POPL 2026 paper

2. Overview                                  (2 pages)
   - Running example (Bayesian regression)
   - GFI operations on the example

3. The λ_MLX Calculus                        (3 pages)
   - Types (extending λ_GEN Figure 10)
   - Terms and typing rules
   - Handler state types

4. Denotational Semantics                    (3 pages)
   - QBS interpretation
   - Handler transition semantics
   - Splice semantics

5. Program Transformations                   (3 pages)
   - generate{−}, update{−}, regenerate{−}
   - edit{−} (constraint, selection, proposal)

6. Correctness Results                       (5 pages)
   - Theorem A: GFI correctness (main proofs)
   - Theorem C: Edit/backward duality
   - Theorem D: Combinator compositionality (statement + Map/Unfold proofs)

7. Broadcasting Correctness                  (3 pages)
   - Logical relations
   - Lemmas (batch sampling, log-prob, shape-agnosticism)
   - Theorem B

8. Implementation                            (2 pages)
   - Handler architecture
   - MLX backend
   - Vectorized inference

9. Evaluation                                (4 pages)
   - Benchmarks (HMM, regression, mixture)
   - Comparisons (Gen.jl, GenJAX or Pyro)
   - Vectorization ablation

10. Related Work                             (2 pages)

11. Conclusion                               (0.5 pages)

Appendix (supplementary):
   - Full proofs for Mix, Recurse, Scan, Mask, Contramap, Dimap
   - Handler soundness proof
   - Diff-aware update proof
   - Worked example (end-to-end)
   - All benchmark raw data
                                    Total: ~30 pages + appendix
```

---

## Non-Negotiable Quality Standards

1. **Every theorem must have all hypotheses stated.** No implicit
   assumptions. If broadcasting requires broadcastable distributions,
   say so in the theorem statement, not in a remark afterward.

2. **No "by inspection" in proofs.** Replace with explicit case analysis
   or citation to a specific code location with a checkable property.

3. **Every weight formula must be derived, not asserted.** The reader
   must be able to verify that weight = log(MH ratio) by following
   the algebra. This is the core of the paper's value.

4. **Implementation correspondence must be machine-checkable in
   principle.** Every formal rule should cite a specific file and line
   range. A reviewer should be able to open the code and verify the
   correspondence.

5. **Experiments must be statistically sound.** Report mean ± std over
   multiple runs (minimum 10). Use paired tests when comparing systems.
   Report both time and accuracy — a fast wrong answer is worthless.

6. **Do not overclaim.** The broadcasting theorem has exceptions
   (wishart, inv-wishart). The model body must use broadcastable
   operations. Splice in batched mode only supports DynamicGF. State
   these limitations clearly. Reviewers respect honesty; they punish
   discovered overclaims.
