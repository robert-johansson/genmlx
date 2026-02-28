# Paper 2: Formal Correctness of Generative Function Implementations

**Title:** "Formal Correctness of the Generative Function Interface: From Denotational Semantics to Runtime Contracts"

**Target venue:** [ACM Transactions on Probabilistic Machine Learning (TOPML)](https://dl.acm.org/journal/topml) — regular submission (or companion to Paper 1)

**Status:** All formal material exists in `formal/` (17 files, 6,652 lines).
No paper draft yet.

---

## Venue Fit

TOPML's [Special Issue on Probabilistic Programming](https://dl.acm.org/pb-assets/static_journal_pages/topml/pdf/TOPML_CfP_SI_Probabilistic_Programming-1704743488800.pdf)
(Sept 2025) explicitly listed these topics:
- "semantics for probabilistic programming"
- "verification and testing probabilistic programming paradigms"
- "theoretical analysis"
- "design and implementation of probabilistic programming languages"

This paper hits all four. The special issue deadline passed, but regular
submissions covering these topics are clearly welcome.

**TOPML format requirements:**
- Double-anonymous review
- ACM LaTeX template (single-column)
- Gold Open Access
- No strict page limit (journal format — aim for 30-40 pages for this material)
- Submit via https://mc.manuscriptcentral.com/topml

---

## Thesis

> The Generative Function Interface (GFI) — the compositional abstraction underlying
> Gen.jl, GenJAX, and GenMLX — can be given a rigorous denotational semantics in
> quasi-Borel spaces, and key implementation invariants (weight correctness, handler
> soundness, combinator compositionality, vectorization correctness, inference
> algorithm correctness) can be formally proved. We present 15 theorems covering
> all 8 architectural layers, connect them to 11 executable runtime contracts
> verified over 13 canonical models, and demonstrate a multi-layer verification
> methodology from mathematical foundations to statistical testing.

This is the **first formal treatment of a PPL's compositional inference interface**.

---

## Why This Paper Is Strong for TOPML

1. **Novel contribution:** No PPL has a published formal specification of its
   inference interface. Gen.jl's PLDI paper describes the design but proves nothing.
   GenJAX has vectorization proofs (Awad et al. POPL 2026) but not GFI proofs.

2. **Breadth + depth:** 15 theorems spanning the full GFI (generate, update,
   regenerate, edit, project, assess, propose), all 10 combinators, vectorization,
   and 7 inference algorithms (MH, HMC, NUTS, SMCP3, VI, ADEV, kernel composition).
   No cherry-picking — this covers everything.

3. **The verification ladder:** A reusable methodology:
   ```
   Layer 1: Denotational semantics (QBS)
     ↓ mathematical meaning
   Layer 2: Formal proofs (15 theorems)
     ↓ implementation invariants
   Layer 3: Runtime contracts (11 contracts, 575 checks)
     ↓ statistical verification
   Layer 4: Property-based tests (153 properties, 9 files)
     ↓ randomized algebraic testing
   Layer 5: Compatibility tests (165 Gen.clj + 73 GenJAX)
     ↓ behavioral equivalence
   ```

4. **Pure functional architecture enables this.** The paper argues this is not
   coincidental — GenMLX was designed for formal verification. Compare to Gen.jl
   (mutable traces) and GenJAX (JAX tracing constraints) where equivalent proofs
   are significantly harder.

5. **TOPML cares about correctness.** The journal's scope includes "uncertainty
   quantification" and "stability." Formal correctness proofs for inference
   algorithms are directly relevant.

---

## Corrected Claims (from verification)

- Property-based tests: 153 (not 162)
- Source LOC: ~10,800
- Formal files: 16 formal files + 1 README = 17 total (13 proofs + 3 specs)

---

## Formal Results Summary

### Core results (the paper's backbone — 7 theorems)

| # | Result | Source | Pages est. |
|---|--------|--------|-----------|
| T1 | Handler soundness | `handler-soundness.md` | 2 |
| T2 | Generate weight correctness | `correctness.md` | 1.5 |
| T3 | Update weight correctness | `correctness.md` | 1.5 |
| T4 | Edit/backward duality | `edit-duality.md` | 2 |
| T5 | Combinator compositionality (C1-C5) | `combinators.md` | 2 |
| T6 | Broadcasting correctness (logical relations) | `broadcasting.md` | 2.5 |
| T7 | Diff-aware update correctness | `diff-update.md` | 1.5 |

### Inference algorithm results (8 theorems)

| # | Result | Source | Pages est. |
|---|--------|--------|-----------|
| T8 | MH kernel preserves target | `kernel-composition.md` | 1 |
| T9 | Chain/cycle/mix preserve stationarity | `kernel-composition.md` | 1 |
| T10 | ADEV surrogate gradient correctness | `adev.md` | 1.5 |
| T11 | HMC preserves target (symplecticity) | `hmc-nuts.md` | 1.5 |
| T12 | NUTS detailed balance | `hmc-nuts.md` | 1 |
| T13 | SMCP3 weight correctness | `smcp3.md` | 1.5 |
| T14 | ELBO bound + IWELBO tightness | `vi.md` | 1 |
| T15 | Deterministic GFs satisfy C1-C5 | `deterministic-gf.md` | 0.5 |

---

## Paper Outline (target: 35-40 pages, TOPML journal format)

```
1. Introduction (3 pages)
   - PPLs need compositional interfaces for programmable inference
   - The GFI: adopted by 3 implementations across 3 languages and 3 hardware
     targets, yet never formally specified
   - Contributions: complete denotational semantics, 15 theorems, 11 runtime
     contracts, a reusable verification methodology
   - This is the first formal treatment of a PPL's inference interface

2. Background (3 pages)
   2.1 The Generative Function Interface
       - 8 operations with informal semantics
       - Compositional: distributions, models, combinators implement same interface
       - Why the GFI is worth formalizing (3 implementations, de facto standard)
   2.2 Quasi-Borel Spaces
       - Why QBS over σ-algebras (function spaces, higher-order programs)
       - Stock measures, morphisms, QBS products
       - Prior work: Heunen et al. 2017, Scibior et al. 2018
   2.3 The System Under Study
       - Brief description of GenMLX (purely functional, handler-based)
       - Why pure functional design enables formalization

3. The λ_MLX Calculus (4 pages)
   3.1 Types
       - Base types, batch types ([N]-arrays), ground types
       - Trace types γ (graded by address → distribution type)
       - Handler state types (σ_sim, σ_gen, σ_upd, σ_reg, σ_proj, σ_adev)
       - Edit request types, diff types, selection types
   3.2 Terms
       - Standard λ-calculus + trace, splice, param
       - Edit requests (ConstraintEdit, SelectionEdit, ProposalEdit)
       - ADEV terms (adev-surrogate, stop-gradient)
   3.3 Typing Rules
       - Handler computation rules
       - Splice rules with state scoping

4. Denotational Semantics (4 pages)
   4.1 QBS Interpretation of Types
   4.2 Stock Measures
   4.3 Term Denotations
   4.4 Handler Transition Semantics (all 6 modes)
       - Simulate, Generate, Update, Regenerate, Project, ADEV
   4.5 Splice Semantics (scalar and batched)
   4.6 Program Transformations

5. Core Correctness (6 pages)
   5.1 Handler Soundness (T1)
       - Statement: handler execution = denotation of transformation
       - Proof: induction on number of trace operations
       - Corollary: handler is observationally pure despite volatile!
   5.2 Generate Weight Correctness (T2)
       - weight = Σ constrained log-densities
       - Proof: structural induction on terms
   5.3 Update Weight Correctness (T3)
       - weight = new_score - old_score (telescoping sum)
       - Constraint propagation lemma
   5.4 Edit/Backward Duality (T4)
       - ConstraintEdit: backward recovers original trace
       - SelectionEdit: MH weight is correct
       - ProposalEdit: weight = update + backward - forward

6. Compositionality (4 pages)
   6.1 GFI Contracts C1-C5
       - Score consistency, generate weight, update weight,
         regenerate weight, discard completeness
   6.2 Combinator Compositionality (T5)
       - Theorem: inner GF satisfies C1-C5 ⟹ combinator-wrapped GF does too
       - Proof per combinator (Map, Unfold, Switch, Scan, Mask, Mix,
         Recurse, Contramap, Dimap)
   6.3 Deterministic GFs Satisfy C1-C5 Trivially (T15)
   6.4 Diff-Aware Update Correctness (T7)
       - MapCombinator: only re-execute changed elements
       - Unfold/Scan: prefix-skip optimization

7. Broadcasting Correctness (3 pages)
   7.1 Logical Relations R_τ
       - Definition for all types
   7.2 Key Lemmas
       - Batch sampling independence
       - Log-prob broadcasting
       - Score accumulation broadcasting
       - Handler shape-agnosticism
   7.3 Broadcasting Correctness Theorem (T6)
       - Statement and proof (induction + lemmas)
   7.4 Corollary: Vectorized Inference Commutes

8. Inference Algorithm Correctness (5 pages)
   8.1 Kernel Stationarity
       - MH from regenerate preserves target (T8)
       - Chain/cycle/mix composition preserves stationarity (T9)
   8.2 Hamiltonian Monte Carlo
       - Leapfrog symplecticity → volume preservation
       - HMC preserves target (T11)
       - NUTS detailed balance via symmetric doubling (T12)
   8.3 SMCP3 Weight Correctness (T13)
       - Init weights, incremental step weights
       - Unbiased log-marginal-likelihood estimation
       - Rejuvenation preserves target (via T8-T9)
   8.4 ADEV Gradient Correctness (T10)
       - Reparameterization trick
       - REINFORCE estimator
       - Surrogate loss: mixed estimator has correct gradient
   8.5 Variational Inference (T14)
       - ELBO is a lower bound (Jensen's inequality)
       - IWELBO tightens monotonically in K

9. Runtime Verification (3 pages)
   9.1 The 11 Executable Contracts
       - Table mapping each contract to the theorem(s) that establish it
   9.2 Statistical Verification Power
       - n=50 trials, ε=0.1 violation rate → 99.5% detection probability
       - Tolerance justification (float32 precision)
   9.3 Contract Completeness Analysis
       - Properties covered (10 GFI properties)
       - 5 recommended additional contracts (G1-G5)
   9.4 Property-Based Tests as Empirical Proofs
       - 153 properties across 9 test files
       - Connection to formal theorems
   9.5 The Verification Ladder
       - How the 5 layers reinforce each other
       - As a reusable methodology for PPL development

10. Related Work (2 pages)
    10.1 Formal Semantics of PPLs
         - Staton et al. 2016 (commutative semantics)
         - Heunen et al. 2017 (QBS)
         - Scibior et al. 2018 (denotational validation)
         - Vákár et al. 2019 (differentiable programming semantics)
    10.2 Verified Probabilistic Inference
         - Beutner et al. 2022 (verified SMC)
         - Lee et al. 2020 (verified MCMC)
         - Bichsel et al. 2018 (verified probabilistic inference)
    10.3 Compositional Inference Interfaces
         - Gen.jl (Cusumano-Towner et al. 2019) — design but no proofs
         - GenJAX vectorization (Awad et al. 2025) — partial proofs
    10.4 Runtime Verification and Testing
         - Property-based testing for probabilistic programs
         - Statistical verification

11. Conclusion and Future Work (1 page)
    - Summary: first complete formal treatment of the GFI
    - Lean4 mechanization as next step
    - Certified inference with runtime proof witnesses
    - The verification ladder as methodology

Appendix A: Full Proof Details (proofs sketched in main body)
Appendix B: Complete Contract-Theorem Mapping Table
Appendix C: Property-Based Test Catalog
```

---

## Key Arguments for TOPML Reviewers

### 1. Relevance to probabilistic ML

This isn't just PL theory. The theorems directly concern MCMC correctness,
importance weight computation, SMC unbiasedness, and VI objective properties.
TOPML reviewers who work with these algorithms daily will appreciate formal
guarantees that their implementations are correct.

### 2. Practical impact via runtime contracts

The paper doesn't stop at proofs. The 11 runtime contracts provide a practical
tool: run `verify-gfi-contracts` on any new model to check that it satisfies
the formally-proved invariants. This is usable today, not aspirational.

### 3. The verification ladder is generalizable

Other PPL developers can adopt the methodology: formalize the interface, derive
contracts from the proofs, test with property-based testing, validate against
reference implementations. Each layer catches different classes of bugs.

---

## Relationship to Paper 1 (System Paper)

Two submission strategies:

**Option A: Simultaneous submission as companion papers.**
Paper 1 describes the system; Paper 2 proves it correct. Each references the
other. TOPML reviewers see the full picture.

**Option B: Sequential submission.**
Submit Paper 1 first. After acceptance, submit Paper 2 referencing the published
system paper. Lower risk but slower.

**Recommendation: Option A.** The papers are genuinely complementary and stronger
together. The system paper's Section 5 (Formal Foundations) is a 3-page preview
of Paper 2; reviewers of Paper 1 will want the full treatment.

---

## Effort Estimate

- Write introduction and motivation: 1.5 days
- Condense calculus + semantics for paper format: 2.5 days
- Write core correctness section (T1-T4, T7): 2 days
- Write compositionality + broadcasting (T5, T6, T15): 2 days
- Write inference algorithm section (T8-T14): 2 days
- Write runtime verification section: 1 day
- Related work and conclusion: 1 day
- Anonymize and format for ACM template: 1 day
- Polish: 1 day

**Total: ~14 days** (longer than before — TOPML journal format demands more
polish and thoroughness than a conference paper)
