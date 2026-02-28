# Contracts Linkage — TODO 10.20

> Maps each of the 11 executable GFI contracts to the formal theorem(s)
> that establish its correctness. Quantifies statistical verification
> power and identifies gaps in contract coverage.

---

## 1. Contract-Theorem Mapping

Each executable contract in `contracts.cljs` is a runtime check for a
measure-theoretic invariant that is formally established by one or more
proofs in this directory.

### 1.1 Complete Mapping Table

| # | Contract | Formal Basis | Location |
|---|----------|-------------|----------|
| 1 | `:generate-weight-equals-score` | Generate weight = Σ constrained log-densities. When fully constrained, weight = score. | `correctness.md` §2 |
| 2 | `:update-empty-identity` | Update with same choices ⟹ weight = 0 (no density change). Base + inductive case: all per-site ratios are 1. | `correctness.md` §3, base case 2 |
| 3 | `:update-weight-correctness` | Update weight = new_score - old_score. Telescoping sum of per-site log-density ratios. | `correctness.md` §3, inductive case |
| 4 | `:update-round-trip` | Forward edit + backward edit recovers original. ConstraintEdit duality: discard contains old values, applying discard recovers them. | `edit-duality.md` §2.1 |
| 5 | `:regenerate-empty-identity` | Regenerate with empty selection ⟹ no addresses resampled ⟹ weight = 0 and choices unchanged. | `semantics.md` §4.4 (not-selected branch) |
| 6 | `:project-all-equals-score` | Project with all-selection accumulates density at every address = total score. | `transformations.md` §5, `semantics.md` §4.5 |
| 7 | `:project-none-equals-zero` | Project with none-selection accumulates nothing = 0. | `transformations.md` §5, `semantics.md` §4.5 |
| 8 | `:assess-equals-generate-score` | Assess computes the same density as generate's score field, since both evaluate the joint log-density at the given choices. | `transformations.md` §§1.2, 2 |
| 9 | `:propose-generate-round-trip` | Propose produces valid choices; generating with those choices yields a trace with finite weight (the choices are in the model's support). | `transformations.md` §6 |
| 10 | `:score-decomposition` | Score decomposes as sum of per-address log-densities. Project at individual addresses sums to total score. | `combinators.md` §1 (C1), `correctness.md` §2 |
| 11 | `:broadcast-equivalence` | Vectorized simulate produces [N]-shaped scores that are individually valid. Broadcasting preserves per-particle correctness. | `broadcasting.md` §4 |

### 1.2 Detailed Justifications

**Contract 1: generate-weight-equals-score** (`contracts.cljs:59-64`)

The contract tests: `generate(model, args, trace.choices).weight ≈ trace.score`
when the full choice map is provided as constraints.

Formal basis: From `correctness.md` §2, generate accumulates weight
at constrained addresses: `weight = Σ_{a ∈ obs} log density_a(obs(a))`.
When obs = all choices (fully constrained), every address is constrained,
and weight = Σ_all log density_a = score. ✓

**Contract 2: update-empty-identity** (`contracts.cljs:66-71`)

The contract tests: `update(model, trace, trace.choices).weight ≈ 0`.

Formal basis: From `correctness.md` §3, update weight = Σ_a [log p(new(a))
- log p(old(a))]. When new = old at every address, each term is 0,
so total weight = 0. ✓

**Contract 3: update-weight-correctness** (`contracts.cljs:73-85`)

The contract tests: `update.weight ≈ new_score - old_score`.

Formal basis: From `correctness.md` §3, the inductive case shows that
update weight telescopes to score(new_trace) - score(old_trace). ✓

**Contract 4: update-round-trip** (`contracts.cljs:87-101`)

The contract tests: update with constraint, then update with discard,
recovers original values.

Formal basis: From `edit-duality.md` §2.1, ConstraintEdit duality
proves that applying the backward edit (ConstraintEdit(discard))
recovers the original trace. ✓

**Contract 5: regenerate-empty-identity** (`contracts.cljs:103-116`)

The contract tests: `regenerate(model, trace, sel/none).weight ≈ 0`
and choices are unchanged.

Formal basis: From `semantics.md` §4.4, when no addresses are selected
(a ∉ selected for all a), every address takes the "not selected" branch,
preserving old values and contributing weight ratio 1 (= 0 in log-space).
The regenerate weight at DynamicGF level is new_score - old_score -
proposal_ratio = 0 - 0 - 0 = 0. ✓

**Contract 6: project-all-equals-score** (`contracts.cljs:118-124`)

The contract tests: `project(model, trace, sel/all) ≈ trace.score`.

Formal basis: From `transformations.md` §5 and `semantics.md` §4.5,
project accumulates log-density at selected addresses. With all-selection,
every address is selected, giving Σ_all log density_a = score. ✓

**Contract 7: project-none-equals-zero** (`contracts.cljs:126-131`)

The contract tests: `project(model, trace, sel/none) ≈ 0`.

Formal basis: Same as above — with none-selection, no addresses
contribute, giving 0. ✓

**Contract 8: assess-equals-generate-score** (`contracts.cljs:133-140`)

The contract tests: `assess(model, args, choices).weight ≈
generate(model, args, choices).trace.score`.

Formal basis: From `transformations.md` §1.2, assess computes the joint
density at given choices: `Σ_a log density_a(choices(a))`. From §2,
generate with full constraints also accumulates this same quantity as
the score. Both compute the same sum over the same addresses with the
same values. ✓

**Contract 9: propose-generate-round-trip** (`contracts.cljs:142-152`)

The contract tests: propose produces choices, and generating with those
choices yields finite weight.

Formal basis: From `transformations.md` §6, propose samples from the
model prior, producing choices in the support of each distribution.
Generating with in-support choices produces finite log-densities at
every address, so the weight is finite. ✓

**Contract 10: score-decomposition** (`contracts.cljs:154-167`)

The contract tests: `Σ_a project(trace, {a}) ≈ trace.score`.

Formal basis: The score factorizes as Σ_a log density_a (from the
product structure of the stock measure, `correctness.md` §2). Projecting
at a single address a extracts log density_a. Summing over all
addresses recovers the total score. This is the C1 compositionality
property from `combinators.md` §1. ✓

**Contract 11: broadcast-equivalence** (`contracts.cljs:169-180`)

The contract tests: `vsimulate(model, args, N)` produces [N]-shaped
scores that are finite.

Formal basis: From `broadcasting.md` §4, the broadcasting correctness
theorem establishes that vectorized execution produces N independent
valid traces. Each trace has a valid score (finite log-density), so
the [N]-shaped score array contains N finite values. ✓

---

## 2. Statistical Verification Power

### 2.1 Detection Model

Each contract run is a Bernoulli trial with:
- **H₀:** The contract holds (violation rate ε = 0)
- **H₁:** The contract fails with probability ε > 0

With n independent trials, the probability of missing a violation is:

```
P(0 failures | n trials, violation rate ε) = (1 - ε)^n
```

### 2.2 Power Table

| n (trials) | ε (violation rate) | P(miss) | P(detect ≥ 1) |
|------------|-------------------|---------|---------------|
| 10 | 0.10 | 34.9% | 65.1% |
| 20 | 0.10 | 12.2% | 87.8% |
| 50 | 0.10 | 0.52% | **99.5%** |
| 50 | 0.05 | 7.69% | 92.3% |
| 50 | 0.01 | 60.5% | 39.5% |
| 100 | 0.05 | 0.59% | **99.4%** |
| 100 | 0.01 | 36.6% | 63.4% |

With the default n=50 trials used in `verify-gfi-contracts`
(`contracts.cljs:186-216`), a violation rate of 10% is detected with
99.5% probability.

### 2.3 Tolerance Justification

Contracts use approximate equality (`assert-close` with tolerance):

```
|actual - expected| < tolerance
```

The tolerance accounts for:
1. **Floating-point precision:** MLX float32 has ~7 decimal digits of
   precision. Operations accumulate rounding errors.
2. **Score accumulation:** Models with many trace sites accumulate more
   numerical error (error ∝ √n for n independent operations).
3. **Numerical stability of log-space operations:** `logsumexp` is
   numerically stable, but individual `log` and `exp` operations may
   lose precision near 0 or ∞.

Default tolerance: 1e-4 for scalar tests, 1e-3 for accumulated
quantities. These thresholds are well above float32 machine epsilon
(~1.2e-7) to avoid false positives while being tight enough to detect
genuine violations.

---

## 3. Contract Completeness Analysis

### 3.1 Properties Covered

The 11 contracts collectively verify:

| GFI Property | Contract(s) |
|-------------|-------------|
| simulate correctness | (implicit in generate + project) |
| generate weight | 1, 8 |
| update weight | 2, 3 |
| update reversibility | 4 |
| regenerate identity | 5 |
| project decomposition | 6, 7, 10 |
| assess consistency | 8 |
| propose validity | 9 |
| score factorization | 10 |
| broadcasting | 11 |

### 3.2 Properties NOT Covered

Several GFI properties lack runtime contracts:

**(G1) Regenerate weight = log MH ratio.**
The regenerate weight derivation (`semantics.md` §4.4) is verified
formally but not tested as a contract. A contract would need to
independently compute the MH ratio and compare.

**Recommended contract:**
```
:regenerate-weight-is-mh-ratio
  log(p(u')/p(u) · q(u|u')/q(u'|u)) ≈ regenerate.weight
```

**(G2) Edit/backward duality for ProposalEdit.**
While ConstraintEdit round-trip is tested (contract 4), ProposalEdit
duality is not. Testing this requires a pair of proposal GFs.

**Recommended contract:**
```
:proposal-edit-weight-sum
  forward-weight + backward-weight ≈ 0  (for involutive proposals)
```

**(G3) SMCP3 weight correctness.**
The SMCP3 incremental weights are not directly tested by contracts.
A contract could verify that the accumulated log-ML is consistent
with importance sampling estimates.

**(G4) ADEV surrogate gradient.**
The surrogate loss correctness (formal proof in `adev.md` §4) is not
runtime-tested. A contract could compare surrogate gradient to finite
differences.

**(G5) HMC detailed balance.**
The HMC kernel's stationarity (formal proof in `hmc-nuts.md` §3) is
not directly tested. A contract could run a long chain and verify
convergence to the correct posterior.

### 3.3 Coverage Summary

```
Formal files:     16 (13 proofs + 3 specifications) covering all 8 gap areas
Runtime contracts: 11 contracts covering core GFI operations
Gap:              5 recommended additional contracts (G1-G5)
```

---

## 4. Cross-References to New Proofs

The seven new proof files strengthen the formal foundation that
backs the contracts:

| New Proof File | Contracts Strengthened | How |
|---------------|----------------------|-----|
| `kernel-composition.md` | 4, 5 | Proves MH kernel preserves target via regenerate weight; strengthens round-trip understanding |
| `adev.md` | — | No direct contract, but proves ADEV surrogate correctness (recommended G4) |
| `deterministic-gf.md` | 1-10 | Proves deterministic GFs (score=0) satisfy all contracts trivially |
| `hmc-nuts.md` | — | No direct contract, but proves HMC detailed balance (recommended G5) |
| `vi.md` | — | No direct contract; ELBO/IWELBO are optimization objectives, not GFI contracts |
| `smcp3.md` | — | Proves SMCP3 weight correctness (recommended G3) |
| `contracts-linkage.md` | All | This file — provides the formal justification mapping |

### 4.1 Strengthening Path

The full verification story for GenMLX is:

```
Layer 1: Formal proofs (this directory)
  ↓ establishes mathematical correctness
Layer 2: Runtime contracts (contracts.cljs)
  ↓ verifies implementation matches formalization
Layer 3: Compatibility tests (gen_clj_compat, genjax_compat)
  ↓ verifies behavioral equivalence with reference implementations
Layer 4: Statistical tests (inference convergence, distribution tests)
  ↓ verifies practical inference quality
```

The new proof files close the gap between Layer 1 (which previously
covered only core GFI and broadcasting) and Layers 2-4 (which test
kernels, ADEV, VI, SMCP3, and HMC). With 16 formal files (13 proofs
+ 3 specifications), every major component of GenMLX now has a formal
correctness argument.

---

## 5. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| 11 GFI contracts | Contract functions | `contracts.cljs:59-180` |
| Contract registry | `contracts` var | `contracts.cljs:55` |
| Verification runner | `(verify-gfi-contracts ...)` | `contracts.cljs:186-216` |
| Tolerance checking | `(assert-close ...)` | Test utilities |
| n=50 default trials | `:or {n-trials 50}` | `contracts.cljs:192` |
| ESS computation | `exp(2·logsumexp - logsumexp(2·))` | `vectorized.cljs` |
| Systematic resampling | `u/systematic-resample` | `inference/util.cljs` |
