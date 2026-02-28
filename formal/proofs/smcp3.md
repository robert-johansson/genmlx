# Sequential Monte Carlo with Probabilistic Program Proposals (SMCP3) — TODO 10.19

> Theorem: SMCP3 produces unbiased log-marginal-likelihood estimates and
> correctly maintains importance weights through sequential proposal,
> resampling, and rejuvenation steps. Custom generative functions as
> proposals integrate via the edit interface.

---

## 1. Framework

### 1.1 Sequential Target Distributions

SMCP3 approximates a sequence of target distributions:

```
π₀(z₀), π₁(z₀:₁), …, π_T(z₀:T)
```

where each πₜ extends the previous target with new observations.
In GenMLX, each πₜ is defined by the model generative function
conditioned on observations up to time t.

### 1.2 Particle Approximation

At each step t, SMCP3 maintains N weighted particles:

```
{(z₀:t^{(k)}, w_t^{(k)})}_{k=1}^{N}
```

where z₀:t^{(k)} is the trace of particle k and w_t^{(k)} is its
unnormalized importance weight (in log-space).

---

## 2. SMCP3-Init: Initialization

### 2.1 Algorithm

SMCP3-Init (ref: `smcp3.cljs:22-59`) initializes N particles for
the first target π₀:

```
For k = 1, …, N:
  if proposal is provided:
    (choices_f, score_f) = propose(proposal, proposal-args)
    merged = merge(choices_f, observations₀)
    (trace_k, model_weight) = generate(model, args, merged)
    log w₀^{(k)} = model_weight - score_f
  else:
    (trace_k, log w₀^{(k)}) = generate(model, args, observations₀)
```

### 2.2 Theorem (Init Weight Correctness)

**Statement.** The initialization weights are correct importance weights
for π₀:

```
E[w₀^{(k)}] ∝ π₀(z₀^{(k)}) / q₀(z₀^{(k)})
```

where q₀ is either the proposal distribution or the prior.

**Proof.**

**Case 1: No proposal.** The particles are generated via
`generate(model, args, obs₀)`. By the correctness of generate
(`correctness.md` §2), the weight equals the marginal density of the
observations:

```
log w₀^{(k)} = Σ_{a ∈ obs₀} log density_a(obs₀(a))
```

This is the standard importance weight for p(x₀) under the prior
proposal q₀ = p(z₀), since:

```
w = p(x₀, z₀) / p(z₀) = p(x₀ | z₀) · p(z₀) / p(z₀) = p(x₀ | z₀)
```

which is the likelihood (marginal over the constrained observations). ✓

**Case 2: Custom proposal.** The proposal generates choices with score
`score_f = log q_f(choices_f)`. The merged constraints include both
proposed and observed values. The model weight from generate is:

```
model_weight = log p(obs₀ ∪ choices_f | rest of z₀)
```

The importance weight:

```
log w₀ = model_weight - score_f
       = log p(z₀ | model) - log q_f(z₀ | proposal) + log p(obs₀)
```

This is the correct importance weight for π₀(z₀) ∝ p(z₀, x₀) under
proposal q₀ = q_f:

```
w₀ = p(z₀, x₀) / q_f(z₀) = π₀(z₀) · Z₀ / q_f(z₀)
```

where Z₀ = p(x₀) is the marginal likelihood at time 0. ✓ ∎

---

## 3. SMCP3-Step: Incremental Update

### 3.1 Algorithm

SMCP3-Step (ref: `smcp3.cljs:65-124`) advances particles from πₜ₋₁ to πₜ:

```
For k = 1, …, N:
  1. [Optional] Resample if ESS < threshold (§4)
  2. Extend particle:
     if forward-kernel provided:
       (trace_k', edit_weight, _, _) = edit(model, trace_k,
                                             ProposalEdit(kernel, kernel-args, ...))
       log w_t^{(k)} += edit_weight
     else:
       (trace_k', update_weight) = update(model, trace_k, observations_t)
       log w_t^{(k)} += update_weight
  3. [Optional] Rejuvenate via MCMC (§5)
```

### 3.2 Theorem (Step Weight Correctness)

**Statement.** The incremental weight correctly updates the importance
weight from πₜ₋₁ to πₜ.

**Proof.**

**Case 1: ConstraintEdit (standard update).** By the correctness of
update (`correctness.md` §3):

```
update_weight = score(trace') - score(trace)
              = log πₜ(z₀:t) - log πₜ₋₁(z₀:t₋₁)
              + log πₜ₋₁(z₀:t₋₁) - log πₜ₋₁(z₀:t₋₁)
```

The new addresses (observations at time t) contribute their log-density,
and unchanged addresses contribute zero (their densities cancel). The
accumulated weight:

```
log w_t = log w_{t-1} + update_weight
        = log w_{t-1} + [log πₜ(z₀:t) - log πₜ₋₁(z₀:t₋₁)]
```

This telescopes correctly: the cumulative weight is the importance
weight for πₜ under the sequential proposal. ✓

**Case 2: ProposalEdit (custom kernel).** By the edit-duality theorem
(`edit-duality.md` §3), the ProposalEdit weight is:

```
edit_weight = update_weight + bwd_score - fwd_score
```

This is the correct incremental IS weight when using a custom forward
kernel as the proposal. The forward kernel proposes changes to the
trace, the model update accounts for the density change, and the
backward score accounts for the proposal asymmetry.

This connection to ProposalEdit is the key insight of SMCP3 — it
enables arbitrary generative functions as proposal kernels, with
automatic weight computation via the GFI edit interface. ✓ ∎

---

## 4. ESS and Resampling

### 4.1 Effective Sample Size

The ESS (ref: `smcp3.cljs:88-92`) measures particle diversity:

```
ESS = (Σ_k w_k)² / (Σ_k w_k²)
```

In log-space:

```
log_ess = 2 · logsumexp(log_w) - logsumexp(2 · log_w)
ESS = exp(log_ess)
```

ESS ranges from 1 (one particle dominates) to N (all particles equal).

### 4.2 Systematic Resampling

When ESS < threshold · N (default threshold = 0.5), systematic
resampling is applied:

```
systematic-resample(log_w, N):
  w̃_k = softmax(log_w)                  -- normalized weights
  u₀   ~ Uniform(0, 1/N)
  cumsum = cumulative-sum(w̃)
  for k = 1, …, N:
    u_k = u₀ + (k-1)/N
    index_k = min{j : cumsum_j ≥ u_k}
  return indices
```

### 4.3 Proposition (Resampling Preserves Expectation)

**Statement.** For any function f, systematic resampling preserves
the weighted expectation:

```
E[f̂_resample] = Σ_k w̃_k f(z_k) = f̂_weighted
```

**Proof.** After resampling, particle j appears n_j times where
E[n_j] = N · w̃_j. The resampled estimate is:

```
f̂_resample = (1/N) Σ_k f(z_{index_k})
```

Taking expectations over the resampling randomness:

```
E[f̂_resample] = (1/N) Σ_k E[f(z_{index_k})]
              = (1/N) Σ_j N · w̃_j · f(z_j)
              = Σ_j w̃_j f(z_j)
              = f̂_weighted
```

✓ ∎

### 4.4 Weight Reset

After resampling, all particle weights are reset to uniform:

```
log w_k ← 0  for all k
```

This is correct because resampling has already incorporated the weight
information into the particle selection. Continuing with non-uniform
weights would double-count.

---

## 5. Rejuvenation Preserves Target

### 5.1 Definition

After resampling and extending, an optional rejuvenation step applies
an MCMC kernel to each particle (ref: `smcp3.cljs:115-118`):

```
For k = 1, …, N:
  trace_k ← rejuvenation-fn(trace_k, key_k)
```

### 5.2 Proposition (Rejuvenation Correctness)

**Statement.** If the rejuvenation function is a πₜ-stationary Markov
kernel, then rejuvenation does not change the target distribution
of the particles.

**Proof.** After the SMCP3-Step, each particle (approximately)
targets πₜ (with the approximation quality depending on the number
of particles and resampling). A πₜ-stationary kernel K satisfies
πₜ · K = πₜ, so applying K to each particle preserves πₜ as the
target.

By the kernel composition results from `kernel-composition.md`:
- MH kernels from regenerate are πₜ-stationary (§2)
- Chain and cycle compositions preserve stationarity (§§3-4)
- Mix compositions preserve stationarity (§5)

Therefore, any rejuvenation function constructed from these kernel
combinators is πₜ-stationary and safe to use. ✓ ∎

### 5.3 Purpose

Rejuvenation does not affect the theoretical correctness of the SMC
approximation (it preserves the target). Its purpose is to improve
the practical quality of the approximation by:
1. Diversifying particles after resampling (which creates duplicates)
2. Moving particles toward higher-probability regions of πₜ
3. Reducing the variance of downstream estimates

---

## 6. Log-Marginal-Likelihood Estimation

### 6.1 Accumulation Formula

The log-marginal-likelihood estimate (ref: `smcp3.cljs:120-123`) is
accumulated across all time steps:

```
log p̂(x₀:T) = Σ_{t=0}^{T} log((1/N) Σ_{k=1}^{N} exp(w_t^{(k)}))
             = Σ_{t=0}^{T} [logsumexp(w_t^{(1)}, …, w_t^{(N)}) - log N]
```

where w_t^{(k)} are the incremental weights **before** resampling
at step t.

### 6.2 Theorem (Unbiasedness of Log-ML Estimate)

**Statement.** (Del Moral 2004, Doucet & Johansen 2009) The SMC
marginal likelihood estimate:

```
p̂(x₀:T) = Π_{t=0}^{T} [(1/N) Σ_k exp(w_t^{(k)})]
```

is an unbiased estimator of p(x₀:T).

**Proof sketch.** At each step t, the normalized importance weights
approximate the Radon-Nikodym derivative πₜ/qₜ. The average
importance weight at step t:

```
Ẑ_t = (1/N) Σ_k exp(w_t^{(k)})
```

is an unbiased estimator of the normalizing constant ratio
Zₜ/Zₜ₋₁ = p(xₜ | x₀:t₋₁). The product:

```
p̂(x₀:T) = Π_t Ẑ_t
```

is unbiased for p(x₀:T) = Π_t p(xₜ | x₀:t₋₁) by the independence
of the estimators across resampling steps (conditional on the
resampled particles, the new weights are independent).

The log estimate log p̂(x₀:T) is a consistent (but biased) estimator
of log p(x₀:T), with bias O(1/N). ∎

**Implementation**: The log-ML is accumulated via `mx/add` across
time steps at `smcp3.cljs:120-123`.

---

## 7. Proposal Design Conditions

### 7.1 ProposalEdit Requirements

When using custom generative functions as proposals (the "PP" in SMCP3),
the following conditions must hold for correctness:

**(S1) Support coverage.** The proposal must cover all new addresses
introduced at time t:

```
dom(propose(kernel, args)) ⊇ new addresses in πₜ \ πₜ₋₁
```

If the proposal does not cover all new addresses, the missing addresses
are sampled from the prior, which may be a poor proposal.

**(S2) Backward scorability.** The backward assessment must be able
to score the discard:

```
assess(backward-kernel, args, discard) produces finite weight
```

This is the same condition P2 from `edit-duality.md` §3.

### 7.2 ConstraintEdit Fallback

When no custom proposal is provided, SMCP3-Step falls back to
ConstraintEdit (standard update):

```
edit(model, trace, ConstraintEdit(observations_t))
```

This is equivalent to:
1. Constraining new observation addresses to their observed values
2. Sampling any new latent addresses from the prior
3. Keeping all existing addresses unchanged

The weight is the update weight (new_score - old_score), which
correctly accounts for the density change from πₜ₋₁ to πₜ.

**When ConstraintEdit suffices:** For models where the prior is a
reasonable proposal for new latent variables, ConstraintEdit is
sufficient. This is common when new latent variables are weakly
coupled to observations.

**When ProposalEdit is needed:** For models with strong observation-
latent coupling (e.g., state-space models with nonlinear observations),
the prior is a poor proposal and custom kernels can dramatically
improve particle efficiency.

### 7.3 Proposal Quality and ESS

The quality of the proposal directly affects the ESS:
- **Perfect proposal:** All weights equal → ESS = N
- **Prior proposal:** Weights vary widely → ESS ≪ N
- **Learned proposal:** Intermediate ESS, improves with training

SMCP3 enables training the proposal via the ADEV framework (`adev.md`):
the proposal parameters are optimized to maximize the log-marginal-
likelihood estimate, which implicitly minimizes the variance of the
importance weights.

---

## 8. Connection to Standard SMC

### 8.1 Special Cases

SMCP3 generalizes standard SMC algorithms:

| SMCP3 Configuration | Standard Algorithm |
|---------------------|-------------------|
| No proposal, no rejuvenation | Bootstrap particle filter |
| No proposal, MH rejuvenation | SMC with MH moves |
| Custom proposal, no rejuvenation | SMC with optimal proposal |
| Custom proposal, MH rejuvenation | Full SMCP3 |

### 8.2 Advantage of GFI Integration

The key advantage of SMCP3 in GenMLX is that proposals are specified
as generative functions, which:
1. Can be composed using combinators (Map, Unfold, etc.)
2. Automatically produce correct importance weights via the edit interface
3. Can be trained via ADEV gradient estimation
4. Satisfy the GFI contracts, enabling formal verification

---

## 9. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| SMCP3-Init | `(smcp3-init ...)` | `smcp3.cljs:22-59` |
| SMCP3-Step | `(smcp3-step ...)` | `smcp3.cljs:65-124` |
| Main SMCP3 loop | `(smcp3 ...)` | `smcp3.cljs:130-182` |
| Proposal-based init | `(p/propose proposal ...)` | `smcp3.cljs:36-48` |
| Standard init (no proposal) | `(p/generate model ...)` | `smcp3.cljs:49-52` |
| ProposalEdit extension | `(p/edit model trace edit-req)` | `smcp3.cljs:99-102` |
| ConstraintEdit fallback | `(p/update model trace obs)` | `smcp3.cljs:106-108` |
| ESS computation | `exp(2·logsumexp(lw) - logsumexp(2·lw))` | `smcp3.cljs:88-92` |
| Systematic resampling | `(u/systematic-resample ...)` | `smcp3.cljs:85-91` |
| Rejuvenation | `(rejuvenation-fn trace key)` | `smcp3.cljs:115-117` |
| Log-ML accumulation | `(mx/add log-ml increment)` | `smcp3.cljs:120-123` |
| Weight materialization | `(u/materialize-weights weights)` | `smcp3.cljs:56` |
| Observation sequence | `(vec observations)` | `smcp3.cljs:152` |

---

## References

- Lew, A. K., Mathieu, G., Zhi-Xuan, T., Ghavamizadeh, M., &
  Mansinghka, V. K. (2023). "SMCP3: Sequential Monte Carlo with
  Probabilistic Program Proposals." *AISTATS 2023*.
- Del Moral, P. (2004). *Feynman-Kac Formulae: Genealogical and
  Interacting Particle Systems with Applications.* Springer.
- Doucet, A., & Johansen, A. M. (2009). "A Tutorial on Particle
  Filtering and Smoothing: Fifteen Years Later." In *Handbook of
  Nonlinear Filtering*, Oxford University Press.
- Chopin, N., & Papaspiliopoulos, O. (2020). *An Introduction to
  Sequential Monte Carlo.* Springer.
