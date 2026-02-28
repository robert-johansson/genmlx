# Variational Inference — TODO 10.18

> Theorem: the ELBO is a lower bound on the log-marginal likelihood,
> IWELBO tightens monotonically in K, and programmable VI composes
> objectives with estimators while preserving gradient correctness.

---

## 1. Evidence Lower Bound (ELBO)

### 1.1 Theorem (ELBO Bound)

**Statement.** For a model p(x, z) with observations x and latent
variables z, and any variational distribution q(z):

```
log p(x) ≥ E_q[log p(x, z) - log q(z)] = ELBO(q)
```

with equality iff q(z) = p(z|x) almost everywhere.

**Proof.** By Jensen's inequality applied to the concave log function:

```
log p(x) = log ∫ p(x, z) dz
         = log ∫ [p(x, z) / q(z)] q(z) dz
         = log E_q[p(x, z) / q(z)]
         ≥ E_q[log(p(x, z) / q(z))]         -- Jensen's inequality
         = E_q[log p(x, z) - log q(z)]
         = ELBO(q)
```

The gap between log p(x) and ELBO(q) equals the KL divergence:

```
log p(x) - ELBO(q) = E_q[log q(z) - log p(z|x)]
                    = KL(q ‖ p(z|x))
                    ≥ 0
```

with equality iff KL = 0, i.e., q = p(z|x). ∎

### 1.2 Monte Carlo ELBO

The ELBO is estimated via Monte Carlo (ref: `vi.cljs:17-40`):

```
ELBO ≈ (1/K) Σ_{k=1}^{K} [log p(x, z_k) - log q(z_k)]

where z_k ~ q(z)
```

Each term log p(x, z_k) is the model score (from GFI `assess` or
`generate`), and log q(z_k) is the guide score.

**Implementation**: `elbo-estimate` at `vi.cljs:17-40`.

---

## 2. ADVI: Mean-Field Gaussian Guide

### 2.1 Guide Structure

ADVI (ref: `vi.cljs:46-110`) uses a mean-field Gaussian guide:

```
q(z; μ, σ) = Π_d N(z_d; μ_d, σ_d²)
```

Parameters: μ ∈ ℝ^D (means), σ ∈ ℝ^D_{>0} (standard deviations).
The log-standard-deviations log(σ) are optimized to ensure positivity.

### 2.2 Reparameterized Sampling

Samples are drawn via reparameterization:

```
ε ~ N(0, I)
z = μ + σ ⊙ ε
```

where ⊙ is element-wise multiplication. This enables gradient flow
through the samples to the parameters (μ, σ).

**Implementation**: `vi.cljs:26-27` computes `samples = mu + sigma * eps`.

### 2.3 Gradient of Negative ELBO

```
∇_{μ,σ} (-ELBO) = ∇_{μ,σ} E_q[-log p(x,z) + log q(z)]
                 = E_ε[∇_{μ,σ} (-log p(x, μ+σε) + log N(μ+σε; μ, σ²))]
```

By the reparameterization trick (see `adev.md` §2), the gradient
commutes with the expectation. The implementation uses `mx/grad` on the
negative ELBO function, which computes this automatically via MLX
autograd.

---

## 3. IWELBO: Importance-Weighted ELBO

### 3.1 Definition

The K-sample IWELBO (ref: `vi.cljs:242-256`) is:

```
IWELBO_K = E[log((1/K) Σ_{k=1}^{K} exp(log w_k))]
         = E[logsumexp(log w₁, …, log w_K) - log K]
```

where log w_k = log p(x, z_k) - log q(z_k) and z_k ~iid q.

### 3.2 Theorem (IWELBO Tightness)

**Statement.** (Burda et al. 2015)

(i) IWELBO_K ≥ ELBO for all K ≥ 1, with IWELBO_1 = ELBO.

(ii) IWELBO_K is monotonically non-decreasing in K.

(iii) IWELBO_K → log p(x) as K → ∞.

**Proof.**

**(i)** For K = 1: IWELBO_1 = E[log w₁] = ELBO. For K > 1, by
Jensen's inequality applied to the concave log:

```
IWELBO_K = E[log((1/K) Σ_k w_k)]
         ≥ (1/K) Σ_k E[log w_k]        -- Jensen on concave log
         = (1/K) · K · ELBO
         = ELBO
```

**(ii)** For K+1 vs K samples, IWELBO_{K+1} ≥ IWELBO_K follows from
a coupling argument: given K+1 samples, any K-subset average is
dominated by the (K+1)-sample average. Formally, for i.i.d. w_k:

```
E[log((1/(K+1)) Σ_{k=1}^{K+1} w_k)] ≥ E[log((1/K) Σ_{k=1}^{K} w_k)]
```

This follows from the fact that adding a positive term inside the
logarithm of the average increases the expectation.

**(iii)** By the law of large numbers, (1/K) Σ_k w_k → E_q[p(x,z)/q(z)]
= p(x) almost surely. Therefore:

```
IWELBO_K = E[log((1/K) Σ_k w_k)] → log p(x)
```

by the continuous mapping theorem. ∎

**Implementation**: `iwelbo-objective` at `vi.cljs:242-256`.

---

## 4. VIMCO: Leave-One-Out Baselines

### 4.1 The Gradient Challenge

The IWELBO gradient has two components:
1. **Reparameterization gradient** through log w_k (when z_k is
   reparameterizable) — low variance
2. **REINFORCE gradient** through the discrete selection of which
   samples to use — high variance

VIMCO (Mnih & Rezende 2016) addresses the high-variance component with
leave-one-out (LOO) baselines.

### 4.2 Construction

The VIMCO objective (ref: `vi.cljs:305-338`) is:

```
L̂ = logsumexp(log w₁, …, log w_K) - log K    -- IWELBO estimate

For each k, define the LOO geometric mean:
  ḡ_k = (1/(K-1)) Σ_{j≠k} log w_j

Define the LOO IWELBO:
  L̂_{-k} = logsumexp(log w₁, …, ḡ_k, …, log w_K) - log K
```

where ḡ_k replaces log w_k with the geometric mean of the other weights.

The VIMCO signal for each sample k is:

```
signal_k = (L̂ - L̂_{-k}) · stop_gradient(log q(z_k))
```

### 4.3 Proposition (VIMCO Unbiasedness)

**Statement.** The VIMCO gradient estimator is an unbiased estimator
of ∇_θ IWELBO_K.

**Proof sketch.** The LOO baseline L̂_{-k} is independent of z_k
(it depends only on z_j for j ≠ k). Therefore, by the same argument
as the baseline proposition in `adev.md` §5:

```
E_{z_k}[L̂_{-k} · ∇_θ log q(z_k)] = L̂_{-k} · E_{z_k}[∇_θ log q(z_k)] = 0
```

So subtracting L̂_{-k} does not bias the gradient. The full gradient:

```
∇_θ IWELBO ≈ Σ_k (L̂ - L̂_{-k}) · ∇_θ log q(z_k)   -- REINFORCE part
            + ∇_θ L̂                                  -- reparam part
```

is unbiased. ∎

### 4.4 Variance Reduction

The LOO baseline L̂_{-k} approximates the optimal baseline
E[IWELBO | z_{-k}]. When the weights are well-behaved, L̂_{-k} ≈ L̂,
so the signal L̂ - L̂_{-k} is small, drastically reducing variance
compared to the raw REINFORCE signal L̂ · ∇ log q.

**Implementation**: `vimco-objective` at `vi.cljs:305-338`. The LOO
computation uses a [K,K] matrix with geometric means on the diagonal
(`vi.cljs:329-331`).

---

## 5. Wake-Sleep Objectives

### 5.1 P-Wake (Forward KL Minimization)

The P-Wake objective (ref: `vi.cljs:258-266`) maximizes:

```
L_{P-wake} = E_q[log p(z)]
```

This is a component of the wake-sleep algorithm that trains the model
p while holding q fixed. It maximizes the expected model log-probability
under the variational distribution.

### 5.2 Q-Wake (Reverse KL Minimization)

The Q-Wake objective (ref: `vi.cljs:268-283`) minimizes KL(p(z|x) ‖ q(z))
via self-normalized importance weights:

```
L_{Q-wake} = Σ_k w̃_k · log q(z_k)

where w̃_k = softmax(log w₁, …, log w_K)   -- self-normalized weights
      log w_k = log p(x, z_k) - log q(z_k)
```

### 5.3 Proposition (Q-Wake Approximation)

**Statement.** In the limit K → ∞, the Q-Wake objective converges to:

```
L_{Q-wake} → E_{p(z|x)}[log q(z)] = -H(p(z|x)) - KL(p(z|x) ‖ q(z))
```

**Proof.** As K → ∞, the self-normalized importance weights w̃_k
converge to the true posterior weights:

```
w̃_k = exp(log w_k) / Σ_j exp(log w_j) → p(z_k|x) / q(z_k) · q(z_k) / Σ_j [p(z_j|x)]
     → p(z_k|x) (via law of large numbers on the denominator)
```

Therefore:
```
Σ_k w̃_k log q(z_k) → E_{p(z|x)}[log q(z)]
```

Maximizing this over q parameters minimizes KL(p(z|x) ‖ q(z)), since
H(p(z|x)) is constant w.r.t. q. ∎

**Implementation**: `qwake-objective` at `vi.cljs:268-283`.

---

## 6. REINFORCE Estimator Integration

### 6.1 Non-Reparameterizable Guides

When the guide q contains non-reparameterizable distributions (e.g.,
discrete latent variables), the ELBO gradient requires REINFORCE:

```
∇_θ ELBO = E_q[∇_θ log q(z; θ) · (f(z) - b)]    -- REINFORCE
         + E_q[∇_θ f(z)]                           -- through reparam sites
```

where f(z) = log p(x,z) - log q(z) and b is a baseline.

### 6.2 Implementation

The REINFORCE estimator (ref: `vi.cljs:285-303`) constructs a surrogate
whose gradient equals the REINFORCE gradient:

```
surrogate = stop_gradient(f(z) - baseline) · log q(z)
```

The baseline is set to the sample mean: `b = mean(f(z))`, which is
independent of the current sample (computed across the K samples) and
therefore does not bias the gradient (see `adev.md` §5).

**Implementation**: `reinforce-estimator` at `vi.cljs:285-303`.

---

## 7. Programmable VI Composition

### 7.1 Architecture

Programmable VI (ref: `vi.cljs:340-399`) decouples three components:

```
Programmable VI = Objective × Estimator × Optimizer
```

**Objectives** (what to optimize):
- `:elbo` — Standard ELBO (ref: `vi.cljs:229-240`)
- `:iwelbo` — Importance-weighted ELBO (ref: `vi.cljs:242-256`)
- `:vimco` — VIMCO with LOO baselines (ref: `vi.cljs:305-338`)
- `:pwake` — P-Wake (ref: `vi.cljs:258-266`)
- `:qwake` — Q-Wake with self-normalized weights (ref: `vi.cljs:268-283`)
- Custom function — any `(fn [log-weights] -> scalar)`

**Estimators** (how to compute gradients):
- `:reparam` — Reparameterization (default, low variance)
- `:reinforce` — REINFORCE with baseline (for discrete latents)

**Optimizer:** Adam with configurable learning rate.

### 7.2 Theorem (Composition Correctness)

**Statement.** For any valid objective O and estimator E, the
programmable VI gradient ∇_θ [E(O(q_θ))] is an unbiased estimator of
∇_θ O(q_θ).

**Proof.** Each objective O(q_θ) is a scalar function of the parameters
θ (through the samples and log-probabilities). The estimator E computes
a gradient estimate:

- **Reparam estimator:** ∇_θ O is computed via automatic differentiation
  through the reparameterization. By the reparameterization trick
  (`adev.md` §2), E[∇_θ O] = ∇_θ E[O] for reparameterizable q.

- **REINFORCE estimator:** The surrogate loss
  `stop_grad(O - baseline) · log q` has gradient equal to the
  REINFORCE estimator. By the log-derivative trick (`adev.md` §3),
  E[∇_θ surrogate] = ∇_θ E[O] for any q.

The objectives (ELBO, IWELBO, VIMCO, etc.) are all functions of the
importance weights log w_k, and each has been shown to provide unbiased
gradients (§§1-4 above). Composing a valid objective with a valid
estimator preserves unbiasedness. ∎

### 7.3 Objective Summary

| Objective | Bound | Variance | Applicability |
|-----------|-------|----------|---------------|
| ELBO | Lower (loosest) | Low (reparam) | Continuous latents |
| IWELBO_K | Lower (tighter) | Medium | Continuous latents |
| VIMCO | Lower (tighter) | Low (LOO baseline) | Discrete + continuous |
| P-Wake | Not a bound | Low | Model training |
| Q-Wake | Not a bound | Medium | Guide training |

---

## 8. Compiled VI

### 8.1 Compilation

Compiled VI (ref: `vi.cljs:126-193`, `vi.cljs:401-448`) uses
`mx/compile-fn` to compile the gradient computation:

```
compiled-grad-fn = mx/compile-fn(fn [params, key] →
  mx/grad(neg-objective-fn)(params))
```

### 8.2 Proposition (Compilation Preserves Semantics)

By the same argument as `adev.md` §7: MLX compilation is a graph-level
optimization that preserves all mathematical operations, including
sampling, log-probability computation, and automatic differentiation.
The compiled gradient is identical to the eager gradient for all inputs
with the same shapes.

**Implementation**: `compiled-vi` at `vi.cljs:126-193`,
`compiled-programmable-vi` at `vi.cljs:401-448`.

---

## 9. Convergence Properties

### 9.1 ELBO Convergence

Under standard assumptions (convex q-parameterization in natural
parameters, bounded variance), Adam optimization of the ELBO converges
to a local optimum of KL(q ‖ p(z|x)).

For the mean-field Gaussian guide:
- The ELBO is generally non-convex in (μ, log σ)
- Multiple local optima may exist (mode-seeking behavior of forward KL)
- Adam converges to a stationary point (Kingma & Ba 2015)

### 9.2 IWELBO Convergence

IWELBO optimization with K > 1 converges to a different stationary
point than ELBO (Rainforth et al. 2018):
- IWELBO_K minimizes a tighter bound, leading to q closer to p(z|x)
- In the limit K → ∞, the optimum is q = p(z|x) (assuming sufficient
  q-family expressiveness)
- However, the signal-to-noise ratio of the gradient decreases as K
  increases (Tucker et al. 2019), requiring careful balancing

---

## 10. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| ELBO = E_q[log p - log q] | `(elbo-estimate ...)` | `vi.cljs:17-40` |
| ELBO objective | `(elbo-objective ...)` | `vi.cljs:229-240` |
| ADVI (mean-field Gaussian) | `(vi model ...)` | `vi.cljs:46-110` |
| Reparameterization z = μ+σε | `(mx/add mu (mx/multiply sigma eps))` | `vi.cljs:26-27` |
| IWELBO_K | `(iwelbo-objective K)` | `vi.cljs:242-256` |
| logsumexp(log w) - log K | `(mx/subtract (mx/logsumexp lw) (mx/log (mx/scalar K)))` | `vi.cljs:255` |
| VIMCO LOO baseline | LOO matrix construction | `vi.cljs:305-338` |
| LOO geometric mean | `(mx/divide (mx/subtract sum-lw lw-k) (mx/scalar (dec K)))` | `vi.cljs:325-326` |
| P-Wake objective | `(pwake-objective ...)` | `vi.cljs:258-266` |
| Q-Wake self-normalized | `softmax(log_w) · log q` | `vi.cljs:268-283` |
| REINFORCE surrogate | `stop_grad(f-b) · log_q` | `vi.cljs:285-303` |
| Programmable VI | `(programmable-vi ...)` | `vi.cljs:340-399` |
| Compiled VI | `(compiled-vi ...)` | `vi.cljs:126-193` |
| Compiled programmable VI | `(compiled-programmable-vi ...)` | `vi.cljs:401-448` |
| VIMCO convenience | `(vimco ...)` | `vi.cljs:450-459` |

---

## References

- Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999).
  "An introduction to variational methods for graphical models."
  *Machine Learning*, 37(2), 183-233.
- Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). "Importance
  Weighted Autoencoders." *ICLR 2016*.
- Mnih, A., & Rezende, D. J. (2016). "Variational Inference for Monte
  Carlo Objectives." *ICML 2016*.
- Kingma, D. P., & Welling, M. (2014). "Auto-encoding variational
  Bayes." *ICLR 2014*.
- Rainforth, T., et al. (2018). "Tighter Variational Bounds are Not
  Necessarily Better." *ICML 2018*.
- Tucker, G., et al. (2019). "Doubly Reparameterized Gradient
  Estimators for Monte Carlo Objectives." *UAI 2019*.
