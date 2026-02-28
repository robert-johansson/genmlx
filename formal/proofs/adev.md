# ADEV Gradient Estimation — TODO 10.15

> Theorem: the ADEV surrogate loss has the correct gradient, combining
> reparameterization for continuous distributions and REINFORCE for discrete
> distributions. Baselines reduce variance without introducing bias.

---

## 1. ADEV Handler State Type

### 1.1 State Schema

The ADEV handler extends the simulate state with a REINFORCE
log-probability accumulator:

```
σ_adev = { key          : Key,
            choices      : γ,
            score        : ℝ,
            reinforce-lp : ℝ }        -- accumulated log-prob of non-reparam sites
```

The `reinforce-lp` field tracks the sum of log-probabilities at addresses
where reparameterization is not available, enabling the REINFORCE
gradient estimator for those sites.

### 1.2 Reparameterizable vs Non-Reparameterizable

A distribution D η is **reparameterizable** if there exists a
deterministic function g and a fixed noise distribution ε such that:

```
x ~ D(θ)  ⟺  ε ~ p(ε), x = g(θ, ε)
```

where g is differentiable with respect to θ.

The ADEV handler branches on this property at each trace site
(ref: `adev.cljs:28-45`):

```
⟦trace(a, d)⟧_adev(σ) =
  if has-reparam?(d) then
    -- Reparameterizable: sample via g(θ, ε)
    let (k₁, k₂) = split(σ.key)
        v         = reparam-sample(d, k₂)        -- v = g(θ, ε)
    in (v, { key          : k₁,
              choices      : σ.choices[a ↦ v],
              score        : σ.score + log_prob(d, v),
              reinforce-lp : σ.reinforce-lp })    -- unchanged
  else
    -- Non-reparameterizable: sample + stop gradient + accumulate lp
    let (k₁, k₂) = split(σ.key)
        v         = stop_gradient(sample(d, k₂))
        lp        = log_prob(d, v)
    in (v, { key          : k₁,
              choices      : σ.choices[a ↦ v],
              score        : σ.score + lp,
              reinforce-lp : σ.reinforce-lp + lp })
```

**Key difference:** For reparameterizable sites, gradients flow through
the sampled value v (since v = g(θ, ε) is differentiable in θ). For
non-reparameterizable sites, `stop_gradient` prevents gradient flow
through v, and the log-probability is accumulated in `reinforce-lp`
for the REINFORCE estimator.

**Implementation**: `adev-transition` at `adev.cljs:28-45`.

---

## 2. Reparameterization Trick

### 2.1 Theorem (Reparameterization Gradient)

**Statement.** Let x ~ p(x; θ) be a reparameterizable distribution
with x = g(θ, ε), ε ~ p(ε). Let f(x) be a differentiable function.
Then:

```
∇_θ E_{p(x;θ)}[f(x)] = E_{p(ε)}[∇_θ f(g(θ, ε))]
```

**Proof.** By the change of variables:

```
E_{p(x;θ)}[f(x)] = ∫ f(x) p(x; θ) dx
                   = ∫ f(g(θ, ε)) p(ε) dε        -- substitution x = g(θ,ε)
```

Since the integration measure p(ε)dε does not depend on θ, we can
exchange differentiation and integration (under standard regularity
conditions — dominated convergence):

```
∇_θ ∫ f(g(θ, ε)) p(ε) dε = ∫ ∇_θ f(g(θ, ε)) p(ε) dε
                            = E_{p(ε)}[∇_θ f(g(θ, ε))]
```

∎

### 2.2 Examples in GenMLX

| Distribution | Reparameterization | g(θ, ε) |
|-------------|-------------------|---------|
| gaussian(μ, σ) | ε ~ N(0,1) | μ + σε |
| uniform(a, b) | ε ~ U(0,1) | a + (b-a)ε |
| gamma(α, β) | ε ~ Gamma(α, 1) | ε/β |
| beta(α, β) | Via gamma ratio | g₁/(g₁+g₂) |
| bernoulli(p) | **Not reparameterizable** | — |
| categorical(ps) | **Not reparameterizable** | — |
| poisson(λ) | **Not reparameterizable** | — |

### 2.3 Variance Properties

The reparameterization gradient estimator has **lower variance** than
REINFORCE for continuous distributions because gradients flow through
the function f, leveraging its local structure. For smooth f, the
variance scales as O(1/n) with sample size, compared to O(Var[f]/n)
for REINFORCE.

---

## 3. REINFORCE Estimator

### 3.1 Theorem (Log-Derivative Trick)

**Statement.** For any distribution p(x; θ) and function f(x):

```
∇_θ E_{p(x;θ)}[f(x)] = E_{p(x;θ)}[f(x) · ∇_θ log p(x; θ)]
```

**Proof.** Starting from the definition:

```
∇_θ E[f(x)] = ∇_θ ∫ f(x) p(x; θ) dx
             = ∫ f(x) ∇_θ p(x; θ) dx
```

Using the log-derivative identity ∇_θ p = p · ∇_θ log p:

```
             = ∫ f(x) p(x; θ) · ∇_θ log p(x; θ) dx
             = E_{p(x;θ)}[f(x) · ∇_θ log p(x; θ)]
```

∎

### 3.2 Applicability

REINFORCE applies to **any** distribution where log p(x; θ) is
differentiable in θ, including discrete distributions. This is why
the ADEV handler accumulates `reinforce-lp = Σ log p(x_a; θ)` at
non-reparameterizable sites — it is the ∇_θ log p term needed for
REINFORCE.

---

## 4. Surrogate Loss Correctness

### 4.1 ADEV Surrogate Construction

The ADEV surrogate loss (ref: `adev.cljs:81-93`) is:

```
surrogate(θ) = cost(θ, ε) + stop_gradient(cost(θ, ε) - baseline) · reinforce-lp(θ)
```

where:
- `cost(θ, ε)` is the model's cost function evaluated under ADEV execution
- `reinforce-lp(θ)` is the accumulated log-probability at non-reparam sites
- `baseline` is an optional variance reduction term (§5)
- `stop_gradient` prevents gradient flow through the multiplicative factor

### 4.2 Theorem (Surrogate Gradient Correctness)

**Statement.** The gradient of the surrogate loss equals the gradient
of the expected cost:

```
E[∇_θ surrogate(θ)] = ∇_θ E[cost(θ)]
```

**Proof.** Partition the trace addresses into two sets:
- R = {reparameterizable addresses}
- N = {non-reparameterizable addresses}

The expected cost decomposes as:

```
E[cost] = E_{p(x_N;θ)} E_{p(ε)}[cost(θ, x_N, g_R(θ, ε))]
```

where x_N are the non-reparameterizable samples and g_R are the
reparameterized samples.

Differentiating:

```
∇_θ E[cost] = ∇_θ ∫∫ cost(θ, x_N, g_R(θ, ε)) p(x_N; θ) p(ε) dx_N dε
```

This splits into two terms by the product rule:

**Term 1 (through cost, reparameterized):**
```
∫∫ ∇_θ cost(θ, x_N, g_R(θ, ε)) · p(x_N; θ) p(ε) dx_N dε
= E[∇_θ cost(θ, x_N, g_R(θ, ε))]
```

This is the gradient from the first term of the surrogate: the direct
`cost(θ, ε)` term, where gradients flow through reparameterized values
g_R(θ, ε).

**Term 2 (through p(x_N; θ), REINFORCE):**
```
∫∫ cost(θ, x_N, g_R(θ, ε)) · ∇_θ p(x_N; θ) · p(ε) dx_N dε
= ∫∫ cost · p(x_N; θ) · ∇_θ log p(x_N; θ) · p(ε) dx_N dε
= E[cost · ∇_θ log p(x_N; θ)]
= E[cost · ∇_θ reinforce-lp]
```

This matches the gradient of the second surrogate term:
```
∇_θ [stop_gradient(cost - baseline) · reinforce-lp]
= stop_gradient(cost - baseline) · ∇_θ reinforce-lp
```

Since `stop_gradient(cost - baseline)` is treated as a constant during
differentiation, and E[baseline · ∇_θ reinforce-lp] = 0 (§5), this
gives exactly E[cost · ∇_θ reinforce-lp].

Combining both terms:
```
E[∇_θ surrogate] = E[∇_θ cost] + E[cost · ∇_θ reinforce-lp]
                  = ∇_θ E[cost]
```

∎

### 4.3 Implementation Note

The `stop_gradient` on cost ensures that ∇_θ does not differentiate
through cost again in the REINFORCE term (which would double-count the
reparameterization gradient). The `stop_gradient` on the sampled values
at non-reparam sites (`adev.cljs:38`) ensures that the first surrogate
term does not try to differentiate through discrete samples.

---

## 5. Baseline Variance Reduction

### 5.1 Proposition (Baselines Are Unbiased)

**Statement.** For any constant b (with respect to x):

```
E_{p(x;θ)}[b · ∇_θ log p(x; θ)] = 0
```

**Proof.**

```
E[b · ∇_θ log p(x; θ)] = b · ∫ ∇_θ log p(x; θ) · p(x; θ) dx
                        = b · ∫ ∇_θ p(x; θ) dx
                        = b · ∇_θ ∫ p(x; θ) dx
                        = b · ∇_θ 1
                        = 0
```

∎

### 5.2 Corollary

Subtracting a baseline b from the cost in the REINFORCE term does not
change the expected gradient:

```
E[(cost - b) · ∇_θ log p] = E[cost · ∇_θ log p] - b · E[∇_θ log p]
                           = E[cost · ∇_θ log p] - 0
                           = E[cost · ∇_θ log p]
```

but reduces variance when b ≈ E[cost].

### 5.3 EMA Baseline

The implementation (ref: `adev.cljs:162-165`) uses an exponential
moving average (EMA) baseline:

```
baseline_{t+1} = decay · baseline_t + (1 - decay) · cost_t
```

with default `decay = 0.99`. This tracks E[cost] adaptively. The EMA
baseline is computed from previous iterations and is therefore constant
with respect to the current sample — satisfying the requirement that
b is independent of x for the unbiasedness proof.

**Variance reduction magnitude:** The optimal baseline is b* = E[cost],
which minimizes Var[(cost - b) · ∇_θ log p]. The EMA baseline
approximates b* with exponential smoothing, providing substantial
variance reduction in practice (often 10-100× for high-variance
REINFORCE terms).

---

## 6. Batching Equivalence

### 6.1 Vectorized ADEV Transition

The batched ADEV transition (ref: `adev.cljs:177-194`) operates on
[N]-shaped arrays:

```
⟦trace(a, d)⟧_adev^N(σ) =
  if has-reparam?(d) then
    let (k₁, k₂) = split(σ.key)
        v         = reparam-sample-n(d, k₂, N)     -- [N]-shaped
    in (v, { …, reinforce-lp : σ.reinforce-lp })   -- unchanged
  else
    let (k₁, k₂) = split(σ.key)
        v         = stop_gradient(sample-n(d, k₂, N))  -- [N]-shaped
        lp        = log-prob(d, v)                       -- [N]-shaped
    in (v, { …, reinforce-lp : σ.reinforce-lp + lp })  -- [N]+[N]=[N]
```

### 6.2 Theorem (Batched ADEV Equivalence)

**Statement.** The vectorized ADEV surrogate equals the mean of N
independent scalar ADEV surrogates:

```
vadev-surrogate(θ) = (1/N) Σᵢ adev-surrogate_i(θ)
```

**Proof.** The argument follows the broadcasting correctness framework
from `broadcasting.md`. At each trace site:

- Reparameterized: v[i] = g(θ, ε_i) for independent ε_i. The cost
  contributions are independent across particles.
- Non-reparameterized: v[i] are independent samples with independent
  log-probs lp[i]. The reinforce-lp[i] accumulates per-particle.

The surrogate for particle i is:
```
s_i = cost_i + stop_grad(cost_i - baseline) · reinforce-lp_i
```

The vectorized surrogate computes:
```
vadev-surrogate = mean(cost) + mean(stop_grad(cost - baseline) · reinforce-lp)
```

By linearity and independence:
```
= (1/N) Σᵢ cost_i + (1/N) Σᵢ stop_grad(cost_i - baseline) · reinforce-lp_i
= (1/N) Σᵢ s_i
```

The gradient commutes with the mean:
```
∇_θ vadev-surrogate = (1/N) Σᵢ ∇_θ s_i
```

which is a Monte Carlo estimate of ∇_θ E[cost] with N samples,
reducing variance by a factor of N. ∎

**Implementation**: `vadev-transition` at `adev.cljs:177-194`,
`vadev-surrogate` at `adev.cljs:222-233`.

---

## 7. Compiled ADEV

### 7.1 Compilation Semantics

The compiled ADEV optimizer (ref: `adev.cljs:256-301`) uses `mx/compile-fn`
to compile the gradient computation into a single fused kernel:

```
compiled-grad-fn = mx/compile-fn(fn [params, key] →
  mx/value-and-grad(fn [p] → vadev-surrogate(p, key))(params))
```

### 7.2 Proposition (Compilation Preserves Semantics)

**Statement.** `mx/compile-fn(f)` computes the same function as f for
all inputs.

**Proof sketch.** MLX compilation (`mx/compile-fn`) traces the
computation graph on the first call and caches the compiled kernel.
On subsequent calls with the same shapes, it replays the compiled graph.
The compilation is a graph-level optimization (operator fusion, memory
planning) that does not change the mathematical semantics of any
operation.

The key properties preserved:
1. **Arithmetic identity:** All MLX ops (add, multiply, etc.) produce
   identical results compiled vs. eager.
2. **Random number generation:** The PRNG state threading is preserved
   — same key produces same samples.
3. **Gradient computation:** `mx/value-and-grad` computes the same
   automatic differentiation graph compiled vs. eager.
4. **Stop-gradient:** Compilation preserves `mx/stop-gradient` barriers
   in the backward pass.

**Implementation**: `compiled-adev-optimize` at `adev.cljs:256-301`.
The `mx/tidy` call (`adev.cljs:289`) ensures intermediate arrays are
freed, which is a resource management concern that does not affect
mathematical correctness.

---

## 8. Optimization Loop

### 8.1 Adam Update

The ADEV optimizer (ref: `adev.cljs:129-171`) uses Adam with the
surrogate gradient:

```
for t = 1, 2, …:
  g_t        = ∇_θ surrogate(θ_{t-1})
  m_t        = β₁ m_{t-1} + (1-β₁) g_t
  v_t        = β₂ v_{t-1} + (1-β₂) g_t²
  m̂_t        = m_t / (1 - β₁^t)
  v̂_t        = v_t / (1 - β₂^t)
  θ_t        = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)
  baseline_t = decay · baseline_{t-1} + (1-decay) · cost_t
```

### 8.2 Convergence Remark

Under standard assumptions (bounded gradients, convex or
Polyak-Łojasiewicz objective), Adam converges to a stationary point.
The ADEV surrogate gradient is an unbiased estimate of the true gradient
(Theorem 4.2), so standard stochastic optimization convergence results
apply (Kingma & Ba 2015).

The baseline adaptation (§5.3) improves convergence speed by reducing
gradient variance, but does not affect the stationary point.

---

## 9. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| σ_adev state type | `{:key :choices :score :reinforce-lp}` map | `adev.cljs:28-45` |
| has-reparam?(d) | `(has-reparam? d)` | `adev.cljs:19-22` |
| Reparam branch | `(dc/dist-reparam d k)` | `adev.cljs:37` |
| Non-reparam branch | `(mx/stop-gradient (dc/dist-sample d k))` | `adev.cljs:38` |
| reinforce-lp accumulation | `(mx/add (:reinforce-lp σ) lp)` | `adev.cljs:45` |
| adev-surrogate | `(adev-surrogate ...)` | `adev.cljs:81-93` |
| stop_gradient(cost - baseline) | `(mx/stop-gradient (mx/subtract cost baseline))` | `adev.cljs:90-92` |
| Batched ADEV transition | `vadev-transition` | `adev.cljs:177-194` |
| Batched surrogate | `vadev-surrogate` | `adev.cljs:222-233` |
| EMA baseline | `(mx/add (mx/multiply decay baseline) ...)` | `adev.cljs:162-165` |
| Compiled gradient | `(mx/compile-fn grad-fn)` | `adev.cljs:256-301` |
| Adam optimizer | SGD/Adam in optimization loop | `adev.cljs:129-171` |

---

## References

- Williams, R. J. (1992). "Simple statistical gradient-following
  algorithms for connectionist reinforcement learning." *Machine
  Learning*, 8(3), 229-256.
- Kingma, D. P., & Welling, M. (2014). "Auto-encoding variational
  Bayes." *ICLR 2014*.
- Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic
  Optimization." *ICLR 2015*.
- Lew, A. K., et al. (2023). "ADEV: Sound Automatic Differentiation
  of Expected Values of Probabilistic Programs." *POPL 2023*.
- Schulman, J., et al. (2015). "Gradient Estimation Using Stochastic
  Computation Graphs." *NeurIPS 2015*.
