# Adaptive HMC and NUTS — TODO 10.17

> Theorem: Hamiltonian Monte Carlo preserves the target distribution via
> symplectic integration and Metropolis correction. Dual averaging adapts
> the step size, Welford statistics estimate the mass matrix, and NUTS
> automatically selects trajectory length — all while maintaining correctness.

---

## 1. Hamiltonian Mechanics on Trace Space

### 1.1 Setup

Let q ∈ ℝ^D denote the continuous trace values (flattened to a vector)
and p ∈ ℝ^D the auxiliary momentum variables. The Hamiltonian is:

```
H(q, p) = U(q) + K(p)
```

where:
- **U(q) = -log π(q)** is the potential energy (negative log-target)
- **K(p) = ½ p^T M^{-1} p** is the kinetic energy
- **M** is the mass matrix (positive definite)

The target distribution π(q) relates to the model score:
U(q) = -score(q) where score is the total log-density from the GFI.

**Implementation**: `hamiltonian` at `mcmc.cljs:829-836`.

### 1.2 Hamilton's Equations

```
dq/dt = ∂H/∂p = M^{-1} p
dp/dt = -∂H/∂q = -∇U(q) = ∇ log π(q)
```

These equations define a continuous-time flow that preserves H (energy
conservation), phase-space volume (Liouville's theorem), and the
canonical distribution π(q,p) ∝ exp(-H(q,p)).

---

## 2. Leapfrog Integration

### 2.1 Störmer-Verlet (Leapfrog) Integrator

The leapfrog integrator discretizes Hamilton's equations with step
size ε:

```
p_{1/2} = p₀ - (ε/2) ∇U(q₀)           -- half-step momentum
q₁      = q₀ + ε M^{-1} p_{1/2}        -- full-step position
p₁      = p_{1/2} - (ε/2) ∇U(q₁)       -- half-step momentum
```

**Implementation**: `leapfrog-step` at `mcmc.cljs:842-858`.

### 2.2 Proposition (Symplecticity)

**Statement.** The leapfrog integrator is a symplectic map: it
preserves the symplectic 2-form ω = Σᵢ dqᵢ ∧ dpᵢ.

**Proof.** The leapfrog step decomposes as three shear maps:

```
Φ₁: (q, p) ↦ (q, p - (ε/2) ∇U(q))       -- momentum kick
Φ₂: (q, p) ↦ (q + ε M^{-1} p, p)         -- position drift
Φ₃: (q, p) ↦ (q, p - (ε/2) ∇U(q))       -- momentum kick
```

Each shear map has Jacobian with determinant 1:

For Φ₁: J₁ = [[I, 0], [-(ε/2)∇²U, I]], det(J₁) = 1
For Φ₂: J₂ = [[I, ε M^{-1}], [0, I]], det(J₂) = 1
For Φ₃: J₃ = [[I, 0], [-(ε/2)∇²U, I]], det(J₃) = 1

The composition Φ = Φ₃ ∘ Φ₂ ∘ Φ₁ has det(J_Φ) = 1, confirming
volume preservation. Since each shear is a symplectomorphism (it
preserves ω), the composition is symplectic. ∎

### 2.3 Corollary (Volume Preservation)

Volume preservation implies that the leapfrog map has unit Jacobian
determinant. This eliminates the need for a Jacobian correction in the
MH acceptance ratio — the proposal density ratio is entirely determined
by the Hamiltonian difference.

### 2.4 Fused Leapfrog

The fused leapfrog trajectory (ref: `mcmc.cljs:870-892`) merges
adjacent half-momentum kicks into full kicks for L steps:

```
fused-leapfrog(q₀, p₀, ε, L):
  p_{1/2} = p₀ - (ε/2) ∇U(q₀)          -- initial half-kick
  for i = 1 to L-1:
    qᵢ      = q_{i-1} + ε M^{-1} p_{i-1/2}
    p_{i+1/2} = p_{i-1/2} - ε ∇U(qᵢ)    -- full kick (merged halves)
  q_L     = q_{L-1} + ε M^{-1} p_{L-1/2}
  p_L     = p_{L-1/2} - (ε/2) ∇U(q_L)   -- final half-kick
  return (q_L, p_L)
```

This requires L+1 gradient evaluations instead of 2L for the unfused
version, a 2× reduction. The mathematical result is identical — the
fusion simply reorganizes the computation.

**Implementation**: `leapfrog-trajectory-fused` at `mcmc.cljs:870-892`.

---

## 3. HMC Detailed Balance

### 3.1 HMC Algorithm

One HMC step (ref: `mcmc.cljs:894-918`):

```
hmc-step(q, key):
  1. Sample momentum: p ~ N(0, M)
  2. Run leapfrog: (q*, p*) = leapfrog(q, p, ε, L)
  3. Negate momentum: p* ← -p*
  4. Accept/reject: α = min(1, exp(H(q,p) - H(q*,p*)))
     q' = q* if accepted, q otherwise
  return q'
```

### 3.2 Theorem (HMC Preserves Target)

**Statement.** The HMC kernel satisfies detailed balance with respect
to π(q) ∝ exp(-U(q)).

**Proof.** The HMC proposal consists of:

1. **Momentum augmentation:** Extend the target to
   π(q,p) ∝ exp(-H(q,p)) = exp(-U(q)) · exp(-K(p))

2. **Leapfrog + momentum flip:** Define the proposal map
   T(q,p) = negate-momentum(leapfrog(q,p,ε,L))

   The momentum negation makes T an involution: T(T(q,p)) = (q,p).
   This is because leapfrog is time-reversible under momentum flip.

3. **Volume preservation:** By Proposition 2.2, leapfrog preserves
   volume. Momentum negation also preserves volume (det = (-1)^D · (-1)^D = 1).
   So det(∂T/∂(q,p)) = 1.

4. **MH correction:** The acceptance probability:
   ```
   α = min(1, exp(H(q,p) - H(q*,p*)))
   ```

   accounts for the integration error (H is not exactly preserved).

Since T is an involution with unit Jacobian, and we apply MH correction:

```
π(q,p) · α(q,p → q*,p*) = π(q*,p*) · α(q*,p* → q,p)
```

Both sides equal min(π(q,p), π(q*,p*)), confirming detailed balance
on the extended space (q,p). Marginalizing over p gives detailed
balance on q. ∎

### 3.3 Mass Matrix Generalizations

The implementation supports three mass matrix types:

| Type | M | K(p) | M^{-1}p | Ref |
|------|---|------|---------|-----|
| Identity (nil) | I | ½p^Tp | p | `mcmc.cljs:790-792` |
| Diagonal | diag(m) | ½Σ pᵢ²/mᵢ | p/m | `mcmc.cljs:793-797` |
| Dense | M | ½p^T M^{-1} p | solve(M,p) | `mcmc.cljs:798-801` |

All three preserve the correctness of HMC — the mass matrix only
affects efficiency (mixing rate), not the stationary distribution.

---

## 4. Dual Averaging for Step-Size Adaptation

### 4.1 Find Reasonable Epsilon

The initial step-size is found by doubling or halving until the
acceptance probability is approximately 50% (ref: `mcmc.cljs:1020-1041`):

```
find-reasonable-epsilon(q, score, grad):
  ε = 1.0
  (q*, p*) = leapfrog-step(q, p~N(0,I), ε, grad)
  ΔH = H(q,p) - H(q*,p*)
  direction = sign(ΔH - log(0.5))     -- +1 if too high, -1 if too low
  while sign(ΔH - log(0.5)) = direction:
    ε ← ε · 2^direction               -- double or halve
    (q*, p*) = leapfrog-step(q, p, ε, grad)
    ΔH = H(q,p) - H(q*,p*)
  return ε
```

This heuristic (Algorithm 4, Hoffman & Gelman 2014) finds an ε where
the HMC acceptance rate is near 50%, a reasonable starting point for
dual averaging.

### 4.2 Dual Averaging Algorithm

The step-size is adapted during warmup using dual averaging
(Algorithm 5, Hoffman & Gelman 2014). Ref: `mcmc.cljs:1069-1111`.

```
Parameters: δ = 0.65 (target acceptance), γ = 0.05, t₀ = 10, κ = 0.75
Initialize: log_ε̄₀ = 0, H̄₀ = 0, μ = log(10 · ε₀)

For warmup iteration t = 1, 2, …, T:
  1. Run HMC step with current ε_t, observe acceptance rate α_t
  2. H̄_t = (1 - 1/(t + t₀)) H̄_{t-1} + (1/(t + t₀))(δ - α_t)
  3. log ε_t = μ - (√t / γ) H̄_t
  4. log ε̄_t = t^{-κ} log ε_t + (1 - t^{-κ}) log ε̄_{t-1}

After warmup: ε = exp(log ε̄_T)
```

### 4.3 Convergence Property

**Proposition.** The dual averaging algorithm converges to a step-size
ε* such that the average acceptance rate equals δ.

**Proof sketch.** Dual averaging (Nesterov 2009) solves the stochastic
optimization problem:

```
minimize f(log ε) = E[δ - α(ε)]²
```

The update H̄_t tracks a running estimate of E[δ - α(ε)], and the
Robbins-Monro conditions (step sizes 1/(t + t₀) → 0, Σ 1/(t+t₀) = ∞)
ensure convergence to a root of E[δ - α(ε)] = 0. The averaged iterate
log ε̄_t converges at the optimal O(1/√t) rate due to Polyak-Ruppert
averaging with exponent κ < 1.

---

## 5. Welford Online Variance Estimation

### 5.1 Algorithm

The Welford algorithm (ref: `mcmc.cljs:1047-1062`) computes running
mean and variance in a single pass with numerical stability:

```
welford-update(state, x):
  n'    = state.n + 1
  delta = x - state.mean
  mean' = state.mean + delta / n'
  delta2 = x - mean'
  M2'   = state.M2 + delta · delta2
  return { n: n', mean: mean', M2: M2' }

welford-variance(state):
  if state.n < 10 then nil
  else state.M2 / (state.n - 1)      -- Bessel's correction
```

### 5.2 Proposition (Correctness)

**Statement.** After n updates with values x₁, …, xₙ, the Welford
state satisfies:
- `mean = (1/n) Σᵢ xᵢ`
- `M2 = Σᵢ (xᵢ - mean)²`

**Proof.** By induction on n.

**Base (n=1):** mean = x₁, M2 = 0. Trivially correct.

**Inductive step:** Assume correct for n-1. After update with xₙ:

```
mean_n = mean_{n-1} + (xₙ - mean_{n-1}) / n
       = ((n-1) mean_{n-1} + xₙ) / n
       = (1/n) Σᵢ xᵢ                           ✓
```

For M2, using the identity:
```
M2_n = M2_{n-1} + (xₙ - mean_{n-1})(xₙ - mean_n)
```

This is Welford's recurrence, which is algebraically equivalent to
Σᵢ (xᵢ - mean_n)² (proof: expand and use the relation between
consecutive means). ∎

### 5.3 Numerical Stability

The Welford algorithm avoids catastrophic cancellation that occurs in
the naive formula Var = E[x²] - E[x]². The update uses only
differences (xᵢ - mean), which are small relative to the values
themselves, maintaining precision in floating-point arithmetic.

### 5.4 Application to Mass Matrix Adaptation

During HMC warmup (ref: `mcmc.cljs:1106-1109`), Welford statistics
estimate the marginal variances of each dimension. The diagonal mass
matrix is set to:

```
M = diag(welford-variance(state))
```

This adapts the kinetic energy to match the posterior geometry:
dimensions with high variance get high mass (large momentum), and
dimensions with low variance get low mass. This improves mixing by
ensuring all dimensions traverse at similar rates.

The mass matrix is updated only during warmup and frozen during
sampling, preserving the stationarity of the HMC kernel during the
collection phase.

---

## 6. NUTS: No-U-Turn Sampler

### 6.1 U-Turn Criterion

The NUTS algorithm (Hoffman & Gelman 2014) automatically selects the
trajectory length by detecting U-turns. The criterion
(ref: `mcmc.cljs:1337-1345`) is:

```
u-turn?(q⁻, q⁺, p⁻, p⁺) =
  (q⁺ - q⁻) · p⁻ < 0  OR  (q⁺ - q⁻) · p⁺ < 0
```

A U-turn occurs when the trajectory starts curving back, indicated by
the momentum becoming anti-aligned with the displacement.

### 6.2 Binary Tree Construction

NUTS builds a balanced binary tree of leapfrog states by recursive
doubling (ref: `mcmc.cljs:1360-1393`):

```
build-tree(q, p, direction, depth, ε):
  if depth = 0 then
    -- Base case: single leapfrog step
    (q', p') = leapfrog-step(q, p, direction · ε)
    n'       = I[H(q₀,p₀) - H(q',p') > log(slice)]
    return (q⁻=q', q⁺=q', p⁻=p', p⁺=p', q'=q', n'=n', ok=¬u-turn)

  else
    -- Build first half-tree
    (q⁻, q⁺, p⁻, p⁺, q', n', ok) = build-tree(q, p, dir, depth-1, ε)
    if ok then
      -- Build second half-tree
      if direction = +1 then
        (q⁻₂, q⁺, p⁻₂, p⁺, q'', n'', ok₂) = build-tree(q⁺, p⁺, dir, depth-1, ε)
      else
        (q⁻, q⁺₂, p⁻, p⁺₂, q'', n'', ok₂) = build-tree(q⁻, p⁻, dir, depth-1, ε)

      -- Multinomial sampling: accept new candidate with probability n''/(n'+n'')
      if uniform() < n'' / (n' + n'') then q' ← q''
      n' ← n' + n''
      ok ← ok₂ AND ¬u-turn?(q⁻, q⁺, p⁻, p⁺)
    return (q⁻, q⁺, p⁻, p⁺, q', n', ok)
```

### 6.3 Theorem (NUTS Detailed Balance)

**Statement.** The NUTS sampler preserves the target distribution π.

**Proof.** The NUTS proposal is a generalization of HMC where the
trajectory length L is chosen adaptively. The key properties:

1. **Symmetric doubling:** At each depth, the tree is extended in a
   random direction (forward or backward with equal probability). This
   makes the proposal distribution symmetric over trajectory lengths.

2. **Multinomial sampling within tree:** The candidate q' is selected
   from the tree nodes with probability proportional to exp(-H(q',p')).
   This is equivalent to sampling from the distribution:
   ```
   P(q' | tree) ∝ exp(-H(q', p')) · I[q' ∈ tree]
   ```

3. **U-turn termination:** The tree stops growing when a U-turn is
   detected. Since the doubling direction is symmetric and the
   termination criterion is applied to the full tree boundaries
   (not individual branches), the stopping rule does not break the
   symmetry needed for detailed balance.

4. **Volume preservation:** Each leapfrog step in the tree preserves
   volume (Proposition 2.2). The multinomial selection and U-turn
   criterion operate on the Hamiltonian values, not on the phase-space
   volume.

The formal argument follows Hoffman & Gelman (2014), Theorem 1: the
multinomial criterion with symmetric doubling ensures that the detailed
balance condition is satisfied on the joint space of (q, tree structure).
∎

### 6.4 Practical Tree Depth

The maximum tree depth is bounded (default: 10 in the implementation,
ref: `mcmc.cljs:1395-1542`), giving at most 2^10 = 1024 leapfrog steps.
The U-turn criterion typically stops the tree much earlier, giving
adaptive trajectory lengths between 1 and ~100 steps.

---

## 7. Vectorized HMC

### 7.1 N Parallel Chains

Vectorized HMC (ref: `mcmc.cljs:1287-1331`) runs N independent HMC
chains simultaneously using [N,D]-shaped arrays:

```
vectorized-hmc-step(q, key):        -- q : [N, D]
  p        = sample-momentum(M, N)  -- p : [N, D]
  (q*, p*) = fused-leapfrog(q, p, ε, L)  -- all [N, D]
  ΔH       = H(q,p) - H(q*,p*)     -- [N]-shaped
  accept?  = uniform([N]) < exp(min(0, ΔH))  -- [N]-shaped boolean
  q'       = where(accept?, q*, q)  -- per-chain accept/reject
  return q'
```

### 7.2 Proposition (Vectorized Correctness)

**Statement.** The vectorized HMC produces N independent chains, each
with the correct stationary distribution π.

**Proof.** By the broadcasting correctness framework from
`broadcasting.md`, vectorized operations on [N,D]-shaped arrays are
equivalent to N independent scalar operations when:

1. **Independent randomness:** Each chain uses independent momentum
   samples and acceptance randomness (via key splitting).
2. **No cross-chain interaction:** The leapfrog integrator operates
   element-wise across the N dimension (position and momentum updates
   are per-chain).
3. **Per-chain accept/reject:** `mx/where` selects independently for
   each chain based on its own ΔH.

Since each chain is an independent HMC chain, and HMC preserves π
(Theorem 3.2), each chain has stationary distribution π. ∎

### 7.3 Fused Vectorized Leapfrog

The vectorized fused leapfrog (ref: `mcmc.cljs:1228-1247`) operates
on [N,D]-shaped arrays with the same fusion optimization as the scalar
version (§2.4):

```
vectorized-fused-leapfrog(q, p, ε, L):   -- q, p : [N, D]
  grad     = ∇U(q)                        -- [N, D]
  p        = p - (ε/2) · grad             -- half-kick, [N, D]
  for i = 1 to L-1:
    q    = q + ε · inv-mass(p)             -- [N, D]
    grad = ∇U(q)                           -- [N, D]
    p    = p - ε · grad                    -- full-kick, [N, D]
  q = q + ε · inv-mass(p)                  -- final position
  grad = ∇U(q)
  p = p - (ε/2) · grad                    -- final half-kick
  return (q, p)
```

All operations broadcast naturally over the N dimension. The gradient
∇U(q) for [N,D]-shaped q is computed via `mx/grad` applied to the
vectorized score function, returning [N,D]-shaped gradients.

---

## 8. Compiled HMC Chains

### 8.1 Compilation Strategy

The compiled HMC chain (ref: `mcmc.cljs:920-968`) compiles K consecutive
HMC steps into a single Metal kernel:

```
compiled-chain = mx/compile-fn(fn [q, score, grad, keys] →
  fold(hmc-step, [q, score, grad], keys))
```

This eliminates K-1 round-trips between CPU and GPU, achieving near-
linear speedup for K steps per compiled block.

### 8.2 Proposition (Compilation Preserves Stationarity)

Compilation does not affect the mathematical operations — it only
changes execution scheduling. Each compiled step performs the same
leapfrog integration, Hamiltonian evaluation, and accept/reject logic.
Therefore, the compiled chain preserves π by the same argument as
Theorem 3.2.

---

## 9. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| H(q,p) = U(q) + K(p) | `(hamiltonian score kinetic)` | `mcmc.cljs:829-836` |
| U(q) = -score(q) | Negated model score | — |
| K(p) = ½p^T M^{-1} p | `(kinetic-energy p metric)` | `mcmc.cljs:803-816` |
| M^{-1} p | `(inv-mass-multiply p metric)` | `mcmc.cljs:818-827` |
| Sample p ~ N(0,M) | `(sample-momentum metric D key)` | `mcmc.cljs:788-801` |
| Leapfrog step | `(leapfrog-step ...)` | `mcmc.cljs:842-858` |
| Fused leapfrog | `(leapfrog-trajectory-fused ...)` | `mcmc.cljs:870-892` |
| HMC step | `(hmc-step ...)` | `mcmc.cljs:894-918` |
| Compiled K-step chain | `(make-compiled-hmc-chain ...)` | `mcmc.cljs:920-968` |
| find-reasonable-epsilon | `(find-reasonable-epsilon ...)` | `mcmc.cljs:1020-1041` |
| Welford update | `(welford-update state x)` | `mcmc.cljs:1047-1055` |
| Welford variance | `(welford-variance state)` | `mcmc.cljs:1057-1062` |
| Dual averaging warmup | `(dual-averaging-warmup ...)` | `mcmc.cljs:1069-1111` |
| U-turn criterion | `(compute-u-turn? ...)` | `mcmc.cljs:1337-1345` |
| NUTS build-tree | `(build-tree ...)` | `mcmc.cljs:1360-1393` |
| NUTS main loop | `(nuts ...)` | `mcmc.cljs:1395-1542` |
| Vectorized HMC step | `(vectorized-hmc-step ...)` | `mcmc.cljs:1249-1285` |
| Vectorized fused leapfrog | `(vectorized-leapfrog-fused ...)` | `mcmc.cljs:1228-1247` |
| Vectorized HMC | `(vectorized-hmc ...)` | `mcmc.cljs:1287-1331` |

---

## References

- Neal, R. M. (2011). "MCMC using Hamiltonian dynamics." Chapter 5 in
  *Handbook of Markov Chain Monte Carlo*, CRC Press.
- Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler:
  Adaptively Setting Path Lengths in Hamiltonian Monte Carlo."
  *JMLR*, 15, 1593-1623.
- Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987).
  "Hybrid Monte Carlo." *Physics Letters B*, 195(2), 216-222.
- Nesterov, Y. (2009). "Primal-dual subgradient methods for convex
  problems." *Mathematical Programming*, 120, 221-259.
- Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian
  Monte Carlo." arXiv:1701.02434.
