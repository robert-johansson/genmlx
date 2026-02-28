# Kernel Composition — TODO 10.14

> Theorem: composed Markov kernels preserve the target distribution.
> MH kernels derived from regenerate are stationary; chain, cycle, and mix
> combinators preserve stationarity and (under mild conditions) ergodicity.

---

## 1. Definitions

### 1.1 Markov Kernel

A **Markov kernel** on trace space Γ is a measurable function

```
K : Γ × Key → Γ
```

that, for each trace u and PRNG key k, produces a new trace u' = K(u, k).
Marginalizing over the key randomness gives a transition probability:

```
K(u, ·) : Γ → 𝒫(Γ)
```

In GenMLX, kernels have the ClojureScript signature `(fn [trace key] -> trace)`.

### 1.2 Stationarity

A kernel K is **stationary** (or **invariant**) with respect to a target
distribution π if:

```
π · K = π

i.e., for all measurable A ⊆ Γ:
∫ π(du) K(u, A) = π(A)
```

When K is π-stationary, iterating K from any initial trace produces a
Markov chain whose stationary distribution is π.

### 1.3 Detailed Balance

A kernel K satisfies **detailed balance** with respect to π if:

```
π(du) K(u, du') = π(du') K(u', du)
```

Detailed balance implies stationarity (integrate both sides over du).
It is a sufficient but not necessary condition.

### 1.4 Ergodicity

A π-stationary kernel K is **ergodic** if, for π-almost all initial
traces u₀, the empirical distribution of {K^n(u₀)} converges to π:

```
1/N Σ_{n=1}^{N} δ_{K^n(u₀)} → π  as N → ∞
```

Sufficient conditions include π-irreducibility and aperiodicity.

---

## 2. MH Kernel from Regenerate

### 2.1 Construction

The MH kernel from regenerate (ref: `kernel.cljs:16-26`) takes a selection
S and constructs a kernel by:

1. Propose u' via `regenerate(model, trace, S)`, obtaining weight w
2. Accept with probability α = min(1, exp(w))
3. Return u' if accepted, u otherwise

```
mh-kernel(S)(u, key) =
  let (k₁, k₂) = split(key)
      (u', w)   = regenerate(model, u, S)     -- uses k₁
      α         = min(1, exp(w))
      accept?   = uniform(0,1; k₂) < α
  in if accept? then u' else u
```

### 2.2 Theorem (MH Kernel Preserves Target)

**Statement.** Let π be the target distribution of a generative function
G with denotation (μ, f), so π(du) = μ(du) with density w(u) = dμ/dν(u).
Then the MH kernel constructed above satisfies detailed balance with
respect to π.

**Proof.** From `semantics.md` §4.4, the regenerate weight is:

```
w = log p(u') - log p(u) - proposal_ratio
```

where proposal_ratio = Σ_{a ∈ S} [log p(u'(a)) - log p(u(a))].

As derived in `semantics.md` §4.4, this equals the log MH ratio:

```
w = log[p(u')/p(u)] + log[q(u|u')/q(u'|u)]
```

where q(u'|u) = Π_{a ∈ S} density_{d_a}(u'(a)) is the prior proposal
distribution at the selected addresses.

The MH acceptance probability α = min(1, exp(w)) ensures detailed balance
by the standard Metropolis-Hastings theorem (Hastings 1970):

```
π(du) q(u, du') min(1, exp(w)) = π(du') q(u', du) min(1, exp(-w))
```

Both sides equal min(π(du) q(u, du'), π(du') q(u', du)), confirming
detailed balance. Since detailed balance implies stationarity, the
MH kernel preserves π. ∎

---

## 3. Chain Preserves Stationarity

### 3.1 Definition

The `chain` combinator (ref: `kernel.cljs:41-51`) composes N kernels
sequentially:

```
chain([K₁, K₂, …, Kₙ])(u, key) =
  let [k₁, k₂, …, kₙ] = split_n(key, N)
  in Kₙ(…K₂(K₁(u, k₁), k₂)…, kₙ)
```

### 3.2 Theorem (Chain Preserves Stationarity)

**Statement.** If K₁, K₂, …, Kₙ are each π-stationary, then
K = Kₙ ∘ … ∘ K₁ is π-stationary.

**Proof.** By induction on N.

**Base case (N=1):** K₁ is π-stationary by hypothesis.

**Inductive step:** Assume K' = K_{n-1} ∘ … ∘ K₁ is π-stationary
(induction hypothesis). Then:

```
π · K = π · (K' ; Kₙ)
      = (π · K') · Kₙ        -- composition of kernels
      = π · Kₙ                -- by IH: π · K' = π
      = π                     -- by hypothesis on Kₙ
```

Therefore K is π-stationary. ∎

### 3.3 Remark on Ergodicity

A chain of π-stationary kernels need not be ergodic even if each
component is. However, if the combined chain is π-irreducible (can reach
any region of positive π-measure) and aperiodic, then it is ergodic.
In practice, chaining MH kernels targeting different address selections
improves irreducibility by exploring different dimensions.

---

## 4. Cycle Preserves Stationarity and Ergodicity

### 4.1 Definition

The `cycle-kernels` combinator (ref: `kernel.cljs:70-81`) repeatedly
applies kernels from a vector in round-robin order for n total
applications:

```
cycle([K₁, …, Kₘ], n)(u, key) =
  let [k₁, …, kₙ] = split_n(key, n)
  in fold(fn [u_i, i] → K_{(i mod m)+1}(u_i, k_{i+1}),
          u, range(n))
```

### 4.2 Theorem (Cycle Preserves Stationarity)

**Statement.** If K₁, …, Kₘ are each π-stationary, then
cycle([K₁,…,Kₘ], n) is π-stationary for any n ≥ 1.

**Proof.** Each application in the cycle applies some K_{j} which is
π-stationary. By Theorem 3.2, the composition of π-stationary kernels
is π-stationary. The cycle of length n composes n such kernels
(repeating cyclically), so the result is π-stationary. ∎

### 4.3 Gibbs as Cycle

**Corollary (Gibbs Sampling).** Let S₁, …, Sₘ be a partition of the
trace addresses. Define K_j = mh-kernel(S_j). Then:

```
gibbs([S₁, …, Sₘ]) = cycle([K₁, …, Kₘ], m)
```

is π-stationary. If each S_j is a single address and the model has
no zero-probability regions, the Gibbs sampler is ergodic.

This corresponds to `kernel.cljs:216-223` where `gibbs` is implemented
as `chain` of `prior` kernels over individual addresses.

---

## 5. Mix Preserves Stationarity

### 5.1 Definition

The `mix-kernels` combinator (ref: `kernel.cljs:83-94`) randomly selects
one kernel per step from a weighted collection:

```
mix([(w₁,K₁), …, (wₘ,Kₘ)])(u, key) =
  let (k₁, k₂) = split(key)
      j         = categorical([w₁,…,wₘ], k₁)
  in K_j(u, k₂)
```

where Σᵢ wᵢ = 1 and wᵢ ≥ 0.

### 5.2 Theorem (Mixture Preserves Stationarity)

**Statement.** If K₁, …, Kₘ are each π-stationary, then
K_mix = Σᵢ wᵢ Kᵢ is π-stationary for any convex weights.

**Proof.**

```
π · K_mix = π · (Σᵢ wᵢ Kᵢ)
          = Σᵢ wᵢ (π · Kᵢ)        -- linearity of kernel composition
          = Σᵢ wᵢ π                 -- each Kᵢ is π-stationary
          = π · (Σᵢ wᵢ)
          = π                        -- weights sum to 1
```

∎

### 5.3 Remark on Ergodicity

A mixture kernel is ergodic if at least one component kernel Kᵢ with
wᵢ > 0 is ergodic. Intuitively, the chain has positive probability
of applying the ergodic kernel at each step, which is sufficient for
the overall chain to explore the full state space.

---

## 6. Repeat Kernel

### 6.1 Definition

The `repeat-kernel` combinator (ref: `kernel.cljs:53-61`) applies a
single kernel n times:

```
repeat(K, n)(u, key) =
  let [k₁, …, kₙ] = split_n(key, n)
  in K(…K(K(u, k₁), k₂)…, kₙ)
```

### 6.2 Corollary

**Corollary.** If K is π-stationary, then repeat(K, n) is π-stationary
for any n ≥ 1.

**Proof.** repeat(K, n) = chain([K, K, …, K]), which is π-stationary
by Theorem 3.2. ∎

---

## 7. Random Walk MH

### 7.1 Construction

The random walk kernel (ref: `kernel.cljs:150-172`) proposes by
adding Gaussian noise to the current value:

```
random-walk(addr, std)(u, key) =
  let (k₁, k₂, k₃) = split_3(key)
      x             = u[addr]
      noise         = sample(gaussian(0, std), k₁)
      x'            = x + noise
      u'            = update(model, u, {addr: x'})
      w_update      = u'.weight
      α             = min(1, exp(w_update))
  in if uniform(0,1; k₃) < α then u'.trace else u
```

### 7.2 Proposition (Random Walk Symmetry)

**Statement.** The Gaussian random walk proposal is symmetric:
q(x'|x) = q(x|x').

**Proof.** The proposal distribution is q(x'|x) = N(x'; x, σ²).
By the symmetry of the Gaussian density:

```
q(x'|x) = (2πσ²)^{-1/2} exp(-(x'-x)²/(2σ²))
q(x|x') = (2πσ²)^{-1/2} exp(-(x-x')²/(2σ²))
```

Since (x'-x)² = (x-x')², we have q(x'|x) = q(x|x'). ∎

### 7.3 Corollary (Simplified MH Ratio)

**Corollary.** For a symmetric random walk proposal, the MH acceptance
ratio simplifies to:

```
α = min(1, p(u')/p(u)) = min(1, exp(score(u') - score(u)))
```

The proposal ratio q(u|u')/q(u'|u) = 1 cancels from the MH ratio.
This is why the implementation uses `update` (which returns
weight = new_score - old_score) directly as the log acceptance ratio,
without a separate proposal correction.

**Implementation note:** For multi-address random walks
(`kernel.cljs:155-156, 169-172`), the implementation chains single-address walks.
By Theorem 3.2, this preserves stationarity.

---

## 8. Custom Proposal MH

### 8.1 Construction

The custom proposal kernel (ref: `kernel.cljs:181-214`) uses a pair of
generative functions (forward, backward) as the proposal mechanism:

```
proposal(forward, backward, symmetric?)(u, key) =
  let (k₁, k₂, k₃) = split_3(key)
      (fwd-choices, fwd-score) = propose(forward, {u.choices}, k₁)
      (u', w_update, disc)     = update(model, u, fwd-choices)

      log_α = if symmetric? then
                w_update
              else
                let (_, bwd-score) = assess(backward, {u'.choices}, disc)
                in w_update + bwd-score - fwd-score

  in if uniform(0,1; k₃) < exp(min(0, log_α)) then u' else u
```

### 8.2 Theorem (Custom Proposal MH Ratio)

**Statement.** The weight `w_update + bwd-score - fwd-score` is the
correct log MH acceptance ratio for maintaining detailed balance.

**Proof.** This is a direct consequence of the ProposalEdit detailed
balance theorem from `edit-duality.md` §3. The terms are:

```
w_update  = score(u') - score(u)           -- from correctness.md §3
fwd-score = log q_f(fwd-choices | u)       -- forward proposal density
bwd-score = log q_b(disc | u')             -- backward proposal density
```

So:

```
log_α = log[p(u')/p(u)] + log[q_b(disc|u')/q_f(fwd-choices|u)]
      = log[p(u') · q_b(disc|u') / (p(u) · q_f(fwd-choices|u))]
```

This is the standard Hastings ratio (Hastings 1970). By the same
argument as in `edit-duality.md` §3, this ensures detailed balance. ∎

### 8.3 Symmetric Proposal Shortcut

When `symmetric?` is true, the forward and backward proposals are the
same distribution, and fwd-score = bwd-score. The MH ratio simplifies
to `w_update = score(u') - score(u)`, matching the random walk case
(§7.3). The implementation (`kernel.cljs:205-214`) correctly skips the
backward assessment in this case.

---

## 9. Seed Kernel

### 9.1 Definition

The `seed` combinator (ref: `kernel.cljs:63-68`) wraps a kernel to use
a fixed PRNG key:

```
seed(K, fixed-key)(u, _) = K(u, fixed-key)
```

### 9.2 Remark

A seeded kernel is deterministic (given u, it always produces the same
u'). It is still π-stationary if K is, since stationarity is a property
of the transition probability kernel, and a fixed-key kernel defines a
valid (deterministic) transition. However, it is **not ergodic** — the
chain is periodic (it cycles). Seeded kernels are primarily useful for
debugging and reproducibility, not for production inference.

---

## 10. Collect Samples and Resource Safety

### 10.1 Sample Collection

The `collect-samples` function (ref: `kernel.cljs:100-127`) orchestrates
burn-in and thinning:

```
collect-samples(K, trace₀, key, n, burn-in, thin) =
  let trace_burned = iterate(K, trace₀, burn-in)
      samples      = [K^{burn-in + i·thin}(trace₀) for i in 1..n]
  in samples
```

### 10.2 Stationarity After Burn-in

If K is π-stationary, then the distribution of K^t(u₀) approaches π
as t → ∞ (assuming ergodicity). The burn-in period discards the initial
transient, and thinning reduces autocorrelation between samples.

The collected samples are (approximately) i.i.d. from π when:
1. Burn-in is long enough for the chain to mix
2. Thinning interval exceeds the autocorrelation time
3. The kernel is ergodic

### 10.3 Resource Guard

The implementation uses `u/with-resource-guard` (`kernel.cljs:109`) to
periodically call `mx/eval!`, preventing unbounded computation graph
growth. This is semantically transparent — `mx/eval!` materializes lazy
computations without changing their values.

---

## 11. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| Markov kernel K : Γ × Key → Γ | `(fn [trace key] -> trace)` | `kernel.cljs:16` |
| mh-kernel(S) | `(mh-kernel model selection)` | `kernel.cljs:16-26` |
| chain([K₁,…,Kₙ]) | `(chain model [k1 k2 ...])` | `kernel.cljs:41-51` |
| repeat(K, n) | `(repeat-kernel model kernel n)` | `kernel.cljs:53-61` |
| seed(K, key) | `(seed kernel key)` | `kernel.cljs:63-68` |
| cycle([K₁,…,Kₘ], n) | `(cycle-kernels model [k1...] n)` | `kernel.cljs:70-81` |
| mix([(w₁,K₁),…]) | `(mix-kernels model [[w1 k1]...])` | `kernel.cljs:83-94` |
| collect-samples | `(collect-samples ...)` | `kernel.cljs:100-127` |
| random-walk(addr, σ) | `(random-walk model addr std)` | `kernel.cljs:150-172` |
| proposal(f, b, sym?) | `(proposal model fwd bwd opts)` | `kernel.cljs:181-214` |
| gibbs([S₁,…,Sₘ]) | `(gibbs model addrs)` | `kernel.cljs:216-223` |
| MH acceptance | `(u/accept-mh? weight key)` | `inference/util.cljs` |
| Regenerate weight = log MH ratio | `weight_regen` derivation | `semantics.md` §4.4 |
| ProposalEdit detailed balance | Forward-backward swap | `edit-duality.md` §3 |

---

## References

- Hastings, W. K. (1970). "Monte Carlo sampling methods using Markov
  chains and their applications." *Biometrika*, 57(1), 97-109.
- Tierney, L. (1994). "Markov chains for exploring posterior
  distributions." *Annals of Statistics*, 22(4), 1701-1728.
- Roberts, G. O., & Rosenthal, J. S. (2004). "General state space
  Markov chains and MCMC algorithms." *Probability Surveys*, 1, 20-71.
- Cusumano-Towner, M. F., et al. (2019). "Gen: A General-Purpose
  Probabilistic Programming System." *PLDI 2019*.
