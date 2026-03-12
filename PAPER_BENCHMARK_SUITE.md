# GenMLX Paper Benchmark Suite — TOPML Journal

## Overview

This document specifies the complete benchmark suite for the GenMLX TOPML paper.
The suite produces **all** experimental results, figures, and tables for the paper.

**Key design goal:** The current paper benchmarks use dynamic addresses, loops, and
`mx/eval!` inside model bodies — so L1-L4 compilation never fires. This redesigned
suite includes **static-address models** alongside the existing dynamic models,
enabling the compilation ladder story to be told with real measurements.

**Three tiers:**
1. **GenMLX benchmarks** — all experiments (nbb + Bun)
2. **Gen.jl benchmarks** — cross-system comparison subset (Julia)
3. **GenJAX benchmarks** — cross-system comparison subset (Python/JAX)

---

## Experiment Summary

| # | Experiment | Levels Exercised | Cross-System | Figure(s) | Table(s) |
|---|-----------|-----------------|-------------|-----------|----------|
| 1 | Compilation Ladder | L0→L4 | GenMLX only | fig_ladder_bar | tab_ladder |
| 2 | Vectorization Scaling | L0 | GenMLX, GenJAX | fig_particle_scaling | — |
| 3 | FFI Bottleneck | L0 | GenMLX, GenJAX | fig_ffi_scaling | tab_ffi |
| 4A | LinReg Correctness | L0–L3 | All three | fig_linreg_posterior | tab_linreg |
| 4B | HMM Correctness | L0, L2 | All three | fig_hmm_logml | — |
| 4C | GMM Correctness | L0 | All three | fig_gmm_logml | — |
| 5 | Auto-Conjugacy | L3, L3.5 | GenMLX only | fig_conjugacy_bars | tab_conjugacy |
| 6 | Compiled Inference | L2 | GenMLX only | — | tab_compiled |
| 7 | Cross-System Comparison | L0 (+ L4 row) | All three | fig_system_bars | tab_system |
| 8 | Neal's Funnel | L0 | GenMLX, Gen.jl | fig_funnel_posterior | tab_funnel |
| 9 | Changepoint Detection | L0 | GenMLX, Gen.jl | fig_changepoint | tab_changepoint |
| 10 | Fused Optimization | L4 | GenMLX only | fig_loss_curves | tab_optimization |
| 11 | Verification | All | Cross-system | fig_verification | tab_verification |

**Total: 11 experiments, ~12 figures, ~8 tables**

---

## Models

### Model A: Static Linear Regression (L1–L4 flagship)

The star model. Static addresses → L1 compiles. Normal-Normal → L3 eliminates.
`fit` → L4 auto-selects. Analytic posterior → ground truth.

```clojure
;; Static version: 7 trace sites, all literal keywords
(def static-linreg
  (gen [x1 x2 x3 x4 x5]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x1) intercept) 1))
      (trace :y2 (dist/gaussian (mx/add (mx/multiply slope x2) intercept) 1))
      (trace :y3 (dist/gaussian (mx/add (mx/multiply slope x3) intercept) 1))
      (trace :y4 (dist/gaussian (mx/add (mx/multiply slope x4) intercept) 1))
      (trace :y5 (dist/gaussian (mx/add (mx/multiply slope x5) intercept) 1))
      slope)))
```

**Compilation ladder on this model:**
- L0: Handler-based simulate/generate (~10 FFI calls per evaluation)
- L1: Compiled simulate/generate (noise transforms, 1 Metal dispatch)
- L2: Compiled MH chain (pre-generated proposals, 1 dispatch per N steps)
- L3: Normal-Normal conjugacy → :slope/:intercept eliminated → exact posterior
- L4: `(fit static-linreg args data)` → auto-selects :exact, returns analytic posterior

### Model B: Dynamic Linear Regression (L0 baseline)

```clojure
;; Dynamic version: loop over data, computed addresses
(def dynamic-linreg
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))
```

Same model, different expression. Dynamic addresses → L1 can't compile. Shows
graceful degradation: same semantics, handler path only.

### Model C: HMM (temporal, L0 + L2)

```clojure
;; 2-state Gaussian-emission HMM via Unfold combinator
;; T=50 timesteps, A=[[0.9,0.1],[0.1,0.9]], emissions N(-2,1)/N(2,1)
```

Temporal structure → L2 compiled particle filter. The Unfold combinator
enables compiled scan (T steps as single Metal dispatch).

### Model D: GMM (mixture, L0)

```clojure
;; K=3 Gaussian mixture, N=8 data points
;; Means [-4, 0, 4], sigma=1.0, equal weights
;; Ground truth: exact enumeration (3^8 = 6561 configurations)
```

Mixture with discrete latents. Tests enumeration, IS, Gibbs.

### Model E: Bayesian Mean Estimation (pure conjugate, L3)

```clojure
(def mean-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1))
      mu)))
```

Pure Normal-Normal. L3 eliminates `:mu` entirely → exact posterior, zero
sampling needed. Ground truth: `mu_post = (sum(y) / n) / (1 + sigma^2/(n * prior^2))`.

### Model F: Beta-Bernoulli (conjugate, L3)

```clojure
(def coin-model
  (gen []
    (let [theta (trace :theta (dist/beta-dist 2 2))]
      (trace :y1 (dist/bernoulli theta))
      (trace :y2 (dist/bernoulli theta))
      (trace :y3 (dist/bernoulli theta))
      (trace :y4 (dist/bernoulli theta))
      (trace :y5 (dist/bernoulli theta))
      theta)))
```

Beta-Bernoulli conjugacy. Ground truth: Beta(2 + n_heads, 2 + n_tails).

### Model G: Gamma-Poisson (conjugate, L3)

```clojure
(def count-model
  (gen []
    (let [rate (trace :rate (dist/gamma 2 1))]
      (trace :y1 (dist/poisson rate))
      (trace :y2 (dist/poisson rate))
      (trace :y3 (dist/poisson rate))
      (trace :y4 (dist/poisson rate))
      rate)))
```

Gamma-Poisson conjugacy. Ground truth: Gamma(2 + sum(y), 1 + n).

### Model H: Mixed Conjugate (partial L3)

```clojure
(def mixed-model
  (gen []
    (let [mu-a  (trace :mu-a (dist/gaussian 0 10))   ;; conjugate (eliminated)
          mu-b  (trace :mu-b (dist/gaussian 0 10))   ;; conjugate (eliminated)
          sigma (trace :sigma (dist/gamma 2 1))]      ;; NOT conjugate (sampled)
      (trace :y1 (dist/gaussian mu-a sigma))
      (trace :y2 (dist/gaussian mu-a sigma))
      (trace :y3 (dist/gaussian mu-b sigma))
      (trace :y4 (dist/gaussian mu-b sigma))
      sigma)))
```

3 latent dimensions → L3 eliminates 2 (mu-a, mu-b), samples 1 (sigma).
Demonstrates partial conjugacy: 67% dimensionality reduction.

### Model I: Neal's Funnel (pathological, L0)

```clojure
;; v ~ N(0, 3), x_i ~ N(0, exp(v/2)) for i=1..10
;; Pathological geometry: exponentially varying scale
```

Tests gradient-based MCMC. NUTS essential for good exploration.

### Model J: Changepoint (data-dependent structure, L0)

```clojure
;; T=100 timesteps, p_change=0.05
;; if (sample :change_t bernoulli(0.05)) then new_segment else same
;; Requires mx/eval! inside model body for branching
```

Data-dependent random structure. Only expressible with dynamic addresses.
Tests SMC (structure-exploiting) vs IS (prior proposal).

---

## Experiment Details

### Exp 1: Compilation Ladder (NEW — the star experiment)

**Purpose:** Demonstrate progressive compilation on the same model.

**Model:** Static Linear Regression (Model A), 5 observations.

**Runs:**

| Level | Configuration | What fires |
|-------|--------------|-----------|
| L0-handler | `p/generate` via handler path (force handler) | Handler transitions only |
| L1-compiled | `p/generate` via compiled path | Compiled simulate + generate |
| L2-mh-handler | 500-step MH via handler loop | Per-step handler generate |
| L2-mh-compiled | 500-step MH via compiled chain | Pre-generated noise, 1 dispatch |
| L3-conjugate | `p/generate` with auto-conjugacy | Normal-Normal elimination |
| L4-fit | `(fit static-linreg args data)` | Auto method selection → :exact |
| L4-fit-learn | `(fit static-linreg args data {:learn [:slope :intercept]})` | Compiled Adam + gradient |

**Metrics per run:**
- Wall-clock time (ms), averaged over 50 runs with 10 warmup
- Speedup vs L0-handler
- Posterior accuracy: |E[slope] - true| and |E[intercept] - true|
- KL divergence from analytic posterior (where applicable)
- Number of Metal dispatch calls (estimated from architecture)

**Expected results:**

| Level | Expected time | Speedup | Accuracy |
|-------|-------------|---------|----------|
| L0-handler | ~5ms | 1x | Depends on N particles |
| L1-compiled | ~1ms | ~5x | Same (verified identical) |
| L2-mh-compiled | ~100ms/500 steps | ~10x vs L2-handler | Better (more steps/second) |
| L3-conjugate | <0.1ms | ~50x | Exact posterior |
| L4-fit | <0.1ms | ~50x | Exact (auto-detected) |
| L4-fit-learn | ~20ms/200 iter | ~9x vs handler loop | Converges to true |

**Figure: `fig_ladder_bar`** — Grouped bar chart: time (ms) at each level. Log scale
y-axis. Annotations showing what changes at each level. This is the hero figure.

**Table: `tab_ladder`** — Full metrics for each level.

**Gen.jl/GenJAX:** Not applicable (they don't have compilation ladders). But we
include their L0-equivalent times in the cross-system comparison (Exp 7).

---

### Exp 2: Vectorization Scaling (UPDATED)

**Purpose:** Show shape-based batching is O(1) in particle count.

**Model:** Dynamic Linear Regression (Model B), 7 observation sites.

**Sweep:** N ∈ {1, 10, 50, 100, 500, 1000, 5000, 10000}

**Runs per N:**
- Sequential: N independent `p/generate` calls
- Batched: Single `dyn/vgenerate` call with [N]-shaped arrays

**Metrics:**
- Time (ms) for sequential vs batched
- Speedup ratio
- Memory usage (estimated)

**Expected:** Sequential O(N), batched O(1). At N=1000, ~1700x speedup.

**Figure: `fig_particle_scaling`** — Log-log plot with two lines:
- Sequential (steep, slope ~1)
- Batched (flat, <1ms)
- Annotations at N=100, 500, 1000 showing speedup

**GenJAX comparison:** Run equivalent model with `jax.vmap` for batching.
Plot as third line to compare GenMLX broadcasting vs JAX vmap.

---

### Exp 3: FFI Bottleneck (UPDATED)

**Purpose:** Show per-site overhead and how vector distributions eliminate it.

**Model:** Linear regression with D features, M=50 data points, N=10000 particles.

**Sweep:** D ∈ {10, 25, 50, 100, 200}

**Three configurations:**
1. Per-site: One `dist/gaussian` call per observation (D FFI calls)
2. gaussian-vec: Single `dist/gaussian-vec` call (1 FFI call)
3. GenJAX: Equivalent model on JAX (JIT, vectorized)

**Metrics:**
- Time (ms) per configuration per D
- Speedup: gaussian-vec / per-site
- Speedup: gaussian-vec / GenJAX

**Figure: `fig_ffi_scaling`** — Log-scale y-axis, D on x-axis. Three lines:
- Per-site (steep, O(D))
- gaussian-vec (flat, O(1))
- GenJAX (moderate slope)

**Table: `tab_ffi`** — Speedup ratios at each D.

---

### Exp 4A: Linear Regression Correctness (UPDATED)

**Purpose:** Validate inference algorithms against analytic posterior.

**Model:** Static Linear Regression (Model A), 20 observations (centered x-values).

**Algorithms:**
1. Compiled MH (500 samples, L1 compiled)
2. Vectorized Compiled Trajectory MH (10 chains × 500 steps, L2)
3. HMC (500 samples, 10 leapfrog steps)
4. NUTS (500 samples, adaptive)
5. ADVI (1000 iterations)
6. Vectorized IS (10000 particles, L0)
7. Conjugate exact (L3, zero-annotation)
8. fit API (L4, auto-selected)

**Ground truth:** Analytic Normal-Normal conjugate posterior.

**Metrics per algorithm:**
- E[slope], E[intercept] (point estimates)
- |error| from analytic posterior
- ESS (effective sample size)
- R-hat (multi-chain convergence, where applicable)
- Wall-clock time (ms)

**Figure: `fig_linreg_posterior`** — Marginal posterior distributions for slope
and intercept. Overlay analytic posterior (black dashed). One panel per algorithm.

**Table: `tab_linreg`** — Algorithm × metric matrix. Highlight L3 exact row.

**Cross-system:** Run algorithms 1-6 (or closest equivalents) on Gen.jl and GenJAX.

---

### Exp 4B: HMM Correctness (UPDATED)

**Purpose:** Demonstrate IS weight degeneracy vs SMC on temporal models.

**Model:** HMM (Model C), T=50 timesteps.

**Algorithms:**
1. IS (prior proposal), N ∈ {100, 500, 1000, 5000}
2. Vectorized IS, N ∈ {100, 500, 1000, 5000}
3. SMC (bootstrap), N ∈ {10, 50, 100}
4. Compiled Particle Filter (L2), N = 100

**Ground truth:** Forward algorithm, log P(y) ≈ -79.1.

**Metrics:**
- log-ML estimate ± std
- ESS
- Time (ms)

**Figure: `fig_hmm_logml`** — log-ML estimates vs N. Show IS converging
slowly (weight degeneracy) vs SMC converging quickly.

**Cross-system:** Run IS and SMC on Gen.jl and GenJAX.

---

### Exp 4C: GMM Correctness (UNCHANGED)

**Purpose:** Test discrete latent variable inference.

**Model:** GMM (Model D), K=3, N=8.

**Algorithms:**
1. Exact enumeration (3^8 = 6561 configurations)
2. IS (prior proposal), N = 10000
3. Vectorized IS, N = 10000
4. Gibbs (500 sweeps)

**Ground truth:** Exact enumeration, log P(y) ≈ -20.38.

**Metrics:**
- log-ML estimate
- Posterior assignment probabilities
- Marginal mean accuracy (MAE)
- Time (ms)

**Figure: `fig_gmm_logml`** — Bar chart of log-ML estimates.

**Cross-system:** Run IS on Gen.jl and GenJAX.

---

### Exp 5: Automatic Conjugacy (NEW — key L3 showcase)

**Purpose:** Demonstrate zero-annotation conjugacy detection and elimination.

**Sub-experiments:**

| # | Model | Conjugate Family | Ground Truth |
|---|-------|-----------------|-------------|
| 5A | Mean Estimation (E) | Normal-Normal | Exact conjugate posterior |
| 5B | Coin Flip (F) | Beta-Bernoulli | Beta(2+k, 2+n-k) |
| 5C | Count Data (G) | Gamma-Poisson | Gamma(2+sum, 1+n) |
| 5D | Mixed Model (H) | Partial conjugacy | 2/3 eliminated, 1/3 sampled |
| 5E | Static LinReg (A) | Normal-Normal | Full elimination → exact |

**For each sub-experiment:**

Run 1: Standard IS (no conjugacy, 10000 particles)
Run 2: L3 auto-conjugacy enabled (zero annotations)

**Metrics:**
- Posterior mean ± std
- Variance of estimator (over 20 independent runs)
- ESS
- Time (ms)
- Variance reduction ratio (Run 1 variance / Run 2 variance)
- Eliminated addresses (auto-detected)

**Expected results:**
- 5A: ~50x variance reduction (prior fully eliminated)
- 5B: Exact Beta posterior (zero variance)
- 5C: Exact Gamma posterior (zero variance)
- 5D: ~33x variance reduction (67% of dimensions eliminated)
- 5E: Exact posterior (all latents eliminated)

**Figure: `fig_conjugacy_bars`** — Grouped bars: variance(standard IS) vs
variance(L3 auto-conjugacy) per model. Log scale. Show variance reduction factor.

**Table: `tab_conjugacy`** — Family, model, eliminated addrs, variance reduction,
ESS improvement, time.

**GenMLX only** — Gen.jl and GenJAX don't have automatic conjugacy detection.

---

### Exp 6: Compiled Inference Sweeps (UPDATED — L2 showcase)

**Purpose:** Show compiled MCMC chains and SMC sweeps vs handler loops.

**Sub-experiments:**

| # | Algorithm | Model | Handler baseline | Compiled path |
|---|-----------|-------|-----------------|---------------|
| 6A | MH chain | Static LinReg (A) | 500-step handler loop | Pre-generated noise, compiled scan |
| 6B | SMC sweep | HMM (C) | T-step handler loop | Compiled particle filter |
| 6C | Score fn | Static LinReg (A) | Handler-based scoring | Tensor-native compiled score |
| 6D | HMC overhead | Static LinReg (A) | Hand-optimized MLX | GenMLX GFI HMC |

**Metrics:**
- Time (ms): handler loop vs compiled
- Speedup ratio
- Correctness: same posterior/log-ML (verified identical)

**Expected:**

| Config | Handler | Compiled | Speedup |
|--------|---------|----------|---------|
| 6A MH (500 steps) | ~1100ms | ~100ms | ~11x |
| 6B SMC (T=50, N=100) | ~2000ms | ~25ms | ~80x |
| 6C Score function | ~54ms | ~8ms | ~7x |
| 6D HMC abstraction | hand: 357ms | GFI: 391ms | 0.91x (9% overhead) |

**Table: `tab_compiled`** — Configuration, handler time, compiled time, speedup,
correctness verification.

---

### Exp 7: Cross-System Comparison (UPDATED — adds L4 row)

**Purpose:** Fair comparison of GenMLX vs Gen.jl vs GenJAX.

**Models × Algorithms:**

| Config | Model | Algorithm | Particles/Samples |
|--------|-------|-----------|------------------|
| IS-linreg | LinReg (B) | Vectorized IS | N=1000 |
| IS-gmm | GMM (D) | Vectorized IS | N=1000 |
| IS-hmm | HMM (C) | Vectorized IS | N=1000 |
| MH-linreg | LinReg (static A) | MH | 5000 steps |
| SMC-hmm | HMM (C) | SMC | N=100 |
| **L4-linreg** | LinReg (static A) | **fit API** | auto-selected |

The L4-linreg row is GenMLX-only and shows the compilation ladder payoff.

**Timing protocol (all systems):**
- 10 warmup runs (GPU/JIT warm)
- 50 timed runs
- Report: mean ± std, min

**Hardware:**
- GenMLX: Apple Silicon (Metal GPU)
- Gen.jl: Same machine (CPU, Julia JIT)
- GenJAX: Same machine (CPU, JAX XLA JIT)

**Figure: `fig_system_bars`** — Grouped bars on log scale. 5 groups (configs),
3 bars each (systems). Plus L4 annotation for GenMLX.

**Table: `tab_system`** — Config × system matrix with times and speedup ratios.
Include hardware notes column.

---

### Exp 8: Neal's Funnel (UNCHANGED + L4 row)

**Purpose:** Test gradient-based MCMC on pathological geometry.

**Model:** Neal's Funnel (Model I), D=10.

**Algorithms:**
1. NUTS (500 samples, adaptive dual-averaging)
2. HMC (500 samples, fixed step size)
3. MALA (500 samples)
4. Compiled MH (500 samples)
5. fit API (auto-selected method)

**Ground truth:** v ~ N(0, 3) marginally.

**Metrics:**
- E[v] error, std(v) error
- ESS (per dimension)
- ESS/sec (throughput)
- R-hat

**Figure: `fig_funnel_posterior`** — Marginal posterior of v and x_1.
NUTS (fills the funnel) vs MH (trapped in narrow region).

**Table: `tab_funnel`** — Algorithm comparison.

**Cross-system:** Compare NUTS on Gen.jl (considered gold standard for NUTS).

---

### Exp 9: Changepoint Detection (UNCHANGED)

**Purpose:** Show data-dependent random structure handling.

**Model:** Changepoint (Model J), T=100.

**Algorithms:**
1. SMC (N=250, via Unfold)
2. SMC (N=500, via Unfold)
3. IS (N=1000, prior proposal)

**Ground truth:** DP forward algorithm, log P(y) ≈ -174.17.

**Metrics:**
- log-ML error ± std
- Detected changepoints vs true
- ESS
- Time per particle

**Figure: `fig_changepoint`** — Time series with detected changepoints overlaid
on true changepoints. Top panel: data. Bottom panel: posterior change probability.

**Table: `tab_changepoint`** — Algorithm comparison.

---

### Exp 10: Fused Optimization (NEW — L4 showcase)

**Purpose:** Demonstrate compiled Adam + gradient in single Metal dispatch.

**Sub-experiments:**

| # | Task | Description |
|---|------|-------------|
| 10A | Compiled Adam | `co/learn` on static-linreg, 200 iterations |
| 10B | Fused MH+Adam | `co/fused-learn` with compiled MH + Adam |
| 10C | Method selection | `ms/select-method` on 6 diverse models |
| 10D | fit API e2e | `(fit model args data)` on all model types |

**10A: Compiled Adam vs handler loop**
- Handler: loop calling `p/generate` + extract gradient + Adam step + `mx/materialize!`
- Compiled: `co/make-compiled-opt-step` → single `mx/compile-fn` per iteration
- Model: Static LinReg (A), learn slope + intercept
- 200 iterations

**10B: Fused MH+Adam**
- Pre-generate all MCMC noise
- MH chain + Adam update in one compiled dispatch
- Compare vs separate MH sweep + Adam step

**10C: Method selection accuracy**
Run `ms/select-method` on:
- Static LinReg (fully conjugate) → expect :exact
- Mixed Model H (partial conjugate) → expect :hmc (2 eliminated, 1 residual ≤ 10)
- HMM (temporal) → expect :smc
- Dynamic LinReg (dynamic addresses) → expect :handler-is
- Funnel (static, 11 dims > 10) → expect :vi
- Empty model (no trace sites) → expect :exact

Verify all selections are correct.

**10D: fit API end-to-end**
Run `(fit model args data)` on each model type. Verify:
- Correct method auto-selected
- Posterior converges to ground truth (where known)
- Returns proper result map with all fields

**Metrics:**
- 10A: Time, speedup, loss convergence
- 10B: Time, speedup, posterior accuracy
- 10C: Method selected vs expected (6/6 correct)
- 10D: Method, time, posterior accuracy per model

**Figure: `fig_loss_curves`** — Loss vs iteration. Handler loop (orange, slow
convergence) vs compiled Adam (blue, same convergence, 9x less wall time).
Two panels: same loss trajectory, different x-axis (iteration vs time).

**Table: `tab_optimization`** — Configuration, handler time, compiled time,
speedup, final loss.

---

### Exp 11: Verification (UPDATED)

**Purpose:** Validate semantic correctness.

**Three verification levels:**

**11A: GFI Contracts**
- 11 contracts (simulate-trace, generate-weight, update-discard, etc.)
- 5 model architectures (single-site, multi-site, hierarchical, combinator, dynamic)
- Matrix: 55 checks

**11B: Gen.clj Compatibility**
- 165 tests ported from Gen.clj
- Expected: 162/165 (3 pre-existing beta/gamma edge cases documented)

**11C: GenJAX Compatibility**
- 73 tests ported from GenJAX
- Expected: 73/73

**11D: Level Certification**
- L0: 68/68 gates
- L1: 438/438 (schema 174 + compiled-sim 82 + partial 92 + combinator 90)
- L4: 41/41 gates

**Figure: `fig_verification`** — Two-panel:
- Left: Heatmap (11 contracts × 5 models), all green
- Right: Stacked bar showing pass counts per compatibility suite

**Table: `tab_verification`** — Suite, total, pass, fail, notes.

---

## File Structure

```
test/genmlx/paper/
  # GenMLX benchmark files (one per experiment)
  bench_01_compilation_ladder.cljs
  bench_02_vectorization.cljs
  bench_03_ffi_bottleneck.cljs
  bench_04a_linreg.cljs
  bench_04b_hmm.cljs
  bench_04c_gmm.cljs
  bench_05_conjugacy.cljs
  bench_06_compiled_inference.cljs
  bench_07_cross_system.cljs
  bench_08_funnel.cljs
  bench_09_changepoint.cljs
  bench_10_fused_optimization.cljs
  bench_11_verification.cljs
  # Shared utilities
  bench_util.cljs              # Timing, JSON output, assertion helpers
  models.cljs                  # All model definitions (imported by benchmarks)

bench/genjl/
  bench_linreg.jl              # IS, MH, HMC on linear regression
  bench_hmm.jl                 # IS, SMC on HMM
  bench_gmm.jl                 # IS on GMM
  bench_funnel.jl              # NUTS on funnel
  bench_system.jl              # System comparison timings
  util.jl                      # Timing helpers

bench/genjax/
  bench_linreg.py              # IS on linear regression
  bench_hmm.py                 # IS on HMM
  bench_gmm.py                 # IS on GMM
  bench_system.py              # System comparison timings
  util.py                      # Timing helpers

results/
  paper/                       # All results go here
    exp01_ladder/
    exp02_vectorization/
    exp03_ffi/
    exp04a_linreg/
    exp04b_hmm/
    exp04c_gmm/
    exp05_conjugacy/
    exp06_compiled/
    exp07_system/
    exp08_funnel/
    exp09_changepoint/
    exp10_optimization/
    exp11_verification/
```

---

## Runner

```bash
#!/bin/bash
# run_all_benchmarks.sh — execute complete paper benchmark suite

RUNNER="bun run --bun nbb"
DIR="test/genmlx/paper"

echo "=== GenMLX Paper Benchmark Suite ==="
echo "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
echo "MLX: $(node -e 'console.log(require("@frost-beta/mlx").version)' 2>/dev/null)"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

for exp in 01 02 03 04a 04b 04c 05 06 07 08 09 10 11; do
  echo "--- Experiment ${exp} ---"
  $RUNNER "${DIR}/bench_${exp}_*.cljs"
  echo ""
done

echo "=== Complete ==="
```

---

## Timing Protocol

All benchmarks follow the same protocol:

1. **Process warmup:** Import model, run 1 generate call to warm MLX shader cache
2. **GPU warmup:** 10 iterations of the full benchmark to warm Metal caches
3. **Measurement:** Nested loop:
   - Outer: 5 repetitions (captures run-to-run variance)
   - Inner: 10 iterations (take minimum to filter GC noise)
   - Record: min(inner) for each outer iteration
   - Report: mean ± std of outer mins
4. **Cold start note:** First invocation in a fresh process can be ~10x slower
   due to Metal shader compilation. Document this.

```clojure
;; Standard timing template
(defn benchmark [label f & {:keys [warmup-n outer-n inner-n]
                            :or {warmup-n 10 outer-n 5 inner-n 10}}]
  ;; Warmup
  (dotimes [_ warmup-n] (f) (mx/materialize!))
  ;; Measure
  (let [outer-times
        (for [_ (range outer-n)]
          (let [inner-times
                (for [_ (range inner-n)]
                  (let [t0 (js/performance.now)]
                    (f)
                    (mx/materialize!)
                    (- (js/performance.now) t0)))]
            (apply min inner-times)))
        mean-ms (/ (reduce + outer-times) (count outer-times))
        std-ms  (Math/sqrt (/ (reduce + (map #(* (- % mean-ms) (- % mean-ms))
                                             outer-times))
                              (max 1 (dec (count outer-times)))))]
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)}))
```

---

## JSON Output Format

Every experiment outputs a JSON file with:

```json
{
  "experiment": "exp01_compilation_ladder",
  "timestamp": "2026-03-12T...",
  "hardware": { "chip": "Apple M4", "memory_gb": 32, "gpu_cores": 10 },
  "software": { "runtime": "bun+nbb", "mlx_version": "0.20.3" },
  "model": { "name": "static-linreg", "n_trace_sites": 7, "static": true },
  "results": [
    {
      "level": "L0-handler",
      "mean_ms": 5.2,
      "std_ms": 0.3,
      "speedup": 1.0,
      "posterior": { "slope": { "mean": 2.01, "error": 0.01 }, ... }
    },
    ...
  ]
}
```

---

## Cross-System Model Specifications

### Gen.jl (Julia)

```julia
using Gen

# Linear regression (equivalent to Model B dynamic)
@gen function linreg(xs::Vector{Float64})
    slope ~ normal(0.0, 10.0)
    intercept ~ normal(0.0, 10.0)
    for (j, x) in enumerate(xs)
        {(:y, j)} ~ normal(slope * x + intercept, 1.0)
    end
    return slope
end

# HMM (equivalent to Model C)
@gen function hmm_step(t::Int, prev_state::Int, params)
    A, mus = params
    state ~ categorical(A[prev_state, :])
    y ~ normal(mus[state], 1.0)
    return state
end
hmm = Unfold(hmm_step)

# GMM (equivalent to Model D)
@gen function gmm(data::Vector{Float64}, K::Int, mus, sigma)
    for (i, y) in enumerate(data)
        z = {:z => i} ~ categorical([1/K for _ in 1:K])
        {:y => i} ~ normal(mus[z], sigma)
    end
end
```

### GenJAX (Python/JAX)

```python
import genjax
from genjax import gen, trace, Normal, Categorical
import jax.numpy as jnp

# Linear regression
@gen
def linreg(xs):
    slope = trace("slope", Normal(0.0, 10.0))
    intercept = trace("intercept", Normal(0.0, 10.0))
    for j, x in enumerate(xs):
        trace(f"y{j}", Normal(slope * x + intercept, 1.0))
    return slope

# Note: GenJAX requires JAX-traceable code (no Python control flow
# over random values for vmap). Dynamic models may need restructuring.
```

---

## What This Suite Proves (Paper Claims)

| Claim | Supported by |
|-------|-------------|
| Shape-based batching is O(1) in particles | Exp 2 |
| Per-site FFI overhead is the bottleneck | Exp 3 |
| GenMLX produces correct posteriors | Exp 4A/B/C |
| Compilation ladder gives progressive speedup | **Exp 1** (NEW) |
| Zero-annotation conjugacy reduces variance | **Exp 5** (NEW) |
| Compiled inference fuses multi-step sweeps | Exp 6 |
| GenMLX is competitive with Gen.jl/GenJAX | Exp 7 |
| Gradient-based MCMC handles pathologies | Exp 8 |
| Data-dependent structure is supported | Exp 9 |
| Fused optimization is 9x faster | **Exp 10** (NEW) |
| GFI semantics are preserved at all levels | Exp 11 |
| Models run unchanged across all levels | Exp 1 + Exp 11 |

---

## On shadow-cljs (AOT Compilation)

**Should we compile GenMLX with shadow-cljs?**

Short answer: **Yes, as a follow-up experiment.** Not for the initial suite.

**Why it matters:**
- nbb (the interpreter) adds ~200μs per FFI call in host overhead
- For batched inference (Exp 2): irrelevant — MLX computation dominates
- For sequential MCMC (Exp 7, MH): critical — 500 steps × ~200μs = 100ms of pure overhead
- This is the main reason Gen.jl is 12x faster on scalar MH

**What shadow-cljs would change:**
- AOT compilation → eliminate interpreter overhead
- Dead code elimination → smaller runtime
- Tree shaking → faster startup
- Expected: ~3-5x speedup on sequential MCMC, closing gap with Gen.jl to ~2-4x

**How to add it:**
1. Set up shadow-cljs build (`:target :node-script`)
2. Run the cross-system benchmarks (Exp 7) with compiled JS
3. Add a "GenMLX (compiled)" row to the system comparison table
4. Report: nbb vs shadow-cljs vs Gen.jl vs GenJAX

**For the paper:** Include as a discussion point: "nbb interpretation overhead
accounts for ~X% of sequential MCMC time. AOT compilation via shadow-cljs
reduces this to Y%, bringing GenMLX within Zx of Gen.jl on scalar algorithms."

**Recommendation:** Run the initial suite with nbb. Then do one shadow-cljs
experiment on the cross-system comparison to quantify the AOT benefit. Include
both numbers in the paper.

---

## Implementation Priority

1. **bench_util.cljs + models.cljs** — shared infrastructure
2. **bench_01_compilation_ladder.cljs** — the hero experiment (NEW)
3. **bench_05_conjugacy.cljs** — L3 showcase (NEW)
4. **bench_10_fused_optimization.cljs** — L4 showcase (NEW)
5. **bench_04a_linreg.cljs** — updated with L3/L4 rows
6. **bench_02_vectorization.cljs** — updated from existing
7. **bench_07_cross_system.cljs** — updated with L4 row
8. Remaining experiments (can largely adapt from existing paper_bench_*.cljs)
9. Gen.jl benchmarks (Julia)
10. GenJAX benchmarks (Python)
