# GenMLX Cross-System Verification Plan

## Goal

Verify GenMLX correctness "to the bones" using Gen.jl and GenJAX as oracle
implementations. All three implement the Generative Function Interface (GFI),
so outputs should agree numerically (up to floating-point tolerance).

## Why this matters

GenMLX was built without deep probabilistic programming background knowledge.
Gen.jl (Julia, MIT/probcomp) and GenJAX (JAX, femtomc) are established
implementations with academic pedigree. Using them as ground truth gives
confidence that GenMLX's math is correct.

---

## API Differences Between Systems

| Aspect | Gen.jl | GenJAX | GenMLX |
|---|---|---|---|
| Score sign | `+log P` | **`-log P` (negated!)** | `+log P` |
| generate signature | `generate(gf, args, constraints)` | `gf.generate(constraints, *args)` | `(p/generate gf args constraints)` |
| Choicemaps | Dedicated ChoiceMap type, `:x => val` | Plain Python dicts `{"x": val}` | Clojure maps with Value/Node protocol |
| Addressing | Symbols `:x`, pairs `:x => :y` | Strings `"x"`, nested dicts | Keywords `:x`, paths `[:x :y]` |
| Combinators | Map, Unfold, Switch, Recurse | Vmap, Scan, Cond | Map, Unfold, Switch, Scan, Mask, Mix, Recurse |
| Randomness | Implicit global RNG | Explicit JAX PRNG keys | Threaded PRNG keys via metadata |
| Distributions | 20 built-in | 33 (via TFP) | 27 + Wishart, IID, Mixture, Product |
| Inference | IS, MH, HMC, MALA, ESS, SMC, VI | MH, MALA, HMC, SMC, VI, ADEV | IS, MH, HMC, NUTS, MALA, Gibbs, ESS, SMC, VI, ADEV, MAP + analytical |

### Critical: GenJAX negates scores

GenJAX's `get_score()` returns `-log P(choices)`, while Gen.jl and GenMLX
return `+log P(choices)`. All comparisons must account for this sign flip.

---

## Verification Categories

### 1. Distribution Log-Probabilities (Foundation)

Everything downstream depends on correct log-probs. For each distribution
shared across systems, evaluate `log-prob(dist, value, params...)` at
identical points and compare numerically.

**Distributions shared across all three systems (15):**

| Distribution | Gen.jl | GenJAX | GenMLX | Params |
|---|---|---|---|---|
| Normal/Gaussian | `normal(mu, std)` | `normal(loc, scale)` | `(dist/gaussian mu sigma)` | mu, sigma |
| Uniform | `uniform(lo, hi)` | `uniform(lo, hi)` | `(dist/uniform lo hi)` | lo, hi |
| Bernoulli | `bernoulli(p)` | `bernoulli(probs=p)` | `(dist/bernoulli p)` | p |
| Beta | `beta(a, b)` | `beta(a, b)` | `(dist/beta-dist a b)` | alpha, beta |
| Gamma | `gamma(shape, scale)` | `gamma(conc, rate)` | `(dist/gamma shape rate)` | shape, rate/scale |
| Exponential | `exponential(rate)` | `exponential(rate)` | `(dist/exponential rate)` | rate |
| Poisson | `poisson(lambda)` | `poisson(rate)` | `(dist/poisson rate)` | rate |
| Categorical | `categorical(probs)` | `categorical(logits)` | `(dist/categorical logits)` | probs vs logits |
| Dirichlet | `dirichlet(alpha)` | `dirichlet(conc)` | `(dist/dirichlet alpha)` | concentration |
| MVN | `mvnormal(mu, cov)` | `multivariate_normal(mu, cov)` | `(dist/multivariate-normal mu cov)` | mean, cov |
| Laplace | `laplace(loc, scale)` | `laplace(loc, scale)` | `(dist/laplace loc scale)` | loc, scale |
| Student-t | `cauchy` (special case) | `student_t(df, loc, scale)` | `(dist/student-t df loc scale)` | df, loc, scale |
| Cauchy | `cauchy(loc, scale)` | `cauchy(loc, scale)` | `(dist/cauchy loc scale)` | loc, scale |
| Binomial | `binom(n, p)` | `binomial(n, p)` | `(dist/binomial n p)` | n, p |
| Geometric | `geometric(p)` | `geometric(p)` | `(dist/geometric p)` | p |

**Parameterization gotchas:**
- Gamma: Gen.jl uses (shape, **scale**), GenJAX uses (concentration, **rate**). rate = 1/scale.
- Categorical: Gen.jl uses **probs**, GenJAX uses **logits**, GenMLX uses **logits**.
- GenMLX beta is `beta-dist` (not `beta` — that's a ClojureScript builtin).

**Test method:** For each distribution, pick 5 representative (value, params) tuples
including edge cases. Evaluate log-prob in all three systems. Compare with tolerance 1e-5.

**Edge cases per distribution:**
- Values at/near support boundaries
- Extreme parameter values (very small sigma, large alpha)
- Known analytical values (e.g., standard normal at 0 = -0.9189...)

### 2. GFI Operation Semantics (Core Contracts)

**Method:** Define identical models in all three systems. Fix choicemap values
via `generate` or `assess` (no randomness). Compare scores and weights.

#### 2a. Score = sum of log-probs (simulate/assess)

For a model with known choices, `score = sum_i log-prob(dist_i, value_i, params_i)`.

Test models:
1. **Single site:** `slope ~ Normal(0, 10)` — score = logpdf(normal, slope, 0, 10)
2. **Two sites:** `slope ~ Normal(0,10), intercept ~ Normal(0,10)` — score = sum
3. **Dependent sites:** `mu ~ Normal(0,1); x ~ Normal(mu, 1)` — score = logpdf(mu) + logpdf(x|mu)
4. **Mixed types:** `p ~ Beta(2,2); x ~ Bernoulli(p)` — score = logpdf(p) + logpmf(x|p)
5. **Loop model:** `y_i ~ Normal(slope * x_i + intercept, 1)` for i=0..4

For each model: fix all choices, call `assess` in all three systems, compare weight (= score).

#### 2b. Generate weight (importance weight)

`weight = log P(constrained choices | unconstrained choices sampled from prior)`

For the default internal proposal (sample unconstrained from prior):
- Constrain observations, leave latents free
- Weight should equal log-prob of constrained values given sampled latents

Test: fix the same latent values across systems (via seeding or assess),
constrain observations, compare weights.

#### 2c. Update weight (differential)

`weight = new_score - old_score` (for same-structure updates without added/removed sites)

More precisely:
- Changed constrained addresses: `weight += new_lp - old_lp`
- Unchanged addresses: no contribution
- New addresses: `weight += new_lp`
- Removed addresses: `weight -= old_lp`

Test: create a trace, update one address, compare weight in Gen.jl and GenMLX.
(GenJAX update is available but less tested.)

#### 2d. Regenerate weight

`weight = new_score - old_score` for selected addresses only.

Property: `old_score == new_score - weight`

Test: create trace, regenerate selected addresses, verify invariant in
Gen.jl and GenMLX.

#### 2e. Assess = fully constrained generate

`assess(gf, args, all_choices).weight == generate(gf, args, all_choices).weight`

And this equals the joint log-probability (score).

### 3. Combinator Semantics

**Combinators shared across systems:**

| Combinator | Gen.jl | GenJAX | GenMLX |
|---|---|---|---|
| Map | `Map(kernel)` | `gf.vmap(...)` | `(map-gf kernel)` |
| Unfold | `Unfold(kernel)` | `Scan(kernel)` | `(unfold-gf kernel)` |
| Switch | `Switch(gf1, gf2, ...)` | `Cond(true_gf, false_gf)` | `(switch-gf gf1 gf2 ...)` |

**Test method:** Define equivalent combinator models in each system, fix choices,
compare scores. Key properties:
- Map score = sum of per-element kernel scores
- Unfold score = sum of sequential step scores
- Switch score = score of selected branch only

### 4. Inference Algorithm Posteriors

Use models with known analytical posteriors (conjugate pairs). Run inference
in all three systems, compare posterior statistics.

**Conjugate test models:**

| Model | Prior | Likelihood | Analytical Posterior |
|---|---|---|---|
| Normal-Normal | `mu ~ N(0, sigma_0)` | `x_i ~ N(mu, sigma)` | `N(mu_post, sigma_post)` |
| Beta-Bernoulli | `p ~ Beta(a, b)` | `x_i ~ Bernoulli(p)` | `Beta(a + sum(x), b + n - sum(x))` |
| Gamma-Poisson | `lambda ~ Gamma(a, b)` | `x_i ~ Poisson(lambda)` | `Gamma(a + sum(x), b + n)` |
| Dirichlet-Categorical | `p ~ Dirichlet(alpha)` | `x_i ~ Categorical(p)` | `Dirichlet(alpha + counts)` |
| Normal-Normal (variance) | `sigma ~ InvGamma(a,b)` | `x_i ~ N(mu, sigma)` | `InvGamma(a', b')` |

**Inference algorithms to cross-validate:**

| Algorithm | Gen.jl | GenJAX | GenMLX |
|---|---|---|---|
| Importance Sampling | `importance_sampling` | N/A (manual) | `importance-sampling` |
| MH | `mh(trace, selection)` | `mh(trace, selection)` | `(mcmc/mh trace selection)` |
| HMC | `hmc(trace, selection)` | `hmc(trace, selection)` | `(mcmc/hmc trace selection)` |
| MALA | `mala(trace, selection, tau)` | `mala(trace, selection, step)` | `(mcmc/mala trace selection step)` |
| SMC | `particle_filter_step!` | `smc.extend` | `(smc/smc ...)` |
| VI (ELBO) | `black_box_vi!` | `elbo_vi` | `(vi/vi ...)` |

**What to compare:**
- Posterior mean (should match analytical within MC error)
- Posterior variance (should match analytical within MC error)
- Log-marginal-likelihood estimate (IS, SMC)
- Acceptance rates (MH, HMC, MALA — should be in similar ranges)
- ELBO convergence (VI)

### 5. Gradient Correctness

Compare gradients of log-prob w.r.t. parameters and values.

**Method:** For each distribution, compute:
1. `d/dx log-prob(dist, x, params)` — gradient w.r.t. value
2. `d/d(param_i) log-prob(dist, x, params)` — gradient w.r.t. each parameter

Compare against:
- Gen.jl: `logpdf_grad` (all distributions have this)
- GenJAX: JAX autodiff (`jax.grad`)
- Finite differences as fallback: `(f(x+h) - f(x-h)) / 2h`

### 6. Numerical Stability

Test edge cases where naive implementations fail:
- `log-prob(normal, x, 0, 1e-10)` — very small sigma
- `log-prob(beta, 0.0001, 0.5, 0.5)` — near boundary
- `log-prob(gamma, 1e-10, 0.5, 1.0)` — near zero
- `log-prob(categorical, k, [1e-300, 1-1e-300])` — extreme probabilities
- `logsumexp` correctness for very negative values

---

## Implementation Approaches

### Approach A: Static Cross-System Test Harness (primary)

Write identical models in all three languages. Fix inputs deterministically
(no randomness). Output results to JSON. Compare numerically.

```
cross_system_tests/
  VERIFICATION_PLAN.md          ← this file
  test_genjax.py                ← GenJAX test runner
  test_gen_jl.jl                ← Gen.jl test runner
  test_genmlx.cljs              ← GenMLX test runner
  models/                       ← model definitions per system
    gen_jl/
      distributions.jl          ← log-prob evaluation at fixed points
      gfi_contracts.jl          ← GFI operation tests
      combinators.jl            ← combinator tests
      inference.jl              ← inference convergence tests
    genjax/
      distributions.py
      gfi_contracts.py
      combinators.py
      inference.py
    genmlx/
      distributions.cljs
      gfi_contracts.cljs
      combinators.cljs
      inference.cljs
  results/                      ← JSON outputs for comparison
    gen_jl/
    genjax/
    genmlx/
  compare.py                    ← numerical comparison script
  run_all.sh                    ← run all systems, compare results
```

**Workflow:**
1. Each system evaluates the same computations, writes results to JSON
2. `compare.py` loads all JSON files, compares numerically with tolerance
3. Reports pass/fail per test, highlights discrepancies

### Approach B: MCP Servers (interactive, for debugging)

Create MCP servers for Gen.jl and GenJAX so Claude Code can query them
interactively during verification and debugging.

**Gen.jl MCP Server:**
- Julia process with Gen loaded
- Endpoints: `eval_logpdf`, `simulate`, `generate`, `assess`, `update`
- Communication: stdin/stdout JSON-RPC (MCP protocol)

**GenJAX MCP Server:**
- Python process with GenJAX loaded
- Same endpoints
- Communication: stdin/stdout JSON-RPC (MCP protocol)

**Use cases:**
- When a distribution log-prob disagrees, query the oracle interactively
- When a GFI weight is wrong, step through the computation
- When inference doesn't converge, compare intermediate states

---

## Verification Matrix

| Category | # Tests | Oracle(s) | Method | Priority |
|---|---|---|---|---|
| Distribution log-probs (15 shared) | ~75 | Gen.jl + GenJAX | Fixed-point comparison | **P0** |
| Distribution log-probs (GenMLX-only) | ~60 | scipy/analytical | Reference values | P1 |
| GFI score (assess) | ~10 models | Gen.jl + GenJAX | Same choices, compare | **P0** |
| GFI weight (generate) | ~10 models | Gen.jl + GenJAX | Same constraints, compare | **P0** |
| GFI weight (update) | ~10 models | Gen.jl | Same old/new, compare | P1 |
| GFI weight (regenerate) | ~10 models | Gen.jl | Invariant check | P1 |
| Combinator scores | 3 combinators | Gen.jl | Same structure, compare | P1 |
| Conjugate posteriors | 5 families | Analytical + all three | Mean/variance comparison | P1 |
| Inference convergence | IS, MH, HMC, SMC, VI | Gen.jl + GenJAX | Posterior statistics | P2 |
| Gradient correctness | 15 distributions | Gen.jl + finite diff | Value comparison | P2 |
| Numerical stability | ~20 edge cases | All three | Tolerance comparison | P2 |

**Estimated total: ~250-300 cross-system verification points**

---

## Execution Plan

### Phase 1: Distribution Log-Probs (P0)
1. Write log-prob evaluation scripts for all three systems
2. 15 shared distributions x 5 test points = 75 comparisons
3. Output to JSON, run comparison
4. Fix any discrepancies found

### Phase 2: GFI Contracts (P0)
1. Define 10 canonical models in all three systems
2. Fix choicemaps, evaluate assess/generate in all three
3. Compare scores and weights
4. Fix any discrepancies

### Phase 3: Extended Coverage (P1)
1. Update and regenerate contracts (Gen.jl only — GenJAX less mature here)
2. Combinator score verification
3. Conjugate posterior convergence tests

### Phase 4: Deep Verification (P2)
1. Inference algorithm cross-validation
2. Gradient correctness
3. Numerical stability edge cases
4. Build MCP servers if interactive debugging needed

---

## Success Criteria

- All P0 tests pass with tolerance < 1e-5
- All P1 tests pass with tolerance < 1e-4
- All P2 tests pass with tolerance < 1e-3 (statistical tests may need wider tolerance)
- Zero unexplained discrepancies between systems
- Any explained discrepancies documented (e.g., parameterization differences)
