# Level 3.5 Completion Plan: Extended Analytical Elimination

## Executive Summary

Level 3 made GenMLX **infer more and sample less** — for top-level static models
using `p/generate`. Level 3.5 extends this to the rest of the system:

1. **MCMC integration** — wire auto-handlers into `p/regenerate` so MH/Gibbs
   chains benefit from analytical elimination
2. **Combinator-aware conjugacy** — detect conjugate pairs inside `unfold`,
   `map`, `scan`, `switch`, `mix` kernels
3. **Multivariate conjugacy** — matrix-Normal, Wishart-Normal, and other
   matrix-valued conjugate families
4. **Assess integration** — wire auto-handlers into `p/assess` for correct
   marginal likelihood scoring

**Same architecture as L3.** No new abstractions. The address-based dispatch,
conjugacy table, affine analysis, dependency graph, and rewrite engine all
extend naturally. L3.5 is scope expansion, not architectural change.

**Why L3.5 before L4?** L4 (single fused graph) is a compilation-layer change.
L3.5 is a coverage-layer change — making existing analytical elimination reach
more of the inference stack. The ROI is high: every MCMC chain on a conjugate
model gets free variance reduction. Every `unfold` model with conjugate
kernels gets analytical elimination per step. And this work is prerequisite
for L4's automatic method selection (L4-M3), which needs to know what's
analytically tractable across the full model including combinators.

---

## Current State

### What Level 3 provides

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Conjugacy detection (5 families) | `conjugacy.cljs` | 165 | Complete |
| Affine expression analysis | `affine.cljs` | 379 | Complete |
| Dependency graph + d-separation | `dep_graph.cljs` | 262 | Complete |
| Graph rewriting engine (3 rules) | `rewrite.cljs` | 225 | Complete |
| Address-based analytical handlers | `inference/auto_analytical.cljs` | 393 | Complete |
| Auto-wiring in DynamicGF | `dynamic.cljs` | +133 | Complete |

### What Level 3 does NOT cover

| Gap | Impact | Current behavior |
|-----|--------|-----------------|
| `p/regenerate` unaware of auto-handlers | MH/Gibbs chains get no analytical benefit | Goes to vanilla `h/regenerate-transition` |
| `p/assess` unaware of auto-handlers | Score assessments don't use marginal LL | Goes to vanilla `h/assess-transition` |
| Conjugacy inside combinators | `unfold(N(mu,1))` with conjugate prior undetected | Only top-level DynamicGF trace sites detected |
| Multivariate conjugacy | Matrix-Normal, Wishart-Normal families missing | Only scalar 1D conjugacy |
| `make-score-fn` ignores conjugate structure | Compiled MCMC chains can't skip marginalized params | Treats all latents as free parameters |

### L3 conjugate families (5)

| Family | Prior | Likelihood | Key |
|--------|-------|-----------|-----|
| Normal-Normal (NN) | `gaussian` | `gaussian` | Mean param |
| Beta-Bernoulli (BB) | `beta-dist` | `bernoulli` | Prob param |
| Gamma-Poisson (GP) | `gamma-dist` | `poisson` | Rate param |
| Gamma-Exponential (GE) | `gamma-dist` | `exponential` | Rate param |
| Dirichlet-Categorical (DC) | `dirichlet` | `categorical` | Logits param |

---

## Architecture Overview

### How auto-handlers work today (L3)

```
1. gen macro captures source form
2. schema.cljs extracts trace sites, deps, dist-types
3. conjugacy.cljs detects conjugate pairs from schema
4. affine.cljs classifies dependency types (direct/affine/nonlinear)
5. dep_graph.cljs builds DAG, finds Kalman chains
6. rewrite.cljs generates rules (Kalman > Conjugacy > RaoBlackwell)
7. rewrite.cljs applies rules → produces {addr handler-fn} map
8. dynamic.cljs stores handlers on schema as :auto-handlers
9. DynamicGF.generate wraps h/generate-transition with address dispatch
```

### What L3.5 adds to the pipeline

```
Steps 1-8: UNCHANGED

9a. DynamicGF.generate: UNCHANGED (already wired)
9b. DynamicGF.regenerate: NEW — wrap h/regenerate-transition with address dispatch
9c. DynamicGF.assess: NEW — wrap h/assess-transition with address dispatch

10. Combinator-aware detection: NEW
    - When a combinator's kernel is a DynamicGF with :auto-handlers,
      the combinator's own generate/regenerate calls on the kernel
      automatically invoke those handlers (no combinator code change needed)
    - NEW: detect conjugacy at the combinator level — e.g., a prior at
      the outer model feeds into an observation inside an unfold kernel

11. Multivariate families: NEW
    - Extend conjugacy table with matrix-valued families
    - New update functions in auto_analytical.cljs
    - Affine analysis extended for matrix expressions
```

---

## Investigation Gates

Five gates. Each has: experiment, success criterion, decision.

### Gate 0: Regenerate Handler Correctness

**Experiment:** Take the NN model from L3 benchmarks (mu ~ N(0,10), y_i ~ N(mu,1)).
Wire auto-handlers into `p/regenerate`. Run 5000-step MH chain targeting `:mu`
with auto-handlers vs without. Compare posterior mean, variance, and acceptance
rate.

**Success criterion:**
- Posterior mean within 2σ of exact analytical posterior
- Posterior variance within 20% of exact
- Auto-handler regenerate produces valid traces (all GFI contracts hold)
- Weight semantics are correct: `new-score - old-score` matches handler path

**Decision:** If correctness holds → proceed with regenerate integration.
If weight accounting is wrong → investigate and fix before proceeding.

### Gate 1: Regenerate Performance Impact

**Experiment:** Same NN model. Compare wall-clock time for 5000-step MH chain
with and without auto-handlers in regenerate. Also measure ESS/second.

**Success criterion:**
- No more than 10% wall-clock slowdown from address dispatch overhead
- If analytical elimination reduces the number of sites to resample,
  ESS/second should improve

**Decision:** If slowdown > 10% → profile to find overhead source.
If ESS/second improves → the approach is validated for MCMC.

### Gate 2: Combinator Kernel Conjugacy

**Experiment:** Build an unfold model:
```clojure
(def kernel (gen [t state]
  (let [z (trace :z (dist/gaussian (mx/multiply 0.9 state) 1.0))]
    z)))
(def model (gen [T]
  (let [mu (trace :mu (dist/gaussian 0 10))]
    (splice :steps (unfold kernel T mu)))))
```
Where `:mu` is a prior and `:z` at step 0 is an observation.

Test whether L3's existing auto-handlers on the outer model correctly
intercept the `:mu` → `:z` conjugate pair when `:z` at step 0 is constrained.

**Success criterion:**
- If the outer model's auto-handlers work for top-level conjugate pairs
  where the obs site is INSIDE a combinator splice → easy case, no new code
- If they don't (because the splice boundary breaks address matching) →
  need combinator-level detection

**Decision:** Determines scope of combinator work. If top-level pairs
that cross splice boundaries already work → WP-2 is simpler. If not →
need explicit cross-boundary detection.

### Gate 3: Multivariate NN Correctness

**Experiment:** Implement Matrix-Normal conjugate update. Test on:
```clojure
(let [mu (trace :mu (dist/multivariate-gaussian [0 0] [[100 0][0 100]]))
      y1 (trace :y1 (dist/multivariate-gaussian mu [[1 0][0 1]]))]
  mu)
```
Compare marginal LL against known formula.

**Success criterion:**
- Marginal LL matches closed-form within 1e-4
- Posterior mean and covariance match analytical solution
- Works for dimensions 2, 5, 10

**Decision:** If correct → proceed with multivariate families.
If MLX matrix ops (cholesky, solve) have issues → defer multivariate to L4.

### Gate 4: End-to-End MCMC + Analytical Benchmark

**Experiment:** The multi-group mixed model from L3 benchmarks (3 NN groups +
2 non-conjugate params). Run a Gibbs sampler that:
- Analytically marginalizes the 3 conjugate groups
- Samples the 2 non-conjugate params via MH

Compare against L3's IS benchmark (200 particles, 15 trials).

**Success criterion:**
- ESS/second of Gibbs+analytical > ESS/second of IS+analytical
- Posterior statistics match within 2σ
- Demonstrates that L3.5 MCMC integration provides benefit beyond L3 IS

**Decision:** If Gibbs+analytical wins → L3.5 MCMC integration is validated.
If not → investigate whether the overhead of Gibbs setup outweighs the
analytical benefit for this model size.

---

## Work Packages

### WP-0: Regenerate Auto-Handler Integration

**Goal:** Wire auto-handlers into `p/regenerate` in DynamicGF so that MH/Gibbs
chains automatically benefit from analytical elimination.

**Why it matters:** MCMC is the most common inference method. If a model has
conjugate structure, `p/regenerate` should use it. This is the highest-ROI
change in L3.5 — one code change benefits every MH, Gibbs, MALA, HMC, and
NUTS chain run on a conjugate model.

**Architecture:**

The key challenge is that regenerate and generate have different semantics:

| Aspect | generate | regenerate |
|--------|----------|------------|
| **Input** | constraints (observations) | selection (which addrs to resample) |
| **Prior sites** | Sample from prior | If selected: resample. If not: keep old value |
| **Obs sites** | If constrained: use constraint. If not: sample | If selected: resample. If not: keep old value |
| **Weight** | Marginal LL of constrained obs | `(new-lp - old-lp)` for resampled sites |
| **Score** | Sum of all log-probs | Sum of all log-probs |

**For conjugate pairs in regenerate, the analytical handler must:**

Case A — Prior is selected, obs is constrained (common in Gibbs):
- Don't sample from prior; compute posterior given observations
- Sample from posterior (not prior) → Rao-Blackwellization
- Weight: `log p(obs | posterior-sample) - log p(obs | old-value)` adjusted
  for the proposal distribution change
- This is the **Gibbs update**: given observations, sample prior from
  its full conditional (= posterior)

Case B — Prior is NOT selected, obs is constrained:
- Prior keeps old value (standard regenerate behavior)
- Obs handler: use constraint value, compute marginal LL → same as generate
- Net effect: score contribution from obs uses exact marginal LL

Case C — Obs is NOT constrained (unconstrained observation):
- Fall through to base regenerate handler
- No analytical benefit

**The simplest correct approach:** For the initial implementation, handle only
Case B (prior not selected, obs constrained). This requires minimal changes:
the obs handler computes marginal LL exactly as in generate, the prior handler
initializes the posterior from its sampled value. Case A (Gibbs posterior
sampling) is a separate, more complex feature (WP-0b).

**Implementation in `dynamic.cljs`:**

In the `regenerate` method of DynamicGF, add auto-handler dispatch:

```clojure
;; In DynamicGF regenerate method, before handler fallback:
(let [use-auto? (and (:auto-handlers schema)
                     ;; Check if any conjugate obs addr is in old-choices
                     ;; (i.e., was constrained in the generate that produced this trace)
                     (some (fn [{:keys [obs-addr]}]
                             (cm/has-value? (cm/get-submap (:choices trace) obs-addr)))
                           (:conjugate-pairs schema)))
      transition (if use-auto?
                   (auto-analytical/make-address-dispatch
                     h/regenerate-transition (:auto-handlers schema))
                   h/regenerate-transition)
      result (rt/run-handler transition
               {:choices cm/EMPTY :score SCORE-ZERO
                :weight SCORE-ZERO
                :key key :selection selection
                :old-choices (:choices trace)
                ;; L3.5: add analytical state
                :auto-posteriors {}
                :auto-kalman-beliefs {}
                :auto-kalman-noise-vars {}
                :constraints (:choices trace) ;; obs values from old trace
                :executor execute-sub
                :param-store (::param-store (meta this))}
               (fn [rt] (apply body-fn rt (:args trace))))]
```

**Key insight:** The observation values are in `:old-choices` (from the trace
that was passed to regenerate). We set `:constraints` to the old trace's
choices so that the existing obs handlers (which check `:constraints`) work
unchanged.

**Modification to auto-handler obs handler:** Currently, the obs handler in
`auto_analytical.cljs` checks `(cm/get-submap (:constraints state) addr)`.
In regenerate, we need it to also check the old-choices for observation values.
The simplest fix: set `:constraints` in the regenerate state to include the
old observation values. Then the existing handler code works without modification.

**But:** We must NOT set constraints for addresses that the selection says
to resample. If an obs address is selected for resampling, it should NOT be
treated as constrained — it should be resampled from the likelihood.

**Refined approach:** Build a "regenerate constraints" map that includes only
observation values for conjugate obs addresses that are NOT in the selection:

```clojure
(let [obs-addrs (set (map :obs-addr (:conjugate-pairs schema)))
      regen-constraints
      (reduce (fn [cm addr]
                (if (and (contains? obs-addrs addr)
                         (not (sel/selected? selection addr)))
                  (let [old-val (cm/get-submap (:choices trace) addr)]
                    (if (cm/has-value? old-val)
                      (cm/set-value cm addr (cm/get-value old-val))
                      cm))
                  cm))
              cm/EMPTY
              (cm/addresses (:choices trace)))]
  ;; Use regen-constraints as :constraints in the handler state
  ...)
```

**New handler semantics for regenerate:**

For **prior handler** in regenerate:
- If prior is selected: sample from prior (base handler), but also init posterior
- If prior is NOT selected: use old value, init posterior from it
- In both cases: NO score/weight contribution (marginalized)

For **obs handler** in regenerate:
- If obs is in constraints (i.e., not selected): compute marginal LL → add to score+weight
- If obs IS selected: return nil → fall through to base (resample from likelihood)

The existing prior handler already returns posterior mean without score contribution.
For regenerate, we need a variant that:
1. Uses the old value (not posterior mean) when the prior is not selected
2. Still initializes the posterior state for downstream obs handlers

**New function in `auto_analytical.cljs`:**

```clojure
(defn make-regenerate-prior-handler
  "Build a prior handler for regenerate context.
   If prior is selected: base handler resamples, then we init posterior from new value.
   If prior is NOT selected: use old value, init posterior from it."
  [prior-addr init-posterior base-handler]
  (fn [state addr dist]
    (let [selected? (sel/selected? (:selection state) addr)
          old-val (cm/get-submap (:old-choices state) addr)]
      (if selected?
        ;; Selected: let base handler resample, then init posterior
        (let [[val state'] (base-handler state addr dist)
              posterior (init-posterior (:params dist))]
          [val (assoc-in state' [:auto-posteriors prior-addr] posterior)])
        ;; Not selected: use old value, init posterior
        (when (cm/has-value? old-val)
          (let [value (cm/get-value old-val)
                lp (dc/dist-log-prob dist value)
                posterior (init-posterior (:params dist))]
            [value (-> state
                     (update :choices cm/set-value addr value)
                     (update :score #(mx/add % lp))
                     (assoc-in [:auto-posteriors prior-addr] posterior))]))))))
```

**Files:**
- Modify: `src/genmlx/dynamic.cljs` (add auto-handler dispatch in regenerate)
- Modify: `src/genmlx/inference/auto_analytical.cljs` (add regenerate-aware handlers)
- New: `test/genmlx/l3_5_regenerate_test.cljs`

**Tests (~35):**
- NN model: regenerate with prior selected, obs constrained → valid trace
- NN model: regenerate with prior NOT selected, obs constrained → marginal LL correct
- NN model: MH chain with auto-handlers → posterior convergence
- BB model: same three tests
- GP model: same three tests
- Kalman chain: regenerate with auto-handlers → beliefs cascade correctly
- Mixed model: regenerate with partial conjugacy → correct weight
- Fallback: regenerate on non-conjugate model → unchanged behavior
- Contract: regenerate weight = (new-score - old-score) semantic check
- Edge: empty selection → no resampling, posterior still initialized
- Edge: all sites selected → full resample, posterior from new values

**Gate 0 experiment: correctness validation.**
**Gate 1 experiment: performance measurement.**

---

### WP-1: Assess Auto-Handler Integration

**Goal:** Wire auto-handlers into `p/assess` so that marginal likelihood
scoring uses analytical elimination.

**Why it matters:** `p/assess` is used for model comparison (Bayes factors),
cross-validation, and as a subroutine in some inference algorithms. When a model
has conjugate structure, `p/assess` should return the marginal LL (integrating
out conjugate priors) rather than the joint LL at a specific prior value.

**Architecture:**

`p/assess` takes `(model args choices)` where ALL trace sites must have values
in `choices`. The current handler:
1. For each site: looks up value in choices, computes log-prob
2. Returns sum of log-probs (joint LL)

With auto-handlers:
1. For conjugate prior sites: skip (marginalized out)
2. For conjugate obs sites: compute marginal LL instead of conditional LL
3. For non-conjugate sites: standard log-prob

**But:** assess requires all sites to have values. The auto-handler for priors
returns posterior mean, which may differ from the provided value. We need a
variant that:
- Uses the provided value for the prior (not posterior mean)
- Still initializes posterior state for downstream obs handlers
- Computes marginal LL at the obs values

**Implementation in `dynamic.cljs`:**

```clojure
;; In DynamicGF assess method:
(if (and (:auto-handlers schema)
         (auto-analytical/some-conjugate-obs-constrained?
           (:conjugate-pairs schema) choices))
  (let [transition (auto-analytical/make-address-dispatch
                     h/assess-transition (:auto-handlers schema))
        result (rt/run-handler transition
                 {:choices cm/EMPTY :score SCORE-ZERO
                  :weight SCORE-ZERO
                  :key key :constraints choices
                  :auto-posteriors {}
                  :auto-kalman-beliefs {}
                  :auto-kalman-noise-vars {}
                  :executor execute-sub-assess
                  :param-store (::param-store (meta this))}
                 (fn [rt] (apply body-fn rt args)))]
    {:weight (:score result)})
  ;; existing handler path
  ...)
```

**The existing obs handler already works for assess:** it checks `:constraints`
for the obs value and computes marginal LL. The prior handler returns posterior
mean without score contribution. Together they produce the marginal LL.

**Subtlety:** In assess, ALL sites are in choices. The prior handler currently
ignores the provided prior value and returns posterior mean. For assess, the
returned value doesn't matter (no trace is constructed), only the score matters.
Since the prior handler adds 0 to score (marginalized), and the obs handler
adds marginal LL, the total score is the marginal LL. This is correct.

**Files:**
- Modify: `src/genmlx/dynamic.cljs` (add auto-handler dispatch in assess)
- New: `test/genmlx/l3_5_assess_test.cljs`

**Tests (~20):**
- NN model: assess with auto-handlers → marginal LL matches exact formula
- NN model: assess without auto-handlers → joint LL (different value)
- BB model: assess marginal LL correctness
- GP model: assess marginal LL correctness
- Kalman chain: assess marginal LL via Kalman filter
- Mixed model: assess with partial conjugacy
- Assess result matches generate weight (for same observations)
- Edge: all sites are conjugate → full marginal LL
- Edge: no conjugate sites → standard joint LL (fallback)

---

### WP-2: Combinator-Aware Conjugacy Detection

**Goal:** Detect conjugate pairs where the prior is at the outer model level
and the observation is inside a combinator kernel (unfold, map, scan, etc.),
AND where conjugate pairs exist entirely within a combinator kernel.

**Why it matters:** Most real models use combinators. A time-series model is
`unfold(kernel)`. A hierarchical model uses `map(kernel)`. If the kernel has
conjugate structure, every step/element benefits from analytical elimination.

**Architecture:**

There are two cases:

**Case 1: Kernel-internal conjugacy**
The kernel itself is a `gen` function with its own schema. If the kernel has
conjugate pairs, they are already detected by L3's existing machinery. When
the combinator calls `p/generate` on the kernel, the kernel's auto-handlers
activate.

**This already works.** No new code needed. The combinator calls `p/generate`
on the kernel DynamicGF, which checks its own `:auto-handlers`. Gate 2
validates this.

**Case 2: Cross-boundary conjugacy**
A prior at the outer model level, with observations inside a combinator:
```clojure
(gen [xs]
  (let [mu (trace :mu (dist/gaussian 0 10))]  ;; prior at outer level
    (splice :obs (map-gf (gen [x]
      (trace :y (dist/gaussian mu 1)))  ;; obs inside map kernel
      xs))))
```

Here `:mu` → `:y` is NN conjugate, but the observation `:y` is inside a
`map` combinator. L3's schema extraction sees `:mu` at the outer level
and the `splice :obs` site, but doesn't see inside the splice to detect
the conjugate pair.

**This is harder.** The schema system would need to:
1. Look inside splice sites to find their kernel's schema
2. Match outer trace addresses with inner kernel dependencies
3. Build cross-boundary conjugate pairs

**Approach:** Extend schema augmentation to look through splice boundaries.
When a splice site references a combinator whose kernel has a schema, check
if any of the kernel's trace sites depend on outer-scope variables that are
also trace addresses in the outer schema.

**Implementation:**

```clojure
;; In conjugacy.cljs or a new cross_boundary.cljs

(defn detect-cross-boundary-pairs
  "Detect conjugate pairs that cross a splice boundary.
   outer-schema: the outer model's schema
   splice-sites: splice sites from the outer schema
   Returns additional conjugate pairs for the outer schema."
  [outer-schema]
  (let [splice-sites (:splice-sites outer-schema)
        outer-site-map (into {} (map (juxt :addr identity))
                              (filter :static? (:trace-sites outer-schema)))]
    ;; For each splice site, check if its GF's kernel has trace sites
    ;; that depend on outer-scope trace addresses
    ;; This requires the splice GF to have a schema (i.e., be a DynamicGF
    ;; or a combinator with a known kernel)
    ...))
```

**Challenge:** At schema extraction time (in the `gen` macro), the splice GF
may not yet be constructed. Schema extraction works on source forms, not
runtime values. So cross-boundary detection must happen at construction time
(when `make-gen-fn` is called), not at macro-expansion time.

**Practical approach for L3.5:** Rather than full cross-boundary detection
(which requires significant schema infrastructure), focus on the case where
the **kernel's generate already uses auto-handlers**. When a combinator calls
`p/generate(kernel, args, constraints)`, the kernel's auto-handlers fire
if the kernel is a DynamicGF with `:auto-handlers`. This gives us
kernel-internal conjugacy for free.

For cross-boundary conjugacy, add a **post-construction detection pass** in
`make-gen-fn` that:
1. Inspects splice site GFs (if available as runtime values)
2. Checks if they are combinators with known kernel schemas
3. Builds cross-boundary pairs by matching outer trace addrs with kernel deps

**Scope decision:** Case 1 (kernel-internal) is free. Case 2 (cross-boundary)
is the real work. Gate 2 determines which case we're in and how much work
Case 2 requires.

**Files:**
- New: `src/genmlx/cross_boundary.cljs` (cross-boundary detection)
- Modify: `src/genmlx/conjugacy.cljs` (integrate cross-boundary pairs)
- Modify: `src/genmlx/dynamic.cljs` (post-construction detection in make-gen-fn)
- New: `test/genmlx/l3_5_combinator_conjugacy_test.cljs`

**Tests (~30):**
- Map with NN kernel: `p/generate` uses auto-handlers per element
- Unfold with NN kernel: `p/generate` uses auto-handlers per step
- Scan with NN kernel: same
- Switch with conjugate branches: auto-handlers on selected branch
- Mix with conjugate components: auto-handlers on sampled component
- Nested combinator: unfold(map(NN kernel)) — two levels of auto-handlers
- Cross-boundary NN: outer prior, inner obs → detected and handled
- Cross-boundary BB: same pattern
- Outer model with mixed conjugate/non-conjugate splices
- Performance: unfold T=50 with NN kernel, measure variance reduction
- Edge: kernel without schema → fallback, no error
- Edge: combinator with non-DynamicGF kernel → fallback

**Gate 2 experiment: kernel-internal conjugacy validation.**

---

### WP-3: Multivariate Conjugacy

**Goal:** Add conjugate families for multivariate distributions:
Matrix-Normal (multivariate Gaussian prior + likelihood), and
Wishart-Normal (Wishart prior on precision, Normal likelihood).

**Why it matters:** Many real models have multivariate parameters (e.g.,
a 2D position, a feature vector). Currently L3 only handles scalar conjugacy.
Multivariate conjugacy eliminates entire vectors/matrices at once.

**Architecture:**

**Prerequisite:** GenMLX needs `dist/multivariate-gaussian` (or equivalent).
Check if this distribution exists. If not, implementing it is part of this WP.

**New conjugate families:**

| Family | Prior | Likelihood | Natural param | Update |
|--------|-------|-----------|---------------|--------|
| MVN-MVN | `mv-gaussian(mu0, Sigma0)` | `mv-gaussian(mu, Sigma_obs)` | Mean vector | Kalman-like update |
| Wishart-MVN | `wishart(nu, V)` | `mv-gaussian(0, Lambda^-1)` | Precision matrix | Wishart update |

**MVN-MVN conjugate update:**
```
Prior:      mu ~ N(m0, S0)
Likelihood: y | mu ~ N(mu, R)
Posterior:  mu | y ~ N(m1, S1)
  S1 = (S0^-1 + R^-1)^-1
  m1 = S1 * (S0^-1 * m0 + R^-1 * y)
Marginal:   y ~ N(m0, S0 + R)
  log p(y) = -0.5 * (d*log(2pi) + log|S0+R| + (y-m0)^T (S0+R)^-1 (y-m0))
```

**MLX matrix operations needed:**
- `mx/matmul` — matrix multiplication (exists)
- Matrix inverse — may need `mx/linalg-solve` or similar
- Log-determinant — `mx/linalg-slogdet` or via Cholesky
- Cholesky decomposition — `mx/linalg-cholesky` if available

**Gate 3 determines feasibility:** If MLX has the required linear algebra ops,
this WP is straightforward. If not, we either implement them or defer.

**Implementation:**

1. **Extend conjugacy table:**
```clojure
;; In conjugacy.cljs
{[:multivariate-gaussian :multivariate-gaussian]
 {:family :mvn-mvn
  :natural-param-idx 0}}
```

2. **Add update functions:**
```clojure
;; In auto_analytical.cljs
(defn mvn-update-step
  "Multivariate Normal-Normal conjugate update."
  [{:keys [mean covariance]} obs-value obs-covariance]
  (let [S0-inv (mx/linalg-inv covariance)
        R-inv (mx/linalg-inv obs-covariance)
        S1-inv (mx/add S0-inv R-inv)
        S1 (mx/linalg-inv S1-inv)
        m1 (mx/matmul S1 (mx/add (mx/matmul S0-inv mean)
                                   (mx/matmul R-inv obs-value)))
        ;; Marginal LL
        S-marginal (mx/add covariance obs-covariance)
        innov (mx/subtract obs-value mean)
        ll (mvn-log-prob innov S-marginal)]
    {:mean m1 :covariance S1 :ll ll}))
```

3. **Register handlers:**
```clojure
(defn make-auto-mvn-handlers [prior-addr obs-addrs]
  (make-conjugate-handlers prior-addr obs-addrs
    (fn [{:keys [mean covariance]}]
      {:mean mean :covariance covariance})
    :mean
    (fn [posterior obs-value {:keys [covariance]}]
      (let [{:keys [mean covariance ll]} (mvn-update-step posterior obs-value covariance)]
        {:posterior {:mean mean :covariance covariance} :ll ll}))))
```

**Files:**
- Modify: `src/genmlx/conjugacy.cljs` (add MVN-MVN to table)
- Modify: `src/genmlx/inference/auto_analytical.cljs` (add MVN update + handlers)
- Possibly new: `src/genmlx/dist.cljs` (add `multivariate-gaussian` if missing)
- New: `test/genmlx/l3_5_multivariate_test.cljs`

**Tests (~25):**
- MVN-MVN: 2D prior + 1 obs → marginal LL matches formula
- MVN-MVN: 2D prior + 5 obs → sequential update, posterior correctness
- MVN-MVN: 5D, 10D dimensions → verify scaling
- MVN-MVN: diagonal covariance → matches scalar NN (regression test)
- MVN-MVN: full covariance → correct cross-correlations in posterior
- Auto-detection in gen model → auto-handlers activate
- Generate with MVN auto-handlers → weight = marginal LL
- Regenerate with MVN auto-handlers → correct behavior
- Edge: singular covariance → graceful fallback (not analytically tractable)
- Edge: mismatched dimensions → detection rejects pair

**Gate 3 experiment: correctness and MLX matrix ops availability.**

---

### WP-4: Score Function Integration

**Goal:** Make `make-score-fn` (used by compiled MCMC) aware of conjugate
structure, so marginalized parameters are excluded from the optimization
space and marginal LL is used for scoring.

**Why it matters:** Compiled MCMC chains (`make-compiled-chain` in mcmc.cljs)
use `make-score-fn` to build a score function over latent parameters. If some
parameters are analytically marginalized, they should be excluded from the
score function's parameter vector, reducing dimensionality and improving
MCMC mixing.

**Architecture:**

Current `make-score-fn`:
```
latent addresses → flatten to [K]-vector → p/generate(model, args, cm) → weight
```

With conjugate awareness:
```
non-conjugate latent addresses → flatten to [K']-vector (K' < K)
→ p/generate(model, args, cm) → weight (includes marginal LL from auto-handlers)
```

The auto-handlers in `p/generate` already handle the analytical part. The only
change is: exclude conjugate prior addresses from the parameter vector that
MCMC proposals operate on.

**Implementation in `inference/util.cljs`:**

```clojure
(defn make-conjugate-aware-score-fn
  "Build a score function that excludes analytically marginalized parameters.
   Returns {:score-fn (fn [params] -> scalar), :addr-index {addr -> idx},
            :initial-params [K'] vector}."
  [model args observations]
  (let [schema (:schema model)
        eliminated (when schema
                     (get-in schema [:analytical-plan :rewrite-result :eliminated]))
        ;; Filter out eliminated addresses
        all-addrs (p/get-trace-addresses model args)
        latent-addrs (if eliminated
                       (remove eliminated all-addrs)
                       all-addrs)]
    (make-score-fn model args observations latent-addrs)))
```

**Integration with compiled MCMC:**
- `make-compiled-chain` already takes addresses as parameter
- Pass filtered addresses (excluding conjugate priors)
- The compiled chain operates in lower dimension
- `p/generate` inside the chain automatically invokes auto-handlers

**Files:**
- Modify: `src/genmlx/inference/util.cljs` (add conjugate-aware score fn)
- Modify: `src/genmlx/inference/mcmc.cljs` (use conjugate-aware score fn when available)
- New: `test/genmlx/l3_5_score_fn_test.cljs`

**Tests (~15):**
- NN model: score fn excludes marginalized prior
- NN model: compiled MH chain in reduced dimension → posterior correct
- Mixed model: 3 NN priors eliminated, 2 params remain → correct dim
- Score fn output matches p/generate weight
- Fallback: non-conjugate model → standard score fn (unchanged)
- Performance: compare MH in K vs K' dimensions → mixing improvement

**Gate 4 experiment: end-to-end MCMC + analytical benchmark.**

---

## Dependency Graph

```
WP-0 (Regenerate integration)     [independent]
  │
  └── WP-4 (Score function)       [uses regenerate for validation]

WP-1 (Assess integration)         [independent]

WP-2 (Combinator conjugacy)       [independent, Gate 2 first]

WP-3 (Multivariate conjugacy)     [independent, Gate 3 first]
```

All WPs are largely independent. WP-0 and WP-1 are low-risk extensions of
existing patterns. WP-2 requires investigation (Gate 2). WP-3 requires
MLX capability check (Gate 3). WP-4 builds on WP-0's regenerate support.

## Recommended Execution Order

1. **Gate 2** first — determines WP-2 scope (quick experiment, ~30 min)
2. **WP-0** → Gate 0, Gate 1 (regenerate — highest ROI, enables WP-4)
3. **WP-1** (assess — straightforward, parallel with WP-0)
4. **Gate 3** — determines WP-3 feasibility (check MLX matrix ops)
5. **WP-2** → full combinator support (scope determined by Gate 2)
6. **WP-3** → multivariate families (feasibility determined by Gate 3)
7. **WP-4** → Gate 4 (score function + end-to-end benchmark — capstone)

**Stop points:** Each WP delivers independent value:
- After WP-0: MH/Gibbs chains benefit from conjugacy
- After WP-1: Model comparison via assess uses marginal LL
- After WP-2: Combinator models get analytical elimination
- After WP-3: Multivariate models get analytical elimination
- After WP-4: Compiled MCMC operates in reduced dimension

---

## Estimated Scope

| WP | New/modified code | Tests | Difficulty | Gates |
|----|-------------------|-------|------------|-------|
| WP-0: Regenerate integration | ~120 lines | ~35 | Medium | Gate 0, 1 |
| WP-1: Assess integration | ~40 lines | ~20 | Easy | — |
| WP-2: Combinator conjugacy | ~180 lines | ~30 | Medium-Hard | Gate 2 |
| WP-3: Multivariate conjugacy | ~200 lines | ~25 | Hard | Gate 3 |
| WP-4: Score function integration | ~80 lines | ~15 | Medium | Gate 4 |
| **Total** | **~620 lines** | **~125** | | |

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Regenerate weight accounting wrong | Medium | High | Gate 0 validates before full implementation |
| Address dispatch overhead in MCMC hot loop | Low | Medium | Gate 1 measures; if too slow, cache dispatch lookup |
| Kernel-internal conjugacy doesn't work through combinator calls | Low | Medium | Gate 2 checks; if fails, add explicit forwarding |
| Cross-boundary detection too complex for L3.5 | Medium | Low | Defer to L4; kernel-internal still works |
| MLX lacks matrix inverse/cholesky | Medium | Medium | Gate 3 checks; if missing, defer multivariate |
| Multivariate update numerically unstable | Low | Medium | Use Cholesky parameterization; condition number checks |
| Score fn dimension reduction doesn't improve mixing | Low | Low | MCMC still works correctly, just no improvement |
| Regenerate + Kalman cascade interaction | Medium | Medium | Test thoroughly; Kalman state in regenerate is new territory |

---

## Files to Create/Modify

| File | Action | WP |
|------|--------|-----|
| `src/genmlx/dynamic.cljs` | Modify: add auto-handler dispatch in regenerate + assess | WP-0, WP-1 |
| `src/genmlx/inference/auto_analytical.cljs` | Modify: add regenerate-aware handlers | WP-0 |
| `src/genmlx/cross_boundary.cljs` | New: cross-boundary conjugacy detection | WP-2 |
| `src/genmlx/conjugacy.cljs` | Modify: integrate cross-boundary pairs, add MVN family | WP-2, WP-3 |
| `src/genmlx/inference/auto_analytical.cljs` | Modify: add MVN update + handlers | WP-3 |
| `src/genmlx/inference/util.cljs` | Modify: conjugate-aware score fn | WP-4 |
| `src/genmlx/inference/mcmc.cljs` | Modify: use conjugate-aware score fn | WP-4 |
| `test/genmlx/l3_5_regenerate_test.cljs` | New | WP-0 |
| `test/genmlx/l3_5_assess_test.cljs` | New | WP-1 |
| `test/genmlx/l3_5_combinator_conjugacy_test.cljs` | New | WP-2 |
| `test/genmlx/l3_5_multivariate_test.cljs` | New | WP-3 |
| `test/genmlx/l3_5_score_fn_test.cljs` | New | WP-4 |
| `test/genmlx/l3_5_gate_test.cljs` | New | Gates |
| `test/genmlx/l3_5_evaluation_benchmark.cljs` | New | Final evaluation |

---

## What "Level 3.5 Done" Means

1. **Every GFI operation** (generate, regenerate, assess) benefits from
   analytical elimination when conjugate structure is detected.
2. **MCMC chains** (MH, Gibbs, MALA, HMC, NUTS) automatically operate in
   reduced dimension, excluding analytically marginalized parameters.
3. **Combinator kernels** with conjugate structure get per-step/per-element
   analytical elimination.
4. **Multivariate models** with matrix-valued conjugate priors get analytical
   elimination (if MLX matrix ops are available).
5. **All 5 existing conjugate families** work across all operations.
6. **Handler remains ground truth.** Auto-handlers produce identical
   traces and scores. Analytical elimination is optimization, not behavior change.
7. **Fallback is always safe.** Non-conjugate models, non-static models,
   and unconstrained observations all fall through to standard handlers
   with zero overhead.

---

## Relationship to Level 4

Level 3.5 provides the coverage that Level 4's automatic method selection
(L4-M3) needs. L4-M3 asks: "given a model and data, choose the inference
strategy." To choose correctly, the system must know what's analytically
tractable — not just at the top level, but inside combinators and across
all GFI operations. L3.5 provides this.

Specifically:
- L4-M3 needs conjugacy metadata on combinators → WP-2 provides this
- L4-M3 needs to know MCMC can use analytical elimination → WP-0 provides this
- L4-M3 needs multivariate tractability info → WP-3 provides this
- L4's fused optimization loop needs reduced-dimension score fns → WP-4 provides this

Without L3.5, L4 would have to build all of this from scratch. With L3.5,
L4 can focus purely on graph fusion and compilation.

---

## Evaluation: How We Measure Benefit Over Level 3

### The measurement challenge

L3 had a clean story: **exact marginal LL vs IS collapse**. One number
(variance reduction) told the whole story. L3.5 is harder because the
benefit is **coverage expansion** — making analytical elimination reach
more of the inference stack. We need metrics that capture this.

### Headline metric: Operation Coverage Rate

**Definition:** For a conjugate model, what fraction of GFI operation
invocations during a full inference run use analytical elimination?

```
Coverage = (analytically-handled invocations) / (total GFI invocations)
```

| | L3 | L3.5 (target) |
|---|---|---|
| IS (N=200, T=1) | 100% (generate only) | 100% (same) |
| MH (5000 steps) | 0.02% (1 generate init / 5001 total) | ~100% (every regenerate step) |
| Gibbs (5000 steps, K blocks) | 0% (no regenerate support) | ~100% per block |
| SMC (N=100, T=50) | 100% of generate calls | 100% + assess if used |
| Unfold model + IS | 0% (combinator boundary) | ~100% per step |

This is the **primary metric**: L3 covers IS well but barely touches MCMC;
L3.5 makes coverage near-universal for conjugate models.

### Benchmark scenarios (in `l3_5_evaluation_benchmark.cljs`)

#### Scenario 1: MCMC on Normal-Normal model

**Model:** `mu ~ N(0,10), y_i ~ N(mu,1)` with 10 observations.

**Comparison:**
| Method | Level | What it measures |
|--------|-------|-----------------|
| MH chain (5000 steps) | L2 (no auto-handlers) | Baseline MCMC |
| MH chain (5000 steps) | L3.5 (auto-handlers in regenerate) | MCMC + analytical |
| IS (200 particles) | L3 (auto-handlers in generate) | Reference from L3 |

**Metrics:**
- **ESS/second** — efficiency of posterior sampling
- **Posterior accuracy** — mean/variance vs exact analytical posterior
- **Acceptance rate** — should improve if analytical elimination helps proposals
- **Effective dimension** — K (full) vs K' (reduced) parameters in proposal

**Expected result:** L3.5 MH should have higher ESS/second than L2 MH because
the conjugate prior is marginalized out. For a 1-param model this is trivial
(the posterior IS the analytical solution), so also test the mixed model.

#### Scenario 2: Mixed model MCMC (the L3 benchmark model)

**Model:** 3 independent NN groups (mu_i ~ N(0,10), y_ij ~ N(mu_i, 1))
+ 2 non-conjugate parameters. Same model from L3 benchmarks.

**Comparison:**
| Method | Level | Dims sampled |
|--------|-------|-------------|
| IS (200 particles) | L2 | 5 (all) |
| IS (200 particles) | L3 | 2 (3 eliminated) |
| MH chain (5000 steps) | L2 | 5 (all) |
| MH chain (5000 steps) | L3.5 | 2 (3 eliminated) |
| Gibbs (5000 steps, 2 blocks) | L3.5 | 2 (3 eliminated, 2 sampled per block) |

**Metrics:**
- **ESS/second** for each method
- **log-ML std** across 15 trials (comparable to L3's 33.5x variance reduction)
- **Mixing per dimension** — trace plots and R-hat for each non-conjugate param
- **Wall-clock time** — total time for equivalent posterior quality

**Expected result:** L3.5 MH in 2D should mix much faster than L2 MH in 5D.
L3.5 Gibbs with analytical blocks should be best. L3 IS remains competitive
for this model size but MCMC scales better to more parameters.

**Headline number:** Compare L3.5 MCMC ESS/second vs L3 IS ESS/second.
If L3.5 MCMC wins, that's the proof that extending to regenerate matters.

#### Scenario 3: Unfold model with conjugate kernel

**Model:**
```clojure
(def kernel (gen [t state]
  (let [z (trace :z (dist/gaussian (mx/multiply 0.9 state) 1.0))]
    z)))
(def model (gen [T]
  (let [sigma (trace :sigma (dist/gamma 2 1))]
    (splice :steps (unfold kernel T (mx/scalar 0.0))))))
```
With observations on `:z` at each step (T=20, 50).

**Comparison:**
| Method | Level | Per-step elimination? |
|--------|-------|-----------------------|
| IS (200 particles) | L2 | No |
| IS (200 particles) | L3 | No (combinator boundary blocks) |
| IS (200 particles) | L3.5 | Yes (kernel auto-handlers fire) |
| Compiled SMC (200 particles) | L2 | No |

**Metrics:**
- **log-ML accuracy** vs Kalman filter ground truth
- **ESS** at each particle count (50, 100, 200, 500)
- **Variance of log-ML** across 15 trials

**Expected result:** L3 and L2 should be identical (L3 can't see inside unfold).
L3.5 should show the same kind of improvement L3 showed for top-level models,
now applied per-step inside the combinator.

**Headline number:** L3.5 unfold IS variance vs L3 unfold IS variance.
If L3.5 shows variance reduction, combinator-aware conjugacy works.

#### Scenario 4: Multivariate Normal-Normal (dimension scaling)

**Model:** `mu ~ N(0, 100*I_d), y_i ~ N(mu, I_d)` for d = 1, 2, 5, 10.
With 10 observations.

**Comparison:**
| Method | d=1 | d=2 | d=5 | d=10 |
|--------|-----|-----|-----|------|
| IS (200 particles), L2 | ~OK | degrading | collapsed | collapsed |
| IS (200 particles), L3 | exact | N/A | N/A | N/A |
| IS (200 particles), L3.5 | exact | exact | exact | exact |

**Metrics:**
- **log-ML** vs closed-form (exact for all d)
- **ESS** at each dimension
- **IS collapse dimension** — at what d does L2 ESS < 5?

**Expected result:** L2 IS collapses exponentially in d (curse of dimensionality).
L3 only handles d=1. L3.5 with multivariate conjugacy handles all d exactly.

**Headline number:** "L3.5 maintains exact marginal LL at d=10 where L2 IS
collapses to ESS < 1" — demonstrates that multivariate conjugacy breaks
the exponential dimension barrier.

### Summary scorecard

After all benchmarks, produce a summary table:

```
Level 3.5 Evaluation Summary
═══════════════════════════════════════════════════════════════
                              L2        L3        L3.5
───────────────────────────────────────────────────────────────
Operation coverage (MH 5K)    0%        0.02%     ~100%
Operation coverage (IS 200)   0%        100%      100%
Mixed model ESS (IS)          1.1       7.7       7.7
Mixed model ESS (MH)          ???       N/A       ???    ← new
Unfold log-ML std (IS)        high      high      ???    ← new
MVN d=5 log-ML                collapsed exact(1D) exact  ← new
MCMC dimensions sampled       K         K         K'<K
───────────────────────────────────────────────────────────────
```

The **three new rows** (MH ESS, unfold variance, MVN scaling) are the
L3.5-specific improvements. The existing rows (IS coverage, mixed model IS)
should be unchanged — L3.5 doesn't regress L3.

### What "L3.5 is worth it" looks like

1. **MCMC benefit is real:** L3.5 MH ESS/second > L2 MH ESS/second on
   conjugate models (Scenario 1 + 2)
2. **Combinator benefit is real:** L3.5 unfold IS variance < L3 unfold IS
   variance (Scenario 3)
3. **Multivariate benefit is real:** L3.5 handles d > 1 where L2 collapses
   (Scenario 4)
4. **No regression:** All L3 benchmarks produce identical results at L3.5

### What "L3.5 is NOT worth it" looks like

- MCMC overhead from address dispatch eats the analytical benefit (Gate 1 catches this)
- Combinator kernels already get auto-handlers for free and WP-2 adds nothing new
- MLX lacks matrix ops and multivariate is deferred
- Only the score-function dimension reduction (WP-4) shows measurable improvement

In this case, L3.5 would be a minor release (WP-0 + WP-1 only) and we'd
proceed directly to L4.
