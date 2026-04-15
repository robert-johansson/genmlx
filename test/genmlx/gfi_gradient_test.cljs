(ns genmlx.gfi-gradient-test
  "Property-based gradient correctness: AD vs symmetric finite differences.

   Verifies that choice-gradients (autodiff through the GFI generate interface)
   matches symmetric central differences for randomly generated differentiable
   models. This is a fundamental correctness check: the AD system must agree
   with the limit definition of the derivative.

   --- Tolerance derivation ---

   Central finite difference with step h on function f:
     FD(h) = [f(x+h) - f(x-h)] / (2h)

   Error sources:
     Truncation error: |FD(h) - f'(x)| = h^2/6 * |f'''(xi)| for some xi
     Float32 rounding: each f evaluation has error ~ eps_mach * |f(x)|
       where eps_mach ~ 1.2e-7 (float32 machine epsilon)
     Total rounding in FD: ~ 2 * eps_mach * |f(x)| / (2h) = eps_mach * |f| / h

   For h = 1e-3, eps_mach = 1.2e-7:
     Truncation: ~ 1e-6 * |f'''| (negligible for smooth log-densities)
     Rounding:   ~ 1.2e-4 * |f| (dominant term)

   AD error is O(N * eps_mach) where N = number of ops in the computation graph.
   For models with 1-6 sites, N ~ 10-50 ops, so AD error ~ 1e-5.

   Combined tolerance: 1e-2 is conservative, accounting for:
   - Cauchy distributions with large curvature near the mode
   - Chain models where perturbation propagates through dependencies
   - Float32 accumulation across multiple trace sites"
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.gradients :as grad]
            [genmlx.gfi-compiler :as compiler]
            [genmlx.gfi-gen :as model-gen]
            [genmlx.test-helpers :as h])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Constants
;; ---------------------------------------------------------------------------

(def ^:private FD-EPS
  "Finite difference step size.

   Optimal h for central FD in float32:
     h_opt = (3 * eps_mach)^(1/3) ~ (3.6e-7)^(1/3) ~ 7e-3

   We use 1e-3, slightly below optimal, which favors lower truncation error
   at the cost of slightly higher rounding error. This is appropriate because
   our log-density functions (gaussian, laplace, cauchy) have bounded third
   derivatives in the regions where samples typically land."
  1e-3)

(def ^:private GRAD-ABS-TOL
  "Absolute tolerance for AD-vs-FD comparison.

   From the derivation in the ns docstring:
   - FD truncation error: ~ h^2 * |f'''| ~ 1e-6 for smooth densities
   - FD rounding error: ~ eps_mach * |score| / h ~ 1.2e-4 * |score|
   - AD error: ~ N_ops * eps_mach ~ 1e-5

   For typical scores |score| < 20 and N_sites <= 6:
     Total error < 1e-6 + 2.4e-3 + 1e-5 ~ 3e-3

   We use 5e-2 to account for edge cases:
   - Cauchy near mode with small scale: |f'''(0)| = 8/gamma^3,
     so FD truncation ~ h^2/6 * 8/gamma^3. For gamma=0.1, ~ 0.013.
   - Laplace cusp: |c-i|/scale has discontinuous f'' at c=i.
     When FD straddles the cusp, error depends on |c-i|/h.
   - Chain dependency amplification through multiple sites."
  5e-2)

(def ^:private GRAD-REL-TOL
  "Relative tolerance for AD-vs-FD comparison.

   When both gradients are large, relative comparison is more appropriate
   than absolute. The relative FD error for smooth functions is:
     |FD - f'| / |f'| ~ h^2/6 * |f'''|/|f'| + eps_mach / (h * |f'|)
   For h = 1e-3 and typical |f'''|/|f'| ratios, this is ~ 1e-3.
   We use 5e-2 (50x margin) for robustness across distribution families."
  5e-2)

;; ---------------------------------------------------------------------------
;; Generators — differentiable models only
;; ---------------------------------------------------------------------------

(def ^:private gen-diff-dist
  "Generator for differentiable distribution types.

   Differentiable distributions in GenMLX (those with reparam clauses in defdist):
   - :gaussian — reparam via location-scale: mu + sigma * N(0,1)
   - :laplace  — reparam via inverse CDF: loc + scale * sign(u-0.5) * log(1-2|u-0.5|)
   - :cauchy   — reparam via inverse CDF: loc + scale * tan(pi * (u - 0.5))

   Excluded:
   - :uniform     — has reparam but boundary discontinuity in log-prob
                     (mx/where creates non-differentiable points)
   - :exponential — has reparam but support boundary at 0 creates issues
   - :bernoulli   — discrete, not differentiable
   - :delta       — zero gradient everywhere (degenerate)"
  (gen/elements [:gaussian :laplace :cauchy]))

(def gen-differentiable-spec
  "Model spec where all sites use differentiable distributions.
   Built by construction (not filtered) so shrinking never fails."
  (gen/bind (gen/choose 1 6)
    (fn [n]
      (gen/bind (gen/vector gen-diff-dist n)
        (fn [dist-types]
          (gen/fmap
            (fn [{:keys [sites]}]
              {:sites sites :args [] :return (:addr (peek sites))})
            (model-gen/gen-sites dist-types)))))))

;; ---------------------------------------------------------------------------
;; Finite difference computation
;; ---------------------------------------------------------------------------

(defn- fd-gradient
  "Compute the symmetric central finite difference gradient of the model's
   joint log-density with respect to the choice at `addr`.

   Mathematical definition:
     FD(addr) = [log p(choices[addr := v+eps]) - log p(choices[addr := v-eps])] / (2*eps)

   where p is the joint density over all choices given args, and
   choices[addr := x] denotes the choicemap with addr replaced by x.

   We use p/assess to evaluate the joint log-density at perturbed points.
   assess(model, args, choices).weight = log p(choices; args)."
  [gf args choices addr eps]
  (let [v (h/realize (cm/get-value (cm/get-submap choices addr)))
        choices+ (cm/set-value choices addr (mx/scalar (+ v eps)))
        choices- (cm/set-value choices addr (mx/scalar (- v eps)))
        score+ (h/realize (:weight (p/assess gf args choices+)))
        score- (h/realize (:weight (p/assess gf args choices-)))]
    (/ (- score+ score-) (* 2.0 eps))))

(defn- near-laplace-cusp?
  "True if the value at `addr` is within 2*eps of a Laplace location parameter.

   The Laplace log-density log Lap(v; loc, scale) = -log(2s) - |v-loc|/s
   has a cusp (non-differentiable point) at v = loc. Central FD with step eps
   gives unreliable results when |v - loc| < 2*eps because the perturbation
   straddles the absolute value cusp.

   Additionally, if a Laplace site depends on an upstream site at `addr`,
   then perturbing `addr` changes the Laplace location parameter, which can
   also cause FD to straddle the cusp.

   This function checks both direct Laplace sites at `addr` and downstream
   Laplace sites whose location depends on `addr`."
  [spec choices addr eps]
  (let [v (h/realize (cm/get-value (cm/get-submap choices addr)))
        sites (:sites spec)
        site-map (into {} (map (fn [s] [(:addr s) s]) sites))]
    (or
      ;; Direct: this site is Laplace and its value is near its own loc
      (let [site (get site-map addr)]
        (when (= :laplace (:dist site))
          (let [loc-arg (first (:args site))
                loc-val (if (keyword? loc-arg)
                          (h/realize (cm/get-value (cm/get-submap choices loc-arg)))
                          loc-arg)]
            (< (js/Math.abs (- v loc-val)) (* 2.0 eps)))))
      ;; Indirect: a downstream Laplace site depends on addr as its loc,
      ;; and its value is near the current value of addr
      (some (fn [site]
              (when (and (= :laplace (:dist site))
                         (= addr (first (:args site))))
                (let [site-v (h/realize (cm/get-value (cm/get-submap choices (:addr site))))]
                  (< (js/Math.abs (- site-v v)) (* 2.0 eps)))))
            sites))))

(defn- gradients-match?
  "Compare AD and FD gradients with combined absolute/relative tolerance.

   Three comparison modes:
   1. Both near zero (|ad| < 1e-6 and |fd| < 1e-6): pass
      Rationale: when the true gradient is zero, both methods produce
      values dominated by numerical noise. Comparing noise is meaningless.

   2. Absolute check: |ad - fd| < GRAD-ABS-TOL
      Appropriate when gradients are O(1) or smaller.

   3. Relative check: |ad - fd| / max(|ad|, |fd|) < GRAD-REL-TOL
      Appropriate when gradients are large (e.g., Cauchy near mode).
      The FD relative error is bounded by h^2 * |f'''|/|f'| for smooth f.

   Passes if either the absolute or relative check succeeds."
  [ad-g fd-g]
  (let [abs-ad (js/Math.abs ad-g)
        abs-fd (js/Math.abs fd-g)
        abs-diff (js/Math.abs (- ad-g fd-g))
        max-mag (js/Math.max abs-ad abs-fd)]
    (or (and (< abs-ad 1e-6) (< abs-fd 1e-6))
        (< abs-diff GRAD-ABS-TOL)
        (and (> max-mag 1e-6)
             (< (/ abs-diff max-mag) GRAD-REL-TOL)))))

;; ---------------------------------------------------------------------------
;; Property 1: AD matches FD for all sites in random differentiable models
;; ---------------------------------------------------------------------------

(defspec law:gradient-matches-finite-diff 30
  ;; For any differentiable model drawn from gen-differentiable-spec:
  ;;   For all trace sites addr_i:
  ;;     |AD_gradient(addr_i) - FD_gradient(addr_i)| < GRAD-TOL
  ;;
  ;; This is a universal property: it must hold for every model structure
  ;; (1-6 sites, any dependency pattern among gaussian/laplace/cauchy sites)
  ;; and every sampled trace (any point in the support).
  ;;
  ;; The AD gradient is computed by choice-gradients, which:
  ;; 1. Packs all target choice values into a single parameter array
  ;; 2. Defines score(params) = generate(model, args, choicemap(params)).weight
  ;; 3. Applies mx/grad to score
  ;;
  ;; The FD gradient perturbs one choice at a time and uses p/assess to
  ;; evaluate the joint log-density at perturbed points.
  ;;
  ;; Both should compute the same quantity: d(log p(tau; x)) / d(tau_i).
  (prop/for-all [spec gen-differentiable-spec]
    (let [gf (compiler/spec->gf spec)
          trace (p/simulate gf [])
          addrs (mapv :addr (:sites spec))
          choices (:choices trace)]
      (try
        (let [ad-grads (grad/choice-gradients gf trace addrs)]
          (every?
            (fn [addr]
              ;; Skip FD check at Laplace cusp points where FD is unreliable.
              ;; The Laplace |v-loc|/scale term has a non-differentiable cusp at
              ;; v = loc. Central FD straddling this point gives O(1) error.
              ;; AD correctly computes the subgradient, but FD cannot match it.
              (if (near-laplace-cusp? spec choices addr FD-EPS)
                true
                (let [ad-g (h/realize (get ad-grads addr))
                      fd-g (fd-gradient gf [] choices addr FD-EPS)
                      ok? (gradients-match? ad-g fd-g)]
                  (when-not ok?
                    (println "\n  GRADIENT MISMATCH at" addr
                             "in model" (pr-str (mapv :dist (:sites spec)))
                             "\n    AD:" ad-g "FD:" fd-g
                             "diff:" (js/Math.abs (- ad-g fd-g))))
                  ok?)))
            addrs))
        (catch :default e
          (println "\n  ERROR for spec:" (pr-str spec) "\n   " (str e))
          false)))))

;; ---------------------------------------------------------------------------
;; Property 2: gradient through dependency chain
;;
;; For models with dependent sites (a -> b -> c), perturbing an upstream
;; choice affects the log-density of all downstream sites. The AD gradient
;; must account for this chain rule effect.
;;
;; Specifically, for a chain a ~ N(0,1), b ~ N(a,1):
;;   score = log N(a;0,1) + log N(b;a,1)
;;         = -0.5*log(2pi) - 0.5*a^2 - 0.5*log(2pi) - 0.5*(b-a)^2
;;
;;   d(score)/d(a) = -a + (b-a) = b - 2a
;;   d(score)/d(b) = -(b-a) = a - b
;;
;; The FD approach naturally captures this because it perturbs the value
;; in the choicemap and re-evaluates the full joint. The AD approach must
;; correctly propagate gradients through the generate interface.
;;
;; We generate random chain-like models (all sites differentiable, some
;; with keyword dependencies on earlier sites) and verify FD matches AD.
;; ---------------------------------------------------------------------------

(def gen-chain-spec
  "Model spec with 2-5 differentiable sites, at least one dependency.
   Built by construction: generates differentiable specs with >= 2 sites,
   then filters for at least one dependency. Since gen-site-args uses
   keyword dep-refs ~70% of the time when available, a 2+ site model
   almost always has deps. The such-that has high hit rate (~95%)."
  (gen/such-that
    (fn [spec]
      (some (fn [site] (seq (:deps site))) (:sites spec)))
    (gen/bind (gen/choose 2 5)
      (fn [n]
        (gen/bind (gen/vector gen-diff-dist n)
          (fn [dist-types]
            (gen/fmap
              (fn [{:keys [sites]}]
                {:sites sites :args [] :return (:addr (peek sites))})
              (model-gen/gen-sites dist-types))))))
    100))

(defspec law:gradient-through-chain 30
  ;; For chain models where at least one site depends on a previous site:
  ;;   AD gradients must still match FD gradients.
  ;;
  ;; This specifically tests that the backward pass through generate correctly
  ;; handles the case where changing choice A affects the log-prob of choice B
  ;; (because B's distribution parameters depend on A's value).
  ;;
  ;; The chain rule for the joint:
  ;;   d/d(tau_i) [sum_j log p(tau_j | parents_j)]
  ;;   = d/d(tau_i) log p(tau_i | parents_i)              [direct term]
  ;;     + sum_{j: i in parents_j} d/d(tau_i) log p(tau_j | parents_j) [indirect terms]
  ;;
  ;; FD captures both terms automatically. AD must propagate through the
  ;; distribution parameter computation in the generate call.
  (prop/for-all [spec gen-chain-spec]
    (let [gf (compiler/spec->gf spec)
          trace (p/simulate gf [])
          addrs (mapv :addr (:sites spec))
          choices (:choices trace)]
      (try
        (let [ad-grads (grad/choice-gradients gf trace addrs)]
          (every?
            (fn [addr]
              (if (near-laplace-cusp? spec choices addr FD-EPS)
                true
                (let [ad-g (h/realize (get ad-grads addr))
                      fd-g (fd-gradient gf [] choices addr FD-EPS)
                      ok? (gradients-match? ad-g fd-g)]
                  (when-not ok?
                    (println "\n  CHAIN GRADIENT MISMATCH at" addr
                             "in model" (pr-str (:sites spec))
                             "\n    AD:" ad-g "FD:" fd-g
                             "diff:" (js/Math.abs (- ad-g fd-g))))
                  ok?)))
            addrs))
        (catch :default e
          (println "\n  CHAIN ERROR for spec:" (pr-str spec) "\n   " (str e))
          false)))))

;; ---------------------------------------------------------------------------
;; Property 3: gradient is antisymmetric under value reflection for gaussian
;;
;; For a single site x ~ N(mu, sigma):
;;   d/dx log N(x; mu, sigma) = -(x - mu) / sigma^2
;;
;; This gradient is an odd function of (x - mu):
;;   grad(mu + delta) = -grad(mu - delta)
;;
;; We verify this algebraic identity holds for the AD gradient,
;; which is a stronger check than FD comparison (no FD truncation error).
;;
;; Tolerance: relative 1e-5 of the gradient magnitude.
;;
;; The identity g(mu+d) + g(mu-d) = 0 is exact in exact arithmetic.
;; In float32, each gradient has error ~ N_ops * eps_mach * |g|
;; where N_ops ~ 10 (gaussian log-prob backward pass), eps_mach ~ 1.2e-7.
;; The sum |g1 + g2| / max(|g1|, |g2|) ~ 2 * N_ops * eps_mach ~ 2.4e-6.
;;
;; When sigma is small (0.1), gradients scale as O(1/sigma^2), reaching
;; ~100-8000 for typical samples. Absolute error |g1+g2| can be up to
;; ~2e-3, but relative error stays bounded by float32 precision.
;;
;; We use max(1e-5, 1e-4 * max(|g1|, |g2|)) to handle both:
;; - Small gradients: absolute 1e-5 catches float32 noise floor
;; - Large gradients: relative 1e-4 (42x margin over theoretical 2.4e-6)
;; ---------------------------------------------------------------------------

(def gen-gaussian-only-spec
  "Model spec with exactly one gaussian site, no dependencies.
   Built by construction: always generates a single-site gaussian model."
  (gen/fmap
    (fn [{:keys [sites]}]
      {:sites sites :args [] :return (:addr (first sites))})
    (model-gen/gen-sites [:gaussian])))

(defspec law:gaussian-gradient-antisymmetry 30
  ;; For x ~ N(mu, sigma):
  ;;   d/dx log N(x; mu, sigma) = -(x - mu) / sigma^2
  ;;
  ;; This is an odd function of (x - mu):
  ;;   grad(mu + d) = -d/sigma^2
  ;;   grad(mu - d) = +d/sigma^2
  ;;   => grad(mu + d) + grad(mu - d) = 0
  ;;
  ;; We simulate, compute the AD gradient, then reflect the value about mu
  ;; and verify the gradients sum to zero (up to float32 precision).
  (prop/for-all [spec gen-gaussian-only-spec]
    (let [site (first (:sites spec))
          addr (:addr site)
          ;; Extract distribution parameters from the spec
          [mu-arg sigma-arg] (:args site)
          mu (if (number? mu-arg) mu-arg 0.0)
          sigma (if (number? sigma-arg) sigma-arg 1.0)
          gf (compiler/spec->gf spec)]
      (try
        (let [;; Simulate and get value
              trace (p/simulate gf [])
              v (h/realize (cm/get-value (cm/get-submap (:choices trace) addr)))
              ;; Reflect about mu: v' = 2*mu - v, so (v' - mu) = -(v - mu)
              v-reflected (- (* 2.0 mu) v)
              ;; Gradient at original
              g1 (h/realize (get (grad/choice-gradients gf trace [addr]) addr))
              ;; Gradient at reflected point: build new trace via generate
              reflected-choices (cm/set-value (:choices trace) addr (mx/scalar v-reflected))
              gen-result (p/generate gf [] reflected-choices)
              trace2 (:trace gen-result)
              g2 (h/realize (get (grad/choice-gradients gf trace2 [addr]) addr))
              ;; Sum should be zero (antisymmetry)
              sum (+ g1 g2)
              max-mag (js/Math.max (js/Math.abs g1) (js/Math.abs g2))
              ;; Tolerance: max of absolute floor and relative bound
              ;; See derivation above the generator definition.
              tol (js/Math.max 1e-5 (* 1e-4 max-mag))]
          (when-not (< (js/Math.abs sum) tol)
            (println "\n  ANTISYMMETRY FAIL:" addr
                     "v=" v "v'=" v-reflected
                     "g1=" g1 "g2=" g2 "sum=" sum "tol=" tol))
          (< (js/Math.abs sum) tol))
        (catch :default e
          (println "\n  ANTISYMMETRY ERROR:" (str e))
          false)))))

;; ---------------------------------------------------------------------------
;; Runner
;; ---------------------------------------------------------------------------

(defn -main []
  (t/run-tests 'genmlx.gfi-gradient-test))

(-main)
