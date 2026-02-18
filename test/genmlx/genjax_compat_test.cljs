(ns genmlx.genjax-compat-test
  "Tests adapted from GenJAX (https://github.com/femtomc/genjax/tree/main/tests).
   Verifies GFI invariants, inference convergence, gradient flow, combinators,
   and numerical stability."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.diagnostics :as diag])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- check [label ok?]
  (if ok?
    (do (vswap! pass-count inc)
        (println (str "  PASS: " label)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " label)))))

(defn- approx= [a b tol]
  (< (js/Math.abs (- a b)) tol))

(defn- finite? [x]
  (and (not (js/isNaN x)) (js/isFinite x)))

(defn- ev [x] (mx/eval! x) (mx/item x))

;; ---------------------------------------------------------------------------
;; Section 1: GFI Invariants (adapted from test_core.py)
;; ---------------------------------------------------------------------------

(println "\n=== Section 1: GFI Invariants ===")

;; -- 1.1: Simulate/Generate consistency --
;; Generating with ALL choices from a simulate trace should give weight == score
(println "\n-- 1.1 Simulate/Generate consistency --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                [x y]))
      trace (p/simulate model [])
      choices (:choices trace)
      score (ev (:score trace))
      ;; Generate with exactly the same choices
      {:keys [trace weight]} (p/generate model [] choices)
      gen-score (ev (:score trace))
      gen-weight (ev weight)]
  (check "generate with all choices: weight ≈ score"
         (approx= gen-weight gen-score 1e-5))
  (check "generate score equals original score"
         (approx= gen-score score 1e-5)))

;; With a more complex model
(let [model (gen [n]
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (mx/eval! mu)
                (let [m (mx/item mu)]
                  (doseq [i (range n)]
                    (dyn/trace (keyword (str "y" i)) (dist/gaussian m 1)))
                  m)))
      trace (p/simulate model [5])
      choices (:choices trace)
      score (ev (:score trace))
      {:keys [weight]} (p/generate model [5] choices)
      w (ev weight)]
  (check "hierarchical model: generate weight ≈ score"
         (approx= w score 1e-4)))

;; -- 1.2: Generate with empty constraints == simulate (weight ≈ 0) --
(println "\n-- 1.2 Generate with empty constraints --")

(let [model (gen [] (dyn/trace :x (dist/gaussian 0 1)))
      {:keys [weight]} (p/generate model [] cm/EMPTY)
      w (ev weight)]
  (check "empty constraints: weight ≈ 0" (approx= w 0.0 1e-10)))

;; -- 1.3: Update weight invariant --
;; For update: weight = new_trace.score - old_trace.score (for fully changed choices)
(println "\n-- 1.3 Update weight invariant --")

(let [model (gen [] (dyn/trace :x (dist/gaussian 0 1)))
      trace (p/simulate model [])
      old-score (ev (:score trace))
      new-val (mx/scalar 2.5)
      constraints (cm/choicemap :x new-val)
      {:keys [trace weight]} (p/update model trace constraints)
      new-score (ev (:score trace))
      w (ev weight)]
  (check "update weight = new_score - old_score"
         (approx= w (- new-score old-score) 1e-5)))

;; Multi-address update
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                [x y]))
      trace (p/simulate model [])
      old-score (ev (:score trace))
      constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
      {:keys [trace weight]} (p/update model trace constraints)
      new-score (ev (:score trace))
      w (ev weight)]
  (check "multi-address update: weight = new_score - old_score"
         (approx= w (- new-score old-score) 1e-5)))

;; Update with same value → weight ≈ 0
(let [model (gen [] (dyn/trace :x (dist/gaussian 0 1)))
      trace (p/simulate model [])
      old-val (cm/get-value (cm/get-submap (:choices trace) :x))
      constraints (cm/choicemap :x old-val)
      {:keys [weight]} (p/update model trace constraints)
      w (ev weight)]
  (check "update with same value: weight ≈ 0" (approx= w 0.0 1e-5)))

;; -- 1.4: Regenerate invariants --
(println "\n-- 1.4 Regenerate invariants --")

;; Regenerate with no selection → weight ≈ 0, choices unchanged
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                [x y]))
      trace (p/simulate model [])
      old-x (ev (cm/get-value (cm/get-submap (:choices trace) :x)))
      {:keys [trace weight]} (p/regenerate model trace sel/none)
      new-x (ev (cm/get-value (cm/get-submap (:choices trace) :x)))
      w (ev weight)]
  (check "regenerate none: weight ≈ 0" (approx= w 0.0 1e-5))
  (check "regenerate none: choices unchanged" (approx= old-x new-x 1e-10)))

;; Regenerate selected: unselected addresses preserved
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                [x y]))
      trace (p/simulate model [])
      old-y (ev (cm/get-value (cm/get-submap (:choices trace) :y)))
      {:keys [trace]} (p/regenerate model trace (sel/select :x))
      new-y (ev (cm/get-value (cm/get-submap (:choices trace) :y)))]
  (check "regenerate :x: :y preserved" (approx= old-y new-y 1e-10)))

;; -- 1.5: Distribution GFI consistency --
(println "\n-- 1.5 Distribution GFI consistency --")

(doseq [[name d] [["gaussian" (dist/gaussian 0 1)]
                   ["uniform" (dist/uniform 0 1)]
                   ["exponential" (dist/exponential 1)]
                   ["laplace" (dist/laplace 0 1)]]]
  (let [trace (p/simulate d [])
        v (:retval trace)
        score (ev (:score trace))
        lp (ev (dist/log-prob d v))]
    (check (str name " GFI: simulate score = log-prob")
           (approx= score lp 1e-5))))

;; -- 1.6: Nested addressing --
(println "\n-- 1.6 Nested addressing --")

(let [inner (gen [] (dyn/trace :z (dist/gaussian 0 1)))
      outer (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/splice :inner inner)
                x))
      trace (p/simulate outer [])
      choices (:choices trace)]
  (check "nested: top-level :x exists"
         (cm/has-value? (cm/get-submap choices :x)))
  (check "nested: :inner/:z exists"
         (cm/has-value? (cm/get-submap (cm/get-submap choices :inner) :z))))

;; -- 1.7: Deterministic computation (no random choices) --
(println "\n-- 1.7 Deterministic computation --")

(let [model (gen [x] (* x x))
      trace (p/simulate model [5])
      score (ev (:score trace))]
  (check "deterministic: retval = 25" (= (:retval trace) 25))
  (check "deterministic: score = 0" (approx= score 0.0 1e-10)))

;; ============================================================================
;; Section 2: MCMC Convergence to Exact Posteriors (adapted from test_mcmc.py)
;; ============================================================================

(println "\n=== Section 2: MCMC Convergence ===")

;; -- 2.1: Normal-Normal conjugate --
;; Prior: mu ~ N(0, sigma0), Likelihood: x_i ~ N(mu, sigma)
;; Posterior: mu ~ N(post_mean, post_var)
(println "\n-- 2.1 Normal-Normal conjugate (MH) --")

(let [sigma0 10.0
      sigma 1.0
      xs [3.0 3.5 2.5 4.0 3.0]
      n (count xs)
      ;; Exact posterior
      post-var (/ 1.0 (+ (/ 1.0 (* sigma0 sigma0)) (/ n (* sigma sigma))))
      post-mean (* post-var (/ (reduce + xs) (* sigma sigma)))
      post-std (js/Math.sqrt post-var)
      ;; Model
      model (gen [xs sigma]
              (let [mu (dyn/trace :mu (dist/gaussian 0 sigma0))]
                (mx/eval! mu)
                (let [m (mx/item mu)]
                  (doseq [[i x] (map-indexed vector xs)]
                    (dyn/trace (keyword (str "y" i)) (dist/gaussian m sigma)))
                  m)))
      observations (reduce (fn [cm [i x]]
                             (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar x)))
                           cm/EMPTY (map-indexed vector xs))
      traces (mcmc/mh {:samples 500 :burn 100
                        :selection (sel/select :mu)}
                       model [xs sigma] observations)
      mus (mapv (fn [t]
                  (let [v (cm/get-value (cm/get-submap (:choices t) :mu))]
                    (mx/eval! v) (mx/item v)))
                traces)
      sample-mu (/ (reduce + mus) (count mus))]
  (check (str "MH posterior mean ≈ " (.toFixed post-mean 2) " (got " (.toFixed sample-mu 2) ")")
         (approx= sample-mu post-mean 0.3))
  (println (str "    exact=" (.toFixed post-mean 3) " sampled=" (.toFixed sample-mu 3))))

;; -- 2.2: Normal-Normal conjugate (HMC) --
;; HMC/MALA models must not call mx/eval! inside body (breaks gradient tracing)
(println "\n-- 2.2 Normal-Normal conjugate (HMC) --")

(let [sigma0 10.0
      sigma 1.0
      xs [3.0 3.5 2.5 4.0 3.0]
      n (count xs)
      post-var (/ 1.0 (+ (/ 1.0 (* sigma0 sigma0)) (/ n (* sigma sigma))))
      post-mean (* post-var (/ (reduce + xs) (* sigma sigma)))
      ;; HMC-compatible model: no eval!/item in body
      model (gen [xs sigma]
              (let [mu (dyn/trace :mu (dist/gaussian 0 sigma0))]
                (doseq [[i x] (map-indexed vector xs)]
                  (dyn/trace (keyword (str "y" i)) (dist/gaussian mu sigma)))
                mu))
      observations (reduce (fn [cm [i x]]
                             (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar x)))
                           cm/EMPTY (map-indexed vector xs))
      samples (mcmc/hmc {:samples 200 :step-size 0.05 :leapfrog-steps 10
                          :burn 50 :addresses [:mu]}
                         model [xs sigma] observations)
      mus (mapv first samples)
      sample-mu (/ (reduce + mus) (count mus))]
  (check (str "HMC posterior mean ≈ " (.toFixed post-mean 2))
         (approx= sample-mu post-mean 0.3))
  (println (str "    exact=" (.toFixed post-mean 3) " sampled=" (.toFixed sample-mu 3))))

;; -- 2.3: Normal-Normal conjugate (MALA) --
(println "\n-- 2.3 Normal-Normal conjugate (MALA) --")

(let [sigma0 10.0
      sigma 1.0
      xs [3.0 3.5 2.5 4.0 3.0]
      n (count xs)
      post-var (/ 1.0 (+ (/ 1.0 (* sigma0 sigma0)) (/ n (* sigma sigma))))
      post-mean (* post-var (/ (reduce + xs) (* sigma sigma)))
      model (gen [xs sigma]
              (let [mu (dyn/trace :mu (dist/gaussian 0 sigma0))]
                (doseq [[i x] (map-indexed vector xs)]
                  (dyn/trace (keyword (str "y" i)) (dist/gaussian mu sigma)))
                mu))
      observations (reduce (fn [cm [i x]]
                             (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar x)))
                           cm/EMPTY (map-indexed vector xs))
      samples (mcmc/mala {:samples 200 :step-size 0.1
                           :burn 100 :addresses [:mu]}
                          model [xs sigma] observations)
      mus (mapv first samples)
      sample-mu (/ (reduce + mus) (count mus))]
  (check (str "MALA posterior mean ≈ " (.toFixed post-mean 2))
         (approx= sample-mu post-mean 1.0))
  (println (str "    exact=" (.toFixed post-mean 3) " sampled=" (.toFixed sample-mu 3))))

;; -- 2.4: Beta-Bernoulli conjugate (MH) --
(println "\n-- 2.4 Beta-Bernoulli conjugate (MH) --")

(let [alpha-prior 2.0 beta-prior 2.0
      obs [1.0 1.0 1.0 0.0 1.0 0.0 1.0]
      n (count obs)
      sum-x (reduce + obs)
      ;; Exact posterior: Beta(alpha + sum, beta + n - sum)
      alpha-post (+ alpha-prior sum-x)
      beta-post (+ beta-prior (- n sum-x))
      post-mean (/ alpha-post (+ alpha-post beta-post))
      ;; Model
      model (gen [n]
              (let [p (dyn/trace :p (dist/beta-dist alpha-prior beta-prior))]
                (mx/eval! p)
                (let [pv (mx/item p)]
                  (doseq [i (range n)]
                    (dyn/trace (keyword (str "x" i)) (dist/bernoulli pv)))
                  pv)))
      observations (reduce (fn [cm [i x]]
                             (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar x)))
                           cm/EMPTY (map-indexed vector obs))
      traces (mcmc/mh {:samples 500 :burn 100
                        :selection (sel/select :p)}
                       model [n] observations)
      ps (mapv (fn [t]
                 (let [v (cm/get-value (cm/get-submap (:choices t) :p))]
                   (mx/eval! v) (mx/item v)))
               traces)
      sample-p (/ (reduce + ps) (count ps))]
  (check (str "Beta-Bernoulli posterior mean ≈ " (.toFixed post-mean 2))
         (approx= sample-p post-mean 0.15))
  (println (str "    exact=" (.toFixed post-mean 3) " sampled=" (.toFixed sample-p 3))))

;; -- 2.5: Acceptance rate validation --
(println "\n-- 2.5 Acceptance rate validation --")

(let [;; Model with observations that create likelihood tension
      model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (mx/eval! mu)
                (let [m (mx/item mu)]
                  (dyn/trace :y0 (dist/gaussian m 1))
                  (dyn/trace :y1 (dist/gaussian m 1))
                  m)))
      obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 3.5))
      {:keys [trace]} (p/generate model [] obs)
      ;; Run MH and count acceptances
      n-steps 200
      accepted (volatile! 0)
      _ (reduce (fn [t _]
                  (let [t' (mcmc/mh-step t (sel/select :mu))]
                    (when (not= t t') (vswap! accepted inc))
                    t'))
                trace (range n-steps))
      rate (/ @accepted n-steps)]
  (check (str "MH acceptance rate in (0, 1): " (.toFixed rate 2))
         (and (> rate 0.0) (< rate 1.0))))

;; -- 2.6: Chain stationarity --
(println "\n-- 2.6 Chain stationarity --")

(let [model (gen [] (dyn/trace :x (dist/gaussian 3 1)))
      obs (cm/choicemap :x (mx/scalar 3.0))
      traces (mcmc/mh {:samples 200 :burn 50
                        :selection (sel/select :x)}
                       model [] obs)
      vals (mapv (fn [t]
                   (let [v (cm/get-value (cm/get-submap (:choices t) :x))]
                     (mx/eval! v) (mx/item v)))
                 traces)
      half (quot (count vals) 2)
      first-half (subvec vals 0 half)
      second-half (subvec vals half)
      mean1 (/ (reduce + first-half) (count first-half))
      mean2 (/ (reduce + second-half) (count second-half))]
  (check (str "stationarity: |mean1 - mean2| < 0.5 (" (.toFixed mean1 2) " vs " (.toFixed mean2 2) ")")
         (< (js/Math.abs (- mean1 mean2)) 0.5)))

;; ============================================================================
;; Section 3: Importance Sampling & SMC (adapted from test_smc.py)
;; ============================================================================

(println "\n=== Section 3: Importance Sampling & SMC ===")

;; -- 3.1: IS log marginal likelihood convergence --
(println "\n-- 3.1 IS log-ML convergence with particles --")

(let [model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 5))]
                (mx/eval! mu)
                (dyn/trace :y (dist/gaussian (mx/item mu) 1))
                mu))
      obs (cm/choicemap :y (mx/scalar 2.0))
      ;; Run IS with increasing particles and check convergence
      log-mls (mapv (fn [n-particles]
                      (let [{:keys [log-ml-estimate]}
                            (is/importance-sampling {:samples n-particles}
                                                    model [] obs)]
                        (mx/eval! log-ml-estimate)
                        (mx/item log-ml-estimate)))
                    [50 200 500])
      ;; Variance should decrease (later estimates should be more consistent)
      ;; Just check all are finite
      all-finite (every? finite? log-mls)]
  (check "IS log-ML estimates are all finite" all-finite)
  (println (str "    log-MLs: " (mapv #(.toFixed % 2) log-mls))))

;; -- 3.2: IS posterior concentration --
(println "\n-- 3.2 IS posterior concentration --")

(let [model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 5))]
                (mx/eval! mu)
                (dyn/trace :y (dist/gaussian (mx/item mu) 0.1))
                mu))
      obs (cm/choicemap :y (mx/scalar 3.0))
      ;; With tight likelihood, posterior should concentrate near y=3
      {:keys [traces log-weights]}
      (is/importance-sampling {:samples 200} model [] obs)
      ;; Weighted mean
      weights-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights))
      log-probs (mx/subtract weights-arr (mx/logsumexp weights-arr))
      _ (mx/eval! log-probs)
      probs (mx/->clj (mx/exp log-probs))
      mus (mapv (fn [t]
                  (let [v (cm/get-value (cm/get-submap (:choices t) :mu))]
                    (mx/eval! v) (mx/item v)))
                traces)
      weighted-mean (reduce + (map * mus probs))]
  (check (str "IS weighted mean ≈ 3.0 (got " (.toFixed weighted-mean 2) ")")
         (approx= weighted-mean 3.0 0.5)))

;; -- 3.3: Importance resampling --
(println "\n-- 3.3 Importance resampling --")

(let [model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 5))]
                (mx/eval! mu)
                (dyn/trace :y (dist/gaussian (mx/item mu) 1))
                mu))
      obs (cm/choicemap :y (mx/scalar 2.0))
      resampled (is/importance-resampling {:samples 50 :particles 200}
                                          model [] obs)
      mus (mapv (fn [t]
                  (let [v (cm/get-value (cm/get-submap (:choices t) :mu))]
                    (mx/eval! v) (mx/item v)))
                resampled)
      mean-mu (/ (reduce + mus) (count mus))]
  (check "resampled traces exist" (= 50 (count resampled)))
  (check (str "resampled mean ≈ 2.0 (got " (.toFixed mean-mu 2) ")")
         (approx= mean-mu 2.0 1.0)))

;; -- 3.4: SMC with sequential observations --
(println "\n-- 3.4 SMC sequential --")

(let [model (gen [xs]
              (let [mu (dyn/trace :mu (dist/gaussian 0 5))]
                (mx/eval! mu)
                (let [m (mx/item mu)]
                  (doseq [[i x] (map-indexed vector xs)]
                    (dyn/trace (keyword (str "y" i)) (dist/gaussian m 1)))
                  m)))
      ;; Sequential observations
      obs-seq [(cm/choicemap :y0 (mx/scalar 2.0))
               (cm/choicemap :y0 (mx/scalar 2.0) :y1 (mx/scalar 2.5))
               (cm/choicemap :y0 (mx/scalar 2.0) :y1 (mx/scalar 2.5) :y2 (mx/scalar 1.5))]
      result (smc/smc {:particles 20}
                       model [[2.0 2.5 1.5]] obs-seq)]
  (check "SMC returns traces" (= 20 (count (:traces result))))
  (check "SMC log-ML is finite" (do (mx/eval! (:log-ml-estimate result))
                                     (finite? (mx/item (:log-ml-estimate result))))))

;; ============================================================================
;; Section 4: Variational Inference (adapted from test_vi.py)
;; ============================================================================

(println "\n=== Section 4: Variational Inference ===")

;; -- 4.1: VI converges to known target --
(println "\n-- 4.1 VI convergence to known target --")

(let [;; Target: N(3, 1) — VI should find mu ≈ 3, sigma ≈ 1
      log-density (fn [params]
                    (let [diff (mx/subtract params (mx/scalar 3.0))]
                      (mx/multiply (mx/scalar -0.5) (mx/sum (mx/multiply diff diff)))))
      init-params (mx/scalar 0.0)
      result (vi/vi {:iterations 100 :learning-rate 0.05 :elbo-samples 5}
                     log-density init-params)
      mu (ev (:mu result))
      sigma (ev (:sigma result))]
  (check (str "VI mu ≈ 3.0 (got " (.toFixed mu 2) ")")
         (approx= mu 3.0 0.5))
  (check (str "VI sigma ≈ 1.0 (got " (.toFixed sigma 2) ")")
         (approx= sigma 1.0 0.5)))

;; -- 4.2: VI ELBO improves over iterations --
(println "\n-- 4.2 ELBO improves --")

(let [log-density (fn [params]
                    (mx/multiply (mx/scalar -0.5) (mx/sum (mx/square params))))
      init-params (mx/array [5.0 5.0])
      result (vi/vi {:iterations 100 :learning-rate 0.02 :elbo-samples 5}
                     log-density init-params)
      history (:elbo-history result)
      n (count history)]
  (when (> n 4)
    (let [early-avg (/ (reduce + (take 2 history)) 2)
          late-avg (/ (reduce + (take-last 2 history)) 2)]
      (check (str "ELBO improves: early=" (.toFixed early-avg 1) " late=" (.toFixed late-avg 1))
             (> late-avg early-avg)))))

;; -- 4.3: VI with model-equivalent log-density --
;; Note: vi-from-model uses vmap which conflicts with handler mutable state.
;; Test the same posterior via manual log-density instead.
(println "\n-- 4.3 VI model-equivalent --")

(let [;; Equivalent to: mu ~ N(0,10), y ~ N(mu,1), observe y=5
      log-density (fn [params]
                    (let [mu params]
                      (mx/add (dist/log-prob (dist/gaussian 0 10) mu)
                              (dist/log-prob (dist/gaussian mu 1) (mx/scalar 5.0)))))
      init-params (mx/scalar 0.0)
      result (vi/vi {:iterations 200 :learning-rate 0.05 :elbo-samples 5}
                     log-density init-params)
      mu (ev (:mu result))]
  (check (str "VI model-equiv mu ≈ 5.0 (got " (.toFixed mu 2) ")")
         (approx= mu 5.0 1.5)))

;; ============================================================================
;; Section 5: Combinator GFI Contracts (adapted from test_core.py)
;; ============================================================================

(println "\n=== Section 5: Combinator GFI Contracts ===")

;; -- 5.1: Map combinator --
(println "\n-- 5.1 Map combinator GFI --")

(let [kernel (gen [x] (dyn/trace :y (dist/gaussian x 1)))
      mapped (comb/map-combinator kernel)
      ;; Simulate
      trace (p/simulate mapped [[1.0 2.0 3.0]])
      retvals (:retval trace)
      score (ev (:score trace))]
  (check "map simulate: returns 3 values" (= 3 (count retvals)))
  (check "map simulate: score is finite" (finite? score))

  ;; Generate with constraints
  (let [constraints (-> cm/EMPTY
                        (cm/set-choice [0] (-> cm/EMPTY (cm/set-choice [:y] (mx/scalar 1.0))))
                        (cm/set-choice [1] (-> cm/EMPTY (cm/set-choice [:y] (mx/scalar 2.0))))
                        (cm/set-choice [2] (-> cm/EMPTY (cm/set-choice [:y] (mx/scalar 3.0)))))
        {:keys [trace weight]} (p/generate mapped [[1.0 2.0 3.0]] constraints)
        gen-score (ev (:score trace))
        w (ev weight)]
    (check "map generate: weight ≈ score (full constraints)"
           (approx= w gen-score 1e-4)))

  ;; Update
  (let [trace (p/simulate mapped [[1.0 2.0 3.0]])
        old-score (ev (:score trace))
        constraints (-> cm/EMPTY
                        (cm/set-choice [1] (-> cm/EMPTY (cm/set-choice [:y] (mx/scalar 99.0)))))
        {:keys [trace weight]} (p/update mapped trace constraints)
        new-score (ev (:score trace))
        w (ev weight)]
    (check (str "map update: weight ≈ new_score - old_score (w=" (.toFixed w 2) " diff=" (.toFixed (- new-score old-score) 2) ")")
           (approx= w (- new-score old-score) 0.5))))

;; -- 5.2: Unfold combinator --
(println "\n-- 5.2 Unfold combinator GFI --")

(let [step (gen [t state]
             (let [noise (dyn/trace :noise (dist/gaussian 0 0.1))]
               (mx/eval! noise)
               (+ state (mx/item noise))))
      unfolded (comb/unfold-combinator step)
      trace (p/simulate unfolded [5 0.0])
      states (:retval trace)
      score (ev (:score trace))]
  (check "unfold simulate: 5 states" (= 5 (count states)))
  (check "unfold simulate: score finite" (finite? score))

  ;; Generate with constraint
  (let [constraints (-> cm/EMPTY
                        (cm/set-choice [0] (-> cm/EMPTY (cm/set-choice [:noise] (mx/scalar 0.1)))))
        {:keys [trace weight]} (p/generate unfolded [5 0.0] constraints)
        w (ev weight)]
    (check "unfold generate: weight finite" (finite? w))))

;; -- 5.3: Switch combinator --
(println "\n-- 5.3 Switch combinator GFI --")

(let [branch-a (gen [] (dyn/trace :v (dist/gaussian 0 1)))
      branch-b (gen [] (dyn/trace :v (dist/gaussian 10 1)))
      switched (comb/switch-combinator branch-a branch-b)]
  ;; Branch 0
  (let [trace (p/simulate switched [0])
        v (ev (cm/get-value (cm/get-submap (:choices trace) :v)))
        score (ev (:score trace))]
    (check "switch branch 0: value near 0" (< (js/Math.abs v) 5))
    (check "switch branch 0: score finite" (finite? score)))
  ;; Branch 1
  (let [trace (p/simulate switched [1])
        v (ev (cm/get-value (cm/get-submap (:choices trace) :v)))]
    (check "switch branch 1: value near 10" (< (js/Math.abs (- v 10)) 5))))

;; ============================================================================
;; Section 6: Gradient Flow (adapted from test_adev.py)
;; ============================================================================

(println "\n=== Section 6: Gradient Flow ===")

;; -- 6.1: Gradient through Gaussian log-prob --
(println "\n-- 6.1 Gradient through Gaussian log-prob --")

(let [;; d/dmu log N(x|mu,1) = (x - mu) / sigma^2 = x - mu
      f (fn [mu] (dist/log-prob (dist/gaussian mu 1) (mx/scalar 3.0)))
      grad-f (mx/grad f)
      mu0 (mx/scalar 1.0)
      g (grad-f mu0)
      _ (mx/eval! g)
      gv (mx/item g)]
  ;; Gradient should be (3 - 1) = 2
  (check (str "d/dmu log N(3|mu,1) at mu=1 ≈ 2.0 (got " (.toFixed gv 3) ")")
         (approx= gv 2.0 1e-4)))

;; -- 6.2: Gradient through model score function --
(println "\n-- 6.2 Gradient through model score --")

(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1)))
      score-fn (fn [x-val]
                 (:weight (p/generate model []
                            (cm/choicemap :x x-val))))
      grad-score (mx/grad score-fn)
      x0 (mx/scalar 2.0)
      g (grad-score x0)
      _ (mx/eval! g)
      gv (mx/item g)]
  ;; d/dx log N(x|0,1) = -x, so at x=2 should be -2
  (check (str "d/dx score at x=2 ≈ -2.0 (got " (.toFixed gv 3) ")")
         (approx= gv -2.0 1e-3)))

;; -- 6.3: value-and-grad correctness --
(println "\n-- 6.3 value-and-grad --")

(let [f (fn [x] (mx/multiply (mx/scalar -0.5) (mx/square x)))
      vg (mx/value-and-grad f)
      x0 (mx/scalar 3.0)
      [v g] (vg x0)
      _ (mx/eval! v g)]
  (check "value = -4.5" (approx= (mx/item v) -4.5 1e-5))
  (check "grad = -3.0" (approx= (mx/item g) -3.0 1e-5)))

;; -- 6.4: Gradient of multi-address score --
(println "\n-- 6.4 Multi-address score gradient --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                [x y]))
      score-fn (fn [params]
                 (let [cm (cm/choicemap :x (mx/index params 0)
                                        :y (mx/index params 1))]
                   (:weight (p/generate model [] cm))))
      grad-fn (mx/grad score-fn)
      params (mx/array [1.0 2.0])
      g (grad-fn params)
      _ (mx/eval! g)
      gv (mx/->clj g)]
  ;; d/dx log N(x|0,1) = -x, d/dy log N(y|0,1) = -y
  (check (str "grad[0] ≈ -1.0 (got " (.toFixed (first gv) 3) ")")
         (approx= (first gv) -1.0 1e-3))
  (check (str "grad[1] ≈ -2.0 (got " (.toFixed (second gv) 3) ")")
         (approx= (second gv) -2.0 1e-3)))

;; -- 6.5: Reparameterized sampling gradient --
(println "\n-- 6.5 Reparameterized sampling --")

(let [;; E[X] where X ~ N(mu, 1), so d/dmu E[X] = 1
      ;; Approximate via reparameterized sample: X = mu + eps, eps ~ N(0,1)
      key (rng/fresh-key)
      f (fn [mu]
          (let [eps (rng/normal key [])
                x (mx/add mu eps)]
            x))
      grad-f (mx/grad f)
      mu (mx/scalar 5.0)
      g (grad-f mu)
      _ (mx/eval! g)]
  (check (str "d/dmu (mu + eps) = 1.0 (got " (.toFixed (mx/item g) 3) ")")
         (approx= (mx/item g) 1.0 1e-5)))

;; -- 6.6: Gradient through compiled function --
(println "\n-- 6.6 Compiled gradient --")

(let [f (fn [x] (mx/exp (mx/negative (mx/multiply (mx/scalar 0.5) (mx/square x)))))
      compiled-grad (mx/compile-fn (mx/grad f))
      x0 (mx/scalar 1.0)
      ;; d/dx exp(-x^2/2) = -x * exp(-x^2/2) = -1 * exp(-0.5) ≈ -0.6065
      g (compiled-grad x0)
      _ (mx/eval! g)]
  (check (str "compiled grad at x=1 ≈ -0.6065 (got " (.toFixed (mx/item g) 4) ")")
         (approx= (mx/item g) (* -1.0 (js/Math.exp -0.5)) 1e-3)))

;; ============================================================================
;; Section 7: Diagnostics (adapted from test_mcmc.py)
;; ============================================================================

(println "\n=== Section 7: Diagnostics ===")

;; -- 7.1: ESS --
(println "\n-- 7.1 ESS --")

(let [;; IID samples should have ESS ≈ n
      samples (mapv (fn [_] (let [v (mx/random-normal [])]
                              (mx/eval! v) v))
                    (range 50))
      effective (diag/ess samples)]
  (check (str "IID ESS ≈ 50 (got " (.toFixed effective 0) ")")
         (> effective 25)))

;; Correlated samples should have lower ESS
(let [samples (loop [i 0 v 0.0 acc []]
                (if (>= i 50)
                  acc
                  (let [v' (+ (* 0.99 v) (* 0.1 (- (* 2 (js/Math.random)) 1)))]
                    (recur (inc i) v' (conj acc (mx/scalar v'))))))
      effective (diag/ess samples)]
  (check (str "correlated ESS < 50 (got " (.toFixed effective 0) ")")
         (< effective 40)))

;; -- 7.2: R-hat --
(println "\n-- 7.2 R-hat --")

;; Same distribution → R-hat ≈ 1 (use plain numbers — r-hat handles them)
(let [chain1 (mapv (fn [_] (+ 3 (* 0.5 (- (* 2 (js/Math.random)) 1)))) (range 50))
      chain2 (mapv (fn [_] (+ 3 (* 0.5 (- (* 2 (js/Math.random)) 1)))) (range 50))
      rh (diag/r-hat [chain1 chain2])]
  (check (str "converged R-hat ≈ 1.0 (got " (.toFixed rh 3) ")")
         (< rh 1.2)))

;; Different distributions → R-hat >> 1
(let [chain1 (mapv (fn [_] (* 0.5 (- (* 2 (js/Math.random)) 1))) (range 50))
      chain2 (mapv (fn [_] (+ 10 (* 0.5 (- (* 2 (js/Math.random)) 1)))) (range 50))
      rh (diag/r-hat [chain1 chain2])]
  (check (str "diverged R-hat >> 1.0 (got " (.toFixed rh 2) ")")
         (> rh 2.0)))

;; -- 7.3: Summary statistics --
(println "\n-- 7.3 Summary statistics --")

(let [samples (mapv #(mx/scalar %) [1.0 2.0 3.0 4.0 5.0])
      mu (diag/sample-mean samples)
      sd (diag/sample-std samples)
      _ (mx/eval! mu sd)]
  (check "mean of [1..5] = 3.0" (approx= (mx/item mu) 3.0 1e-4))
  (check "std of [1..5] > 0" (> (mx/item sd) 0)))

;; ============================================================================
;; Section 8: Numerical Stability (adapted from test_core.py, test_distributions.py)
;; ============================================================================

(println "\n=== Section 8: Numerical Stability ===")

;; -- 8.1: Extreme Gaussian parameters --
(println "\n-- 8.1 Extreme parameters --")

(let [;; Very large sigma
      d1 (dist/gaussian 0 1000)
      lp1 (ev (dist/log-prob d1 (mx/scalar 0.0)))
      ;; Very small sigma
      d2 (dist/gaussian 0 0.001)
      lp2 (ev (dist/log-prob d2 (mx/scalar 0.0)))
      ;; Large value
      d3 (dist/gaussian 0 1)
      lp3 (ev (dist/log-prob d3 (mx/scalar 100.0)))]
  (check "large sigma: log-prob finite" (finite? lp1))
  (check "small sigma: log-prob finite" (finite? lp2))
  (check "large value: log-prob finite" (finite? lp3)))

;; -- 8.2: Distributions at boundary values --
(println "\n-- 8.2 Boundary values --")

(let [;; Uniform at exact boundaries
      u (dist/uniform 0 1)
      lp-lo (ev (dist/log-prob u (mx/scalar 0.0)))
      lp-hi (ev (dist/log-prob u (mx/scalar 1.0)))
      lp-out (ev (dist/log-prob u (mx/scalar 1.5)))]
  (check "uniform at lo: finite" (finite? lp-lo))
  (check "uniform at hi: finite" (finite? lp-hi))
  (check "uniform outside: -Inf" (= lp-out ##-Inf)))

(let [;; Exponential at 0
      e (dist/exponential 1)
      lp0 (ev (dist/log-prob e (mx/scalar 0.0)))
      lp-neg (ev (dist/log-prob e (mx/scalar -1.0)))]
  (check "exponential at 0: finite" (finite? lp0))
  (check "exponential at -1: -Inf" (= lp-neg ##-Inf)))

;; -- 8.3: Large model score --
(println "\n-- 8.3 Large model --")

(let [n 10
      model (gen [n]
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (mx/eval! mu)
                (let [m (mx/item mu)]
                  (doseq [i (range n)]
                    (dyn/trace (keyword (str "y" i)) (dist/gaussian m 1)))
                  m)))
      trace (p/simulate model [n])
      score (ev (:score trace))]
  (check (str "10-obs model: score finite (" (.toFixed score 1) ")")
         (finite? score)))

;; -- 8.4: Multivariate Normal edge cases --
(println "\n-- 8.4 MVN stability --")

(let [;; Near-identity covariance
      k 10
      cov (mx/add (mx/eye k) (mx/multiply (mx/scalar 0.001) (mx/random-normal [k k])))
      ;; Make it symmetric
      cov-sym (mx/multiply (mx/scalar 0.5) (mx/add cov (mx/transpose cov)))
      ;; Add diagonal dominance
      cov-spd (mx/add cov-sym (mx/multiply (mx/scalar (float k)) (mx/eye k)))
      _ (mx/eval! cov-spd)
      mvn (dist/multivariate-normal (mx/zeros [k]) cov-spd)
      sample (dist/sample mvn)
      _ (mx/eval! sample)
      lp (dist/log-prob mvn sample)
      _ (mx/eval! lp)]
  (check "10-dim MVN: sample finite" (every? finite? (mx/->clj sample)))
  (check "10-dim MVN: log-prob finite" (finite? (mx/item lp))))

;; ============================================================================
;; Section 9: Score Consistency Across Operations
;;            (adapted from test_core.py TestGenerateConsistency)
;; ============================================================================

(println "\n=== Section 9: Score Consistency ===")

;; -- 9.1: Simulate → Generate round-trip for hierarchical model --
(println "\n-- 9.1 Hierarchical model round-trip --")

(let [model (gen [xs]
              (let [slope (dyn/trace :slope (dist/gaussian 0 10))
                    intercept (dyn/trace :intercept (dist/gaussian 0 10))]
                (mx/eval! slope intercept)
                (let [s (mx/item slope) i (mx/item intercept)]
                  (doseq [[j x] (map-indexed vector xs)]
                    (dyn/trace (keyword (str "y" j))
                               (dist/gaussian (+ (* s x) i) 1)))
                  [s i])))
      xs [1.0 2.0 3.0]
      trace (p/simulate model [xs])
      choices (:choices trace)
      score (ev (:score trace))
      {:keys [weight]} (p/generate model [xs] choices)
      w (ev weight)]
  (check "hierarchical round-trip: weight ≈ score"
         (approx= w score 1e-3)))

;; -- 9.2: Update preserves score when no change --
(println "\n-- 9.2 Update with no-op --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                [x y]))
      trace (p/simulate model [])
      old-score (ev (:score trace))
      ;; Update with same values
      old-x (cm/get-value (cm/get-submap (:choices trace) :x))
      old-y (cm/get-value (cm/get-submap (:choices trace) :y))
      {:keys [trace weight]} (p/update model trace
                               (cm/choicemap :x old-x :y old-y))
      new-score (ev (:score trace))
      w (ev weight)]
  (check "no-op update: score unchanged" (approx= old-score new-score 1e-5))
  (check "no-op update: weight ≈ 0" (approx= w 0.0 1e-5)))

;; -- 9.3: Generate/Update consistency --
;; Generating constrained, then updating to new values, should match
;; generating directly with those new values
(println "\n-- 9.3 Generate/Update consistency --")

(let [model (gen [] (dyn/trace :x (dist/gaussian 0 1)))
      val-a (mx/scalar 1.0)
      val-b (mx/scalar 3.0)
      ;; Path 1: generate with val-a, then update to val-b
      {:keys [trace]} (p/generate model [] (cm/choicemap :x val-a))
      {:keys [trace weight]} (p/update model trace (cm/choicemap :x val-b))
      path1-score (ev (:score trace))
      path1-weight (ev weight)
      ;; Path 2: generate directly with val-b
      {:keys [trace weight]} (p/generate model [] (cm/choicemap :x val-b))
      path2-score (ev (:score trace))]
  (check "generate→update score = direct generate score"
         (approx= path1-score path2-score 1e-5)))

;; ============================================================================
;; Results
;; ============================================================================

(println "\n=== GenJAX Compatibility Test Results ===")
(println (str "  Passed: " @pass-count))
(println (str "  Failed: " @fail-count))
(println (str "  Total:  " (+ @pass-count @fail-count)))

(when (pos? @fail-count)
  (js/process.exit 1))
