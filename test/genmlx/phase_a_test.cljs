(ns genmlx.phase-a-test
  "Tests for Phase A features: propose, custom proposal MH, mask combinator,
   vectorized update/regenerate, new distributions, unfold/switch update/regenerate."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.trace :as tr]
            [genmlx.combinators :as comb]
            [genmlx.vectorized :as vec]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Phase A Tests ===\n")

;; ---------------------------------------------------------------------------
;; 1.1 IPropose — verify weight = joint log-prob
;; ---------------------------------------------------------------------------

(println "-- IPropose on Distribution: weight = log-prob of sample --")
(let [d (dist/gaussian 0 1)
      {:keys [choices weight retval]} (p/propose d [])]
  (mx/eval! weight retval)
  (let [manual-lp (dist/log-prob d retval)]
    (mx/eval! manual-lp)
    (assert-close "propose weight matches manual log-prob"
      (mx/item manual-lp) (mx/item weight) 1e-5)))

(println "\n-- IPropose on DynamicGF: weight = sum of log-probs --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                x))
      {:keys [choices weight retval]} (p/propose model [])
      ;; propose weight should equal the trace score
      trace (p/simulate model [])
      ;; Re-run with same choices to verify score consistency
      x-val (cm/get-value (cm/get-submap choices :x))
      _ (mx/eval! x-val weight)
      verify-lp (dist/log-prob (dist/gaussian 0 1) x-val)]
  (mx/eval! verify-lp)
  (assert-close "propose weight = log-prob of sampled choice"
    (mx/item verify-lp) (mx/item weight) 1e-5))

;; ---------------------------------------------------------------------------
;; IAssess — verify weight = joint log-prob of given choices
;; ---------------------------------------------------------------------------

(println "\n-- IAssess on Distribution: known value --")
(let [d (dist/gaussian 0 1)
      {:keys [weight]} (p/assess d [] (cm/->Value (mx/scalar 0.0)))]
  (mx/eval! weight)
  ;; log(1/sqrt(2*pi)) = -0.91894...
  (assert-close "assess N(0,1) at 0" -0.91894 (mx/item weight) 1e-4))

(println "\n-- IAssess on DynamicGF: all choices specified --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                (mx/add x y)))
      choices (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
      {:keys [weight]} (p/assess model [] choices)]
  (mx/eval! weight)
  ;; weight should be log p(x=1, y=2) = log N(1;0,1) + log N(2;0,1)
  (let [lp-x (mx/item (do (let [v (dist/log-prob (dist/gaussian 0 1) (mx/scalar 1.0))] (mx/eval! v) v)))
        lp-y (mx/item (do (let [v (dist/log-prob (dist/gaussian 0 1) (mx/scalar 2.0))] (mx/eval! v) v)))
        expected (+ lp-x lp-y)]
    (assert-close "assess sum of log-probs" expected (mx/item weight) 1e-4)))

;; ---------------------------------------------------------------------------
;; 5.1 New distributions — verify log-prob correctness
;; ---------------------------------------------------------------------------

(println "\n-- Cauchy: log-prob = -log(pi * scale * (1 + z^2)) --")
(let [d (dist/cauchy 0 1)]
  (let [lp (dist/log-prob d (mx/scalar 0.0))]
    (mx/eval! lp)
    (assert-close "cauchy(0,1) lp at 0" (- (js/Math.log js/Math.PI)) (mx/item lp) 1e-5))
  (let [lp (dist/log-prob d (mx/scalar 1.0))]
    (mx/eval! lp)
    ;; -log(pi * (1 + 1)) = -log(2*pi)
    (assert-close "cauchy(0,1) lp at 1" (- (js/Math.log (* 2 js/Math.PI))) (mx/item lp) 1e-5)))

(println "\n-- Cauchy: batch sampling shape and log-prob broadcasting --")
(let [d (dist/cauchy 2 3)
      samples (dc/dist-sample-n d nil 200)
      _ (mx/eval! samples)
      lps (dc/dist-log-prob d samples)
      _ (mx/eval! lps)]
  (assert-true "cauchy batch shape" (= [200] (mx/shape samples)))
  (assert-true "cauchy batch lp shape" (= [200] (mx/shape lps))))

(println "\n-- Geometric: log p(k) = k*log(1-p) + log(p) --")
(let [d (dist/geometric 0.3)]
  (doseq [k [0 1 2 5]]
    (let [lp (dist/log-prob d (mx/scalar k))]
      (mx/eval! lp)
      (let [expected (+ (* k (js/Math.log 0.7)) (js/Math.log 0.3))]
        (assert-close (str "geometric lp at k=" k) expected (mx/item lp) 0.01)))))

(println "\n-- Binomial: log C(n,k) + k*log(p) + (n-k)*log(1-p) --")
(let [d (dist/binomial 10 0.5)]
  ;; P(X=5) = C(10,5) * 0.5^10 = 252/1024
  (let [lp (dist/log-prob d (mx/scalar 5))]
    (mx/eval! lp)
    (assert-close "binomial(10,0.5) lp at 5"
      (js/Math.log (/ 252 1024)) (mx/item lp) 0.01))
  ;; Verify support has 11 values (0..10)
  (assert-true "binomial support size" (= 11 (count (dist/support d)))))

(println "\n-- Discrete Uniform: log p = -log(n) --")
(let [d (dist/discrete-uniform 1 6)]
  (let [lp (dist/log-prob d (mx/scalar 3))]
    (mx/eval! lp)
    (assert-close "uniform(1,6) lp" (- (js/Math.log 6)) (mx/item lp) 1e-5))
  ;; Out of support
  (let [lp (dist/log-prob d (mx/scalar 0))]
    (mx/eval! lp)
    (assert-true "uniform(1,6) lp at 0 = -Inf" (= ##-Inf (mx/item lp)))))

(println "\n-- Truncated Normal: samples in bounds, lp > standard normal lp --")
(let [d (dist/truncated-normal 0 1 -1 1)
      samples (mapv (fn [_] (let [v (dist/sample d)] (mx/eval! v) (mx/item v)))
                    (range 200))]
  (assert-true "all truncated-normal in [-1,1]"
    (every? #(and (>= % -1.0) (<= % 1.0)) samples))
  ;; At v=0, truncated-normal lp should be higher than standard normal
  ;; because the probability mass is concentrated in [-1,1]
  (let [tn-lp (dist/log-prob d (mx/scalar 0.0))
        n-lp (dist/log-prob (dist/gaussian 0 1) (mx/scalar 0.0))]
    (mx/eval! tn-lp n-lp)
    (assert-true "truncated lp > standard lp at 0"
      (> (mx/item tn-lp) (mx/item n-lp)))))

(println "\n-- Truncated Normal: batch sampling respects bounds --")
(let [d (dist/truncated-normal 0 1 -2 2)
      batch (dc/dist-sample-n d nil 500)]
  (mx/eval! batch)
  (let [vals (mx/->clj batch)]
    (assert-true "all batch in [-2,2]"
      (every? #(and (>= % -2.0) (<= % 2.0)) vals))))

(println "\n-- Inverse Gamma: shape 3 scale 2, mean = scale/(shape-1) = 1 --")
(let [d (dist/inv-gamma 3 2)
      samples (mapv (fn [_] (let [v (dist/sample d)] (mx/eval! v) (mx/item v)))
                    (range 2000))
      mean (/ (reduce + samples) (count samples))]
  (assert-true "all inv-gamma > 0" (every? pos? samples))
  (assert-close "inv-gamma mean near 1" 1.0 mean 0.15))

;; ---------------------------------------------------------------------------
;; 3.2 Unfold update — verify score changes correctly
;; ---------------------------------------------------------------------------

(println "\n-- Unfold update: score changes when constraint changes --")
(let [step (gen [t state]
             (let [next (dyn/trace :x (dist/gaussian state 0.1))]
               (mx/eval! next)
               (mx/item next)))
      unfold (comb/unfold-combinator step)
      trace (p/simulate unfold [3 0.0])
      ;; Constrain step 1's :x to a specific value
      new-constraints (cm/set-choice cm/EMPTY [1] (cm/choicemap :x (mx/scalar 5.0)))
      {:keys [trace weight discard]} (p/update unfold trace new-constraints)]
  (assert-true "unfold update trace exists" (instance? tr/Trace trace))
  (mx/eval! weight)
  (assert-true "unfold update weight is finite" (js/isFinite (mx/item weight)))
  ;; Weight should be nonzero since we changed a value
  (assert-true "unfold update weight != 0" (not= 0.0 (mx/item weight))))

(println "\n-- Switch update and regenerate --")
(let [b0 (gen [] (let [x (dyn/trace :x (dist/gaussian 0 1))] (mx/eval! x) (mx/item x)))
      b1 (gen [] (let [x (dyn/trace :x (dist/gaussian 10 1))] (mx/eval! x) (mx/item x)))
      sw (comb/switch-combinator b0 b1)
      trace (p/simulate sw [0])
      ;; Update: change :x to 2.0
      {:keys [trace weight]} (p/update sw trace (cm/choicemap :x (mx/scalar 2.0)))
      ;; Verify the choice was updated
      x-val (cm/get-value (cm/get-submap (:choices trace) :x))]
  (mx/eval! x-val weight)
  (assert-close "switch update sets x=2" 2.0 (mx/item x-val) 1e-5)
  (assert-true "switch update weight finite" (js/isFinite (mx/item weight))))

;; ---------------------------------------------------------------------------
;; 1.5 Vectorized update/regenerate
;; ---------------------------------------------------------------------------

(println "\n-- Vectorized regenerate: changes selected addresses --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :y (dist/gaussian 0 1)))
      n 50
      key (rng/fresh-key)
      vtrace (dyn/vsimulate model [] n key)
      ;; Get original x values
      old-x (cm/get-value (cm/get-submap (:choices vtrace) :x))
      old-y (cm/get-value (cm/get-submap (:choices vtrace) :y))
      _ (mx/eval! old-x old-y)
      ;; Regenerate only :x
      {:keys [vtrace]} (dyn/vregenerate model vtrace (sel/select :x) key)
      new-x (cm/get-value (cm/get-submap (:choices vtrace) :x))
      new-y (cm/get-value (cm/get-submap (:choices vtrace) :y))
      _ (mx/eval! new-x new-y)]
  ;; x should have changed (overwhelmingly likely with 50 independent samples)
  (assert-true "vregenerate changed :x"
    (not= (mx/->clj old-x) (mx/->clj new-x)))
  ;; y should be unchanged
  (assert-true "vregenerate kept :y"
    (= (mx/->clj old-y) (mx/->clj new-y))))

;; ---------------------------------------------------------------------------
;; 1.4 Mask combinator
;; ---------------------------------------------------------------------------

(println "\n-- Mask combinator: active vs inactive --")
(let [inner (gen [] (let [x (dyn/trace :x (dist/gaussian 5 0.1))] (mx/eval! x) (mx/item x)))
      masked (comb/mask-combinator inner)
      active-trace (p/simulate masked [true])
      inactive-trace (p/simulate masked [false])]
  ;; Active: retval near 5, choices populated, score nonzero
  (assert-true "active retval near 5" (< (js/Math.abs (- (:retval active-trace) 5)) 2))
  (assert-true "active has :x choice"
    (cm/has-value? (cm/get-submap (:choices active-trace) :x)))
  (mx/eval! (:score active-trace))
  (assert-true "active score is finite" (js/isFinite (mx/item (:score active-trace))))
  ;; Inactive: nil retval, empty choices, score = 0
  (assert-true "inactive retval nil" (nil? (:retval inactive-trace)))
  (assert-true "inactive choices empty"
    (not (cm/has-value? (cm/get-submap (:choices inactive-trace) :x))))
  (mx/eval! (:score inactive-trace))
  (assert-close "inactive score = 0" 0.0 (mx/item (:score inactive-trace)) 1e-10))

;; ---------------------------------------------------------------------------
;; Gibbs sampling — verify posterior concentration
;; ---------------------------------------------------------------------------

(println "\n-- Gibbs: posterior on binary latent --")
(let [model (gen []
              (let [z (dyn/trace :z (dist/bernoulli 0.5))]
                (mx/eval! z)
                (let [z-val (mx/item z)
                      mu (if (> z-val 0.5) 5 -5)]
                  (dyn/trace :x (dist/gaussian mu 1))
                  z-val)))
      ;; x=4.5 is much more likely under z=1 (mu=5) than z=0 (mu=-5)
      observations (cm/choicemap :x (mx/scalar 4.5))
      traces (mcmc/gibbs
               {:samples 100 :burn 50}
               model [] observations
               [{:addr :z :support [(mx/scalar 0.0) (mx/scalar 1.0)]}])
      z-vals (mapv (fn [t]
                      (mx/realize (cm/get-value (cm/get-submap (:choices t) :z))))
                    traces)
      z-mean (/ (reduce + z-vals) (count z-vals))]
  (assert-true "gibbs: z concentrates near 1 (>0.8)" (> z-mean 0.8)))

;; ---------------------------------------------------------------------------
;; Mixture distribution — verify log-prob correctness
;; ---------------------------------------------------------------------------

(println "\n-- Mixture distribution: log-prob is logsumexp --")
(let [c1 (dist/gaussian -5 1)
      c2 (dist/gaussian 5 1)
      mix (dc/mixture [c1 c2] (mx/array [(js/Math.log 0.3) (js/Math.log 0.7)]))
      ;; At v=5, the mixture lp should be close to log(0.7 * N(5;5,1))
      ;; because the c1 component contributes negligibly
      v (mx/scalar 5.0)
      lp (dc/dist-log-prob mix v)
      _ (mx/eval! lp)
      ;; Manual: log(0.3 * N(5;-5,1) + 0.7 * N(5;5,1))
      ;; N(5;5,1) = 1/sqrt(2pi) ≈ 0.3989, N(5;-5,1) ≈ 0 (10 sigma away)
      ;; ≈ log(0.7 * 0.3989) = log(0.2792)
      expected (js/Math.log (* 0.7 (/ 1.0 (js/Math.sqrt (* 2 js/Math.PI)))))]
  (assert-close "mixture lp at v=5" expected (mx/item lp) 0.01))

(println "\nAll Phase A tests complete.")
