(ns genmlx.l3-5-assess-test
  "Level 3.5 WP-1: Assess auto-handler integration tests.

   Verifies that p/assess returns joint LL (sum of all log-probs) for
   all models — conjugate and non-conjugate alike. The GFI identity:
   simulate.score == assess(choices).weight must hold.

   Run: bun run --bun nbb test/genmlx/l3_5_assess_test.cljs"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close
  ([desc expected actual tol]
   (let [diff (js/Math.abs (- expected actual))]
     (if (<= diff tol)
       (do (swap! pass-count inc)
           (println (str "  PASS: " desc " (expected=" (.toFixed expected 6)
                         " actual=" (.toFixed actual 6) " diff=" (.toFixed diff 8) ")")))
       (do (swap! fail-count inc)
           (println (str "  FAIL: " desc " (expected=" (.toFixed expected 6)
                         " actual=" (.toFixed actual 6) " diff=" (.toFixed diff 8)
                         " tol=" tol ")")))))))

(defn- strip-analytical
  "Remove auto-handlers from a gen-fn, forcing standard handler path."
  [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :conjugate-pairs
                            :has-conjugate? :analytical-plan)))

(defn- has-auto-handlers? [gf]
  (boolean (:auto-handlers (:schema gf))))

;; ---------------------------------------------------------------------------
;; Joint LL helpers
;; ---------------------------------------------------------------------------

(defn- gaussian-log-prob
  "Log probability density of x under N(mean, variance)."
  [x mean variance]
  (* -0.5 (+ (js/Math.log (* 2 js/Math.PI))
              (js/Math.log variance)
              (/ (* (- x mean) (- x mean)) variance))))

(defn- nn-joint-ll
  "Joint LL for Normal-Normal model.
   log p(mu, y1, y2) = log N(mu; prior-mean, prior-var) + sum_i log N(yi; mu, obs-var)"
  [prior-mean prior-var obs-var mu ys]
  (+ (gaussian-log-prob mu prior-mean prior-var)
     (reduce + (map #(gaussian-log-prob % mu obs-var) ys))))

;; Keep marginal LL formulas for verifying generate weight (not assess)
(defn- nn-marginal-ll
  "Exact marginal LL for Normal-Normal model."
  [prior-mean prior-var obs-var ys]
  (let [result (reduce
                 (fn [{:keys [mean var ll]} yi]
                   (let [pred-var (+ var obs-var)
                         innov (- yi mean)
                         ll-i (* -0.5 (+ (js/Math.log (* 2 js/Math.PI))
                                         (js/Math.log pred-var)
                                         (/ (* innov innov) pred-var)))
                         K (/ var pred-var)
                         new-mean (+ mean (* K innov))
                         new-var (- var (* K var))]
                     {:mean new-mean :var new-var :ll (+ ll ll-i)}))
                 {:mean prior-mean :var prior-var :ll 0.0}
                 ys)]
    (:ll result)))

;; ---------------------------------------------------------------------------
;; Model Definitions
;; ---------------------------------------------------------------------------

;; Normal-Normal: mu ~ N(0, 10), y_i ~ N(mu, 1)
(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

;; Beta-Bernoulli: p ~ Beta(2, 5), y_i ~ Bernoulli(p)
(def bb-model
  (gen []
    (let [p (trace :p (dist/beta-dist 2 5))]
      (trace :y1 (dist/bernoulli p))
      (trace :y2 (dist/bernoulli p))
      (trace :y3 (dist/bernoulli p))
      p)))

;; Gamma-Poisson: rate ~ Gamma(3, 2), y_i ~ Poisson(rate)
(def gp-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 3 2))]
      (trace :y1 (dist/poisson rate))
      (trace :y2 (dist/poisson rate))
      rate)))

;; Mixed model: one conjugate pair + one non-conjugate site
(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      ;; y3 depends on sigma (not conjugate)
      (trace :y3 (dist/gaussian 0 sigma))
      mu)))

;; Kalman chain: z0 ~ N(0,10), z1 ~ N(z0, 1), y0 ~ N(z0, 0.5), y1 ~ N(z1, 0.5)
(def kalman-model
  (gen []
    (let [z0 (trace :z0 (dist/gaussian 0 10))
          z1 (trace :z1 (dist/gaussian z0 1))]
      (trace :y0 (dist/gaussian z0 0.5))
      (trace :y1 (dist/gaussian z1 0.5))
      z0)))

;; No conjugacy: all non-conjugate
(def no-conjugate-model
  (gen []
    (let [x (trace :x (dist/uniform 0 10))]
      (trace :y1 (dist/gaussian (mx/sin x) 1))
      x)))

;; =========================================================================
;; Tests
;; =========================================================================

(println "\n===== Level 3.5 WP-1: Assess Auto-Handler Integration =====\n")

;; ---------------------------------------------------------------------------
;; Test 1: Schema detection — models have auto-handlers
;; ---------------------------------------------------------------------------

(println "\n-- 1. Schema detection --")

(assert-true "NN model has auto-handlers"
  (has-auto-handlers? nn-model))

(assert-true "BB model has auto-handlers"
  (has-auto-handlers? bb-model))

(assert-true "GP model has auto-handlers"
  (has-auto-handlers? gp-model))

(assert-true "Mixed model has auto-handlers"
  (has-auto-handlers? mixed-model))

(assert-true "Kalman model has auto-handlers"
  (has-auto-handlers? kalman-model))

(assert-true "No-conjugate model does NOT have auto-handlers"
  (not (has-auto-handlers? no-conjugate-model)))

;; ---------------------------------------------------------------------------
;; Test 2: NN model — assess returns joint LL
;; ---------------------------------------------------------------------------

(println "\n-- 2. NN model: assess joint LL --")

(let [model (dyn/auto-key nn-model)
      choices (-> cm/EMPTY
                  (cm/set-value :mu (mx/scalar 0.0))
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
      result (p/assess model [] choices)
      assess-weight (mx/item (:weight result))
      ;; Joint LL = log N(0; 0, 100) + log N(3; 0, 1) + log N(4; 0, 1)
      exact-joint (nn-joint-ll 0.0 100.0 1.0 0.0 [3.0 4.0])]
  (mx/eval! (:weight result))
  (assert-close "NN assess joint LL matches exact formula"
    exact-joint assess-weight 1e-3))

;; ---------------------------------------------------------------------------
;; Test 3: NN model — GFI identity: simulate.score == assess.weight
;; ---------------------------------------------------------------------------

(println "\n-- 3. NN model: GFI identity (simulate.score == assess.weight) --")

(let [model (dyn/auto-key nn-model)
      trace (p/simulate model [])
      sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
      choices (:choices trace)
      assess-result (p/assess model [] choices)
      assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
  (assert-close "NN simulate.score == assess.weight (GFI identity)"
    sim-score assess-weight 1e-6))

;; ---------------------------------------------------------------------------
;; Test 4: NN model — with/without auto-handlers both return joint LL
;; ---------------------------------------------------------------------------

(println "\n-- 4. NN model: analytical vs stripped both return joint LL --")

(let [model-with (dyn/auto-key nn-model)
      model-without (dyn/auto-key (strip-analytical nn-model))
      choices (-> cm/EMPTY
                  (cm/set-value :mu (mx/scalar 3.5))
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
      with-result (p/assess model-with [] choices)
      without-result (p/assess model-without [] choices)
      with-weight (do (mx/eval! (:weight with-result)) (mx/item (:weight with-result)))
      without-weight (do (mx/eval! (:weight without-result)) (mx/item (:weight without-result)))]
  (assert-close "Both paths return same joint LL"
    with-weight without-weight 1e-6)
  (println (str "    Analytical path: " (.toFixed with-weight 6)
                " Handler path: " (.toFixed without-weight 6))))

;; ---------------------------------------------------------------------------
;; Test 5: BB model — GFI identity: simulate.score == assess.weight
;; ---------------------------------------------------------------------------

(println "\n-- 5. BB model: GFI identity --")

(let [model (dyn/auto-key bb-model)
      trace (p/simulate model [])
      sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
      choices (:choices trace)
      assess-result (p/assess model [] choices)
      assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
  (assert-close "BB simulate.score == assess.weight (GFI identity)"
    sim-score assess-weight 1e-6))

;; ---------------------------------------------------------------------------
;; Test 6: BB model — with/without auto-handlers both return joint LL
;; ---------------------------------------------------------------------------

(println "\n-- 6. BB model: analytical vs stripped both return joint LL --")

(let [model-with (dyn/auto-key bb-model)
      model-without (dyn/auto-key (strip-analytical bb-model))
      choices (-> cm/EMPTY
                  (cm/set-value :p (mx/scalar 0.3))
                  (cm/set-value :y1 (mx/scalar 1.0))
                  (cm/set-value :y2 (mx/scalar 0.0))
                  (cm/set-value :y3 (mx/scalar 1.0)))
      with-result (p/assess model-with [] choices)
      without-result (p/assess model-without [] choices)
      with-weight (do (mx/eval! (:weight with-result)) (mx/item (:weight with-result)))
      without-weight (do (mx/eval! (:weight without-result)) (mx/item (:weight without-result)))]
  (assert-close "BB both paths return same joint LL"
    with-weight without-weight 1e-6))

;; ---------------------------------------------------------------------------
;; Test 7: GP model — GFI identity: simulate.score == assess.weight
;; ---------------------------------------------------------------------------

(println "\n-- 7. GP model: GFI identity --")

(let [model (dyn/auto-key gp-model)
      trace (p/simulate model [])
      sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
      choices (:choices trace)
      assess-result (p/assess model [] choices)
      assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
  (assert-close "GP simulate.score == assess.weight (GFI identity)"
    sim-score assess-weight 1e-6))

;; ---------------------------------------------------------------------------
;; Test 8: GP model — with/without auto-handlers both return joint LL
;; ---------------------------------------------------------------------------

(println "\n-- 8. GP model: analytical vs stripped both return joint LL --")

(let [model-with (dyn/auto-key gp-model)
      model-without (dyn/auto-key (strip-analytical gp-model))
      choices (-> cm/EMPTY
                  (cm/set-value :rate (mx/scalar 2.0))
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 1.0)))
      with-result (p/assess model-with [] choices)
      without-result (p/assess model-without [] choices)
      with-weight (do (mx/eval! (:weight with-result)) (mx/item (:weight with-result)))
      without-weight (do (mx/eval! (:weight without-result)) (mx/item (:weight without-result)))]
  (assert-close "GP both paths return same joint LL"
    with-weight without-weight 1e-6))

;; ---------------------------------------------------------------------------
;; Test 9: Kalman chain — GFI identity
;; ---------------------------------------------------------------------------

(println "\n-- 9. Kalman chain: GFI identity --")

(let [model (dyn/auto-key kalman-model)
      trace (p/simulate model [])
      sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
      choices (:choices trace)
      assess-result (p/assess model [] choices)
      assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
  (assert-close "Kalman simulate.score == assess.weight (GFI identity)"
    sim-score assess-weight 1e-6))

;; ---------------------------------------------------------------------------
;; Test 10: Mixed model — auto-handler and standard both return joint LL
;; ---------------------------------------------------------------------------

(println "\n-- 10. Mixed model: both paths return joint LL --")

(let [model (dyn/auto-key mixed-model)
      model-stripped (dyn/auto-key (strip-analytical mixed-model))
      choices (-> cm/EMPTY
                  (cm/set-value :mu (mx/scalar 2.0))
                  (cm/set-value :sigma (mx/scalar 1.5))
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
      with-result (p/assess model [] choices)
      without-result (p/assess model-stripped [] choices)
      with-weight (do (mx/eval! (:weight with-result)) (mx/item (:weight with-result)))
      without-weight (do (mx/eval! (:weight without-result)) (mx/item (:weight without-result)))]
  (assert-close "Mixed model: auto-handler assess == standard assess (both joint)"
    with-weight without-weight 1e-6)
  (println (str "    Auto-handler: " (.toFixed with-weight 6)
                " Standard: " (.toFixed without-weight 6))))

;; ---------------------------------------------------------------------------
;; Test 11: Mixed model — assess (joint) differs from generate (marginal)
;; ---------------------------------------------------------------------------

(println "\n-- 11. Mixed model: assess (joint) ≠ generate (marginal) --")

(let [model (dyn/auto-key mixed-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0))
              (cm/set-value :y3 (mx/scalar 0.5)))
      gen-result (p/generate model [] obs)
      gen-weight (do (mx/eval! (:weight gen-result)) (mx/item (:weight gen-result)))
      ;; Assess with all choices (mu and sigma from the trace)
      trace-choices (:choices (:trace gen-result))
      mu-val (cm/get-value (cm/get-submap trace-choices :mu))
      sigma-val (cm/get-value (cm/get-submap trace-choices :sigma))
      choices (-> obs
                  (cm/set-value :mu mu-val)
                  (cm/set-value :sigma sigma-val))
      assess-result (p/assess model [] choices)
      assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
  ;; assess = joint LL (includes prior log-probs), generate = marginal weight.
  ;; They should differ for conjugate models.
  (assert-true "Mixed model: assess (joint) ≠ generate (marginal)"
    (> (js/Math.abs (- assess-weight gen-weight)) 0.01))
  (println (str "    Assess (joint): " (.toFixed assess-weight 6)
                " Generate (marginal): " (.toFixed gen-weight 6))))

;; ---------------------------------------------------------------------------
;; Test 12: No conjugacy — fallback to standard assess
;; ---------------------------------------------------------------------------

(println "\n-- 12. No conjugacy: standard fallback --")

(let [model (dyn/auto-key no-conjugate-model)
      choices (-> cm/EMPTY
                  (cm/set-value :x (mx/scalar 3.0))
                  (cm/set-value :y1 (mx/scalar 0.5)))
      result (p/assess model [] choices)
      weight (do (mx/eval! (:weight result)) (mx/item (:weight result)))]
  (assert-true "No-conjugate model assess returns finite weight"
    (js/isFinite weight))
  ;; Joint LL = log p(x|U(0,10)) + log p(y1|N(sin(3), 1))
  (let [sin3 (js/Math.sin 3.0)
        expected (+ (- (js/Math.log 10))
                    (* -0.5 (+ (js/Math.log (* 2 js/Math.PI))
                               (let [d (- 0.5 sin3)] (* d d)))))]
    (assert-close "No-conjugate assess matches manual joint LL"
      expected weight 1e-3)))

;; ---------------------------------------------------------------------------
;; Test 13: Assess retval is correct
;; ---------------------------------------------------------------------------

(println "\n-- 13. Assess retval --")

(let [model (dyn/auto-key nn-model)
      choices (-> cm/EMPTY
                  (cm/set-value :mu (mx/scalar 5.0))
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
      result (p/assess model [] choices)]
  (assert-true "Assess returns a retval"
    (some? (:retval result))))

;; ---------------------------------------------------------------------------
;; Test 14: Consistency — multiple runs produce same result
;; ---------------------------------------------------------------------------

(println "\n-- 14. Consistency across runs --")

(let [model (dyn/auto-key nn-model)
      choices (-> cm/EMPTY
                  (cm/set-value :mu (mx/scalar 0.0))
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
      w1 (mx/item (:weight (p/assess model [] choices)))
      w2 (mx/item (:weight (p/assess model [] choices)))
      w3 (mx/item (:weight (p/assess model [] choices)))]
  (assert-close "Assess is deterministic (run 1 = run 2)" w1 w2 1e-10)
  (assert-close "Assess is deterministic (run 2 = run 3)" w2 w3 1e-10))

;; ---------------------------------------------------------------------------
;; Test 15: NN model — joint LL depends on prior value
;; ---------------------------------------------------------------------------

(println "\n-- 15. Joint LL depends on prior value --")

(let [model (dyn/auto-key nn-model)
      obs-y1 (mx/scalar 3.0)
      obs-y2 (mx/scalar 4.0)
      w1 (mx/item (:weight (p/assess model []
                    (-> cm/EMPTY (cm/set-value :mu (mx/scalar 0.0))
                        (cm/set-value :y1 obs-y1) (cm/set-value :y2 obs-y2)))))
      w2 (mx/item (:weight (p/assess model []
                    (-> cm/EMPTY (cm/set-value :mu (mx/scalar 5.0))
                        (cm/set-value :y1 obs-y1) (cm/set-value :y2 obs-y2)))))
      w3 (mx/item (:weight (p/assess model []
                    (-> cm/EMPTY (cm/set-value :mu (mx/scalar -10.0))
                        (cm/set-value :y1 obs-y1) (cm/set-value :y2 obs-y2)))))]
  ;; Joint LL depends on the prior value (different mu → different log p(mu))
  (assert-true "Joint LL differs for mu=0 vs mu=5"
    (> (js/Math.abs (- w1 w2)) 0.01))
  (assert-true "Joint LL differs for mu=5 vs mu=-10"
    (> (js/Math.abs (- w2 w3)) 0.01))
  ;; Verify each matches exact joint formula
  (assert-close "mu=0 matches exact joint LL"
    (nn-joint-ll 0.0 100.0 1.0 0.0 [3.0 4.0]) w1 1e-3)
  (assert-close "mu=5 matches exact joint LL"
    (nn-joint-ll 0.0 100.0 1.0 5.0 [3.0 4.0]) w2 1e-3)
  (assert-close "mu=-10 matches exact joint LL"
    (nn-joint-ll 0.0 100.0 1.0 -10.0 [3.0 4.0]) w3 1e-3))

;; ---------------------------------------------------------------------------
;; Test 16: All sites conjugate — joint LL matches formula
;; ---------------------------------------------------------------------------

(println "\n-- 16. All sites conjugate --")

(let [model (dyn/auto-key nn-model)
      choices (-> cm/EMPTY
                  (cm/set-value :mu (mx/scalar 2.0))
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
      result (p/assess model [] choices)
      weight (do (mx/eval! (:weight result)) (mx/item (:weight result)))]
  (assert-true "All-conjugate model: assess returns finite weight"
    (js/isFinite weight))
  (let [exact (nn-joint-ll 0.0 100.0 1.0 2.0 [3.0 4.0])]
    (assert-close "All-conjugate model: weight = exact joint LL"
      exact weight 1e-3)))

;; ---------------------------------------------------------------------------
;; Test 17: NN model with different obs values — joint LL
;; ---------------------------------------------------------------------------

(println "\n-- 17. Different observation values --")

(let [model (dyn/auto-key nn-model)]
  (doseq [[y1 y2 label] [[0.0 0.0 "y=(0,0)"]
                          [1.0 1.0 "y=(1,1)"]
                          [5.0 -5.0 "y=(5,-5)"]
                          [10.0 10.0 "y=(10,10)"]]]
    (let [mu 0.0
          choices (-> cm/EMPTY
                      (cm/set-value :mu (mx/scalar mu))
                      (cm/set-value :y1 (mx/scalar y1))
                      (cm/set-value :y2 (mx/scalar y2)))
          weight (mx/item (:weight (p/assess model [] choices)))
          exact (nn-joint-ll 0.0 100.0 1.0 mu [y1 y2])]
      (assert-close (str "NN joint LL correct for " label)
        exact weight 1e-3))))

;; ---------------------------------------------------------------------------
;; Test 18: GE (Gamma-Exponential) model — GFI identity
;; ---------------------------------------------------------------------------

(println "\n-- 18. GE model: GFI identity --")

(def ge-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 2 1))]
      (trace :y1 (dist/exponential rate))
      (trace :y2 (dist/exponential rate))
      rate)))

(let [model (dyn/auto-key ge-model)
      trace (p/simulate model [])
      sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
      choices (:choices trace)
      assess-result (p/assess model [] choices)
      assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
  (assert-close "GE simulate.score == assess.weight (GFI identity)"
    sim-score assess-weight 1e-6))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n===== Results =====")
(println (str "PASS: " @pass-count " / " (+ @pass-count @fail-count)))
(when (pos? @fail-count)
  (println (str "FAIL: " @fail-count)))
(println "===================\n")
