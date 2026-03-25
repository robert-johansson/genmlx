(ns genmlx.l3-5-assess-test
  "Level 3.5 WP-1: Assess auto-handler integration tests.
   Verifies that p/assess returns joint LL for all models."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- strip-analytical
  "Remove auto-handlers from a gen-fn, forcing standard handler path."
  [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :conjugate-pairs
                            :has-conjugate? :analytical-plan)))

(defn- has-auto-handlers? [gf]
  (boolean (:auto-handlers (:schema gf))))

(defn- gaussian-log-prob
  "Log probability density of x under N(mean, variance)."
  [x mean variance]
  (* -0.5 (+ (js/Math.log (* 2 js/Math.PI))
              (js/Math.log variance)
              (/ (* (- x mean) (- x mean)) variance))))

(defn- nn-joint-ll
  "Joint LL for Normal-Normal model."
  [prior-mean prior-var obs-var mu ys]
  (+ (gaussian-log-prob mu prior-mean prior-var)
     (reduce + (map #(gaussian-log-prob % mu obs-var) ys))))

;; ---------------------------------------------------------------------------
;; Model Definitions
;; ---------------------------------------------------------------------------

(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

(def bb-model
  (gen []
    (let [p (trace :p (dist/beta-dist 2 5))]
      (trace :y1 (dist/bernoulli p))
      (trace :y2 (dist/bernoulli p))
      (trace :y3 (dist/bernoulli p))
      p)))

(def gp-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 3 2))]
      (trace :y1 (dist/poisson rate))
      (trace :y2 (dist/poisson rate))
      rate)))

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian 0 sigma))
      mu)))

(def kalman-model
  (gen []
    (let [z0 (trace :z0 (dist/gaussian 0 10))
          z1 (trace :z1 (dist/gaussian z0 1))]
      (trace :y0 (dist/gaussian z0 0.5))
      (trace :y1 (dist/gaussian z1 0.5))
      z0)))

(def no-conjugate-model
  (gen []
    (let [x (trace :x (dist/uniform 0 10))]
      (trace :y1 (dist/gaussian (mx/sin x) 1))
      x)))

(def ge-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 2 1))]
      (trace :y1 (dist/exponential rate))
      (trace :y2 (dist/exponential rate))
      rate)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest schema-detection-test
  (testing "Schema detection"
    (is (has-auto-handlers? nn-model) "NN model has auto-handlers")
    (is (has-auto-handlers? bb-model) "BB model has auto-handlers")
    (is (has-auto-handlers? gp-model) "GP model has auto-handlers")
    (is (has-auto-handlers? mixed-model) "Mixed model has auto-handlers")
    (is (has-auto-handlers? kalman-model) "Kalman model has auto-handlers")
    (is (not (has-auto-handlers? no-conjugate-model)) "No-conjugate model does NOT have auto-handlers")))

(deftest nn-assess-joint-ll-test
  (testing "NN model: assess joint LL"
    (let [model (dyn/auto-key nn-model)
          choices (-> cm/EMPTY
                      (cm/set-value :mu (mx/scalar 0.0))
                      (cm/set-value :y1 (mx/scalar 3.0))
                      (cm/set-value :y2 (mx/scalar 4.0)))
          result (p/assess model [] choices)
          assess-weight (do (mx/eval! (:weight result)) (mx/item (:weight result)))
          exact-joint (nn-joint-ll 0.0 100.0 1.0 0.0 [3.0 4.0])]
      (is (h/close? exact-joint assess-weight 1e-3) "NN assess joint LL matches exact formula"))))

(deftest nn-gfi-identity-test
  (testing "NN model: GFI identity (simulate.score == assess.weight)"
    (let [model (dyn/auto-key nn-model)
          trace (p/simulate model [])
          sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          choices (:choices trace)
          assess-result (p/assess model [] choices)
          assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
      (is (h/close? sim-score assess-weight 1e-6)
          "NN simulate.score == assess.weight (GFI identity)"))))

(deftest nn-analytical-vs-stripped-test
  (testing "NN model: analytical vs stripped both return joint LL"
    (let [model-with (dyn/auto-key nn-model)
          model-without (dyn/auto-key (strip-analytical nn-model))
          choices (-> cm/EMPTY
                      (cm/set-value :mu (mx/scalar 3.5))
                      (cm/set-value :y1 (mx/scalar 3.0))
                      (cm/set-value :y2 (mx/scalar 4.0)))
          with-weight (do (mx/eval! (:weight (p/assess model-with [] choices)))
                          (mx/item (:weight (p/assess model-with [] choices))))
          without-weight (do (mx/eval! (:weight (p/assess model-without [] choices)))
                             (mx/item (:weight (p/assess model-without [] choices))))]
      (is (h/close? with-weight without-weight 1e-6) "Both paths return same joint LL"))))

(deftest bb-gfi-identity-test
  (testing "BB model: GFI identity"
    (let [model (dyn/auto-key bb-model)
          trace (p/simulate model [])
          sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          choices (:choices trace)
          assess-result (p/assess model [] choices)
          assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
      (is (h/close? sim-score assess-weight 1e-6)
          "BB simulate.score == assess.weight (GFI identity)"))))

(deftest bb-analytical-vs-stripped-test
  (testing "BB model: analytical vs stripped both return joint LL"
    (let [model-with (dyn/auto-key bb-model)
          model-without (dyn/auto-key (strip-analytical bb-model))
          choices (-> cm/EMPTY
                      (cm/set-value :p (mx/scalar 0.3))
                      (cm/set-value :y1 (mx/scalar 1.0))
                      (cm/set-value :y2 (mx/scalar 0.0))
                      (cm/set-value :y3 (mx/scalar 1.0)))
          with-weight (do (mx/eval! (:weight (p/assess model-with [] choices)))
                          (mx/item (:weight (p/assess model-with [] choices))))
          without-weight (do (mx/eval! (:weight (p/assess model-without [] choices)))
                             (mx/item (:weight (p/assess model-without [] choices))))]
      (is (h/close? with-weight without-weight 1e-6) "BB both paths return same joint LL"))))

(deftest gp-gfi-identity-test
  (testing "GP model: GFI identity"
    (let [model (dyn/auto-key gp-model)
          trace (p/simulate model [])
          sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          choices (:choices trace)
          assess-result (p/assess model [] choices)
          assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
      (is (h/close? sim-score assess-weight 1e-6)
          "GP simulate.score == assess.weight (GFI identity)"))))

(deftest gp-analytical-vs-stripped-test
  (testing "GP model: analytical vs stripped both return joint LL"
    (let [model-with (dyn/auto-key gp-model)
          model-without (dyn/auto-key (strip-analytical gp-model))
          choices (-> cm/EMPTY
                      (cm/set-value :rate (mx/scalar 2.0))
                      (cm/set-value :y1 (mx/scalar 3.0))
                      (cm/set-value :y2 (mx/scalar 1.0)))
          with-weight (do (mx/eval! (:weight (p/assess model-with [] choices)))
                          (mx/item (:weight (p/assess model-with [] choices))))
          without-weight (do (mx/eval! (:weight (p/assess model-without [] choices)))
                             (mx/item (:weight (p/assess model-without [] choices))))]
      (is (h/close? with-weight without-weight 1e-6) "GP both paths return same joint LL"))))

(deftest kalman-gfi-identity-test
  (testing "Kalman chain: GFI identity"
    (let [model (dyn/auto-key kalman-model)
          trace (p/simulate model [])
          sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          choices (:choices trace)
          assess-result (p/assess model [] choices)
          assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
      (is (h/close? sim-score assess-weight 1e-6)
          "Kalman simulate.score == assess.weight (GFI identity)"))))

(deftest mixed-model-both-paths-test
  (testing "Mixed model: both paths return joint LL"
    (let [model (dyn/auto-key mixed-model)
          model-stripped (dyn/auto-key (strip-analytical mixed-model))
          choices (-> cm/EMPTY
                      (cm/set-value :mu (mx/scalar 2.0))
                      (cm/set-value :sigma (mx/scalar 1.5))
                      (cm/set-value :y1 (mx/scalar 3.0))
                      (cm/set-value :y2 (mx/scalar 4.0))
                      (cm/set-value :y3 (mx/scalar 0.5)))
          with-weight (do (mx/eval! (:weight (p/assess model [] choices)))
                          (mx/item (:weight (p/assess model [] choices))))
          without-weight (do (mx/eval! (:weight (p/assess model-stripped [] choices)))
                             (mx/item (:weight (p/assess model-stripped [] choices))))]
      (is (h/close? with-weight without-weight 1e-6)
          "Mixed model: auto-handler assess == standard assess (both joint)"))))

(deftest mixed-model-assess-vs-generate-test
  (testing "Mixed model: assess (joint) differs from generate (marginal)"
    (let [model (dyn/auto-key mixed-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          gen-result (p/generate model [] obs)
          gen-weight (do (mx/eval! (:weight gen-result)) (mx/item (:weight gen-result)))
          trace-choices (:choices (:trace gen-result))
          mu-val (cm/get-value (cm/get-submap trace-choices :mu))
          sigma-val (cm/get-value (cm/get-submap trace-choices :sigma))
          choices (-> obs
                      (cm/set-value :mu mu-val)
                      (cm/set-value :sigma sigma-val))
          assess-result (p/assess model [] choices)
          assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
      (is (> (js/Math.abs (- assess-weight gen-weight)) 0.01)
          "Mixed model: assess (joint) != generate (marginal)"))))

(deftest no-conjugacy-fallback-test
  (testing "No conjugacy: standard fallback"
    (let [model (dyn/auto-key no-conjugate-model)
          choices (-> cm/EMPTY
                      (cm/set-value :x (mx/scalar 3.0))
                      (cm/set-value :y1 (mx/scalar 0.5)))
          result (p/assess model [] choices)
          weight (do (mx/eval! (:weight result)) (mx/item (:weight result)))]
      (is (js/isFinite weight) "No-conjugate model assess returns finite weight")
      (let [sin3 (js/Math.sin 3.0)
            expected (+ (- (js/Math.log 10))
                        (* -0.5 (+ (js/Math.log (* 2 js/Math.PI))
                                   (let [d (- 0.5 sin3)] (* d d)))))]
        (is (h/close? expected weight 1e-3) "No-conjugate assess matches manual joint LL")))))

(deftest assess-retval-test
  (testing "Assess retval"
    (let [model (dyn/auto-key nn-model)
          choices (-> cm/EMPTY
                      (cm/set-value :mu (mx/scalar 5.0))
                      (cm/set-value :y1 (mx/scalar 3.0))
                      (cm/set-value :y2 (mx/scalar 4.0)))
          result (p/assess model [] choices)]
      (is (some? (:retval result)) "Assess returns a retval"))))

(deftest consistency-across-runs-test
  (testing "Consistency across runs"
    (let [model (dyn/auto-key nn-model)
          choices (-> cm/EMPTY
                      (cm/set-value :mu (mx/scalar 0.0))
                      (cm/set-value :y1 (mx/scalar 3.0))
                      (cm/set-value :y2 (mx/scalar 4.0)))
          w1 (mx/item (:weight (p/assess model [] choices)))
          w2 (mx/item (:weight (p/assess model [] choices)))
          w3 (mx/item (:weight (p/assess model [] choices)))]
      (is (h/close? w1 w2 1e-10) "Assess is deterministic (run 1 = run 2)")
      (is (h/close? w2 w3 1e-10) "Assess is deterministic (run 2 = run 3)"))))

(deftest joint-ll-depends-on-prior-test
  (testing "Joint LL depends on prior value"
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
      (is (> (js/Math.abs (- w1 w2)) 0.01) "Joint LL differs for mu=0 vs mu=5")
      (is (> (js/Math.abs (- w2 w3)) 0.01) "Joint LL differs for mu=5 vs mu=-10")
      (is (h/close? (nn-joint-ll 0.0 100.0 1.0 0.0 [3.0 4.0]) w1 1e-3)
          "mu=0 matches exact joint LL")
      (is (h/close? (nn-joint-ll 0.0 100.0 1.0 5.0 [3.0 4.0]) w2 1e-3)
          "mu=5 matches exact joint LL")
      (is (h/close? (nn-joint-ll 0.0 100.0 1.0 -10.0 [3.0 4.0]) w3 1e-3)
          "mu=-10 matches exact joint LL"))))

(deftest all-sites-conjugate-test
  (testing "All sites conjugate"
    (let [model (dyn/auto-key nn-model)
          choices (-> cm/EMPTY
                      (cm/set-value :mu (mx/scalar 2.0))
                      (cm/set-value :y1 (mx/scalar 3.0))
                      (cm/set-value :y2 (mx/scalar 4.0)))
          result (p/assess model [] choices)
          weight (do (mx/eval! (:weight result)) (mx/item (:weight result)))]
      (is (js/isFinite weight) "All-conjugate model: assess returns finite weight")
      (let [exact (nn-joint-ll 0.0 100.0 1.0 2.0 [3.0 4.0])]
        (is (h/close? exact weight 1e-3) "All-conjugate model: weight = exact joint LL")))))

(deftest different-observation-values-test
  (testing "Different observation values"
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
          (is (h/close? exact weight 1e-3)
              (str "NN joint LL correct for " label)))))))

(deftest ge-model-gfi-identity-test
  (testing "GE model: GFI identity"
    (let [model (dyn/auto-key ge-model)
          trace (p/simulate model [])
          sim-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          choices (:choices trace)
          assess-result (p/assess model [] choices)
          assess-weight (do (mx/eval! (:weight assess-result)) (mx/item (:weight assess-result)))]
      (is (h/close? sim-score assess-weight 1e-6)
          "GE simulate.score == assess.weight (GFI identity)"))))

(cljs.test/run-tests)
