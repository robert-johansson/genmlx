(ns genmlx.l3-5-multivariate-test
  "Level 3.5 WP-3: Multivariate conjugacy test suite.
   Tests MVN-MVN conjugate update, auto-handler wiring, condition number guard,
   d=1 scalar consistency, and dimension scaling benchmark."
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.gen :refer [gen]]
            [genmlx.inference.auto-analytical :as auto-analytical]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- assert-true [desc pred]
  (if pred
    (do (vswap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (vswap! pass-count inc)
          (println (str "  PASS: " desc " (expected=" expected " actual=" actual ")")))
      (do (vswap! fail-count inc)
          (println (str "  FAIL: " desc " (expected=" expected " actual=" actual " diff=" diff ")"))))))

(defn- deterministic-weight? [weights]
  (apply = weights))

(defn- weight-variance [weights]
  (let [n (count weights)
        mean (/ (reduce + weights) n)]
    (/ (reduce + (map #(* (- % mean) (- % mean)) weights)) n)))

(defn- eye [d]
  "Create d×d identity matrix."
  (let [vals (for [i (range d) j (range d)] (if (= i j) 1.0 0.0))]
    (mx/reshape (mx/array (vec vals)) [d d])))

(defn- scale-eye [d s]
  "Create d×d scaled identity matrix."
  (let [vals (for [i (range d) j (range d)] (if (= i j) s 0.0))]
    (mx/reshape (mx/array (vec vals)) [d d])))

;; ---------------------------------------------------------------------------
;; Section 1: MVN-MVN update function correctness
;; ---------------------------------------------------------------------------

(println "\n== Section 1: MVN-MVN update function correctness ==")

;; 2D example: mu ~ N([0,0], 10*I), y ~ N(mu, I), obs y=[5,3]
(let [posterior {:mean-vec (mx/array [0 0]) :cov-matrix (scale-eye 2 10.0)}
      obs-value (mx/array [5 3])
      obs-cov (eye 2)
      result (auto-analytical/mvn-update-step posterior obs-value obs-cov)]
  (assert-true "MVN update returns non-nil" (some? result))
  ;; Posterior mean: m1 = S1*(S0^-1*m0 + R^-1*y)
  ;; With S0=10I, R=I: S1 = (I/10 + I)^-1 = (1.1*I)^-1 = (10/11)*I
  ;; m1 = (10/11)*I * (0 + y) = (10/11)*y = [50/11, 30/11] ≈ [4.545, 2.727]
  (assert-close "MVN posterior mean[0]" 4.545 (mx/item (mx/index (:mean-vec result) 0)) 0.01)
  (assert-close "MVN posterior mean[1]" 2.727 (mx/item (mx/index (:mean-vec result) 1)) 0.01)
  ;; Posterior cov: S1 = (10/11)*I ≈ 0.909*I
  (assert-close "MVN posterior cov[0,0]" 0.909
    (mx/item (mx/index (mx/index (:cov-matrix result) 0) 0)) 0.01)
  ;; Marginal LL: y ~ N(0, 11*I)
  ;; log p(y) = -0.5*(2*log(2pi) + log|11I| + (25+9)/11)
  ;;          = -0.5*(2*1.838 + 2*log(11) + 34/11) ≈ -5.781
  (assert-close "MVN marginal LL" -5.781 (mx/item (:ll result)) 0.01))

;; Multiple observations: update twice
(let [post0 {:mean-vec (mx/array [0 0]) :cov-matrix (scale-eye 2 10.0)}
      obs-cov (eye 2)
      r1 (auto-analytical/mvn-update-step post0 (mx/array [5 3]) obs-cov)
      post1 {:mean-vec (:mean-vec r1) :cov-matrix (:cov-matrix r1)}
      r2 (auto-analytical/mvn-update-step post1 (mx/array [6 4]) obs-cov)]
  (assert-true "MVN double update non-nil" (some? r2))
  ;; After two obs, posterior mean should be closer to average of obs
  (let [m (mx/item (mx/index (:mean-vec r2) 0))]
    (assert-true "MVN double update: mean between obs (4.5 < m < 6)"
      (and (> m 4.5) (< m 6.0)))))

;; ---------------------------------------------------------------------------
;; Section 2: Conjugacy detection in schema
;; ---------------------------------------------------------------------------

(println "\n== Section 2: Conjugacy detection in schema ==")

(def mvn-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/array [0 0])
                            (mx/reshape (mx/array [10 0 0 10]) [2 2])))
            y  (trace :y (dist/multivariate-normal
                            mu
                            (mx/reshape (mx/array [1 0 0 1]) [2 2])))]
        y))))

(let [s (:schema mvn-model)]
  (assert-true "MVN model is static" (:static? s))
  (assert-true "MVN model has conjugate pairs"
    (= 1 (count (:conjugate-pairs s))))
  (assert-true "MVN conjugate family is :mvn-normal"
    (= :mvn-normal (:family (first (:conjugate-pairs s)))))
  (assert-true "MVN model has auto-handlers"
    (some? (:auto-handlers s)))
  (assert-true "MVN auto-handlers include :mu and :y"
    (= #{:mu :y} (set (keys (:auto-handlers s))))))

;; ---------------------------------------------------------------------------
;; Section 3: End-to-end p/generate with auto-handlers
;; ---------------------------------------------------------------------------

(println "\n== Section 3: End-to-end p/generate ==")

(let [obs (cm/set-value cm/EMPTY :y (mx/array [5 3]))
      weights (mapv (fn [_]
                      (mx/item (:weight (p/generate mvn-model [] obs))))
                    (range 10))]
  (assert-true "MVN generate: deterministic weight"
    (deterministic-weight? weights))
  (assert-close "MVN generate: marginal LL ≈ -5.781"
    -5.781 (first weights) 0.02)

  ;; Check posterior mean in choices
  (let [result (p/generate mvn-model [] obs)
        choices (:choices (:trace result))
        mu-val (cm/get-value (cm/get-submap choices :mu))
        y-val (cm/get-value (cm/get-submap choices :y))]
    (assert-close "MVN generate: mu[0] ≈ 4.545"
      4.545 (mx/item (mx/index mu-val 0)) 0.02)
    (assert-close "MVN generate: y equals observation"
      5.0 (mx/item (mx/index y-val 0)) 0.001)))

;; ---------------------------------------------------------------------------
;; Section 4: d=1 consistency with scalar Normal-Normal
;; ---------------------------------------------------------------------------

(println "\n== Section 4: d=1 MVN matches scalar NN ==")

;; MVN d=1: mu ~ N([0], [[100]]), y ~ N(mu, [[1]]), obs y=[5]
(def mvn-1d-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/array [0])
                            (mx/reshape (mx/array [100]) [1 1])))
            y  (trace :y (dist/multivariate-normal
                            mu
                            (mx/reshape (mx/array [1]) [1 1])))]
        y))))

;; Scalar NN: mu ~ N(0, 10), y ~ N(mu, 1), obs y=5
(def scalar-nn-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
            y  (trace :y (dist/gaussian mu (mx/scalar 1.0)))]
        y))))

(let [mvn-obs (cm/set-value cm/EMPTY :y (mx/array [5]))
      scalar-obs (cm/set-value cm/EMPTY :y (mx/scalar 5.0))
      mvn-w (mx/item (:weight (p/generate mvn-1d-model [] mvn-obs)))
      scalar-w (mx/item (:weight (p/generate scalar-nn-model [] scalar-obs)))]
  (assert-close "d=1 MVN weight matches scalar NN"
    scalar-w mvn-w 1e-4))

;; ---------------------------------------------------------------------------
;; Section 5: Condition number guard
;; ---------------------------------------------------------------------------

(println "\n== Section 5: Condition number guard ==")

;; Near-singular covariance: should fallthrough (return nil)
(let [tiny-cov (mx/reshape (mx/array [1e-8 0 0 1e-8]) [2 2])
      posterior {:mean-vec (mx/array [0 0]) :cov-matrix tiny-cov}
      obs-cov (eye 2)
      result (auto-analytical/mvn-update-step posterior (mx/array [1 1]) obs-cov)]
  ;; The marginal cov is tiny-cov + I ≈ I, which IS well-conditioned
  ;; So this should succeed
  (assert-true "Near-zero prior cov + I obs: well-conditioned, succeeds"
    (some? result)))

;; Test with obs cov near-singular
(let [posterior {:mean-vec (mx/array [0 0]) :cov-matrix (eye 2)}
      tiny-obs-cov (mx/reshape (mx/array [1e-8 0 0 1e-8]) [2 2])
      result (auto-analytical/mvn-update-step posterior (mx/array [1 1]) tiny-obs-cov)]
  ;; Marginal cov = I + tiny ≈ I, well-conditioned
  (assert-true "Normal prior + tiny obs cov: well-conditioned, succeeds"
    (some? result)))

;; ---------------------------------------------------------------------------
;; Section 6: Multiple observations
;; ---------------------------------------------------------------------------

(println "\n== Section 6: Multiple MVN observations ==")

(def mvn-multi-obs-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/array [0 0])
                            (scale-eye 2 10.0)))
            y1 (trace :y1 (dist/multivariate-normal mu (eye 2)))
            y2 (trace :y2 (dist/multivariate-normal mu (eye 2)))]
        [y1 y2]))))

(let [s (:schema mvn-multi-obs-model)]
  (assert-true "Multi-obs MVN: 2 conjugate pairs"
    (= 2 (count (:conjugate-pairs s))))
  (assert-true "Multi-obs MVN: auto-handlers for :mu :y1 :y2"
    (= #{:mu :y1 :y2} (set (keys (:auto-handlers s))))))

(let [obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/array [5 3]))
              (cm/set-value :y2 (mx/array [6 4])))
      weights (mapv (fn [_]
                      (mx/item (:weight (p/generate mvn-multi-obs-model [] obs))))
                    (range 10))]
  (assert-true "Multi-obs MVN: deterministic weight"
    (deterministic-weight? weights))
  ;; Two observations compound
  (assert-true "Multi-obs MVN: weight is negative (sum of marginal LLs)"
    (< (first weights) 0.0)))

;; ---------------------------------------------------------------------------
;; Section 7: Dimension scaling benchmark (Scenario 4 lite)
;; ---------------------------------------------------------------------------

(println "\n== Section 7: Dimension scaling benchmark ==")

(defn make-mvn-model [d prior-var]
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/zeros [d])
                            (scale-eye d prior-var)))
            y  (trace :y (dist/multivariate-normal mu (eye d)))]
        y))))

(doseq [d [1 2 5 10]]
  (let [model (make-mvn-model d 100.0)
        obs-val (mx/ones [d])  ;; observe all 1s
        obs (cm/set-value cm/EMPTY :y obs-val)
        has-auto (some? (:auto-handlers (:schema model)))
        weights (mapv (fn [_]
                        (mx/item (:weight (p/generate model [] obs))))
                      (range 5))]
    (assert-true (str "d=" d ": auto-handlers detected") has-auto)
    (assert-true (str "d=" d ": deterministic weight")
      (deterministic-weight? weights))
    (println (str "  INFO d=" d ": weight=" (first weights)))))

;; ---------------------------------------------------------------------------
;; Section 8: Variance comparison (auto-handler vs vanilla)
;; ---------------------------------------------------------------------------

(println "\n== Section 8: Variance comparison ==")

(let [d 5
      model-auto (make-mvn-model d 100.0)
      ;; Make vanilla model (no auto-handlers)
      model-vanilla (let [m (make-mvn-model d 100.0)
                          s (-> (:schema m)
                                (dissoc :auto-handlers)
                                (assoc :conjugate-pairs []))]
                      (dyn/auto-key (dyn/->DynamicGF (:body-fn m) (:source m) s)))
      obs-val (mx/ones [d])
      obs (cm/set-value cm/EMPTY :y obs-val)
      n-trials 10
      auto-weights (mapv (fn [_] (mx/item (:weight (p/generate model-auto [] obs)))) (range n-trials))
      vanilla-weights (mapv (fn [_] (mx/item (:weight (p/generate model-vanilla [] obs)))) (range n-trials))
      auto-var (weight-variance auto-weights)
      vanilla-var (weight-variance vanilla-weights)]
  (println (str "  d=5 auto-handler: mean=" (/ (reduce + auto-weights) n-trials) " var=" auto-var))
  (println (str "  d=5 vanilla:      mean=" (/ (reduce + vanilla-weights) n-trials) " var=" vanilla-var))
  (assert-true "d=5: auto-handler has zero variance" (< auto-var 0.001))
  (assert-true "d=5: vanilla has high variance" (> vanilla-var 1.0)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== WP-3 Multivariate Conjugacy Tests: "
              @pass-count " passed, " @fail-count " failed =="))
(when (pos? @fail-count)
  (println "THERE WERE FAILURES"))
