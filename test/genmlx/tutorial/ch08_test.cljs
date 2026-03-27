(ns genmlx.tutorial.ch08-test
  "Test file for Tutorial Chapter 8: Going Fast — Vectorization and Compilation."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

(def pass (atom 0))
(def fail (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass inc) (println "  PASS:" msg))
    (do (swap! fail inc) (println "  FAIL:" msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass inc) (println "  PASS:" msg))
      (do (swap! fail inc) (println "  FAIL:" msg (str "expected=" expected " actual=" actual))))))

;; Simple model for vectorization tests
(def simple-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :x (dist/gaussian mu 1))
      mu)))

(def obs (cm/choicemap :x (mx/scalar 5.0)))

;; ============================================================
;; Listing 8.1: vsimulate
;; ============================================================
(println "\n== Listing 8.1: vsimulate ==")

(let [model (dyn/auto-key simple-model)
      vtrace (dyn/vsimulate model [] 100 (rng/fresh-key))]
  (assert-true "vsimulate returns a trace" (some? vtrace))
  (assert-true "score is [100]-shaped"
               (= [100] (mx/shape (:score vtrace))))
  ;; :mu should be [100]-shaped
  (let [mu-arr (cm/get-value (cm/get-submap (:choices vtrace) :mu))]
    (assert-true ":mu is [100]-shaped" (= [100] (mx/shape mu-arr)))))

;; ============================================================
;; Listing 8.2: vgenerate
;; ============================================================
(println "\n== Listing 8.2: vgenerate ==")

(let [model (dyn/auto-key simple-model)
      vtrace (dyn/vgenerate model [] obs 100 (rng/fresh-key))]
  (assert-true "vgenerate returns a trace" (some? vtrace))
  (assert-true "weight is [100]-shaped"
               (= [100] (mx/shape (:weight vtrace))))
  ;; :x should be scalar (constrained), :mu should be [100]-shaped
  (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
    (assert-true ":x is constrained (scalar or broadcast)"
                 (some? x-val))))

;; ============================================================
;; Listing 8.3: ESS and log-ML from vectorized traces
;; ============================================================
(println "\n== Listing 8.3: ESS and log-ML ==")

(let [model (dyn/auto-key simple-model)
      vtrace (dyn/vgenerate model [] obs 200 (rng/fresh-key))
      ess-val (vec/vtrace-ess vtrace)
      log-ml-val (vec/vtrace-log-ml-estimate vtrace)
      ess (if (mx/array? ess-val) (mx/item ess-val) ess-val)
      log-ml (if (mx/array? log-ml-val) (mx/item log-ml-val) log-ml-val)]
  (assert-true "ESS is positive" (> ess 0))
  (assert-true "ESS <= N" (<= ess 200))
  (assert-true "log-ML is finite" (js/Number.isFinite log-ml))
  (assert-true "log-ML is negative" (< log-ml 0)))

;; ============================================================
;; Listing 8.4: Resampling
;; ============================================================
(println "\n== Listing 8.4: resampling ==")

(let [model (dyn/auto-key simple-model)
      vtrace (dyn/vgenerate model [] obs 50 (rng/fresh-key))
      resampled (vec/resample-vtrace vtrace (rng/fresh-key))]
  (assert-true "resampled has same structure" (some? (:choices resampled)))
  (assert-true "resampled score is [50]-shaped"
               (= [50] (mx/shape (:score resampled)))))

;; ============================================================
;; Listing 8.5: Schema extraction
;; ============================================================
(println "\n== Listing 8.5: schema ==")

(let [model simple-model
      schema (:schema model)]
  (assert-true "model has a schema" (some? schema))
  (assert-true "schema has :trace-sites" (some? (:trace-sites schema)))
  (let [sites (:trace-sites schema)
        addrs (mapv :addr sites)]
    (assert-true "schema knows about :mu" (some #{:mu} addrs))
    (assert-true "schema knows about :x" (some #{:x} addrs)))
  (assert-true "schema has :static? flag" (contains? schema :static?)))

;; ============================================================
;; Listing 8.6: Compiled simulate
;; ============================================================
(println "\n== Listing 8.6: compiled simulate ==")

;; A static model (all addresses known at construction time)
(def static-model
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian 0 1))]
      (mx/add a b))))

(let [model (dyn/auto-key static-model)
      schema (:schema static-model)]
  ;; Check if model is static
  (assert-true "static model is classified as static" (:static? schema))
  ;; Simulate should work (uses compiled path if available)
  (let [trace (p/simulate model [])]
    (assert-true "compiled model simulates" (some? trace))
    (assert-true "has :a" (cm/has-value? (cm/get-submap (:choices trace) :a)))
    (assert-true "has :b" (cm/has-value? (cm/get-submap (:choices trace) :b)))
    (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace))))))

;; ============================================================
;; Listing 8.7: Performance comparison (conceptual)
;; ============================================================
(println "\n== Listing 8.7: scalar vs vectorized ==")

;; Scalar: N separate simulations
(let [model (dyn/auto-key simple-model)
      t0 (js/Date.now)
      _ (dotimes [_ 20] (p/simulate model []))
      scalar-ms (- (js/Date.now) t0)
      ;; Vectorized: one call for N particles
      t1 (js/Date.now)
      _ (dyn/vsimulate model [] 20 (rng/fresh-key))
      vec-ms (- (js/Date.now) t1)]
  (assert-true "scalar runs" (>= scalar-ms 0))
  (assert-true "vectorized runs" (>= vec-ms 0))
  (println (str "  scalar 20x: " scalar-ms "ms, vectorized 20: " vec-ms "ms")))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 8 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
