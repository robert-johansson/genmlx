(ns genmlx.tensor-trace-test
  "Level 2 WP-0 tests: TensorTrace, TensorChoiceMap, pack/unpack, make-tensor-score.

   Tests cover:
   1. TensorChoiceMap — IChoiceMap protocol implementation
   2. TensorTrace — construction, field access, choices access
   3. make-addr-index / make-latent-addr-index
   4. pack/unpack round-trip
   5. tensor-trace->trace / trace->tensor-trace conversion
   6. make-tensor-score — correctness against handler assess
   7. make-tensor-score-with-index — returns index + score-fn
   8. make-tensor-score-fn — tensor-native vs GFI fallback
   9. Multiple distribution types (gaussian, uniform, bernoulli, etc.)
   10. Edge cases (non-static models, no observations, all observed)"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled-ops :as compiled]
            [genmlx.tensor-trace :as tt]
            [genmlx.inference.util :as iu]))

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

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toFixed diff 6) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " expected=" expected " actual=" actual " diff=" diff))))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " expected=" expected " actual=" actual)))))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def linear-model
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))
          y-pred (mx/add (mx/multiply slope (mx/ensure-array x)) intercept)]
      (trace :y (dist/gaussian y-pred 1))
      slope)))

(def five-site-model
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 1))
          c (trace :c (dist/gaussian 0 2))
          d (trace :d (dist/gaussian (mx/add b c) 0.5))
          e (trace :e (dist/gaussian d 0.1))]
      e)))

(def multi-dist-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))
          rate (trace :rate (dist/exponential 1))
          u (trace :u (dist/uniform 0 1))]
      x)))

;; ---------------------------------------------------------------------------
;; 1. TensorChoiceMap tests
;; ---------------------------------------------------------------------------

(println "\n== TensorChoiceMap ==")

(let [vals (mx/array [3.5 -2.0 5.0])
      ai {:slope 0 :intercept 1 :y 2}
      tcm (tt/->TensorChoiceMap vals ai)]
  (assert-true "TensorChoiceMap is not a leaf" (not (cm/has-value? tcm)))
  (assert-true "get-submap :slope returns Value"
               (cm/has-value? (cm/get-submap tcm :slope)))
  (assert-close "slope value is 3.5" 3.5
                (mx/item (cm/get-value (cm/get-submap tcm :slope))) 1e-6)
  (assert-close "intercept value is -2.0" -2.0
                (mx/item (cm/get-value (cm/get-submap tcm :intercept))) 1e-6)
  (assert-close "y value is 5.0" 5.0
                (mx/item (cm/get-value (cm/get-submap tcm :y))) 1e-6)
  (assert-equal "submaps count is 3" 3 (count (cm/-submaps tcm)))
  ;; Missing address returns EMPTY (matches choicemap.cljs contract)
  (assert-true "get-submap :missing returns EMPTY from -get-submap"
               (= cm/EMPTY (cm/-get-submap tcm :missing))))

;; ---------------------------------------------------------------------------
;; 2. make-addr-index tests
;; ---------------------------------------------------------------------------

(println "\n== make-addr-index ==")

(let [schema (:schema linear-model)
      ai (tt/make-addr-index schema)]
  (assert-equal "linear-model has 3 entries" 3 (count ai))
  (assert-equal "slope is index 0" 0 (:slope ai))
  (assert-equal "intercept is index 1" 1 (:intercept ai))
  (assert-equal "y is index 2" 2 (:y ai)))

(let [schema (:schema five-site-model)
      ai (tt/make-addr-index schema)]
  (assert-equal "five-site has 5 entries" 5 (count ai)))

;; ---------------------------------------------------------------------------
;; 3. make-latent-addr-index tests
;; ---------------------------------------------------------------------------

(println "\n== make-latent-addr-index ==")

(let [schema (:schema linear-model)
      obs (cm/choicemap :y (mx/scalar 1.0))
      lai (tt/make-latent-addr-index schema obs)]
  (assert-equal "latent index has 2 entries" 2 (count lai))
  (assert-equal "slope is latent index 0" 0 (:slope lai))
  (assert-equal "intercept is latent index 1" 1 (:intercept lai))
  (assert-true ":y excluded" (nil? (:y lai))))

(let [schema (:schema five-site-model)
      obs (cm/choicemap :e (mx/scalar 0.5))
      lai (tt/make-latent-addr-index schema obs)]
  (assert-equal "5-site with 1 obs has 4 latent" 4 (count lai)))

;; ---------------------------------------------------------------------------
;; 4. Pack/Unpack round-trip tests
;; ---------------------------------------------------------------------------

(println "\n== pack/unpack ==")

(let [ai {:slope 0 :intercept 1 :y 2}
      vm {:slope (mx/scalar 3.5) :intercept (mx/scalar -2.0) :y (mx/scalar 5.0)}
      packed (tt/pack-values vm ai)]
  (assert-equal "packed shape is [3]" [3] (mx/shape packed))
  (let [unpacked (tt/unpack-values packed ai)]
    (assert-close "round-trip slope" 3.5 (mx/item (:slope unpacked)) 1e-6)
    (assert-close "round-trip intercept" -2.0 (mx/item (:intercept unpacked)) 1e-6)
    (assert-close "round-trip y" 5.0 (mx/item (:y unpacked)) 1e-6)))

;; ---------------------------------------------------------------------------
;; 5. TensorTrace construction and field access
;; ---------------------------------------------------------------------------

(println "\n== TensorTrace ==")

(let [ai {:slope 0 :intercept 1}
      vals (mx/array [3.5 -2.0])
      t (tt/make-tensor-trace {:gen-fn linear-model :args [2.0]
                                :values vals :addr-index ai
                                :score (mx/scalar -5.0) :retval (mx/scalar 3.5)})]
  (assert-true "TensorTrace has :gen-fn" (some? (:gen-fn t)))
  (assert-equal "args preserved" [2.0] (:args t))
  (assert-close "score preserved" -5.0 (mx/item (:score t)) 1e-6)
  (assert-close "retval preserved" 3.5 (mx/item (:retval t)) 1e-6)
  (assert-equal "values shape" [2] (mx/shape (:values t)))
  (assert-equal "addr-index" {:slope 0 :intercept 1} (:addr-index t))
  ;; choices is a TensorChoiceMap
  (assert-true "choices is TensorChoiceMap"
               (instance? tt/TensorChoiceMap (:choices t)))
  (assert-close "choices :slope" 3.5
                (mx/item (cm/get-value (cm/get-submap (:choices t) :slope))) 1e-6))

;; ---------------------------------------------------------------------------
;; 6. trace->tensor-trace and tensor-trace->trace
;; ---------------------------------------------------------------------------

(println "\n== trace conversion ==")

(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [2.0])
      ai (tt/make-addr-index (:schema linear-model))
      ;; Convert to TensorTrace
      t-trace (tt/trace->tensor-trace trace ai)]
  (assert-true "trace->tensor produces TensorTrace"
               (instance? tt/TensorTrace t-trace))
  (assert-close "score preserved"
                (mx/item (:score trace)) (mx/item (:score t-trace)) 1e-6)
  ;; Convert back
  (let [back (tt/tensor-trace->trace t-trace)]
    (assert-true "tensor->trace produces Trace"
                 (instance? tr/Trace back))
    (assert-close "round-trip score"
                  (mx/item (:score trace)) (mx/item (:score back)) 1e-6)
    (assert-close "round-trip slope"
                  (mx/item (cm/get-value (cm/get-submap (:choices trace) :slope)))
                  (mx/item (cm/get-value (cm/get-submap (:choices back) :slope)))
                  1e-6)
    (assert-close "round-trip intercept"
                  (mx/item (cm/get-value (cm/get-submap (:choices trace) :intercept)))
                  (mx/item (cm/get-value (cm/get-submap (:choices back) :intercept)))
                  1e-6)))

;; ---------------------------------------------------------------------------
;; 7. make-tensor-score — correctness
;; ---------------------------------------------------------------------------

(println "\n== make-tensor-score ==")

;; Basic correctness: tensor-score matches handler score
(let [model (dyn/auto-key linear-model)
      schema (:schema linear-model)
      source (:source linear-model)]
  (dotimes [trial 5]
    (let [trace (p/simulate model [2.0])
          choices (:choices trace)
          y-val (cm/get-value (cm/get-submap choices :y))
          obs (cm/choicemap :y y-val)
          score-fn (compiled/make-tensor-score schema source [2.0] obs)
          slope-val (cm/get-value (cm/get-submap choices :slope))
          intercept-val (cm/get-value (cm/get-submap choices :intercept))
          latent-tensor (mx/stack [slope-val intercept-val])
          ts (score-fn latent-tensor)]
      (mx/eval! ts)
      (assert-close (str "trial " trial ": tensor-score matches handler")
                    (mx/item (:score trace)) (mx/item ts) 1e-4))))

;; 5-site model: no observations (all latent)
(let [model (dyn/auto-key five-site-model)
      schema (:schema five-site-model)
      source (:source five-site-model)]
  (let [trace (p/simulate model [])
        choices (:choices trace)
        score-fn (compiled/make-tensor-score schema source [] cm/EMPTY)]
    (assert-true "all-latent score-fn exists" (some? score-fn))
    (when score-fn
      (let [ai (tt/make-addr-index schema)
            vm (into {} (map (fn [[addr _]]
                               [addr (cm/get-value (cm/get-submap choices addr))])
                             ai))
            latent-tensor (tt/pack-values vm ai)
            ts (score-fn latent-tensor)]
        (mx/eval! ts)
        (assert-close "all-latent score matches"
                      (mx/item (:score trace)) (mx/item ts) 1e-4)))))

;; 5-site model with observations
(let [model (dyn/auto-key five-site-model)
      schema (:schema five-site-model)
      source (:source five-site-model)]
  (let [trace (p/simulate model [])
        choices (:choices trace)
        e-val (cm/get-value (cm/get-submap choices :e))
        obs (cm/choicemap :e e-val)
        score-fn (compiled/make-tensor-score schema source [] obs)]
    (assert-true "partial-obs score-fn exists" (some? score-fn))
    (when score-fn
      (let [obs-addrs #{:e}
            latent-addrs (vec (remove obs-addrs (:dep-order schema)))
            latent-index (into {} (map-indexed (fn [i a] [a i]) latent-addrs))
            vm (into {} (map (fn [[addr _]]
                               [addr (cm/get-value (cm/get-submap choices addr))])
                             latent-index))
            latent-tensor (tt/pack-values vm latent-index)
            ts (score-fn latent-tensor)]
        (mx/eval! ts)
        (assert-close "partial-obs score matches handler"
                      (mx/item (:score trace)) (mx/item ts) 1e-4)))))

;; ---------------------------------------------------------------------------
;; 8. make-tensor-score-with-index
;; ---------------------------------------------------------------------------

(println "\n== make-tensor-score-with-index ==")

(let [schema (:schema linear-model)
      source (:source linear-model)
      obs (cm/choicemap :y (mx/scalar 1.0))
      result (compiled/make-tensor-score-with-index schema source [2.0] obs)]
  (assert-true "returns non-nil" (some? result))
  (assert-true "has :score-fn" (fn? (:score-fn result)))
  (assert-equal "latent-index" {:slope 0 :intercept 1} (:latent-index result)))

;; ---------------------------------------------------------------------------
;; 9. make-tensor-score returns nil for non-static models
;; ---------------------------------------------------------------------------

(println "\n== non-static model returns nil ==")

(let [dynamic-model (gen [n]
                      (dotimes [i n]
                        (trace (keyword (str "x" i)) (dist/gaussian 0 1)))
                      nil)
      schema (:schema dynamic-model)
      source (:source dynamic-model)]
  (assert-true "dynamic model has-loops?" (:has-loops? schema))
  (assert-true "dynamic model not static" (not (:static? schema)))
  (assert-true "make-tensor-score returns nil for dynamic model"
               (nil? (compiled/make-tensor-score schema source [] cm/EMPTY))))

;; ---------------------------------------------------------------------------
;; 10. make-tensor-score-fn — integration (tensor-native vs fallback)
;; ---------------------------------------------------------------------------

(println "\n== make-tensor-score-fn ==")

;; Static model → tensor-native path
(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [2.0])
      y-val (cm/get-value (cm/get-submap (:choices trace) :y))
      obs (cm/choicemap :y y-val)
      result (iu/make-tensor-score-fn linear-model [2.0] obs [:slope :intercept])]
  (assert-true "tensor-native? for static model" (:tensor-native? result))
  (let [{:keys [score-fn latent-index]} result
        slope-val (cm/get-value (cm/get-submap (:choices trace) :slope))
        intercept-val (cm/get-value (cm/get-submap (:choices trace) :intercept))
        tensor (mx/stack [slope-val intercept-val])
        s (score-fn tensor)]
    (mx/eval! s)
    (assert-close "tensor-native score matches"
                  (mx/item (:score trace)) (mx/item s) 1e-4)))

;; Dynamic model → GFI fallback
(let [dynamic-model (gen [n]
                      (dotimes [i n]
                        (trace (keyword (str "x" i)) (dist/gaussian 0 1)))
                      nil)
      result (iu/make-tensor-score-fn dynamic-model [3] cm/EMPTY [:x0 :x1 :x2])]
  (assert-true "not tensor-native for dynamic model" (not (:tensor-native? result)))
  (assert-true "has score-fn" (fn? (:score-fn result)))
  (assert-true "has latent-index" (map? (:latent-index result))))

;; ---------------------------------------------------------------------------
;; 11. Multi-distribution model
;; ---------------------------------------------------------------------------

(println "\n== multi-dist tensor-score ==")

(let [model (dyn/auto-key multi-dist-model)
      schema (:schema multi-dist-model)
      source (:source multi-dist-model)]
  (dotimes [_ 3]
    (let [trace (p/simulate model [])
          choices (:choices trace)
          score-fn (compiled/make-tensor-score schema source [] cm/EMPTY)]
      (assert-true "multi-dist score-fn exists" (some? score-fn))
      (when score-fn
        (let [ai (tt/make-addr-index schema)
              vm (into {} (map (fn [[addr _]]
                                 [addr (cm/get-value (cm/get-submap choices addr))])
                               ai))
              tensor (tt/pack-values vm ai)
              ts (score-fn tensor)]
          (mx/eval! ts)
          (assert-close "multi-dist score matches"
                        (mx/item (:score trace)) (mx/item ts) 1e-4))))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed =="))
(when (pos? @fail-count)
  (println (str "FAILURES: " @fail-count)))
