(ns genmlx.tensor-trace-test
  "Level 2 WP-0 tests: TensorTrace, TensorChoiceMap, pack/unpack, make-tensor-score."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
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

(deftest tensor-choicemap-test
  (testing "TensorChoiceMap protocol implementation"
    (let [vals (mx/array [3.5 -2.0 5.0])
          ai {:slope 0 :intercept 1 :y 2}
          tcm (tt/->TensorChoiceMap vals ai)]
      (is (not (cm/has-value? tcm)) "TensorChoiceMap is not a leaf")
      (is (cm/has-value? (cm/get-submap tcm :slope)) "get-submap :slope returns Value")
      (is (h/close? 3.5 (mx/item (cm/get-value (cm/get-submap tcm :slope))) 1e-6)
          "slope value is 3.5")
      (is (h/close? -2.0 (mx/item (cm/get-value (cm/get-submap tcm :intercept))) 1e-6)
          "intercept value is -2.0")
      (is (h/close? 5.0 (mx/item (cm/get-value (cm/get-submap tcm :y))) 1e-6)
          "y value is 5.0")
      (is (= 3 (count (cm/-submaps tcm))) "submaps count is 3")
      (is (= cm/EMPTY (cm/-get-submap tcm :missing))
          "get-submap :missing returns EMPTY from -get-submap"))))

;; ---------------------------------------------------------------------------
;; 2. make-addr-index tests
;; ---------------------------------------------------------------------------

(deftest make-addr-index-test
  (testing "linear-model addr-index"
    (let [schema (:schema linear-model)
          ai (tt/make-addr-index schema)]
      (is (= 3 (count ai)) "linear-model has 3 entries")
      (is (= 0 (:slope ai)) "slope is index 0")
      (is (= 1 (:intercept ai)) "intercept is index 1")
      (is (= 2 (:y ai)) "y is index 2")))

  (testing "five-site-model addr-index"
    (let [schema (:schema five-site-model)
          ai (tt/make-addr-index schema)]
      (is (= 5 (count ai)) "five-site has 5 entries"))))

;; ---------------------------------------------------------------------------
;; 3. make-latent-addr-index tests
;; ---------------------------------------------------------------------------

(deftest make-latent-addr-index-test
  (testing "linear-model latent index"
    (let [schema (:schema linear-model)
          obs (cm/choicemap :y (mx/scalar 1.0))
          lai (tt/make-latent-addr-index schema obs)]
      (is (= 2 (count lai)) "latent index has 2 entries")
      (is (= 0 (:slope lai)) "slope is latent index 0")
      (is (= 1 (:intercept lai)) "intercept is latent index 1")
      (is (nil? (:y lai)) ":y excluded")))

  (testing "five-site-model latent index"
    (let [schema (:schema five-site-model)
          obs (cm/choicemap :e (mx/scalar 0.5))
          lai (tt/make-latent-addr-index schema obs)]
      (is (= 4 (count lai)) "5-site with 1 obs has 4 latent"))))

;; ---------------------------------------------------------------------------
;; 4. Pack/Unpack round-trip tests
;; ---------------------------------------------------------------------------

(deftest pack-unpack-test
  (testing "round-trip pack/unpack"
    (let [ai {:slope 0 :intercept 1 :y 2}
          vm {:slope (mx/scalar 3.5) :intercept (mx/scalar -2.0) :y (mx/scalar 5.0)}
          packed (tt/pack-values vm ai)]
      (is (= [3] (mx/shape packed)) "packed shape is [3]")
      (let [unpacked (tt/unpack-values packed ai)]
        (is (h/close? 3.5 (mx/item (:slope unpacked)) 1e-6) "round-trip slope")
        (is (h/close? -2.0 (mx/item (:intercept unpacked)) 1e-6) "round-trip intercept")
        (is (h/close? 5.0 (mx/item (:y unpacked)) 1e-6) "round-trip y")))))

;; ---------------------------------------------------------------------------
;; 5. TensorTrace construction and field access
;; ---------------------------------------------------------------------------

(deftest tensor-trace-construction-test
  (testing "TensorTrace field access"
    (let [ai {:slope 0 :intercept 1}
          vals (mx/array [3.5 -2.0])
          t (tt/make-tensor-trace {:gen-fn linear-model :args [2.0]
                                    :values vals :addr-index ai
                                    :score (mx/scalar -5.0) :retval (mx/scalar 3.5)})]
      (is (some? (:gen-fn t)) "TensorTrace has :gen-fn")
      (is (= [2.0] (:args t)) "args preserved")
      (is (h/close? -5.0 (mx/item (:score t)) 1e-6) "score preserved")
      (is (h/close? 3.5 (mx/item (:retval t)) 1e-6) "retval preserved")
      (is (= [2] (mx/shape (:values t))) "values shape")
      (is (= {:slope 0 :intercept 1} (:addr-index t)) "addr-index")
      (is (instance? tt/TensorChoiceMap (:choices t)) "choices is TensorChoiceMap")
      (is (h/close? 3.5 (mx/item (cm/get-value (cm/get-submap (:choices t) :slope))) 1e-6)
          "choices :slope"))))

;; ---------------------------------------------------------------------------
;; 6. trace->tensor-trace and tensor-trace->trace
;; ---------------------------------------------------------------------------

(deftest trace-conversion-test
  (testing "trace->tensor-trace and back"
    (let [model (dyn/auto-key linear-model)
          trace (p/simulate model [2.0])
          ai (tt/make-addr-index (:schema linear-model))
          t-trace (tt/trace->tensor-trace trace ai)]
      (is (instance? tt/TensorTrace t-trace) "trace->tensor produces TensorTrace")
      (is (h/close? (mx/item (:score trace)) (mx/item (:score t-trace)) 1e-6)
          "score preserved")
      (let [back (tt/tensor-trace->trace t-trace)]
        (is (instance? tr/Trace back) "tensor->trace produces Trace")
        (is (h/close? (mx/item (:score trace)) (mx/item (:score back)) 1e-6)
            "round-trip score")
        (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :slope)))
                      (mx/item (cm/get-value (cm/get-submap (:choices back) :slope)))
                      1e-6)
            "round-trip slope")
        (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :intercept)))
                      (mx/item (cm/get-value (cm/get-submap (:choices back) :intercept)))
                      1e-6)
            "round-trip intercept")))))

;; ---------------------------------------------------------------------------
;; 7. make-tensor-score — correctness
;; ---------------------------------------------------------------------------

(deftest make-tensor-score-test
  (testing "tensor-score matches handler score"
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
          (is (h/close? (mx/item (:score trace)) (mx/item ts) 1e-4)
              (str "trial " trial ": tensor-score matches handler"))))))

  (testing "all-latent score (no observations)"
    (let [model (dyn/auto-key five-site-model)
          schema (:schema five-site-model)
          source (:source five-site-model)]
      (let [trace (p/simulate model [])
            choices (:choices trace)
            score-fn (compiled/make-tensor-score schema source [] cm/EMPTY)]
        (is (some? score-fn) "all-latent score-fn exists")
        (when score-fn
          (let [ai (tt/make-addr-index schema)
                vm (into {} (map (fn [[addr _]]
                                   [addr (cm/get-value (cm/get-submap choices addr))])
                                 ai))
                latent-tensor (tt/pack-values vm ai)
                ts (score-fn latent-tensor)]
            (mx/eval! ts)
            (is (h/close? (mx/item (:score trace)) (mx/item ts) 1e-4)
                "all-latent score matches"))))))

  (testing "partial observations"
    (let [model (dyn/auto-key five-site-model)
          schema (:schema five-site-model)
          source (:source five-site-model)]
      (let [trace (p/simulate model [])
            choices (:choices trace)
            e-val (cm/get-value (cm/get-submap choices :e))
            obs (cm/choicemap :e e-val)
            score-fn (compiled/make-tensor-score schema source [] obs)]
        (is (some? score-fn) "partial-obs score-fn exists")
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
            (is (h/close? (mx/item (:score trace)) (mx/item ts) 1e-4)
                "partial-obs score matches handler")))))))

;; ---------------------------------------------------------------------------
;; 8. make-tensor-score-with-index
;; ---------------------------------------------------------------------------

(deftest make-tensor-score-with-index-test
  (testing "returns score-fn and latent-index"
    (let [schema (:schema linear-model)
          source (:source linear-model)
          obs (cm/choicemap :y (mx/scalar 1.0))
          result (compiled/make-tensor-score-with-index schema source [2.0] obs)]
      (is (some? result) "returns non-nil")
      (is (fn? (:score-fn result)) "has :score-fn")
      (is (= {:slope 0 :intercept 1} (:latent-index result)) "latent-index"))))

;; ---------------------------------------------------------------------------
;; 9. make-tensor-score returns nil for non-static models
;; ---------------------------------------------------------------------------

(deftest non-static-model-nil-test
  (testing "dynamic model returns nil"
    (let [dynamic-model (gen [n]
                          (dotimes [i n]
                            (trace (keyword (str "x" i)) (dist/gaussian 0 1)))
                          nil)
          schema (:schema dynamic-model)
          source (:source dynamic-model)]
      (is (:has-loops? schema) "dynamic model has-loops?")
      (is (not (:static? schema)) "dynamic model not static")
      (is (nil? (compiled/make-tensor-score schema source [] cm/EMPTY))
          "make-tensor-score returns nil for dynamic model"))))

;; ---------------------------------------------------------------------------
;; 10. make-tensor-score-fn — integration (tensor-native vs fallback)
;; ---------------------------------------------------------------------------

(deftest make-tensor-score-fn-test
  (testing "static model gets tensor-native path"
    (let [model (dyn/auto-key linear-model)
          trace (p/simulate model [2.0])
          y-val (cm/get-value (cm/get-submap (:choices trace) :y))
          obs (cm/choicemap :y y-val)
          result (iu/make-tensor-score-fn linear-model [2.0] obs [:slope :intercept])]
      (is (:tensor-native? result) "tensor-native? for static model")
      (let [{:keys [score-fn latent-index]} result
            slope-val (cm/get-value (cm/get-submap (:choices trace) :slope))
            intercept-val (cm/get-value (cm/get-submap (:choices trace) :intercept))
            tensor (mx/stack [slope-val intercept-val])
            s (score-fn tensor)]
        (mx/eval! s)
        (is (h/close? (mx/item (:score trace)) (mx/item s) 1e-4)
            "tensor-native score matches"))))

  (testing "dynamic model gets GFI fallback"
    (let [dynamic-model (gen [n]
                          (dotimes [i n]
                            (trace (keyword (str "x" i)) (dist/gaussian 0 1)))
                          nil)
          result (iu/make-tensor-score-fn dynamic-model [3] cm/EMPTY [:x0 :x1 :x2])]
      (is (not (:tensor-native? result)) "not tensor-native for dynamic model")
      (is (fn? (:score-fn result)) "has score-fn")
      (is (map? (:latent-index result)) "has latent-index"))))

;; ---------------------------------------------------------------------------
;; 11. Multi-distribution model
;; ---------------------------------------------------------------------------

(deftest multi-dist-tensor-score-test
  (testing "multi-distribution tensor-score"
    (let [model (dyn/auto-key multi-dist-model)
          schema (:schema multi-dist-model)
          source (:source multi-dist-model)]
      (dotimes [_ 3]
        (let [trace (p/simulate model [])
              choices (:choices trace)
              score-fn (compiled/make-tensor-score schema source [] cm/EMPTY)]
          (is (some? score-fn) "multi-dist score-fn exists")
          (when score-fn
            (let [ai (tt/make-addr-index schema)
                  vm (into {} (map (fn [[addr _]]
                                     [addr (cm/get-value (cm/get-submap choices addr))])
                                   ai))
                  tensor (tt/pack-values vm ai)
                  ts (score-fn tensor)]
              (mx/eval! ts)
              (is (h/close? (mx/item (:score trace)) (mx/item ts) 1e-4)
                  "multi-dist score matches"))))))))

(cljs.test/run-tests)
