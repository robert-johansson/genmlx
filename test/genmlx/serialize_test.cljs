;; @tier fast
(ns genmlx.serialize-test
  "Tests for serialization/deserialization of traces and choicemaps."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.serialize :as ser]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private fs (js/require "fs"))

(deftest scalar-round-trip
  (testing "MLX scalar round-trip"
    (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          tr1 (p/simulate model [])
          json (ser/save-choices tr1)
          choices (ser/load-choices json)
          v-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
          v-loaded (mx/item (cm/get-value (cm/get-submap choices :x)))]
      (is (h/close? v-orig v-loaded 1e-6) "scalar round-trip float32"))))

(deftest int32-round-trip
  (testing "MLX int32 round-trip"
    (let [model (dyn/auto-key (gen [] (trace :k (dist/bernoulli 0.5))))
          tr1 (p/simulate model [])
          json (ser/save-choices tr1)
          choices (ser/load-choices json)
          v-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :k)))
          v-loaded (mx/item (cm/get-value (cm/get-submap choices :k)))]
      (is (h/close? v-orig v-loaded 1e-6) "int round-trip"))))

(deftest nested-choicemap-round-trip
  (testing "Nested ChoiceMap round-trip"
    (let [model (dyn/auto-key
                  (gen []
                    (let [x (trace :slope (dist/gaussian 0 10))
                          y (trace :intercept (dist/gaussian 0 5))]
                      (trace :obs (dist/gaussian (mx/add x y) 1)))))
          tr1 (p/simulate model [])
          json (ser/save-choices tr1)
          choices (ser/load-choices json)
          slope-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :slope)))
          slope-loaded (mx/item (cm/get-value (cm/get-submap choices :slope)))
          intercept-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :intercept)))
          intercept-loaded (mx/item (cm/get-value (cm/get-submap choices :intercept)))]
      (is (h/close? slope-orig slope-loaded 1e-6) "slope round-trip")
      (is (h/close? intercept-orig intercept-loaded 1e-6) "intercept round-trip"))))

(deftest reconstruct-trace-end-to-end
  (testing "reconstruct-trace end-to-end"
    (let [model (dyn/auto-key
                  (gen [mu]
                    (let [x (trace :x (dist/gaussian mu 1))]
                      x)))
          tr1 (p/simulate model [5.0])
          json (ser/save-choices tr1)
          tr2 (ser/reconstruct-trace model [5.0] json)
          x-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
          x-recon (mx/item (cm/get-value (cm/get-submap (:choices tr2) :x)))
          score-orig (mx/item (:score tr1))
          score-recon (mx/item (:score tr2))]
      (is (h/close? x-orig x-recon 1e-6) "reconstructed x matches")
      (is (h/close? score-orig score-recon 1e-5) "reconstructed score matches"))))

(deftest save-load-trace
  (testing "save-trace / load-trace"
    (let [model (dyn/auto-key
                  (gen [mu sigma]
                    (let [x (trace :x (dist/gaussian mu sigma))]
                      x)))
          tr1 (p/simulate model [3.0 2.0])
          json (ser/save-trace tr1 :gen-fn-id "test-model")
          tr2 (ser/load-trace model json)
          x-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
          x-loaded (mx/item (cm/get-value (cm/get-submap (:choices tr2) :x)))]
      (is (h/close? x-orig x-loaded 1e-6) "trace round-trip x")
      (let [parsed (js->clj (js/JSON.parse json) :keywordize-keys true)]
        (is (= "test-model" (:gen-fn-id parsed)) "gen-fn-id present")))))

(deftest version-check
  (testing "wrong version throws"
    (let [bad-json (js/JSON.stringify (clj->js {:version 99 :format "bad" :choices {}}))]
      (is (try (ser/load-choices bad-json)
               false
               (catch :default e
                 (= 99 (:got (ex-data e)))))
          "wrong version throws"))))

(deftest file-io
  (testing "choices file I/O"
    (let [model (dyn/auto-key
                  (gen []
                    (let [x (trace :x (dist/gaussian 0 1))]
                      x)))
          tr1 (p/simulate model [])
          path "/tmp/genmlx_serialize_test.json"]
      (ser/save-choices-to-file! tr1 path :gen-fn-id "file-test")
      (let [choices (ser/load-choices-from-file path)
            x-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
            x-loaded (mx/item (cm/get-value (cm/get-submap choices :x)))]
        (is (h/close? x-orig x-loaded 1e-6) "file I/O round-trip"))
      (.unlinkSync fs path)))

  (testing "full trace file I/O"
    (let [model (dyn/auto-key
                  (gen [mu]
                    (trace :x (dist/gaussian mu 1))))
          tr1 (p/simulate model [2.0])
          path "/tmp/genmlx_trace_test.json"]
      (ser/save-trace-to-file! tr1 path)
      (let [tr2 (ser/load-trace-from-file model path)
            x-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
            x-loaded (mx/item (cm/get-value (cm/get-submap (:choices tr2) :x)))]
        (is (h/close? x-orig x-loaded 1e-6) "trace file I/O round-trip"))
      (.unlinkSync fs path))))

(deftest array-round-trip
  (testing "MLX array round-trip"
    (let [model (dyn/auto-key
                  (gen []
                    (trace :mv (dist/multivariate-normal [0 0] [[1 0] [0 1]]))))
          tr1 (p/simulate model [])
          json (ser/save-choices tr1)
          choices (ser/load-choices json)
          v-orig (mx/->clj (cm/get-value (cm/get-submap (:choices tr1) :mv)))
          v-loaded (mx/->clj (cm/get-value (cm/get-submap choices :mv)))]
      (is (= 2 (count v-loaded)) "array shape preserved")
      (is (h/close? (nth v-orig 0) (nth v-loaded 0) 1e-5) "array element 0")
      (is (h/close? (nth v-orig 1) (nth v-loaded 1) 1e-5) "array element 1"))))

(deftest empty-choicemap-round-trip
  (testing "empty choicemap"
    (let [model (dyn/auto-key (gen [] (mx/scalar 42.0)))
          tr1 (p/simulate model [])
          json (ser/save-choices tr1)
          choices (ser/load-choices json)]
      (is (= {} (:m choices)) "empty choices round-trips"))))

(deftest reconstruct-trace-from-file-test
  (testing "reconstruct-trace-from-file"
    (let [model (dyn/auto-key
                  (gen [mu]
                    (let [x (trace :x (dist/gaussian mu 1))
                          y (trace :y (dist/gaussian x 0.5))]
                      (mx/add x y))))
          tr1 (p/simulate model [0.0])
          path "/tmp/genmlx_reconstruct_test.json"]
      (ser/save-choices-to-file! tr1 path)
      (let [tr2 (ser/reconstruct-trace-from-file model [0.0] path)
            x-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
            x-recon (mx/item (cm/get-value (cm/get-submap (:choices tr2) :x)))
            y-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :y)))
            y-recon (mx/item (cm/get-value (cm/get-submap (:choices tr2) :y)))]
        (is (h/close? x-orig x-recon 1e-6) "reconstruct from file x")
        (is (h/close? y-orig y-recon 1e-6) "reconstruct from file y"))
      (.unlinkSync fs path))))

;; ---------------------------------------------------------------------------
;; Combinator traces, full dtype table, batched scores (genmlx-000i)
;; ---------------------------------------------------------------------------

(deftest map-combinator-round-trip
  (testing "Map traces save and load with integer element addresses intact"
    ;; Pre-fix: (name 0) threw on save, so no Map/Unfold/Scan trace could
    ;; be serialized at all; the load side keywordized everything to :0.
    (let [kernel (dyn/auto-key (gen [x] (trace :y (dist/gaussian x 1))))
          mapped (comb/map-combinator kernel)
          tr1 (p/simulate mapped [[1.0 2.0 3.0]])
          json (ser/save-choices tr1)
          choices (ser/load-choices json)]
      (doseq [i (range 3)]
        (let [orig (mx/item (cm/get-choice (:choices tr1) [i :y]))
              loaded (mx/item (cm/get-choice choices [i :y]))]
          (is (h/close? orig loaded 1e-6)
              (str "element " i " value survives under INTEGER address"))))
      ;; Fully-constrained generate reproduces the original score exactly
      (let [{tr2 :trace} (p/generate mapped [[1.0 2.0 3.0]] choices)]
        (is (h/close? (h/realize (:score tr1)) (h/realize (:score tr2)) 1e-5)
            "reconstructed Map trace scores identically")))))

(deftest unfold-combinator-round-trip
  (testing "Unfold traces round-trip through save/load + generate"
    (let [step (dyn/auto-key (gen [t state]
                 (trace :x (dist/gaussian state 0.5))))
          unfold (comb/unfold-combinator step)
          tr1 (p/simulate unfold [4 (mx/scalar 0.0)])
          json (ser/save-choices tr1)
          choices (ser/load-choices json)
          {tr2 :trace} (p/generate unfold [4 (mx/scalar 0.0)] choices)]
      (is (h/close? (h/realize (:score tr1)) (h/realize (:score tr2)) 1e-5)
          "reconstructed Unfold trace scores identically"))))

(deftest namespaced-keyword-address-round-trip
  (testing "namespaced keyword addresses keep their namespace"
    ;; Pre-fix the codec used (name k), which drops the namespace.
    (let [choices (cm/set-choice cm/EMPTY [:obs/y] (mx/scalar 2.5))
          loaded (ser/load-choices (ser/save-choices {:choices choices}))]
      (is (h/close? 2.5 (mx/item (cm/get-choice loaded [:obs/y])) 1e-6)
          ":obs/y survives with namespace"))))

(deftest uint32-categorical-round-trip
  (testing "uint32 leaves (categorical/token samples) serialize"
    ;; Pre-fix the dtype table only knew float32/int32 — every trace
    ;; containing a categorical sample failed with Unknown dtype code: 4.
    (let [model (dyn/auto-key
                 (gen [] (trace :k (dist/categorical
                                    (mx/array [0.2 0.3 0.5])))))
          tr1 (p/simulate model [])
          loaded (ser/load-choices (ser/save-choices tr1))
          orig (mx/item (cm/get-choice (:choices tr1) [:k]))
          got (mx/item (cm/get-choice loaded [:k]))]
      (is (= orig got) "categorical index value survives")
      (is (= (mx/dtype (cm/get-value (cm/get-submap loaded :k)))
             (mx/dtype (cm/get-value (cm/get-submap (:choices tr1) :k))))
          "uint32 dtype survives"))))

(deftest uint32-full-range-round-trip
  (testing "uint32 values above the float32 24-bit mantissa are bit-exact"
    ;; Pre-fix, ->clj read uint32 through .toFloat32 and mx/array rebuilt it
    ;; through Float32Array + astype — BOTH silently rounded values >= 2^24,
    ;; so a serialized PRNG key restored to a DIFFERENT key with the same
    ;; dtype tag: silently non-reproducible replay (genmlx-st0y).
    (let [k (first (rng/split (rng/fresh-key 424242)))
          _ (mx/eval! k)
          truth (vec (js->clj (.toUint32 k)))
          _ (is (boolean (some #(> % 16777216) truth))
                "test key exercises the >2^24 range")
          restored (ser/data->value (ser/value->data k))]
      (is (= truth (mx/->clj k)) "->clj reads uint32 exactly")
      (is (= truth (mx/->clj restored)) "value->data/data->value bit-exact")
      (is (= (mx/dtype k) (mx/dtype restored)) "dtype preserved")
      (let [[a b] (rng/split restored)]
        (is (= [2] (vec (mx/shape a)) (vec (mx/shape b)))
            "restored key is a usable PRNG key")))))

(deftest batched-score-save-trace
  (testing "save-trace handles [N]-shaped batched scores"
    ;; Pre-fix mx/realize on an [N] score threw (item needs size 1).
    (let [trace (tr/make-trace {:gen-fn nil :args []
                                :choices (cm/choicemap :x (mx/zeros [5]))
                                :retval nil
                                :score (mx/array [1.0 2.0 3.0 4.0 5.0])})
          json (ser/save-trace trace)
          data (js->clj (js/JSON.parse json) :keywordize-keys true)]
      (is (string? json) "save-trace returns JSON, does not throw")
      (is (= "array" (get-in data [:score :type])) "score saved as array data")
      (is (= [1 2 3 4 5] (mapv int (get-in data [:score :value])))
          "score values intact"))))

(deftest legacy-unprefixed-keys-load-as-keywords
  (testing "v1 files written by the old codec still load"
    (let [json (js/JSON.stringify
                (clj->js {:version 1
                          :format "genmlx-choices-v1"
                          :choices {"x" {:type "scalar" :value 1.5
                                         :dtype "float32"}}}))
          loaded (ser/load-choices json)]
      (is (h/close? 1.5 (mx/item (cm/get-choice loaded [:x])) 1e-6)
          "unprefixed legacy key decodes as keyword :x"))))

(cljs.test/run-tests)
