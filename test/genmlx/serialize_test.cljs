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
            [genmlx.serialize :as ser])
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
      (ser/save-choices-to-file tr1 path :gen-fn-id "file-test")
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
      (ser/save-trace-to-file tr1 path)
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
      (ser/save-choices-to-file tr1 path)
      (let [tr2 (ser/reconstruct-trace-from-file model [0.0] path)
            x-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
            x-recon (mx/item (cm/get-value (cm/get-submap (:choices tr2) :x)))
            y-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :y)))
            y-recon (mx/item (cm/get-value (cm/get-submap (:choices tr2) :y)))]
        (is (h/close? x-orig x-recon 1e-6) "reconstruct from file x")
        (is (h/close? y-orig y-recon 1e-6) "reconstruct from file y"))
      (.unlinkSync fs path))))

(cljs.test/run-tests)
