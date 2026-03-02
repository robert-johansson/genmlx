(ns genmlx.serialize-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.serialize :as ser])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private fs (js/require "fs"))

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

(println "\n=== Serialization Tests ===")

;; ---------------------------------------------------------------------------
;; MLX scalar round-trip
;; ---------------------------------------------------------------------------

(println "\n-- MLX scalar round-trip --")
(let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
      tr1 (p/simulate model [])
      json (ser/save-choices tr1)
      choices (ser/load-choices json)
      v-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
      v-loaded (mx/item (cm/get-value (cm/get-submap choices :x)))]
  (assert-close "scalar round-trip float32" v-orig v-loaded 1e-6))

;; ---------------------------------------------------------------------------
;; MLX int32 scalar round-trip
;; ---------------------------------------------------------------------------

(println "\n-- MLX int32 round-trip --")
(let [model (dyn/auto-key (gen [] (trace :k (dist/bernoulli 0.5))))
      tr1 (p/simulate model [])
      json (ser/save-choices tr1)
      choices (ser/load-choices json)
      v-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :k)))
      v-loaded (mx/item (cm/get-value (cm/get-submap choices :k)))]
  (assert-close "int round-trip" v-orig v-loaded 1e-6))

;; ---------------------------------------------------------------------------
;; Nested ChoiceMap round-trip
;; ---------------------------------------------------------------------------

(println "\n-- Nested ChoiceMap round-trip --")
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
  (assert-close "slope round-trip" slope-orig slope-loaded 1e-6)
  (assert-close "intercept round-trip" intercept-orig intercept-loaded 1e-6))

;; ---------------------------------------------------------------------------
;; save-choices / load-choices / reconstruct-trace end-to-end
;; ---------------------------------------------------------------------------

(println "\n-- reconstruct-trace end-to-end --")
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
  (assert-close "reconstructed x matches" x-orig x-recon 1e-6)
  (assert-close "reconstructed score matches" score-orig score-recon 1e-5))

;; ---------------------------------------------------------------------------
;; save-trace / load-trace round-trip
;; ---------------------------------------------------------------------------

(println "\n-- save-trace / load-trace --")
(let [model (dyn/auto-key
              (gen [mu sigma]
                (let [x (trace :x (dist/gaussian mu sigma))]
                  x)))
      tr1 (p/simulate model [3.0 2.0])
      json (ser/save-trace tr1 :gen-fn-id "test-model")
      tr2 (ser/load-trace model json)
      x-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
      x-loaded (mx/item (cm/get-value (cm/get-submap (:choices tr2) :x)))]
  (assert-close "trace round-trip x" x-orig x-loaded 1e-6)
  ;; Verify gen-fn-id is in the JSON
  (let [parsed (js->clj (js/JSON.parse json) :keywordize-keys true)]
    (assert-true "gen-fn-id present" (= "test-model" (:gen-fn-id parsed)))))

;; ---------------------------------------------------------------------------
;; Version check
;; ---------------------------------------------------------------------------

(println "\n-- Version check --")
(let [bad-json (js/JSON.stringify (clj->js {:version 99 :format "bad" :choices {}}))]
  (assert-true "wrong version throws"
               (try (ser/load-choices bad-json)
                    false
                    (catch :default e
                      (= 99 (:got (ex-data e)))))))

;; ---------------------------------------------------------------------------
;; File I/O
;; ---------------------------------------------------------------------------

(println "\n-- File I/O --")
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
    (assert-close "file I/O round-trip" x-orig x-loaded 1e-6))
  ;; Clean up
  (.unlinkSync fs path))

;; Full trace file I/O
(println "\n-- Full trace file I/O --")
(let [model (dyn/auto-key
              (gen [mu]
                (trace :x (dist/gaussian mu 1))))
      tr1 (p/simulate model [2.0])
      path "/tmp/genmlx_trace_test.json"]
  (ser/save-trace-to-file tr1 path)
  (let [tr2 (ser/load-trace-from-file model path)
        x-orig (mx/item (cm/get-value (cm/get-submap (:choices tr1) :x)))
        x-loaded (mx/item (cm/get-value (cm/get-submap (:choices tr2) :x)))]
    (assert-close "trace file I/O round-trip" x-orig x-loaded 1e-6))
  (.unlinkSync fs path))

;; ---------------------------------------------------------------------------
;; Empty choicemap
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; MLX array (non-scalar) round-trip
;; ---------------------------------------------------------------------------

(println "\n-- MLX array round-trip --")
(let [model (dyn/auto-key
              (gen []
                (trace :mv (dist/multivariate-normal [0 0] [[1 0] [0 1]]))))
      tr1 (p/simulate model [])
      json (ser/save-choices tr1)
      choices (ser/load-choices json)
      v-orig (mx/->clj (cm/get-value (cm/get-submap (:choices tr1) :mv)))
      v-loaded (mx/->clj (cm/get-value (cm/get-submap choices :mv)))]
  (assert-true "array shape preserved" (= 2 (count v-loaded)))
  (assert-close "array element 0" (nth v-orig 0) (nth v-loaded 0) 1e-5)
  (assert-close "array element 1" (nth v-orig 1) (nth v-loaded 1) 1e-5))

;; ---------------------------------------------------------------------------
;; Empty choicemap
;; ---------------------------------------------------------------------------

(println "\n-- Empty choicemap --")
(let [model (dyn/auto-key (gen [] (mx/scalar 42.0)))
      tr1 (p/simulate model [])
      json (ser/save-choices tr1)
      choices (ser/load-choices json)]
  (assert-true "empty choices round-trips" (= {} (:m choices))))

;; ---------------------------------------------------------------------------
;; reconstruct-trace-from-file
;; ---------------------------------------------------------------------------

(println "\n-- reconstruct-trace-from-file --")
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
    (assert-close "reconstruct from file x" x-orig x-recon 1e-6)
    (assert-close "reconstruct from file y" y-orig y-recon 1e-6))
  (.unlinkSync fs path))

(println "\n=== Serialization Tests Complete ===")
