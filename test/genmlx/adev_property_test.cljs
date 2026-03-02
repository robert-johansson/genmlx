(ns genmlx.adev-property-test
  "Property-based tests for ADEV (Automatic Differentiation of Expected Values)
   gradient estimation using test.check."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.adev :as adev])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- report-result [name result]
  (if (:pass? result)
    (do (vswap! pass-count inc)
        (println "  PASS:" name (str "(" (:num-tests result) " trials)")))
    (do (vswap! fail-count inc)
        (println "  FAIL:" name)
        (println "    seed:" (:seed result))
        (when-let [s (get-in result [:shrunk :smallest])]
          (println "    shrunk:" s)))))

(defn- check [name prop & {:keys [num-tests] :or {num-tests 50}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

;; ---------------------------------------------------------------------------
;; Model: mixed reparam + non-reparam
;; ---------------------------------------------------------------------------

(def mixed-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))    ;; reparam
            b (trace :b (dist/bernoulli 0.5))]   ;; non-reparam (REINFORCE)
        (mx/eval! x b)
        (mx/item x)))))

;; Pure gaussian model (all reparam)
(def gauss-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/eval! x y)
        (+ (mx/item x) (mx/item y))))))

;; Cost function: negative score (optimize model score)
(def score-cost (fn [trace] (mx/negative (:score trace))))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))
(def gen-n-samples (gen/elements [5 10]))
(def gen-iterations (gen/elements [5 10]))

(println "\n=== ADEV Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Reparam detection
;; ---------------------------------------------------------------------------

(println "-- reparam detection --")

;; Property 1: has-reparam?: gaussian/uniform=true, bernoulli/categorical=false
(check "has-reparam?: correct for known distributions"
  (prop/for-all [_ (gen/return nil)]
    (and
      ;; Reparameterizable
      (adev/has-reparam? (dist/gaussian 0 1))
      (adev/has-reparam? (dist/uniform 0 1))
      ;; Not reparameterizable
      (not (adev/has-reparam? (dist/bernoulli 0.5)))
      (not (adev/has-reparam? (dist/categorical [0.3 0.7]))))))

;; ---------------------------------------------------------------------------
;; ADEV execution
;; ---------------------------------------------------------------------------

(println "\n-- adev-execute --")

;; Property 2: adev-execute: trace score is finite
(check "adev-execute: trace score finite"
  (prop/for-all [k gen-key]
    (let [result (adev/adev-execute mixed-model [] k)
          score (eval-weight (:score (:trace result)))]
      (finite? score))))

;; Property 3: adev-execute: reinforce-lp is finite
(check "adev-execute: reinforce-lp finite"
  (prop/for-all [k gen-key]
    (let [result (adev/adev-execute mixed-model [] k)
          rlp (eval-weight (:reinforce-lp result))]
      (finite? rlp))))

;; ---------------------------------------------------------------------------
;; ADEV surrogate
;; ---------------------------------------------------------------------------

(println "\n-- adev-surrogate --")

;; Property 4: adev-surrogate: returns finite scalar
(check "adev-surrogate: finite scalar"
  (prop/for-all [k gen-key]
    (let [result (adev/adev-surrogate mixed-model [] score-cost k)]
      (mx/eval! result)
      (finite? (mx/item result)))))

;; ---------------------------------------------------------------------------
;; Vectorized ADEV
;; ---------------------------------------------------------------------------

(println "\n-- vectorized ADEV --")

;; Vectorized-safe model (no mx/item or mx/eval! in body)
(def vmixed-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))    ;; reparam
            b (trace :b (dist/bernoulli 0.5))]   ;; non-reparam
        x))))

;; Property 5: vadev-execute: score shape = [N], reinforce-lp shape = [N]
(check "vadev-execute: shapes = [N]"
  (prop/for-all [n gen-n-samples
                 k gen-key]
    (let [result (adev/vadev-execute vmixed-model [] n k)]
      (mx/eval! (:score result) (:reinforce-lp result))
      (and (= [n] (mx/shape (:score result)))
           (= [n] (mx/shape (:reinforce-lp result)))))))

;; ---------------------------------------------------------------------------
;; ADEV optimization
;; ---------------------------------------------------------------------------

(println "\n-- adev-optimize --")

;; Property 6: adev-optimize: loss-history length = iterations
(check "adev-optimize: loss-history length = iterations"
  (prop/for-all [iters gen-iterations
                 k gen-key]
    (let [result (adev/adev-optimize
                   {:iterations iters :lr 0.01 :n-samples 1 :key k}
                   gauss-model [] score-cost [:x :y]
                   (mx/array [0.5 0.5]))]
      (= iters (count (:loss-history result)))))
  :num-tests 15)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== ADEV Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
