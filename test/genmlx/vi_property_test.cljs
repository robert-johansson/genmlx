(ns genmlx.vi-property-test
  "Property-based tests for variational inference: ADVI, compiled VI,
   programmable VI, objectives, and model-based VI using test.check."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.util :as u])
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
;; Target log-density: 2D isotropic Gaussian N(0, I)
;; ---------------------------------------------------------------------------

(def log-density
  "Simple 2D Gaussian target: log p(z) = -0.5 * sum(z^2) + const"
  (fn [z]
    (mx/negative (mx/multiply (mx/scalar 0.5) (mx/sum (mx/square z))))))

(def init-params (mx/array [1.0 1.0]))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))
(def gen-iterations (gen/elements [20 30]))
(def gen-lr (gen/elements [0.01 0.05]))
(def gen-elbo-samples (gen/elements [3 5]))

;; ---------------------------------------------------------------------------
;; Model for vi-from-model
;; ---------------------------------------------------------------------------

(def two-gauss
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/eval! x y)
        (+ (mx/item x) (mx/item y))))))

(def two-gauss-obs (cm/choicemap :y (mx/scalar 1.0)))

(println "\n=== VI Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; ADVI properties
;; ---------------------------------------------------------------------------

(println "-- ADVI --")

;; Property 1: vi: ELBO history has length ≤ iterations
(check "vi: ELBO history length ≤ iterations"
  (prop/for-all [n gen-iterations
                 k gen-key]
    (let [result (vi/vi {:iterations n :learning-rate 0.05 :elbo-samples 3 :key k}
                         log-density init-params)]
      (<= (count (:elbo-history result)) n)))
  :num-tests 20)

;; Property 2: vi: mu is finite
(check "vi: mu is finite"
  (prop/for-all [k gen-key]
    (let [result (vi/vi {:iterations 30 :learning-rate 0.05 :elbo-samples 3 :key k}
                         log-density init-params)
          mu (:mu result)]
      (mx/eval! mu)
      (every? js/isFinite (mx/->clj mu))))
  :num-tests 20)

;; Property 3: vi: sigma > 0 (exp of log-sigma)
(check "vi: sigma > 0"
  (prop/for-all [k gen-key]
    (let [result (vi/vi {:iterations 30 :learning-rate 0.05 :elbo-samples 3 :key k}
                         log-density init-params)
          sigma (:sigma result)]
      (mx/eval! sigma)
      (every? #(> % 0) (mx/->clj sigma))))
  :num-tests 20)

;; Property 4: vi: sample-fn(n) produces n samples
(check "vi: sample-fn produces n samples"
  (prop/for-all [k gen-key]
    (let [result (vi/vi {:iterations 20 :learning-rate 0.05 :elbo-samples 3 :key k}
                         log-density init-params)
          samples ((:sample-fn result) 15)]
      (= 15 (count samples))))
  :num-tests 20)

;; ---------------------------------------------------------------------------
;; Compiled VI properties
;; ---------------------------------------------------------------------------

(println "\n-- compiled VI --")

;; Property 5: compiled-vi: mu is finite
(check "compiled-vi: mu is finite"
  (prop/for-all [k gen-key]
    (let [result (vi/compiled-vi {:iterations 30 :learning-rate 0.05
                                   :elbo-samples 3 :key k :device :cpu}
                                  log-density init-params)
          mu (:mu result)]
      (mx/eval! mu)
      (every? js/isFinite (mx/->clj mu))))
  :num-tests 15)

;; Property 6: compiled-vi: sigma > 0
(check "compiled-vi: sigma > 0"
  (prop/for-all [k gen-key]
    (let [result (vi/compiled-vi {:iterations 30 :learning-rate 0.05
                                   :elbo-samples 3 :key k :device :cpu}
                                  log-density init-params)
          sigma (:sigma result)]
      (mx/eval! sigma)
      (every? #(> % 0) (mx/->clj sigma))))
  :num-tests 15)

;; ---------------------------------------------------------------------------
;; Objective functions
;; ---------------------------------------------------------------------------

(println "\n-- VI objectives --")

(def log-p-fn
  "Model log-density for testing objectives."
  (fn [z] (mx/negative (mx/multiply (mx/scalar 0.5) (mx/sum (mx/square z))))))

(def log-q-fn
  "Guide log-density for testing objectives."
  (fn [z] (mx/negative (mx/multiply (mx/scalar 0.5) (mx/sum (mx/square z))))))

;; Property 7: elbo-objective: returns finite value
(check "elbo-objective: returns finite value"
  (prop/for-all [k gen-key]
    (let [obj-fn (vi/elbo-objective log-p-fn log-q-fn)
          samples (rng/normal k [5 2])
          result (obj-fn samples)]
      (mx/eval! result)
      (finite? (mx/item result))))
  :num-tests 30)

;; Property 8: iwelbo-objective: returns finite value
(check "iwelbo-objective: returns finite value"
  (prop/for-all [k gen-key]
    (let [obj-fn (vi/iwelbo-objective log-p-fn log-q-fn)
          samples (rng/normal k [5 2])
          result (obj-fn samples)]
      (mx/eval! result)
      (finite? (mx/item result))))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; Programmable VI
;; ---------------------------------------------------------------------------

(println "\n-- programmable VI --")

;; Parameterized guide
(defn my-log-q [z params]
  (let [mu params]
    ;; log N(z; mu, I) = -0.5 * sum((z - mu)^2) + const
    (mx/negative (mx/multiply (mx/scalar 0.5)
                               (mx/sum (mx/square (mx/subtract z mu)))))))

(defn my-sample-fn [params key n]
  (let [d (first (mx/shape params))
        eps (rng/normal (rng/ensure-key key) [n d])]
    (mx/add params eps)))

;; Property 9: programmable-vi: loss history length ≤ iterations
(check "programmable-vi: loss history length ≤ iterations"
  (prop/for-all [k gen-key]
    (let [result (vi/programmable-vi
                   {:iterations 20 :learning-rate 0.05 :n-samples 5
                    :objective :elbo :key k}
                   log-p-fn my-log-q my-sample-fn (mx/array [1.0 1.0]))]
      (<= (count (:loss-history result)) 20)))
  :num-tests 15)

;; Property 10: programmable-vi: final params are finite
(check "programmable-vi: final params finite"
  (prop/for-all [k gen-key]
    (let [result (vi/programmable-vi
                   {:iterations 20 :learning-rate 0.05 :n-samples 5
                    :objective :elbo :key k}
                   log-p-fn my-log-q my-sample-fn (mx/array [1.0 1.0]))]
      (mx/eval! (:params result))
      (every? js/isFinite (mx/->clj (:params result)))))
  :num-tests 15)

;; ---------------------------------------------------------------------------
;; vi-from-model
;; ---------------------------------------------------------------------------

(println "\n-- vi-from-model --")

;; Property 11: vi-from-model: mu is finite
(check "vi-from-model: mu is finite"
  (prop/for-all [k gen-key]
    (let [result (vi/vi-from-model
                   {:iterations 20 :learning-rate 0.05 :elbo-samples 3 :key k}
                   two-gauss [] two-gauss-obs [:x])
          mu (:mu result)]
      (mx/eval! mu)
      (every? js/isFinite (mx/->clj mu))))
  :num-tests 10)

;; Property 12: vi-from-model: sigma > 0
(check "vi-from-model: sigma > 0"
  (prop/for-all [k gen-key]
    (let [result (vi/vi-from-model
                   {:iterations 20 :learning-rate 0.05 :elbo-samples 3 :key k}
                   two-gauss [] two-gauss-obs [:x])
          sigma (:sigma result)]
      (mx/eval! sigma)
      (every? #(> % 0) (mx/->clj sigma))))
  :num-tests 10)

;; ---------------------------------------------------------------------------
;; Additional VI objectives
;; ---------------------------------------------------------------------------

(println "\n-- additional VI objectives --")

;; Property 13: pwake-objective: returns finite value
(check "pwake-objective: returns finite value"
  (prop/for-all [k gen-key]
    (let [obj-fn (vi/pwake-objective log-p-fn log-q-fn)
          samples (rng/normal k [5 2])
          result (obj-fn samples)]
      (mx/eval! result)
      (finite? (mx/item result))))
  :num-tests 30)

;; Property 14: reinforce-estimator: returns finite value
(check "reinforce-estimator: returns finite value"
  (prop/for-all [k gen-key]
    (let [base-obj (vi/elbo-objective log-p-fn log-q-fn)
          reinforce-fn (vi/reinforce-estimator base-obj log-q-fn)
          samples (rng/normal k [5 2])
          result (reinforce-fn samples)]
      (mx/eval! result)
      (finite? (mx/item result))))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== VI Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
