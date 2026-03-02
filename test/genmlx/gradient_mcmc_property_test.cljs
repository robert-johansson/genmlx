(ns genmlx.gradient-mcmc-property-test
  "Property-based tests for gradient MCMC: MALA, HMC, NUTS using test.check.
   Verifies sample counts, finiteness, and minimum acceptance rates.
   Uses small sample counts and compile?=false for fast property testing."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc])
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

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

;; ---------------------------------------------------------------------------
;; Model: simple 1-address Gaussian (minimal for gradient MCMC)
;; ---------------------------------------------------------------------------

(def simple-model
  (gen [sigma]
    (let [mu (trace :mu (dist/gaussian 0 sigma))]
      (trace :y (dist/gaussian mu 1))
      mu)))

(def obs (cm/choicemap :y (mx/scalar 2.0)))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

(println "\n=== Gradient MCMC Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; MALA properties
;; ---------------------------------------------------------------------------

(println "-- MALA --")

;; Property 1: mala: produces correct number of samples
(check "mala: correct sample count"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/mala {:samples 10 :step-size 0.01 :burn 2
                               :addresses [:mu] :compile? false :key k}
                              simple-model [10.0] obs)]
      (= 10 (count samples))))
  :num-tests 10)

;; Property 2: mala: all samples have finite values
(check "mala: all samples finite"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/mala {:samples 10 :step-size 0.01 :burn 2
                               :addresses [:mu] :compile? false :key k}
                              simple-model [10.0] obs)]
      (every? (fn [s] (every? js/isFinite s)) samples)))
  :num-tests 10)

;; Property 3: mala: at least 1 acceptance in 20 samples
(check "mala: at least 1 acceptance"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/mala {:samples 20 :step-size 0.01 :burn 2
                               :addresses [:mu] :compile? false :key k}
                              simple-model [10.0] obs)]
      ;; At least 2 distinct samples means some acceptance
      (> (count (set (mapv first samples))) 1)))
  :num-tests 10)

;; ---------------------------------------------------------------------------
;; Vectorized MALA
;; ---------------------------------------------------------------------------

(println "\n-- vectorized MALA --")

;; Property 4: vectorized-mala: correct sample count
(check "vectorized-mala: correct sample count"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/vectorized-mala
                    {:samples 5 :step-size 0.01 :burn 2
                     :addresses [:mu] :n-chains 3
                     :key k :device :cpu}
                    simple-model [10.0] obs)]
      (= 5 (count samples))))
  :num-tests 5)

;; Property 5: vectorized-mala: all samples finite
(check "vectorized-mala: all samples finite"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/vectorized-mala
                    {:samples 5 :step-size 0.01 :burn 2
                     :addresses [:mu] :n-chains 3
                     :key k :device :cpu}
                    simple-model [10.0] obs)]
      (every? (fn [s] (every? (fn [row] (every? js/isFinite row)) s)) samples)))
  :num-tests 5)

;; ---------------------------------------------------------------------------
;; HMC properties
;; ---------------------------------------------------------------------------

(println "\n-- HMC --")

;; Property 6: hmc: correct sample count
(check "hmc: correct sample count"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/hmc {:samples 10 :step-size 0.05 :leapfrog-steps 5
                              :burn 2 :addresses [:mu] :compile? false :key k}
                             simple-model [10.0] obs)]
      (= 10 (count samples))))
  :num-tests 10)

;; Property 7: hmc: all samples finite, no NaN
(check "hmc: all samples finite"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/hmc {:samples 10 :step-size 0.05 :leapfrog-steps 5
                              :burn 2 :addresses [:mu] :compile? false :key k}
                             simple-model [10.0] obs)]
      (every? (fn [s] (every? js/isFinite s)) samples)))
  :num-tests 10)

;; Property 8: hmc: at least 1 acceptance in 20 samples
(check "hmc: at least 1 acceptance"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/hmc {:samples 20 :step-size 0.05 :leapfrog-steps 5
                              :burn 2 :addresses [:mu] :compile? false :key k}
                             simple-model [10.0] obs)]
      (> (count (set (mapv first samples))) 1)))
  :num-tests 10)

;; ---------------------------------------------------------------------------
;; NUTS properties
;; ---------------------------------------------------------------------------

(println "\n-- NUTS --")

;; Property 9: nuts: correct sample count
(check "nuts: correct sample count"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/nuts {:samples 10 :step-size 0.05 :max-depth 3
                               :burn 2 :addresses [:mu] :compile? false :key k}
                              simple-model [10.0] obs)]
      (= 10 (count samples))))
  :num-tests 5)

;; Property 10: nuts: all samples finite, no NaN
(check "nuts: all samples finite"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/nuts {:samples 10 :step-size 0.05 :max-depth 3
                               :burn 2 :addresses [:mu] :compile? false :key k}
                              simple-model [10.0] obs)]
      (every? (fn [s] (every? js/isFinite s)) samples)))
  :num-tests 5)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Gradient MCMC Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
