(ns genmlx.mcmc-property-test
  "Property-based tests for scalar MCMC: MH, custom proposal MH,
   compiled MH, Gibbs, and involutive MCMC using test.check."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
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

(defn- choice-val [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

;; ---------------------------------------------------------------------------
;; Model: linear regression with 3 data points
;; ---------------------------------------------------------------------------

(def linreg
  (dyn/auto-key
    (gen [xs]
      (let [slope (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                        intercept) 1)))
        (mx/eval! slope intercept)
        [(mx/item slope) (mx/item intercept)]))))

(def xs [1.0 2.0 3.0])
(def obs (cm/choicemap :y0 (mx/scalar 1.5)
                       :y1 (mx/scalar 3.0)
                       :y2 (mx/scalar 4.5)))

;; Bernoulli model for Gibbs
(def bernoulli-model
  (dyn/auto-key
    (gen []
      (let [b (trace :b (dist/bernoulli 0.5))
            _ (mx/eval! b)
            p-y (if (> (mx/item b) 0.5) 0.8 0.2)
            y (trace :y (dist/bernoulli p-y))]
        (mx/eval! y)
        (mx/item b)))))

(def bernoulli-obs (cm/choicemap :y (mx/scalar 1.0)))

;; Proposal GF for custom MH
(def slope-proposal
  (dyn/auto-key
    (gen [choices]
      (let [old-slope (choice-val choices :slope)
            new-slope (trace :slope (dist/gaussian (or old-slope 0) 0.5))]
        (mx/eval! new-slope)
        (mx/item new-slope)))))

;; Involution: identity (swap trace and aux)
(def identity-involution
  (fn [trace-cm aux-cm]
    [trace-cm aux-cm]))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))
(def gen-samples (gen/elements [10 20]))
(def gen-burn (gen/elements [0 5]))

(println "\n=== MCMC Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; MH properties
;; ---------------------------------------------------------------------------

(println "-- MH --")

;; Property 1: mh-step: returned trace has same addresses
(check "mh-step: trace has same addresses"
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          new-trace (mcmc/mh-step trace (sel/select :slope) k)]
      (and (some? (choice-val (:choices new-trace) :slope))
           (some? (choice-val (:choices new-trace) :intercept))))))

;; Property 2: mh-step: returned trace has finite score
(check "mh-step: finite score"
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          new-trace (mcmc/mh-step trace (sel/select :slope) k)]
      (finite? (eval-weight (:score new-trace))))))

;; Property 3: mh-step(sel/none): trace unchanged
(check "mh-step(sel/none): no change"
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          new-trace (mcmc/mh-step trace sel/none k)]
      ;; With sel/none, regenerate weight is 0, always accept → same values
      (let [old-slope (choice-val (:choices trace) :slope)
            new-slope (choice-val (:choices new-trace) :slope)]
        (and (finite? old-slope) (finite? new-slope)
             (<= (js/Math.abs (- old-slope new-slope)) 1e-6))))))

;; Property 4: mh: produces correct number of samples
(check "mh: correct sample count"
  (prop/for-all [n gen-samples
                 k gen-key]
    (let [samples (mcmc/mh {:samples n :selection (sel/select :slope) :key k}
                            linreg [xs] obs)]
      (= n (count samples)))))

;; Property 5: mh: all traces have finite scores
(check "mh: all traces have finite scores"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/mh {:samples 10 :selection (sel/select :slope) :key k}
                            linreg [xs] obs)]
      (every? #(finite? (eval-weight (:score %))) samples))))

;; ---------------------------------------------------------------------------
;; Custom proposal MH
;; ---------------------------------------------------------------------------

(println "\n-- custom proposal MH --")

;; Property 6: mh-custom-step: returned trace has same addresses
(check "mh-custom-step: trace has same addresses"
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          new-trace (mcmc/mh-custom-step trace linreg slope-proposal k)]
      (and (some? (choice-val (:choices new-trace) :slope))
           (some? (choice-val (:choices new-trace) :intercept))))))

;; Property 7: mh-custom-step: returned trace has finite score
(check "mh-custom-step: finite score"
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          new-trace (mcmc/mh-custom-step trace linreg slope-proposal k)]
      (finite? (eval-weight (:score new-trace))))))

;; Property 8: mh-custom: correct sample count
(check "mh-custom: correct sample count"
  (prop/for-all [n gen-samples
                 k gen-key]
    (let [samples (mcmc/mh-custom {:samples n :proposal-gf slope-proposal :key k}
                                   linreg [xs] obs)]
      (= n (count samples)))))

;; ---------------------------------------------------------------------------
;; Compiled MH
;; ---------------------------------------------------------------------------

(println "\n-- compiled MH --")

;; Property 9: compiled-mh: correct sample count
(check "compiled-mh: correct sample count"
  (prop/for-all [n gen-samples
                 k gen-key]
    (let [samples (mcmc/compiled-mh {:samples n :burn 5 :addresses [:slope :intercept]
                                      :proposal-std 0.1 :key k :compile? false}
                                     linreg [xs] obs)]
      (= n (count samples))))
  :num-tests 20)

;; Property 10: compiled-mh: all returned values are finite
(check "compiled-mh: all values finite"
  (prop/for-all [k gen-key]
    (let [samples (mcmc/compiled-mh {:samples 10 :burn 5 :addresses [:slope :intercept]
                                      :proposal-std 0.1 :key k :compile? false}
                                     linreg [xs] obs)]
      (every? (fn [s] (every? js/isFinite s)) samples)))
  :num-tests 20)

;; ---------------------------------------------------------------------------
;; Gibbs sampling
;; ---------------------------------------------------------------------------

(println "\n-- Gibbs --")

;; Property 11: gibbs-step: score is finite
(check "gibbs: score finite after step"
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate bernoulli-model [] bernoulli-obs)
          new-trace (mcmc/gibbs-step-with-support trace :b
                      [(mx/scalar 0.0) (mx/scalar 1.0)] k)]
      (finite? (eval-weight (:score new-trace))))))

;; Property 12: gibbs: correct sample count
(check "gibbs: correct sample count"
  (prop/for-all [n gen-samples
                 k gen-key]
    (let [schedule [{:addr :b :support [(mx/scalar 0.0) (mx/scalar 1.0)]}]
          samples (mcmc/gibbs {:samples n :key k}
                               bernoulli-model [] bernoulli-obs schedule)]
      (= n (count samples)))))

;; ---------------------------------------------------------------------------
;; Involutive MCMC
;; ---------------------------------------------------------------------------

(println "\n-- involutive MCMC --")

;; Simple aux proposal for involutive MCMC
(def aux-proposal
  (dyn/auto-key
    (gen [choices]
      (let [x (trace :noise (dist/gaussian 0 0.1))]
        (mx/eval! x)
        (mx/item x)))))

;; Involution that adds aux noise to :slope
(def add-noise-involution
  (fn [trace-cm aux-cm]
    (let [slope-sub (cm/get-submap trace-cm :slope)
          noise-sub (cm/get-submap aux-cm :noise)]
      (if (and (cm/has-value? slope-sub) (cm/has-value? noise-sub))
        (let [slope-val (cm/get-value slope-sub)
              noise-val (cm/get-value noise-sub)
              new-slope (mx/add slope-val noise-val)
              new-noise (mx/negative noise-val)
              new-trace (cm/set-value trace-cm :slope new-slope)
              new-aux (cm/set-value aux-cm :noise new-noise)]
          [new-trace new-aux])
        [trace-cm aux-cm]))))

;; Property 13: involutive-mh-step: returned trace has finite score
(check "involutive-mh-step: finite score"
  (prop/for-all [k gen-key]
    (let [{:keys [trace]} (p/generate linreg [xs] obs)
          new-trace (mcmc/involutive-mh-step trace linreg aux-proposal
                                              add-noise-involution k)]
      (finite? (eval-weight (:score new-trace))))))

;; Property 14: involutive-mh: correct sample count
(check "involutive-mh: correct sample count"
  (prop/for-all [n gen-samples
                 k gen-key]
    (let [samples (mcmc/involutive-mh {:samples n :proposal-gf aux-proposal
                                        :involution add-noise-involution :key k}
                                       linreg [xs] obs)]
      (= n (count samples)))))

;; Property 15: mh-step accept/reject: log-alpha=0 always accepts
(check "accept-mh?(0) always true"
  (prop/for-all [k gen-key]
    (u/accept-mh? 0 k))
  :num-tests 100)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== MCMC Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
