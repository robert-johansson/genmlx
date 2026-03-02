(ns genmlx.smcp3-property-test
  "Property-based tests for SMCP3 (Sequential Monte Carlo with
   Probabilistic Program Proposals) using test.check."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.smcp3 :as smcp3]
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
;; Model: simple observation model
;; ---------------------------------------------------------------------------

(def model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 5))
            y (trace :y (dist/gaussian mu 1))]
        (mx/eval! mu y)
        (mx/item mu)))))

;; Prior proposal
(def prior-proposal
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 1))]
        (mx/eval! mu)
        (mx/item mu)))))

(def obs-seq [(cm/choicemap :y (mx/scalar 2.0))])

(def particle-pool [5 10 15])
(def gen-n-particles (gen/elements particle-pool))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

(println "\n=== SMCP3 Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; SMCP3 init without proposal
;; ---------------------------------------------------------------------------

(println "-- smcp3-init without proposal --")

;; Property 1: smcp3-init without proposal: trace count = particles
(check "smcp3-init (no proposal): trace count = particles"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smcp3/smcp3-init model [] (first obs-seq) nil n k)]
      (= n (count (:traces result))))))

;; Property 2: smcp3-init without proposal: all weights finite
(check "smcp3-init (no proposal): all weights finite"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smcp3/smcp3-init model [] (first obs-seq) nil n k)]
      (every? #(finite? (eval-weight %)) (:log-weights result)))))

;; Property 3: smcp3-init without proposal: log-ml-increment finite
(check "smcp3-init (no proposal): log-ml-increment finite"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smcp3/smcp3-init model [] (first obs-seq) nil n k)]
      (finite? (eval-weight (:log-ml-increment result))))))

;; ---------------------------------------------------------------------------
;; SMCP3 init with proposal
;; ---------------------------------------------------------------------------

(println "\n-- smcp3-init with proposal --")

;; Property 4: smcp3-init with proposal: trace count = particles
(check "smcp3-init (with proposal): trace count = particles"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smcp3/smcp3-init model [] (first obs-seq) prior-proposal n k)]
      (= n (count (:traces result))))))

;; Property 5: smcp3-init with proposal: all weights finite
(check "smcp3-init (with proposal): all weights finite"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smcp3/smcp3-init model [] (first obs-seq) prior-proposal n k)]
      (every? #(finite? (eval-weight %)) (:log-weights result)))))

;; ---------------------------------------------------------------------------
;; SMCP3 step
;; ---------------------------------------------------------------------------

(println "\n-- smcp3-step --")

;; We need a 2-step model for testing smcp3-step
(def model-2step
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 5))
            y0 (trace :y0 (dist/gaussian mu 1))
            y1 (trace :y1 (dist/gaussian mu 1))]
        (mx/eval! mu y0 y1)
        (mx/item mu)))))

(def obs-0 (cm/choicemap :y0 (mx/scalar 2.0)))
(def obs-1 (cm/choicemap :y0 (mx/scalar 2.0) :y1 (mx/scalar 3.0)))

;; Property 6: smcp3-step: trace count preserved
(check "smcp3-step: trace count preserved"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [[k1 k2] (rng/split k)
          init (smcp3/smcp3-init model-2step [] obs-0 nil n k1)
          step-result (smcp3/smcp3-step
                        (:traces init) (:log-weights init)
                        model-2step obs-1
                        nil nil  ;; no forward/backward kernels
                        n 0.5 nil k2)]
      (= n (count (:traces step-result))))))

;; Property 7: smcp3-step: all updated weights finite
(check "smcp3-step: all weights finite"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [[k1 k2] (rng/split k)
          init (smcp3/smcp3-init model-2step [] obs-0 nil n k1)
          step-result (smcp3/smcp3-step
                        (:traces init) (:log-weights init)
                        model-2step obs-1
                        nil nil n 0.5 nil k2)]
      (every? #(finite? (eval-weight %)) (:log-weights step-result)))))

;; ---------------------------------------------------------------------------
;; SMCP3 full pipeline
;; ---------------------------------------------------------------------------

(println "\n-- smcp3 full pipeline --")

;; Property 8: smcp3: full pipeline trace count = particles
(check "smcp3: trace count = particles"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smcp3/smcp3 {:particles n :key k}
                               model [] obs-seq)]
      (= n (count (:traces result))))))

;; Property 9: smcp3: log-ml-estimate finite
(check "smcp3: log-ml-estimate finite"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smcp3/smcp3 {:particles n :key k}
                               model [] obs-seq)]
      (finite? (eval-weight (:log-ml-estimate result))))))

;; Property 10: smcp3: all final traces have finite scores
(check "smcp3: all traces have finite scores"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [result (smcp3/smcp3 {:particles n :key k}
                               model [] obs-seq)]
      (every? (fn [trace]
                (mx/eval! (:score trace))
                (finite? (mx/item (:score trace))))
              (:traces result)))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== SMCP3 Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
