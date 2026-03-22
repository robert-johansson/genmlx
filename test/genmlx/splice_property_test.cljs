(ns genmlx.splice-property-test
  "Property-based tests for splice (compositional modeling).

   Splice is the mechanism by which a parent generative function calls a
   sub-generative-function, creating hierarchical traces. These tests verify
   the algebraic laws that the GFI must satisfy for spliced models:
   score decomposition, importance weighting, update identity, invertibility,
   projection totality, and additive independence."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- trace-score [trace]
  (mx/eval! (:score trace))
  (mx/item (:score trace)))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Inner model (sub-model): samples z ~ N(mu, 1)
(def inner-model
  (gen [mu]
    (trace :z (dist/gaussian mu 1))))

;; Standalone auto-keyed inner for assess calls outside splice
(def inner-model-ak (dyn/auto-key inner-model))

;; Outer model: samples x ~ N(0,2), splices inner-model with mu=x
(def outer-model
  (dyn/auto-key (gen []
    (let [x (trace :x (dist/gaussian 0 2))]
      (splice :sub inner-model x)
      x))))

;; Dual-splice model: two independent splices, each with mu=0
(def dual-splice-model
  (dyn/auto-key (gen []
    (let [a (splice :left inner-model (mx/scalar 0))
          b (splice :right inner-model (mx/scalar 0))]
      (mx/add a b)))))

;; ---------------------------------------------------------------------------
;; Generators (SCI-safe: gen/elements from pre-built pools)
;; ---------------------------------------------------------------------------

;; Pool of constraint values for update round-trip
(def constraint-pool
  (mapv mx/scalar [-5.0 -3.0 -1.0 0.0 1.0 3.0 5.0]))

(def gen-constraint-val
  (gen/elements constraint-pool))

;; Dummy generator: we use gen/return because models are fixed,
;; but test.check still provides randomness via the seed (which
;; controls the PRNG for each auto-keyed simulate call).
(def gen-run-idx
  (gen/elements (range 50)))

;; ---------------------------------------------------------------------------
;; S1. splice: score = parent score + child score
;; ---------------------------------------------------------------------------

(defspec s1-splice-score-equals-parent-log-prob-plus-child-log-prob 50
  (prop/for-all [_ gen-run-idx]
    (let [tr (p/simulate outer-model [])
          score (trace-score tr)
          choices (:choices tr)
          ;; Extract values
          x-val (cm/get-value (cm/get-submap choices :x))
          z-val (cm/get-value (cm/get-submap (cm/get-submap choices :sub) :z))
          _ (mx/eval! x-val z-val)
          ;; Compute individual log-probs
          x-lp (mx/item (dist/log-prob (dist/gaussian 0 2) x-val))
          z-lp (mx/item (dist/log-prob (dist/gaussian x-val 1) z-val))]
      (close? score (+ x-lp z-lp) 0.01))))

;; ---------------------------------------------------------------------------
;; S2. splice: generate(full) weight = score
;; ---------------------------------------------------------------------------

(defspec s2-splice-generate-full-weight-equals-score 50
  (prop/for-all [_ gen-run-idx]
    (let [tr (p/simulate outer-model [])
          {:keys [trace weight]} (p/generate outer-model [] (:choices tr))
          w (eval-weight weight)
          s (trace-score trace)]
      (close? s w 0.01))))

;; ---------------------------------------------------------------------------
;; S3. splice: update(same) weight = 0
;; ---------------------------------------------------------------------------

(defspec s3-splice-update-same-weight-equals-0 50
  (prop/for-all [_ gen-run-idx]
    (let [tr (p/simulate outer-model [])
          {:keys [weight]} (p/update outer-model tr (:choices tr))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; S4. splice: update round-trip via discard
;; ---------------------------------------------------------------------------

(defspec s4-splice-update-round-trip-via-discard 50
  (prop/for-all [cval gen-constraint-val]
    (let [tr (p/simulate outer-model [])
          orig-score (trace-score tr)
          ;; Constrain the sub-model's :z address
          constraint (cm/set-choice cm/EMPTY [:sub :z] cval)
          {:keys [trace discard]} (p/update outer-model tr constraint)
          ;; Round-trip: apply discard to recover original
          {:keys [trace]} (p/update outer-model trace discard)
          recovered-score (trace-score trace)]
      (close? orig-score recovered-score 0.01))))

;; ---------------------------------------------------------------------------
;; S5. splice: project(all) = score
;; ---------------------------------------------------------------------------

(defspec s5-splice-project-all-equals-score 50
  (prop/for-all [_ gen-run-idx]
    (let [tr (p/simulate outer-model [])
          score (trace-score tr)
          proj (eval-weight (p/project outer-model tr sel/all))]
      (close? score proj 0.01))))

;; ---------------------------------------------------------------------------
;; S6. splice: dual splice score = sum of sub-scores
;; ---------------------------------------------------------------------------

(defspec s6-dual-splice-score-equals-sum-of-individual-sub-scores 50
  (prop/for-all [_ gen-run-idx]
    (let [tr (p/simulate dual-splice-model [])
          score (trace-score tr)
          choices (:choices tr)
          ;; Extract sub-choicemaps
          left-choices (cm/get-submap choices :left)
          right-choices (cm/get-submap choices :right)
          ;; Assess each sub-model independently (both with mu=0)
          left-w (eval-weight (:weight (p/assess inner-model-ak [(mx/scalar 0)] left-choices)))
          right-w (eval-weight (:weight (p/assess inner-model-ak [(mx/scalar 0)] right-choices)))]
      (close? score (+ left-w right-w) 0.01))))

(t/run-tests)
