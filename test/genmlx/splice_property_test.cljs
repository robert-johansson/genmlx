(ns genmlx.splice-property-test
  "Property-based tests for splice (compositional modeling).

   Splice is the mechanism by which a parent generative function calls a
   sub-generative-function, creating hierarchical traces. These tests verify
   the algebraic laws that the GFI must satisfy for spliced models:
   score decomposition, importance weighting, update identity, invertibility,
   projection totality, and additive independence."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test infrastructure (follows gfi_property_test.cljs)
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
        (println "    shrunk:" (get-in result [:shrunk :smallest])))))

(defn- check [name prop & {:keys [num-tests] :or {num-tests 50}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

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

(println "\n=== Splice Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; S1. splice: score = parent score + child score
;;
;; Law: The total score of a spliced model decomposes additively into the
;; log-probability contributions of the parent's trace sites and the child's
;; trace sites. This is the fundamental compositional scoring identity.
;;
;; Verification: simulate outer-model, then compute log-prob at :x and at
;; [:sub :z] individually and check they sum to the trace score.
;; ---------------------------------------------------------------------------

(println "-- splice score decomposition --")

(check "S1: splice score = parent log-prob + child log-prob"
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
;;
;; Law: When all addresses are constrained (full observation), the importance
;; weight equals the trace score. This is the GFI importance weighting identity
;; extended to hierarchical models with splice.
;;
;; generate(model, args, all-choices).weight = generate(model, args, all-choices).trace.score
;; ---------------------------------------------------------------------------

(println "\n-- splice generate --")

(check "S2: splice generate(full) weight = score"
  (prop/for-all [_ gen-run-idx]
    (let [tr (p/simulate outer-model [])
          {:keys [trace weight]} (p/generate outer-model [] (:choices tr))
          w (eval-weight weight)
          s (trace-score trace)]
      (close? s w 0.01))))

;; ---------------------------------------------------------------------------
;; S3. splice: update(same) weight = 0
;;
;; Law: Updating a trace with its own choices produces weight 0. This is the
;; GFI update identity: the ratio new_score/old_score = 1, so log-weight = 0.
;; For splice models, this must hold across the hierarchical boundary.
;; ---------------------------------------------------------------------------

(println "\n-- splice update --")

(check "S3: splice update(same) weight = 0"
  (prop/for-all [_ gen-run-idx]
    (let [tr (p/simulate outer-model [])
          {:keys [weight]} (p/update outer-model tr (:choices tr))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; S4. splice: update round-trip via discard
;;
;; Law: Update is invertible. If we update trace T with constraint C to get
;; (T', discard D), then updating T' with D recovers the original score.
;; For splice, this tests that the discard correctly captures the old value
;; at a hierarchical address like [:sub :z].
;;
;; update(T, C) -> (T', D)
;; update(T', D) -> (T'', _)
;; T''.score approx T.score
;; ---------------------------------------------------------------------------

(check "S4: splice update round-trip via discard"
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
;;
;; Law: Projecting a trace onto all addresses yields the total score.
;; For splice models, this means the projection correctly traverses
;; the hierarchical choice structure.
;;
;; project(model, trace, ALL) = trace.score
;; ---------------------------------------------------------------------------

(println "\n-- splice project --")

(check "S5: splice project(all) = score"
  (prop/for-all [_ gen-run-idx]
    (let [tr (p/simulate outer-model [])
          score (trace-score tr)
          proj (eval-weight (p/project outer-model tr sel/all))]
      (close? score proj 0.01))))

;; ---------------------------------------------------------------------------
;; S6. splice: dual splice score = sum of sub-scores
;;
;; Law: When a model contains two independent spliced sub-models, the total
;; score is the sum of each sub-model's individual score. This tests that
;; splice addressing (:left vs :right) correctly namespaces the sub-models
;; and that scores accumulate additively across independent sub-computations.
;;
;; score(dual) = assess(inner, left-args, left-choices).weight
;;             + assess(inner, right-args, right-choices).weight
;; ---------------------------------------------------------------------------

(println "\n-- splice dual independence --")

(check "S6: dual splice score = sum of individual sub-scores"
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

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Splice Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
