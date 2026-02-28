(ns genmlx.combinator-extra-property-test
  "Property-based tests for Mix, Recurse, Contramap, and Dimap combinators.
   Verifies GFI contracts and structural invariants."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb])
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

(defn- eval-score [trace]
  (mx/eval! (:score trace))
  (mx/item (:score trace)))

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

(println "\n=== Combinator Extra Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Mix Combinator (6)
;; ---------------------------------------------------------------------------

(println "-- Mix combinator --")

;; Two Gaussian components with equal log-weights
(def mix-comp1 (gen [] (dyn/trace :y (dist/gaussian 0 1))))
(def mix-comp2 (gen [] (dyn/trace :y (dist/gaussian 5 2))))
(def mix-equal-weights (mx/log (mx/array [0.5 0.5])))
(def mixed (comb/mix-combinator [mix-comp1 mix-comp2] mix-equal-weights))

(check "Mix: simulate produces valid component-idx in {0, 1}"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate mixed [])
          idx (mx/item (cm/get-choice (:choices trace) [:component-idx]))]
      (or (== idx 0) (== idx 1)))))

(check "Mix: score is finite"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate mixed [])
          s (eval-score trace)]
      (finite? s))))

(check "Mix: generate(empty) weight near 0"
  (prop/for-all [_ (gen/return nil)]
    (let [{:keys [weight]} (p/generate mixed [] cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Mix: update(same) weight near 0"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate mixed [])
          {:keys [weight]} (p/update mixed trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Mix: generate(full) weight near score"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate mixed [])
          {:keys [trace weight]} (p/generate mixed [] (:choices trace))
          w (eval-weight weight)
          s (eval-score trace)]
      (close? s w 0.01))))

(check "Mix: regenerate(none) weight near 0"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate mixed [])
          {:keys [weight]} (p/regenerate mixed trace sel/none)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; Recurse Combinator (5)
;; ---------------------------------------------------------------------------

(println "\n-- Recurse combinator --")

;; Simple recursive model: traces :v, no splice (avoids regenerate complexity)
;; The recurse combinator just provides fixed-point wrapping.
(def recursive-model
  (comb/recurse
    (fn [self]
      (gen [depth]
        (let [v (dyn/trace :v (dist/gaussian 0 1))]
          (mx/eval! v)
          (mx/item v))))))

;; Pool: depth doesn't matter since no recursion, but tests the combinator wrapping
(def recurse-pool
  [{:depth 0 :args [0] :label "recurse(depth=0)"}
   {:depth 1 :args [1] :label "recurse(depth=1)"}])

(def gen-recurse-spec (gen/elements recurse-pool))

(check "Recurse: simulate produces trace with finite score"
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          s (eval-score trace)]
      (finite? s))))

(check "Recurse: generate(full) weight near score"
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          {:keys [trace weight]} (p/generate recursive-model (:args spec) (:choices trace))
          w (eval-weight weight)
          s (eval-score trace)]
      (close? s w 0.01))))

(check "Recurse: update(same) weight near 0"
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          {:keys [weight]} (p/update recursive-model trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Recurse: regenerate(none) weight near 0"
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          {:keys [weight]} (p/regenerate recursive-model trace sel/none)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Recurse: project(all) near score"
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          s (eval-score trace)
          proj (eval-weight (p/project recursive-model trace sel/all))]
      (close? s proj 0.01))))

;; ---------------------------------------------------------------------------
;; Contramap (5)
;; ---------------------------------------------------------------------------

(println "\n-- Contramap combinator --")

(def inner-gf
  (gen [x]
    (let [y (dyn/trace :y (dist/gaussian x 1))]
      (mx/eval! y) (mx/item y))))

;; Identity contramap: passes args unchanged
(def contramap-identity (comb/contramap-gf inner-gf identity))

;; Doubling contramap: doubles the first arg
(def contramap-double (comb/contramap-gf inner-gf (fn [args] (mapv #(* 2 %) args))))

(def contramap-pool
  [{:gf contramap-identity :args [3.0] :label "contramap(identity)"}
   {:gf contramap-double   :args [3.0] :label "contramap(double)"}])

(def gen-contramap-spec (gen/elements contramap-pool))

(check "Contramap(identity): score equals inner GF score"
  (prop/for-all [_ (gen/return nil)]
    (let [constraint (cm/choicemap :y (mx/scalar 2.5))
          {:keys [trace]} (p/generate contramap-identity [3.0] constraint)
          cm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [3.0] constraint)
          inner-score (eval-score trace)]
      (close? cm-score inner-score 0.01))))

(check "Contramap(identity): retval matches inner GF retval"
  (prop/for-all [_ (gen/return nil)]
    (let [constraint (cm/choicemap :y (mx/scalar 2.5))
          {:keys [trace]} (p/generate contramap-identity [3.0] constraint)
          cm-retval (:retval trace)
          {:keys [trace]} (p/generate inner-gf [3.0] constraint)
          inner-retval (:retval trace)]
      (close? cm-retval inner-retval 1e-6))))

(check "Contramap: generate(empty) weight near 0"
  (prop/for-all [spec gen-contramap-spec]
    (let [{:keys [weight]} (p/generate (:gf spec) (:args spec) cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Contramap: update(same) weight near 0"
  (prop/for-all [spec gen-contramap-spec]
    (let [trace (p/simulate (:gf spec) (:args spec))
          {:keys [weight]} (p/update (:gf spec) trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Contramap: score unchanged by arg transformation (structural)"
  (prop/for-all [_ (gen/return nil)]
    ;; contramap-double with args [3.0] should behave like inner-gf with args [6.0]
    (let [constraint (cm/choicemap :y (mx/scalar 2.5))
          {:keys [trace]} (p/generate contramap-double [3.0] constraint)
          cm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [6.0] constraint)
          inner-score (eval-score trace)]
      (close? cm-score inner-score 0.01))))

;; ---------------------------------------------------------------------------
;; Dimap (4)
;; ---------------------------------------------------------------------------

(println "\n-- Dimap combinator --")

;; dimap: double input, negate output
(def dimapped (comb/dimap inner-gf
                          (fn [args] (mapv #(* 2 %) args))  ; double input
                          (fn [retval] (- retval))))          ; negate output

(check "Dimap: score unchanged vs inner (with matching args and constraints)"
  (prop/for-all [_ (gen/return nil)]
    ;; dimap with args [3.0] transforms to inner args [6.0], same score
    (let [constraint (cm/choicemap :y (mx/scalar 2.5))
          {:keys [trace]} (p/generate dimapped [3.0] constraint)
          dm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [6.0] constraint)
          inner-score (eval-score trace)]
      (close? dm-score inner-score 0.01))))

(check "Dimap: retval transformed by output function"
  (prop/for-all [_ (gen/return nil)]
    (let [constraint (cm/choicemap :y (mx/scalar 2.5))
          {:keys [trace]} (p/generate dimapped [3.0] constraint)
          dm-retval (:retval trace)
          {:keys [trace]} (p/generate inner-gf [6.0] constraint)
          inner-retval (:retval trace)]
      ;; dimap negates, so dm-retval = -(inner-retval)
      (close? dm-retval (- inner-retval) 1e-6))))

(check "Dimap: generate(empty) weight near 0"
  (prop/for-all [_ (gen/return nil)]
    (let [{:keys [weight]} (p/generate dimapped [3.0] cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Dimap: update(same) weight near 0"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate dimapped [3.0])
          {:keys [weight]} (p/update dimapped trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Combinator Extra Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
