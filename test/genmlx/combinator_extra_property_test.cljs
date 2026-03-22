(ns genmlx.combinator-extra-property-test
  "Property-based tests for Mix, Recurse, Contramap, and Dimap combinators.
   Verifies GFI contracts and structural invariants."
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

;; ---------------------------------------------------------------------------
;; Generator pools
;; ---------------------------------------------------------------------------

(def key-pool [(rng/fresh-key 42) (rng/fresh-key 99) (rng/fresh-key 123)
               (rng/fresh-key 7) (rng/fresh-key 255)])
(def gen-key (gen/elements key-pool))

(def arg-pool [1.0 2.0 3.0 -1.0 0.5])
(def gen-arg (gen/elements arg-pool))

(def constraint-pool [0.5 1.0 -0.5 2.0 -1.0])
(def gen-constraint (gen/elements constraint-pool))

(println "\n=== Combinator Extra Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Mix Combinator (5)
;; ---------------------------------------------------------------------------

(println "-- Mix combinator --")

;; Two Gaussian components with varying weights
(def mix-comp1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1)))))
(def mix-comp2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 5 2)))))

(def mix-weight-pool [0.1 0.3 0.5 0.7 0.9])
(def gen-mix-weight (gen/elements mix-weight-pool))

(defn make-mixed [w1]
  (let [log-w (mx/log (mx/array [w1 (- 1.0 w1)]))]
    (comb/mix-combinator [mix-comp1 mix-comp2] log-w)))

(check "Mix: simulate produces valid component-idx in {0, 1}"
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          trace (p/simulate m [])
          idx (mx/item (cm/get-choice (:choices trace) [:component-idx]))]
      (or (== idx 0) (== idx 1)))))

(check "Mix: generate(empty) weight near 0"
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          {:keys [weight]} (p/generate m [] cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Mix: update(same) weight near 0"
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          trace (p/simulate m [])
          {:keys [weight]} (p/update m trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Mix: generate(full) weight near score"
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          trace (p/simulate m [])
          {:keys [trace weight]} (p/generate m [] (:choices trace))
          w (eval-weight weight)
          s (eval-score trace)]
      (close? s w 0.01))))

(check "Mix: regenerate(none) weight near 0"
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          trace (p/simulate m [])
          {:keys [weight]} (p/regenerate m trace sel/none)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; Recurse Combinator (4)
;; ---------------------------------------------------------------------------

(println "\n-- Recurse combinator --")

;; Simple recursive model: traces :v, no splice (avoids regenerate complexity)
;; The recurse combinator just provides fixed-point wrapping.
(def recursive-model
  (comb/recurse
    (fn [self]
      (dyn/auto-key
        (gen [depth]
          (let [v (trace :v (dist/gaussian 0 1))]
            (mx/eval! v)
            (mx/item v)))))))

;; Pool: depth doesn't matter since no recursion, but tests the combinator wrapping
(def recurse-pool
  [{:depth 0 :args [0] :label "recurse(depth=0)"}
   {:depth 1 :args [1] :label "recurse(depth=1)"}])

(def gen-recurse-spec (gen/elements recurse-pool))

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
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/gaussian x 1))]
        (mx/eval! y) (mx/item y)))))

;; Identity contramap: passes args unchanged
(def contramap-identity (comb/contramap-gf inner-gf identity))

;; Doubling contramap: doubles the first arg
(def contramap-double (comb/contramap-gf inner-gf (fn [args] (mapv #(* 2 %) args))))

(def contramap-pool
  [{:gf contramap-identity :label "contramap(identity)"}
   {:gf contramap-double   :label "contramap(double)"}])

(def gen-contramap-spec (gen/elements contramap-pool))

(check "Contramap(identity): score equals inner GF score"
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate contramap-identity [arg] constraint)
          cm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [arg] constraint)
          inner-score (eval-score trace)]
      (close? cm-score inner-score 0.01))))

(check "Contramap(identity): retval matches inner GF retval"
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate contramap-identity [arg] constraint)
          cm-retval (:retval trace)
          {:keys [trace]} (p/generate inner-gf [arg] constraint)
          inner-retval (:retval trace)]
      (close? cm-retval inner-retval 1e-6))))

(check "Contramap: generate(empty) weight near 0"
  (prop/for-all [spec gen-contramap-spec
                 arg gen-arg]
    (let [{:keys [weight]} (p/generate (:gf spec) [arg] cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Contramap: update(same) weight near 0"
  (prop/for-all [spec gen-contramap-spec
                 arg gen-arg]
    (let [trace (p/simulate (:gf spec) [arg])
          {:keys [weight]} (p/update (:gf spec) trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Contramap: score unchanged by arg transformation (structural)"
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    ;; contramap-double with args [arg] should behave like inner-gf with args [2*arg]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate contramap-double [arg] constraint)
          cm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [(* 2 arg)] constraint)
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
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    ;; dimap with args [arg] transforms to inner args [2*arg], same score
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate dimapped [arg] constraint)
          dm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [(* 2 arg)] constraint)
          inner-score (eval-score trace)]
      (close? dm-score inner-score 0.01))))

(check "Dimap: retval transformed by output function"
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate dimapped [arg] constraint)
          dm-retval (:retval trace)
          {:keys [trace]} (p/generate inner-gf [(* 2 arg)] constraint)
          inner-retval (:retval trace)]
      ;; dimap negates, so dm-retval = -(inner-retval)
      (close? dm-retval (- inner-retval) 1e-6))))

(check "Dimap: generate(empty) weight near 0"
  (prop/for-all [arg gen-arg]
    (let [{:keys [weight]} (p/generate dimapped [arg] cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(check "Dimap: update(same) weight near 0"
  (prop/for-all [arg gen-arg]
    (let [trace (p/simulate dimapped [arg])
          {:keys [weight]} (p/update dimapped trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Combinator Extra Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
