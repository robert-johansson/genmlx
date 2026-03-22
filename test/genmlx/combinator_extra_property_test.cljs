(ns genmlx.combinator-extra-property-test
  "Property-based tests for Mix, Recurse, Contramap, and Dimap combinators.
   Verifies GFI contracts and structural invariants."
  (:require [cljs.test :as t]
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
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

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

;; ---------------------------------------------------------------------------
;; Mix Combinator
;; ---------------------------------------------------------------------------

(def mix-comp1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1)))))
(def mix-comp2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 5 2)))))

(def mix-weight-pool [0.1 0.3 0.5 0.7 0.9])
(def gen-mix-weight (gen/elements mix-weight-pool))

(defn make-mixed [w1]
  (let [log-w (mx/log (mx/array [w1 (- 1.0 w1)]))]
    (comb/mix-combinator [mix-comp1 mix-comp2] log-w)))

(defspec mix-component-idx-in-0-1 50
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          trace (p/simulate m [])
          idx (mx/item (cm/get-choice (:choices trace) [:component-idx]))]
      (or (== idx 0) (== idx 1)))))

(defspec mix-generate-empty-weight-near-zero 50
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          {:keys [weight]} (p/generate m [] cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec mix-update-same-weight-near-zero 50
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          trace (p/simulate m [])
          {:keys [weight]} (p/update m trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec mix-generate-full-weight-near-score 50
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          trace (p/simulate m [])
          {:keys [trace weight]} (p/generate m [] (:choices trace))
          w (eval-weight weight)
          s (eval-score trace)]
      (close? s w 0.01))))

(defspec mix-regenerate-none-weight-near-zero 50
  (prop/for-all [w gen-mix-weight]
    (let [m (make-mixed w)
          trace (p/simulate m [])
          {:keys [weight]} (p/regenerate m trace sel/none)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; Recurse Combinator
;; ---------------------------------------------------------------------------

(def recursive-model
  (comb/recurse
    (fn [self]
      (dyn/auto-key
        (gen [depth]
          (let [v (trace :v (dist/gaussian 0 1))]
            (mx/eval! v)
            (mx/item v)))))))

(def recurse-pool
  [{:depth 0 :args [0] :label "recurse(depth=0)"}
   {:depth 1 :args [1] :label "recurse(depth=1)"}])

(def gen-recurse-spec (gen/elements recurse-pool))

(defspec recurse-generate-full-weight-near-score 50
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          {:keys [trace weight]} (p/generate recursive-model (:args spec) (:choices trace))
          w (eval-weight weight)
          s (eval-score trace)]
      (close? s w 0.01))))

(defspec recurse-update-same-weight-near-zero 50
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          {:keys [weight]} (p/update recursive-model trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec recurse-regenerate-none-weight-near-zero 50
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          {:keys [weight]} (p/regenerate recursive-model trace sel/none)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec recurse-project-all-near-score 50
  (prop/for-all [spec gen-recurse-spec]
    (let [trace (p/simulate recursive-model (:args spec))
          s (eval-score trace)
          proj (eval-weight (p/project recursive-model trace sel/all))]
      (close? s proj 0.01))))

;; ---------------------------------------------------------------------------
;; Contramap
;; ---------------------------------------------------------------------------

(def inner-gf
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/gaussian x 1))]
        (mx/eval! y) (mx/item y)))))

(def contramap-identity (comb/contramap-gf inner-gf identity))
(def contramap-double (comb/contramap-gf inner-gf (fn [args] (mapv #(* 2 %) args))))

(def contramap-pool
  [{:gf contramap-identity :label "contramap(identity)"}
   {:gf contramap-double   :label "contramap(double)"}])

(def gen-contramap-spec (gen/elements contramap-pool))

(defspec contramap-identity-score-equals-inner 50
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate contramap-identity [arg] constraint)
          cm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [arg] constraint)
          inner-score (eval-score trace)]
      (close? cm-score inner-score 0.01))))

(defspec contramap-identity-retval-matches-inner 50
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate contramap-identity [arg] constraint)
          cm-retval (:retval trace)
          {:keys [trace]} (p/generate inner-gf [arg] constraint)
          inner-retval (:retval trace)]
      (close? cm-retval inner-retval 1e-6))))

(defspec contramap-generate-empty-weight-near-zero 50
  (prop/for-all [spec gen-contramap-spec
                 arg gen-arg]
    (let [{:keys [weight]} (p/generate (:gf spec) [arg] cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec contramap-update-same-weight-near-zero 50
  (prop/for-all [spec gen-contramap-spec
                 arg gen-arg]
    (let [trace (p/simulate (:gf spec) [arg])
          {:keys [weight]} (p/update (:gf spec) trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec contramap-double-score-matches-inner-doubled-arg 50
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate contramap-double [arg] constraint)
          cm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [(* 2 arg)] constraint)
          inner-score (eval-score trace)]
      (close? cm-score inner-score 0.01))))

;; ---------------------------------------------------------------------------
;; Dimap
;; ---------------------------------------------------------------------------

(def dimapped (comb/dimap inner-gf
                          (fn [args] (mapv #(* 2 %) args))
                          (fn [retval] (- retval))))

(defspec dimap-score-unchanged-vs-inner 50
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate dimapped [arg] constraint)
          dm-score (eval-score trace)
          {:keys [trace]} (p/generate inner-gf [(* 2 arg)] constraint)
          inner-score (eval-score trace)]
      (close? dm-score inner-score 0.01))))

(defspec dimap-retval-transformed-by-output-fn 50
  (prop/for-all [arg gen-arg
                 cval gen-constraint]
    (let [constraint (cm/choicemap :y (mx/scalar cval))
          {:keys [trace]} (p/generate dimapped [arg] constraint)
          dm-retval (:retval trace)
          {:keys [trace]} (p/generate inner-gf [(* 2 arg)] constraint)
          inner-retval (:retval trace)]
      (close? dm-retval (- inner-retval) 1e-6))))

(defspec dimap-generate-empty-weight-near-zero 50
  (prop/for-all [arg gen-arg]
    (let [{:keys [weight]} (p/generate dimapped [arg] cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec dimap-update-same-weight-near-zero 50
  (prop/for-all [arg gen-arg]
    (let [trace (p/simulate dimapped [arg])
          {:keys [weight]} (p/update dimapped trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(t/run-tests)
