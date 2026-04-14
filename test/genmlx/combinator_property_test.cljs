(ns genmlx.combinator-property-test
  "Property-based combinator invariant tests using test.check.
   Verifies Map, Unfold, Switch, Scan, Mask, Contramap, MapRetval, Mix,
   and Dimap structure and score invariants.

   Uses gen/elements with pre-built combinator instances to avoid
   SCI interop crashes during test.check shrink traversal."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
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

(defn- sum-meta-scores
  "Sum scores from metadata vector, returning JS number."
  [scores]
  (reduce (fn [acc s] (mx/eval! s) (+ acc (mx/item s)))
          0.0 scores))

;; ---------------------------------------------------------------------------
;; Shared kernel
;; ---------------------------------------------------------------------------

(def kernel
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/gaussian x 1))]
        (mx/eval! y)
        (mx/item y)))))

;; ---------------------------------------------------------------------------
;; Pre-built combinator pools
;; ---------------------------------------------------------------------------

(def map-pool
  [{:n 1 :args [[1.0]]        :label "map(n=1)"}
   {:n 3 :args [[1.0 2.0 3.0]] :label "map(n=3)"}
   {:n 5 :args [[1.0 2.0 3.0 4.0 5.0]] :label "map(n=5)"}])

(def gen-map-spec (gen/elements map-pool))
(def mapped (comb/map-combinator kernel))

(def unfold-step
  (dyn/auto-key
    (gen [t state]
      (let [y (trace :y (dist/gaussian state 1))]
        (mx/eval! y)
        (mx/item y)))))

(def unfolded (comb/unfold-combinator unfold-step))

(def unfold-pool
  [{:n 1 :args [1 0.0] :label "unfold(n=1)"}
   {:n 3 :args [3 0.0] :label "unfold(n=3)"}
   {:n 5 :args [5 0.0] :label "unfold(n=5)"}])

(def gen-unfold-spec (gen/elements unfold-pool))

(def switch-g1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1)))))
(def switch-g2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 5 2)))))
(def switched (comb/switch-combinator switch-g1 switch-g2))

(def switch-pool
  [{:idx 0 :args [0] :label "switch(idx=0)"}
   {:idx 1 :args [1] :label "switch(idx=1)"}])

(def gen-switch-spec (gen/elements switch-pool))

(def scan-kernel
  (dyn/auto-key
    (gen [carry x]
      (let [y (trace :y (dist/gaussian carry 1))]
        (mx/eval! y)
        [(mx/item y) (mx/item y)]))))

(def scanned (comb/scan-combinator scan-kernel))

(def scan-pool
  [{:n 1 :args [0.0 [1.0]]           :label "scan(n=1)"}
   {:n 3 :args [0.0 [1.0 2.0 3.0]]   :label "scan(n=3)"}])

(def gen-scan-spec (gen/elements scan-pool))

(def masked (comb/mask-combinator kernel))

(def mask-pool
  [{:active? true  :args [true 3.0]  :label "mask(true)"}
   {:active? false :args [false 3.0] :label "mask(false)"}])

(def gen-mask-spec (gen/elements mask-pool))

(def contramapped
  (comb/contramap-gf kernel (fn [args] [(+ (first args) 1.0)])))

(def retval-mapped
  (comb/map-retval kernel (fn [v] (* v 2.0))))

(def kernel-wide
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/gaussian x 2))]
        (mx/eval! y)
        (mx/item y)))))

(def mixed
  (comb/mix-combinator [kernel kernel-wide] (mx/array [0.0 0.0])))

(def dimapped
  (comb/dimap kernel
              (fn [args] [(+ (first args) 1.0)])
              (fn [v] (* v 2.0))))

;; ---------------------------------------------------------------------------
;; Properties
;; ---------------------------------------------------------------------------

(defspec map-n-integer-keyed-sub-traces 50
  (prop/for-all [spec gen-map-spec]
    (let [trace (p/simulate mapped (:args spec))
          choices (:choices trace)
          n (:n spec)]
      (every? (fn [i]
                (not= (cm/get-submap choices i) cm/EMPTY))
              (range n)))))

(defspec map-score-equals-sum-element-scores 50
  (prop/for-all [spec gen-map-spec]
    (let [trace (p/simulate mapped (:args spec))
          total (eval-score trace)
          element-scores (::comb/element-scores (meta trace))]
      (if element-scores
        (close? total (sum-meta-scores element-scores) 0.01)
        true))))

(defspec map-generate-empty-weight-approx-zero 50
  (prop/for-all [spec gen-map-spec]
    (let [{:keys [weight]} (p/generate mapped (:args spec) cm/EMPTY)]
      (close? 0.0 (eval-weight weight) 0.01))))

(defspec map-generate-full-weight-approx-score 50
  (prop/for-all [spec gen-map-spec]
    (let [trace (p/simulate mapped (:args spec))
          {:keys [trace weight]} (p/generate mapped (:args spec) (:choices trace))
          w (eval-weight weight)
          s (eval-score trace)]
      (close? s w 0.01))))

(defspec map-update-same-weight-approx-zero 50
  (prop/for-all [spec gen-map-spec]
    (let [trace (p/simulate mapped (:args spec))
          {:keys [weight]} (p/update mapped trace (:choices trace))]
      (close? 0.0 (eval-weight weight) 0.01))))

(defspec unfold-n-step-sub-traces 50
  (prop/for-all [spec gen-unfold-spec]
    (let [trace (p/simulate unfolded (:args spec))
          choices (:choices trace)
          n (:n spec)]
      (every? (fn [i]
                (not= (cm/get-submap choices i) cm/EMPTY))
              (range n)))))

(defspec unfold-score-equals-sum-step-scores 50
  (prop/for-all [spec gen-unfold-spec]
    (let [trace (p/simulate unfolded (:args spec))
          total (eval-score trace)
          step-scores (::comb/step-scores (meta trace))]
      (if step-scores
        (close? total (sum-meta-scores step-scores) 0.01)
        true))))

(defspec unfold-gfi-contracts 50
  (prop/for-all [spec gen-unfold-spec]
    (let [{:keys [weight]} (p/generate unfolded (:args spec) cm/EMPTY)
          gen-w (eval-weight weight)
          trace (p/simulate unfolded (:args spec))
          {:keys [weight]} (p/update unfolded trace (:choices trace))
          upd-w (eval-weight weight)]
      (and (close? 0.0 gen-w 0.01)
           (close? 0.0 upd-w 0.01)))))

(defspec switch-gfi-contracts 50
  (prop/for-all [spec gen-switch-spec]
    (let [{:keys [weight]} (p/generate switched (:args spec) cm/EMPTY)
          gen-w (eval-weight weight)
          trace (p/simulate switched (:args spec))
          s (eval-score trace)
          {:keys [trace weight]} (p/generate switched (:args spec) (:choices trace))
          full-w (eval-weight weight)
          full-s (eval-score trace)]
      (and (close? 0.0 gen-w 0.01)
           (close? full-s full-w 0.01)))))

(defspec scan-retval-structure 50
  (prop/for-all [spec gen-scan-spec]
    (let [trace (p/simulate scanned (:args spec))
          retval (:retval trace)]
      (and (contains? retval :carry)
           (contains? retval :outputs)
           (= (:n spec) (count (:outputs retval)))))))

(defspec scan-score-equals-sum-step-scores 50
  (prop/for-all [spec gen-scan-spec]
    (let [trace (p/simulate scanned (:args spec))
          total (eval-score trace)
          step-scores (::comb/step-scores (meta trace))]
      (if step-scores
        (close? total (sum-meta-scores step-scores) 0.01)
        true))))

(defspec map-n1-score-approx-kernel-score 50
  (prop/for-all [_ (gen/return nil)]
    (let [constraint-val (mx/scalar 2.5)
          map-constraint (cm/set-choice cm/EMPTY [0] (cm/choicemap :y constraint-val))
          kernel-constraint (cm/choicemap :y constraint-val)
          {:keys [trace]} (p/generate mapped [[3.0]] map-constraint)
          map-score (eval-score trace)
          {:keys [trace]} (p/generate kernel [3.0] kernel-constraint)
          kernel-score (eval-score trace)]
      (close? kernel-score map-score 0.01))))

(defspec mask-true-is-kernel-false-score-zero 50
  (prop/for-all [spec gen-mask-spec]
    (let [trace (p/simulate masked (:args spec))
          s (eval-score trace)]
      (if (:active? spec)
        (finite? s)
        (close? 0.0 s 0.01)))))

(defspec all-combinators-project-all-approx-score 50
  (prop/for-all [which (gen/elements [:map :unfold :switch :scan :mask-true
                                      :contramap :map-retval :mix])]
    (let [[gf args] (case which
                      :map        [mapped [[1.0 2.0]]]
                      :unfold     [unfolded [2 0.0]]
                      :switch     [switched [0]]
                      :scan       [scanned [0.0 [1.0 2.0]]]
                      :mask-true  [masked [true 3.0]]
                      :contramap  [contramapped [2.0]]
                      :map-retval [retval-mapped [3.0]]
                      :mix        [mixed [3.0]])
          trace (p/simulate gf args)
          s (eval-score trace)
          proj (eval-weight (p/project gf trace sel/all))]
      (close? s proj 0.01))))

(defspec all-combinators-project-none-approx-zero 50
  (prop/for-all [which (gen/elements [:map :unfold :switch :scan :mask-true
                                      :contramap :map-retval :mix])]
    (let [[gf args] (case which
                      :map        [mapped [[1.0 2.0]]]
                      :unfold     [unfolded [2 0.0]]
                      :switch     [switched [0]]
                      :scan       [scanned [0.0 [1.0 2.0]]]
                      :mask-true  [masked [true 3.0]]
                      :contramap  [contramapped [2.0]]
                      :map-retval [retval-mapped [3.0]]
                      :mix        [mixed [3.0]])
          trace (p/simulate gf args)
          proj (eval-weight (p/project gf trace sel/none))]
      (close? 0.0 proj 0.01))))

(defspec dimap-regenerate-all-yields-finite-weight 50
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate dimapped [2.0])
          {:keys [trace weight]} (p/regenerate dimapped trace sel/all)
          w (eval-weight weight)
          s (eval-score trace)]
      (and (finite? w) (finite? s)))))

(defspec dimap-regenerate-none-preserves-trace 50
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate dimapped [2.0])
          original-score (eval-score trace)
          {:keys [trace weight]} (p/regenerate dimapped trace sel/none)
          regen-score (eval-score trace)
          w (eval-weight weight)]
      (and (close? original-score regen-score 0.01)
           (close? 0.0 w 0.01)))))

(t/run-tests)
