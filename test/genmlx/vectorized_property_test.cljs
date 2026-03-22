(ns genmlx.vectorized-property-test
  "Property-based vectorized inference tests using test.check.
   Verifies vsimulate/vgenerate shape invariants, statistical equivalence,
   VectorizedTrace operations, and batched vs scalar consistency."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

(defn- shape= [arr expected-shape]
  (= (mx/shape arr) expected-shape))

;; ---------------------------------------------------------------------------
;; Models and fixture pools
;; ---------------------------------------------------------------------------

;; Independent model -- for vsimulate tests
(def ind-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))
          y (trace :y (dist/gaussian 0 1))]
      (mx/add x y))))

;; Dependent model -- for vgenerate tests (y depends on x)
;; This ensures weight is [N]-shaped when :y is constrained and :x is unconstrained
(def dep-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))
          y (trace :y (dist/gaussian x 1))]
      y)))

(def n-pool [5 10 15 20])
(def gen-n (gen/elements n-pool))

;; Partial obs (constrain only :y, leave :x free) -> [N]-shaped weight
(def partial-obs-pool
  [(cm/choicemap :y (mx/scalar 1.0))
   (cm/choicemap :y (mx/scalar 0.0))
   (cm/choicemap :y (mx/scalar -1.0))
   (cm/choicemap :y (mx/scalar 2.0))])

(def gen-partial-obs (gen/elements partial-obs-pool))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

;; ---------------------------------------------------------------------------
;; vsimulate Shape (3)
;; ---------------------------------------------------------------------------

(defspec vsimulate-score-shape-is-n 50
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vsimulate ind-model [] n k)]
      (shape= (:score vt) [n]))))

(defspec vsimulate-all-choice-leaves-are-n-shaped 50
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vsimulate ind-model [] n k)
          choices (:choices vt)]
      (and (shape= (cm/get-value (cm/get-submap choices :x)) [n])
           (shape= (cm/get-value (cm/get-submap choices :y)) [n])))))

(defspec vsimulate-n-particles-matches-n 50
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vsimulate ind-model [] n k)]
      (= (:n-particles vt) n))))

;; ---------------------------------------------------------------------------
;; vgenerate Shape (4)
;; ---------------------------------------------------------------------------

(defspec vgenerate-weight-shape-is-n 50
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)]
      (shape= (:weight vt) [n]))))

(defspec vgenerate-constrained-site-is-scalar 50
  (prop/for-all [n gen-n
                 k gen-key]
    (let [obs (cm/choicemap :y (mx/scalar 2.0))
          vt (dyn/vgenerate dep-model [] obs n k)
          y-val (cm/get-value (cm/get-submap (:choices vt) :y))
          _ (mx/eval! y-val)]
      ;; Constrained site is scalar
      (close? 2.0 (mx/realize y-val) 1e-6))))

(defspec vgenerate-unconstrained-sites-are-n-shaped 50
  (prop/for-all [n gen-n
                 k gen-key]
    (let [obs (cm/choicemap :y (mx/scalar 1.0))
          vt (dyn/vgenerate dep-model [] obs n k)
          x-arr (cm/get-value (cm/get-submap (:choices vt) :x))]
      (shape= x-arr [n]))))

(defspec vgenerate-empty-constraints-weight-is-scalar-0 50
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vgenerate ind-model [] cm/EMPTY n k)
          w (:weight vt)
          _ (mx/eval! w)]
      ;; Empty constraints -> weight = 0 (scalar)
      (close? 0.0 (mx/item w) 0.01))))

;; ---------------------------------------------------------------------------
;; Statistical Equivalence (1)
;; ---------------------------------------------------------------------------

(defspec vsimulate-mean-score-near-mean-of-n-sequential-simulates 30
  (prop/for-all [k gen-key]
    (let [n 20
          ;; Vectorized
          vt (dyn/vsimulate ind-model [] n k)
          _ (mx/eval! (:score vt))
          v-scores (mx/->clj (:score vt))
          v-mean (/ (reduce + v-scores) n)
          ;; Sequential
          seq-scores (mapv (fn [_]
                             (let [t (p/simulate (dyn/auto-key ind-model) [])
                                   _ (mx/eval! (:score t))]
                               (mx/item (:score t))))
                           (range n))
          s-mean (/ (reduce + seq-scores) n)]
      ;; Both should be around -log(2pi) ~ -1.84 for standard Gaussian
      ;; Loose tolerance since different random draws
      (and (finite? v-mean) (finite? s-mean)
           (< (js/Math.abs (- v-mean s-mean)) 3.0)))))

;; ---------------------------------------------------------------------------
;; VectorizedTrace Operations (3)
;; ---------------------------------------------------------------------------

(defspec vtrace-ess-in-0-n 50
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          ess (vec/vtrace-ess vt)]
      (and (> ess 0) (<= ess (+ n 0.01))))))

(defspec resample-vtrace-produces-near-uniform-weights 50
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          [k1 _] (rng/split k)
          resampled (vec/resample-vtrace vt k1)
          _ (mx/eval! (:weight resampled))
          ws (mx/->clj (:weight resampled))]
      ;; After resampling, weights should be uniform (zeros in log-space)
      (every? #(close? 0.0 % 0.01) ws))))

(defspec resample-vtrace-preserves-n-particles 50
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          [k1 _] (rng/split k)
          resampled (vec/resample-vtrace vt k1)]
      (= (:n-particles resampled) n))))

;; ---------------------------------------------------------------------------
;; Batched vs Scalar (2)
;; ---------------------------------------------------------------------------

(defspec n-1-score-and-weight-shapes-are-1 50
  (prop/for-all [k gen-key
                 obs gen-partial-obs]
    (let [vt-sim (dyn/vsimulate ind-model [] 1 k)
          vt-gen (dyn/vgenerate dep-model [] obs 1 k)]
      (and (shape= (:score vt-sim) [1])
           (shape= (:weight vt-gen) [1])))))

(defspec resample-preserves-score-shape-n 50
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          [k1 _] (rng/split k)
          resampled (vec/resample-vtrace vt k1)]
      (shape= (:score resampled) [n]))))

(t/run-tests)
