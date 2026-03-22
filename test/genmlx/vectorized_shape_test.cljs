(ns genmlx.vectorized-shape-test
  "Vectorized execution: shape correctness for vsimulate and vgenerate.
   vsimulate: choices [N]-shaped, score [N]-shaped.
   vgenerate: unconstrained choices [N]-shaped, constrained choices scalar."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def single-gaussian
  (dyn/auto-key
   (gen [mu]
     (trace :x (dist/gaussian mu 1)))))

(def two-site-model
  (dyn/auto-key
   (gen []
     (let [x (trace :x (dist/gaussian 0 1))
           y (trace :y (dist/gaussian x 1))]
       y))))

(def N 16)

;; ---------------------------------------------------------------------------
;; vsimulate shape tests
;; ---------------------------------------------------------------------------

(deftest vsimulate-choices-are-n-shaped
  (let [vtr (dyn/vsimulate single-gaussian [(mx/scalar 0.0)] N (h/deterministic-key))
        x-val (cm/get-value (cm/get-submap (:choices vtr) :x))]
    (mx/eval! x-val)
    (is (= [N] (mx/shape x-val))
        "choice :x has shape [N]")))

(deftest vsimulate-score-is-n-shaped
  (let [vtr (dyn/vsimulate single-gaussian [(mx/scalar 0.0)] N (h/deterministic-key))]
    (mx/eval! (:score vtr))
    (is (= [N] (mx/shape (:score vtr))))))

(deftest vsimulate-multi-site-all-n-shaped
  (testing "multi-site model: all choices and score [N]-shaped"
    (let [{:keys [choices score]} (dyn/vsimulate two-site-model [] N (h/deterministic-key))
          x-val (cm/get-value (cm/get-submap choices :x))
          y-val (cm/get-value (cm/get-submap choices :y))]
      (mx/eval! x-val)
      (mx/eval! y-val)
      (mx/eval! score)
      (is (= [N] (mx/shape x-val)) ":x is [N]-shaped")
      (is (= [N] (mx/shape y-val)) ":y is [N]-shaped")
      (is (= [N] (mx/shape score)) "score is [N]-shaped"))))

(deftest vsimulate-n-particles-field-matches
  (let [vtr (dyn/vsimulate single-gaussian [(mx/scalar 0.0)] N (h/deterministic-key))]
    (is (= N (:n-particles vtr)))))

(deftest vsimulate-scores-are-finite
  (let [scores (h/realize-vec (:score (dyn/vsimulate single-gaussian [(mx/scalar 0.0)]
                                                     N (h/deterministic-key))))]
    (is (every? js/isFinite scores)
        "all particle scores finite")))

;; ---------------------------------------------------------------------------
;; vgenerate shape tests
;; ---------------------------------------------------------------------------

(deftest vgenerate-unconstrained-choices-are-n-shaped
  (testing "unconstrained site in vgenerate has [N]-shaped values"
    (let [constraints (cm/choicemap :x (mx/scalar 0.0))
          vtr (dyn/vgenerate two-site-model [] constraints N (h/deterministic-key))
          y-val (cm/get-value (cm/get-submap (:choices vtr) :y))]
      (mx/eval! y-val)
      (is (= [N] (mx/shape y-val))
          "unconstrained :y is [N]-shaped"))))

(deftest vgenerate-no-constraints-score-is-n-shaped
  (testing "no constraints: all sites sampled [N]-shaped"
    (let [vtr (dyn/vgenerate two-site-model [] cm/EMPTY N (h/deterministic-key))]
      (mx/eval! (:score vtr))
      (is (= [N] (mx/shape (:score vtr)))))))

(deftest vgenerate-partial-constraint-score-is-n-shaped
  (testing "partial constraints: score [N]-shaped from unconstrained sites"
    (let [constraints (cm/choicemap :x (mx/scalar 0.0))
          vtr (dyn/vgenerate two-site-model [] constraints N (h/deterministic-key))]
      (mx/eval! (:score vtr))
      (is (= [N] (mx/shape (:score vtr)))))))

;; ---------------------------------------------------------------------------
;; VectorizedTrace is a proper record
;; ---------------------------------------------------------------------------

(deftest vsimulate-produces-vectorized-trace
  (let [vtr (dyn/vsimulate single-gaussian [(mx/scalar 0.0)] N (h/deterministic-key))]
    (is (instance? vec/VectorizedTrace vtr)
        "vsimulate returns VectorizedTrace")))

(deftest vgenerate-produces-vectorized-trace
  (let [vtr (dyn/vgenerate single-gaussian [(mx/scalar 0.0)] cm/EMPTY N (h/deterministic-key))]
    (is (instance? vec/VectorizedTrace vtr)
        "vgenerate returns VectorizedTrace")))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
