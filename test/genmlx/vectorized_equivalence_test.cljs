(ns genmlx.vectorized-equivalence-test
  "Vectorized equivalence: vsimulate produces N independent samples
   with correct distributional properties. ESS computation."
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
;; Test model: x ~ N(0,1), known analytical score = log N(x; 0, 1)
;; ---------------------------------------------------------------------------

(def gaussian-model
  (dyn/auto-key
   (gen []
     (trace :x (dist/gaussian 0 1)))))

(def two-site-model
  (dyn/auto-key
   (gen []
     (let [x (trace :x (dist/gaussian 0 1))
           y (trace :y (dist/gaussian x 1))]
       y))))

(def N 200)

;; ---------------------------------------------------------------------------
;; vsimulate produces N distinct samples
;; ---------------------------------------------------------------------------

(deftest vsimulate-produces-n-scores
  (let [scores (h/realize-vec (:score (dyn/vsimulate gaussian-model [] N
                                                     (h/deterministic-key))))]
    (is (= N (count scores)))))

(deftest vsimulate-scores-have-positive-variance
  (testing "N particles produce diverse scores (not all identical)"
    (let [scores (h/realize-vec (:score (dyn/vsimulate gaussian-model [] N
                                                       (h/deterministic-key))))
          variance (h/sample-variance scores)]
      (is (pos? variance)
          "vsimulate scores are not all identical"))))

(deftest vsimulate-choices-have-positive-variance
  (testing "N particles produce diverse choice values"
    (let [vtr (dyn/vsimulate gaussian-model [] N (h/deterministic-key))
          x-vals (h/realize-vec (cm/get-value (cm/get-submap (:choices vtr) :x)))
          variance (h/sample-variance x-vals)]
      (is (pos? variance)
          "choice values are diverse"))))

;; ---------------------------------------------------------------------------
;; Score consistency: each particle's score matches analytical log-prob
;; ---------------------------------------------------------------------------

(deftest vsimulate-scores-match-analytical-log-prob
  (testing "each particle score = log N(x_i; 0, 1)"
    (let [vtr (dyn/vsimulate gaussian-model [] 16 (h/deterministic-key))
          x-vals (h/realize-vec (cm/get-value (cm/get-submap (:choices vtr) :x)))
          scores (h/realize-vec (:score vtr))]
      (doseq [[x-val score] (map vector x-vals scores)]
        (is (h/close? (h/gaussian-lp x-val 0 1) score 1e-4)
            (str "score matches log N(" x-val "; 0, 1)"))))))

(deftest vsimulate-two-site-scores-match-analytical
  (testing "each particle score = log N(x; 0,1) + log N(y; x,1)"
    (let [vtr (dyn/vsimulate two-site-model [] 16 (h/deterministic-key))
          {:keys [choices score]} vtr
          x-vals (h/realize-vec (cm/get-value (cm/get-submap choices :x)))
          y-vals (h/realize-vec (cm/get-value (cm/get-submap choices :y)))
          scores (h/realize-vec score)]
      (doseq [[x-val y-val sc] (map vector x-vals y-vals scores)]
        (let [expected (+ (h/gaussian-lp x-val 0 1)
                          (h/gaussian-lp y-val x-val 1))]
          (is (h/close? expected sc 1e-4)
              "two-site score = joint log-prob"))))))

;; ---------------------------------------------------------------------------
;; ESS computation
;; ---------------------------------------------------------------------------

(defn log-ess
  "Effective sample size from log-weights (seq of JS numbers).
   ESS = 1 / Σ(w_i²) where w_i = exp(lw_i - max) / Σ exp(lw_j - max)."
  [log-weights]
  (let [lw (vec log-weights)
        max-lw (apply max lw)
        shifted (mapv #(js/Math.exp (- % max-lw)) lw)
        total (reduce + shifted)
        normalized (mapv #(/ % total) shifted)
        sum-sq (reduce + (mapv #(* % %) normalized))]
    (/ 1.0 sum-sq)))

(deftest ess-uniform-weights-equal-n
  (let [ess (log-ess (repeat 50 0.0))]
    (is (h/close? 50.0 ess 1e-6)
        "uniform weights → ESS = N")))

(deftest ess-single-dominant-weight-near-one
  (let [ess (log-ess (into [100.0] (repeat 49 -100.0)))]
    (is (< ess 1.01)
        "single dominant weight → ESS ≈ 1")))

(deftest ess-bounded-by-n
  (testing "ESS from vgenerate importance weights ≤ N"
    (let [constraints (cm/choicemap :x (mx/scalar 0.0))
          vtr (dyn/vgenerate two-site-model [] constraints N (h/deterministic-key))
          ;; Use scores as proxy log-weights (since weight is scalar)
          scores (h/realize-vec (:score vtr))
          ess (log-ess scores)]
      (is (<= ess (+ N 0.01))
          "ESS ≤ N"))))

;; ---------------------------------------------------------------------------
;; Deterministic reproducibility
;; ---------------------------------------------------------------------------

(deftest vsimulate-same-key-reproduces-scores
  (testing "same PRNG key → identical scores"
    (let [scores1 (h/realize-vec (:score (dyn/vsimulate gaussian-model [] 8
                                                        (rng/fresh-key 42))))
          scores2 (h/realize-vec (:score (dyn/vsimulate gaussian-model [] 8
                                                        (rng/fresh-key 42))))]
      (is (h/all-close? scores1 scores2 1e-6)))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
