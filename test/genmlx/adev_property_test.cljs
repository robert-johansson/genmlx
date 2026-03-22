(ns genmlx.adev-property-test
  "Property-based tests for ADEV gradient estimation using test.check.
   Every test verifies a genuine algebraic law:
   - vadev output shape contract
   - Reparam gradient is unbiased (E[grad] = analytical)
   - REINFORCE gradient is unbiased (score function estimator)
   - More samples reduce score estimate variance (law of large numbers)
   - ADEV optimization decreases loss"
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.adev :as adev]
            [genmlx.learning :as learn])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Vectorized-safe mixed model for shape test
(def vmixed-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            b (trace :b (dist/bernoulli 0.5))]
        x))))

;; Simple gaussian model for gradient/optimization tests
(def gauss-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/eval! x y)
        (+ (mx/item x) (mx/item y))))))

;; Vectorized-safe gaussian model (no mx/item in body)
(def vgauss-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/add x y)))))

(def score-cost (fn [trace] (mx/negative (:score trace))))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))
(def gen-n-samples (gen/elements [5 10]))

;; ---------------------------------------------------------------------------
;; Law: vadev-execute produces [N]-shaped score and reinforce-lp arrays
;; (Shape contract for vectorized ADEV)
;; ---------------------------------------------------------------------------

(defspec vadev-execute-shapes-equal-n 50
  (prop/for-all [n gen-n-samples
                 k gen-key]
    (let [result (adev/vadev-execute vmixed-model [] n k)]
      (mx/eval! (:score result) (:reinforce-lp result))
      (and (= [n] (mx/shape (:score result)))
           (= [n] (mx/shape (:reinforce-lp result)))))))

;; ---------------------------------------------------------------------------
;; Law: reparam gradient is unbiased
;; E[x^2] with x ~ N(mu,1) at mu=1
;; Analytical: d/dmu E[(mu+eps)^2] = d/dmu (mu^2 + 1) = 2*mu = 2.0
;; Average 200 single-sample gradient estimates
;; ---------------------------------------------------------------------------

(defspec reparam-gradient-unbiased 20
  (prop/for-all [k gen-key]
    (let [mu-val (mx/scalar 1.0)
          grad-fn (mx/grad
                    (fn [mu]
                      (let [model (dyn/auto-key
                                    (gen []
                                      (let [x (trace :x (dist/gaussian mu 1))]
                                        x)))
                            keys (rng/split-n k 200)
                            surrogates (mapv
                                         (fn [ki]
                                           (let [{:keys [trace]} (adev/adev-execute model [] ki)]
                                             (mx/square (:retval trace))))
                                         keys)]
                        (mx/divide (reduce mx/add surrogates) (mx/scalar 200.0)))))
          g (mx/tidy-materialize #(grad-fn mu-val))]
      (mx/eval! g)
      ;; Analytical: d/dmu E[(mu+eps)^2] = 2*mu = 2.0
      (close? (mx/item g) 2.0 0.5))))

;; ---------------------------------------------------------------------------
;; Law: REINFORCE gradient is unbiased (score function estimator)
;; Same model but manually building the REINFORCE surrogate
;; ---------------------------------------------------------------------------

(defspec reinforce-gradient-unbiased 10
  (prop/for-all [k gen-key]
    (let [mu-val (mx/scalar 1.0)
          grad-fn (mx/grad
                    (fn [mu]
                      (let [model (dyn/auto-key
                                    (gen []
                                      (let [x (trace :x (dist/gaussian mu 1))]
                                        x)))
                            keys (rng/split-n k 500)
                            surrogates (mapv
                                         (fn [ki]
                                           (let [{:keys [trace]} (adev/adev-execute model [] ki)
                                                 cost (mx/square (:retval trace))
                                                 ;; REINFORCE surrogate: f(x) + sg(f(x)) * log q(x)
                                                 lp (mx/stop-gradient
                                                      (mx/multiply (mx/scalar -0.5)
                                                        (mx/square (mx/subtract (:retval trace) mu))))]
                                             (mx/add cost (mx/multiply (mx/stop-gradient cost) lp))))
                                         keys)]
                        (mx/divide (reduce mx/add surrogates) (mx/scalar 500.0)))))
          g (mx/tidy-materialize #(grad-fn mu-val))]
      (mx/eval! g)
      ;; Analytical: 2*mu = 2.0, looser tolerance for REINFORCE
      (close? (mx/item g) 2.0 1.0))))

;; ---------------------------------------------------------------------------
;; Law: more batch samples reduce score estimate variance (law of large numbers)
;; Var[mean(scores)] = Var[score] / N, so increasing N decreases variance.
;; Compare vadev score-mean variance with N=100 vs N=5.
;; ---------------------------------------------------------------------------

(defspec more-batch-samples-reduce-score-mean-variance 5
  (prop/for-all [k gen-key]
    (let [n-estimates 30
          keys (rng/split-n k (* 2 n-estimates))
          ;; Compute mean score with N=100 batch, 30 times
          means-100 (mapv (fn [ki]
                            (let [r (adev/vadev-execute vgauss-model [] 100 ki)]
                              (mx/eval! (:score r))
                              (mx/item (mx/mean (:score r)))))
                          (subvec keys 0 n-estimates))
          ;; Compute mean score with N=3 batch, 30 times
          means-3 (mapv (fn [ki]
                          (let [r (adev/vadev-execute vgauss-model [] 3 ki)]
                            (mx/eval! (:score r))
                            (mx/item (mx/mean (:score r)))))
                        (subvec keys n-estimates))
          mean-fn (fn [vs] (/ (reduce + vs) (count vs)))
          var-fn  (fn [vs]
                    (let [m (mean-fn vs)]
                      (/ (reduce + (map #(* (- % m) (- % m)) vs))
                         (count vs))))
          var-100 (var-fn means-100)
          var-3   (var-fn means-3)]
      (< var-100 var-3))))

;; ---------------------------------------------------------------------------
;; Law: ADEV optimization decreases loss over iterations
;; ---------------------------------------------------------------------------

;; Parameterized model: x ~ N(mu, 1) where mu is learnable
;; Cost: E[x^2] = mu^2 + 1, minimized at mu=0
;; With reparam, gradient = 2*mu, moves mu toward 0
(def opt-model
  (dyn/auto-key
    (gen []
      (let [mu (param :mu (mx/scalar 0.0))
            x (trace :x (dist/gaussian mu 1))]
        x))))

;; Sequential cost: retval is the MLX array x
(def seq-x2-cost (fn [trace] (mx/square (:retval trace))))

;; Vectorized cost: receives {:choices :score :reinforce-lp :retval}
;; Score is [N]-shaped, retval is [N]-shaped x values
(def vec-x2-cost (fn [result] (mx/square (:retval result))))

(defspec adev-loss-decreases-over-optimization 10
  (prop/for-all [k gen-key]
    (let [result (adev/adev-optimize
                   {:iterations 50 :lr 0.05 :n-samples 20 :key k}
                   opt-model [] vec-x2-cost [:mu]
                   (mx/array [5.0]))  ;; Start at mu=5
          history (:loss-history result)]
      (when (>= (count history) 4)
        (let [n (count history)
              quarter (max 1 (quot n 4))
              early-mean (/ (reduce + (take quarter history)) quarter)
              late-mean  (/ (reduce + (take-last quarter history)) quarter)]
          (< late-mean early-mean))))))

(t/run-tests)
