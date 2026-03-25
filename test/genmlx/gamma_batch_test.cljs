(ns genmlx.gamma-batch-test
  "Tests for gamma, inv-gamma, beta, and dirichlet batch sampling."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Gamma batch sampling
;; ---------------------------------------------------------------------------

(deftest gamma-alpha-2-rate-1
  (testing "gamma alpha=2.0, rate=1.0"
    (let [d (dist/gamma-dist (mx/scalar 2.0) (mx/scalar 1.0))
          n 2000
          samples (dc/dist-sample-n d (rng/fresh-key 42) n)
          _ (mx/eval! samples)
          sh (mx/shape samples)
          vals (mx/->clj samples)
          mean (/ (reduce + vals) (count vals))
          variance (/ (reduce + (map #(* (- % mean) (- % mean)) vals)) (count vals))]
      (is (= sh [n]) "shape is [2000]")
      (is (every? pos? vals) "all samples > 0")
      (is (h/close? 2.0 mean 0.15) "mean ~ 2.0")
      (is (h/close? 2.0 variance 0.3) "variance ~ 2.0"))))

(deftest gamma-alpha-10-rate-2
  (testing "gamma alpha=10.0, rate=2.0"
    (let [d (dist/gamma-dist (mx/scalar 10.0) (mx/scalar 2.0))
          n 2000
          samples (dc/dist-sample-n d (rng/fresh-key 123) n)
          _ (mx/eval! samples)
          vals (mx/->clj samples)
          mean (/ (reduce + vals) (count vals))
          variance (/ (reduce + (map #(* (- % mean) (- % mean)) vals)) (count vals))]
      (is (every? pos? vals) "all samples > 0")
      (is (h/close? 5.0 mean 0.2) "mean ~ 5.0")
      (is (h/close? 2.5 variance 0.4) "variance ~ 2.5"))))

(deftest gamma-alpha-1-rate-1
  (testing "gamma alpha=1.0, rate=1.0 (exponential)"
    (let [d (dist/gamma-dist (mx/scalar 1.0) (mx/scalar 1.0))
          n 2000
          samples (dc/dist-sample-n d (rng/fresh-key 77) n)
          _ (mx/eval! samples)
          vals (mx/->clj samples)
          mean (/ (reduce + vals) (count vals))]
      (is (every? pos? vals) "all samples > 0")
      (is (h/close? 1.0 mean 0.1) "mean ~ 1.0"))))

(deftest gamma-alpha-0-5-rate-1
  (testing "gamma alpha=0.5, rate=1.0 (Ahrens-Dieter path)"
    (let [d (dist/gamma-dist (mx/scalar 0.5) (mx/scalar 1.0))
          n 2000
          samples (dc/dist-sample-n d (rng/fresh-key 99) n)
          _ (mx/eval! samples)
          vals (mx/->clj samples)
          mean (/ (reduce + vals) (count vals))
          variance (/ (reduce + (map #(* (- % mean) (- % mean)) vals)) (count vals))]
      (is (every? pos? vals) "all samples > 0")
      (is (h/close? 0.5 mean 0.1) "mean ~ 0.5")
      (is (h/close? 0.5 variance 0.15) "variance ~ 0.5"))))

;; ---------------------------------------------------------------------------
;; Inv-gamma batch sampling
;; ---------------------------------------------------------------------------

(deftest inv-gamma-shape-3-scale-2
  (testing "inv-gamma shape=3.0, scale=2.0"
    (let [d (dist/inv-gamma (mx/scalar 3.0) (mx/scalar 2.0))
          n 2000
          samples (dc/dist-sample-n d (rng/fresh-key 55) n)
          _ (mx/eval! samples)
          sh (mx/shape samples)
          vals (mx/->clj samples)
          mean (/ (reduce + vals) (count vals))]
      (is (= sh [n]) "shape is [2000]")
      (is (every? pos? vals) "all samples > 0")
      (is (h/close? 1.0 mean 0.15) "mean ~ 1.0"))))

;; ---------------------------------------------------------------------------
;; Beta batch sampling
;; ---------------------------------------------------------------------------

(deftest beta-alpha-2-beta-5
  (testing "beta alpha=2.0, beta=5.0"
    (let [d (dist/beta-dist (mx/scalar 2.0) (mx/scalar 5.0))
          n 2000
          samples (dc/dist-sample-n d (rng/fresh-key 33) n)
          _ (mx/eval! samples)
          sh (mx/shape samples)
          vals (mx/->clj samples)
          mean (/ (reduce + vals) (count vals))]
      (is (= sh [n]) "shape is [2000]")
      (is (every? #(and (pos? %) (< % 1.0)) vals) "all samples in (0,1)")
      (is (h/close? (/ 2.0 7.0) mean 0.03) "mean ~ 0.2857"))))

(deftest beta-alpha-0-5-beta-0-5
  (testing "beta alpha=0.5, beta=0.5 (U-shaped)"
    (let [d (dist/beta-dist (mx/scalar 0.5) (mx/scalar 0.5))
          n 2000
          samples (dc/dist-sample-n d (rng/fresh-key 44) n)
          _ (mx/eval! samples)
          vals (mx/->clj samples)
          mean (/ (reduce + vals) (count vals))]
      (is (every? #(and (>= % 0.0) (<= % 1.0)) vals) "all samples in [0,1]")
      (is (h/close? 0.5 mean 0.05) "mean ~ 0.5"))))

;; ---------------------------------------------------------------------------
;; Dirichlet batch sampling
;; ---------------------------------------------------------------------------

(deftest dirichlet-batch
  (testing "dirichlet [2.0, 3.0, 5.0]"
    (let [d (dist/dirichlet (mx/array [2.0 3.0 5.0]))
          n 2000
          samples (dc/dist-sample-n d (rng/fresh-key 66) n)
          _ (mx/eval! samples)
          sh (mx/shape samples)]
      (is (= sh [n 3]) "shape is [2000 3]")
      (let [row-sums (mx/sum samples [1])
            _ (mx/eval! row-sums)
            sums (mx/->clj row-sums)]
        (is (every? #(< (js/Math.abs (- % 1.0)) 0.01) sums) "all rows sum to ~1.0"))
      (let [col-means (mx/divide (mx/sum samples [0]) (mx/scalar (float n)))
            _ (mx/eval! col-means)
            means (mx/->clj col-means)]
        (is (h/close? 0.2 (nth means 0) 0.03) "mean[0] ~ 0.2")
        (is (h/close? 0.3 (nth means 1) 0.03) "mean[1] ~ 0.3")
        (is (h/close? 0.5 (nth means 2) 0.03) "mean[2] ~ 0.5")))))

;; ---------------------------------------------------------------------------
;; vgenerate benchmark: Bayesian model with gamma/beta priors
;; ---------------------------------------------------------------------------

(deftest vgenerate-gamma-beta-model
  (testing "vgenerate: Bayesian model with gamma/beta priors (N=200)"
    (let [model (gen []
                  (let [noise-prec (trace :noise-prec (dist/gamma-dist (mx/scalar 2.0) (mx/scalar 1.0)))
                        noise-std  (mx/divide (mx/scalar 1.0) (mx/sqrt noise-prec))
                        weight     (trace :weight (dist/gaussian 0 5))
                        bias       (trace :bias (dist/gaussian 0 5))]
                    (trace :y0 (dist/gaussian (mx/add bias weight) noise-std))
                    (trace :y1 (dist/gaussian (mx/add bias (mx/multiply weight (mx/scalar 2.0))) noise-std))
                    (trace :y2 (dist/gaussian (mx/add bias (mx/multiply weight (mx/scalar 3.0))) noise-std))
                    nil))
          obs (cm/choicemap :y0 (mx/scalar 2.1) :y1 (mx/scalar 3.9) :y2 (mx/scalar 6.2))
          n 200
          ;; Warm up both paths
          _ (do (let [{:keys [trace weight]} (p/generate (dyn/auto-key model) [] obs)]
                  (mx/eval! (:score trace) weight))
                (let [vt (dyn/vgenerate model [] obs n nil)]
                  (mx/eval! (:score vt) (:weight vt))))
          ;; Sequential
          t0 (js/Date.now)
          _ (doseq [_ (range n)]
              (let [{:keys [trace weight]} (p/generate (dyn/auto-key model) [] obs)]
                (mx/eval! (:score trace) weight)))
          t1 (js/Date.now)
          ;; Batched
          t2 (js/Date.now)
          vt (dyn/vgenerate model [] obs n nil)
          _ (mx/eval! (:score vt) (:weight vt))
          t3 (js/Date.now)]
      (is (some? vt) "vgenerate produces result"))))

(cljs.test/run-tests)
