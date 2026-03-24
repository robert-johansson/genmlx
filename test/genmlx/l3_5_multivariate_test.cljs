(ns genmlx.l3-5-multivariate-test
  "Level 3.5 WP-3: Multivariate conjugacy test suite.
   Tests MVN-MVN conjugate update, auto-handler wiring, condition number guard,
   d=1 scalar consistency, and dimension scaling benchmark."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.gen :refer [gen]]
            [genmlx.inference.auto-analytical :as auto-analytical]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- deterministic-weight? [weights]
  (apply = weights))

(defn- weight-variance [weights]
  (let [n (count weights)
        mean (/ (reduce + weights) n)]
    (/ (reduce + (map #(* (- % mean) (- % mean)) weights)) n)))

(defn- eye [d]
  (let [vals (for [i (range d) j (range d)] (if (= i j) 1.0 0.0))]
    (mx/reshape (mx/array (vec vals)) [d d])))

(defn- scale-eye [d s]
  (let [vals (for [i (range d) j (range d)] (if (= i j) s 0.0))]
    (mx/reshape (mx/array (vec vals)) [d d])))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def mvn-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/array [0 0])
                            (mx/reshape (mx/array [10 0 0 10]) [2 2])))
            y  (trace :y (dist/multivariate-normal
                            mu
                            (mx/reshape (mx/array [1 0 0 1]) [2 2])))]
        y))))

(def mvn-1d-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/array [0])
                            (mx/reshape (mx/array [100]) [1 1])))
            y  (trace :y (dist/multivariate-normal
                            mu
                            (mx/reshape (mx/array [1]) [1 1])))]
        y))))

(def scalar-nn-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
            y  (trace :y (dist/gaussian mu (mx/scalar 1.0)))]
        y))))

(def mvn-multi-obs-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/array [0 0])
                            (scale-eye 2 10.0)))
            y1 (trace :y1 (dist/multivariate-normal mu (eye 2)))
            y2 (trace :y2 (dist/multivariate-normal mu (eye 2)))]
        [y1 y2]))))

(defn make-mvn-model [d prior-var]
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/multivariate-normal
                            (mx/zeros [d])
                            (scale-eye d prior-var)))
            y  (trace :y (dist/multivariate-normal mu (eye d)))]
        y))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest mvn-update-correctness-test
  (testing "MVN-MVN update function correctness"
    (let [posterior {:mean-vec (mx/array [0 0]) :cov-matrix (scale-eye 2 10.0)}
          obs-value (mx/array [5 3])
          obs-cov (eye 2)
          result (auto-analytical/mvn-update-step posterior obs-value obs-cov)]
      (is (some? result) "MVN update returns non-nil")
      (is (h/close? 4.545 (mx/item (mx/index (:mean-vec result) 0)) 0.01) "MVN posterior mean[0]")
      (is (h/close? 2.727 (mx/item (mx/index (:mean-vec result) 1)) 0.01) "MVN posterior mean[1]")
      (is (h/close? 0.909
            (mx/item (mx/index (mx/index (:cov-matrix result) 0) 0)) 0.01) "MVN posterior cov[0,0]")
      (is (h/close? -5.781 (mx/item (:ll result)) 0.01) "MVN marginal LL")))

  (testing "MVN double update"
    (let [post0 {:mean-vec (mx/array [0 0]) :cov-matrix (scale-eye 2 10.0)}
          obs-cov (eye 2)
          r1 (auto-analytical/mvn-update-step post0 (mx/array [5 3]) obs-cov)
          post1 {:mean-vec (:mean-vec r1) :cov-matrix (:cov-matrix r1)}
          r2 (auto-analytical/mvn-update-step post1 (mx/array [6 4]) obs-cov)]
      (is (some? r2) "MVN double update non-nil")
      (let [m (mx/item (mx/index (:mean-vec r2) 0))]
        (is (and (> m 4.5) (< m 6.0)) "MVN double update: mean between obs (4.5 < m < 6)")))))

(deftest conjugacy-detection-test
  (testing "Conjugacy detection in schema"
    (let [s (:schema mvn-model)]
      (is (:static? s) "MVN model is static")
      (is (= 1 (count (:conjugate-pairs s))) "MVN model has conjugate pairs")
      (is (= :mvn-normal (:family (first (:conjugate-pairs s)))) "MVN conjugate family is :mvn-normal")
      (is (some? (:auto-handlers s)) "MVN model has auto-handlers")
      (is (= #{:mu :y} (set (keys (:auto-handlers s)))) "MVN auto-handlers include :mu and :y"))))

(deftest end-to-end-generate-test
  (testing "End-to-end p/generate"
    (let [obs (cm/set-value cm/EMPTY :y (mx/array [5 3]))
          weights (mapv (fn [_]
                          (mx/item (:weight (p/generate mvn-model [] obs))))
                        (range 10))]
      (is (deterministic-weight? weights) "MVN generate: deterministic weight")
      (is (h/close? -5.781 (first weights) 0.02) "MVN generate: marginal LL approx -5.781")

      (let [result (p/generate mvn-model [] obs)
            choices (:choices (:trace result))
            mu-val (cm/get-value (cm/get-submap choices :mu))
            y-val (cm/get-value (cm/get-submap choices :y))]
        (is (h/close? 4.545 (mx/item (mx/index mu-val 0)) 0.02) "MVN generate: mu[0] approx 4.545")
        (is (h/close? 5.0 (mx/item (mx/index y-val 0)) 0.001) "MVN generate: y equals observation")))))

(deftest d1-consistency-test
  (testing "d=1 MVN matches scalar NN"
    (let [mvn-obs (cm/set-value cm/EMPTY :y (mx/array [5]))
          scalar-obs (cm/set-value cm/EMPTY :y (mx/scalar 5.0))
          mvn-w (mx/item (:weight (p/generate mvn-1d-model [] mvn-obs)))
          scalar-w (mx/item (:weight (p/generate scalar-nn-model [] scalar-obs)))]
      (is (h/close? scalar-w mvn-w 1e-4) "d=1 MVN weight matches scalar NN"))))

(deftest condition-number-guard-test
  (testing "Condition number guard"
    (let [tiny-cov (mx/reshape (mx/array [1e-8 0 0 1e-8]) [2 2])
          posterior {:mean-vec (mx/array [0 0]) :cov-matrix tiny-cov}
          obs-cov (eye 2)
          result (auto-analytical/mvn-update-step posterior (mx/array [1 1]) obs-cov)]
      (is (some? result) "Near-zero prior cov + I obs: well-conditioned, succeeds"))

    (let [posterior {:mean-vec (mx/array [0 0]) :cov-matrix (eye 2)}
          tiny-obs-cov (mx/reshape (mx/array [1e-8 0 0 1e-8]) [2 2])
          result (auto-analytical/mvn-update-step posterior (mx/array [1 1]) tiny-obs-cov)]
      (is (some? result) "Normal prior + tiny obs cov: well-conditioned, succeeds"))))

(deftest multiple-mvn-observations-test
  (testing "Multiple MVN observations"
    (let [s (:schema mvn-multi-obs-model)]
      (is (= 2 (count (:conjugate-pairs s))) "Multi-obs MVN: 2 conjugate pairs")
      (is (= #{:mu :y1 :y2} (set (keys (:auto-handlers s))))
          "Multi-obs MVN: auto-handlers for :mu :y1 :y2"))

    (let [obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/array [5 3]))
                  (cm/set-value :y2 (mx/array [6 4])))
          weights (mapv (fn [_]
                          (mx/item (:weight (p/generate mvn-multi-obs-model [] obs))))
                        (range 10))]
      (is (deterministic-weight? weights) "Multi-obs MVN: deterministic weight")
      (is (< (first weights) 0.0) "Multi-obs MVN: weight is negative (sum of marginal LLs)"))))

(deftest dimension-scaling-test
  (testing "Dimension scaling benchmark"
    (doseq [d [1 2 5 10]]
      (let [model (make-mvn-model d 100.0)
            obs-val (mx/ones [d])
            obs (cm/set-value cm/EMPTY :y obs-val)
            has-auto (some? (:auto-handlers (:schema model)))
            weights (mapv (fn [_]
                            (mx/item (:weight (p/generate model [] obs))))
                          (range 5))]
        (is has-auto (str "d=" d ": auto-handlers detected"))
        (is (deterministic-weight? weights) (str "d=" d ": deterministic weight"))))))

(deftest variance-comparison-test
  (testing "Variance comparison"
    (let [d 5
          model-auto (make-mvn-model d 100.0)
          model-vanilla (let [m (make-mvn-model d 100.0)
                              s (-> (:schema m)
                                    (dissoc :auto-handlers)
                                    (assoc :conjugate-pairs []))]
                          (dyn/auto-key (dyn/->DynamicGF (:body-fn m) (:source m) s)))
          obs-val (mx/ones [d])
          obs (cm/set-value cm/EMPTY :y obs-val)
          n-trials 10
          auto-weights (mapv (fn [_] (mx/item (:weight (p/generate model-auto [] obs)))) (range n-trials))
          vanilla-weights (mapv (fn [_] (mx/item (:weight (p/generate model-vanilla [] obs)))) (range n-trials))
          auto-var (weight-variance auto-weights)
          vanilla-var (weight-variance vanilla-weights)]
      (is (< auto-var 0.001) "d=5: auto-handler has zero variance")
      (is (> vanilla-var 1.0) "d=5: vanilla has high variance"))))

(cljs.test/run-tests)
