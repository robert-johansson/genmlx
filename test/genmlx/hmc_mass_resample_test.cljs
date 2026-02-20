(ns genmlx.hmc-mass-resample-test
  "Tests for HMC mass matrix (14.1), residual resampling (14.3),
   and stratified resampling (14.4)."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(defn assert-true [msg pred]
  (if pred
    (println (str "  PASS: " msg))
    (println (str "  FAIL: " msg))))

(defn assert-close [msg expected actual tol]
  (let [ok (< (js/Math.abs (- expected actual)) tol)]
    (if ok
      (println (str "  PASS: " msg " (expected ~" expected ", got " (.toFixed actual 4) ")"))
      (println (str "  FAIL: " msg " (expected ~" expected ", got " (.toFixed actual 4) ")")))))

;; ---------------------------------------------------------------------------
;; Models for testing
;; ---------------------------------------------------------------------------

;; Simple 2D Gaussian with different scales — good for testing mass matrices
(def aniso-model
  (gen [_]
    (let [x (dyn/trace :x (dist/gaussian 0 1))
          y (dyn/trace :y (dist/gaussian 0 10))]
      [x y])))

;; Simple model for SMC testing
(def coin-model
  (gen [n]
    (let [p (dyn/trace :p (dist/uniform 0.01 0.99))]
      (mx/eval! p)
      (let [pv (mx/item p)]
        (doseq [i (range n)]
          (dyn/trace (keyword (str "y" i))
                     (dist/bernoulli pv)))
        pv))))

;; ---------------------------------------------------------------------------
;; 14.1: HMC mass matrix tests
;; ---------------------------------------------------------------------------

(println "\n=== 14.1: HMC Mass Matrix ===")

(println "\n-- HMC with identity metric (baseline) --")
(let [obs cm/EMPTY
      key (rng/fresh-key 42)
      samples (mcmc/hmc {:samples 20 :step-size 0.1 :leapfrog-steps 5
                          :burn 5 :addresses [:x :y] :key key}
                        aniso-model [nil] obs)]
  (assert-true "HMC identity returns samples" (pos? (count samples))))

(println "\n-- HMC with diagonal metric --")
(let [obs cm/EMPTY
      key (rng/fresh-key 43)
      ;; Diagonal metric matching the prior scales: σ² = [1, 100]
      metric (mx/array [1.0 100.0])
      samples (mcmc/hmc {:samples 20 :step-size 0.1 :leapfrog-steps 5
                          :burn 5 :addresses [:x :y] :metric metric :key key}
                        aniso-model [nil] obs)]
  (assert-true "HMC diagonal metric returns samples" (pos? (count samples)))
  (assert-true "HMC diagonal metric returns vectors" (vector? (first samples))))

(println "\n-- HMC with dense metric --")
(let [obs cm/EMPTY
      key (rng/fresh-key 44)
      ;; Dense metric (2x2 positive definite)
      metric (mx/array [[1.0 0.0] [0.0 100.0]])
      samples (mcmc/hmc {:samples 20 :step-size 0.1 :leapfrog-steps 5
                          :burn 5 :addresses [:x :y] :metric metric :key key}
                        aniso-model [nil] obs)]
  (assert-true "HMC dense metric returns samples" (pos? (count samples)))
  (assert-true "HMC dense metric returns vectors" (vector? (first samples))))

(println "\n-- NUTS with diagonal metric --")
(let [obs cm/EMPTY
      key (rng/fresh-key 45)
      metric (mx/array [1.0 100.0])
      samples (mcmc/nuts {:samples 10 :step-size 0.1 :max-depth 3
                           :addresses [:x :y] :metric metric :key key}
                         aniso-model [nil] obs)]
  (assert-true "NUTS diagonal metric returns samples" (pos? (count samples))))

(println "\n-- NUTS with dense metric --")
(let [obs cm/EMPTY
      key (rng/fresh-key 46)
      metric (mx/array [[1.0 0.0] [0.0 100.0]])
      samples (mcmc/nuts {:samples 10 :step-size 0.1 :max-depth 3
                           :addresses [:x :y] :metric metric :key key}
                         aniso-model [nil] obs)]
  (assert-true "NUTS dense metric returns samples" (pos? (count samples))))

;; ---------------------------------------------------------------------------
;; 14.3 & 14.4: Resampling method tests
;; ---------------------------------------------------------------------------

(println "\n=== 14.3: Residual Resampling ===")

(println "\n-- SMC with residual resampling --")
(let [obs-seq [(cm/from-map {:y0 (mx/scalar 1.0)})
               (cm/from-map {:y1 (mx/scalar 1.0)})
               (cm/from-map {:y2 (mx/scalar 1.0)})]
      key (rng/fresh-key 50)
      result (smc/smc {:particles 20 :ess-threshold 0.5
                        :resample-method :residual :key key}
                      coin-model [3] obs-seq)]
  (assert-true "Residual: returns traces" (= 20 (count (:traces result))))
  (assert-true "Residual: returns weights" (= 20 (count (:log-weights result))))
  (assert-true "Residual: returns log-ML" (number? (mx/item (:log-ml-estimate result)))))

(println "\n=== 14.4: Stratified Resampling ===")

(println "\n-- SMC with stratified resampling --")
(let [obs-seq [(cm/from-map {:y0 (mx/scalar 1.0)})
               (cm/from-map {:y1 (mx/scalar 1.0)})
               (cm/from-map {:y2 (mx/scalar 1.0)})]
      key (rng/fresh-key 51)
      result (smc/smc {:particles 20 :ess-threshold 0.5
                        :resample-method :stratified :key key}
                      coin-model [3] obs-seq)]
  (assert-true "Stratified: returns traces" (= 20 (count (:traces result))))
  (assert-true "Stratified: returns weights" (= 20 (count (:log-weights result))))
  (assert-true "Stratified: returns log-ML" (number? (mx/item (:log-ml-estimate result)))))

(println "\n-- SMC with systematic (default, still works) --")
(let [obs-seq [(cm/from-map {:y0 (mx/scalar 1.0)})
               (cm/from-map {:y1 (mx/scalar 1.0)})
               (cm/from-map {:y2 (mx/scalar 1.0)})]
      key (rng/fresh-key 52)
      result (smc/smc {:particles 20 :ess-threshold 0.5
                        :key key}
                      coin-model [3] obs-seq)]
  (assert-true "Systematic (default): returns traces" (= 20 (count (:traces result))))
  (assert-true "Systematic (default): returns log-ML" (number? (mx/item (:log-ml-estimate result)))))

(println "\nAll 14.1/14.3/14.4 tests complete.")
