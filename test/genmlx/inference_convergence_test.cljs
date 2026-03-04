(ns genmlx.inference-convergence-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Inference Convergence Tests ===")

;; ---------------------------------------------------------------------------
;; 21.10 — Gamma-Poisson conjugate
;; ---------------------------------------------------------------------------
;;
;; Prior:      lambda ~ Gamma(alpha=3, rate=1)
;; Likelihood: x_i ~ Poisson(lambda), i=1..5
;; Data:       [2, 4, 3, 5, 1] (sum=15)
;; Posterior:  Gamma(alpha + sum = 18, rate + n = 6)
;; E[lambda|data] = 18/6 = 3.0

(println "\n-- 21.10: Gamma-Poisson conjugate --")

;; Note: GenMLX gamma-dist uses rate parameterization: gamma-dist(shape, rate)
;; E[X] = shape/rate
(def gamma-poisson-model
  (gen [data]
    (let [lam (trace :lambda (dist/gamma-dist 3 1))]
      (mx/eval! lam)
      (let [lam-val (mx/item lam)]
        (doseq [[i x] (map-indexed vector data)]
          (trace (keyword (str "x" i)) (dist/poisson lam-val)))
        lam-val))))

(def gp-data [2 4 3 5 1])
(def gp-observations
  (reduce (fn [cm [i x]]
            (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar x)))
          cm/EMPTY (map-indexed vector gp-data)))

;; IS test
(println "\n  IS (100 particles)")
(let [{:keys [traces log-weights]} (is/importance-sampling
                                     {:samples 100} gamma-poisson-model [gp-data] gp-observations)
      ;; Compute weighted mean
      raw-weights (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
      max-w (apply max raw-weights)
      exp-weights (mapv (fn [w] (js/Math.exp (- w max-w))) raw-weights)
      sum-w (reduce + exp-weights)
      norm-weights (mapv (fn [w] (/ w sum-w)) exp-weights)
      lambda-vals (mapv :retval traces)
      weighted-mean (reduce + (map * lambda-vals norm-weights))]
  (assert-close "IS posterior mean(lambda) ≈ 3.0" 3.0 weighted-mean 0.5)
  (assert-true "IS returns 100 traces" (= 100 (count traces))))

;; MH test
(println "\n  MH (500 samples)")
(let [;; For MH we need a model without mx/eval!/mx/item inside
      ;; since MH uses regenerate which needs the trace sites to be pure MLX
      model-mh (gen [data]
                 (let [lam (trace :lambda (dist/gamma-dist 3 1))]
                   (mx/eval! lam)
                   (let [lam-val (mx/item lam)]
                     (doseq [[i x] (map-indexed vector data)]
                       (trace (keyword (str "x" i)) (dist/poisson lam-val)))
                     lam-val)))
      traces (mcmc/mh {:samples 500 :burn 200 :selection (sel/select :lambda)}
                       model-mh [gp-data] gp-observations)
      lambda-vals (mapv (fn [t]
                          (let [v (cm/get-value (cm/get-submap (:choices t) :lambda))]
                            (mx/eval! v) (mx/item v)))
                        traces)
      lambda-mean (/ (reduce + lambda-vals) (count lambda-vals))]
  (assert-close "MH posterior mean(lambda) ≈ 3.0" 3.0 lambda-mean 0.5)
  (assert-true "MH returns 500 traces" (= 500 (count traces))))

;; ---------------------------------------------------------------------------
;; 21.11 — HMC/NUTS acceptance rate
;; ---------------------------------------------------------------------------

(println "\n-- 21.11: HMC/NUTS acceptance rates --")

;; Normal-Normal model (no eval!/item in body for gradient-based methods)
(def normal-normal
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [i (range 5)]
        (trace (keyword (str "obs" i)) (dist/gaussian mu 1)))
      mu)))

(def nn-observations
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "obs" i))]
                           (mx/scalar (+ 3.0 (* 0.1 (- i 2))))))
          cm/EMPTY (range 5)))

;; HMC acceptance rate
(println "\n  HMC acceptance rate")
(let [samples (mcmc/hmc {:samples 100 :burn 50 :step-size 0.05 :leapfrog-steps 10
                          :addresses [:mu] :compile? false :device :cpu}
                         normal-normal [] nn-observations)
      ;; HMC returns list of JS arrays
      mu-vals (mapv first samples)
      ;; Check no NaN
      has-nan (some js/isNaN mu-vals)
      ;; Count unique values (proxy for acceptance)
      n-unique (count (set mu-vals))
      acceptance-rate (/ (double n-unique) (count mu-vals))]
  (assert-true "HMC: no NaN in samples" (not has-nan))
  (assert-true "HMC: acceptance rate > 0.3" (> acceptance-rate 0.3))
  (assert-close "HMC: posterior mean ≈ 3" 3.0 (/ (reduce + mu-vals) (count mu-vals)) 1.0))

;; NUTS acceptance rate
(println "\n  NUTS acceptance rate")
(let [samples (mcmc/nuts {:samples 50 :burn 50 :step-size 0.05
                           :addresses [:mu] :compile? false :device :cpu}
                          normal-normal [] nn-observations)
      mu-vals (mapv first samples)
      has-nan (some js/isNaN mu-vals)
      n-unique (count (set mu-vals))
      acceptance-rate (/ (double n-unique) (count mu-vals))]
  (assert-true "NUTS: no NaN in samples" (not has-nan))
  (assert-true "NUTS: acceptance rate > 0.3" (> acceptance-rate 0.3))
  (assert-close "NUTS: posterior mean ≈ 3" 3.0 (/ (reduce + mu-vals) (count mu-vals)) 1.0))

(println "\nAll inference convergence tests complete.")
