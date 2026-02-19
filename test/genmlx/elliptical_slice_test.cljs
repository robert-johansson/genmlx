(ns genmlx.elliptical-slice-test
  "Tests for elliptical slice sampling."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
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

(println "\n=== Elliptical Slice Sampling Tests ===\n")

;; ---------------------------------------------------------------------------
;; 1. Convergence — single Gaussian prior
;; ---------------------------------------------------------------------------

(println "-- ESS convergence: single address --")
;; Model: x ~ N(0, prior_std), y ~ N(x, 1), observe y=3
;; Posterior: x ~ N(3 * prior_std^2 / (prior_std^2 + 1), ...)
(let [prior-std 5.0
      obs-val 3.0
      ;; Posterior mean = obs * prior_std^2 / (prior_std^2 + 1)
      posterior-mean (/ (* obs-val prior-std prior-std) (+ (* prior-std prior-std) 1))
      model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 prior-std))]
                (mx/eval! x)
                (dyn/trace :obs (dist/gaussian (mx/item x) 1))
                (mx/item x)))
      observations (cm/choicemap :obs (mx/scalar obs-val))
      traces (mcmc/elliptical-slice
               {:samples 200 :burn 50 :selection [:x] :prior-std prior-std}
               model [] observations)
      x-vals (mapv (fn [t]
                     (mx/realize (cm/get-value (cm/get-submap (:choices t) :x))))
                   traces)
      x-mean (/ (reduce + x-vals) (count x-vals))]
  (assert-true "ESS: 200 samples" (= 200 (count traces)))
  (assert-close "ESS: posterior mean" posterior-mean x-mean 1.0)
  ;; ESS always accepts
  (let [ar (:acceptance-rate (meta traces))]
    (assert-close "ESS: acceptance rate = 1.0" 1.0 ar 0.01)))

;; ---------------------------------------------------------------------------
;; 2. Multiple addresses
;; ---------------------------------------------------------------------------

(println "\n-- ESS convergence: two addresses --")
(let [prior-std 5.0
      model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 prior-std))
                    z (dyn/trace :z (dist/gaussian 0 prior-std))]
                (mx/eval! x z)
                (dyn/trace :obs-x (dist/gaussian (mx/item x) 1))
                (dyn/trace :obs-z (dist/gaussian (mx/item z) 1))
                [(mx/item x) (mx/item z)]))
      observations (cm/merge-cm
                     (cm/choicemap :obs-x (mx/scalar 3.0))
                     (cm/choicemap :obs-z (mx/scalar -2.0)))
      traces (mcmc/elliptical-slice
               {:samples 200 :burn 50 :selection [:x :z] :prior-std prior-std}
               model [] observations)
      x-vals (mapv (fn [t]
                     (mx/realize (cm/get-value (cm/get-submap (:choices t) :x))))
                   traces)
      z-vals (mapv (fn [t]
                     (mx/realize (cm/get-value (cm/get-submap (:choices t) :z))))
                   traces)
      x-mean (/ (reduce + x-vals) (count x-vals))
      z-mean (/ (reduce + z-vals) (count z-vals))
      post-mean-x (/ (* 3.0 prior-std prior-std) (+ (* prior-std prior-std) 1))
      post-mean-z (/ (* -2.0 prior-std prior-std) (+ (* prior-std prior-std) 1))]
  (assert-close "ESS: x posterior mean near analytical" post-mean-x x-mean 1.0)
  (assert-close "ESS: z posterior mean near analytical" post-mean-z z-mean 1.0))

(println "\nAll elliptical slice sampling tests complete.")
