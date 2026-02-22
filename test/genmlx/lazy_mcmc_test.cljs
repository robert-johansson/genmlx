(ns genmlx.lazy-mcmc-test
  "Correctness tests for lazy (sync-free) MCMC variants."
  (:require [genmlx.mlx :as mx]
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

(println "\n=== Lazy MCMC Correctness Tests ===\n")

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def single-gaussian
  (gen []
    (dyn/trace :x (dist/gaussian 0 1))))

(def linear-regression
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def linreg-xs [1.0 2.0 3.0 4.0 5.0])

(defn- obs-linear-regression []
  (-> cm/EMPTY
      (cm/set-choice [:y0] (mx/scalar 3.1))
      (cm/set-choice [:y1] (mx/scalar 5.2))
      (cm/set-choice [:y2] (mx/scalar 6.9))
      (cm/set-choice [:y3] (mx/scalar 9.1))
      (cm/set-choice [:y4] (mx/scalar 10.8))))

(defn- sample-val
  "Extract a scalar value from a sample (handles both number and vector)."
  [s]
  (if (number? s) s (first s)))

;; ---------------------------------------------------------------------------
;; compiled-mh-lazy tests
;; ---------------------------------------------------------------------------

(println "-- compiled-mh-lazy (single gaussian) --")
(let [obs (cm/choicemap :x (mx/scalar 0.5))
      samples (mcmc/compiled-mh-lazy
                {:samples 500 :addresses [:x]}
                single-gaussian [] obs)
      rate (:acceptance-rate (meta samples))
      vals (mapv sample-val samples)
      mean (/ (reduce + vals) (count vals))]
  (assert-true "returns 500 samples" (= 500 (count samples)))
  (assert-true "acceptance rate > 0" (> rate 0.0))
  (println "    acceptance rate:" (.toFixed rate 3))
  ;; Score function = log N(0,1), exploring the prior
  (assert-close "mean near 0" 0.0 mean 0.5))

(println "\n-- compiled-mh-lazy (linear regression) --")
(let [samples (mcmc/compiled-mh-lazy
                {:samples 500 :burn 200 :addresses [:slope :intercept]
                 :proposal-std 0.5}
                linear-regression [linreg-xs] (obs-linear-regression))
      rate (:acceptance-rate (meta samples))
      slopes (mapv #(nth % 0) samples)
      intercepts (mapv #(nth % 1) samples)
      mean-slope (/ (reduce + slopes) (count slopes))
      mean-intercept (/ (reduce + intercepts) (count intercepts))]
  (assert-true "returns 500 samples" (= 500 (count samples)))
  (assert-true "acceptance rate > 0" (> rate 0.0))
  (println "    acceptance rate:" (.toFixed rate 3))
  ;; True slope ~2.0, intercept ~1.0 (wide tolerance for MCMC)
  (assert-close "slope near 2.0" 2.0 mean-slope 1.5)
  (assert-close "intercept near 1.0" 1.0 mean-intercept 2.0))

;; ---------------------------------------------------------------------------
;; hmc-lazy tests
;; ---------------------------------------------------------------------------

(println "\n-- hmc-lazy (linear regression, eval-interval=5) --")
(let [samples (mcmc/hmc-lazy
                {:samples 200 :burn 100 :step-size 0.01
                 :leapfrog-steps 10 :addresses [:slope :intercept]
                 :eval-interval 5}
                linear-regression [linreg-xs] (obs-linear-regression))
      rate (:acceptance-rate (meta samples))
      slopes (mapv #(nth % 0) samples)
      intercepts (mapv #(nth % 1) samples)
      mean-slope (/ (reduce + slopes) (count slopes))
      mean-intercept (/ (reduce + intercepts) (count intercepts))]
  (assert-true "returns 200 samples" (= 200 (count samples)))
  (assert-true "acceptance rate > 0" (> rate 0.0))
  (println "    acceptance rate:" (.toFixed rate 3))
  (println "    mean slope:" (.toFixed mean-slope 3)
           " mean intercept:" (.toFixed mean-intercept 3))
  ;; Wide tolerance — HMC with small step-size mixes slowly
  (assert-close "slope reasonable" 2.0 mean-slope 4.0)
  (assert-close "intercept reasonable" 1.0 mean-intercept 4.0))

(println "\n-- hmc-lazy with eval-interval=1 --")
(let [samples (mcmc/hmc-lazy
                {:samples 50 :burn 50 :step-size 0.01
                 :leapfrog-steps 10 :addresses [:slope :intercept]
                 :eval-interval 1}
                linear-regression [linreg-xs] (obs-linear-regression))
      rate (:acceptance-rate (meta samples))]
  (assert-true "returns 50 samples" (= 50 (count samples)))
  (assert-true "acceptance rate > 0" (> rate 0.0))
  (println "    acceptance rate:" (.toFixed rate 3)))

;; ---------------------------------------------------------------------------
;; Comparison: lazy vs eager produce comparable results
;; ---------------------------------------------------------------------------

(println "\n-- lazy vs eager comparison (compiled MH, single gaussian) --")
(let [obs (cm/choicemap :x (mx/scalar 0.5))
      eager (mcmc/compiled-mh
              {:samples 500 :burn 200 :addresses [:x] :proposal-std 1.0}
              single-gaussian [] obs)
      lazy  (mcmc/compiled-mh-lazy
              {:samples 500 :burn 200 :addresses [:x] :proposal-std 1.0}
              single-gaussian [] obs)
      eager-mean (/ (reduce + (mapv sample-val eager)) (count eager))
      lazy-mean  (/ (reduce + (mapv sample-val lazy)) (count lazy))
      eager-var (/ (reduce + (mapv #(let [v (- (sample-val %) eager-mean)]
                                      (* v v)) eager)) (count eager))
      lazy-var  (/ (reduce + (mapv #(let [v (- (sample-val %) lazy-mean)]
                                      (* v v)) lazy)) (count lazy))]
  ;; Both should explore N(0,1) — wide tolerance for MCMC
  (assert-close "eager mean near 0" 0.0 eager-mean 1.0)
  (assert-close "lazy mean near 0" 0.0 lazy-mean 1.0)
  ;; Variance should be positive (chain is mixing)
  (assert-true "eager variance > 0" (> eager-var 0.05))
  (assert-true "lazy variance > 0" (> lazy-var 0.05))
  (println "    eager: mean=" (.toFixed eager-mean 3) " var=" (.toFixed eager-var 3))
  (println "    lazy:  mean=" (.toFixed lazy-mean 3) " var=" (.toFixed lazy-var 3)))

(println "\nLazy MCMC tests complete.")
