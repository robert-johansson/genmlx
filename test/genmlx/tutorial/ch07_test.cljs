(ns genmlx.tutorial.ch07-test
  "Test file for Tutorial Chapter 7: The Inference Toolkit."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.importance :as importance]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.diagnostics :as diag]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

(def pass (atom 0))
(def fail (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass inc) (println "  PASS:" msg))
    (do (swap! fail inc) (println "  FAIL:" msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass inc) (println "  PASS:" msg))
      (do (swap! fail inc) (println "  FAIL:" msg (str "expected=" expected " actual=" actual))))))

;; Simple model for testing
(def simple-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :x (dist/gaussian mu 1))
      mu)))

(def obs (cm/choicemap :x (mx/scalar 5.0)))

;; ============================================================
;; Listing 7.1: Importance sampling
;; ============================================================
(println "\n== Listing 7.1: importance sampling ==")

(let [{:keys [traces log-weights log-ml-estimate]}
      (importance/importance-sampling {:samples 50} simple-model [] obs)]
  (assert-true "IS returns 50 traces" (= 50 (count traces)))
  (assert-true "log-ML is finite" (js/Number.isFinite (mx/item log-ml-estimate))))

;; ============================================================
;; Listing 7.2: Kernel constructors
;; ============================================================
(println "\n== Listing 7.2: kernel constructors ==")

(let [k1 (kern/mh-kernel (sel/select :mu))]
  (assert-true "mh-kernel is a fn" (fn? k1))
  (assert-true "mh-kernel is symmetric" (kern/symmetric? k1)))

(let [k2 (kern/random-walk :mu 0.5)]
  (assert-true "random-walk is a fn" (fn? k2))
  (assert-true "random-walk is symmetric" (kern/symmetric? k2)))

(let [k3 (kern/prior :mu)]
  (assert-true "prior is a fn" (fn? k3)))

(let [k4 (kern/gibbs :mu)]
  (assert-true "gibbs is a fn" (fn? k4)))

;; ============================================================
;; Listing 7.3: Kernel composition
;; ============================================================
(println "\n== Listing 7.3: kernel composition ==")

(let [composed (kern/chain (kern/prior :mu) (kern/random-walk :mu 0.5))]
  (assert-true "chain produces a fn" (fn? composed)))

(let [repeated (kern/repeat-kernel 10 (kern/prior :mu))]
  (assert-true "repeat-kernel produces a fn" (fn? repeated)))

(let [cycled (kern/cycle-kernels 6 [(kern/prior :mu) (kern/random-walk :mu 0.5)])]
  (assert-true "cycle-kernels produces a fn" (fn? cycled)))

;; ============================================================
;; Listing 7.4: run-kernel
;; ============================================================
(println "\n== Listing 7.4: run-kernel ==")

(let [model (dyn/auto-key simple-model)
      init-trace (:trace (p/generate model [] obs))
      kernel (kern/random-walk :mu 1.0)
      samples (kern/run-kernel {:samples 50 :burn 20} kernel init-trace)]
  (assert-true "run-kernel returns samples" (= 50 (count samples)))
  (assert-true "samples are traces" (some? (:choices (first samples))))
  (let [rate (:acceptance-rate (meta samples))]
    (assert-true "acceptance rate > 0" (> rate 0))
    (assert-true "acceptance rate < 1" (< rate 1))))

;; ============================================================
;; Listing 7.5: MH with built-in function
;; ============================================================
(println "\n== Listing 7.5: mcmc/mh ==")

(let [samples (mcmc/mh {:samples 50 :burn 20 :selection (sel/select :mu)}
                        simple-model [] obs)]
  (assert-true "mcmc/mh returns 50 traces" (= 50 (count samples)))
  (assert-true "has acceptance rate in metadata" (number? (:acceptance-rate (meta samples)))))

;; ============================================================
;; Listing 7.6: HMC
;; ============================================================
(println "\n== Listing 7.6: HMC ==")

(let [samples (mcmc/hmc {:samples 30 :burn 10 :step-size 0.1 :leapfrog-steps 5
                          :addresses [:mu]}
                         simple-model [] obs)]
  (assert-true "HMC returns 30 traces" (= 30 (count samples)))
  ;; HMC compiled path may use tensor traces
  (assert-true "HMC samples are non-nil" (some? (first samples))))

;; ============================================================
;; Listing 7.7: SMC
;; ============================================================
(println "\n== Listing 7.7: SMC ==")

;; SMC for a sequential model
(def hmm-step
  (gen [t state]
    (let [next-state (trace :z (dist/gaussian state 1))]
      (trace :obs (dist/gaussian next-state 0.5))
      next-state)))

(let [model (dyn/auto-key (genmlx.combinators/unfold-combinator hmm-step))
      ;; Observations at 3 timesteps
      obs-seq [(cm/choicemap 0 (cm/choicemap :obs (mx/scalar 1.0)))
               (cm/choicemap 0 (cm/choicemap :obs (mx/scalar 1.0))
                             1 (cm/choicemap :obs (mx/scalar 2.0)))
               (cm/choicemap 0 (cm/choicemap :obs (mx/scalar 1.0))
                             1 (cm/choicemap :obs (mx/scalar 2.0))
                             2 (cm/choicemap :obs (mx/scalar 3.0)))]
      result (smc/smc {:particles 20 :ess-threshold 0.5}
                       model [3 0.0] (last obs-seq))]
  (assert-true "SMC returns traces" (pos? (count (:traces result))))
  (assert-true "SMC log-ML is finite" (js/Number.isFinite (mx/item (:log-ml-estimate result)))))

;; ============================================================
;; Listing 7.8: Diagnostics
;; ============================================================
(println "\n== Listing 7.8: diagnostics ==")

(let [model (dyn/auto-key simple-model)
      init-trace (:trace (p/generate model [] obs))
      kernel (kern/random-walk :mu 1.0)
      traces (kern/run-kernel {:samples 100 :burn 50} kernel init-trace)
      ;; Extract mu values as MLX arrays for diagnostics
      mu-arrays (mapv #(cm/get-choice (:choices %) [:mu]) traces)]
  (let [mean (mx/item (diag/sample-mean mu-arrays))
        std (mx/item (diag/sample-std mu-arrays))
        qs (diag/sample-quantiles mu-arrays)]
    (assert-true "mean is finite" (js/Number.isFinite mean))
    (assert-true "std is positive" (> std 0))
    (assert-true "quantiles has :median" (contains? qs :median))
    (assert-true "quantiles has :q025" (contains? qs :q025))
    (assert-true "median > q025" (< (:q025 qs) (:median qs)))))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 7 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
