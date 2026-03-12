(ns genmlx.fused-mh-api-test
  "Test fused-mh public API (M5)."
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.inference.mcmc :as mcmc]))

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc) (println (str "  PASS: " desc)))
    (do (swap! fail-count inc) (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc) (println (str "  PASS: " desc " (diff=" (.toFixed diff 4) ")")))
      (do (swap! fail-count inc) (println (str "  FAIL: " desc " expected=" expected " actual=" actual))))))

;; Static linreg (tensor-native score)
(def static-linreg
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))
          y-pred (mx/add (mx/multiply slope (mx/ensure-array x)) intercept)]
      (trace :y (dist/gaussian y-pred 1))
      slope)))

(def obs (cm/choicemap :y (mx/scalar 5.0)))

(println "\n== fused-mh public API ==")

;; 1. Basic usage
(println "\n-- basic usage --")
(let [result (mcmc/fused-mh
               {:samples 500 :burn 200 :addresses [:slope :intercept]
                :proposal-std 0.5}
               static-linreg [2.0] obs)]
  (assert-true "returns :samples" (some? (:samples result)))
  (assert-true "returns :final-params" (some? (:final-params result)))
  (assert-true "returns :chain-fn" (fn? (:chain-fn result)))
  (assert-true "samples shape [500,2]" (= [500 2] (mx/shape (:samples result))))
  (assert-true "final-params shape [2]" (= [2] (mx/shape (:final-params result))))
  (let [samples-js (mx/->clj (:samples result))
        slopes (mapv first samples-js)
        mean-slope (/ (reduce + slopes) (count slopes))]
    (assert-true "posterior slope finite" (js/isFinite mean-slope))))

;; 2. Reuse chain-fn (amortized compilation)
(println "\n-- chain-fn reuse --")
(let [;; First call: compiles
      result1 (mcmc/fused-mh
                {:samples 200 :burn 100 :addresses [:slope :intercept]
                 :proposal-std 0.5}
                static-linreg [2.0] obs)
      cfn (:chain-fn result1)
      ;; Second call: reuses compiled fn (should be fast)
      t0 (js/Date.now)
      result2 (mcmc/fused-mh
                {:samples 200 :burn 100 :addresses [:slope :intercept]
                 :proposal-std 0.5 :chain-fn cfn}
                static-linreg [2.0] obs)
      t1 (js/Date.now)]
  (assert-true "reuse returns samples" (= [200 2] (mx/shape (:samples result2))))
  (assert-true "reuse is fast (<500ms)" (< (- t1 t0) 500))
  (println (str "  (reuse took " (- t1 t0) "ms)")))

;; 3. With thinning
(println "\n-- thinning --")
(let [result (mcmc/fused-mh
               {:samples 100 :burn 50 :thin 3 :addresses [:slope :intercept]
                :proposal-std 0.5}
               static-linreg [2.0] obs)]
  (assert-true "thin=3 samples shape [100,2]" (= [100 2] (mx/shape (:samples result)))))

;; 4. Benchmark: fused vs block-based compiled-mh
(println "\n-- benchmark: fused vs block-based --")
(let [;; Pre-compile fused
      warmup (mcmc/fused-mh {:samples 10 :burn 10 :addresses [:slope :intercept]
                              :proposal-std 0.5}
                            static-linreg [2.0] obs)
      cfn (:chain-fn warmup)
      ;; Timed fused (cached)
      t0 (js/Date.now)
      _ (mcmc/fused-mh {:samples 1000 :burn 500 :addresses [:slope :intercept]
                         :proposal-std 0.5 :chain-fn cfn}
                       static-linreg [2.0] obs)
      t1 (js/Date.now)
      ms-fused (- t1 t0)
      ;; Timed block-based
      _ (mcmc/compiled-mh {:samples 10 :burn 10 :addresses [:slope :intercept]
                           :proposal-std 0.5}
                          static-linreg [2.0] obs)
      t2 (js/Date.now)
      _ (mcmc/compiled-mh {:samples 1000 :burn 500 :addresses [:slope :intercept]
                           :proposal-std 0.5}
                          static-linreg [2.0] obs)
      t3 (js/Date.now)
      ms-block (- t3 t2)]
  (println (str "  Fused (cached): " ms-fused "ms"))
  (println (str "  Block-based:    " ms-block "ms"))
  (println (str "  Fused speedup:  " (.toFixed (/ ms-block ms-fused) 1) "x"))
  (assert-true "both complete" (and (pos? ms-fused) (pos? ms-block))))

;; Summary
(println (str "\n== fused-mh API: " @pass-count "/" (+ @pass-count @fail-count) " passed =="))
