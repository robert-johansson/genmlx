(ns genmlx.resource-test
  "Stress tests for Metal resource management.
   Verifies that inference loops don't leak Metal buffers over extended runs."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(println "\n=== Resource Management Tests ===\n")

;; Simple 5-site Gaussian model
(def model
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def xs [1.0 2.0 3.0])
(def observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.5 5.1 7.3])))

;; ---------------------------------------------------------------------------
;; Test 1: MH stress — 500 iterations shouldn't grow memory linearly
;; ---------------------------------------------------------------------------
(println "-- MH stress test (500 iterations) --")
(let [_ (mx/clear-cache!)
      _ (mx/reset-peak-memory!)
      ;; Warm up
      {:keys [trace]} (p/generate model [xs] observations)
      _ (mx/eval! (:score trace))
      ;; Run MH for 50 iterations, check memory
      _ (mcmc/mh {:samples 50 :selection (sel/select :slope :intercept)}
                 model [xs] observations)
      mem-50 (mx/get-active-memory)
      ;; Run MH for 500 iterations
      _ (mcmc/mh {:samples 500 :selection (sel/select :slope :intercept)}
                 model [xs] observations)
      mem-500 (mx/get-active-memory)]
  ;; Memory at 500 iters should not be 10x memory at 50 iters
  ;; (i.e., should be bounded, not linearly growing)
  (println "    active memory after 50 iters:" mem-50)
  (println "    active memory after 500 iters:" mem-500)
  (assert-true "memory bounded (500 iters < 5x 50 iters)"
               (or (< mem-500 (* 5 (max mem-50 1024)))
                   ;; If memory is small in both cases, that's fine too
                   (< mem-500 (* 10 1024 1024)))))

;; ---------------------------------------------------------------------------
;; Test 2: IS stress — 200 samples complete without crash
;; ---------------------------------------------------------------------------
(println "\n-- IS stress test (200 samples) --")
(let [_ (mx/clear-cache!)
      result (is/importance-sampling {:samples 200} model [xs] observations)]
  (assert-true "IS completed 200 samples" (= 200 (count (:traces result))))
  (assert-true "IS has weights" (= 200 (count (:log-weights result)))))

;; ---------------------------------------------------------------------------
;; Test 3: collect-samples with array-heavy step-fn
;; ---------------------------------------------------------------------------
(println "\n-- collect-samples resource test --")
(let [_ (mx/clear-cache!)
      ;; Step function that creates many intermediate arrays per step
      step-fn (fn [state _key]
                (let [;; Create several intermediate arrays
                      a (mx/add state (mx/scalar 0.1))
                      b (mx/multiply a (mx/scalar 0.99))
                      c (mx/add b (rng/normal (rng/fresh-key) [10]))
                      d (mx/sum c)]
                  (mx/eval! d)
                  {:state d :accepted? true}))
      results (kern/collect-samples
                {:samples 200 :burn 50}
                step-fn
                mx/item
                (mx/scalar 0.0))]
  (assert-true "collect-samples completed 200 samples" (= 200 (count results)))
  (let [active-mem (mx/get-active-memory)]
    (println "    active memory after 250 total iterations:" active-mem)
    ;; Active memory should stay reasonable (under 50MB)
    (assert-true "active memory bounded" (< active-mem (* 50 1024 1024)))))

;; ---------------------------------------------------------------------------
;; Test 4: clear-cache effect
;; ---------------------------------------------------------------------------
(println "\n-- clear-cache effect test --")
(let [;; Create and eval a bunch of arrays to fill cache
      _ (doseq [_ (range 100)]
          (let [a (rng/normal (rng/fresh-key) [100])]
            (mx/eval! a)))
      cache-before (mx/get-cache-memory)
      _ (mx/clear-cache!)
      cache-after (mx/get-cache-memory)]
  (println "    cache before clear:" cache-before)
  (println "    cache after clear:" cache-after)
  (assert-true "clear-cache reduces cache" (<= cache-after cache-before)))

(println "\n=== Resource Management Tests Complete ===")
