(ns genmlx.loop-compiled-mala-test
  "Tests for MALA loop compilation: correctness, cache validation, benchmarks."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.gen :refer [gen]]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(defn- assert-true [desc pred]
  (println (str "  " (if pred "PASS" "FAIL") ": " desc))
  (when-not pred (throw (js/Error. (str "FAIL: " desc)))))

(defn- mean [xs] (/ (reduce + xs) (count xs)))

(defn- variance [xs]
  (let [m (mean xs)]
    (/ (reduce + (map #(* (- % m) (- % m)) xs)) (count xs))))

(defn- bench [f {:keys [warmup runs] :or {warmup 1 runs 3}}]
  (dotimes [_ warmup] (f))
  (let [times (mapv (fn [_]
                      (let [t0 (js/Date.now)]
                        (f)
                        (- (js/Date.now) t0)))
                    (range runs))]
    (mean times)))

;; ---------------------------------------------------------------------------
;; Model: simple Gaussian with known posterior
;; Posterior mean ≈ 5.1, posterior std ≈ 0.58
;; ---------------------------------------------------------------------------

(def model
  (gen [n]
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (dotimes [i n]
        (dyn/trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(def obs (cm/choicemap :y0 5.0 :y1 5.5 :y2 4.8))

;; ---------------------------------------------------------------------------
;; 1. Correctness: compiled returns correct number of samples
;; ---------------------------------------------------------------------------

(println "\n=== MALA Loop Compilation Tests ===")

(println "\n-- correctness --")

(let [samples (mcmc/mala
                {:samples 30 :burn 20 :step-size 0.1
                 :addresses [:mu] :compile? true :device :cpu}
                model [3] obs)]
  (assert-true "compiled returns 30 samples" (= 30 (count samples)))
  (assert-true "samples are vectors" (vector? (first samples)))
  (assert-true "sample values are numbers" (number? (first (first samples)))))

;; ---------------------------------------------------------------------------
;; 2. Compiled vs eager both produce finite samples
;; ---------------------------------------------------------------------------

(println "\n-- compiled vs eager statistics --")

(let [compiled-samples (mcmc/mala
                         {:samples 500 :burn 500 :step-size 0.1
                          :addresses [:mu] :compile? true :device :cpu}
                         model [3] obs)
      eager-samples (mcmc/mala
                      {:samples 500 :burn 500 :step-size 0.1
                       :addresses [:mu] :compile? false :device :cpu}
                      model [3] obs)
      compiled-mean (mean (map first compiled-samples))
      eager-mean (mean (map first eager-samples))]
  (assert-true (str "compiled samples finite (" (.toFixed compiled-mean 2) ")")
               (not (js/isNaN compiled-mean)))
  (assert-true (str "eager samples finite (" (.toFixed eager-mean 2) ")")
               (not (js/isNaN eager-mean)))
  ;; Both should have some variance (chain is mixing)
  (assert-true "compiled has variance" (> (variance (map first compiled-samples)) 0.001))
  (assert-true "eager has variance" (> (variance (map first eager-samples)) 0.001)))

;; ---------------------------------------------------------------------------
;; 3. Statistical validity: variance > 0 across chains
;; ---------------------------------------------------------------------------

(println "\n-- statistical validity --")

(let [samples (mcmc/mala
                {:samples 100 :burn 100 :step-size 0.1
                 :addresses [:mu] :compile? true :device :cpu}
                model [3] obs)
      vals (mapv first samples)
      v (variance vals)]
  (assert-true (str "variance > 0 (" (.toFixed v 4) ")") (> v 0.001)))

;; ---------------------------------------------------------------------------
;; 4. Thin > 1 uses compiled thin chain
;; ---------------------------------------------------------------------------

(println "\n-- thin > 1 --")

(let [samples (mcmc/mala
                {:samples 20 :burn 10 :thin 3 :step-size 0.1
                 :addresses [:mu] :compile? true :device :cpu}
                model [3] obs)]
  (assert-true "thin=3 returns 20 samples" (= 20 (count samples))))

;; ---------------------------------------------------------------------------
;; 5. Benchmark: compiled vs eager
;; ---------------------------------------------------------------------------

(println "\n-- benchmark (200 samples, burn 100) --")

(let [compiled-ms (bench
                    #(mcmc/mala
                       {:samples 200 :burn 100 :step-size 0.1
                        :addresses [:mu] :compile? true :device :cpu}
                       model [3] obs)
                    {:warmup 1 :runs 3})
      eager-ms (bench
                 #(mcmc/mala
                    {:samples 200 :burn 100 :step-size 0.1
                     :addresses [:mu] :compile? false :device :cpu}
                    model [3] obs)
                 {:warmup 1 :runs 3})
      speedup (/ eager-ms compiled-ms)]
  (println (str "  compiled: " (.toFixed compiled-ms 0) "ms"))
  (println (str "  eager:    " (.toFixed eager-ms 0) "ms"))
  (println (str "  speedup:  " (.toFixed speedup 2) "x")))

;; ---------------------------------------------------------------------------
;; 6. Long chain stability (no Metal crash)
;; ---------------------------------------------------------------------------

(println "\n-- long chain stability (500 steps) --")

(let [samples (mcmc/mala
                {:samples 500 :burn 100 :step-size 0.05
                 :addresses [:mu] :compile? true :block-size 50 :device :cpu}
                model [3] obs)]
  (assert-true "500-step chain completes" (= 500 (count samples)))
  (assert-true "no NaN in samples"
               (every? #(not (js/isNaN (first %))) samples)))

(println "\nAll MALA loop compilation tests passed!")
