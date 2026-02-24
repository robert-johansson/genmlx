(ns genmlx.loop-compiled-hmc-test
  "Tests for HMC loop compilation: correctness, benchmarks, stability."
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
;; ---------------------------------------------------------------------------

(def model
  (gen [n]
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (dotimes [i n]
        (dyn/trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(def obs (cm/choicemap :y0 5.0 :y1 5.5 :y2 4.8))

;; ---------------------------------------------------------------------------
;; 1. Correctness
;; ---------------------------------------------------------------------------

(println "\n=== HMC Loop Compilation Tests ===")

(println "\n-- correctness --")

(let [samples (mcmc/hmc
                {:samples 30 :burn 20 :step-size 0.05 :leapfrog-steps 5
                 :addresses [:mu] :compile? true :device :cpu}
                model [3] obs)]
  (assert-true "compiled returns 30 samples" (= 30 (count samples)))
  (assert-true "samples are vectors" (vector? (first samples)))
  (assert-true "sample values are numbers" (number? (first (first samples)))))

;; ---------------------------------------------------------------------------
;; 2. Compiled vs eager both produce finite samples
;; ---------------------------------------------------------------------------

(println "\n-- compiled vs eager statistics --")

(let [compiled-samples (mcmc/hmc
                         {:samples 200 :burn 200 :step-size 0.05 :leapfrog-steps 10
                          :addresses [:mu] :compile? true :device :cpu}
                         model [3] obs)
      eager-samples (mcmc/hmc
                      {:samples 200 :burn 200 :step-size 0.05 :leapfrog-steps 10
                       :addresses [:mu] :compile? false :device :cpu}
                      model [3] obs)
      compiled-mean (mean (map first compiled-samples))
      eager-mean (mean (map first eager-samples))]
  (assert-true (str "compiled samples finite (" (.toFixed compiled-mean 2) ")")
               (not (js/isNaN compiled-mean)))
  (assert-true (str "eager samples finite (" (.toFixed eager-mean 2) ")")
               (not (js/isNaN eager-mean)))
  (assert-true "compiled has variance" (> (variance (map first compiled-samples)) 0.001))
  (assert-true "eager has variance" (> (variance (map first eager-samples)) 0.001)))

;; ---------------------------------------------------------------------------
;; 3. Statistical validity
;; ---------------------------------------------------------------------------

(println "\n-- statistical validity --")

(let [samples (mcmc/hmc
                {:samples 100 :burn 100 :step-size 0.05 :leapfrog-steps 10
                 :addresses [:mu] :compile? true :device :cpu}
                model [3] obs)
      vals (mapv first samples)
      v (variance vals)]
  (assert-true (str "variance > 0 (" (.toFixed v 4) ")") (> v 0.001)))

;; ---------------------------------------------------------------------------
;; 4. Thin > 1 uses compiled thin chain
;; ---------------------------------------------------------------------------

(println "\n-- thin > 1 --")

(let [samples (mcmc/hmc
                {:samples 20 :burn 10 :thin 3 :step-size 0.05 :leapfrog-steps 5
                 :addresses [:mu] :compile? true :device :cpu}
                model [3] obs)]
  (assert-true "thin=3 returns 20 samples" (= 20 (count samples))))

;; ---------------------------------------------------------------------------
;; 5. Non-identity metric falls back to eager
;; ---------------------------------------------------------------------------

(println "\n-- non-identity metric fallback --")

(let [samples (mcmc/hmc
                {:samples 20 :burn 10 :step-size 0.05 :leapfrog-steps 5
                 :addresses [:mu] :compile? true :metric (mx/array [1.0])
                 :device :cpu}
                model [3] obs)]
  (assert-true "diagonal metric returns 20 samples" (= 20 (count samples))))

;; ---------------------------------------------------------------------------
;; 6. Benchmark: compiled vs eager
;; ---------------------------------------------------------------------------

(println "\n-- benchmark (100 samples, burn 50, L=10) --")

(let [compiled-ms (bench
                    #(mcmc/hmc
                       {:samples 100 :burn 50 :step-size 0.05 :leapfrog-steps 10
                        :addresses [:mu] :compile? true :device :cpu}
                       model [3] obs)
                    {:warmup 1 :runs 3})
      eager-ms (bench
                 #(mcmc/hmc
                    {:samples 100 :burn 50 :step-size 0.05 :leapfrog-steps 10
                     :addresses [:mu] :compile? false :device :cpu}
                    model [3] obs)
                 {:warmup 1 :runs 3})
      speedup (/ eager-ms compiled-ms)]
  (println (str "  compiled: " (.toFixed compiled-ms 0) "ms"))
  (println (str "  eager:    " (.toFixed eager-ms 0) "ms"))
  (println (str "  speedup:  " (.toFixed speedup 2) "x")))

;; ---------------------------------------------------------------------------
;; 7. Long chain stability
;; ---------------------------------------------------------------------------

(println "\n-- long chain stability (300 steps) --")

(let [samples (mcmc/hmc
                {:samples 300 :burn 50 :step-size 0.05 :leapfrog-steps 5
                 :addresses [:mu] :compile? true :block-size 20 :device :cpu}
                model [3] obs)]
  (assert-true "300-step chain completes" (= 300 (count samples)))
  (assert-true "no NaN in samples"
               (every? #(not (js/isNaN (first %))) samples)))

(println "\nAll HMC loop compilation tests passed!")
