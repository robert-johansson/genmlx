(ns genmlx.kernel-dsl-test
  "Tests for kernel DSL: random-walk, prior, proposal, gibbs."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.kernel :as kern])
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

(println "\n=== Kernel DSL Tests ===\n")

;; Shared model: mu ~ N(0, 10), obs_i ~ N(mu, 1) for i=0..4, all obs=3.0
;; Posterior: mu ~ N(3, ~0.45)
(def model
  (gen []
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (mx/eval! mu)
      (let [mu-val (mx/item mu)]
        (doseq [i (range 5)]
          (dyn/trace (keyword (str "obs" i))
                     (dist/gaussian mu-val 1)))
        mu-val))))

(def observations
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "obs" i))]
                           (mx/scalar 3.0)))
          cm/EMPTY (range 5)))

(defn extract-mu-mean [traces]
  (let [mu-vals (mapv (fn [t]
                        (mx/realize (cm/get-value (cm/get-submap (:choices t) :mu))))
                      traces)]
    (/ (reduce + mu-vals) (count mu-vals))))

;; ---------------------------------------------------------------------------
;; 1. random-walk single address
;; ---------------------------------------------------------------------------

(println "-- random-walk: single address --")
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/random-walk :mu 0.5)
      traces (kern/run-kernel {:samples 200 :burn 100} k trace)
      mu-mean (extract-mu-mean traces)
      ar (:acceptance-rate (meta traces))]
  (assert-true "random-walk: 200 samples" (= 200 (count traces)))
  (assert-close "random-walk: posterior mu near 3" 3.0 mu-mean 1.0)
  (assert-true "random-walk: acceptance rate > 0" (> ar 0))
  (println "    acceptance rate:" ar))

;; ---------------------------------------------------------------------------
;; 2. random-walk multi-address
;; ---------------------------------------------------------------------------

(println "\n-- random-walk: multi-address (map form) --")
(def model2
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (mx/eval! slope intercept)
      (let [s (mx/item slope) b (mx/item intercept)]
        (doseq [[j x] (map-indexed vector xs)]
          (dyn/trace (keyword (str "y" j))
                     (dist/gaussian (+ (* s x) b) 1)))
        s))))

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ;; True slope=2, intercept=1 → y = 2x+1
      obs (reduce (fn [cm [j x]]
                    (cm/set-choice cm [(keyword (str "y" j))]
                                  (mx/scalar (+ (* 2.0 x) 1.0))))
                  cm/EMPTY (map-indexed vector xs))
      {:keys [trace]} (p/generate model2 [xs] obs)
      k (kern/random-walk {:slope 0.3 :intercept 0.3})
      traces (kern/run-kernel {:samples 200 :burn 300} k trace)
      slope-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :slope)))) traces)
      intercept-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :intercept)))) traces)
      slope-mean (/ (reduce + slope-vals) (count slope-vals))
      intercept-mean (/ (reduce + intercept-vals) (count intercept-vals))]
  (assert-close "random-walk(map): slope near 2" 2.0 slope-mean 1.5)
  (assert-close "random-walk(map): intercept near 1" 1.0 intercept-mean 2.0))

;; ---------------------------------------------------------------------------
;; 3. prior
;; ---------------------------------------------------------------------------

(println "\n-- prior: resample from prior --")
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/prior :mu)
      traces (kern/run-kernel {:samples 200 :burn 100} k trace)
      mu-mean (extract-mu-mean traces)
      ar (:acceptance-rate (meta traces))]
  (assert-true "prior: 200 samples" (= 200 (count traces)))
  (assert-close "prior: posterior mu near 3" 3.0 mu-mean 1.0)
  (assert-true "prior: acceptance rate > 0" (> ar 0)))

;; ---------------------------------------------------------------------------
;; 4. proposal symmetric
;; ---------------------------------------------------------------------------

(println "\n-- proposal: symmetric custom proposal --")
;; Proposal GF: takes [current-choices], proposes new :mu from N(current_mu, 0.5)
(def sym-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (dyn/trace :mu (dist/gaussian (mx/item cur-mu) 0.5)))))

(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/proposal sym-proposal)
      traces (kern/run-kernel {:samples 200 :burn 100} k trace)
      mu-mean (extract-mu-mean traces)
      ar (:acceptance-rate (meta traces))]
  (assert-true "proposal(sym): 200 samples" (= 200 (count traces)))
  (assert-close "proposal(sym): posterior mu near 3" 3.0 mu-mean 1.0)
  (assert-true "proposal(sym): acceptance rate > 0" (> ar 0)))

;; ---------------------------------------------------------------------------
;; 5. proposal asymmetric
;; ---------------------------------------------------------------------------

(println "\n-- proposal: asymmetric forward/backward --")
;; Forward: propose mu from N(current_mu + 0.1, 0.5) (biased drift)
;; Backward: same structure, score backward from new trace
(def fwd-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (dyn/trace :mu (dist/gaussian (+ (mx/item cur-mu) 0.1) 0.5)))))

(def bwd-proposal
  (gen [current-choices]
    (let [cur-mu (cm/get-value (cm/get-submap current-choices :mu))]
      (mx/eval! cur-mu)
      (dyn/trace :mu (dist/gaussian (+ (mx/item cur-mu) 0.1) 0.5)))))

(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/proposal fwd-proposal :backward bwd-proposal)
      traces (kern/run-kernel {:samples 200 :burn 100} k trace)
      mu-mean (extract-mu-mean traces)
      ar (:acceptance-rate (meta traces))]
  (assert-true "proposal(asym): 200 samples" (= 200 (count traces)))
  (assert-close "proposal(asym): posterior mu near 3" 3.0 mu-mean 1.0)
  (assert-true "proposal(asym): acceptance rate > 0" (> ar 0)))

;; ---------------------------------------------------------------------------
;; 6. gibbs with keywords (prior-based)
;; ---------------------------------------------------------------------------

(println "\n-- gibbs: keyword args (prior-based) --")
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/gibbs :mu)
      traces (kern/run-kernel {:samples 200 :burn 100} k trace)
      mu-mean (extract-mu-mean traces)
      ar (:acceptance-rate (meta traces))]
  (assert-true "gibbs(kw): 200 samples" (= 200 (count traces)))
  (assert-close "gibbs(kw): posterior mu near 3" 3.0 mu-mean 1.0)
  (assert-true "gibbs(kw): acceptance rate > 0" (> ar 0)))

;; ---------------------------------------------------------------------------
;; 7. gibbs with std map (random-walk-based)
;; ---------------------------------------------------------------------------

(println "\n-- gibbs: std map (random-walk-based) --")
(let [xs [1.0 2.0 3.0 4.0 5.0]
      obs (reduce (fn [cm [j x]]
                    (cm/set-choice cm [(keyword (str "y" j))]
                                  (mx/scalar (+ (* 2.0 x) 1.0))))
                  cm/EMPTY (map-indexed vector xs))
      {:keys [trace]} (p/generate model2 [xs] obs)
      k (kern/gibbs {:slope 0.3 :intercept 0.3})
      traces (kern/run-kernel {:samples 200 :burn 100} k trace)
      slope-vals (mapv (fn [t] (mx/realize (cm/get-value (cm/get-submap (:choices t) :slope)))) traces)
      slope-mean (/ (reduce + slope-vals) (count slope-vals))]
  (assert-close "gibbs(map): slope near 2" 2.0 slope-mean 1.0))

;; ---------------------------------------------------------------------------
;; 8. compatibility — DSL kernels compose with existing combinators
;; ---------------------------------------------------------------------------

(println "\n-- compatibility: compose with chain, repeat-kernel, mix-kernels --")

;; chain: random-walk + prior
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/chain (kern/random-walk :mu 0.5) (kern/prior :mu))
      traces (kern/run-kernel {:samples 50 :burn 50} k trace)
      mu-mean (extract-mu-mean traces)]
  (assert-close "chain(rw+prior): posterior mu near 3" 3.0 mu-mean 1.5))

;; repeat-kernel with random-walk
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/repeat-kernel 3 (kern/random-walk :mu 0.5))
      traces (kern/run-kernel {:samples 50 :burn 50} k trace)
      mu-mean (extract-mu-mean traces)]
  (assert-close "repeat(rw): posterior mu near 3" 3.0 mu-mean 1.5))

;; mix-kernels with DSL kernels
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/mix-kernels [[(kern/random-walk :mu 0.5) 0.7]
                           [(kern/prior :mu) 0.3]])
      traces (kern/run-kernel {:samples 50 :burn 50} k trace)
      mu-mean (extract-mu-mean traces)]
  (assert-close "mix(rw+prior): posterior mu near 3" 3.0 mu-mean 1.5))

;; run-kernel with callback
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/random-walk :mu 0.5)
      callback-count (atom 0)
      traces (kern/run-kernel {:samples 10 :burn 0
                               :callback (fn [_] (swap! callback-count inc))}
                              k trace)]
  (assert-true "run-kernel callback fires" (= 10 @callback-count)))

(println "\nAll kernel DSL tests complete.")
