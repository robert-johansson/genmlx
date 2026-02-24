(ns genmlx.vectorized-grad-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]
            [genmlx.vectorized :as v])
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

(println "\n=== Vectorized Gradient Infrastructure Tests ===\n")

;; Simple Gaussian model: mu ~ N(0, σ), obs ~ N(mu, 1)
(def model
  (gen [mu-prior-std]
    (let [mu (dyn/trace :mu (dist/gaussian 0 mu-prior-std))]
      (dyn/trace :obs (dist/gaussian mu 1))
      mu)))

(def observations (cm/choicemap :obs (mx/scalar 3.0)))

;; ---------------------------------------------------------------------------
;; Phase 1: Vectorized gradient correctness
;; ---------------------------------------------------------------------------

(println "-- Vectorized gradient vs N scalar gradients --")
(let [addresses [:mu]
      n-chains 5
      ;; Create N independent parameter sets
      params-list (mapv (fn [_]
                          (let [{:keys [trace]} (p/generate model [10] observations)]
                            (u/extract-params trace addresses)))
                        (range n-chains))
      ;; Scalar gradients (N independent calls)
      scalar-grad-fn (mx/grad (u/make-score-fn model [10] observations addresses))
      scalar-grads (mapv (fn [p]
                           (let [g (scalar-grad-fn p)]
                             (mx/eval! g)
                             (mx/->clj g)))
                         params-list)
      ;; Vectorized gradient (one call)
      vec-grad-fn (u/make-vectorized-grad-score model [10] observations addresses)
      params-matrix (mx/stack params-list)
      vec-grads-arr (vec-grad-fn params-matrix)
      _ (mx/eval! vec-grads-arr)
      vec-grads (mx/->clj vec-grads-arr)]
  (doseq [i (range n-chains)]
    (let [s-g (if (vector? (nth scalar-grads i)) (first (nth scalar-grads i)) (nth scalar-grads i))
          v-g (if (vector? (nth vec-grads i)) (first (nth vec-grads i)) (nth vec-grads i))]
      (assert-close (str "chain " i " gradient matches")
                    s-g v-g 0.01))))

;; ---------------------------------------------------------------------------
;; Phase 2: Loop-Compiled MALA
;; ---------------------------------------------------------------------------

(println "\n-- mala (compiled) --")
(let [samples (mcmc/mala
                {:samples 50 :burn 20 :step-size 0.01 :addresses [:mu]
                 :compile? true :device :cpu}
                model [10] observations)]
  (assert-true "mala compiled returns correct sample count" (= 50 (count samples))))

;; ---------------------------------------------------------------------------
;; Phase 3: Vectorized MALA
;; ---------------------------------------------------------------------------

(println "\n-- vectorized-mala --")
(let [samples (mcmc/vectorized-mala
                {:samples 30 :burn 10 :step-size 0.005 :addresses [:mu]
                 :n-chains 5 :device :cpu}
                model [10] observations)]
  (assert-true "vectorized-mala returns correct sample count" (= 30 (count samples)))
  (let [rate (:acceptance-rate (meta samples))]
    (assert-true "vectorized-mala has positive acceptance rate" (> rate 0))))

;; ---------------------------------------------------------------------------
;; Phase 4: Vectorized HMC
;; ---------------------------------------------------------------------------

(println "\n-- vectorized-hmc --")
(let [samples (mcmc/vectorized-hmc
                {:samples 20 :burn 5 :step-size 0.005 :leapfrog-steps 5
                 :addresses [:mu] :n-chains 3 :device :cpu}
                model [10] observations)]
  (assert-true "vectorized-hmc returns correct sample count" (= 20 (count samples)))
  (let [rate (:acceptance-rate (meta samples))]
    (assert-true "vectorized-hmc has positive acceptance rate" (> rate 0))))

;; ---------------------------------------------------------------------------
;; Phase 5: Vectorized MAP
;; ---------------------------------------------------------------------------

(println "\n-- vectorized-map-optimize --")
(let [result (mcmc/vectorized-map-optimize
               {:iterations 500 :lr 0.01 :addresses [:mu] :n-restarts 5 :device :cpu}
               model [10] observations)]
  (assert-true "vectorized-map has trace" (some? (:trace result)))
  (assert-true "vectorized-map has score" (number? (:score result)))
  ;; MAP estimate for mu should be near 3.0 (posterior mean of Gaussian with obs=3)
  (let [p (:params result)
        best-mu (if (sequential? p) (first p) p)]
    (assert-close "MAP estimate near posterior mode" 3.0 best-mu 1.0)))

;; ---------------------------------------------------------------------------
;; Phase 6: GPU resampling
;; ---------------------------------------------------------------------------

(println "\n-- GPU systematic resampling --")
(let [log-weights (mx/array [-1.0 -2.0 -0.5 -3.0 -1.5])
      key (rng/fresh-key 42)
      indices (v/systematic-resample-indices-gpu log-weights 5 key)
      _ (mx/eval! indices)
      idx-clj (mx/->clj indices)]
  (assert-true "GPU resample returns 5 indices" (= 5 (count idx-clj)))
  (assert-true "GPU resample indices in range" (every? #(and (>= % 0) (< % 5)) idx-clj))
  ;; Weight -0.5 (index 2) should be resampled most often
  (let [freqs (frequencies idx-clj)]
    (assert-true "GPU resample: highest-weight particle appears"
                 (contains? freqs 2))))

;; Check GPU resampling produces valid indices for larger N
(println "\n-- GPU resampling validity (N=100) --")
(let [n 100
      log-weights (mx/array (mapv #(- (rand) 2.0) (range n)))
      key (rng/fresh-key 123)
      gpu-indices (v/systematic-resample-indices-gpu log-weights n key)
      _ (mx/eval! gpu-indices)
      gpu-clj (mx/->clj gpu-indices)]
  (assert-true "GPU resample returns N indices" (= n (count gpu-clj)))
  (assert-true "GPU resample all indices in [0, N)"
               (every? #(and (>= % 0) (< % n)) gpu-clj))
  ;; Higher-weight particles should appear more often
  (let [freqs (frequencies gpu-clj)
        max-freq (apply max (vals freqs))]
    (assert-true "GPU resample has some duplicates (expected for resampling)"
                 (> max-freq 1))))

(println "\nAll vectorized gradient tests complete.")
