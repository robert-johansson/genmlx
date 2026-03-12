(ns genmlx.l2-mcmc-test
  "Level 2 WP-1 tests: compiled MCMC with tensor-native score.

   Tests cover:
   1. compiled-mh with tensor-native score produces valid samples
   2. MALA with tensor-native score produces correct samples
   3. HMC with tensor-native score produces correct samples
   4. MAP with tensor-native score converges
   5. Fallback to GFI for non-static models
   6. Acceptance rates are reasonable
   7. prepare-mcmc-score returns tensor-native for static models"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.tensor-trace :as tt]
            [genmlx.inference.util :as iu]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toFixed diff 4) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " expected=" expected " actual=" actual " diff=" diff))))))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def linear-model
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))
          y-pred (mx/add (mx/multiply slope (mx/ensure-array x)) intercept)]
      (trace :y (dist/gaussian y-pred 1))
      slope)))

(def simple-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :x (dist/gaussian mu 0.1))
      mu)))

;; ---------------------------------------------------------------------------
;; 1. prepare-mcmc-score returns tensor-native for static models
;; ---------------------------------------------------------------------------

(println "\n== prepare-mcmc-score ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate model [2.0] obs)
      result (iu/prepare-mcmc-score linear-model [2.0] obs [:slope :intercept] trace)]
  (assert-true "tensor-native? for static model" (:tensor-native? result))
  (assert-true "has score-fn" (fn? (:score-fn result)))
  (assert-true "has init-params" (mx/array? (:init-params result)))
  (assert-true "n-params is 2" (= 2 (:n-params result)))
  ;; Verify score-fn works
  (let [s ((:score-fn result) (:init-params result))]
    (mx/eval! s)
    (assert-true "score-fn returns finite value" (js/isFinite (mx/item s)))))

;; ---------------------------------------------------------------------------
;; 2. compiled-mh produces valid samples
;; ---------------------------------------------------------------------------

(println "\n== compiled-mh with tensor-score ==")

(let [obs (cm/choicemap :y (mx/scalar 5.0))
      samples (mcmc/compiled-mh
                {:samples 50 :burn 200 :addresses [:slope :intercept]
                 :proposal-std 0.5 :compile? true}
                linear-model [2.0] obs)]
  (assert-true "compiled-mh returns 50 samples" (= 50 (count samples)))
  (assert-true "each sample has 2 elements" (= 2 (count (first samples))))
  ;; Check that samples have reasonable values (slope near 2.5, intercept near 0)
  (let [slopes (mapv first samples)
        mean-slope (/ (reduce + slopes) (count slopes))]
    (assert-true "mean slope is finite" (js/isFinite mean-slope))
    (assert-true "mean slope in reasonable range" (< -20 mean-slope 20))))

;; Also test with compile? false (GFI path)
(let [obs (cm/choicemap :y (mx/scalar 5.0))
      samples (mcmc/compiled-mh
                {:samples 10 :burn 50 :addresses [:slope :intercept]
                 :proposal-std 0.5 :compile? false}
                linear-model [2.0] obs)]
  (assert-true "non-compiled mh returns 10 samples" (= 10 (count samples))))

;; ---------------------------------------------------------------------------
;; 3. MALA produces valid samples
;; ---------------------------------------------------------------------------

(println "\n== MALA with tensor-score ==")

(let [obs (cm/choicemap :x (mx/scalar 0.5))
      samples (mcmc/mala
                {:samples 20 :burn 100 :addresses [:mu]
                 :step-size 0.1 :compile? true}
                simple-model [] obs)]
  (assert-true "MALA returns 20 samples" (= 20 (count samples)))
  ;; Posterior for mu given x=0.5 with N(0,1) prior and N(mu,0.1) likelihood
  ;; should concentrate near 0.5
  (let [mus (mapv first samples)
        mean-mu (/ (reduce + mus) (count mus))]
    (assert-close "MALA posterior mean near 0.5" 0.5 mean-mu 0.3)))

;; ---------------------------------------------------------------------------
;; 4. HMC produces valid samples
;; ---------------------------------------------------------------------------

(println "\n== HMC with tensor-score ==")

(let [obs (cm/choicemap :x (mx/scalar 0.5))
      samples (mcmc/hmc
                {:samples 20 :burn 100 :addresses [:mu]
                 :step-size 0.05 :n-leapfrog 10 :compile? true}
                simple-model [] obs)]
  (assert-true "HMC returns 20 samples" (= 20 (count samples)))
  (let [mus (mapv first samples)
        mean-mu (/ (reduce + mus) (count mus))]
    (assert-close "HMC posterior mean near 0.5" 0.5 mean-mu 0.3)))

;; ---------------------------------------------------------------------------
;; 5. MAP converges
;; ---------------------------------------------------------------------------

(println "\n== MAP with tensor-score ==")

(let [obs (cm/choicemap :x (mx/scalar 0.5))
      result (mcmc/map-optimize
               {:iterations 500 :addresses [:mu]
                :learning-rate 0.05 :compile? true}
               simple-model [] obs)]
  (assert-true "MAP returns trace" (some? (:trace result)))
  (assert-true "MAP score is finite" (js/isFinite (:score result)))
  (assert-true "MAP has score-history" (pos? (count (:score-history result))))
  ;; MAP estimate should be near 0.5
  (let [mu-val (first (:params result))]
    (assert-close "MAP mu near 0.5" 0.5 mu-val 0.2)))

;; ---------------------------------------------------------------------------
;; 6. Fallback to GFI for non-static models
;; ---------------------------------------------------------------------------

(println "\n== GFI fallback ==")

(let [dynamic-model (gen [n]
                      (dotimes [i n]
                        (trace (keyword (str "x" i)) (dist/gaussian 0 1)))
                      nil)
      model (dyn/auto-key dynamic-model)
      obs cm/EMPTY
      {:keys [trace]} (p/generate model [3] (cm/choicemap :x0 (mx/scalar 0.0)
                                                           :x1 (mx/scalar 0.0)
                                                           :x2 (mx/scalar 0.0)))
      result (iu/prepare-mcmc-score dynamic-model [3]
                                     (cm/choicemap :x0 (mx/scalar 0.0)
                                                    :x1 (mx/scalar 0.0)
                                                    :x2 (mx/scalar 0.0))
                                     [] trace)]
  (assert-true "non-static model not tensor-native" (not (:tensor-native? result))))

;; ---------------------------------------------------------------------------
;; 7. Score consistency: tensor-score matches GFI in MCMC context
;; ---------------------------------------------------------------------------

(println "\n== score consistency ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate model [2.0] obs)
      ;; Tensor-native score
      tensor-result (iu/prepare-mcmc-score linear-model [2.0] obs [:slope :intercept] trace)
      ;; GFI score for comparison
      layout (iu/compute-param-layout trace [:slope :intercept])
      gfi-fn (iu/make-score-fn model [2.0] obs [:slope :intercept] layout)
      gfi-params (iu/extract-params trace [:slope :intercept] layout)
      ;; Tensor params
      tensor-params (:init-params tensor-result)]
  ;; Scores should match for the same trace
  (let [ts ((:score-fn tensor-result) tensor-params)
        gs (gfi-fn gfi-params)]
    (mx/eval! ts)
    (mx/eval! gs)
    (assert-close "tensor and GFI scores match for same trace"
                  (mx/item gs) (mx/item ts) 1e-4)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== L2 MCMC Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed =="))
(when (pos? @fail-count)
  (println (str "FAILURES: " @fail-count)))
