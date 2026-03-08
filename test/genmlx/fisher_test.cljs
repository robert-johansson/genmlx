(ns genmlx.fisher-test
  "Tests for Tier 3e: Fisher information matrix.
   Tests observed Fisher, Laplace approximation,
   natural gradient, and standard errors."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.fisher :as fisher]
            [genmlx.inference.differentiable :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [label pred]
  (if pred
    (println (str "  PASS: " label))
    (println (str "  FAIL: " label))))

(defn assert-close [label expected actual tol]
  (let [ok (< (js/Math.abs (- expected actual)) tol)]
    (if ok
      (println (str "  PASS: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")"))
      (println (str "  FAIL: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")")))))

;; ---------------------------------------------------------------------------
;; Test 1: Analytical Fisher for single Gaussian
;; ---------------------------------------------------------------------------

;; Model: y ~ N(mu, 1) with known sigma=1, learnable mu
;; Observe K data points y_1,...,y_K
;; Fisher information for mu: F(mu) = K/sigma^2 = K

(println "\n=== Test 1: Observed Fisher — Gaussian mean (1 param) ===")

(def K-obs 10)
(def true-mu 3.0)
(def obs-data (mapv (fn [i] (+ true-mu (* 0.5 (- i 4.5)))) (range K-obs)))

(def model-1
  (gen []
    (let [mu (param :mu 0.0)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu 1.0)))
      mu)))

(def obs-1
  (apply cm/choicemap
    (mapcat (fn [i] [(keyword (str "y" i)) (mx/scalar (nth obs-data i))])
            (range K-obs))))

;; Analytical Fisher = K = 10
(let [params (mx/array [true-mu])
      {:keys [fisher log-ml]} (fisher/observed-fisher
                                 {:n-particles 5000 :key (rng/fresh-key 42)}
                                 model-1 [] obs-1 [:mu] params)]
  (mx/materialize! fisher)
  (let [f-val (mx/item (mx/mat-get fisher 0 0))]
    (println (str "  Observed Fisher: " (.toFixed f-val 2) " (analytical: " K-obs ".00)"))
    (println (str "  log-ML: " (.toFixed (mx/item log-ml) 2)))
    (assert-close "Fisher ≈ K" (double K-obs) f-val 2.0)))

;; ---------------------------------------------------------------------------
;; Test 2: 2D Fisher — Gaussian mean + log-scale
;; ---------------------------------------------------------------------------

;; Model: y ~ N(mu, exp(log_sigma)^2)
;; Observed Fisher at (mu, log_sigma):
;;   F_11 = K/sigma^2 = 10
;;   F_22 = 2*SS/sigma^2 where SS = Σ(y_i - mu)² (≈41.25 at sigma=1, mu=3)
;;   F_12 = F_21 ≈ 0

(println "\n=== Test 2: 2D Fisher — Gaussian (mu, log-sigma) ===")

(def model-2
  (gen []
    (let [mu (param :mu 0.0)
          log-sigma (param :log-sigma 0.0)
          sigma (mx/exp log-sigma)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu sigma)))
      mu)))

(let [sigma-val 1.0
      params (mx/array [true-mu (js/Math.log sigma-val)])
      {:keys [fisher log-ml]} (fisher/observed-fisher
                                 {:n-particles 5000 :key (rng/fresh-key 77)}
                                 model-2 [] obs-1 [:mu :log-sigma] params)]
  (mx/materialize! fisher)
  (let [f00 (mx/item (mx/mat-get fisher 0 0))
        f11 (mx/item (mx/mat-get fisher 1 1))
        f01 (mx/item (mx/mat-get fisher 0 1))]
    (println (str "  F[0,0] (mu,mu): " (.toFixed f00 2) " (analytical: " (.toFixed (/ K-obs (* sigma-val sigma-val)) 2) ")"))
    ;; SS = Σ(y_i - mu)² = Σ(0.5*(i-4.5))² for i=0..9
    (let [SS (reduce + (map (fn [y] (* (- y true-mu) (- y true-mu))) obs-data))
          expected-F11 (* 2.0 SS)]
      (println (str "  F[1,1] (ls,ls): " (.toFixed f11 2) " (analytical: " (.toFixed expected-F11 2) ")"))
      (println (str "  F[0,1] (off-diag): " (.toFixed f01 2) " (analytical: 0.00)"))
      (assert-close "F[0,0] ≈ K/σ²" (/ K-obs (* sigma-val sigma-val)) f00 3.0)
      (assert-close "F[1,1] ≈ 2·SS/σ²" expected-F11 f11 5.0)
      (assert-close "F[0,1] ≈ 0" 0.0 f01 2.0))))

;; ---------------------------------------------------------------------------
;; Test 3: Standard errors / Cramér-Rao bound
;; ---------------------------------------------------------------------------

(println "\n=== Test 3: Standard errors ===")

(let [params (mx/array [true-mu])
      {:keys [fisher]} (fisher/observed-fisher
                          {:n-particles 5000 :key (rng/fresh-key 42)}
                          model-1 [] obs-1 [:mu] params)
      {:keys [std-errors covariance]} (fisher/parameter-std-errors fisher)]
  (mx/materialize! std-errors)
  (let [se (mx/item (mx/index std-errors 0))
        ;; Analytical: SE = 1/sqrt(K) = 1/sqrt(10) ≈ 0.316
        analytical-se (/ 1.0 (js/Math.sqrt K-obs))]
    (println (str "  SE(mu): " (.toFixed se 3) " (analytical: " (.toFixed analytical-se 3) ")"))
    (assert-close "SE ≈ 1/√K" analytical-se se 0.1)))

;; ---------------------------------------------------------------------------
;; Test 4: Laplace log-evidence
;; ---------------------------------------------------------------------------

(println "\n=== Test 4: Laplace log-evidence ===")

(let [params (mx/array [true-mu])
      fisher-result (fisher/observed-fisher
                      {:n-particles 5000 :key (rng/fresh-key 42)}
                      model-1 [] obs-1 [:mu] params)
      {:keys [log-evidence log-ml log-det-fisher]} (fisher/laplace-log-evidence fisher-result 1)]
  (println (str "  log p(y|θ*): " (.toFixed log-ml 2)))
  (println (str "  log|F|: " (.toFixed log-det-fisher 2)))
  (println (str "  Laplace log p(y): " (.toFixed log-evidence 2)))
  (assert-true "log-evidence is finite" (js/isFinite log-evidence))
  ;; Laplace correction: 0.5*log(2π) - 0.5*log(10) ≈ 0.919 - 1.151 = -0.232
  ;; So log-evidence < log-ML (Occam's razor penalty)
  (assert-true "Laplace evidence < log-ML (Occam penalty)" (< log-evidence log-ml)))

;; ---------------------------------------------------------------------------
;; Test 5: Natural gradient step
;; ---------------------------------------------------------------------------

(println "\n=== Test 5: Natural gradient step ===")

;; Natural gradient for Gaussian mean: F⁻¹ · ∇ = (K)⁻¹ · K·(ȳ - μ) = (ȳ - μ)
;; So one natural gradient step with lr=1 should jump directly to the MLE!

(let [params (mx/array [0.0])  ;; Start far from truth
      key (rng/fresh-key 42)
      {:keys [fisher]} (fisher/observed-fisher
                         {:n-particles 5000 :key key}
                         model-1 [] obs-1 [:mu] params)
      {:keys [grad]} (diff/log-ml-gradient
                       {:n-particles 5000 :key key}
                       model-1 [] obs-1 [:mu] params)
      _ (mx/materialize! grad)
      new-params (fisher/natural-gradient-step fisher grad params {:lr 1.0})
      new-mu (mx/item (mx/index new-params 0))
      y-bar (/ (reduce + obs-data) K-obs)]
  (println (str "  Start: mu=0.000"))
  (println (str "  After 1 natural gradient step: mu=" (.toFixed new-mu 3)))
  (println (str "  Sample mean (MLE): " (.toFixed y-bar 3)))
  (assert-close "Natural grad → MLE in 1 step" y-bar new-mu 0.5))

;; ---------------------------------------------------------------------------
;; Test 6: Fisher symmetry and positive definiteness (2D)
;; ---------------------------------------------------------------------------

(println "\n=== Test 6: Fisher symmetry & positive definiteness ===")

(let [params (mx/array [true-mu 0.0])
      {:keys [fisher]} (fisher/observed-fisher
                         {:n-particles 5000 :key (rng/fresh-key 77)}
                         model-2 [] obs-1 [:mu :log-sigma] params)]
  (mx/materialize! fisher)
  (let [f00 (mx/item (mx/mat-get fisher 0 0))
        f11 (mx/item (mx/mat-get fisher 1 1))
        f01 (mx/item (mx/mat-get fisher 0 1))
        f10 (mx/item (mx/mat-get fisher 1 0))
        ;; Check symmetry: |F01 - F10| < ε
        sym-diff (js/Math.abs (- f01 f10))
        ;; Check positive definiteness: diagonal elements > 0
        det-val (- (* f00 f11) (* f01 f10))]
    (println (str "  |F[0,1] - F[1,0]| = " (.toFixed sym-diff 4)))
    (println (str "  det(F) = " (.toFixed det-val 2)))
    (assert-true "Fisher is symmetric" (< sym-diff 0.5))
    (assert-true "Fisher diagonal positive" (and (> f00 0) (> f11 0)))
    (assert-true "Fisher determinant positive" (> det-val 0))))

(println "\nDone.")
