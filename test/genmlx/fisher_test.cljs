(ns genmlx.fisher-test
  "Tests for Tier 3e: Fisher information matrix."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.fisher :as fisher]
            [genmlx.inference.differentiable :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Shared setup
;; ---------------------------------------------------------------------------

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

(def model-2
  (gen []
    (let [mu (param :mu 0.0)
          log-sigma (param :log-sigma 0.0)
          sigma (mx/exp log-sigma)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu sigma)))
      mu)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest observed-fisher-gaussian-1d
  (testing "Observed Fisher for single Gaussian mean"
    (let [params (mx/array [true-mu])
          {:keys [fisher log-ml]} (fisher/observed-fisher
                                     {:n-particles 5000 :key (rng/fresh-key 42)}
                                     model-1 [] obs-1 [:mu] params)]
      (mx/materialize! fisher)
      (let [f-val (mx/item (mx/mat-get fisher 0 0))]
        (is (h/close? (double K-obs) f-val 2.0) "Fisher ~ K")))))

(deftest observed-fisher-gaussian-2d
  (testing "2D Fisher for Gaussian (mu, log-sigma)"
    (let [sigma-val 1.0
          params (mx/array [true-mu (js/Math.log sigma-val)])
          {:keys [fisher]} (fisher/observed-fisher
                              {:n-particles 5000 :key (rng/fresh-key 77)}
                              model-2 [] obs-1 [:mu :log-sigma] params)]
      (mx/materialize! fisher)
      (let [f00 (mx/item (mx/mat-get fisher 0 0))
            f11 (mx/item (mx/mat-get fisher 1 1))
            f01 (mx/item (mx/mat-get fisher 0 1))
            SS (reduce + (map (fn [y] (* (- y true-mu) (- y true-mu))) obs-data))
            expected-F11 (* 2.0 SS)]
        (is (h/close? (/ K-obs (* sigma-val sigma-val)) f00 3.0) "F[0,0] ~ K/sigma^2")
        (is (h/close? expected-F11 f11 5.0) "F[1,1] ~ 2*SS/sigma^2")
        (is (h/close? 0.0 f01 2.0) "F[0,1] ~ 0")))))

(deftest standard-errors
  (testing "Standard errors / Cramer-Rao bound"
    (let [params (mx/array [true-mu])
          {:keys [fisher]} (fisher/observed-fisher
                              {:n-particles 5000 :key (rng/fresh-key 42)}
                              model-1 [] obs-1 [:mu] params)
          {:keys [std-errors]} (fisher/parameter-std-errors fisher)]
      (mx/materialize! std-errors)
      (let [se (mx/item (mx/index std-errors 0))
            analytical-se (/ 1.0 (js/Math.sqrt K-obs))]
        (is (h/close? analytical-se se 0.1) "SE ~ 1/sqrt(K)")))))

(deftest laplace-log-evidence
  (testing "Laplace log-evidence"
    (let [params (mx/array [true-mu])
          fisher-result (fisher/observed-fisher
                          {:n-particles 5000 :key (rng/fresh-key 42)}
                          model-1 [] obs-1 [:mu] params)
          {:keys [log-evidence log-ml]} (fisher/laplace-log-evidence fisher-result 1)]
      (is (js/isFinite log-evidence) "log-evidence is finite")
      (is (< log-evidence log-ml) "Laplace evidence < log-ML (Occam penalty)"))))

(deftest natural-gradient-step
  (testing "Natural gradient step jumps to MLE"
    (let [params (mx/array [0.0])
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
      (is (h/close? y-bar new-mu 0.5) "Natural grad -> MLE in 1 step"))))

(deftest fisher-symmetry-positive-definiteness
  (testing "Fisher symmetry and positive definiteness (2D)"
    (let [params (mx/array [true-mu 0.0])
          {:keys [fisher]} (fisher/observed-fisher
                             {:n-particles 5000 :key (rng/fresh-key 77)}
                             model-2 [] obs-1 [:mu :log-sigma] params)]
      (mx/materialize! fisher)
      (let [f00 (mx/item (mx/mat-get fisher 0 0))
            f11 (mx/item (mx/mat-get fisher 1 1))
            f01 (mx/item (mx/mat-get fisher 0 1))
            f10 (mx/item (mx/mat-get fisher 1 0))
            sym-diff (js/Math.abs (- f01 f10))
            det-val (- (* f00 f11) (* f01 f10))]
        (is (< sym-diff 0.5) "Fisher is symmetric")
        (is (and (> f00 0) (> f11 0)) "Fisher diagonal positive")
        (is (> det-val 0) "Fisher determinant positive")))))

(cljs.test/run-tests)
