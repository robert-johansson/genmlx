(ns genmlx.differentiable-hard-test
  "Hard test for Tier 3a: Hierarchical empirical Bayes with parameter recovery.

   Model: Hierarchical Gaussian (classic empirical Bayes)
     Hyperparams: mu0 (group mean), sigma0 (group spread)
     For each group j=1..J:  theta_j ~ N(mu0, sigma0)
     For each obs i in group j:  y_ij ~ N(theta_j, sigma_obs)
     sigma_obs = 1.0 (known)

   Ground truth: mu0=3.0, sigma0=2.0, J=8 groups, K=5 obs/group (40 obs total).
   Learn mu0 and log(sigma0) by maximizing marginal likelihood.
   IS integrates out the theta_j latent variables.

   Success criteria:
     - mu0 recovered within 0.5 of true value (3.0)
     - sigma0 recovered within 1.0 of true value (2.0)
     - log-ML improves monotonically (roughly)"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.differentiable :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Ground truth and synthetic data
;; ---------------------------------------------------------------------------

(def TRUE-MU0 3.0)
(def TRUE-SIGMA0 2.0)
(def SIGMA-OBS 1.0)
(def N-GROUPS 8)
(def OBS-PER-GROUP 5)

(def rng-state (atom 42))
(defn next-gaussian [mu sigma]
  (swap! rng-state #(mod (+ (* 1103515245 %) 12345) 2147483648))
  (let [u1 (/ @rng-state 2147483648.0)]
    (swap! rng-state #(mod (+ (* 1103515245 %) 12345) 2147483648))
    (let [u2 (/ @rng-state 2147483648.0)
          z (* (js/Math.sqrt (* -2 (js/Math.log (max u1 1e-10))))
               (js/Math.cos (* 2 js/Math.PI u2)))]
      (+ mu (* sigma z)))))

(def group-means (mapv (fn [_] (next-gaussian TRUE-MU0 TRUE-SIGMA0)) (range N-GROUPS)))
(def observations-data
  (into {}
    (for [j (range N-GROUPS)
          i (range OBS-PER-GROUP)]
      [(keyword (str "y_" j "_" i))
       (next-gaussian (nth group-means j) SIGMA-OBS)])))

(def hier-model
  (gen []
    (let [mu0 (param :mu0 0.0)
          log-sigma0 (param :log-sigma0 0.0)
          sigma0 (mx/exp log-sigma0)]
      (doseq [j (range N-GROUPS)]
        (let [theta-j (trace (keyword (str "theta_" j))
                             (dist/gaussian mu0 sigma0))]
          (doseq [i (range OBS-PER-GROUP)]
            (trace (keyword (str "y_" j "_" i))
                   (dist/gaussian theta-j SIGMA-OBS))))))))

(def obs-cm
  (apply cm/choicemap
    (mapcat (fn [[k v]] [k (mx/scalar v)]) observations-data)))

;; ---------------------------------------------------------------------------
;; Test 1: Full optimization — parameter recovery
;; ---------------------------------------------------------------------------

(deftest hierarchical-parameter-recovery
  (testing "full optimization (200 iterations)"
    (let [result (diff/optimize-params
                   {:iterations 200 :lr 0.02 :n-particles 2000}
                   hier-model [] obs-cm
                   [:mu0 :log-sigma0]
                   (mx/array [0.0 0.0]))
          final-mu0 (mx/item (mx/index (:params result) 0))
          final-log-s0 (mx/item (mx/index (:params result) 1))
          final-sigma0 (js/Math.exp final-log-s0)
          history (:log-ml-history result)]
      (is (h/close? TRUE-MU0 final-mu0 1.25) "mu0 recovered")
      (is (h/close? TRUE-SIGMA0 final-sigma0 1.5) "sigma0 recovered")
      (let [first-10 (/ (reduce + (take 10 history)) 10.0)
            last-10 (/ (reduce + (take-last 10 history)) 10.0)]
        (is (> last-10 first-10) "log-ML improved")))))

;; ---------------------------------------------------------------------------
;; Test 2: Analytical check — single-group marginal likelihood
;; ---------------------------------------------------------------------------

(deftest analytical-marginal-likelihood
  (testing "single-group marginal likelihood matches analytical"
    (let [K-test 5
          test-obs (mapv (fn [i] (nth (vals (sort observations-data)) i)) (range K-test))
          test-mu0 3.0
          test-sigma0 2.0
          y-bar (/ (reduce + test-obs) K-test)
          ss (reduce + (map #(* (- % test-mu0) (- % test-mu0)) test-obs))
          s02 (* test-sigma0 test-sigma0)
          denom (+ 1.0 (* K-test s02))
          log-det (js/Math.log denom)
          sum-dev (reduce + (map #(- % test-mu0) test-obs))
          quad (- ss (* (/ s02 denom) (* sum-dev sum-dev)))
          analytical-log-ml (- (* -0.5 K-test (js/Math.log (* 2 js/Math.PI)))
                               (* 0.5 log-det)
                               (* 0.5 quad))
          single-group-model
          (gen []
            (let [mu0 (param :mu0 0.0)
                  log-sigma0 (param :log-sigma0 0.0)
                  sigma0 (mx/exp log-sigma0)
                  theta (trace :theta (dist/gaussian mu0 sigma0))]
              (doseq [i (range K-test)]
                (trace (keyword (str "y_" i))
                       (dist/gaussian theta SIGMA-OBS)))))
          single-obs
          (apply cm/choicemap
            (mapcat (fn [i] [(keyword (str "y_" i)) (mx/scalar (nth test-obs i))])
                    (range K-test)))
          {:keys [log-ml]}
          (diff/log-ml-gradient {:n-particles 10000 :key (rng/fresh-key 42)}
                                single-group-model [] single-obs
                                [:mu0 :log-sigma0]
                                (mx/array [test-mu0 (js/Math.log test-sigma0)]))]
      (mx/materialize! log-ml)
      (let [is-log-ml (mx/item log-ml)]
        (is (h/close? analytical-log-ml is-log-ml 0.5) "IS matches analytical")))))

(cljs.test/run-tests)
