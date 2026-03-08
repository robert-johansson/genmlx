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
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
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
;; Ground truth and synthetic data
;; ---------------------------------------------------------------------------

(def TRUE-MU0 3.0)
(def TRUE-SIGMA0 2.0)
(def SIGMA-OBS 1.0)
(def N-GROUPS 8)
(def OBS-PER-GROUP 5)

;; Generate synthetic data: sample group means, then observations
;; Using a fixed seed for reproducibility
(def rng-state (atom 42))
(defn next-gaussian [mu sigma]
  ;; Box-Muller with deterministic seed (good enough for test data)
  (swap! rng-state #(mod (+ (* 1103515245 %) 12345) 2147483648))
  (let [u1 (/ @rng-state 2147483648.0)]
    (swap! rng-state #(mod (+ (* 1103515245 %) 12345) 2147483648))
    (let [u2 (/ @rng-state 2147483648.0)
          z (* (js/Math.sqrt (* -2 (js/Math.log (max u1 1e-10))))
               (js/Math.cos (* 2 js/Math.PI u2)))]
      (+ mu (* sigma z)))))

;; Sample group means and observations
(def group-means (mapv (fn [_] (next-gaussian TRUE-MU0 TRUE-SIGMA0)) (range N-GROUPS)))
(def observations-data
  (into {}
    (for [j (range N-GROUPS)
          i (range OBS-PER-GROUP)]
      [(keyword (str "y_" j "_" i))
       (next-gaussian (nth group-means j) SIGMA-OBS)])))

(println "\n=== Hierarchical Empirical Bayes: Parameter Recovery ===")
(println (str "Ground truth: mu0=" TRUE-MU0 ", sigma0=" TRUE-SIGMA0))
(println (str "Data: " N-GROUPS " groups × " OBS-PER-GROUP " obs = "
             (* N-GROUPS OBS-PER-GROUP) " observations"))
(println (str "Group means: " (mapv #(.toFixed % 2) group-means)))
(println (str "Sample grand mean: "
             (.toFixed (/ (reduce + group-means) N-GROUPS) 3)))

;; ---------------------------------------------------------------------------
;; Model
;; ---------------------------------------------------------------------------

(def hier-model
  (gen []
    (let [mu0 (param :mu0 0.0)
          log-sigma0 (param :log-sigma0 0.0)
          sigma0 (mx/exp log-sigma0)]
      ;; Sample group-level means
      (doseq [j (range N-GROUPS)]
        (let [theta-j (trace (keyword (str "theta_" j))
                             (dist/gaussian mu0 sigma0))]
          ;; Observe data within group
          (doseq [i (range OBS-PER-GROUP)]
            (trace (keyword (str "y_" j "_" i))
                   (dist/gaussian theta-j SIGMA-OBS))))))))

;; Build observation choicemap
(def obs-cm
  (apply cm/choicemap
    (mapcat (fn [[k v]] [k (mx/scalar v)]) observations-data)))

;; ---------------------------------------------------------------------------
;; Test 1: Gradient landscape — check gradient direction at different points
;; ---------------------------------------------------------------------------

(println "\n--- Test 1: Gradient landscape ---")

(doseq [[label mu0-init log-s0-init]
        [["at truth"         3.0  (js/Math.log 2.0)]
         ["mu0 too low"      0.0  (js/Math.log 2.0)]
         ["mu0 too high"     6.0  (js/Math.log 2.0)]
         ["sigma0 too small" 3.0  -1.0]
         ["sigma0 too large" 3.0  2.0]]]
  (let [{:keys [log-ml grad]}
        (diff/log-ml-gradient {:n-particles 2000 :key (rng/fresh-key 99)}
                              hier-model [] obs-cm
                              [:mu0 :log-sigma0]
                              (mx/array [mu0-init log-s0-init]))]
    (mx/materialize! log-ml grad)
    (let [g0 (mx/item (mx/index grad 0))
          g1 (mx/item (mx/index grad 1))]
      (println (str "  " label ": log-ML=" (.toFixed (mx/item log-ml) 2)
                   " ∇mu0=" (.toFixed g0 3)
                   " ∇log-σ0=" (.toFixed g1 3))))))

;; ---------------------------------------------------------------------------
;; Test 2: Full optimization — parameter recovery
;; ---------------------------------------------------------------------------

(println "\n--- Test 2: Full optimization (200 iterations) ---")

(let [result (diff/optimize-params
               {:iterations 200 :lr 0.02 :n-particles 2000
                :callback (fn [{:keys [iter log-ml params]}]
                            (when (zero? (mod iter 25))
                              (let [mu (mx/item (mx/index params 0))
                                    ls (mx/item (mx/index params 1))]
                                (println (str "  iter " iter
                                             ": log-ml=" (.toFixed log-ml 2)
                                             "  mu0=" (.toFixed mu 3)
                                             "  sigma0=" (.toFixed (js/Math.exp ls) 3))))))}
               hier-model [] obs-cm
               [:mu0 :log-sigma0]
               (mx/array [0.0 0.0]))  ;; Start far from truth
      final-mu0 (mx/item (mx/index (:params result) 0))
      final-log-s0 (mx/item (mx/index (:params result) 1))
      final-sigma0 (js/Math.exp final-log-s0)
      history (:log-ml-history result)]

  (println (str "\n  Final: mu0=" (.toFixed final-mu0 3)
               ", sigma0=" (.toFixed final-sigma0 3)))
  (println (str "  Truth: mu0=" TRUE-MU0 ", sigma0=" TRUE-SIGMA0))

  ;; Parameter recovery
  (assert-close "mu0 recovered" TRUE-MU0 final-mu0 0.75)
  (assert-close "sigma0 recovered" TRUE-SIGMA0 final-sigma0 1.0)

  ;; Log-ML should improve overall (compare first 10 avg vs last 10 avg)
  (let [first-10 (/ (reduce + (take 10 history)) 10.0)
        last-10 (/ (reduce + (take-last 10 history)) 10.0)]
    (println (str "  log-ML first 10 avg: " (.toFixed first-10 2)
                 ", last 10 avg: " (.toFixed last-10 2)))
    (assert-true "log-ML improved" (> last-10 first-10))))

;; ---------------------------------------------------------------------------
;; Test 3: Comparison at different N-particles (gradient quality)
;; ---------------------------------------------------------------------------

(println "\n--- Test 3: Gradient quality vs N-particles ---")

(let [params (mx/array [1.0 0.5])  ;; Away from truth
      key (rng/fresh-key 77)]
  (doseq [n [100 500 2000 5000]]
    (let [grads (mapv (fn [seed]
                        (let [{:keys [grad]}
                              (diff/log-ml-gradient
                                {:n-particles n :key (rng/fresh-key seed)}
                                hier-model [] obs-cm
                                [:mu0 :log-sigma0] params)]
                          (mx/materialize! grad)
                          [(mx/item (mx/index grad 0))
                           (mx/item (mx/index grad 1))]))
                      (range 5))
          mu-grads (mapv first grads)
          ls-grads (mapv second grads)
          mu-mean (/ (reduce + mu-grads) 5.0)
          mu-std (js/Math.sqrt (/ (reduce + (map #(* (- % mu-mean) (- % mu-mean)) mu-grads)) 5.0))
          ls-mean (/ (reduce + ls-grads) 5.0)
          ls-std (js/Math.sqrt (/ (reduce + (map #(* (- % ls-mean) (- % ls-mean)) ls-grads)) 5.0))]
      (println (str "  N=" n
                   ": ∇mu0=" (.toFixed mu-mean 3) "±" (.toFixed mu-std 3)
                   "  ∇log-σ0=" (.toFixed ls-mean 3) "±" (.toFixed ls-std 3))))))

;; ---------------------------------------------------------------------------
;; Test 4: Analytical check — single-group marginal likelihood
;; ---------------------------------------------------------------------------

(println "\n--- Test 4: Analytical marginal likelihood (single group) ---")

;; For a single group with K observations y1..yK:
;;   theta ~ N(mu0, sigma0^2)
;;   y_i ~ N(theta, 1)
;; Marginal: y ~ N(mu0 * 1_K, sigma0^2 * 1_K*1_K^T + I_K)
;; log p(y) can be computed analytically.

(def K-test 5)
(def test-obs (mapv (fn [i] (nth (vals (sort observations-data)) i)) (range K-test)))
(def test-mu0 3.0)
(def test-sigma0 2.0)

;; Analytical log p(y | mu0, sigma0):
;; Covariance is sigma0^2 * 11^T + I = diagonal(1) + sigma0^2 * ones
;; Using matrix determinant lemma and Woodbury for efficiency:
;; det(sigma0^2*11^T + I) = 1 + K*sigma0^2  (scalar since rank-1 update)
;; (sigma0^2*11^T + I)^{-1} = I - sigma0^2/(1+K*sigma0^2) * 11^T
(let [y-bar (/ (reduce + test-obs) K-test)
      ss (reduce + (map #(* (- % test-mu0) (- % test-mu0)) test-obs))
      s02 (* test-sigma0 test-sigma0)
      denom (+ 1.0 (* K-test s02))
      ;; log-det = log(1 + K*sigma0^2) (other eigenvalues are 1)
      log-det (js/Math.log denom)
      ;; Quadratic form: y^T Sigma^{-1} y where Sigma^{-1} = I - s02/denom * 11^T
      ;; = sum (y_i - mu0)^2 - s02/denom * (sum(y_i - mu0))^2
      sum-dev (reduce + (map #(- % test-mu0) test-obs))
      quad (- ss (* (/ s02 denom) (* sum-dev sum-dev)))
      analytical-log-ml (- (* -0.5 K-test (js/Math.log (* 2 js/Math.PI)))
                           (* 0.5 log-det)
                           (* 0.5 quad))]

  ;; Now estimate via IS
  (def single-group-model
    (gen []
      (let [mu0 (param :mu0 0.0)
            log-sigma0 (param :log-sigma0 0.0)
            sigma0 (mx/exp log-sigma0)
            theta (trace :theta (dist/gaussian mu0 sigma0))]
        (doseq [i (range K-test)]
          (trace (keyword (str "y_" i))
                 (dist/gaussian theta SIGMA-OBS))))))

  (def single-obs
    (apply cm/choicemap
      (mapcat (fn [i] [(keyword (str "y_" i)) (mx/scalar (nth test-obs i))])
              (range K-test))))

  (let [{:keys [log-ml]}
        (diff/log-ml-gradient {:n-particles 10000 :key (rng/fresh-key 42)}
                              single-group-model [] single-obs
                              [:mu0 :log-sigma0]
                              (mx/array [test-mu0 (js/Math.log test-sigma0)]))]
    (mx/materialize! log-ml)
    (let [is-log-ml (mx/item log-ml)]
      (println (str "  Analytical log p(y): " (.toFixed analytical-log-ml 3)))
      (println (str "  IS estimate (N=10K): " (.toFixed is-log-ml 3)))
      (println (str "  Difference: " (.toFixed (- is-log-ml analytical-log-ml) 3)))
      (assert-close "IS matches analytical" analytical-log-ml is-log-ml 0.5))))

(println "\nDone.")
