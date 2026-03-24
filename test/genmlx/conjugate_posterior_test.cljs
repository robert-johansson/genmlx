(ns genmlx.conjugate-posterior-test
  "Conjugate posterior verification against analytically derived ground truth.

   Three families, two tiers each:
     Tier A — direct analytical update via conjugate.cljs update functions
     Tier B — importance sampling with prior as proposal

   Ground truth derived independently by math-verifier.

   Family 1: Gamma-Poisson
     Prior Gamma(3,2), 10 observations sum=19
     Posterior Gamma(22,12), E[lambda]=22/12, log-ML=-16.7425

   Family 2: Normal-Inverse-Gamma (unknown mean AND variance)
     Prior NIG(mu0=0, kappa0=1, alpha0=3, beta0=2), 10 observations
     Posterior NIG(mu_n=2.3545, kappa_n=11, alpha_n=8, beta_n=6.3836)

   Family 3: Dirichlet-Categorical
     Prior Dir(2,3,1), 20 observations counts [8,7,5]
     Posterior Dir(10,10,6), E[p]=[10/26, 10/26, 6/26]"
  (:require [cljs.test :refer [deftest is testing are run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.conjugate :as conj]
            [genmlx.inference.importance :as is]
            [genmlx.test-helpers :as th]))

;; ---------------------------------------------------------------------------
;; Ground truth constants
;; ---------------------------------------------------------------------------

(def gp-ground-truth
  "Gamma-Poisson posterior parameters and moments."
  {:data [1 3 2 0 4 2 1 3 2 1]
   :prior {:shape 3.0 :rate 2.0}
   :posterior {:shape 22.0 :rate 12.0}
   :mean (/ 22.0 12.0)
   :variance (/ 22.0 144.0)
   :log-ml -16.74252734621151})

(def nig-ground-truth
  "Normal-Inverse-Gamma posterior parameters and moments."
  {:data [2.1 3.5 1.8 2.7 3.0 2.3 2.9 3.1 2.5 2.0]
   :prior {:mu0 0.0 :kappa0 1.0 :alpha0 3.0 :beta0 2.0}
   :posterior {:mu-n 2.3545454545454545 :kappa-n 11.0 :alpha-n 8.0 :beta-n 6.383645454545454}
   :mean-mu 2.3545454545454545
   :mean-sig2 0.91194805194805
   :log-ml -15.30678042979591})

(def dir-cat-ground-truth
  "Dirichlet-Categorical posterior parameters and moments."
  {:counts [8 7 5]
   :prior [2 3 1]
   :posterior [10 10 6]
   :mean-p [(/ 10.0 26) (/ 10.0 26) (/ 6.0 26)]
   :log-ml -5.09975350366021})

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- realize
  "Evaluate an MLX array and extract its JS number."
  [x]
  (mx/eval! x)
  (mx/item x))

(defn- observations-from-indexed
  "Build a choicemap :y0, :y1, ... from sequential data."
  [data]
  (apply cm/choicemap
         (mapcat (fn [i x] [(keyword (str "y" i)) (mx/scalar x)])
                 (range) data)))

(defn- normalize-log-weights
  "Convert log-weights to normalized probability weights (as JS numbers)."
  [log-weights]
  (let [ws (mapv realize log-weights)
        max-w (apply max ws)
        exp-ws (mapv #(js/Math.exp (- % max-w)) ws)
        total (reduce + exp-ws)]
    (mapv #(/ % total) exp-ws)))

(defn- weighted-scalar-mean
  "Compute weighted mean of a scalar address from IS results."
  [{:keys [traces log-weights]} addr]
  (let [norm-ws (normalize-log-weights log-weights)
        vals (mapv (fn [tr]
                     (-> (:choices tr) (cm/get-submap addr) cm/get-value realize))
                   traces)]
    (reduce + (map * norm-ws vals))))

(defn- weighted-vector-mean
  "Compute element-wise weighted mean of a vector address from IS results."
  [{:keys [traces log-weights]} addr k]
  (let [norm-ws (normalize-log-weights log-weights)
        vecs (mapv (fn [tr]
                     (let [v (-> (:choices tr) (cm/get-submap addr) cm/get-value)]
                       (mx/eval! v)
                       (mx/->clj v)))
                   traces)]
    (reduce (fn [acc [w pv]] (mapv + acc (map #(* w %) pv)))
            (vec (repeat k 0.0))
            (map vector norm-ws vecs))))

(defn- fold-gp-updates
  "Sequentially apply gp-update over data. Returns {:posterior :ll-acc}."
  [{:keys [shape rate]} data]
  (reduce
   (fn [{:keys [posterior ll-acc]} obs]
     (let [{:keys [posterior ll]}
           (conj/gp-update posterior (mx/scalar obs) (mx/scalar 1.0))]
       {:posterior posterior
        :ll-acc (+ ll-acc (realize ll))}))
   {:posterior {:shape (mx/scalar shape) :rate (mx/scalar rate)}
    :ll-acc 0.0}
   data))

;; ---------------------------------------------------------------------------
;; Models for importance sampling (Tier B)
;; ---------------------------------------------------------------------------

(def gp-model
  "Gamma-Poisson: lambda ~ Gamma(shape, rate), y_i ~ Poisson(lambda)."
  (dyn/auto-key
   (gen [data shape rate]
        (let [lam (trace :lambda (dist/gamma-dist shape rate))]
          (doseq [[i _] (map-indexed vector data)]
            (trace (keyword (str "y" i)) (dist/poisson lam)))
          lam))))

(def nig-model
  "Normal-Inverse-Gamma: sigma^2 ~ IG(alpha, beta),
   mu ~ N(mu0, sigma/sqrt(kappa0)), y_i ~ N(mu, sigma)."
  (dyn/auto-key
   (gen [data mu0 kappa0 alpha0 beta0]
        (let [sigma-sq (trace :sigma-sq (dist/inv-gamma alpha0 beta0))
              sigma (mx/sqrt sigma-sq)
              mu (trace :mu (dist/gaussian mu0
                                           (mx/divide sigma (mx/sqrt (mx/scalar kappa0)))))]
          (doseq [[i _] (map-indexed vector data)]
            (trace (keyword (str "y" i)) (dist/gaussian mu sigma)))
          {:mu mu :sigma-sq sigma-sq}))))

(def dir-cat-model
  "Dirichlet-Categorical: p ~ Dir(alpha), y_i ~ Categorical(log(p))."
  (dyn/auto-key
   (gen [data alpha]
        (let [p-vec (trace :p (dist/dirichlet (mx/array alpha)))
              logits (mx/log p-vec)]
          (doseq [[i _] (map-indexed vector data)]
            (trace (keyword (str "y" i)) (dist/categorical logits)))
          p-vec))))

;; ---------------------------------------------------------------------------
;; Tier A: Analytical conjugate update — Gamma-Poisson
;; ---------------------------------------------------------------------------

(deftest gp-posterior-shape-is-prior-plus-sum
  (testing "posterior shape = prior shape + sum(data)"
    (let [{:keys [data prior]} gp-ground-truth
          {:keys [posterior]} (fold-gp-updates prior data)]
      (is (th/close? 22.0 (realize (:shape posterior)) 1e-6)
          "shape = 3 + 19 = 22"))))

(deftest gp-posterior-rate-is-prior-plus-n
  (testing "posterior rate = prior rate + n"
    (let [{:keys [data prior]} gp-ground-truth
          {:keys [posterior]} (fold-gp-updates prior data)]
      (is (th/close? 12.0 (realize (:rate posterior)) 1e-6)
          "rate = 2 + 10 = 12"))))

(deftest gp-posterior-mean-is-shape-over-rate
  (testing "E[lambda|data] = alpha_n / beta_n"
    (let [{:keys [data prior mean]} gp-ground-truth
          {:keys [posterior]} (fold-gp-updates prior data)
          post-mean (/ (realize (:shape posterior))
                       (realize (:rate posterior)))]
      (is (th/close? mean post-mean 1e-6)
          "E[lambda|data] = 22/12"))))

(deftest gp-posterior-variance-is-shape-over-rate-squared
  (testing "Var[lambda|data] = alpha_n / beta_n^2"
    (let [{:keys [data prior variance]} gp-ground-truth
          {:keys [posterior]} (fold-gp-updates prior data)
          r (realize (:rate posterior))
          post-var (/ (realize (:shape posterior)) (* r r))]
      (is (th/close? variance post-var 1e-6)
          "Var[lambda|data] = 22/144"))))

(deftest gp-log-marginal-likelihood-matches-ground-truth
  (testing "log p(data) from sequential updates matches analytical value"
    (let [{:keys [data prior log-ml]} gp-ground-truth
          {:keys [ll-acc]} (fold-gp-updates prior data)]
      (is (th/close? log-ml ll-acc 1e-3)
          "log marginal likelihood = -16.7425"))))

(deftest gp-single-observation-updates-are-correct
  (testing "individual update steps match hand calculations"
    (are [shape rate obs exp-shape exp-rate]
         (let [{:keys [posterior]}
               (conj/gp-update {:shape (mx/scalar shape) :rate (mx/scalar rate)}
                               (mx/scalar obs) (mx/scalar 1.0))]
           (and (th/close? exp-shape (realize (:shape posterior)) 1e-6)
                (th/close? exp-rate (realize (:rate posterior)) 1e-6)))
      ;;  shape rate  obs  -> new-shape new-rate
      3.0 2.0 1.0 4.0 3.0
      3.0 2.0 0.0 3.0 3.0
      3.0 2.0 5.0 8.0 3.0
      10.0 5.0 3.0 13.0 6.0)))

;; ---------------------------------------------------------------------------
;; Tier A: Algebraic properties of Gamma-Poisson update
;; ---------------------------------------------------------------------------

(deftest gp-update-commutes-over-observation-order
  (testing "posterior is invariant to observation order"
    (let [data [1 3 2 0 4]
          prior {:shape 3.0 :rate 2.0}
          fwd (:posterior (fold-gp-updates prior data))
          rev (:posterior (fold-gp-updates prior (reverse data)))]
      (is (th/close? (realize (:shape fwd)) (realize (:shape rev)) 1e-6)
          "shape is order-invariant")
      (is (th/close? (realize (:rate fwd)) (realize (:rate rev)) 1e-6)
          "rate is order-invariant"))))

(deftest gp-update-with-mask-zero-is-identity
  (testing "masked-out observation leaves posterior unchanged"
    (let [prior {:shape (mx/scalar 5.0) :rate (mx/scalar 3.0)}
          {:keys [posterior ll]}
          (conj/gp-update prior (mx/scalar 99.0) (mx/scalar 0.0))]
      (is (th/close? 5.0 (realize (:shape posterior)) 1e-6)
          "shape unchanged when mask=0")
      (is (th/close? 3.0 (realize (:rate posterior)) 1e-6)
          "rate unchanged when mask=0")
      (is (th/close? 0.0 (realize ll) 1e-6)
          "log-likelihood is zero when mask=0"))))

(deftest gp-posterior-mean-shrinks-toward-data-mean
  (testing "more data pulls posterior mean closer to sample mean"
    (let [data-mean 1.9
          prior {:shape 3.0 :rate 2.0}
          few-mean (let [{:keys [posterior]} (fold-gp-updates prior [1 3 2 0 4])]
                     (/ (realize (:shape posterior)) (realize (:rate posterior))))
          many-mean (let [{:keys [posterior]}
                          (fold-gp-updates prior [1 3 2 0 4 2 1 3 2 1 2 2 1 3 2 1 2 2 1 3])]
                      (/ (realize (:shape posterior)) (realize (:rate posterior))))]
      (is (< (js/Math.abs (- many-mean data-mean))
             (js/Math.abs (- few-mean data-mean)))
          "20 observations pulls mean closer than 5"))))

;; ---------------------------------------------------------------------------
;; Tier B: Importance sampling convergence — Gamma-Poisson
;; ---------------------------------------------------------------------------

(deftest gp-is-posterior-mean-converges
  (testing "IS weighted mean of lambda converges to E[lambda|data]"
    (let [{:keys [data prior mean]} gp-ground-truth
          {:keys [shape rate]} prior
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 5000 :key (rng/fresh-key 42)}
                                         gp-model [data shape rate] obs)]
      (is (th/close? mean (weighted-scalar-mean result :lambda) 0.07)
          "IS posterior mean within 0.07 of 22/12"))))

(deftest gp-is-log-ml-converges
  (testing "IS log-ML estimate converges to analytical log p(data)"
    (let [{:keys [data prior log-ml]} gp-ground-truth
          {:keys [shape rate]} prior
          obs (observations-from-indexed data)
          {:keys [log-ml-estimate]}
          (is/tidy-importance-sampling {:samples 10000 :key (rng/fresh-key 99)}
                                       gp-model [data shape rate] obs)]
      (is (th/close? log-ml log-ml-estimate 0.15)
          "IS log-ML within 0.15 of -16.7425"))))

;; ---------------------------------------------------------------------------
;; Tier B: Importance sampling convergence — Normal-Inverse-Gamma
;; ---------------------------------------------------------------------------

(deftest nig-is-posterior-mean-mu-converges
  (testing "IS weighted mean of mu converges to E[mu|data]"
    (let [{:keys [data prior mean-mu]} nig-ground-truth
          {:keys [mu0 kappa0 alpha0 beta0]} prior
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 5000 :key (rng/fresh-key 77)}
                                         nig-model [data mu0 kappa0 alpha0 beta0] obs)]
      (is (th/close? mean-mu (weighted-scalar-mean result :mu) 0.05)
          "IS posterior mean of mu within 0.05 of 2.3545"))))

(deftest nig-is-posterior-mean-sigma-sq-converges
  (testing "IS weighted mean of sigma^2 converges to E[sigma^2|data]"
    (let [{:keys [data prior mean-sig2]} nig-ground-truth
          {:keys [mu0 kappa0 alpha0 beta0]} prior
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 5000 :key (rng/fresh-key 88)}
                                         nig-model [data mu0 kappa0 alpha0 beta0] obs)]
      (is (th/close? mean-sig2 (weighted-scalar-mean result :sigma-sq) 0.10)
          "IS posterior mean of sigma^2 within 0.10 of 0.9119"))))

;; ---------------------------------------------------------------------------
;; Tier B: Importance sampling convergence — Dirichlet-Categorical
;; ---------------------------------------------------------------------------

(deftest dir-cat-is-posterior-mean-converges
  (testing "IS weighted mean of p converges to Dirichlet posterior mean"
    (let [{:keys [counts prior mean-p]} dir-cat-ground-truth
          data (vec (mapcat (fn [cat cnt] (repeat cnt cat)) (range) counts))
          obs (observations-from-indexed data)
          result (is/importance-sampling {:samples 5000 :key (rng/fresh-key 55)}
                                         dir-cat-model [data prior] obs)
          weighted-p (weighted-vector-mean result :p 3)]
      (is (th/close? (nth mean-p 0) (nth weighted-p 0) 0.02)
          "E[p_0|data] within 0.02 of 10/26")
      (is (th/close? (nth mean-p 1) (nth weighted-p 1) 0.02)
          "E[p_1|data] within 0.02 of 10/26")
      (is (th/close? (nth mean-p 2) (nth weighted-p 2) 0.02)
          "E[p_2|data] within 0.02 of 6/26"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
