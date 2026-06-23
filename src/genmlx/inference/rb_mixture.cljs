(ns genmlx.inference.rb-mixture
  "Rao-Blackwellized (collapsed) Gaussian-mixture marginal-likelihood scorer.

   Computes the EXACT log marginal likelihood log p(y) of a 1-D Gaussian
   mixture by analytically integrating out the continuous component means
   (Rao-Blackwellization) and enumerating only the discrete cluster
   assignments. For a known observation noise sigma and a conjugate
   Gaussian prior on each component mean, the per-component data likelihood
   collapses to a closed-form Gaussian-Gaussian marginal, so no Monte Carlo
   is needed for small N.

   Model
   -----
     data        y_1..y_N            (1-D scalars)
     components  K
     means       mu_k ~ N(m0, s0^2)  (conjugate Gaussian prior)
     noise       sigma               (FIXED / known positive scalar)
     assignments z_i in {1..K}       (prior: uniform 1/K per point, or a
                                       symmetric-Dirichlet-marginalized
                                       multinomial — see :weights)

   Exact marginal
   --------------
     p(y) = sum_{z in {1..K}^N} [ p(z) * prod_k m( {y_i : z_i = k} ) ]

   where m(S) is the Gaussian-Gaussian marginal likelihood of the points in
   set S sharing one latent mean mu ~ N(m0, s0^2) with known noise sigma.
   This Rao-Blackwellizes the continuous means (integrated out in closed
   form) and enumerates only the K^N discrete assignments — exact for small
   N (e.g. K=2,N=12 -> 4096 terms; K=3,N=9 -> 19683 terms). Everything is
   done in log-space with log-sum-exp for numerical stability.

   The collapsed estimator is DETERMINISTIC and EXACT: it has zero Monte
   Carlo variance, whereas a naive importance sampler over both assignments
   and means would be high-variance (the curse of integrating a discrete
   K^N sum AND continuous means by sampling). That variance reduction is the
   point of Rao-Blackwellization.

   K=1 special case: collapsed-gmm-log-evidence with :k 1 reduces exactly to
   single-gaussian-log-evidence (one shared mean over all points) — a
   non-circular cross-check of the enumeration against the closed form.

   References
   ----------
   - Marin, Mengersen & Robert (2005), \"Bayesian modelling and inference on
     mixtures of distributions\", Handbook of Statistics 25 — the collapsed
     mixture marginal likelihood.
   - Neal (2000), \"Markov chain sampling methods for Dirichlet process
     mixture models\", JCGS 9(2):249-265 — Algorithm 3 (collapsed Gibbs;
     the same per-point predictive marginal used here).
   - Rasmussen (2000) / Kamper's notes on the Normal-Gamma / Gaussian-Gaussian
     conjugate predictive — the closed-form per-component marginal.

   Pure + native-free: plain JS Math only, no MLX, no native bindings.")

;; ---------------------------------------------------------------------------
;; Numerics
;; ---------------------------------------------------------------------------

(defn- log-sum-exp
  "Numerically stable log(sum(exp(xs))) over a non-empty seq of doubles."
  [xs]
  (let [m (apply max xs)]
    (if (= m ##-Inf)
      ##-Inf
      (+ m (js/Math.log (reduce + (map #(js/Math.exp (- % m)) xs)))))))

(def ^:private LOG-2PI (js/Math.log (* 2.0 js/Math.PI)))

(defn- log-gamma
  "Lanczos approximation to log Gamma(x), x > 0. Used only by the optional
   symmetric-Dirichlet assignment prior (Dirichlet-multinomial marginal)."
  [x]
  (let [g 7
        c [0.99999999999980993 676.5203681218851 -1259.1392167224028
           771.32342877765313 -176.61502916214059 12.507343278686905
           -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7]]
    (if (< x 0.5)
      (- (js/Math.log (/ js/Math.PI (js/Math.sin (* js/Math.PI x))))
         (log-gamma (- 1.0 x)))
      (let [x  (- x 1.0)
            a  (reduce (fn [acc i] (+ acc (/ (nth c i) (+ x i))))
                       (first c) (range 1 (+ g 2)))
            t  (+ x g 0.5)]
        (+ (* 0.5 LOG-2PI)
           (* (+ x 0.5) (js/Math.log t))
           (- t)
           (js/Math.log a))))))

;; ---------------------------------------------------------------------------
;; Gaussian-Gaussian marginal (one shared mean, known noise)
;; ---------------------------------------------------------------------------

(defn- gg-log-marginal
  "Log marginal likelihood of a SET of points sharing one latent mean:
     mu ~ N(m0, s0^2),  y_i ~ N(mu, sigma^2)  for y_i in ys.

   Closed form (integrate mu out / complete the square):
     log p(ys) = -n/2 log(2 pi sigma^2)
                 + 1/2 log(prec_prior / prec_post)
                 - sum_i (y_i - m0)^2 / (2 sigma^2)
                 + b^2 / (2 prec_post)
   with prec_prior = 1/s0^2, prec_post = prec_prior + n/sigma^2,
        b = sum_i (y_i - m0)/sigma^2.

   The empty set contributes 0 (an unused component multiplies the joint by
   1 — its mean integrates to the prior normalizer)."
  [ys m0 s0 sigma]
  (let [n (count ys)]
    (if (zero? n)
      0.0
      (let [s0sq       (* s0 s0)
            snsq       (* sigma sigma)
            prec-prior (/ 1.0 s0sq)
            prec-post  (+ prec-prior (/ n snsq))
            ds         (map #(- % m0) ys)
            sumsq      (reduce + (map #(* % %) ds))
            sumd       (reduce + ds)
            b          (/ sumd snsq)]
        (+ (* -0.5 n (+ LOG-2PI (js/Math.log snsq)))
           (* 0.5 (js/Math.log (/ prec-prior prec-post)))
           (* -0.5 (/ sumsq snsq))
           (* 0.5 (/ (* b b) prec-post)))))))

;; ---------------------------------------------------------------------------
;; Discrete assignment enumeration
;; ---------------------------------------------------------------------------

(defn- assignments
  "All assignment vectors in {0..k-1}^n (the K^N discrete configurations)."
  [k n]
  (reduce (fn [acc _] (for [a acc, c (range k)] (conj a c)))
          [[]]
          (range n)))

(defn- log-prior-uniform
  "log p(z) for the per-point uniform prior: each point independently lands
   in one of k components with probability 1/k, so p(z) = (1/k)^n. This is a
   constant across the K^N terms for fixed k, but DOES vary with k (the
   -n*log(k) Occam term that penalizes adding components)."
  [_counts k n]
  (* (double n) (- (js/Math.log k))))

(defn- log-prior-dirichlet
  "log p(z) under a symmetric Dirichlet(alpha,...,alpha) prior on the mixing
   weights, marginalized out (Dirichlet-multinomial):
     p(z) = [ Gamma(k*alpha) / Gamma(k*alpha + n) ]
            * prod_k [ Gamma(alpha + count_k) / Gamma(alpha) ].
   Softer Occam slope than :uniform (unused components are integrated over
   rather than charged a fixed 1/k each)."
  [counts k n alpha]
  (- (+ (reduce + (map #(log-gamma (+ alpha %)) counts))
        (log-gamma (* k alpha)))
     (+ (* k (log-gamma alpha))
        (log-gamma (+ (* k alpha) n)))))

;; ---------------------------------------------------------------------------
;; Public API
;; ---------------------------------------------------------------------------

(defn single-gaussian-log-evidence
  "Exact log marginal likelihood of ys under a single-mean Gaussian model:
     mu ~ N(m0, s0^2),  y_i ~ N(mu, sigma^2).
   The K=1 special case of collapsed-gmm-log-evidence (and a non-circular
   closed-form cross-check of it)."
  [ys {:keys [m0 s0 sigma]}]
  (gg-log-marginal (vec ys) m0 s0 sigma))

(defn collapsed-gmm-log-evidence
  "Exact log marginal likelihood log p(ys) of a 1-D K-component Gaussian
   mixture with conjugate Gaussian priors on the means and known noise,
   computed by Rao-Blackwellizing the means and enumerating assignments.

   ys   1-D scalar data (seq of numbers)
   opts:
     :k        number of components (default 1)
     :m0       prior mean of each component mean       (default 0.0)
     :s0       prior std of each component mean         (default 1.0)
     :sigma    known observation noise std (> 0)        (default 1.0)
     :weights  assignment prior, :uniform (default; p(z)=(1/k)^n) or
               :dirichlet (symmetric-Dirichlet-marginalized multinomial)
     :alpha    symmetric Dirichlet concentration, :dirichlet only (default 1.0)

   Returns the exact log marginal likelihood as a double. Deterministic
   (zero Monte Carlo variance). Cost is K^N enumerated terms — intended for
   small N; throws if K^N exceeds :max-combinations (default 200000)."
  [ys {:keys [k m0 s0 sigma weights alpha max-combinations]
       :or   {k 1 m0 0.0 s0 1.0 sigma 1.0 weights :uniform alpha 1.0
              max-combinations 200000}}]
  (let [ysv   (vec ys)
        n     (count ysv)
        idx   (range n)
        n-cfg (js/Math.pow k n)]
    (when (> n-cfg max-combinations)
      (throw (ex-info (str "collapsed-gmm: K^N = " n-cfg " configurations exceeds "
                           "max-combinations " max-combinations
                           " (K=" k ", N=" n "). Reduce N or K.")
                      {:k k :n n :n-configurations n-cfg
                       :max-combinations max-combinations})))
    (if (zero? n)
      0.0
      (log-sum-exp
       (for [z (assignments k n)]
         (let [zv       (vec z)
               clusters (group-by zv idx)
               counts   (mapv #(count (get clusters % [])) (range k))
               lik      (reduce + (for [c (range k)]
                                    (gg-log-marginal (map ysv (get clusters c []))
                                                     m0 s0 sigma)))
               lprior   (case weights
                          :uniform   (log-prior-uniform counts k n)
                          :dirichlet (log-prior-dirichlet counts k n alpha)
                          (throw (ex-info (str "unknown :weights " weights)
                                          {:weights weights})))]
           (+ lprior lik)))))))
