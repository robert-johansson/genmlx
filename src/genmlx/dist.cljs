(ns genmlx.dist
  "MLX-native probability distributions for GenMLX.
   Every distribution is a record implementing three protocols:
   - IDistribution (sample, log-prob)
   - IDifferentiable (sample-reparam for gradient flow)
   - IGenerativeFunction/IGenerate (GFI participant)

   All log-probs return MLX scalars (stay on GPU for autograd).
   Reparameterized sampling enables gradient flow."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.handler :as h]))

;; ---------------------------------------------------------------------------
;; Distribution protocols
;; ---------------------------------------------------------------------------

(defprotocol IDistribution
  (sample   [d] [d key] "Sample from the distribution.")
  (log-prob [d value]    "Differentiable log-probability as MLX scalar."))

(defprotocol IDifferentiable
  (sample-reparam [d key] "Reparameterized sample for gradient flow."))

(defprotocol IEnumerable
  (support [d] "Return the support as a sequence of values."))

;; ---------------------------------------------------------------------------
;; Distribution -> GFI bridge
;; ---------------------------------------------------------------------------

(defn dist-simulate [dist]
  (let [v  (sample dist)
        lp (log-prob dist v)]
    (tr/make-trace {:gen-fn dist :args [] :choices (cm/->Value v)
                    :retval v :score lp})))

(defn dist-generate [dist constraints]
  (if (cm/has-value? constraints)
    (let [v  (cm/get-value constraints)
          lp (log-prob dist v)]
      {:trace (tr/make-trace {:gen-fn dist :args [] :choices (cm/->Value v)
                              :retval v :score lp})
       :weight lp})
    {:trace (dist-simulate dist) :weight (mx/scalar 0.0)}))

;; ---------------------------------------------------------------------------
;; Constants
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI (js/Math.log (* 2.0 js/Math.PI)))

;; ---------------------------------------------------------------------------
;; Gaussian
;; ---------------------------------------------------------------------------

(defrecord Gaussian [mu sigma]
  IDistribution
  (sample [_]
    (mx/add mu (mx/multiply sigma (mx/random-normal []))))
  (sample [_ key]
    (if key
      (mx/add mu (mx/multiply sigma (rng/normal key [])))
      (mx/add mu (mx/multiply sigma (mx/random-normal [])))))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))
          z (mx/divide (mx/subtract v mu) sigma)]
      (mx/negative
        (mx/add (mx/scalar (* 0.5 LOG-2PI))
                (mx/add (mx/log sigma)
                        (mx/multiply (mx/scalar 0.5) (mx/multiply z z)))))))

  IDifferentiable
  (sample-reparam [_ key]
    (mx/add mu (mx/multiply sigma (rng/normal key []))))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn gaussian
  "Create a Gaussian (normal) distribution with mean mu and std sigma."
  [mu sigma]
  (->Gaussian (if (mx/array? mu) mu (mx/scalar mu))
              (if (mx/array? sigma) sigma (mx/scalar sigma))))

(def normal gaussian)

;; ---------------------------------------------------------------------------
;; Uniform
;; ---------------------------------------------------------------------------

(defrecord Uniform [lo hi]
  IDistribution
  (sample [_]
    (mx/add lo (mx/multiply (mx/subtract hi lo) (mx/random-uniform []))))
  (sample [_ key]
    (if key
      (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key [])))
      (mx/add lo (mx/multiply (mx/subtract hi lo) (mx/random-uniform [])))))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))
          in-bounds (mx/multiply (mx/less-equal lo v) (mx/less-equal v hi))
          log-density (mx/negative (mx/log (mx/subtract hi lo)))]
      (mx/where in-bounds log-density (mx/scalar ##-Inf))))

  IDifferentiable
  (sample-reparam [_ key]
    (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key []))))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn uniform
  "Create a continuous uniform distribution on [lo, hi]."
  [lo hi]
  (->Uniform (if (mx/array? lo) lo (mx/scalar lo))
             (if (mx/array? hi) hi (mx/scalar hi))))

;; ---------------------------------------------------------------------------
;; Bernoulli
;; ---------------------------------------------------------------------------

(defrecord Bernoulli [p]
  IDistribution
  (sample [_]
    (let [u (mx/random-uniform [])]
      (mx/where (mx/less u p) (mx/scalar 1.0) (mx/scalar 0.0))))
  (sample [_ key]
    (if key
      (let [u (rng/uniform key [])]
        (mx/where (mx/less u p) (mx/scalar 1.0) (mx/scalar 0.0)))
      (sample _)))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))]
      ;; v*log(p) + (1-v)*log(1-p)
      (mx/add (mx/multiply v (mx/log p))
              (mx/multiply (mx/subtract (mx/scalar 1.0) v)
                           (mx/log (mx/subtract (mx/scalar 1.0) p))))))

  IEnumerable
  (support [_] [(mx/scalar 0.0) (mx/scalar 1.0)])

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn bernoulli
  "Create a Bernoulli distribution with probability p."
  [prob]
  (->Bernoulli (if (mx/array? prob) prob (mx/scalar prob))))

(defn flip
  "Alias for bernoulli."
  [prob]
  (bernoulli prob))

;; ---------------------------------------------------------------------------
;; Lanczos approximation for gamma function (needed by Beta, Gamma, etc.)
;; ---------------------------------------------------------------------------

(defn- js-gamma [x]
  (if (<= x 0)
    js/Infinity
    (let [g 7
          c [0.99999999999980993 676.5203681218851 -1259.1392167224028
             771.32342877765313 -176.61502916214059 12.507343278686905
             -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7]
          x (dec x)]
      (if (< x 0.5)
        (/ js/Math.PI (* (js/Math.sin (* js/Math.PI x))
                         (js-gamma (- 1 x))))
        (let [t (+ x g 0.5)
              rest-c (rest c)
              s (reduce (fn [a [i ci]]
                          (+ a (/ ci (+ x i 1))))
                        (first c)
                        (map-indexed vector rest-c))]
          (* (js/Math.sqrt (* 2 js/Math.PI))
             (js/Math.pow t (+ x 0.5))
             (js/Math.exp (- t))
             s))))))

(defn- log-gamma [x]
  (if (<= x 0)
    js/Infinity
    (let [g 7
          c [0.99999999999980993 676.5203681218851 -1259.1392167224028
             771.32342877765313 -176.61502916214059 12.507343278686905
             -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7]
          x (dec x)
          t (+ x g 0.5)
          rest-c (rest c)
          s (reduce (fn [a [i ci]]
                      (+ a (/ ci (+ x i 1))))
                    (first c)
                    (map-indexed vector rest-c))]
      (+ (* 0.5 (js/Math.log (* 2 js/Math.PI)))
         (* (+ x 0.5) (js/Math.log t))
         (- t)
         (js/Math.log s)))))

;; ---------------------------------------------------------------------------
;; Beta
;; ---------------------------------------------------------------------------

(defrecord Beta [alpha beta-param]
  IDistribution
  (sample [this] (sample this nil))
  (sample [_ key]
    ;; Beta via inverse CDF approximation using normal + sigmoid
    ;; For simple cases, use the JS Math.random fallback
    (let [a (do (mx/eval! alpha) (mx/item alpha))
          b (do (mx/eval! beta-param) (mx/item beta-param))]
      ;; Use Johnk's algorithm for beta sampling
      (loop []
        (let [u1 (js/Math.random)
              u2 (js/Math.random)
              x (js/Math.pow u1 (/ 1.0 a))
              y (js/Math.pow u2 (/ 1.0 b))]
          (if (<= (+ x y) 1.0)
            (mx/scalar (/ x (+ x y)))
            (recur))))))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))
          ;; log Beta(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)
          a-val (do (mx/eval! alpha) (mx/item alpha))
          b-val (do (mx/eval! beta-param) (mx/item beta-param))
          log-beta-val (mx/scalar (- (+ (log-gamma a-val)
                                        (log-gamma b-val))
                                     (log-gamma (+ a-val b-val))))]
      (mx/subtract
        (mx/add (mx/multiply (mx/subtract alpha (mx/scalar 1.0)) (mx/log v))
                (mx/multiply (mx/subtract beta-param (mx/scalar 1.0))
                             (mx/log (mx/subtract (mx/scalar 1.0) v))))
        log-beta-val)))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn beta-dist
  "Create a Beta distribution with parameters alpha and beta."
  [alpha beta-param]
  (->Beta (if (mx/array? alpha) alpha (mx/scalar alpha))
          (if (mx/array? beta-param) beta-param (mx/scalar beta-param))))

;; ---------------------------------------------------------------------------
;; Gamma
;; ---------------------------------------------------------------------------

(defrecord Gamma [shape-param rate]
  IDistribution
  (sample [this] (sample this nil))
  (sample [_ key]
    ;; Marsaglia and Tsang's method
    (let [a (do (mx/eval! shape-param) (mx/item shape-param))
          r (do (mx/eval! rate) (mx/item rate))
          d (- a (/ 1.0 3.0))
          c (/ 1.0 (js/Math.sqrt (* 9.0 d)))]
      (loop []
        (let [x (if key
                  (let [k (rng/normal key [])]
                    (mx/eval! k) (mx/item k))
                  (let [k (mx/random-normal [])]
                    (mx/eval! k) (mx/item k)))
              v (js/Math.pow (+ 1.0 (* c x)) 3)
              u (js/Math.random)]
          (if (and (> v 0)
                   (< (js/Math.log u) (+ (* 0.5 x x) (- d) (* d (- v 1 (js/Math.log v))))))
            (mx/scalar (/ (* d v) r))
            (recur))))))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))
          k shape-param
          ;; log p(x) = (k-1)*log(x) - rate*x + k*log(rate) - lgamma(k)
          log-gamma-k (mx/scalar (log-gamma (do (mx/eval! k) (mx/item k))))]
      (mx/subtract
        (mx/subtract
          (mx/add (mx/multiply (mx/subtract k (mx/scalar 1.0)) (mx/log v))
                  (mx/multiply k (mx/log rate)))
          (mx/multiply rate v))
        log-gamma-k)))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn gamma-dist
  "Create a Gamma distribution with shape and rate parameters."
  [shape-param rate]
  (->Gamma (if (mx/array? shape-param) shape-param (mx/scalar shape-param))
           (if (mx/array? rate) rate (mx/scalar rate))))

;; ---------------------------------------------------------------------------
;; Exponential
;; ---------------------------------------------------------------------------

(defrecord Exponential [rate]
  IDistribution
  (sample [_]
    (let [u (mx/random-uniform [])]
      (mx/divide (mx/negative (mx/log (mx/subtract (mx/scalar 1.0) u))) rate)))
  (sample [_ key]
    (if key
      (let [u (rng/uniform key [])]
        (mx/divide (mx/negative (mx/log (mx/subtract (mx/scalar 1.0) u))) rate))
      (sample _)))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))
          log-density (mx/subtract (mx/log rate) (mx/multiply rate v))
          non-neg (mx/greater-equal v (mx/scalar 0.0))]
      (mx/where non-neg log-density (mx/scalar ##-Inf))))

  IDifferentiable
  (sample-reparam [_ key]
    (let [u (rng/uniform key [])]
      (mx/divide (mx/negative (mx/log (mx/subtract (mx/scalar 1.0) u))) rate)))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn exponential
  "Create an Exponential distribution with the given rate."
  [rate]
  (->Exponential (if (mx/array? rate) rate (mx/scalar rate))))

;; ---------------------------------------------------------------------------
;; Categorical
;; ---------------------------------------------------------------------------

(defrecord Categorical [logits]
  IDistribution
  (sample [_]
    (mx/random-categorical logits))
  (sample [_ key]
    (if key
      (rng/categorical key logits)
      (mx/random-categorical logits)))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v mx/int32))
          ;; log-softmax: logits - logsumexp(logits)
          log-probs (mx/subtract logits (mx/logsumexp logits))]
      (mx/take-idx log-probs v)))

  IEnumerable
  (support [_]
    (let [n (do (mx/eval! logits) (first (mx/shape logits)))]
      (mapv #(mx/scalar (int %) mx/int32) (range n))))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn categorical
  "Create a Categorical distribution from log-probabilities (logits)."
  [logits]
  (->Categorical (if (mx/array? logits) logits (mx/array logits))))

;; ---------------------------------------------------------------------------
;; Poisson
;; ---------------------------------------------------------------------------

(defrecord Poisson [rate]
  IDistribution
  (sample [this] (sample this nil))
  (sample [_ key]
    ;; Knuth's algorithm
    (let [l (do (mx/eval! rate) (js/Math.exp (- (mx/item rate))))]
      (loop [k 0 p 1.0]
        (let [p (* p (js/Math.random))]
          (if (> p l)
            (recur (inc k) p)
            (mx/scalar k))))))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))
          ;; log p(k) = k*log(rate) - rate - lgamma(k+1)
          k-val (do (mx/eval! v) (mx/item v))
          log-gamma-k1 (mx/scalar (log-gamma (inc k-val)))]
      (mx/subtract
        (mx/subtract (mx/multiply v (mx/log rate)) rate)
        log-gamma-k1)))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn poisson
  "Create a Poisson distribution with the given rate."
  [rate]
  (->Poisson (if (mx/array? rate) rate (mx/scalar rate))))

;; ---------------------------------------------------------------------------
;; Laplace
;; ---------------------------------------------------------------------------

(defrecord Laplace [loc scale]
  IDistribution
  (sample [_]
    (let [u (mx/subtract (mx/random-uniform []) (mx/scalar 0.5))]
      (mx/subtract loc
        (mx/multiply scale
          (mx/multiply (mx/sign u)
            (mx/log (mx/subtract (mx/scalar 1.0)
                                  (mx/multiply (mx/scalar 2.0) (mx/abs u)))))))))
  (sample [_ key]
    (if key
      (let [u (mx/subtract (rng/uniform key []) (mx/scalar 0.5))]
        (mx/subtract loc
          (mx/multiply scale
            (mx/multiply (mx/sign u)
              (mx/log (mx/subtract (mx/scalar 1.0)
                                    (mx/multiply (mx/scalar 2.0) (mx/abs u))))))))
      (sample _)))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))]
      ;; -log(2*scale) - |v - loc| / scale
      (mx/subtract
        (mx/negative (mx/log (mx/multiply (mx/scalar 2.0) scale)))
        (mx/divide (mx/abs (mx/subtract v loc)) scale))))

  IDifferentiable
  (sample-reparam [_ key]
    (let [u (mx/subtract (rng/uniform key []) (mx/scalar 0.5))]
      (mx/subtract loc
        (mx/multiply scale
          (mx/multiply (mx/sign u)
            (mx/log (mx/subtract (mx/scalar 1.0)
                                  (mx/multiply (mx/scalar 2.0) (mx/abs u)))))))))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn laplace
  "Create a Laplace distribution with location and scale."
  [loc scale]
  (->Laplace (if (mx/array? loc) loc (mx/scalar loc))
             (if (mx/array? scale) scale (mx/scalar scale))))

;; ---------------------------------------------------------------------------
;; Student-t
;; ---------------------------------------------------------------------------

(defrecord StudentT [df loc scale]
  IDistribution
  (sample [this] (sample this nil))
  (sample [_ key]
    ;; Use ratio of normals and chi-squared
    (let [df-val (do (mx/eval! df) (mx/item df))
          ;; Sample chi-squared(df) = sum of df squared normals
          chi2 (loop [i 0 acc 0.0]
                 (if (>= i (int df-val))
                   acc
                   (let [z (let [n (mx/random-normal [])] (mx/eval! n) (mx/item n))]
                     (recur (inc i) (+ acc (* z z))))))
          z (let [n (mx/random-normal [])] (mx/eval! n) (mx/item n))
          t (* z (js/Math.sqrt (/ df-val chi2)))]
      (mx/add loc (mx/multiply scale (mx/scalar t)))))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))
          z (mx/divide (mx/subtract v loc) scale)
          df-val (do (mx/eval! df) (mx/item df))
          ;; log p(t) = lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi) - log(scale)
          ;;            - ((df+1)/2)*log(1 + t^2/df)
          half-df (/ df-val 2.0)
          half-df1 (/ (inc df-val) 2.0)
          log-norm (mx/scalar (- (log-gamma half-df1)
                                 (log-gamma half-df)
                                 (* 0.5 (js/Math.log (* df-val js/Math.PI)))))]
      (mx/subtract
        (mx/subtract log-norm (mx/log scale))
        (mx/multiply (mx/scalar half-df1)
                     (mx/log (mx/add (mx/scalar 1.0)
                                      (mx/divide (mx/multiply z z) df)))))))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn student-t
  "Create a Student-t distribution with df degrees of freedom, location and scale."
  [df loc scale]
  (->StudentT (if (mx/array? df) df (mx/scalar df))
              (if (mx/array? loc) loc (mx/scalar loc))
              (if (mx/array? scale) scale (mx/scalar scale))))

;; ---------------------------------------------------------------------------
;; Log-Normal
;; ---------------------------------------------------------------------------

(defrecord LogNormal [mu sigma]
  IDistribution
  (sample [_]
    (mx/exp (mx/add mu (mx/multiply sigma (mx/random-normal [])))))
  (sample [_ key]
    (if key
      (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key []))))
      (sample _)))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/scalar v))
          log-v (mx/log v)
          z (mx/divide (mx/subtract log-v mu) sigma)]
      ;; log p(x) = -(log(x) + 0.5*log(2pi) + log(sigma) + 0.5*z^2)
      (mx/negative
        (mx/add log-v
          (mx/add (mx/scalar (* 0.5 LOG-2PI))
                  (mx/add (mx/log sigma)
                          (mx/multiply (mx/scalar 0.5) (mx/multiply z z))))))))

  IDifferentiable
  (sample-reparam [_ key]
    (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key [])))))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn log-normal
  "Create a Log-Normal distribution with parameters mu and sigma."
  [mu sigma]
  (->LogNormal (if (mx/array? mu) mu (mx/scalar mu))
               (if (mx/array? sigma) sigma (mx/scalar sigma))))

;; ---------------------------------------------------------------------------
;; Multivariate Normal (via Cholesky)
;; ---------------------------------------------------------------------------

;; Pre-computed fields avoid per-call overhead:
;;   L-inv:       L^{-1} — matmul on GPU replaces solve-triangular on CPU
;;   k:           dimension (plain int, no mx/shape per call)
;;   norm-const:  -0.5 * (k*log(2pi) + log|Sigma|) — the entire constant term
;;   neg-half:    pre-allocated (mx/scalar -0.5)
(defrecord MultivariateNormal [mean-vec cov-matrix cholesky-L L-inv k norm-const neg-half]
  IDistribution
  (sample [this] (sample this nil))
  (sample [_ key]
    (let [z (if key (rng/normal key [k]) (mx/random-normal [k]))]
      ;; sample = mean + L @ z
      (mx/add mean-vec
              (mx/flatten (mx/matmul cholesky-L (mx/reshape z [k 1]))))))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/array v))
          diff (mx/subtract v mean-vec)
          ;; y = L^{-1} @ diff  (GPU matmul, no CPU sync)
          y (mx/flatten (mx/matmul L-inv (mx/reshape diff [k 1])))
          mahal (mx/sum (mx/multiply y y))]
      ;; -0.5 * mahal + norm-const
      (mx/add (mx/multiply neg-half mahal) norm-const)))

  IDifferentiable
  (sample-reparam [_ key]
    (let [z (rng/normal key [k])]
      (mx/add mean-vec
              (mx/flatten (mx/matmul cholesky-L (mx/reshape z [k 1]))))))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn multivariate-normal
  "Create a Multivariate Normal distribution.
   mean-vec: [k] array, cov-matrix: [k k] positive definite array.
   Cholesky decomposition and L-inverse are computed once at construction."
  [mean-vec cov-matrix]
  (let [mu (if (mx/array? mean-vec) mean-vec (mx/array mean-vec))
        cov (if (mx/array? cov-matrix) cov-matrix (mx/array cov-matrix))
        cov-2d (if (= 1 (mx/ndim cov))
                 (let [k (first (mx/shape mu))]
                   (mx/reshape cov [k k]))
                 cov)
        L (mx/cholesky cov-2d)
        _ (mx/eval! L)
        Li (mx/tri-inv L false)
        k (first (mx/shape mu))
        ;; norm-const = -0.5 * (k*log(2pi) + log|Sigma|)
        ;; log|Sigma| = 2 * sum(log(diag(L)))
        log-det-sigma (mx/multiply (mx/scalar 2.0)
                                   (mx/sum (mx/log (mx/diag L))))
        nc (mx/multiply (mx/scalar -0.5)
                        (mx/add (mx/scalar (* k LOG-2PI)) log-det-sigma))
        neg-half (mx/scalar -0.5)]
    (mx/eval! Li nc)
    (->MultivariateNormal mu cov-2d L Li k nc neg-half)))

;; ---------------------------------------------------------------------------
;; Dirichlet
;; ---------------------------------------------------------------------------

(defrecord Dirichlet [alpha]
  IDistribution
  (sample [this] (sample this nil))
  (sample [_ key]
    ;; Sample via gamma distribution: x_i ~ Gamma(alpha_i, 1), then normalize
    (let [alpha-vals (mx/->clj alpha)
          k (count alpha-vals)
          gammas (mapv (fn [a]
                         (let [g (sample (->Gamma (mx/scalar a) (mx/scalar 1.0)))]
                           (mx/eval! g) (mx/item g)))
                       alpha-vals)
          total (reduce + gammas)
          normalized (mapv #(/ % total) gammas)]
      (mx/array normalized)))
  (log-prob [_ v]
    (let [v (if (mx/array? v) v (mx/array v))
          alpha-vals (mx/->clj alpha)
          k (count alpha-vals)
          ;; log p(x) = sum((alpha_i - 1) * log(x_i)) - log B(alpha)
          ;; log B(alpha) = sum(lgamma(alpha_i)) - lgamma(sum(alpha_i))
          sum-alpha (reduce + alpha-vals)
          log-beta (- (reduce + (map log-gamma alpha-vals))
                      (log-gamma sum-alpha))
          log-terms (mx/sum
                      (mx/multiply (mx/subtract alpha (mx/scalar 1.0))
                                   (mx/log v)))]
      (mx/subtract log-terms (mx/scalar log-beta))))

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn dirichlet
  "Create a Dirichlet distribution with concentration parameters alpha."
  [alpha]
  (->Dirichlet (if (mx/array? alpha) alpha (mx/array alpha))))

;; ---------------------------------------------------------------------------
;; Delta (point mass)
;; ---------------------------------------------------------------------------

(defrecord Delta [v]
  IDistribution
  (sample [_] v)
  (sample [_ _key] v)
  (log-prob [_ value]
    (let [value (if (mx/array? value) value (mx/scalar value))
          eq (mx/equal v value)]
      (mx/where eq (mx/scalar 0.0) (mx/scalar ##-Inf))))

  IEnumerable
  (support [_] [v])

  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))

(defn delta
  "Create a Delta (point mass) distribution at value v."
  [v]
  (->Delta (if (mx/array? v) v (mx/scalar v))))

;; ---------------------------------------------------------------------------
;; Register bridge functions with handler
;; ---------------------------------------------------------------------------

(h/set-dist-fns!
  (fn [dist key] (sample dist key))
  (fn [dist value] (log-prob dist value)))
