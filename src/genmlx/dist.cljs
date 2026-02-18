(ns genmlx.dist
  "MLX-native probability distributions for GenMLX.
   All distributions use a single Distribution record with open multimethods.
   Extensible from any namespace via defdist or manual defmethod.

   All log-probs return MLX scalars (stay on GPU for autograd).
   Reparameterized sampling enables gradient flow."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc])
  (:require-macros [genmlx.dist.macros :refer [defdist]]))

;; ---------------------------------------------------------------------------
;; Constants
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI (js/Math.log (* 2.0 js/Math.PI)))

;; ---------------------------------------------------------------------------
;; Lanczos approximation for log-gamma (needed by Beta, Gamma, etc.)
;; ---------------------------------------------------------------------------

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
;; Laplace inverse CDF helper
;; ---------------------------------------------------------------------------

(defn- laplace-icdf
  "Inverse CDF for Laplace: loc - scale * sign(u) * log(1 - 2|u|)."
  [loc scale u]
  (->> (mx/abs u)
       (mx/multiply (mx/scalar 2.0))
       (mx/subtract (mx/scalar 1.0))
       mx/log
       (mx/multiply (mx/sign u))
       (mx/multiply scale)
       (mx/subtract loc)))

;; ---------------------------------------------------------------------------
;; Public API wrappers (backward compatible)
;; ---------------------------------------------------------------------------

(defn sample
  "Sample from a distribution."
  ([d] (dc/dist-sample d nil))
  ([d key] (dc/dist-sample d key)))

(defn log-prob
  "Compute differentiable log-probability."
  [d value]
  (dc/dist-log-prob d value))

(defn sample-reparam
  "Reparameterized sample for gradient flow."
  [d key]
  (dc/dist-reparam d key))

(defn support
  "Return the support as a sequence of values."
  [d]
  (dc/dist-support d))

;; ---------------------------------------------------------------------------
;; Gaussian
;; ---------------------------------------------------------------------------

(defdist gaussian
  "Gaussian (normal) distribution with mean mu and std sigma."
  [mu sigma]
  (sample [key]
    (mx/add mu (mx/multiply sigma (rng/normal key []))))
  (log-prob [v]
    (let [z (mx/divide (mx/subtract v mu) sigma)]
      (mx/negative
        (mx/add (mx/scalar (* 0.5 LOG-2PI))
                (mx/log sigma)
                (mx/multiply (mx/scalar 0.5) (mx/square z))))))
  (reparam [key]
    (mx/add mu (mx/multiply sigma (rng/normal key [])))))

(def normal gaussian)

;; ---------------------------------------------------------------------------
;; Uniform
;; ---------------------------------------------------------------------------

(defdist uniform
  "Continuous uniform distribution on [lo, hi]."
  [lo hi]
  (sample [key]
    (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key []))))
  (log-prob [v]
    (let [in-bounds (mx/multiply (mx/less-equal lo v) (mx/less-equal v hi))
          log-density (mx/negative (mx/log (mx/subtract hi lo)))]
      (mx/where in-bounds log-density (mx/scalar ##-Inf))))
  (reparam [key]
    (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key [])))))

;; ---------------------------------------------------------------------------
;; Bernoulli
;; ---------------------------------------------------------------------------

(defdist bernoulli
  "Bernoulli distribution with probability p."
  [p]
  (sample [key]
    (let [u (rng/uniform key [])]
      (mx/where (mx/less u p) (mx/scalar 1.0) (mx/scalar 0.0))))
  (log-prob [v]
    (mx/add (mx/multiply v (mx/log p))
            (mx/multiply (mx/subtract (mx/scalar 1.0) v)
                         (mx/log (mx/subtract (mx/scalar 1.0) p)))))
  (support []
    [(mx/scalar 0.0) (mx/scalar 1.0)]))

(defn flip
  "Alias for bernoulli."
  [prob]
  (bernoulli prob))

;; ---------------------------------------------------------------------------
;; Beta
;; ---------------------------------------------------------------------------

(defdist beta-dist
  "Beta distribution with parameters alpha and beta."
  [alpha beta-param]
  (sample [key]
    ;; Johnk's algorithm for beta sampling
    (let [a (mx/realize alpha)
          b (mx/realize beta-param)]
      (loop [k key]
        (let [[k1 k2] (rng/split k)
              u1 (mx/realize (rng/uniform k1 []))
              u2 (mx/realize (rng/uniform k2 []))
              x (js/Math.pow u1 (/ 1.0 a))
              y (js/Math.pow u2 (/ 1.0 b))]
          (if (<= (+ x y) 1.0)
            (mx/scalar (/ x (+ x y)))
            (let [[k' _] (rng/split k2)]
              (recur k')))))))
  (log-prob [v]
    (let [a-val (mx/realize alpha)
          b-val (mx/realize beta-param)
          log-beta-val (mx/scalar (- (+ (log-gamma a-val)
                                        (log-gamma b-val))
                                     (log-gamma (+ a-val b-val))))]
      (-> (mx/add (mx/multiply (mx/subtract alpha (mx/scalar 1.0)) (mx/log v))
                  (mx/multiply (mx/subtract beta-param (mx/scalar 1.0))
                               (mx/log (mx/subtract (mx/scalar 1.0) v))))
          (mx/subtract log-beta-val)))))

;; ---------------------------------------------------------------------------
;; Gamma
;; ---------------------------------------------------------------------------

(defdist gamma-dist
  "Gamma distribution with shape and rate parameters."
  [shape-param rate]
  (sample [key]
    ;; Marsaglia and Tsang's method
    (let [a (mx/realize shape-param)
          r (mx/realize rate)
          d (- a (/ 1.0 3.0))
          c (/ 1.0 (js/Math.sqrt (* 9.0 d)))]
      (loop [k key]
        (let [[k1 k2] (rng/split k)
              x (mx/realize (rng/normal k1 []))
              v (js/Math.pow (+ 1.0 (* c x)) 3)
              u (mx/realize (rng/uniform k2 []))]
          (if (and (> v 0)
                   (< (js/Math.log u) (+ (* 0.5 x x) (- d) (* d (- v 1 (js/Math.log v))))))
            (mx/scalar (/ (* d v) r))
            (let [[k' _] (rng/split k2)]
              (recur k')))))))
  (log-prob [v]
    (let [k shape-param
          log-gamma-k (mx/scalar (log-gamma (mx/realize k)))]
      (-> (mx/add (mx/multiply (mx/subtract k (mx/scalar 1.0)) (mx/log v))
                  (mx/multiply k (mx/log rate)))
          (mx/subtract (mx/multiply rate v))
          (mx/subtract log-gamma-k)))))

;; ---------------------------------------------------------------------------
;; Exponential
;; ---------------------------------------------------------------------------

(defdist exponential
  "Exponential distribution with the given rate."
  [rate]
  (sample [key]
    (let [u (rng/uniform key [])]
      (mx/divide (mx/negative (mx/log (mx/subtract (mx/scalar 1.0) u))) rate)))
  (log-prob [v]
    (let [log-density (mx/subtract (mx/log rate) (mx/multiply rate v))
          non-neg (mx/greater-equal v (mx/scalar 0.0))]
      (mx/where non-neg log-density (mx/scalar ##-Inf))))
  (reparam [key]
    (let [u (rng/uniform key [])]
      (mx/divide (mx/negative (mx/log (mx/subtract (mx/scalar 1.0) u))) rate))))

;; ---------------------------------------------------------------------------
;; Categorical
;; ---------------------------------------------------------------------------

(defdist categorical
  "Categorical distribution from log-probabilities (logits)."
  [logits]
  (sample [key]
    (rng/categorical key logits))
  (log-prob [v]
    (let [v (mx/ensure-array v mx/int32)
          log-probs (mx/subtract logits (mx/logsumexp logits))]
      (mx/take-idx log-probs v)))
  (support []
    (let [n (do (mx/eval! logits) (first (mx/shape logits)))]
      (mapv #(mx/scalar (int %) mx/int32) (range n)))))

;; ---------------------------------------------------------------------------
;; Poisson
;; ---------------------------------------------------------------------------

(defdist poisson
  "Poisson distribution with the given rate."
  [rate]
  (sample [key]
    ;; Knuth's algorithm
    (let [l (js/Math.exp (- (mx/realize rate)))]
      (loop [k 0 p 1.0 rk key]
        (let [[rk1 rk2] (rng/split rk)
              p (* p (mx/realize (rng/uniform rk1 [])))]
          (if (> p l)
            (recur (inc k) p rk2)
            (mx/scalar k))))))
  (log-prob [v]
    (let [k-val (mx/realize v)
          log-gamma-k1 (mx/scalar (log-gamma (inc k-val)))]
      (-> (mx/multiply v (mx/log rate))
          (mx/subtract rate)
          (mx/subtract log-gamma-k1)))))

;; ---------------------------------------------------------------------------
;; Laplace
;; ---------------------------------------------------------------------------

(defdist laplace
  "Laplace distribution with location and scale."
  [loc scale]
  (sample [key]
    (laplace-icdf loc scale (mx/subtract (rng/uniform key []) (mx/scalar 0.5))))
  (log-prob [v]
    (mx/subtract
      (mx/negative (mx/log (mx/multiply (mx/scalar 2.0) scale)))
      (mx/divide (mx/abs (mx/subtract v loc)) scale)))
  (reparam [key]
    (laplace-icdf loc scale (mx/subtract (rng/uniform key []) (mx/scalar 0.5)))))

;; ---------------------------------------------------------------------------
;; Student-t
;; ---------------------------------------------------------------------------

(defdist student-t
  "Student-t distribution with df degrees of freedom, location and scale."
  [df loc scale]
  (sample [key]
    (let [df-val (mx/realize df)
          n-keys (rng/split-n key (+ (int df-val) 1))
          chi2 (loop [i 0 acc 0.0]
                 (if (>= i (int df-val))
                   acc
                   (let [z (mx/realize (rng/normal (nth n-keys i) []))]
                     (recur (inc i) (+ acc (* z z))))))
          z (mx/realize (rng/normal (nth n-keys (int df-val)) []))
          t (* z (js/Math.sqrt (/ df-val chi2)))]
      (mx/add loc (mx/multiply scale (mx/scalar t)))))
  (log-prob [v]
    (let [z (mx/divide (mx/subtract v loc) scale)
          df-val (mx/realize df)
          half-df (/ df-val 2.0)
          half-df1 (/ (inc df-val) 2.0)
          log-norm (mx/scalar (- (log-gamma half-df1)
                                 (log-gamma half-df)
                                 (* 0.5 (js/Math.log (* df-val js/Math.PI)))))]
      (-> log-norm
          (mx/subtract (mx/log scale))
          (mx/subtract (mx/multiply (mx/scalar half-df1)
                                    (mx/log (mx/add (mx/scalar 1.0)
                                                    (mx/divide (mx/square z) df)))))))))

;; ---------------------------------------------------------------------------
;; Log-Normal
;; ---------------------------------------------------------------------------

(defdist log-normal
  "Log-Normal distribution with parameters mu and sigma."
  [mu sigma]
  (sample [key]
    (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key [])))))
  (log-prob [v]
    (let [log-v (mx/log v)
          z (mx/divide (mx/subtract log-v mu) sigma)]
      (mx/negative
        (mx/add log-v
                (mx/scalar (* 0.5 LOG-2PI))
                (mx/log sigma)
                (mx/multiply (mx/scalar 0.5) (mx/square z))))))
  (reparam [key]
    (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key []))))))

;; ---------------------------------------------------------------------------
;; Dirichlet
;; ---------------------------------------------------------------------------

(defdist dirichlet
  "Dirichlet distribution with concentration parameters alpha."
  [alpha]
  (sample [key]
    (let [alpha-vals (mx/->clj alpha)
          k (count alpha-vals)
          keys (rng/split-n key k)
          gammas (mapv (fn [a ki]
                         (let [g (dc/dist-sample (gamma-dist (mx/scalar a) (mx/scalar 1.0)) ki)]
                           (mx/realize g)))
                       alpha-vals keys)
          total (reduce + gammas)
          normalized (mapv #(/ % total) gammas)]
      (mx/array normalized)))
  (log-prob [v]
    (let [v (if (mx/array? v) v (mx/array v))
          alpha-vals (mx/->clj alpha)
          sum-alpha (reduce + alpha-vals)
          log-beta (- (reduce + (map log-gamma alpha-vals))
                      (log-gamma sum-alpha))
          log-terms (mx/sum
                      (mx/multiply (mx/subtract alpha (mx/scalar 1.0))
                                   (mx/log v)))]
      (mx/subtract log-terms (mx/scalar log-beta)))))

;; ---------------------------------------------------------------------------
;; Delta (point mass)
;; ---------------------------------------------------------------------------

(defdist delta
  "Delta (point mass) distribution at value v."
  [v]
  (sample [_key] v)
  (log-prob [value]
    (let [eq (mx/equal v value)]
      (mx/where eq (mx/scalar 0.0) (mx/scalar ##-Inf))))
  (support [] [v]))

;; ---------------------------------------------------------------------------
;; Multivariate Normal (via Cholesky) — manual definition
;; ---------------------------------------------------------------------------

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
        log-det-sigma (mx/multiply (mx/scalar 2.0)
                                   (mx/sum (mx/log (mx/diag L))))
        nc (mx/multiply (mx/scalar -0.5)
                        (mx/add (mx/scalar (* k LOG-2PI)) log-det-sigma))
        neg-half (mx/scalar -0.5)]
    (mx/eval! Li nc)
    (dc/->Distribution :multivariate-normal
                       {:mean-vec mu :cov-matrix cov-2d :cholesky-L L
                        :L-inv Li :k k :norm-const nc :neg-half neg-half})))

(defmethod dc/dist-sample :multivariate-normal [d key]
  (let [{:keys [mean-vec cholesky-L k]} (:params d)
        key (rng/ensure-key key)
        z (rng/normal key [k])]
    (mx/add mean-vec
            (mx/flatten (mx/matmul cholesky-L (mx/reshape z [k 1]))))))

(defmethod dc/dist-log-prob :multivariate-normal [d v]
  (let [{:keys [mean-vec L-inv k norm-const neg-half]} (:params d)
        v (if (mx/array? v) v (mx/array v))
        diff (mx/subtract v mean-vec)
        y (mx/flatten (mx/matmul L-inv (mx/reshape diff [k 1])))
        mahal (mx/sum (mx/square y))]
    (mx/add (mx/multiply neg-half mahal) norm-const)))

(defmethod dc/dist-reparam :multivariate-normal [d key]
  (let [{:keys [mean-vec cholesky-L k]} (:params d)
        key (rng/ensure-key key)
        z (rng/normal key [k])]
    (mx/add mean-vec
            (mx/flatten (mx/matmul cholesky-L (mx/reshape z [k 1]))))))
