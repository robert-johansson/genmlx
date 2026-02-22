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

;; Cached MLX scalar constants — avoid per-call allocation.
;; MLX arrays are immutable; mx/add etc. always create new arrays.
(def ^:private ZERO (mx/scalar 0.0))
(def ^:private ONE (mx/scalar 1.0))
(def ^:private TWO (mx/scalar 2.0))
(def ^:private THREE (mx/scalar 3.0))
(def ^:private HALF (mx/scalar 0.5))
(def ^:private NEG-INF (mx/scalar ##-Inf))
(def ^:private LOG-2PI-HALF (mx/scalar (* 0.5 LOG-2PI)))
(def ^:private LOG-PI (mx/scalar (js/Math.log js/Math.PI)))
(def ^:private MLX-PI (mx/scalar js/Math.PI))
(def ^:private SQRT-TWO (mx/scalar (js/Math.sqrt 2.0)))
(def ^:private TINY (mx/scalar 1e-30))
(def ^:private BERNOULLI-SUPPORT [ZERO ONE])

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

(defn- mlx-log-gamma
  "Element-wise log-gamma via Lanczos approximation using MLX operations.
   Works on arrays of any shape (scalar, [N], etc.) via broadcasting."
  [x]
  (let [x' (mx/subtract x ONE)
        t  (mx/add x' (mx/scalar 7.5))
        s  (reduce (fn [acc [i ci]]
                     (mx/add acc (mx/divide (mx/scalar ci)
                                            (mx/add x' (mx/scalar (double i))))))
                   (mx/scalar 0.99999999999980993)
                   [[1 676.5203681218851] [2 -1259.1392167224028]
                    [3 771.32342877765313] [4 -176.61502916214059]
                    [5 12.507343278686905] [6 -0.13857109526572012]
                    [7 9.9843695780195716e-6] [8 1.5056327351493116e-7]])]
    (mx/add LOG-2PI-HALF
            (mx/multiply (mx/add x' HALF) (mx/log t))
            (mx/negative t)
            (mx/log s))))

;; ---------------------------------------------------------------------------
;; Laplace inverse CDF helper
;; ---------------------------------------------------------------------------

(defn- laplace-icdf
  "Inverse CDF for Laplace: loc - scale * sign(u) * log(1 - 2|u|)."
  [loc scale u]
  (->> (mx/abs u)
       (mx/multiply TWO)
       (mx/subtract ONE)
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
;; Parameter validation helpers
;; ---------------------------------------------------------------------------

(defn- check-positive [dist-name param-name v]
  (when (and (number? v) (<= v 0))
    (throw (ex-info (str dist-name ": " param-name " must be positive, got " v)
                    {:distribution dist-name :parameter param-name :value v}))))

(defn- check-less-than [dist-name lo-name lo hi-name hi]
  (when (and (number? lo) (number? hi) (>= lo hi))
    (throw (ex-info (str dist-name ": " lo-name " must be less than " hi-name
                         ", got " lo-name "=" lo " " hi-name "=" hi)
                    {:distribution dist-name :lo lo :hi hi}))))

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
        (mx/add LOG-2PI-HALF
                (mx/log sigma)
                (mx/multiply HALF (mx/square z))))))
  (reparam [key]
    (mx/add mu (mx/multiply sigma (rng/normal key [])))))

(let [gaussian-raw gaussian]
  (defn gaussian
    "Gaussian (normal) distribution with mean mu and std sigma."
    [mu sigma]
    (check-positive "gaussian" "sigma" sigma)
    (gaussian-raw mu sigma)))

(defmethod dc/dist-sample-n :gaussian [d key n]
  (let [{:keys [mu sigma]} (:params d)
        key (rng/ensure-key key)]
    (mx/add mu (mx/multiply sigma (rng/normal key [n])))))

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
      (mx/where in-bounds log-density NEG-INF)))
  (reparam [key]
    (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key [])))))

(let [uniform-raw uniform]
  (defn uniform
    "Continuous uniform distribution on [lo, hi]."
    [lo hi]
    (check-less-than "uniform" "lo" lo "hi" hi)
    (uniform-raw lo hi)))

(defmethod dc/dist-sample-n :uniform [d key n]
  (let [{:keys [lo hi]} (:params d)
        key (rng/ensure-key key)]
    (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key [n])))))

;; ---------------------------------------------------------------------------
;; Bernoulli
;; ---------------------------------------------------------------------------

(defdist bernoulli
  "Bernoulli distribution with probability p."
  [p]
  (sample [key]
    (let [u (rng/uniform key [])]
      (mx/where (mx/less u p) ONE ZERO)))
  (log-prob [v]
    (mx/add (mx/multiply v (mx/log p))
            (mx/multiply (mx/subtract ONE v)
                         (mx/log (mx/subtract ONE p)))))
  (support [] BERNOULLI-SUPPORT))

(defmethod dc/dist-sample-n :bernoulli [d key n]
  (let [{:keys [p]} (:params d)
        key (rng/ensure-key key)
        u (rng/uniform key [n])]
    (mx/where (mx/less u p) ONE ZERO)))

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
      (-> (mx/add (mx/multiply (mx/subtract alpha ONE) (mx/log v))
                  (mx/multiply (mx/subtract beta-param ONE)
                               (mx/log (mx/subtract ONE v))))
          (mx/subtract log-beta-val)))))

(let [beta-dist-raw beta-dist]
  (defn beta-dist
    "Beta distribution with parameters alpha and beta."
    [alpha beta-param]
    (check-positive "beta-dist" "alpha" alpha)
    (check-positive "beta-dist" "beta" beta-param)
    (beta-dist-raw alpha beta-param)))

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
                   (< (js/Math.log u) (+ (* 0.5 x x) (* d (+ 1 (- v) (js/Math.log v))))))
            (mx/scalar (/ (* d v) r))
            (let [[k' _] (rng/split k2)]
              (recur k')))))))
  (log-prob [v]
    (let [k shape-param
          log-gamma-k (mx/scalar (log-gamma (mx/realize k)))]
      (-> (mx/add (mx/multiply (mx/subtract k ONE) (mx/log v))
                  (mx/multiply k (mx/log rate)))
          (mx/subtract (mx/multiply rate v))
          (mx/subtract log-gamma-k)))))

(let [gamma-dist-raw gamma-dist]
  (defn gamma-dist
    "Gamma distribution with shape and rate parameters."
    [shape-param rate]
    (check-positive "gamma-dist" "shape" shape-param)
    (check-positive "gamma-dist" "rate" rate)
    (gamma-dist-raw shape-param rate)))

(defn gamma-sample-n
  "Vectorized Marsaglia-Tsang: sample [n] gamma values with given shape and rate.
   shape-val: JS number, rate: MLX scalar, key: PRNG key, n: int.
   Exposed for reuse by beta, inv-gamma, and dirichlet batch sampling."
  [shape-val rate key n]
  (let [key (rng/ensure-key key)
        ;; For alpha < 1: Ahrens-Dieter boost — sample Gamma(alpha+1), scale by U^(1/alpha)
        alpha<1? (< shape-val 1.0)
        a (if alpha<1? (inc shape-val) shape-val)
        d (- a (/ 1.0 3.0))
        c (/ 1.0 (js/Math.sqrt (* 9.0 d)))
        d-arr (mx/scalar d)
        c-arr (mx/scalar c)
        max-iter 20]
    (loop [iter 0
           result (mx/zeros [n])
           done (mx/zeros [n])  ;; float 0.0/1.0 mask
           k key]
      (if (>= iter max-iter)
        ;; Scale by rate (and Ahrens-Dieter if alpha < 1)
        (let [samples (mx/divide result rate)]
          (if alpha<1?
            (let [[ku _] (rng/split k)
                  u (rng/uniform ku [n])]
              (mx/multiply samples (mx/power u (mx/scalar (/ 1.0 shape-val)))))
            samples))
        (let [[k1 k2 k3] (rng/split-n k 3)
              x (rng/normal k1 [n])
              u (rng/uniform k2 [n])
              ;; v = (1 + c*x)^3
              cx1 (mx/add ONE (mx/multiply c-arr x))
              v (mx/power cx1 (mx/scalar 3.0))
              ;; Accept where: v > 0 AND log(u) < 0.5*x^2 + d*(1 - v + log(v))
              v-pos (mx/greater v ZERO)
              safe-v (mx/maximum v (mx/scalar 1e-30))
              log-accept (mx/add (mx/multiply HALF (mx/square x))
                                 (mx/multiply d-arr
                                              (mx/add (mx/subtract ONE safe-v)
                                                      (mx/log safe-v))))
              accepted (mx/multiply v-pos (mx/less (mx/log u) log-accept))
              ;; Only fill not-yet-done slots
              not-done (mx/equal done ZERO)
              newly-done (mx/multiply accepted not-done)
              new-vals (mx/multiply d-arr safe-v)
              result (mx/where newly-done new-vals result)
              done (mx/where newly-done ONE done)]
          (recur (inc iter) result done k3))))))

(defmethod dc/dist-sample-n :gamma [d key n]
  (let [{:keys [shape-param rate]} (:params d)
        shape-val (mx/realize shape-param)]
    (gamma-sample-n shape-val rate key n)))

;; Beta batch sampling via two independent gamma samples
(defmethod dc/dist-sample-n :beta-dist [d key n]
  (let [{:keys [alpha beta-param]} (:params d)
        key (rng/ensure-key key)
        [k1 k2] (rng/split key)
        one ONE
        g1 (gamma-sample-n (mx/realize alpha) one k1 n)
        g2 (gamma-sample-n (mx/realize beta-param) one k2 n)]
    (mx/divide g1 (mx/add g1 g2))))

;; Dirichlet batch sampling via k independent gamma samples, then normalize
(defmethod dc/dist-sample-n :dirichlet [d key n]
  (let [{:keys [alpha]} (:params d)
        key (rng/ensure-key key)
        alpha-vals (mx/->clj alpha)
        k (count alpha-vals)
        keys (rng/split-n key k)
        one ONE
        ;; Sample k gamma arrays, each [n], then stack to [k n]
        gammas (mx/stack (mapv (fn [a ki] (gamma-sample-n a one ki n))
                               alpha-vals keys))
        ;; gammas is [k n], sum along axis 0 -> [n], then transpose and divide
        totals (mx/sum gammas [0])]
    ;; Result shape [n k]: transpose [k n] -> [n k]
    (mx/transpose (mx/divide gammas totals))))

;; ---------------------------------------------------------------------------
;; Exponential
;; ---------------------------------------------------------------------------

(defdist exponential
  "Exponential distribution with the given rate."
  [rate]
  (sample [key]
    (let [u (rng/uniform key [])]
      (mx/divide (mx/negative (mx/log (mx/subtract ONE u))) rate)))
  (log-prob [v]
    (let [log-density (mx/subtract (mx/log rate) (mx/multiply rate v))
          non-neg (mx/greater-equal v ZERO)]
      (mx/where non-neg log-density NEG-INF)))
  (reparam [key]
    (let [u (rng/uniform key [])]
      (mx/divide (mx/negative (mx/log (mx/subtract ONE u))) rate))))

(let [exponential-raw exponential]
  (defn exponential
    "Exponential distribution with the given rate."
    [rate]
    (check-positive "exponential" "rate" rate)
    (exponential-raw rate)))

(defmethod dc/dist-sample-n :exponential [d key n]
  (let [{:keys [rate]} (:params d)
        key (rng/ensure-key key)
        u (rng/uniform key [n])]
    (mx/divide (mx/negative (mx/log (mx/subtract ONE u))) rate)))

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

(defmethod dc/dist-sample-n :categorical [d key n]
  (let [{:keys [logits]} (:params d)
        keys (rng/split-n (rng/ensure-key key) n)]
    (mx/stack (mapv #(rng/categorical % logits) keys))))

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
    (-> (mx/multiply v (mx/log rate))
        (mx/subtract rate)
        (mx/subtract (mlx-log-gamma (mx/add v ONE))))))

;; ---------------------------------------------------------------------------
;; Laplace
;; ---------------------------------------------------------------------------

(defdist laplace
  "Laplace distribution with location and scale."
  [loc scale]
  (sample [key]
    (laplace-icdf loc scale (mx/subtract (rng/uniform key []) HALF)))
  (log-prob [v]
    (mx/subtract
      (mx/negative (mx/log (mx/multiply TWO scale)))
      (mx/divide (mx/abs (mx/subtract v loc)) scale)))
  (reparam [key]
    (laplace-icdf loc scale (mx/subtract (rng/uniform key []) HALF))))

(defmethod dc/dist-sample-n :laplace [d key n]
  (let [{:keys [loc scale]} (:params d)
        key (rng/ensure-key key)]
    (laplace-icdf loc scale (mx/subtract (rng/uniform key [n]) HALF))))

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
                                    (mx/log (mx/add ONE
                                                    (mx/divide (mx/square z) df)))))))))

(defmethod dc/dist-sample-n :student-t [d key n]
  (let [{:keys [df loc scale]} (:params d)
        df-val (int (mx/realize df))
        [k1 k2] (rng/split (rng/ensure-key key))
        ;; Chi-squared: sum of df squared normals -> [n]
        normals (rng/normal k1 [n df-val])
        chi2 (mx/sum (mx/square normals) [1])
        ;; Standard normal -> [n]
        z (rng/normal k2 [n])
        ;; t = z * sqrt(df / chi2)
        t (mx/multiply z (mx/sqrt (mx/divide df chi2)))]
    (mx/add loc (mx/multiply scale t))))

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
                LOG-2PI-HALF
                (mx/log sigma)
                (mx/multiply HALF (mx/square z))))))
  (reparam [key]
    (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key []))))))

(defmethod dc/dist-sample-n :log-normal [d key n]
  (let [{:keys [mu sigma]} (:params d)
        key (rng/ensure-key key)]
    (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key [n]))))))

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
                         (let [g (dc/dist-sample (gamma-dist (mx/scalar a) ONE) ki)]
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
                      (mx/multiply (mx/subtract alpha ONE)
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
      (mx/where eq ZERO NEG-INF)))
  (support [] [v]))

(defmethod dc/dist-sample-n :delta [d _key n]
  (let [{:keys [v]} (:params d)]
    (mx/broadcast-to v [n])))

;; ---------------------------------------------------------------------------
;; Cauchy
;; ---------------------------------------------------------------------------

(defdist cauchy
  "Cauchy distribution with location and scale."
  [loc scale]
  (sample [key]
    ;; Inverse CDF: loc + scale * tan(pi * (u - 0.5))
    (let [u (rng/uniform key [])
          z (mx/subtract u HALF)]
      (mx/add loc (mx/multiply scale
                    (mx/divide (mx/sin (mx/multiply (mx/scalar js/Math.PI) z))
                               (mx/cos (mx/multiply (mx/scalar js/Math.PI) z)))))))
  (log-prob [v]
    ;; -log(pi * scale * (1 + ((v - loc) / scale)^2))
    (let [z (mx/divide (mx/subtract v loc) scale)]
      (mx/negative
        (mx/add (mx/scalar (js/Math.log js/Math.PI))
                (mx/log scale)
                (mx/log (mx/add ONE (mx/square z)))))))
  (reparam [key]
    (let [u (rng/uniform key [])
          z (mx/subtract u HALF)]
      (mx/add loc (mx/multiply scale
                    (mx/divide (mx/sin (mx/multiply (mx/scalar js/Math.PI) z))
                               (mx/cos (mx/multiply (mx/scalar js/Math.PI) z))))))))

(defmethod dc/dist-sample-n :cauchy [d key n]
  (let [{:keys [loc scale]} (:params d)
        key (rng/ensure-key key)
        u (rng/uniform key [n])
        z (mx/subtract u HALF)]
    (mx/add loc (mx/multiply scale
                  (mx/divide (mx/sin (mx/multiply (mx/scalar js/Math.PI) z))
                             (mx/cos (mx/multiply (mx/scalar js/Math.PI) z)))))))

;; ---------------------------------------------------------------------------
;; Inverse Gamma
;; ---------------------------------------------------------------------------

(defdist inv-gamma
  "Inverse-Gamma distribution with shape and scale parameters."
  [shape-param scale-param]
  (sample [key]
    ;; Sample gamma(shape, 1/scale), then invert
    (let [g (dc/dist-sample (gamma-dist shape-param ONE) key)]
      (mx/divide scale-param g)))
  (log-prob [v]
    ;; log p(v) = shape*log(scale) - log-gamma(shape) - (shape+1)*log(v) - scale/v
    (let [a (mx/realize shape-param)
          log-gamma-a (mx/scalar (log-gamma a))]
      (-> (mx/multiply shape-param (mx/log scale-param))
          (mx/subtract log-gamma-a)
          (mx/subtract (mx/multiply (mx/add shape-param ONE) (mx/log v)))
          (mx/subtract (mx/divide scale-param v))))))

(defmethod dc/dist-sample-n :inv-gamma [d key n]
  (let [{:keys [shape-param scale-param]} (:params d)
        g (gamma-sample-n (mx/realize shape-param) ONE key n)]
    (mx/divide scale-param g)))

;; ---------------------------------------------------------------------------
;; Geometric
;; ---------------------------------------------------------------------------

(defdist geometric
  "Geometric distribution: number of failures before first success, p in (0,1)."
  [p]
  (sample [key]
    ;; Inverse CDF: floor(log(u) / log(1-p))
    (let [u (rng/uniform key [])
          log-u (mx/log u)
          log-1mp (mx/log (mx/subtract ONE p))]
      (mx/floor (mx/divide log-u log-1mp))))
  (log-prob [v]
    ;; log p(k) = k * log(1-p) + log(p)
    (mx/add (mx/multiply v (mx/log (mx/subtract ONE p)))
            (mx/log p)))
  (support []
    ;; Infinite support; return first few for enumeration
    (mapv #(mx/scalar % mx/int32) (range 100))))

(defmethod dc/dist-sample-n :geometric [d key n]
  (let [{:keys [p]} (:params d)
        key (rng/ensure-key key)
        u (rng/uniform key [n])
        log-u (mx/log u)
        log-1mp (mx/log (mx/subtract ONE p))]
    (mx/floor (mx/divide log-u log-1mp))))

;; ---------------------------------------------------------------------------
;; Negative Binomial
;; ---------------------------------------------------------------------------

(defdist neg-binomial
  "Negative binomial (Polya) distribution.
   r: number of successes, p: probability of success."
  [r p]
  (sample [key]
    ;; Gamma-Poisson mixture: lambda ~ Gamma(r, p/(1-p)), then x ~ Poisson(lambda)
    (let [rate (mx/divide p (mx/subtract ONE p))
          g (dc/dist-sample (gamma-dist r rate) key)
          g-val (mx/realize g)
          l (js/Math.exp (- g-val))]
      (loop [k 0 pr 1.0 rk key]
        (let [[rk1 rk2] (rng/split rk)
              pr (* pr (mx/realize (rng/uniform rk1 [])))]
          (if (> pr l)
            (recur (inc k) pr rk2)
            (mx/scalar k))))))
  (log-prob [v]
    ;; log C(v + r - 1, v) + r*log(p) + v*log(1-p)
    (let [log-coeff (mx/subtract (mlx-log-gamma (mx/add v r))
                                 (mx/add (mlx-log-gamma (mx/add v ONE))
                                         (mlx-log-gamma r)))]
      (-> log-coeff
          (mx/add (mx/multiply r (mx/log p)))
          (mx/add (mx/multiply v (mx/log (mx/subtract ONE p))))))))

;; ---------------------------------------------------------------------------
;; Binomial
;; ---------------------------------------------------------------------------

(defdist binomial
  "Binomial distribution: n trials with success probability p."
  [n-trials p]
  (sample [key]
    (let [nt (int (mx/realize n-trials))
          keys (rng/split-n key nt)
          successes (reduce (fn [acc ki]
                              (let [u (mx/realize (rng/uniform ki []))]
                                (if (< u (mx/realize p)) (inc acc) acc)))
                            0 keys)]
      (mx/scalar successes)))
  (log-prob [v]
    ;; log C(n, k) + k*log(p) + (n-k)*log(1-p)
    (let [log-coeff (mx/subtract (mlx-log-gamma (mx/add n-trials ONE))
                                 (mx/add (mlx-log-gamma (mx/add v ONE))
                                         (mlx-log-gamma (mx/add (mx/subtract n-trials v)
                                                                ONE))))]
      (-> log-coeff
          (mx/add (mx/multiply v (mx/log p)))
          (mx/add (mx/multiply (mx/subtract n-trials v)
                               (mx/log (mx/subtract ONE p)))))))
  (support []
    (let [nt (int (mx/realize n-trials))]
      (mapv #(mx/scalar % mx/int32) (range (inc nt))))))

(defmethod dc/dist-sample-n :binomial [d key n]
  (let [{:keys [n-trials p]} (:params d)
        nt (int (mx/realize n-trials))
        key (rng/ensure-key key)
        ;; [n, nt] uniform draws, compare with p, sum successes
        u (rng/uniform key [n nt])
        successes (mx/sum (mx/where (mx/less u p) ONE ZERO) [1])]
    successes))

;; ---------------------------------------------------------------------------
;; Discrete Uniform
;; ---------------------------------------------------------------------------

(defdist discrete-uniform
  "Discrete uniform distribution on integers [lo, hi]."
  [lo hi]
  (sample [key]
    (let [lo-val (int (mx/realize lo))
          hi-val (int (mx/realize hi))
          n (inc (- hi-val lo-val))]
      (mx/scalar (+ lo-val (int (* (mx/realize (rng/uniform key [])) n))) mx/int32)))
  (log-prob [v]
    (let [lo-val (mx/realize lo)
          hi-val (mx/realize hi)
          n (inc (- hi-val lo-val))
          in-range (mx/multiply (mx/greater-equal v lo) (mx/less-equal v hi))]
      (mx/where in-range (mx/scalar (- (js/Math.log n))) NEG-INF)))
  (support []
    (let [lo-val (int (mx/realize lo))
          hi-val (int (mx/realize hi))]
      (mapv #(mx/scalar % mx/int32) (range lo-val (inc hi-val))))))

(defmethod dc/dist-sample-n :discrete-uniform [d key n]
  (let [{:keys [lo hi]} (:params d)
        key (rng/ensure-key key)]
    (rng/randint key (int (mx/realize lo)) (inc (int (mx/realize hi))) [n])))

;; ---------------------------------------------------------------------------
;; Truncated Normal
;; ---------------------------------------------------------------------------

(defdist truncated-normal
  "Truncated normal distribution on [lo, hi] with parameters mu and sigma."
  [mu sigma lo hi]
  (sample [key]
    (let [z (rng/truncated-normal key
              (mx/divide (mx/subtract lo mu) sigma)
              (mx/divide (mx/subtract hi mu) sigma)
              [])]
      (mx/add mu (mx/multiply sigma z))))
  (log-prob [v]
    ;; log p(v) = log N(v; mu, sigma) - log(Phi(b) - Phi(a))
    ;; where a = (lo - mu)/sigma, b = (hi - mu)/sigma
    (let [z (mx/divide (mx/subtract v mu) sigma)
          ;; Standard normal log-pdf
          log-phi (mx/negative
                    (mx/add LOG-2PI-HALF
                            (mx/multiply HALF (mx/square z))))
          ;; Subtract log(sigma)
          log-pdf (mx/subtract log-phi (mx/log sigma))
          ;; Normalization: approximate Phi via erf
          a (mx/divide (mx/subtract lo mu) sigma)
          b (mx/divide (mx/subtract hi mu) sigma)
          phi-a (mx/multiply HALF
                  (mx/add ONE
                          (mx/erf (mx/divide a SQRT-TWO))))
          phi-b (mx/multiply HALF
                  (mx/add ONE
                          (mx/erf (mx/divide b SQRT-TWO))))
          log-norm (mx/log (mx/subtract phi-b phi-a))
          ;; Bounds check
          in-bounds (mx/multiply (mx/greater-equal v lo) (mx/less-equal v hi))]
      (mx/where in-bounds (mx/subtract log-pdf log-norm) NEG-INF)))
  (reparam [key]
    (let [z (rng/truncated-normal key
              (mx/divide (mx/subtract lo mu) sigma)
              (mx/divide (mx/subtract hi mu) sigma)
              [])]
      (mx/add mu (mx/multiply sigma z)))))

(defmethod dc/dist-sample-n :truncated-normal [d key n]
  (let [{:keys [mu sigma lo hi]} (:params d)
        key (rng/ensure-key key)
        a (mx/divide (mx/subtract lo mu) sigma)
        b (mx/divide (mx/subtract hi mu) sigma)
        z (rng/truncated-normal key a b [n])]
    (mx/add mu (mx/multiply sigma z))))

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
        log-det-sigma (mx/multiply TWO
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

(defmethod dc/dist-sample-n :multivariate-normal [d key n]
  (let [{:keys [mean-vec cholesky-L k]} (:params d)
        key (rng/ensure-key key)
        z (rng/normal key [n k])
        ;; z: [n, k], L^T: [k, k] -> samples: [n, k]
        samples (mx/matmul z (mx/transpose cholesky-L))]
    (mx/add mean-vec samples)))

;; ---------------------------------------------------------------------------
;; Broadcasted Normal (independent element-wise Gaussians)
;; ---------------------------------------------------------------------------

(defdist broadcasted-normal
  "Independent element-wise normal distribution.
   mu and sigma are MLX arrays of any shape. Samples N(mu_i, sigma_i) independently."
  [mu sigma]
  (sample [key]
    (let [sh (mx/shape mu)]
      (mx/add mu (mx/multiply sigma (rng/normal key sh)))))
  (log-prob [v]
    (let [z (mx/divide (mx/subtract v mu) sigma)]
      (mx/sum
        (mx/negative
          (mx/add LOG-2PI-HALF
                  (mx/log sigma)
                  (mx/multiply HALF (mx/square z)))))))
  (reparam [key]
    (let [sh (mx/shape mu)]
      (mx/add mu (mx/multiply sigma (rng/normal key sh))))))

(defmethod dc/dist-sample-n :broadcasted-normal [d key n]
  (let [{:keys [mu sigma]} (:params d)
        key (rng/ensure-key key)
        sh (mx/shape mu)]
    (mx/add mu (mx/multiply sigma (rng/normal key (into [n] sh))))))

;; ---------------------------------------------------------------------------
;; Beta-Uniform Mixture — convenience wrapper
;; ---------------------------------------------------------------------------

(defn beta-uniform-mixture
  "Mixture of Beta(alpha, beta-param) with probability theta and
   Uniform(0,1) with probability (1 - theta). Common prior for bounded params."
  [theta alpha beta-param]
  (dc/mixture [(beta-dist (mx/ensure-array alpha) (mx/ensure-array beta-param))
               (uniform ZERO ONE)]
              (mx/array [(js/Math.log theta)
                         (js/Math.log (- 1.0 theta))])))

;; ---------------------------------------------------------------------------
;; Piecewise Uniform
;; ---------------------------------------------------------------------------

(defdist piecewise-uniform
  "Piecewise uniform distribution over bins defined by sorted boundary points.
   bounds: MLX array of N+1 boundary points (sorted).
   probs:  MLX array of N unnormalized bin probabilities."
  [bounds probs]
  (sample [key]
    (let [[k1 k2] (rng/split key)
          ;; Choose bin via categorical on log-probs
          log-probs (mx/log probs)
          bin-idx (mx/item (rng/categorical k1 log-probs))
          ;; Uniform within chosen bin
          lo (mx/index bounds (int bin-idx))
          hi (mx/index bounds (inc (int bin-idx)))
          u (rng/uniform k2 [])]
      (mx/add lo (mx/multiply u (mx/subtract hi lo)))))
  (log-prob [v]
    ;; Vectorized bin assignment using mx/where — works for scalar and [N]-shaped v
    (let [bounds-vals (mx/->clj bounds)
          probs-vals (mx/->clj probs)
          total (reduce + probs-vals)
          n-bins (count probs-vals)]
      (reduce
        (fn [acc i]
          (let [lo (nth bounds-vals i)
                hi (nth bounds-vals (inc i))
                width (- hi lo)
                p (nth probs-vals i)
                log-density (mx/scalar (- (js/Math.log p) (js/Math.log total) (js/Math.log width)))
                in-bin (mx/multiply (mx/greater-equal v (mx/scalar lo))
                                    (mx/less v (mx/scalar hi)))]
            (mx/where in-bin log-density acc)))
        NEG-INF
        (range n-bins)))))

;; ---------------------------------------------------------------------------
;; Wishart
;; ---------------------------------------------------------------------------

(defn- log-multivariate-gamma
  "Log of the multivariate gamma function Gamma_k(a)."
  [a k]
  (+ (* k (dec k) 0.25 (js/Math.log js/Math.PI))
     (reduce + (map (fn [i] (log-gamma (- a (* 0.5 i)))) (range k)))))

(defn wishart
  "Wishart distribution with df degrees of freedom and [k x k] scale matrix V.
   Uses Bartlett decomposition for sampling."
  [df scale-matrix]
  (let [V (if (mx/array? scale-matrix) scale-matrix (mx/array scale-matrix))
        df-val (if (mx/array? df) (mx/realize df) df)
        V-2d (if (= 1 (mx/ndim V))
                (let [k (int (js/Math.sqrt (first (mx/shape V))))]
                  (mx/reshape V [k k]))
                V)
        L (mx/cholesky V-2d)
        _ (mx/eval! L)
        k (first (mx/shape V-2d))
        V-inv (mx/inv V-2d)
        log-det-V (mx/multiply TWO (mx/sum (mx/log (mx/diag L))))
        _ (mx/eval! V-inv log-det-V)]
    (dc/->Distribution :wishart
                        {:df df-val :scale-matrix V-2d :cholesky-L L
                         :V-inv V-inv :log-det-V log-det-V :k k})))

(defmethod dc/dist-sample :wishart [d key]
  (let [{:keys [df cholesky-L k]} (:params d)
        key (rng/ensure-key key)
        keys (rng/split-n key (+ k (* k (dec k) (/ 2))))
        ;; Build lower-triangular A (Bartlett decomposition)
        ;; Diagonal: A_ii ~ sqrt(chi²(df - i + 1)), chi²(n) = Gamma(n/2, 1/2)
        ;; Off-diagonal: A_ij ~ N(0,1)
        ki (atom 0)
        next-key! (fn [] (let [i @ki] (swap! ki inc) (nth keys i)))
        A-data (for [i (range k)]
                 (for [j (range k)]
                   (cond
                     (= i j) ;; diagonal: sqrt(chi²(df - i))
                     (let [chi2-df (- df i)
                           ;; chi²(n) = Gamma(n/2, 2) but we sample Gamma(n/2, 1) * 2
                           g (dc/dist-sample (gamma-dist (mx/scalar (/ chi2-df 2.0))
                                                          ONE)
                                             (next-key!))]
                       (mx/sqrt (mx/multiply TWO g)))
                     (> i j) ;; below diagonal: N(0,1)
                     (rng/normal (next-key!) [])
                     :else ;; above diagonal: 0
                     ZERO)))
        ;; Build A matrix
        A (mx/reshape (mx/stack (mapv (fn [row] (mx/stack (vec row))) A-data)) [k k])
        ;; W = L * A * A^T * L^T
        LA (mx/matmul cholesky-L A)
        W (mx/matmul LA (mx/transpose LA))]
    (do (mx/eval! W) W)))

(defmethod dc/dist-log-prob :wishart [d x]
  (let [{:keys [df V-inv log-det-V k]} (:params d)
        x (if (mx/array? x) x (mx/array x))
        x-2d (if (= 1 (mx/ndim x)) (mx/reshape x [k k]) x)
        log-det-X (mx/multiply TWO
                               (mx/sum (mx/log (mx/diag (mx/cholesky x-2d)))))
        _ (mx/eval! log-det-X)
        ;; log p(X) = ((df-k-1)/2)*log|X| - (1/2)*tr(V^{-1}X) - (df*k/2)*log(2)
        ;;            - (df/2)*log|V| - log_multivariate_gamma(df/2, k)
        half-df (/ df 2.0)
        term1 (mx/multiply (mx/scalar (/ (- df k 1) 2.0)) log-det-X)
        tr-VinvX (mx/sum (mx/multiply V-inv (mx/transpose x-2d))) ;; tr(A*B) = sum(A .* B^T)
        term2 (mx/multiply (mx/scalar -0.5) tr-VinvX)
        term3 (mx/scalar (- (* half-df k (js/Math.log 2.0))))
        term4 (mx/scalar (- (* half-df (mx/realize log-det-V))))
        term5 (mx/scalar (- (log-multivariate-gamma half-df k)))]
    (mx/add term1 term2 term3 term4 term5)))

;; ---------------------------------------------------------------------------
;; Inverse Wishart
;; ---------------------------------------------------------------------------

(defn inv-wishart
  "Inverse Wishart distribution with df degrees of freedom and [k x k] scale matrix Psi.
   Sample: W ~ Wishart(df, Psi^{-1}), return W^{-1}."
  [df scale-matrix]
  (let [Psi (if (mx/array? scale-matrix) scale-matrix (mx/array scale-matrix))
        df-val (if (mx/array? df) (mx/realize df) df)
        Psi-2d (if (= 1 (mx/ndim Psi))
                  (let [k (int (js/Math.sqrt (first (mx/shape Psi))))]
                    (mx/reshape Psi [k k]))
                  Psi)
        k (first (mx/shape Psi-2d))
        Psi-inv (mx/inv Psi-2d)
        _ (mx/eval! Psi-inv)
        ;; Build internal Wishart(df, Psi^{-1}) for sampling
        wish (wishart df-val Psi-inv)
        ;; Precompute log|Psi| for log-prob
        L-psi (mx/cholesky Psi-2d)
        _ (mx/eval! L-psi)
        log-det-Psi (mx/multiply TWO (mx/sum (mx/log (mx/diag L-psi))))
        _ (mx/eval! log-det-Psi)]
    (dc/->Distribution :inv-wishart
                        {:df df-val :scale-matrix Psi-2d :k k
                         :wish wish :log-det-Psi log-det-Psi})))

(defmethod dc/dist-sample :inv-wishart [d key]
  (let [{:keys [wish]} (:params d)
        W (dc/dist-sample wish key)]
    (mx/inv W)))

(defmethod dc/dist-log-prob :inv-wishart [d x]
  (let [{:keys [df scale-matrix log-det-Psi k]} (:params d)
        x (if (mx/array? x) x (mx/array x))
        x-2d (if (= 1 (mx/ndim x)) (mx/reshape x [k k]) x)
        X-inv (mx/inv x-2d)
        log-det-X (mx/multiply TWO
                               (mx/sum (mx/log (mx/diag (mx/cholesky x-2d)))))
        _ (mx/eval! log-det-X)
        ;; log p(X) = (df/2)*log|Psi| - (df*k/2)*log(2) - log_multivariate_gamma(df/2, k)
        ;;            - ((df+k+1)/2)*log|X| - (1/2)*tr(Psi * X^{-1})
        half-df (/ df 2.0)
        term1 (mx/multiply (mx/scalar half-df) log-det-Psi)
        term2 (mx/scalar (- (* half-df k (js/Math.log 2.0))))
        term3 (mx/scalar (- (log-multivariate-gamma half-df k)))
        term4 (mx/multiply (mx/scalar (- (/ (+ df k 1) 2.0))) log-det-X)
        tr-PsiXinv (mx/sum (mx/multiply scale-matrix (mx/transpose X-inv)))
        term5 (mx/multiply (mx/scalar -0.5) tr-PsiXinv)]
    (mx/add term1 term2 term3 term4 term5)))
