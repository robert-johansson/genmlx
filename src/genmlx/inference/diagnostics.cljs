(ns genmlx.inference.diagnostics
  "MCMC diagnostics: ESS, R-hat, trace summary statistics."
  (:require [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; Shared helpers
;; ---------------------------------------------------------------------------

(defn- flatten-samples
  "Stack, materialize, and flatten a vector of parameter samples (MLX arrays)
   to a CLJS seq of doubles. Scalar params pass through; array-valued params
   contribute their first element."
  [samples]
  (let [stacked (mx/stack samples)]
    (mx/materialize! stacked)
    (let [raw-vals (mx/->clj stacked)]
      (if (number? (first raw-vals)) raw-vals (map first raw-vals)))))

;; ---------------------------------------------------------------------------
;; Effective Sample Size (ESS)
;; ---------------------------------------------------------------------------

(defn ess
  "Compute ESS from a vector of parameter samples (MLX arrays).
   Uses the autocorrelation-based estimator.
   Array-valued samples: only element 0 is diagnosed (see flatten-samples) —
   pass a single component explicitly for other marginals.
   Returns a number."
  [samples]
  (let [n (count samples)
        flat-vals (vec (flatten-samples samples))
        mu (/ (reduce + flat-vals) n)
        ;; Compute autocovariance
        centered (mapv #(- % mu) flat-vals)
        var0 (/ (reduce + (map #(* % %) centered)) n)]
    (if (zero? var0)
      n
      (let [;; Compute autocorrelation at increasing lags
            max-lag (min (dec n) (int (/ n 2)))
            autocorr (fn [lag]
                       (/ (reduce + (map (fn [i] (* (nth centered i)
                                                     (nth centered (+ i lag))))
                                         (range (- n lag))))
                          (* n var0)))]
        ;; Sum pairs of autocorrelations until they go negative (Geyer's initial positive sequence)
        (loop [lag 1 sum-rho 0.0]
          (if (>= lag max-lag)
            (/ n (max 1.0 (+ 1.0 (* 2.0 sum-rho))))
            (let [rho-lag (autocorr lag)
                  rho-lag1 (if (< (inc lag) n) (autocorr (inc lag)) 0.0)
                  pair-sum (+ rho-lag rho-lag1)]
              (if (neg? pair-sum)
                (/ n (max 1.0 (+ 1.0 (* 2.0 sum-rho))))
                (recur (+ lag 2) (+ sum-rho pair-sum))))))))))

;; ---------------------------------------------------------------------------
;; Rank-normalized bulk-ESS and tail-ESS (Vehtari et al. 2021,
;; "Rank-normalization, folding, and localization: An improved R-hat")
;;
;; SCOPE: bulk-ess/tail-ess (and r-hat) are MULTI-CHAIN MCMC diagnostics --
;; they take several independent chains and quantify mixing/convergence of a
;; scalar marginal via autocorrelation. They do NOT apply to particle methods
;; (importance sampling / SMC), which have no chains and no autocorrelation;
;; those report weight-based ESS (u/compute-ess) instead. Hence the funnel
;; (HMC/NUTS/MH) bench carries r-hat + bulk/tail-ESS, while the hmm/gmm/
;; changepoint (IS/SMC) benches carry weight-based ESS -- by design, not omission.
;; ---------------------------------------------------------------------------

(defn- chain->doubles
  "Materialize a chain (vector of MLX scalars or plain numbers) to doubles."
  [chain]
  (mapv (fn [s]
          (cond
            (number? s) s
            (mx/array? s) (do (mx/materialize! s) (mx/item s))
            :else s))
        chain))

(defn- quantile
  "Type-7 (linear-interpolation) quantile of a vector of doubles."
  [vals p]
  (let [sorted (vec (sort vals))
        s (count sorted)
        h (* (dec s) p)
        lo (int (js/Math.floor h))
        hi (min (dec s) (inc lo))
        frac (- h lo)]
    (+ (nth sorted lo) (* frac (- (nth sorted hi) (nth sorted lo))))))

(defn- autocovariance
  "Biased autocovariances gamma_t = (1/n) sum_i (x_i-xbar)(x_{i+t}-xbar),
   t = 0..n-1. Returns a vector of n doubles."
  [xs]
  (let [n (count xs)
        v (vec xs)
        xbar (/ (reduce + v) n)
        c (mapv #(- % xbar) v)]
    (mapv (fn [t]
            (/ (loop [i 0 acc 0.0]
                 (if (< i (- n t))
                   (recur (inc i) (+ acc (* (nth c i) (nth c (+ i t)))))
                   acc))
               n))
          (range n))))

(defn- polyval
  "Horner evaluation: [k0 k1 ... km] x -> k0 x^m + k1 x^(m-1) + ... + km."
  [coeffs x]
  (reduce (fn [acc k] (+ (* acc x) k)) 0.0 coeffs))

(defn- inv-normal-cdf
  "Inverse standard-normal CDF (Acklam's rational approximation, |err|<1.15e-9)."
  [p]
  (let [a [-3.969683028665376e+01 2.209460984245205e+02 -2.759285104469687e+02
           1.383577518672690e+02 -3.066479806614716e+01 2.506628277459239e+00]
        b [-5.447609879822406e+01 1.615858368580409e+02 -1.556989798598866e+02
           6.680131188771972e+01 -1.328068155288572e+01 1.0]
        c [-7.784894002430293e-03 -3.223964580411365e-01 -2.400758277161838e+00
           -2.549732539343734e+00 4.374664141464968e+00 2.938163982698783e+00]
        d [7.784695709041462e-03 3.224671290700398e-01 2.445134137142996e+00
           3.754408661907416e+00 1.0]
        plow 0.02425
        phigh (- 1.0 plow)]
    (cond
      (< p plow)   (let [q (js/Math.sqrt (* -2.0 (js/Math.log p)))]
                     (/ (polyval c q) (polyval d q)))
      (<= p phigh) (let [q (- p 0.5) r (* q q)]
                     (/ (* (polyval a r) q) (polyval b r)))
      :else        (let [q (js/Math.sqrt (* -2.0 (js/Math.log (- 1.0 p))))]
                     (- (/ (polyval c q) (polyval d q)))))))

(defn- rank-normalize
  "Average-rank-normalize a pooled vector via the Blom transform
   z_i = Phi^{-1}((r_i - 3/8)/(S - 1/4)). Returns a vector aligned with input."
  [pooled]
  (let [s (count pooled)
        order (vec (sort-by (fn [i] (nth pooled i)) (range s)))
        ranks (double-array s)]
    (loop [k 0]
      (when (< k s)
        (let [vk (nth pooled (nth order k))
              j (loop [j (inc k)]
                  (if (and (< j s) (= (nth pooled (nth order j)) vk))
                    (recur (inc j)) j))
              avg (/ (+ (inc k) j) 2.0)]   ; mean of 1-based ranks (k+1 .. j)
          (doseq [t (range k j)] (aset ranks (nth order t) avg))
          (recur j))))
    (mapv (fn [i] (inv-normal-cdf (/ (- (aget ranks i) 0.375) (- s 0.25))))
          (range s))))

(defn- split-chains
  "Split each chain into two halves, dropping the middle element if odd length."
  [chains-vals]
  (vec (mapcat (fn [c]
                 (let [c (vec c) n (count c) h (quot n 2)]
                   [(subvec c 0 h) (subvec c (- n h) n)]))
               chains-vals)))

(defn- ess-from-chains
  "Multi-chain effective sample size (Vehtari et al. 2021 / Stan) over chains of
   equal length given as vectors of doubles. Returns a number."
  [chains-vals]
  (let [m (count chains-vals)
        n (count (first chains-vals))]
    (if (< n 8)
      (* m n)
      (let [acovs (mapv autocovariance chains-vals)
            means (mapv (fn [xs] (/ (reduce + xs) n)) chains-vals)
            mean-var (* (/ (reduce + (map #(nth % 0) acovs)) m) (/ n (dec n)))
            grand (/ (reduce + means) m)
            between (if (> m 1)
                      (/ (reduce + (map #(let [e (- % grand)] (* e e)) means)) (dec m))
                      0.0)
            var-plus (+ (* mean-var (/ (dec n) n)) between)]
        (if (<= var-plus 0.0)
          (* m n)                          ; degenerate (constant) -> nominal
          (let [rho (fn [t]
                      (if (zero? t) 1.0
                          (- 1.0 (/ (- mean-var (/ (reduce + (map #(nth % t) acovs)) m))
                                    var-plus))))
                ;; Geyer pair-sums Gamma_k = rho(2k) + rho(2k+1)
                gammas (loop [k 0 acc []]
                         (let [te (* 2 k) to (inc te)]
                           (if (> to (dec n))
                             acc
                             (recur (inc k) (conj acc (+ (rho te) (rho to)))))))
                pos-gammas (vec (take-while pos? gammas))]
            (if (empty? pos-gammas)
              (* m n)
              (let [mono (reductions min (first pos-gammas) (rest pos-gammas))
                    tau (max (- (* 2.0 (reduce + mono)) 1.0)
                             (/ 1.0 (js/Math.log10 (* m n))))]
                (/ (* m n) tau)))))))))

(defn bulk-ess
  "Rank-normalized bulk effective sample size (Vehtari et al. 2021).
   chains: vector of equal-length chains, each a vector of MLX scalars or
   numbers. Captures sampling efficiency for the bulk of the distribution."
  [chains]
  (let [chains-d (mapv chain->doubles chains)
        n (count (first chains-d))]
    (if (or (zero? n) (some #(not= n (count %)) chains-d))
      0.0
      (let [pooled (vec (apply concat chains-d))
            z (rank-normalize pooled)
            z-chains (mapv (fn [ci] (subvec z (* ci n) (* (inc ci) n)))
                           (range (count chains-d)))]
        (ess-from-chains (split-chains z-chains))))))

(defn tail-ess
  "Tail effective sample size (Vehtari et al. 2021): the minimum of the ESS of
   the 5%- and 95%-quantile tail indicators (pooled quantiles, split chains).
   chains: vector of equal-length chains of MLX scalars or numbers."
  [chains]
  (let [chains-d (mapv chain->doubles chains)
        n (count (first chains-d))]
    (if (or (zero? n) (some #(not= n (count %)) chains-d))
      0.0
      (let [pooled (vec (apply concat chains-d))
            q05 (quantile pooled 0.05)
            q95 (quantile pooled 0.95)
            indic (fn [thr] (mapv (fn [c] (mapv (fn [x] (if (<= x thr) 1.0 0.0)) c))
                                  chains-d))]
        (min (ess-from-chains (split-chains (indic q05)))
             (ess-from-chains (split-chains (indic q95))))))))

;; ---------------------------------------------------------------------------
;; R-hat (Gelman-Rubin)
;; ---------------------------------------------------------------------------

(defn r-hat
  "Compute R-hat from multiple chains of parameter samples.
   chains: vector of vectors of MLX arrays.
   Returns a number; values close to 1.0 indicate convergence.

   This is the CLASSIC Gelman-Rubin potential scale reduction factor: no
   chain splitting, no rank normalization. It can miss non-stationarity
   within chains and heavy-tail pathologies that the rank-normalized
   split-R-hat of Vehtari et al. 2021 detects; bulk-ess/tail-ess in this
   namespace ARE the Vehtari diagnostics. Documented as-is (genmlx-7ca0)."
  [chains]
  (let [m (count chains)
        n (count (first chains))
        ;; Extract scalar values
        chain-vals (mapv (fn [chain]
                           (mapv (fn [s]
                                   (mx/materialize! s)
                                   (if (mx/array? s) (mx/item s) s))
                                 chain))
                         chains)
        ;; Sum of squared deviations of xs from their mean mu
        ss (fn [xs mu] (reduce + (map #(let [d (- % mu)] (* d d)) xs)))
        ;; Chain means
        chain-means (mapv (fn [vals] (/ (reduce + vals) n)) chain-vals)
        overall-mean (/ (reduce + chain-means) m)
        ;; Between-chain variance
        B (* (/ n (dec m))
             (ss chain-means overall-mean))
        ;; Within-chain variance
        W (/ (reduce + (map (fn [vals mu]
                              (/ (ss vals mu) (dec n)))
                            chain-vals chain-means))
             m)
        ;; R-hat
        var-hat (+ (* (/ (dec n) n) W) (/ B n))]
    (if (zero? W) 1.0 (js/Math.sqrt (/ var-hat W)))))

;; ---------------------------------------------------------------------------
;; Summary statistics
;; ---------------------------------------------------------------------------

(defn sample-mean
  "Mean of parameter samples. Returns MLX array."
  [samples]
  (let [stacked (mx/stack samples)]
    (mx/mean stacked [0])))

(defn sample-std
  "Standard deviation of parameter samples. Returns MLX array."
  [samples]
  (let [stacked (mx/stack samples)]
    (mx/std stacked [0])))

(defn sample-quantiles
  "Compute quantiles of samples.
   Array-valued samples: only element 0 is summarized (see flatten-samples).
   Returns {:median :q025 :q975}."
  [samples]
  (let [n (count samples)
        sorted-vals (sort (flatten-samples samples))
        idx-025 (int (* 0.025 n))
        idx-50  (int (* 0.5 n))
        idx-975 (int (* 0.975 n))]
    {:median (nth sorted-vals idx-50)
     :q025 (nth sorted-vals idx-025)
     :q975 (nth sorted-vals idx-975)}))

(defn summarize
  "Print summary statistics for MCMC samples."
  [samples & {label :name :or {label "param"}}]
  (let [mu (sample-mean samples)
        sd (sample-std samples)
        {:keys [median q025 q975]} (sample-quantiles samples)
        effective (ess samples)]
    (mx/materialize! mu sd)
    {:name label
     :mean (mx/item mu)
     :std (mx/item sd)
     :median median
     :q025 q025
     :q975 q975
     :ess effective}))
