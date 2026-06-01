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
;; R-hat (Gelman-Rubin)
;; ---------------------------------------------------------------------------

(defn r-hat
  "Compute R-hat from multiple chains of parameter samples.
   chains: vector of vectors of MLX arrays.
   Returns a number; values close to 1.0 indicate convergence."
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
