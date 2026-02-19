(ns genmlx.inference.kernel
  "Composable inference kernels with chain, repeat, and seed operators.
   An inference kernel is a function: (fn [trace key] -> trace)
   that transforms a trace via some MCMC or other transition."
  (:require [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Core kernel constructors
;; ---------------------------------------------------------------------------

(defn mh-kernel
  "Create an MH kernel that regenerates the given selection.
   Returns (fn [trace key] -> trace)."
  [selection]
  (fn [trace key]
    (let [gf (:gen-fn trace)
          result (p/regenerate gf trace selection)
          w (mx/realize (:weight result))]
      (if (u/accept-mh? w key)
        (:trace result)
        trace))))

(defn update-kernel
  "Create a kernel that updates with given constraints.
   Returns (fn [trace key] -> trace)."
  [constraints]
  (fn [trace _key]
    (let [gf (:gen-fn trace)
          result (p/update gf trace constraints)]
      (:trace result))))

;; ---------------------------------------------------------------------------
;; Kernel combinators
;; ---------------------------------------------------------------------------

(defn chain
  "Compose inference kernels sequentially.
   (chain k1 k2 k3) applies k1, then k2, then k3.
   Each kernel is (fn [trace key] -> trace)."
  [& kernels]
  (fn [trace key]
    (let [keys (rng/split-n (rng/ensure-key key) (count kernels))]
      (reduce (fn [t [kernel ki]]
                (kernel t ki))
              trace
              (map vector kernels keys)))))

(defn repeat-kernel
  "Apply kernel n times sequentially.
   Returns (fn [trace key] -> trace)."
  [n kernel]
  (fn [trace key]
    (let [keys (rng/split-n (rng/ensure-key key) n)]
      (reduce (fn [t ki] (kernel t ki))
              trace
              keys))))

(defn seed
  "Fix the PRNG key for a kernel. The same key is used every call.
   Returns (fn [trace _key] -> trace)."
  [kernel fixed-key]
  (fn [trace _key]
    (kernel trace fixed-key)))

(defn cycle-kernels
  "Cycle through kernels repeatedly for n total applications.
   (cycle-kernels 10 [k1 k2 k3]) applies k1, k2, k3, k1, k2, k3, k1, k2, k3, k1."
  [n kernels]
  (let [kernel-vec (vec kernels)
        k (count kernel-vec)]
    (fn [trace key]
      (let [keys (rng/split-n (rng/ensure-key key) n)]
        (reduce (fn [t [i ki]]
                  ((nth kernel-vec (mod i k)) t ki))
                trace
                (map-indexed vector keys))))))

(defn mix-kernels
  "Randomly select one kernel per step from a weighted collection.
   kernel-weights: vector of [kernel weight] pairs.
   Returns (fn [trace key] -> trace)."
  [kernel-weights]
  (let [kernels (mapv first kernel-weights)
        weights (mx/array (mapv second kernel-weights))
        log-weights (mx/log weights)]
    (fn [trace key]
      (let [[k1 k2] (rng/split (rng/ensure-key key))
            idx (mx/realize (rng/categorical k1 log-weights))]
        ((nth kernels (int idx)) trace k2)))))

;; ---------------------------------------------------------------------------
;; Kernel execution
;; ---------------------------------------------------------------------------

(defn run-kernel
  "Run a kernel for n-samples iterations with burn-in and thinning.
   Returns a vector of traces with {:acceptance-rate ...} metadata."
  [{:keys [samples burn thin callback key]
    :or {burn 0 thin 1}}
   kernel init-trace]
  (let [total (+ burn (* samples thin))]
    (loop [i 0 trace init-trace
           acc (transient []) n 0 n-accepted 0
           rk key]
      (if (>= n samples)
        (with-meta (persistent! acc)
          {:acceptance-rate (if (pos? i) (/ n-accepted i) 0)})
        (let [[step-key next-key] (rng/split-or-nils rk)
              trace' (kernel trace step-key)
              accepted? (not (identical? trace' trace))
              past-burn? (>= i burn)
              keep? (and past-burn? (zero? (mod (- i burn) thin)))]
          (when (and callback keep?)
            (callback {:iter n :trace trace' :accepted? accepted?}))
          (recur (inc i) trace'
                 (if keep? (conj! acc trace') acc)
                 (if keep? (inc n) n)
                 (if accepted? (inc n-accepted) n-accepted)
                 next-key))))))
