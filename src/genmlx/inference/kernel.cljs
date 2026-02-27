(ns genmlx.inference.kernel
  "Composable inference kernels with chain, repeat, and seed operators.
   An inference kernel is a function: (fn [trace key] -> trace)
   that transforms a trace via some MCMC or other transition."
  (:require [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.choicemap :as cm]
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

(defn collect-samples
  "Generic sample collection loop with burn-in, thinning, and callback.
   - `step-fn`:    (fn [state key] -> {:state new-state :accepted? bool})
   - `extract-fn`: (fn [state] -> sample)
   - `init-state`: initial state
   - Returns vector of samples with {:acceptance-rate ...} metadata."
  [{:keys [samples burn thin callback key]
    :or {burn 0 thin 1}}
   step-fn extract-fn init-state]
  (u/with-resource-guard
    (fn []
      (let [total-iters (+ burn (* samples thin))]
        (loop [i 0, state init-state, acc (transient []), n 0, n-accepted 0, rk key]
          (if (>= n samples)
            (with-meta (persistent! acc)
              {:acceptance-rate (if (pos? total-iters) (/ n-accepted total-iters) 0)})
            (let [[step-key next-key] (rng/split-or-nils rk)
                  {:keys [state accepted?]} (u/tidy-step step-fn state step-key)
                  _  (mx/clear-cache!)
                  past-burn? (>= i burn)
                  keep? (and past-burn? (zero? (mod (- i burn) thin)))]
              (when (and callback keep?)
                (callback {:iter n :value (extract-fn state) :accepted? accepted?}))
              (recur (inc i) state
                     (if keep? (conj! acc (extract-fn state)) acc)
                     (if keep? (inc n) n)
                     (if accepted? (inc n-accepted) n-accepted)
                     next-key))))))))

(defn run-kernel
  "Run a kernel for n-samples iterations with burn-in and thinning.
   Returns a vector of traces with {:acceptance-rate ...} metadata."
  [{:keys [samples burn thin callback key]
    :or {burn 0 thin 1}}
   kernel init-trace]
  (collect-samples
    {:samples samples :burn burn :thin thin :key key
     :callback (when callback
                 (fn [{:keys [iter value accepted?]}]
                   (callback {:iter iter :trace value :accepted? accepted?})))}
    (fn [trace step-key]
      (let [trace' (kernel trace step-key)]
        {:state trace' :accepted? (not (identical? trace' trace))}))
    identity
    init-trace))

;; ---------------------------------------------------------------------------
;; Kernel DSL — higher-level constructors
;; ---------------------------------------------------------------------------

(defn random-walk
  "Gaussian random-walk MH kernel.
   Single address: (random-walk :x 0.5) — proposes x' = x + N(0, 0.5).
   Multi-address:  (random-walk {:x 0.5 :y 0.1}) — chains per-address walks."
  ([addr-or-map std]
   (if (map? addr-or-map)
     (apply chain (map (fn [[a s]] (random-walk a s)) addr-or-map))
     (fn [trace key]
       (let [[k1 k2] (rng/split (rng/ensure-key key))
             cur-val (cm/get-choice (:choices trace) [addr-or-map])
             noise   (mx/multiply (rng/normal k1 (mx/shape cur-val))
                                  (mx/scalar std))
             proposed (mx/add cur-val noise)
             constraints (cm/choicemap addr-or-map proposed)
             result (p/update (:gen-fn trace) trace constraints)
             w (mx/realize (:weight result))]
         (if (u/accept-mh? w k2)
           (:trace result)
           trace)))))
  ([addr-map]
   (if (map? addr-map)
     (apply chain (map (fn [[a s]] (random-walk a s)) addr-map))
     (throw (ex-info "random-walk requires std or a map" {:arg addr-map})))))

(defn prior
  "MH kernel that resamples addresses from the prior via regenerate.
   (prior :x)       — single address
   (prior :x :y :z) — joint resample"
  [& addrs]
  (mh-kernel (apply sel/select addrs)))

(defn proposal
  "MH kernel with a custom proposal generative function.
   The proposal-gf takes [current-trace-choices] as args and proposes new choices.
   Symmetric:  (proposal my-gf)
   Asymmetric: (proposal fwd-gf :backward bwd-gf)"
  [fwd-gf & {:keys [backward]}]
  (if backward
    ;; Asymmetric: weight = update-weight + backward-score - forward-score
    (fn [trace key]
      (let [[_k1 _k2 k3] (rng/split-n (rng/ensure-key key) 3)
            fwd-result    (p/propose fwd-gf [(:choices trace)])
            fwd-choices   (:choices fwd-result)
            fwd-score     (:weight fwd-result)
            update-result (p/update (:gen-fn trace) trace fwd-choices)
            trace'        (:trace update-result)
            update-weight (:weight update-result)
            bwd-result    (p/assess backward [(:choices trace')] (:discard update-result))
            bwd-score     (:weight bwd-result)
            _             (mx/eval! update-weight fwd-score bwd-score)
            log-alpha     (- (+ (mx/item update-weight) (mx/item bwd-score))
                             (mx/item fwd-score))]
        (if (u/accept-mh? log-alpha k3)
          trace'
          trace)))
    ;; Symmetric: weight = update-weight (forward and backward cancel)
    (fn [trace key]
      (let [[_k1 k2]     (rng/split (rng/ensure-key key))
            fwd-result    (p/propose fwd-gf [(:choices trace)])
            fwd-choices   (:choices fwd-result)
            update-result (p/update (:gen-fn trace) trace fwd-choices)
            w             (mx/realize (:weight update-result))]
        (if (u/accept-mh? w k2)
          (:trace update-result)
          trace)))))

(defn gibbs
  "Convenience for the Gibbs cycling pattern.
   (gibbs :x :y :z)         — resample each from prior in sequence
   (gibbs {:x 0.5 :y 0.1})  — random walk on each with given std"
  [& args]
  (if (and (= 1 (count args)) (map? (first args)))
    (random-walk (first args))
    (apply chain (map prior args))))
