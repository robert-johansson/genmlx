(ns genmlx.inference.kernel
  "Composable inference kernels with chain, repeat, and seed operators.
   An inference kernel is a function: (fn [trace key] -> trace)
   that transforms a trace via some MCMC or other transition."
  (:require [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Kernel reversal declarations
;; ---------------------------------------------------------------------------

(defn with-reversal
  "Declare that reverse-kernel is the reversal of kernel.
   Sets metadata on both directions so (reversal (reversal k)) = k."
  [kernel reverse-kernel]
  (let [fwd (vary-meta kernel assoc ::reversal reverse-kernel)
        bwd (vary-meta reverse-kernel assoc ::reversal fwd)]
    ;; Update fwd's reversal to point to the updated bwd
    (vary-meta fwd assoc ::reversal bwd)))

(defn symmetric-kernel
  "Declare kernel as symmetric (its own reversal)."
  [kernel]
  (vary-meta kernel assoc ::symmetric true ::reversal ::self))

(defn reversal
  "Get the declared reversal of a kernel, or nil."
  [kernel]
  (let [r (::reversal (meta kernel))]
    (if (= r ::self) kernel r)))

(defn symmetric?
  "Check if kernel is declared symmetric."
  [kernel]
  (boolean (::symmetric (meta kernel))))

(defn reversed
  "Return the reversal of a kernel. Throws if no reversal declared."
  [kernel]
  (or (reversal kernel)
      (throw (ex-info "No reversal declared for kernel" {}))))

;; ---------------------------------------------------------------------------
;; Core kernel constructors
;; ---------------------------------------------------------------------------

(defn mh-kernel
  "Create an MH kernel that regenerates the given selection.
   Returns (fn [trace key] -> trace). Symmetric by default."
  [selection]
  (symmetric-kernel
    (fn [trace key]
      (let [gf (dyn/auto-key (:gen-fn trace))
            result (p/regenerate gf trace selection)
            w (mx/realize (:weight result))]
        (if (u/accept-mh? w key)
          (:trace result)
          trace)))))

(defn update-kernel
  "Create a kernel that updates with given constraints.
   Returns (fn [trace key] -> trace)."
  [constraints]
  (fn [trace _key]
    (let [gf (dyn/auto-key (:gen-fn trace))
          result (p/update gf trace constraints)]
      (:trace result))))

;; ---------------------------------------------------------------------------
;; Kernel combinators
;; ---------------------------------------------------------------------------

(defn- chain-raw
  "Internal: compose kernels without reversal propagation."
  [kernels]
  (fn [trace key]
    (let [keys (rng/split-n (rng/ensure-key key) (count kernels))]
      (reduce (fn [t [kernel ki]]
                (kernel t ki))
              trace
              (map vector kernels keys)))))

(defn chain
  "Compose inference kernels sequentially.
   (chain k1 k2 k3) applies k1, then k2, then k3.
   Each kernel is (fn [trace key] -> trace).
   If all input kernels have reversals, the composite does too:
   reversal(chain(k1, k2, k3)) = chain(reversal(k3), reversal(k2), reversal(k1))."
  [& kernels]
  (let [result (chain-raw kernels)
        reversals (mapv reversal kernels)]
    (if (every? some? reversals)
      (let [rev (chain-raw (reverse reversals))]
        (with-reversal result rev))
      result)))

(defn- repeat-raw
  "Internal: repeat kernel n times without reversal propagation."
  [n kernel]
  (fn [trace key]
    (let [keys (rng/split-n (rng/ensure-key key) n)]
      (reduce (fn [t ki] (kernel t ki))
              trace
              keys))))

(defn repeat-kernel
  "Apply kernel n times sequentially.
   Returns (fn [trace key] -> trace).
   If kernel has a reversal, the composite does too."
  [n kernel]
  (let [result (repeat-raw n kernel)
        rev (reversal kernel)]
    (if rev
      (with-reversal result (repeat-raw n rev))
      result)))

(defn- seed-raw
  "Internal: seed kernel without reversal propagation."
  [kernel fixed-key]
  (fn [trace _key]
    (kernel trace fixed-key)))

(defn seed
  "Fix the PRNG key for a kernel. The same key is used every call.
   Returns (fn [trace _key] -> trace).
   If kernel has a reversal, the composite does too."
  [kernel fixed-key]
  (let [result (seed-raw kernel fixed-key)
        rev (reversal kernel)]
    (if rev
      (with-reversal result (seed-raw rev fixed-key))
      result)))

(defn- cycle-raw
  "Internal: cycle kernels without reversal propagation."
  [n kernel-vec]
  (let [k (count kernel-vec)]
    (fn [trace key]
      (let [keys (rng/split-n (rng/ensure-key key) n)]
        (reduce (fn [t [i ki]]
                  ((nth kernel-vec (mod i k)) t ki))
                trace
                (map-indexed vector keys))))))

(defn cycle-kernels
  "Cycle through kernels repeatedly for n total applications.
   (cycle-kernels 10 [k1 k2 k3]) applies k1, k2, k3, k1, k2, k3, k1, k2, k3, k1.
   If all kernels have reversals, the composite does too."
  [n kernels]
  (let [kernel-vec (vec kernels)
        result (cycle-raw n kernel-vec)
        reversals (mapv reversal kernel-vec)]
    (if (every? some? reversals)
      (with-reversal result (cycle-raw n (vec (reverse reversals))))
      result)))

(defn- mix-raw
  "Internal: mix kernels without reversal propagation."
  [kernels weights-arr]
  (let [log-weights (mx/log weights-arr)]
    (fn [trace key]
      (let [[k1 k2] (rng/split (rng/ensure-key key))
            idx (mx/realize (rng/categorical k1 log-weights))]
        ((nth kernels (int idx)) trace k2)))))

(defn mix-kernels
  "Randomly select one kernel per step from a weighted collection.
   kernel-weights: vector of [kernel weight] pairs.
   Returns (fn [trace key] -> trace).
   If all kernels have reversals, the composite does too."
  [kernel-weights]
  (let [kernels (mapv first kernel-weights)
        weights-arr (mx/array (mapv second kernel-weights))
        result (mix-raw kernels weights-arr)
        reversals (mapv #(reversal (first %)) kernel-weights)]
    (if (every? some? reversals)
      (with-reversal result (mix-raw reversals weights-arr))
      result)))

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
  (mx/with-resource-guard
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
  "Gaussian random-walk MH kernel. Symmetric by default.
   Single address: (random-walk :x 0.5) — proposes x' = x + N(0, 0.5).
   Multi-address:  (random-walk {:x 0.5 :y 0.1}) — chains per-address walks."
  ([addr-or-map std]
   (if (map? addr-or-map)
     (apply chain (map (fn [[a s]] (random-walk a s)) addr-or-map))
     (symmetric-kernel
       (fn [trace key]
         (let [gf (dyn/auto-key (:gen-fn trace))
               [k1 k2] (rng/split (rng/ensure-key key))
               cur-val (cm/get-choice (:choices trace) [addr-or-map])
               noise   (mx/multiply (rng/normal k1 (mx/shape cur-val))
                                    (mx/scalar std))
               proposed (mx/add cur-val noise)
               constraints (cm/choicemap addr-or-map proposed)
               result (p/update gf trace constraints)
               w (mx/realize (:weight result))]
           (if (u/accept-mh? w k2)
             (:trace result)
             trace))))))
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

(defn- proposal-asymmetric-raw
  "Internal: build asymmetric proposal kernel without reversal metadata."
  [fwd-gf backward]
  (fn [trace key]
    (let [gf (dyn/auto-key (:gen-fn trace))
          [_k1 _k2 k3] (rng/split-n (rng/ensure-key key) 3)
          fwd-result    (p/propose fwd-gf [(:choices trace)])
          fwd-choices   (:choices fwd-result)
          fwd-score     (:weight fwd-result)
          update-result (p/update gf trace fwd-choices)
          trace'        (:trace update-result)
          update-weight (:weight update-result)
          bwd-result    (p/assess backward [(:choices trace')] (:discard update-result))
          bwd-score     (:weight bwd-result)
          _             (mx/materialize! update-weight fwd-score bwd-score)
          log-alpha     (- (+ (mx/item update-weight) (mx/item bwd-score))
                           (mx/item fwd-score))]
      (if (u/accept-mh? log-alpha k3)
        trace'
        trace))))

(defn proposal
  "MH kernel with a custom proposal generative function.
   The proposal-gf takes [current-trace-choices] as args and proposes new choices.
   Symmetric:  (proposal my-gf) — marked symmetric
   Asymmetric: (proposal fwd-gf :backward bwd-gf) — forward/backward reversal pair"
  [fwd-gf & {:keys [backward]}]
  (if backward
    (let [fwd-kern (proposal-asymmetric-raw fwd-gf backward)
          bwd-kern (proposal-asymmetric-raw backward fwd-gf)]
      (with-reversal fwd-kern bwd-kern))
    ;; Symmetric: weight = update-weight (forward and backward cancel)
    (symmetric-kernel
      (fn [trace key]
        (let [gf (dyn/auto-key (:gen-fn trace))
              [_k1 k2]     (rng/split (rng/ensure-key key))
              fwd-result    (p/propose fwd-gf [(:choices trace)])
              fwd-choices   (:choices fwd-result)
              update-result (p/update gf trace fwd-choices)
              w             (mx/realize (:weight update-result))]
          (if (u/accept-mh? w k2)
            (:trace update-result)
            trace))))))

(defn gibbs
  "Convenience for the Gibbs cycling pattern.
   (gibbs :x :y :z)         — resample each from prior in sequence
   (gibbs {:x 0.5 :y 0.1})  — random walk on each with given std"
  [& args]
  (if (and (= 1 (count args)) (map? (first args)))
    (random-walk (first args))
    (apply chain (map prior args))))
