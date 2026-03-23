(ns genmlx.inference.compiled-smc
  "Level 2 compiled SMC: bootstrap particle filter for unfold models.

   Particles are stored as [N,K] tensors. Each extend step broadcasts
   noise transforms over N particles simultaneously.

   Supports three resampling methods:
   - :systematic     — O(N²) broadcasting (default, non-differentiable)
   - :gumbel-top-k   — Gumbel-top-k trick (all-GPU, non-differentiable)
   - :gumbel-softmax — Gumbel-softmax relaxation (differentiable, for mx/grad)

   Architecture:
   - generate-smc-noise: pre-generate [T,N,K] extend noise + [T] resample uniforms
   - cops/make-smc-extend-step: build extend step from kernel schema
   - systematic-resample-tensor: [N] log-weights → resampled [N,K] particles
   - compiled-smc: full bootstrap PF with chunked execution"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.compiled-ops :as cops]
            [genmlx.choicemap :as cm]
            [genmlx.tensor-trace :as tt]
            [genmlx.inference.util :as u]
            [genmlx.inference.differentiable-resample :as dr]))

;; =========================================================================
;; Noise generation
;; =========================================================================

(defn generate-smc-noise
  "Pre-generate all randomness for a T-step, N-particle SMC sweep.
   Returns {:extend-noise [T,N,K] :resample-uniforms [T]}."
  [key T N K]
  (let [[k1 k2] (rng/split key)]
    {:extend-noise (rng/normal k1 [T N K])
     :resample-uniforms (rng/uniform k2 [T])}))

;; =========================================================================
;; Systematic resampling on tensors (O(N) via searchsorted)
;; =========================================================================

(defn systematic-resample-tensor
  "Systematic resampling: [N] log-weights + scalar uniform → resampled [N,K] particles.
   Uses O(N) searchsorted for index computation.

   Returns {:particles [N,K] :ancestors [N] int32}."
  [particles log-weights uniform N]
  (let [probs (mx/softmax log-weights)
        cumsum (mx/cumsum probs)
        indices-float (mx/astype (mx/arange 0 N 1) mx/float32)
        positions (mx/divide (mx/add indices-float uniform) (mx/scalar N))
        ancestors (mx/searchsorted cumsum positions)
        ancestors (mx/minimum ancestors (mx/scalar (dec N) mx/int32))
        resampled (mx/take-idx particles ancestors 0)]
    {:particles resampled :ancestors ancestors}))

;; =========================================================================
;; Extend + resample helpers
;; =========================================================================

(defn- extend-particles
  "Run one extend step: propose new particles via noise transforms.
   Returns {:new-particles [N,K] :new-state [N] :obs-log-prob [N] :ml-inc scalar}."
  [extend-fn noise-t current-state obs-t all-addrs t N]
  (let [kernel-args [(mx/ensure-array t) current-state]
        {:keys [obs-log-prob values-map retval]} (extend-fn noise-t kernel-args obs-t)
        new-particles (mx/stack (mapv #(get values-map %) all-addrs) 1)
        new-state (or retval (get values-map (first all-addrs)))
        ml-inc (mx/subtract (mx/logsumexp obs-log-prob)
                             (mx/scalar (js/Math.log N)))]
    (mx/materialize! obs-log-prob ml-inc new-particles new-state)
    {:new-particles new-particles :new-state new-state
     :obs-log-prob obs-log-prob :ml-inc ml-inc}))

(defn- resample-particles
  "Resample particles and state using the selected method.
   Returns {:particles [N,K] :state [N]}."
  [new-particles new-state obs-log-prob resample-method
   noise t gumbel-noise tau-arr N]
  (let [{:keys [particles ancestors]}
        (case resample-method
          :systematic
          (systematic-resample-tensor new-particles obs-log-prob
                                      (mx/index (:resample-uniforms noise) t) N)
          :gumbel-top-k
          (dr/gumbel-top-k new-particles obs-log-prob (mx/index gumbel-noise t))
          :gumbel-softmax
          (dr/gumbel-softmax new-particles obs-log-prob
                             (mx/index gumbel-noise t) tau-arr))
        resampled-state
        (if ancestors
          (mx/take-idx new-state ancestors 0)
          (dr/gumbel-softmax-1d new-state obs-log-prob
                                (mx/index gumbel-noise t) tau-arr))]
    (mx/materialize! particles resampled-state)
    {:particles particles :state resampled-state}))

;; =========================================================================
;; Compiled SMC: bootstrap particle filter
;; =========================================================================

(defn compiled-smc
  "Compiled bootstrap particle filter for unfold models.

   opts:
     :particles        — number of particles N (default 100)
     :key              — PRNG key
     :callback         — (fn [{:step :ess :resampled?}]) called each step
     :resample-method  — :systematic (default), :gumbel-top-k, or :gumbel-softmax
     :tau              — temperature for :gumbel-softmax (default 1.0)

   kernel: DynamicGF — the unfold kernel (takes [t state & extra])
   init-state: initial state
   observations-seq: seq of ChoiceMaps, one per timestep

   Returns {:log-ml MLX-scalar :particles [N,K] :addr-index {addr->int}
            :final-ess number}"
  [{:keys [particles key callback resample-method tau]
    :or {particles 100 resample-method :systematic tau 1.0}}
   kernel init-state observations-seq]
  (let [schema (:schema kernel)
        source (:source kernel)
        extend-fn (cops/make-smc-extend-step schema source)
        obs-vec (vec observations-seq)
        T (count obs-vec)
        static-sites (filterv :static? (:trace-sites schema))
        all-addrs (mapv :addr static-sites)
        addr-index (into {} (map-indexed (fn [i a] [a i]) all-addrs))
        K (count all-addrs)
        N particles]
    (when-not extend-fn
      (throw (ex-info "Kernel cannot be compiled for SMC" {:kernel kernel})))
    (let [rk (rng/ensure-key key)
          [noise-key gumbel-key run-key] (rng/split-n rk 3)
          noise (generate-smc-noise noise-key T N K)
          gumbel-noise (when (#{:gumbel-top-k :gumbel-softmax} resample-method)
                         (let [gn (dr/generate-gumbel-noise gumbel-key T N)]
                           (mx/materialize! gn)
                           gn))
          tau-arr (mx/scalar tau)
          init-state-n (mx/broadcast-to (mx/ensure-array init-state) [N])]
      (mx/materialize! (:extend-noise noise) (:resample-uniforms noise))
      (loop [t 0
             current-particles nil
             current-state init-state-n
             log-ml (mx/scalar 0.0)]
        (if (>= t T)
          {:log-ml log-ml :particles current-particles
           :addr-index addr-index :final-ess (double N)}
          (let [{:keys [new-particles new-state obs-log-prob ml-inc]}
                (extend-particles extend-fn (mx/index (:extend-noise noise) t)
                                  current-state (nth obs-vec t) all-addrs t N)
                {:keys [particles state]}
                (resample-particles new-particles new-state obs-log-prob
                                    resample-method noise t gumbel-noise tau-arr N)]
            (when (zero? (mod (inc t) 5)) (mx/clear-cache!))
            (when callback (callback {:step t :resampled? true}))
            (recur (inc t) particles state (mx/add log-ml ml-inc))))))))

;; =========================================================================
;; Convenience: extract results
;; =========================================================================

(defn smc-result->traces
  "Convert compiled SMC result to vector of TensorTraces."
  [result kernel]
  (let [{:keys [particles addr-index]} result
        N (first (mx/shape particles))]
    (mapv (fn [i]
            (let [values (mx/index particles i)]
              (tt/make-tensor-trace
                {:gen-fn kernel :args nil
                 :values values :addr-index addr-index
                 :score (mx/scalar 0.0) :retval nil})))
          (range N))))
