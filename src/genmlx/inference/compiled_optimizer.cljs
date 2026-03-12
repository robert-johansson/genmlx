(ns genmlx.inference.compiled-optimizer
  "Compiled optimization step: gradient + Adam update in a single mx/compile-fn.
   Eliminates per-iteration mx/materialize! calls, keeping the MLX lazy graph
   intact across the full step. Part of Level 4 (fused graph) — WP-0, WP-1, WP-2.

   WP-0: compiled-train, make-compiled-opt-step
   WP-1: make-compiled-loss-grad, learn
   WP-2: make-fused-mcmc-train, fused-learn

   Key insight: t (iteration counter) is passed as an MLX scalar argument,
   not closed over. This allows mx/power for bias correction inside the
   compiled graph, avoiding host-side js/Math.pow."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.inference.compiled-gradient :as cg]
            [genmlx.inference.vi :as vi]))

;; ---------------------------------------------------------------------------
;; Compiled optimization step (WP-0)
;; ---------------------------------------------------------------------------

(defn make-compiled-opt-step
  "Build a compiled single optimization step.

   score-fn: (fn [params-tensor] -> MLX scalar) — the objective to MAXIMIZE.
   opts: {:lr :beta1 :beta2 :epsilon} — Adam hyperparameters.

   Returns a compiled function:
     (fn [params adam-m adam-v t-scalar] -> #js [new-params new-m new-v loss])

   where:
     - params, adam-m, adam-v are MLX arrays of shape [D]
     - t-scalar is an MLX scalar (iteration number, 1-indexed)
     - loss is the negative score (what we minimize)
     - new-params has been updated by one Adam step on the gradient of loss

   The compiled function does NOT call mx/materialize! internally —
   the entire gradient + Adam update fuses into a single Metal dispatch.

   Note: This handles deterministic score functions. WP-2 extends this
   pattern for stochastic score functions (MCMC/SMC) where noise must
   be passed as additional arguments."
  [score-fn {:keys [lr beta1 beta2 epsilon]
             :or {lr 0.001 beta1 0.9 beta2 0.999 epsilon 1e-8}}]
  (let [;; We minimize negative score (= maximize score)
        neg-score-fn (fn [params] (mx/negative (score-fn params)))
        vg (mx/value-and-grad neg-score-fn)

        ;; Pre-create scalar constants for Adam hyperparams
        lr-s (mx/scalar lr)
        beta1-s (mx/scalar beta1)
        beta2-s (mx/scalar beta2)
        eps-s (mx/scalar epsilon)
        one-s (mx/scalar 1.0)
        one-minus-b1 (mx/scalar (- 1.0 beta1))
        one-minus-b2 (mx/scalar (- 1.0 beta2))

        step-fn
        (fn [params m v t-scalar]
          (let [;; Forward + backward in one call
                [loss grad] (vg params)

                ;; Moment updates (exponential moving averages)
                new-m (mx/add (mx/multiply beta1-s m)
                              (mx/multiply one-minus-b1 grad))
                new-v (mx/add (mx/multiply beta2-s v)
                              (mx/multiply one-minus-b2 (mx/square grad)))

                ;; Bias correction — mx/power keeps this inside the graph
                m-hat (mx/divide new-m
                                 (mx/subtract one-s (mx/power beta1-s t-scalar)))
                v-hat (mx/divide new-v
                                 (mx/subtract one-s (mx/power beta2-s t-scalar)))

                ;; Parameter update
                update (mx/divide m-hat
                                  (mx/add (mx/sqrt v-hat) eps-s))
                new-params (mx/subtract params (mx/multiply lr-s update))]
            #js [new-params new-m new-v loss]))]

    (mx/compile-fn step-fn)))

;; ---------------------------------------------------------------------------
;; Compiled training loop (WP-0)
;; ---------------------------------------------------------------------------

(defn compiled-train
  "Training loop using a compiled optimization step.

   score-fn:    (fn [params-tensor] -> MLX scalar) — objective to MAXIMIZE.
   init-params: initial MLX array of shape [D].
   opts:
     :iterations  — number of training steps (default 1000)
     :lr          — learning rate (default 0.001)
     :beta1       — Adam beta1 (default 0.9)
     :beta2       — Adam beta2 (default 0.999)
     :epsilon     — Adam epsilon (default 1e-8)
     :callback    — (fn [{:iter :loss :params}]) called when loss is logged
     :log-every   — log loss every N iterations (default 50)

   Returns {:params final-params :loss-history [numbers...]}.

   The host loop increments t as a plain integer, converts to MLX scalar
   for each call. Loss is only materialized at log boundaries."
  [score-fn init-params
   {:keys [iterations lr beta1 beta2 epsilon callback log-every]
    :or {iterations 1000 lr 0.001 beta1 0.9 beta2 0.999
         epsilon 1e-8 log-every 50}}]
  (let [opt-step (make-compiled-opt-step
                  score-fn
                  {:lr lr :beta1 beta1 :beta2 beta2 :epsilon epsilon})
        d (mx/shape init-params)
        ;; Warm-up: trace once to cache the Metal program
        _ (mx/materialize! (aget (opt-step (mx/zeros d) (mx/zeros d)
                                           (mx/zeros d) (mx/scalar 1.0)) 0))]
    (loop [i 0
           params init-params
           m (mx/zeros d)
           v (mx/zeros d)
           losses (transient [])]
      (if (>= i iterations)
        (do
          (mx/materialize! params)
          {:params params :loss-history (persistent! losses)})
        (let [;; Fresh MLX scalar each iteration — negligible vs Metal dispatch cost
              t-scalar (mx/scalar (double (inc i)))
              result (opt-step params m v t-scalar)
              np (aget result 0)
              nm (aget result 1)
              nv (aget result 2)
              loss-arr (aget result 3)

              ;; Only materialize loss at log boundaries
              log? (or (zero? i) (zero? (mod (inc i) log-every)))
              losses' (if log?
                        (do
                          (mx/materialize! loss-arr)
                          (let [loss-val (mx/item loss-arr)]
                            (when callback
                              (callback {:iter i :loss loss-val :params np}))
                            (conj! losses loss-val)))
                        losses)]

          ;; Periodic cleanup every 50 iterations
          (when (and (pos? i) (zero? (mod i 50)))
            (mx/clear-cache!)
            (mx/sweep-dead-arrays!))

          (recur (inc i) np nm nv losses'))))))

;; ===========================================================================
;; WP-1: Compiled Loss-Gradient Function
;; ===========================================================================

(defn- finite-diff-grad
  "Compute gradient of f at params using central finite differences.
   f: (fn [params-tensor] -> MLX scalar)
   params: [D] MLX array
   h: step size (double)
   Returns [D] MLX gradient array.

   Uses central differences: df/dx_i = (f(x+h*e_i) - f(x-h*e_i)) / (2h)
   Each dimension requires 2 function evaluations."
  [f params h]
  (let [d (first (mx/shape params))
        two-h (mx/scalar (* 2.0 h))]
    (mx/stack
     (mapv (fn [i]
             (let [;; Create perturbation vector: h * e_i
                   ei (mx/array (assoc (vec (repeat d 0.0)) i h))
                   f-plus (f (mx/add params ei))
                   f-minus (f (mx/subtract params ei))]
               (mx/divide (mx/subtract f-plus f-minus) two-h)))
           (range d)))))

(defn- handler-train
  "Training loop for handler-based (non-differentiable) score functions.
   Uses finite-difference gradients with Adam optimization.
   Not compiled — each iteration materializes separately.

   score-fn: (fn [params-tensor] -> MLX scalar) — objective to MAXIMIZE.
   init-params: [D] MLX array.
   opts: same as compiled-train plus :fd-h (finite diff step, default 1e-4).

   Returns {:params final-params :loss-history [numbers...]}."
  [score-fn init-params
   {:keys [iterations lr beta1 beta2 epsilon callback log-every fd-h]
    :or {iterations 1000 lr 0.001 beta1 0.9 beta2 0.999
         epsilon 1e-8 log-every 50 fd-h 1e-4}}]
  (let [d (mx/shape init-params)
        ;; Hoist closure and scalar constants outside the loop
        neg-score (fn [p] (mx/negative (score-fn p)))
        beta1-s (mx/scalar beta1)
        beta2-s (mx/scalar beta2)
        one-minus-b1-s (mx/scalar (- 1.0 beta1))
        one-minus-b2-s (mx/scalar (- 1.0 beta2))
        lr-s (mx/scalar lr)
        eps-s (mx/scalar epsilon)]
    (loop [i 0
           params init-params
           m (mx/zeros d)
           v (mx/zeros d)
           losses (transient [])]
      (if (>= i iterations)
        (do
          (mx/materialize! params)
          {:params params :loss-history (persistent! losses)})
        (let [;; Compute loss = -score (minimize)
              loss (neg-score params)
              ;; Finite-difference gradient of loss
              grad (finite-diff-grad neg-score params fd-h)

              _ (mx/materialize! loss grad)

              ;; Adam moment updates (host-side bias correction)
              t (double (inc i))
              new-m (mx/add (mx/multiply beta1-s m)
                            (mx/multiply one-minus-b1-s grad))
              new-v (mx/add (mx/multiply beta2-s v)
                            (mx/multiply one-minus-b2-s (mx/square grad)))
              m-hat (mx/divide new-m (mx/scalar (- 1.0 (js/Math.pow beta1 t))))
              v-hat (mx/divide new-v (mx/scalar (- 1.0 (js/Math.pow beta2 t))))
              update-vec (mx/divide m-hat
                                    (mx/add (mx/sqrt v-hat) eps-s))
              new-params (mx/subtract params (mx/multiply lr-s update-vec))

              _ (mx/materialize! new-params new-m new-v)

              ;; Log loss at boundaries
              log? (or (zero? i) (zero? (mod (inc i) log-every)))
              loss-val (mx/item loss)
              losses' (if log?
                        (do
                          (when callback
                            (callback {:iter i :loss loss-val :params new-params}))
                          (conj! losses loss-val))
                        losses)]

          ;; Periodic cleanup
          (when (and (pos? i) (zero? (mod i 50)))
            (mx/clear-cache!)
            (mx/sweep-dead-arrays!))

          (recur (inc i) new-params new-m new-v losses'))))))

(defn- ad-train
  "Training loop for compiled-generate score functions.
   Uses mx/value-and-grad for exact AD gradients (not compiled, not finite-diff).
   Per-iteration mx/materialize! since mx/compile-fn can't wrap ChoiceMap ops.

   score-fn: (fn [params-tensor] -> MLX scalar) — objective to MAXIMIZE.
   init-params: [D] MLX array.
   opts: same as compiled-train.

   Returns {:params final-params :loss-history [numbers...]}."
  [score-fn init-params
   {:keys [iterations lr beta1 beta2 epsilon callback log-every]
    :or {iterations 1000 lr 0.001 beta1 0.9 beta2 0.999
         epsilon 1e-8 log-every 50}}]
  (let [d (mx/shape init-params)
        neg-score (fn [p] (mx/negative (score-fn p)))
        vg (mx/value-and-grad neg-score)
        beta1-s (mx/scalar beta1)
        beta2-s (mx/scalar beta2)
        one-minus-b1-s (mx/scalar (- 1.0 beta1))
        one-minus-b2-s (mx/scalar (- 1.0 beta2))
        lr-s (mx/scalar lr)
        eps-s (mx/scalar epsilon)]
    (loop [i 0
           params init-params
           m (mx/zeros d)
           v (mx/zeros d)
           losses (transient [])]
      (if (>= i iterations)
        (do
          (mx/materialize! params)
          {:params params :loss-history (persistent! losses)})
        (let [[loss grad] (vg params)
              _ (mx/materialize! loss grad)

              ;; Adam moment updates (host-side bias correction)
              t (double (inc i))
              new-m (mx/add (mx/multiply beta1-s m)
                            (mx/multiply one-minus-b1-s grad))
              new-v (mx/add (mx/multiply beta2-s v)
                            (mx/multiply one-minus-b2-s (mx/square grad)))
              m-hat (mx/divide new-m (mx/scalar (- 1.0 (js/Math.pow beta1 t))))
              v-hat (mx/divide new-v (mx/scalar (- 1.0 (js/Math.pow beta2 t))))
              update-vec (mx/divide m-hat
                                    (mx/add (mx/sqrt v-hat) eps-s))
              new-params (mx/subtract params (mx/multiply lr-s update-vec))

              _ (mx/materialize! new-params new-m new-v)

              ;; Log loss at boundaries
              log? (or (zero? i) (zero? (mod (inc i) log-every)))
              loss-val (mx/item loss)
              losses' (if log?
                        (do
                          (when callback
                            (callback {:iter i :loss loss-val :params new-params}))
                          (conj! losses loss-val))
                        losses)]

          ;; Periodic cleanup
          (when (and (pos? i) (zero? (mod i 50)))
            (mx/clear-cache!)
            (mx/sweep-dead-arrays!))

          (recur (inc i) new-params new-m new-v losses'))))))

(defn make-compiled-loss-grad
  "Build a compiled loss+gradient function for parameter learning.

   Tries three paths in priority order:
   1. Tensor-native score (L2) — pure MLX ops, no GFI. Wraps in
      mx/compile-fn(mx/value-and-grad(negative(score-fn))). Fully compiled.
   2. Compiled-generate score — compiled generate as score fn. Uses
      mx/value-and-grad for exact AD gradients (no mx/compile-fn).
   3. Handler-based score (L0) — full GFI via u/make-score-fn. Uses
      finite-difference gradients (GFI not differentiable through MLX AD).

   Automatically filters out L3/3.5 analytically eliminated addresses.

   model: gen fn with schema
   args: model arguments (vector)
   observations: ChoiceMap of observed values
   addresses: vector of latent addresses to optimize

   Returns:
     {:loss-grad-fn       (fn [params-tensor] -> [loss grad])
      :score-fn           (fn [params-tensor] -> scalar)
      :init-params        [K] MLX array
      :n-params           int
      :compilation-level  :tensor-native | :compiled-generate | :handler
      :latent-index       {addr -> int}}"
  [model args observations addresses]
  (let [model-keyed (dyn/auto-key model)
        ;; L3.5: filter out analytically eliminated addresses
        eliminated (u/get-eliminated-addresses model)
        addresses (u/filter-addresses addresses eliminated)
        ;; Try tensor-native → compiled-generate → handler
        {:keys [score-fn latent-index tensor-native? compiled-generate?]}
        (u/make-tensor-score-fn model args observations addresses)
        ;; Get initial trace for extracting init-params
        {:keys [trace]} (p/generate model-keyed args observations)]
    (if tensor-native?
      ;; Path 1: Tensor-native — fully compiled gradient
      (let [init-params (u/extract-params-by-index trace latent-index)
            n-params (count latent-index)
            neg-score (fn [params] (mx/negative (score-fn params)))
            vg (mx/value-and-grad neg-score)
            compiled-vg (mx/compile-fn vg)]
        {:loss-grad-fn compiled-vg
         :score-fn score-fn
         :init-params init-params
         :n-params n-params
         :compilation-level :tensor-native
         :latent-index latent-index})
      (if compiled-generate?
        ;; Path 2: Compiled-generate — exact AD, not compiled
        (let [init-params (u/extract-params trace addresses)
              n-params (first (mx/shape init-params))
              neg-score (fn [params] (mx/negative (score-fn params)))
              vg (mx/value-and-grad neg-score)]
          {:loss-grad-fn vg
           :score-fn score-fn
           :init-params init-params
           :n-params n-params
           :compilation-level :compiled-generate
           :latent-index latent-index})
        ;; Path 3: Handler-based — finite-difference gradients
        (let [layout (u/compute-param-layout trace addresses)
              init-params (u/extract-params trace addresses layout)
              n-params (:total-size layout)
              gfi-score-fn (u/make-score-fn model args observations addresses layout)
              neg-score (fn [params] (mx/negative (gfi-score-fn params)))
              loss-grad (fn [params]
                          (let [loss (neg-score params)
                                grad (finite-diff-grad neg-score params 1e-4)]
                            (mx/materialize! loss grad)
                            [loss grad]))]
          {:loss-grad-fn loss-grad
           :score-fn gfi-score-fn
           :init-params init-params
           :n-params n-params
           :compilation-level :handler
           :latent-index latent-index})))))

(defn learn
  "Learn model parameters via compiled gradient optimization.

   model: gen fn with schema
   args: model arguments
   observations: ChoiceMap of observed values
   addresses: vector of latent addresses to optimize
   opts:
     :iterations  — number of training steps (default 1000)
     :lr          — learning rate (default 0.01)
     :beta1       — Adam beta1 (default 0.9)
     :beta2       — Adam beta2 (default 0.999)
     :epsilon     — Adam epsilon (default 1e-8)
     :log-every   — log loss every N iterations (default 50)
     :callback    — (fn [{:iter :loss :params}]) called at log boundaries
     :fd-h        — finite-diff step for handler path (default 1e-4)

   Returns:
     {:params           [K] MLX array of optimized parameters
      :loss-history     [numbers...]
      :compilation-level :tensor-native | :handler
      :latent-index     {addr -> int}
      :n-params         int}"
  [model args observations addresses
   {:keys [iterations lr log-every callback]
    :or {iterations 1000 lr 0.01 log-every 50}
    :as opts}]
  (let [{:keys [score-fn init-params n-params compilation-level latent-index]}
        (make-compiled-loss-grad model args observations addresses)
        train-opts (merge {:iterations iterations :lr lr :log-every log-every
                           :callback callback}
                          (select-keys opts [:beta1 :beta2 :epsilon :fd-h]))
        result (case compilation-level
                 :tensor-native (compiled-train score-fn init-params train-opts)
                 :compiled-generate (ad-train score-fn init-params train-opts)
                 :handler (handler-train score-fn init-params train-opts))]
    (assoc result
           :compilation-level compilation-level
           :latent-index latent-index
           :n-params n-params)))

;; ===========================================================================
;; WP-2: Fused Inference + Optimization
;; ===========================================================================

(defn- make-fused-mcmc-opt-step
  "Build a compiled single fused MCMC + Adam step.

   score-fn: (fn [params-tensor] -> MLX scalar) — objective to maximize.
   chain-fn: differentiable MH chain (fn [params noise uniforms] -> final-params).
   opts: {:lr :beta1 :beta2 :epsilon} — Adam hyperparameters.

   Returns a compiled function:
     (fn [params adam-m adam-v t-scalar noise uniforms]
         -> #js [new-params new-m new-v loss])

   Noise [T,K] and uniforms [T] are generated host-side each iteration.
   The compiled function fuses: chain -> score -> grad -> Adam into one
   Metal dispatch."
  [score-fn chain-fn {:keys [lr beta1 beta2 epsilon]
                      :or {lr 0.01 beta1 0.9 beta2 0.999 epsilon 1e-8}}]
  (let [;; Objective: run chain, return negative final score (minimize)
        neg-obj (fn [params noise uniforms]
                  (mx/negative (score-fn (chain-fn params noise uniforms))))
        vg (mx/value-and-grad neg-obj)

        ;; Pre-create scalar constants
        lr-s (mx/scalar lr)
        beta1-s (mx/scalar beta1)
        beta2-s (mx/scalar beta2)
        eps-s (mx/scalar epsilon)
        one-s (mx/scalar 1.0)
        one-minus-b1 (mx/scalar (- 1.0 beta1))
        one-minus-b2 (mx/scalar (- 1.0 beta2))

        step-fn
        (fn [params m v t-scalar noise uniforms]
          (let [[loss grad] (vg params noise uniforms)
                new-m (mx/add (mx/multiply beta1-s m)
                              (mx/multiply one-minus-b1 grad))
                new-v (mx/add (mx/multiply beta2-s v)
                              (mx/multiply one-minus-b2 (mx/square grad)))
                m-hat (mx/divide new-m
                                 (mx/subtract one-s (mx/power beta1-s t-scalar)))
                v-hat (mx/divide new-v
                                 (mx/subtract one-s (mx/power beta2-s t-scalar)))
                update-vec (mx/divide m-hat (mx/add (mx/sqrt v-hat) eps-s))
                new-params (mx/subtract params (mx/multiply lr-s update-vec))]
            #js [new-params new-m new-v loss]))]

    (mx/compile-fn step-fn)))

(defn make-fused-mcmc-train
  "Fused MCMC + optimization: each iteration runs a differentiable MH chain
   then takes a gradient step on the final score.

   The differentiable chain explores the posterior via T MH steps with
   pre-generated noise. mx/value-and-grad differentiates through the chain
   (mx/where has straight-through gradient). Adam updates the parameters.

   model: gen fn with schema
   args: model arguments
   observations: ChoiceMap of observed values
   addresses: vector of latent addresses to optimize
   opts:
     :iterations   — number of training steps (default 500)
     :lr           — learning rate (default 0.01)
     :beta1        — Adam beta1 (default 0.9)
     :beta2        — Adam beta2 (default 0.999)
     :epsilon      — Adam epsilon (default 1e-8)
     :mcmc-steps   — MH steps per iteration (default 10)
     :proposal-std — random walk step size (default 0.1)
     :log-every    — log loss every N iterations (default 50)
     :callback     — (fn [{:iter :loss :params}]) called at log boundaries
     :key          — PRNG key

   Returns {:params           [K] MLX array of optimized parameters
            :loss-history     [numbers...]
            :compilation-level :tensor-native | :handler
            :mcmc-compiled    true
            :latent-index     {addr -> int}
            :n-params         int}"
  [model args observations addresses
   {:keys [iterations lr beta1 beta2 epsilon mcmc-steps proposal-std
           log-every callback key]
    :or {iterations 500 lr 0.01 beta1 0.9 beta2 0.999 epsilon 1e-8
         mcmc-steps 10 proposal-std 0.1 log-every 50}}]
  (let [model-keyed (dyn/auto-key model)
        ;; L3.5: filter analytically eliminated addresses
        eliminated (u/get-eliminated-addresses model)
        addresses (u/filter-addresses addresses eliminated)
        ;; Build score function + initial params
        {:keys [trace]} (p/generate model-keyed args observations)
        {:keys [score-fn init-params n-params tensor-native? latent-index]}
        (u/prepare-mcmc-score model args observations addresses trace)
        _ (mx/materialize! init-params)
        K n-params
        T mcmc-steps
        std-arr (mx/scalar proposal-std)
        compilation-level (if tensor-native? :tensor-native :handler)

        ;; Build differentiable chain once (structure fixed, values change)
        chain-fn (cg/make-differentiable-chain score-fn std-arr T K)

        ;; Build compiled fused step
        opt-step (make-fused-mcmc-opt-step
                  score-fn chain-fn
                  {:lr lr :beta1 beta1 :beta2 beta2 :epsilon epsilon})

        ;; Warm-up: trace once to cache the Metal program
        rk (rng/ensure-key key)
        _ (let [n0 (rng/normal (rng/fresh-key) [T K])
                u0 (rng/uniform (rng/fresh-key) [T])]
            (mx/materialize! n0 u0)
            (mx/materialize! (aget (opt-step (mx/zeros [K]) (mx/zeros [K])
                                             (mx/zeros [K]) (mx/scalar 1.0)
                                             n0 u0) 0)))]
    ;; Training loop: noise generated host-side, compiled step does the rest
    (loop [i 0
           params init-params
           m (mx/zeros [K])
           v (mx/zeros [K])
           losses (transient [])
           rk rk]
      (if (>= i iterations)
        (do
          (mx/materialize! params)
          {:params params
           :loss-history (persistent! losses)
           :compilation-level compilation-level
           :mcmc-compiled true
           :latent-index latent-index
           :n-params n-params})
        (let [;; Fresh noise each iteration (host-side PRNG)
              [nk uk rk'] (rng/split-n rk 3)
              noise (rng/normal nk [T K])
              uniforms (rng/uniform uk [T])
              _ (mx/materialize! noise uniforms)

              ;; Compiled step: chain + grad + Adam
              t-scalar (mx/scalar (double (inc i)))
              result (opt-step params m v t-scalar noise uniforms)
              np (aget result 0)
              nm (aget result 1)
              nv (aget result 2)
              loss-arr (aget result 3)

              ;; Only materialize loss at log boundaries
              log? (or (zero? i) (zero? (mod (inc i) log-every)))
              losses' (if log?
                        (do
                          (mx/materialize! loss-arr)
                          (let [loss-val (mx/item loss-arr)]
                            (when callback
                              (callback {:iter i :loss loss-val :params np}))
                            (conj! losses loss-val)))
                        losses)]

          ;; Periodic cleanup every 50 iterations
          (when (and (pos? i) (zero? (mod i 50)))
            (mx/clear-cache!)
            (mx/sweep-dead-arrays!))

          (recur (inc i) np nm nv losses' rk'))))))

(defn fused-learn
  "Learn parameters via fused inference + optimization.

   Dispatches to the right fused path based on method:
   - :direct  — tensor-native gradient descent (WP-1 learn)
   - :mcmc    — differentiable MH chain + Adam (WP-2 make-fused-mcmc-train)
   - :vi      — variational inference (delegates to vi/vi)

   Falls back to WP-1's learn for :direct.

   model: gen fn with schema
   args: model arguments
   observations: ChoiceMap of observed values
   addresses: vector of latent addresses to optimize
   method: :direct | :mcmc | :vi
   opts: method-specific options (see individual functions)

   Returns {:params :loss-history :compilation-level ...}"
  [model args observations addresses method
   {:keys [] :as opts}]
  (case method
    :direct
    (learn model args observations addresses opts)

    :mcmc
    (make-fused-mcmc-train model args observations addresses opts)

    :vi
    (let [eliminated (u/get-eliminated-addresses model)
          filtered (u/filter-addresses addresses eliminated)
          model-keyed (dyn/auto-key model)
          score-fn (u/make-score-fn model-keyed args observations filtered)
          ;; Bypass mx/vmap — use loop-over-rows like vi-from-model (b5c923d fix).
          ;; mx/vmap crashes on GFI score functions that use volatile! internally.
          vec-score-fn (fn [samples]
                         (let [n (first (mx/shape samples))]
                           (mx/stack (mapv (fn [i] (score-fn (mx/index samples i)))
                                           (range n)))))
          {:keys [trace]} (p/generate model-keyed args observations)
          init-params (u/extract-params trace filtered)
          vi-opts (merge {:iterations (or (:iterations opts) 500)
                          :learning-rate (or (:lr opts) 0.01)
                          :vectorized-log-density vec-score-fn}
                         (select-keys opts [:elbo-samples :beta1 :beta2
                                            :epsilon :callback :key]))
          vi-result (vi/vi vi-opts score-fn init-params)]
      (assoc vi-result
             :compilation-level :vi
             :method :vi))

    ;; Default: try direct first
    (learn model args observations addresses opts)))
