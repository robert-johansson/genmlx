(ns genmlx.compiled
  "Compiled execution paths for GenMLX.

   Level 0: Compiled unfold loops and particle filters (temporal models).
   Level 1: Compiled gen functions — static models (L1-M2), partial prefix (L1-M3).

   Level 0 pattern: user provides a pure MLX step-fn, we pre-generate noise
   and unroll the loop. All randomness injected as input tensors.

   Level 1 pattern: schema → noise-transform-based pure function → mx/compile-fn.
   Distribution-specific noise transforms bypass multimethod dispatch,
   enabling fusion into a single Metal kernel. No mutable state, no handler."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.vectorized :as vec]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.handler :as h]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Core: compiled unfold step loop
;; ---------------------------------------------------------------------------

(defn make-compiled-unfold
  "Build a compiled T-step unfold as one Metal dispatch.
   step-fn: (fn [state noise-row] -> [new-state score]) — pure MLX ops only.
     state:     [state-dim] MLX array
     noise-row: [noise-dim] MLX array (one row of pre-generated noise)
     Returns:   [new-state, score] where new-state is [state-dim], score is scalar.
   n-steps:   number of timesteps T
   state-dim: dimension of state vector
   noise-dim: noise needed per step

   Returns a compiled fn: (init-state [state-dim], noise [T, noise-dim])
                         -> [final-state [state-dim], states [T, state-dim], total-score scalar]

   The returned function is warmed up (Metal program cached on first call)."
  [step-fn n-steps state-dim noise-dim]
  (let [unfold-fn
        (fn [init-state noise-2d]
          (loop [t 0, state init-state, score (mx/scalar 0.0), states []]
            (if (>= t n-steps)
              [state (mx/stack states) score]
              (let [noise-row (mx/reshape
                               (mx/take-idx noise-2d (mx/array [t] mx/int32) 0)
                               [noise-dim])
                    [new-state step-score] (step-fn state noise-row)]
                (recur (inc t)
                       new-state
                       (mx/add score step-score)
                       (conj states new-state))))))
        compiled (mx/compile-fn unfold-fn)]
    ;; Warm up: trace with dummy data to cache Metal program
    (let [dummy-state (mx/zeros [state-dim])
          dummy-noise (mx/zeros [n-steps noise-dim])]
      (let [[s states sc] (compiled dummy-state dummy-noise)]
        (mx/materialize! s states sc)))
    compiled))

(defn make-compiled-unfold-generate
  "Build a compiled T-step unfold with observations (generate mode).
   step-fn: (fn [state noise-row obs-row] -> [new-state score weight])
     state:     [state-dim] MLX array
     noise-row: [noise-dim] MLX array
     obs-row:   [obs-dim] MLX array (observations at this timestep)
     Returns:   [new-state, score, weight] — score is total log-prob,
                weight is incremental importance weight for this step.
   n-steps:   number of timesteps T
   state-dim: dimension of state vector
   noise-dim: noise per step
   obs-dim:   observation dimension per step

   Returns compiled fn: (init-state, noise [T,noise-dim], obs [T,obs-dim])
                       -> [final-state, states [T,state-dim], total-score, total-weight]"
  [step-fn n-steps state-dim noise-dim obs-dim]
  (let [unfold-fn
        (fn [init-state noise-2d obs-2d]
          (loop [t 0, state init-state
                 score (mx/scalar 0.0), weight (mx/scalar 0.0)
                 states []]
            (if (>= t n-steps)
              [state (mx/stack states) score weight]
              (let [noise-row (mx/reshape
                               (mx/take-idx noise-2d (mx/array [t] mx/int32) 0)
                               [noise-dim])
                    obs-row (mx/reshape
                             (mx/take-idx obs-2d (mx/array [t] mx/int32) 0)
                             [obs-dim])
                    [new-state step-score step-weight] (step-fn state noise-row obs-row)]
                (recur (inc t)
                       new-state
                       (mx/add score step-score)
                       (mx/add weight step-weight)
                       (conj states new-state))))))
        compiled (mx/compile-fn unfold-fn)]
    ;; Warm up
    (let [dummy-state (mx/zeros [state-dim])
          dummy-noise (mx/zeros [n-steps noise-dim])
          dummy-obs (mx/zeros [n-steps obs-dim])]
      (let [[s states sc w] (compiled dummy-state dummy-noise dummy-obs)]
        (mx/materialize! s states sc w)))
    compiled))

;; ---------------------------------------------------------------------------
;; High-level API: simulate and generate with Trace output
;; ---------------------------------------------------------------------------

(defn compiled-unfold-simulate
  "Run a compiled unfold and return a standard Trace.
   step-fn:    pure MLX step function (see make-compiled-unfold)
   n-steps:    number of timesteps
   state-dim:  state vector dimension
   noise-dim:  noise per step
   init-state: [state-dim] MLX array
   key:        PRNG key for noise generation
   addr-fn:    (fn [t] -> keyword) maps timestep to trace address (default: identity int)

   Returns: Trace with choices at addr-fn(t) for each timestep."
  [{:keys [step-fn n-steps state-dim noise-dim addr-fn]
    :or {addr-fn identity}}
   init-state key]
  (let [key (rng/ensure-key key)
        compiled (make-compiled-unfold step-fn n-steps state-dim noise-dim)
        noise (rng/normal key [n-steps noise-dim])
        [final-state states total-score] (compiled init-state noise)
        _ (mx/materialize! final-state states total-score)
        ;; Build choicemap from states tensor
        choices (reduce
                 (fn [cm t]
                   (let [state-t (mx/reshape
                                  (mx/take-idx states (mx/array [t] mx/int32) 0)
                                  [state-dim])]
                     (cm/set-value cm (addr-fn t) state-t)))
                 cm/EMPTY
                 (range n-steps))]
    (tr/make-trace {:gen-fn nil :args [n-steps init-state]
                    :choices choices
                    :retval {:final-state final-state :states states}
                    :score total-score})))

(defn compiled-unfold-generate
  "Run a compiled unfold with observations and return {:trace Trace :weight scalar}.
   step-fn:    pure MLX step function with obs (see make-compiled-unfold-generate)
   n-steps:    number of timesteps
   state-dim:  state vector dimension
   noise-dim:  noise per step
   obs-dim:    observation dimension per step
   init-state: [state-dim] MLX array
   obs:        [T, obs-dim] MLX array of observations
   key:        PRNG key
   addr-fn:    (fn [t] -> keyword) maps timestep to trace address

   Returns: {:trace Trace :weight scalar}"
  [{:keys [step-fn n-steps state-dim noise-dim obs-dim addr-fn]
    :or {addr-fn identity}}
   init-state obs key]
  (let [key (rng/ensure-key key)
        compiled (make-compiled-unfold-generate step-fn n-steps state-dim noise-dim obs-dim)
        noise (rng/normal key [n-steps noise-dim])
        [final-state states total-score total-weight] (compiled init-state noise obs)
        _ (mx/materialize! final-state states total-score total-weight)
        ;; Build choicemap from states tensor
        choices (reduce
                 (fn [cm t]
                   (let [state-t (mx/reshape
                                  (mx/take-idx states (mx/array [t] mx/int32) 0)
                                  [state-dim])]
                     (cm/set-value cm (addr-fn t) state-t)))
                 cm/EMPTY
                 (range n-steps))]
    {:trace (tr/make-trace {:gen-fn nil :args [n-steps init-state]
                            :choices choices
                            :retval {:final-state final-state :states states}
                            :score total-score})
     :weight total-weight}))

;; ---------------------------------------------------------------------------
;; Convenience: compile from a step specification
;; ---------------------------------------------------------------------------

(defn make-gaussian-step
  "Build a step-fn for a Gaussian transition model.
   transition-fn: (fn [state] -> [mean, std]) — pure MLX ops.
   Returns a step-fn suitable for compiled unfold.
   noise-dim = state-dim (one noise per state dimension)."
  [transition-fn state-dim]
  (fn [state noise-row]
    (let [[mean std] (transition-fn state)
          new-state (mx/add mean (mx/multiply std noise-row))
          ;; Log-prob: -0.5 * sum((new - mean)^2 / std^2) - sum(log(std)) - D/2*log(2π)
          diff (mx/subtract new-state mean)
          log-prob (mx/subtract
                    (mx/subtract
                     (mx/multiply (mx/scalar -0.5)
                                  (mx/sum (mx/divide (mx/multiply diff diff)
                                                     (mx/multiply std std))))
                     (mx/sum (mx/log std)))
                    (mx/scalar (* 0.5 state-dim (js/Math.log (* 2 js/Math.PI)))))]
      [new-state log-prob])))

(defn make-gaussian-step-with-obs
  "Build a step-fn with Gaussian observations for compiled unfold generate.
   transition-fn: (fn [state] -> [mean, std]) for latent state transition.
   observation-fn: (fn [state] -> [obs-mean, obs-std]) for observation model.
   Returns step-fn: (state, noise, obs) -> [new-state, score, weight]."
  [transition-fn observation-fn state-dim obs-dim]
  (fn [state noise-row obs-row]
    (let [;; Transition
          [t-mean t-std] (transition-fn state)
          new-state (mx/add t-mean (mx/multiply t-std noise-row))
          ;; Transition log-prob
          t-diff (mx/subtract new-state t-mean)
          t-lp (mx/subtract
                (mx/subtract
                 (mx/multiply (mx/scalar -0.5)
                              (mx/sum (mx/divide (mx/multiply t-diff t-diff)
                                                 (mx/multiply t-std t-std))))
                 (mx/sum (mx/log t-std)))
                (mx/scalar (* 0.5 state-dim (js/Math.log (* 2 js/Math.PI)))))
          ;; Observation log-prob
          [o-mean o-std] (observation-fn new-state)
          o-diff (mx/subtract obs-row o-mean)
          o-lp (mx/subtract
                (mx/subtract
                 (mx/multiply (mx/scalar -0.5)
                              (mx/sum (mx/divide (mx/multiply o-diff o-diff)
                                                 (mx/multiply o-std o-std))))
                 (mx/sum (mx/log o-std)))
                (mx/scalar (* 0.5 obs-dim (js/Math.log (* 2 js/Math.PI)))))
          ;; score = transition + observation, weight = observation
          score (mx/add t-lp o-lp)]
      [new-state score o-lp])))

;; ===========================================================================
;; Tier 2c: Compiled Inference Graphs
;; ===========================================================================
;; Full inference sweeps compiled into single Metal dispatches.
;; All randomness pre-generated, no materialization between steps.

;; ---------------------------------------------------------------------------
;; Compiled bootstrap particle filter
;; ---------------------------------------------------------------------------

(defn make-compiled-particle-filter
  "Build a compiled T-step bootstrap particle filter as one Metal dispatch.

   particle-step-fn: (fn [states noise obs-row] -> [new-states, log-weights])
     states:     [N, state-dim] MLX array (batched particle states)
     noise:      [N, noise-dim] MLX array (per-particle noise for this step)
     obs-row:    [obs-dim] MLX array (observation at this timestep)
     Returns:    [new-states [N,state-dim], log-weights [N]] where log-weights
                 are the importance weights for this step (typically observation
                 log-probs under a bootstrap filter).

   n-steps:    number of timesteps T
   n-particles: number of particles N
   state-dim:  state vector dimension
   noise-dim:  noise per particle per step
   obs-dim:    observation dimension per step

   Returns compiled fn:
     (init-states [N,D], noise [T,N,noise-dim], obs [T,obs-dim], uniforms [T,1])
     -> [final-states [N,D], log-ml scalar]

   The uniforms [T,1] are used for deterministic systematic resampling at each step.
   Pre-compile warms up the Metal program cache."
  [particle-step-fn n-steps n-particles state-dim noise-dim obs-dim]
  (let [pf-fn
        (fn [init-states noise-3d obs-2d uniforms-2d]
          (loop [t 0
                 states init-states
                 log-ml (mx/scalar 0.0)]
            (if (>= t n-steps)
              [states log-ml]
              (let [;; Extract noise for this step: [N, noise-dim]
                    noise-t (mx/reshape
                             (mx/take-idx noise-3d (mx/array [t] mx/int32) 0)
                             [n-particles noise-dim])
                    ;; Extract observation: [obs-dim]
                    obs-t (mx/reshape
                           (mx/take-idx obs-2d (mx/array [t] mx/int32) 0)
                           [obs-dim])
                    ;; Extract uniform for resampling: [1]
                    u0 (mx/reshape
                        (mx/take-idx uniforms-2d (mx/array [t] mx/int32) 0)
                        [1])
                    ;; 1. Extend particles
                    [new-states log-weights] (particle-step-fn states noise-t obs-t)
                    ;; 2. Log-ML increment: logsumexp(w) - log(N)
                    ml-inc (mx/subtract (mx/logsumexp log-weights)
                                        (mx/scalar (js/Math.log n-particles)))
                    ;; 3. Resample (deterministic with pre-generated u0)
                    indices (vec/systematic-resample-indices-deterministic
                             log-weights n-particles u0)
                    resampled (mx/take-idx new-states indices 0)]
                (recur (inc t)
                       resampled
                       (mx/add log-ml ml-inc))))))
        compiled (mx/compile-fn pf-fn)]
    ;; Warm up
    (let [dummy-states (mx/zeros [n-particles state-dim])
          dummy-noise (mx/zeros [n-steps n-particles noise-dim])
          dummy-obs (mx/zeros [n-steps obs-dim])
          dummy-u (mx/ones [n-steps 1])]
      (let [[s ml] (compiled dummy-states dummy-noise dummy-obs dummy-u)]
        (mx/materialize! s ml)))
    compiled))

(defn compiled-particle-filter
  "Run a compiled bootstrap particle filter.

   particle-step-fn: (fn [states [N,D], noise [N,K], obs [M]]
                       -> [new-states [N,D], log-weights [N]])
   n-steps:    T
   n-particles: N
   state-dim:  D
   noise-dim:  K (noise per particle per step)
   obs-dim:    M
   init-states: [N, D] initial particle states
   obs:         [T, M] observation tensor
   key:         PRNG key

   Returns {:final-states [N,D], :log-ml scalar (JS number)}"
  [{:keys [particle-step-fn n-steps n-particles state-dim noise-dim obs-dim]}
   init-states obs key]
  (let [key (rng/ensure-key key)
        [noise-key uniform-key] (rng/split key)
        compiled (make-compiled-particle-filter
                  particle-step-fn n-steps n-particles state-dim noise-dim obs-dim)
        noise (rng/normal noise-key [n-steps n-particles noise-dim])
        uniforms (rng/uniform uniform-key [n-steps 1])
        [final-states log-ml] (compiled init-states noise obs uniforms)]
    (mx/materialize! final-states log-ml)
    {:final-states final-states
     :log-ml (mx/item log-ml)}))

;; ---------------------------------------------------------------------------
;; Convenience: Gaussian bootstrap particle filter
;; ---------------------------------------------------------------------------

(defn make-gaussian-particle-step
  "Build a particle-step-fn for a linear-Gaussian state-space model.
   transition-fn: (fn [states [N,D]] -> [mean [N,D], std [D]])
   observation-fn: (fn [states [N,D]] -> [obs-mean [N,M], obs-std [M]])
   Returns particle-step-fn for compiled-particle-filter."
  [transition-fn observation-fn state-dim obs-dim]
  (fn [states noise obs-row]
    (let [;; Transition: states [N,D], noise [N,D]
          [t-mean t-std] (transition-fn states)
          new-states (mx/add t-mean (mx/multiply t-std noise))
          ;; Observation log-prob per particle
          [o-mean o-std] (observation-fn new-states)
          ;; obs-row is [M], o-mean is [N,M] — broadcast subtraction
          o-diff (mx/subtract obs-row o-mean)
          ;; Per-particle log-prob: sum over obs dimensions
          log-weights (mx/subtract
                       (mx/subtract
                        (mx/multiply (mx/scalar -0.5)
                                     (mx/sum (mx/divide (mx/multiply o-diff o-diff)
                                                        (mx/multiply o-std o-std)
                                                        1))) ;; sum along obs dim
                        (mx/sum (mx/log o-std))) ;; this is scalar, broadcasts
                       (mx/scalar (* 0.5 obs-dim (js/Math.log (* 2 js/Math.PI)))))]
      [new-states log-weights])))

;; ===========================================================================
;; Level 1-M2: Compiled Gen Functions for Static Models
;; ===========================================================================
;;
;; Architecture: schema → noise-transform pure function → mx/compile-fn
;;
;;   Schema (from gen macro)
;;     │ trace-sites, dist-types, dist-args, dep-order
;;     ▼
;;   make-compiled-simulate
;;     │ builds pure fn using noise transforms (not multimethod dispatch)
;;     │ wraps in mx/compile-fn → single Metal kernel
;;     ▼
;;   DynamicGF.simulate (in dynamic.cljs)
;;     │ dispatches: compiled path or handler fallback
;;     │ builds Trace + ChoiceMap from compiled result
;;     ▼
;;   Trace (shared data type)

;; ---------------------------------------------------------------------------
;; Function resolution: source-form symbols → actual functions
;; ---------------------------------------------------------------------------

(def ^:private mx-fns
  "MLX function name → actual function."
  {"add" mx/add "subtract" mx/subtract "multiply" mx/multiply
   "divide" mx/divide "negative" mx/negative "scalar" mx/scalar
   "log" mx/log "exp" mx/exp "sqrt" mx/sqrt "abs" mx/abs
   "power" mx/power "square" mx/square "sum" mx/sum "mean" mx/mean
   "maximum" mx/maximum "minimum" mx/minimum "where" mx/where
   "sigmoid" mx/sigmoid "tanh" mx/tanh "sin" mx/sin "cos" mx/cos
   "reshape" mx/reshape "matmul" mx/matmul "logaddexp" mx/logaddexp
   "log1p" mx/log1p "expm1" mx/expm1 "reciprocal" mx/reciprocal
   "floor" mx/floor "ceil" mx/ceil "clip" mx/clip
   "softmax" mx/softmax "ensure-array" mx/ensure-array
   "array" mx/array "zeros" mx/zeros "ones" mx/ones
   "stack" mx/stack "concatenate" mx/concatenate
   "transpose" mx/transpose "inner" mx/inner "outer" mx/outer
   "sign" mx/sign "tan" mx/tan "less-equal" mx/less-equal
   "less" mx/less "equal" mx/equal "lgamma" mx/lgamma
   "index" mx/index "take-idx" mx/take-idx "logsumexp" mx/logsumexp})

(def ^:private cljs-fns
  "ClojureScript core math → MLX equivalents."
  {"+" mx/add "-" mx/subtract "*" mx/multiply "/" mx/divide})

(defn- resolve-fn
  "Resolve a namespace-qualified symbol to an actual function.
   Handles mx/ alias, genmlx.mlx/ full, and unqualified ClojureScript ops."
  [sym]
  (let [ns-part (namespace sym)
        n (name sym)]
    (cond
      (or (= ns-part "mx") (= ns-part "genmlx.mlx")) (get mx-fns n)
      (nil? ns-part) (get cljs-fns n)
      :else nil)))

;; ---------------------------------------------------------------------------
;; Binding environment: walk source → symbol resolution map
;; ---------------------------------------------------------------------------

(defn- trace-call?
  "Is form a (trace :addr ...) call?"
  [form]
  (and (seq? form) (seq form)
       (symbol? (first form))
       (= "trace" (name (first form)))
       (keyword? (second form))))

(declare walk-binding-forms)

(defn- walk-binding-form
  "Walk a single form, collecting let/do bindings into env."
  [env form]
  (cond
    ;; let form: process bindings sequentially, then walk body
    (and (seq? form) (seq form) (symbol? (first form))
         (= "let" (name (first form)))
         (vector? (second form)))
    (let [pairs (partition 2 (second form))
          env' (reduce
                (fn [env [sym val-form]]
                  (if (symbol? sym)
                    (assoc env (name sym)
                           (if (trace-call? val-form)
                             {:kind :trace :addr (second val-form)}
                             {:kind :expr :form val-form}))
                    env))
                env pairs)]
      (walk-binding-forms env' (drop 2 form)))

    ;; do form
    (and (seq? form) (seq form) (symbol? (first form))
         (= "do" (name (first form))))
    (walk-binding-forms env (rest form))

    ;; Other sequences: walk children for nested lets
    (seq? form) (walk-binding-forms env form)
    (vector? form) (walk-binding-forms env form)
    :else env))

(defn- walk-binding-forms [env forms]
  (reduce walk-binding-form env forms))

(defn build-binding-env
  "Walk gen source form, build symbol → resolution map.
   source: (params-vec body-forms...)
   Returns: {name-string → {:kind :param/:trace/:expr ...}}"
  [source]
  (let [params (first source)
        param-env (into {} (map-indexed
                            (fn [i p] [(name p) {:kind :param :index i}])
                            params))]
    (walk-binding-forms param-env (rest source))))

;; ---------------------------------------------------------------------------
;; Expression compiler: source form → pure closure
;; ---------------------------------------------------------------------------

(defn compile-expr
  "Compile a source form into (fn [values-map args-vec] -> value).
   Returns nil if the form contains unsupported constructs.
   binding-env: {name-string → {:kind :param/:trace/:expr ...}}
   visited: set of symbol names for cycle detection on :expr bindings."
  [form binding-env visited]
  (cond
    (number? form) (fn [_v _a] form)
    (boolean? form) (fn [_v _a] form)
    (keyword? form) (fn [_v _a] form)
    (string? form) (fn [_v _a] form)
    (nil? form) (fn [_v _a] nil)

    ;; Symbol → resolve through binding env, then try closed-over var
    (symbol? form)
    (let [info (get binding-env (name form))]
      (case (:kind info)
        :param (let [i (:index info)] (fn [_v a] (nth a i)))
        :trace (let [addr (:addr info)] (fn [v _a] (get v addr)))
        :expr (when-not (contains? visited (name form))
                (compile-expr (:form info) binding-env
                              (conj visited (name form))))
        ;; Not in binding-env: try to resolve as a closed-over var (captured constant)
        (when-let [resolved (try (deref (resolve form)) (catch :default _ nil))]
          (fn [_v _a] resolved))))

    ;; Keyword-as-function: (:key arg) → (get arg :key) — common map access pattern
    (and (seq? form) (seq form) (keyword? (first form)))
    (let [kw (first form)
          carg (compile-expr (second form) binding-env visited)]
      (when carg
        (fn [v a] (get (carg v a) kw))))

    ;; Function call (or trace reference in return position)
    (and (seq? form) (seq form) (symbol? (first form)))
    (let [head-name (name (first form))]
      (cond
        ;; (trace :addr ...) → look up the sampled value
        (and (= head-name "trace") (keyword? (second form)))
        (let [addr (second form)]
          (fn [v _a] (get v addr)))

        ;; Regular function call
        :else
        (let [f (resolve-fn (first form))
              cargs (mapv #(compile-expr % binding-env visited) (rest form))]
          (when (and f (every? some? cargs))
            (case (count cargs)
              0 (fn [_v _a] (f))
              1 (let [c0 (nth cargs 0)]
                  (fn [v a] (f (c0 v a))))
              2 (let [c0 (nth cargs 0) c1 (nth cargs 1)]
                  (fn [v a] (f (c0 v a) (c1 v a))))
              3 (let [c0 (nth cargs 0) c1 (nth cargs 1) c2 (nth cargs 2)]
                  (fn [v a] (f (c0 v a) (c1 v a) (c2 v a))))
              (fn [v a] (apply f (mapv #(% v a) cargs))))))))

    ;; Vector literal
    (vector? form)
    (let [celems (mapv #(compile-expr % binding-env visited) form)]
      (when (every? some? celems)
        (fn [v a] (mapv #(% v a) celems))))

    ;; Map literal — common for Unfold kernel return values {:dep dep :ac ac ...}
    (map? form)
    (let [compiled-entries
          (mapv (fn [[k val-form]]
                  (let [ck (compile-expr k binding-env visited)
                        cv (compile-expr val-form binding-env visited)]
                    (when (and ck cv) [ck cv])))
                form)]
      (when (every? some? compiled-entries)
        (fn [v a] (into {} (mapv (fn [[ck cv]] [(ck v a) (cv v a)]) compiled-entries)))))

    :else nil))

;; ---------------------------------------------------------------------------
;; Noise transform registry
;; ---------------------------------------------------------------------------
;; Each distribution that can be compiled defines:
;;   :noise-fn   — (fn [key] -> noise-array) base random draw
;;   :transform  — (fn [noise arg1 arg2 ...] -> sample) pure MLX ops
;;   :log-prob   — (fn [value arg1 arg2 ...] -> scalar) pure MLX ops
;;
;; Distributions NOT in this map fall back to dc/dist-sample (no compilation).

(def ^:private LOG-2PI-HALF (mx/scalar (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))))
(def ^:private HALF (mx/scalar 0.5))
(def ^:private ONE (mx/scalar 1.0))
(def ^:private ZERO (mx/scalar 0.0))
(def ^:private NEG-INF (mx/scalar ##-Inf))
(def ^:private TWO (mx/scalar 2.0))
(def ^:private LOG-2 (mx/scalar (js/Math.log 2.0)))
(def ^:private LOG-PI (mx/scalar (js/Math.log js/Math.PI)))
(def ^:private MLX-PI (mx/scalar js/Math.PI))

;; ---------------------------------------------------------------------------
;; Arg coercion helpers
;; ---------------------------------------------------------------------------

(defn ensure-mlx-args
  "Ensure all args are MLX arrays. Used by fully-compiled paths (M2/M4)."
  [args-vec]
  (mapv mx/ensure-array args-vec))

(defn ensure-numeric-mlx-args
  "Ensure numeric args are MLX arrays, pass others through.
   Used by prefix-compiled paths (M3) where args may be complex."
  [args-vec]
  (mapv (fn [a]
          (cond (mx/array? a) a
                (number? a) (mx/scalar a)
                :else a))
        args-vec))

(def ^:private noise-transforms
  {:gaussian
   {:noise-fn (fn [key] (rng/normal key []))
    :transform (fn [noise mu sigma] (mx/add mu (mx/multiply sigma noise)))
    :log-prob (fn [v mu sigma]
                (let [z (mx/divide (mx/subtract v mu) sigma)]
                  (mx/negative
                   (mx/add LOG-2PI-HALF (mx/log sigma)
                           (mx/multiply HALF (mx/square z))))))}

   :uniform
   {:noise-fn (fn [key] (rng/uniform key []))
    :transform (fn [noise lo hi]
                 (mx/add lo (mx/multiply (mx/subtract hi lo) noise)))
    :log-prob (fn [v lo hi]
                (mx/where (mx/multiply (mx/less-equal lo v)
                                       (mx/less-equal v hi))
                          (mx/negative (mx/log (mx/subtract hi lo)))
                          NEG-INF))}

   :bernoulli
   {:noise-fn (fn [key] (rng/uniform key []))
    :transform (fn [noise p]
                 (mx/where (mx/less noise p) ONE ZERO))
    :log-prob (fn [v p]
                (mx/add (mx/multiply v (mx/log p))
                        (mx/multiply (mx/subtract ONE v)
                                     (mx/log (mx/subtract ONE p)))))}

   :exponential
   {:noise-fn (fn [key] (rng/uniform key []))
    :transform (fn [noise rate]
                 (mx/divide (mx/negative (mx/log (mx/subtract ONE noise)))
                            rate))
    :log-prob (fn [v rate]
                (mx/subtract (mx/log rate) (mx/multiply rate v)))}

   :log-normal
   {:noise-fn (fn [key] (rng/normal key []))
    :transform (fn [noise mu sigma]
                 (mx/exp (mx/add mu (mx/multiply sigma noise))))
    :log-prob (fn [v mu sigma]
                (let [log-v (mx/log v)
                      z (mx/divide (mx/subtract log-v mu) sigma)]
                  (mx/negative
                   (mx/add LOG-2PI-HALF (mx/log sigma) log-v
                           (mx/multiply HALF (mx/square z))))))}

   :delta
   {:noise-fn nil ;; no randomness
    :transform nil ;; value = first dist-arg
    :log-prob (fn [v param]
                (mx/where (mx/equal v param) ZERO NEG-INF))}

   :laplace
   {:noise-fn (fn [key] (rng/uniform key []))
    :transform (fn [noise loc scale]
                 ;; Inverse CDF: loc - scale * sign(u) * log(1 - 2*|u|)
                 ;; where u = noise - 0.5, matches dist.cljs laplace-icdf
                 (let [u (mx/subtract noise HALF)]
                   (mx/subtract loc
                                (mx/multiply scale
                                             (mx/multiply (mx/sign u)
                                                          (mx/log (mx/subtract ONE (mx/multiply TWO (mx/abs u)))))))))
    :log-prob (fn [v loc scale]
                (mx/subtract
                 (mx/negative (mx/divide (mx/abs (mx/subtract v loc)) scale))
                 (mx/add LOG-2 (mx/log scale))))}

   :cauchy
   {:noise-fn (fn [key] (rng/uniform key []))
    :transform (fn [noise loc scale]
                 ;; Inverse CDF: loc + scale * tan(pi * (u - 0.5))
                 ;; Use sin/cos to match dist.cljs exactly
                 (let [z (mx/subtract noise HALF)
                       pz (mx/multiply MLX-PI z)]
                   (mx/add loc
                           (mx/multiply scale
                                        (mx/divide (mx/sin pz) (mx/cos pz))))))
    :log-prob (fn [v loc scale]
                (let [z (mx/divide (mx/subtract v loc) scale)]
                  (mx/negative
                   (mx/add LOG-PI (mx/log scale)
                           (mx/log (mx/add ONE (mx/square z)))))))}

   :iid-gaussian
   {:noise-fn nil ;; noise shape depends on t (3rd dist-arg)
    :args-noise-fn (fn [eval-args key]
                     (let [t (let [v (nth eval-args 2)]
                               (if (number? v) v (mx/item v)))]
                       (rng/normal key [t])))
    :transform (fn [noise mu sigma _t]
                 (mx/add mu (mx/multiply sigma noise)))
    :log-prob (fn [v mu sigma _t]
                (let [z (mx/divide (mx/subtract v mu) sigma)]
                  (mx/sum
                   (mx/negative
                    (mx/add LOG-2PI-HALF (mx/log sigma)
                            (mx/multiply HALF (mx/square z))))
                   -1)))}})

;; Aliases
(def noise-transforms-full
  (assoc noise-transforms
         :normal (:gaussian noise-transforms)
         :flip (:bernoulli noise-transforms)))

;; ---------------------------------------------------------------------------
;; Return form extraction
;; ---------------------------------------------------------------------------

(defn extract-return-expr
  "Peel through let/do wrappers to find the leaf return expression."
  [form]
  (cond
    (and (seq? form) (seq form) (symbol? (first form))
         (= "let" (name (first form))))
    (extract-return-expr (last (drop 2 form)))

    (and (seq? form) (seq form) (symbol? (first form))
         (= "do" (name (first form))))
    (extract-return-expr (last (rest form)))

    :else form))

;; ---------------------------------------------------------------------------
;; Site spec building (shared by prepare-* and fused compilation)
;; ---------------------------------------------------------------------------

(defn build-fused-site-specs
  "Build site-specs with compiled args for fused compilation.
   Returns vector of {:addr :compiled-args :dist-type} or nil per site."
  [static-sites binding-env]
  (mapv (fn [ts]
          (let [cargs (mapv #(compile-expr % binding-env #{})
                            (:dist-args ts))]
            (when (every? some? cargs)
              {:addr (:addr ts)
               :compiled-args cargs
               :dist-type (:dist-type ts)})))
        static-sites))

;; ---------------------------------------------------------------------------
;; Compiled simulate: schema → noise-transform pure function → mx/compile-fn
;; ---------------------------------------------------------------------------

(defn build-site-step
  "Build the reduce step function for one trace site using noise transforms.
   Returns (fn [{:keys [values score key]} values-map args-vec] -> state)
   or nil if the site can't use noise transforms."
  [site-spec]
  (let [{:keys [addr compiled-args dist-type]} site-spec
        nt (get noise-transforms-full dist-type)]
    (when nt
      (cond
        (:noise-fn nt)
        ;; Standard distribution: generate noise, transform, score
        (let [noise-fn (:noise-fn nt)
              transform-fn (:transform nt)
              log-prob-fn (:log-prob nt)]
          (fn [{:keys [values score key]} args-vec]
            (let [eval-args (mapv #(% values args-vec) compiled-args)
                  [k1 k2] (rng/split key)
                  noise (noise-fn k2)
                  value (apply transform-fn noise eval-args)
                  lp (apply log-prob-fn value eval-args)]
              {:values (assoc values addr value)
               :score (mx/add score lp)
               :key k1})))

        (:args-noise-fn nt)
        ;; Dynamic-shape noise (iid-gaussian): noise depends on eval-args
        (let [args-noise-fn (:args-noise-fn nt)
              transform-fn (:transform nt)
              log-prob-fn (:log-prob nt)]
          (fn [{:keys [values score key]} args-vec]
            (let [eval-args (mapv #(% values args-vec) compiled-args)
                  [k1 k2] (rng/split key)
                  noise (args-noise-fn eval-args k2)
                  value (apply transform-fn noise eval-args)
                  lp (apply log-prob-fn value eval-args)]
              {:values (assoc values addr value)
               :score (mx/add score lp)
               :key k1})))

        :else
        ;; Delta: no noise, value = first arg, lp = 0 (in simulate, value always matches)
        ;; Still consume a key split for PRNG equivalence with handler
        (fn [{:keys [values score key]} args-vec]
          (let [eval-args (mapv #(% values args-vec) compiled-args)
                [k1 _k2] (rng/split key)
                value (first eval-args)]
            {:values (assoc values addr value)
             :score score ;; delta log-prob is 0 in simulate
             :key k1}))))))

(defn prepare-static-sites
  "Common pipeline for static model compilation (M2).
   Returns {:site-specs :retval-fn :addrs :n-sites :binding-env} or nil."
  [schema source]
  (when (and (:static? schema)
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [binding-env (build-binding-env source)
          static-sites (filterv :static? (:trace-sites schema))
          site-specs (build-fused-site-specs static-sites binding-env)]
      (when (every? some? site-specs)
        (let [return-expr (extract-return-expr (:return-form schema))
              retval-fn (compile-expr return-expr binding-env #{})]
          {:site-specs site-specs
           :retval-fn retval-fn
           :addrs (mapv :addr static-sites)
           :n-sites (count static-sites)
           :binding-env binding-env})))))

(defn make-compiled-simulate
  "Build a compiled simulate function from a gen schema and source.

   Returns (fn [key args-vec] -> {:values {addr->val} :score :retval :key})
   or nil if the model can't be compiled.

   Uses noise transforms for inline sampling/scoring (bypasses multimethod
   dispatch). Wraps in mx/compile-fn for single Metal kernel dispatch.

   Compilation fails (returns nil) when:
   - Schema is not static (dynamic addresses, branches, loops)
   - Model has splice sites or param sites
   - Any dist-arg expression uses unsupported constructs
   - A distribution type has no known noise transform"
  [schema source]
  (when-let [{:keys [site-specs retval-fn addrs n-sites]}
             (prepare-static-sites schema source)]
    (let [step-fns (mapv build-site-step site-specs)]
      (when (and (every? some? step-fns) retval-fn)
        ;; Build the inner pure function (all MLX ops, no multimethod dispatch)
        (let [inner-fn
              (fn [key args-vec]
                (let [result
                      (reduce
                       (fn [state step-fn]
                         (step-fn state args-vec))
                       {:values {} :score (mx/scalar 0.0) :key key}
                       step-fns)
                      vals (mapv #(get (:values result) %) addrs)]
                  ;; Return flat JS array: [v0 v1 ... vN score]
                  (to-array (conj vals (:score result)))))
              ;; Build arity-specific wrapper for mx/compile-fn.
              ;; Skip mx/compile-fn when any site uses :args-noise-fn
              ;; (dynamic noise shape breaks Metal graph caching).
              n-params (count (first source))
              has-dynamic-noise?
              (some (fn [ss]
                      (let [nt (get noise-transforms-full (:dist-type ss))]
                        (and nt (:args-noise-fn nt))))
                    site-specs)
              compiled-inner
              (if has-dynamic-noise?
                ;; Raw noise transforms — still faster than multimethod dispatch
                (fn [key args] (inner-fn key args))
                (case n-params
                  0 (let [f (fn [key] (inner-fn key []))
                          cf (mx/compile-fn f)]
                      (fn [key _args] (cf key)))
                  1 (let [f (fn [key a0] (inner-fn key [a0]))
                          cf (mx/compile-fn f)]
                      (fn [key args] (cf key (nth args 0))))
                  2 (let [f (fn [key a0 a1] (inner-fn key [a0 a1]))
                          cf (mx/compile-fn f)]
                      (fn [key args] (cf key (nth args 0) (nth args 1))))
                  3 (let [f (fn [key a0 a1 a2] (inner-fn key [a0 a1 a2]))
                          cf (mx/compile-fn f)]
                      (fn [key args] (cf key (nth args 0) (nth args 1) (nth args 2))))
                  ;; >3 params: no mx/compile-fn, use raw noise transforms
                  (fn [key args] (inner-fn key args))))]
          ;; Outer wrapper: GenMLX interface
          (fn compiled-simulate [key args-vec]
            (let [mlx-args (ensure-mlx-args args-vec)
                  result (compiled-inner key mlx-args)
                  ;; Unpack flat JS array → values map
                  values (loop [i 0 m {}]
                           (if (= i n-sites)
                             m
                             (recur (inc i)
                                    (assoc m (nth addrs i) (aget result i)))))
                  score (aget result n-sites)]
              {:values values
               :score score
               :retval (when retval-fn
                         (retval-fn values args-vec))})))))))

;; ===========================================================================
;; Level 1-M3: Partial Compilation for Dynamic Models
;; ===========================================================================
;;
;; Architecture: compiled static prefix + replay transition for handler
;;
;;   Source form (from gen macro)
;;     │ extract-prefix-sites → static trace sites before first dynamic construct
;;     ▼
;;   make-compiled-prefix
;;     │ builds pure fn for prefix sites (noise transforms + mx/compile-fn)
;;     │ truncates at first non-compilable site
;;     ▼
;;   DynamicGF.simulate (in dynamic.cljs)
;;     │ Phase 1: run compiled prefix → values + partial score
;;     │ Phase 2: run-handler with replay transition (replay prefix, simulate rest)
;;     ▼
;;   Trace (shared data type)

;; ---------------------------------------------------------------------------
;; Prefix extraction: walk source form, collect static prefix sites
;; ---------------------------------------------------------------------------

(defn- contains-gen-call-any?
  "Does this form recursively contain any trace, splice, or param calls?"
  [form]
  (cond
    (and (seq? form) (seq form))
    (let [head (first form)]
      (or (and (symbol? head)
               (let [n (name head)]
                 (or (= n "trace") (= n "splice") (= n "param"))))
          (some contains-gen-call-any? form)))
    (and (vector? form) (seq form))
    (some contains-gen-call-any? form)
    :else false))

(defn- simple-trace-call?
  "Is form exactly (trace :keyword dist-expr)?"
  [form]
  (and (seq? form) (seq form)
       (symbol? (first form))
       (= "trace" (name (first form)))
       (keyword? (second form))
       (= 3 (count form))))

(defn- extract-dist-info
  "Extract dist-type and dist-args from a dist constructor form.
   (dist/gaussian 0 10) → {:dist-type :gaussian :dist-args [0 10]}"
  [dist-form]
  (when (and (seq? dist-form) (seq dist-form) (symbol? (first dist-form)))
    (let [sym (first dist-form)
          ns-part (namespace sym)
          name-part (name sym)]
      (when (or (nil? ns-part)
                (= ns-part "dist")
                (= ns-part "genmlx.dist")
                (and (string? ns-part) (.endsWith ns-part ".dist")))
        {:dist-type (keyword name-part)
         :dist-args (vec (rest dist-form))}))))

(declare walk-prefix-forms)

(defn- walk-prefix-bindings
  "Walk let bindings, collecting prefix trace sites.
   Returns {:prefix [...] :stopped boolean}."
  [bindings prefix]
  (if-not (seq bindings)
    {:prefix prefix :stopped false}
    (let [[_sym val-form] (first bindings)]
      (if (simple-trace-call? val-form)
        ;; Static trace in binding → add to prefix
        (let [addr (second val-form)
              dist-form (nth val-form 2)
              info (extract-dist-info dist-form)]
          (if info
            (recur (rest bindings) (conj prefix (assoc info :addr addr)))
            ;; Unrecognizable dist form → stop
            {:prefix prefix :stopped true}))
        ;; Not a simple trace
        (if (contains-gen-call-any? val-form)
          {:prefix prefix :stopped true}
          (recur (rest bindings) prefix))))))

(defn- walk-prefix-forms
  "Walk body forms sequentially, collecting prefix trace sites.
   Stops at the first form containing gen calls (traces in branches/loops,
   splice, param, or dynamic-address traces)."
  [forms prefix]
  (if-not (seq forms)
    prefix
    (let [form (first forms)
          rest-forms (rest forms)]
      (cond
        ;; let form: walk bindings, then body
        (and (seq? form) (seq form) (symbol? (first form))
             (= "let" (name (first form)))
             (vector? (second form)))
        (let [result (walk-prefix-bindings (partition 2 (second form)) prefix)]
          (if (:stopped result)
            (:prefix result)
            ;; Bindings done → walk let body + remaining forms
            (walk-prefix-forms (concat (drop 2 form) rest-forms) (:prefix result))))

        ;; do form: walk children
        (and (seq? form) (seq form) (symbol? (first form))
             (= "do" (name (first form))))
        (walk-prefix-forms (concat (rest form) rest-forms) prefix)

        ;; Simple trace at top level
        (simple-trace-call? form)
        (let [addr (second form)
              dist-form (nth form 2)
              info (extract-dist-info dist-form)]
          (if info
            (walk-prefix-forms rest-forms (conj prefix (assoc info :addr addr)))
            prefix))

        ;; Contains gen calls → STOP
        (contains-gen-call-any? form)
        prefix

        ;; Pure expression → continue
        :else
        (walk-prefix-forms rest-forms prefix)))))

(defn extract-prefix-sites
  "Walk gen source form, collect static trace sites in execution order
   until a dynamic construct is encountered.
   Returns [{:addr :dist-type :dist-args [...]} ...] in execution order.

   source: (params-vec body-form1 body-form2 ...) as captured by gen macro."
  [source]
  (walk-prefix-forms (rest source) []))

(defn prepare-prefix-sites
  "Common pipeline for prefix compilation (M3).
   Returns {:compiled-sites :addrs :binding-env} or nil."
  [schema source]
  (when (and (not (:static? schema))
             (seq (:trace-sites schema))
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema)))
    (let [raw-prefix (extract-prefix-sites source)]
      (when (seq raw-prefix)
        (let [binding-env (build-binding-env source)
              compiled-sites
              (reduce
               (fn [acc site]
                 (let [cargs (mapv #(compile-expr % binding-env #{})
                                   (:dist-args site))
                       nt (get noise-transforms-full (:dist-type site))]
                   (if (and nt (every? some? cargs))
                     (conj acc (assoc site :compiled-args cargs))
                     (reduced acc))))
               [] raw-prefix)]
          (when (seq compiled-sites)
            {:compiled-sites compiled-sites
             :addrs (mapv :addr compiled-sites)
             :binding-env binding-env}))))))

;; ---------------------------------------------------------------------------
;; Compiled prefix function
;; ---------------------------------------------------------------------------

(defn make-compiled-prefix
  "Build a compiled prefix function from a gen schema and source.

   Returns {:fn (fn [key args-vec] -> {:values {addr->val} :score scalar})
            :addrs [keyword...]}
   or nil if partial compilation isn't applicable.

   Compilation applies when:
   - Model is NOT static (static models use L1-M2 make-compiled-simulate)
   - Model has trace sites but no splice/param sites
   - At least 1 prefix trace site has a compilable noise transform

   Reuses L1-M2 infrastructure: build-binding-env, compile-expr,
   noise-transforms-full, build-site-step, mx/compile-fn."
  [schema source]
  (when-let [{:keys [compiled-sites addrs]} (prepare-prefix-sites schema source)]
    (let [step-fns (mapv build-site-step compiled-sites)
          n-sites (count compiled-sites)]
      (when (every? some? step-fns)
        (let [inner-fn
              (fn [key args-vec]
                (let [result
                      (reduce
                       (fn [state step-fn]
                         (step-fn state args-vec))
                       {:values {} :score (mx/scalar 0.0) :key key}
                       step-fns)
                      vals (mapv #(get (:values result) %) addrs)]
                  (to-array (conj vals (:score result)))))
              n-params (count (first source))
              ;; mx/compile-fn for 0-param models (no risk of complex args).
              ;; For models with params, use raw noise transforms (still
              ;; faster than multimethod dispatch, just without Metal fusion).
              compiled-inner
              (if (zero? n-params)
                (let [f (fn [key] (inner-fn key []))
                      cf (mx/compile-fn f)]
                  (fn [key _args] (cf key)))
                (fn [key args] (inner-fn key args)))]
          {:fn (fn compiled-prefix [key args-vec]
                 (let [mlx-args (ensure-numeric-mlx-args args-vec)
                       result (compiled-inner key mlx-args)
                       values (loop [i 0 m {}]
                                (if (= i n-sites)
                                  m
                                  (recur (inc i)
                                         (assoc m (nth addrs i) (aget result i)))))
                       score (aget result n-sites)]
                   {:values values :score score}))
           :addrs addrs})))))

;; ---------------------------------------------------------------------------
;; Replay transition for partial compilation
;; ---------------------------------------------------------------------------

(defn make-replay-simulate-transition
  "Build a replay transition for partial compilation.
   At prefix sites: split key (PRNG consistency), return pre-computed value.
   Log-prob NOT added (already counted in compiled prefix score).
   At other sites: delegate to h/simulate-transition."
  [compiled-values]
  (fn [state addr dist]
    (if (contains? compiled-values addr)
      ;; Replay: split key for PRNG consistency, return pre-computed value
      (let [[k1 _k2] (rng/split (:key state))
            value (get compiled-values addr)]
        [value (-> state
                   (assoc :key k1)
                   (update :choices cm/set-value addr value))])
      ;; Dynamic site: standard simulate
      (h/simulate-transition state addr dist))))

(defn make-replay-generate-transition
  "Build a replay transition for partial generate compilation.
   At prefix sites: replay pre-computed value, advance key correctly.
     - Unconstrained prefix: split key (matches simulate-transition)
     - Constrained prefix: no key split (matches generate-transition)
     - Score/weight NOT modified (already counted in prefix result)
   At other sites: delegate to h/generate-transition."
  [compiled-values]
  (fn [state addr dist]
    (if (contains? compiled-values addr)
      ;; Replay prefix site: set value, advance key per constraint status
      (let [value (get compiled-values addr)
            constrained? (cm/has-value? (cm/get-submap (:constraints state) addr))]
        [value (cond-> (update state :choices cm/set-value addr value)
                 (not constrained?) (#(let [[k1 _k2] (rng/split (:key %))]
                                        (assoc % :key k1))))])
      ;; Dynamic site: standard generate
      (h/generate-transition state addr dist))))

;; ===========================================================================
;; Level 1-M4: Automatic Branch Rewriting
;; ===========================================================================
;;
;; Detects if/if-not where both branches trace the same address with the
;; same distribution type. Rewrites dist args with mx/where to eliminate
;; branching. Result compiles fully via L1-M2 infrastructure.
;;
;;   Source form (with branch)
;;     │ analyze-rewritable-branch → detect rewritable pattern
;;     │ extract-rewritable-sites → all sites (standard + rewritten)
;;     ▼
;;   make-branch-rewritten-simulate
;;     │ compile cond/args, wrap in mx/where, build step fns
;;     │ wraps in mx/compile-fn → single Metal kernel
;;     ▼
;;   DynamicGF.simulate (in dynamic.cljs)
;;     │ dispatches: compiled path (same as M2)
;;     ▼
;;   Trace (shared data type)

(defn- analyze-rewritable-branch
  "Analyze an if/if-not form for branch rewriting (L1-M4).
   Both branches must be bare (trace :addr (dist/type args...)) calls
   with the same address and same distribution type.
   Returns {:addr :dist-type :cond-form :true-dist-args :false-dist-args :flipped?}
   or nil."
  [form]
  (when (and (seq? form) (seq form) (symbol? (first form))
             (= (count form) 4))
    (let [head (name (first form))
          flipped? (= head "if-not")]
      (when (or (= head "if") flipped?)
        (let [cond-form (nth form 1)
              true-form (nth form 2)
              false-form (nth form 3)]
          (when (and (simple-trace-call? true-form)
                     (simple-trace-call? false-form))
            (let [t-addr (second true-form)
                  f-addr (second false-form)
                  t-info (extract-dist-info (nth true-form 2))
                  f-info (extract-dist-info (nth false-form 2))]
              (when (and t-info f-info
                         (= t-addr f-addr)
                         (= (:dist-type t-info) (:dist-type f-info))
                         (= (count (:dist-args t-info)) (count (:dist-args f-info))))
                {:addr t-addr
                 :dist-type (:dist-type t-info)
                 :cond-form cond-form
                 :true-dist-args (:dist-args t-info)
                 :false-dist-args (:dist-args f-info)
                 :flipped? flipped?}))))))))

;; ---------------------------------------------------------------------------
;; Source form walker for branch rewriting
;; ---------------------------------------------------------------------------

(declare walk-rewrite-forms)

(defn- walk-rewrite-bindings
  "Walk let bindings, collecting sites (standard + rewritten branches).
   Returns {:sites [...] :failed? boolean}."
  [bindings sites]
  (if-not (seq bindings)
    {:sites sites :failed? false}
    (let [[_sym val-form] (first bindings)]
      (cond
        ;; Simple trace binding
        (simple-trace-call? val-form)
        (let [addr (second val-form)
              dist-form (nth val-form 2)
              info (extract-dist-info dist-form)]
          (if info
            (recur (rest bindings) (conj sites (assoc info :addr addr)))
            {:sites sites :failed? true}))

        :else
        (if-let [branch (analyze-rewritable-branch val-form)]
          ;; Rewritable branch binding
          (recur (rest bindings) (conj sites branch))
          ;; Not a rewritable branch — check for gen calls
          (if (contains-gen-call-any? val-form)
            {:sites sites :failed? true}
            ;; Pure binding → continue
            (recur (rest bindings) sites)))))))

(defn- walk-rewrite-forms
  "Walk body forms, collecting sites (standard + rewritten branches).
   Returns sites vector or nil if any form can't be handled."
  [forms sites]
  (if-not (seq forms)
    sites
    (let [form (first forms)
          rest-forms (rest forms)]
      (cond
        ;; let form
        (and (seq? form) (seq form) (symbol? (first form))
             (= "let" (name (first form)))
             (vector? (second form)))
        (let [result (walk-rewrite-bindings (partition 2 (second form)) sites)]
          (if (:failed? result)
            nil
            (walk-rewrite-forms (concat (drop 2 form) rest-forms) (:sites result))))

        ;; do form
        (and (seq? form) (seq form) (symbol? (first form))
             (= "do" (name (first form))))
        (walk-rewrite-forms (concat (rest form) rest-forms) sites)

        ;; Simple trace at top level
        (simple-trace-call? form)
        (let [addr (second form)
              dist-form (nth form 2)
              info (extract-dist-info dist-form)]
          (if info
            (walk-rewrite-forms rest-forms (conj sites (assoc info :addr addr)))
            nil))

        ;; if/if-not → try to rewrite
        (and (seq? form) (seq form) (symbol? (first form))
             (let [n (name (first form))] (or (= n "if") (= n "if-not"))))
        (if-let [branch (analyze-rewritable-branch form)]
          (walk-rewrite-forms rest-forms (conj sites branch))
          nil)

        ;; Contains gen calls → fail
        (contains-gen-call-any? form)
        nil

        ;; Pure expression → continue
        :else
        (walk-rewrite-forms rest-forms sites)))))

(defn extract-rewritable-sites
  "Walk gen source form, collect all trace sites including rewritten branches.
   Returns vector of site specs if ALL sites are compilable, or nil."
  [source]
  (walk-rewrite-forms (rest source) []))

;; ---------------------------------------------------------------------------
;; Compiled branch-rewritten models: shared pipeline + simulate/generate
;; ---------------------------------------------------------------------------

(defn compile-branch-rewritten-site-specs
  "Shared pipeline for branch-rewritten models (L1-M4).
   Builds binding env, compiles site args (standard or mx/where-wrapped),
   and compiles return expression.
   Returns {:site-specs [...] :retval-fn fn :addrs [...]} or nil."
  [schema source raw-sites]
  (let [base-env (build-binding-env source)
        binding-env
        (reduce-kv
         (fn [env k v]
           (if (= (:kind v) :expr)
             (if-let [branch (analyze-rewritable-branch (:form v))]
               (assoc env k {:kind :trace :addr (:addr branch)})
               env)
             env))
         base-env base-env)
        ;; Compile each site's args (standard or where-wrapped)
        site-specs
        (mapv
         (fn [site]
           (if (:cond-form site)
              ;; Rewritten branch: wrap dist args in mx/where.
              ;; Safety: condition must be a JS value (parameter or literal),
              ;; not an MLX array. MLX arrays are always truthy in CLJS's if,
              ;; so mx/where would disagree with the handler path.
             (let [cf (:cond-form site)
                   safe-cond? (or (boolean? cf) (number? cf)
                                  (and (symbol? cf)
                                       (= :param (:kind (get binding-env (name cf))))))]
               (when safe-cond?
                 (let [cond-fn (compile-expr cf binding-env #{})
                       true-fns (mapv #(compile-expr % binding-env #{}) (:true-dist-args site))
                       false-fns (mapv #(compile-expr % binding-env #{}) (:false-dist-args site))
                       flipped? (:flipped? site)]
                   (when (and cond-fn (every? some? true-fns) (every? some? false-fns))
                     (let [[sel-t sel-f] (if flipped? [false-fns true-fns] [true-fns false-fns])
                           compiled-args
                           (mapv (fn [tf ff]
                                   (fn [values args-vec]
                                     (mx/where (mx/ensure-array (cond-fn values args-vec))
                                               (mx/ensure-array (tf values args-vec))
                                               (mx/ensure-array (ff values args-vec)))))
                                 sel-t sel-f)]
                       {:addr (:addr site)
                        :compiled-args compiled-args
                        :dist-type (:dist-type site)})))))
              ;; Standard site
             (let [cargs (mapv #(compile-expr % binding-env #{}) (:dist-args site))]
               (when (every? some? cargs)
                 {:addr (:addr site)
                  :compiled-args cargs
                  :dist-type (:dist-type site)}))))
         raw-sites)]
    (when (every? some? site-specs)
      (let [return-expr (extract-return-expr (:return-form schema))
            retval-fn
            (or
             (when (and (seq? return-expr) (seq return-expr)
                        (symbol? (first return-expr)))
               (when-let [branch (analyze-rewritable-branch return-expr)]
                 (let [addr (:addr branch)]
                   (fn [v _a] (get v addr)))))
             (compile-expr return-expr binding-env #{}))]
        {:site-specs site-specs
         :retval-fn retval-fn
         :addrs (mapv :addr site-specs)}))))

(defn prepare-branch-sites
  "Common pipeline for branch-rewritten compilation (M4).
   Returns {:site-specs :retval-fn :addrs} or nil."
  [schema source]
  (when (and (:has-branches? schema)
             (empty? (:splice-sites schema))
             (empty? (:param-sites schema))
             (not (:has-loops? schema))
             (not (:dynamic-addresses? schema)))
    (when-let [raw-sites (extract-rewritable-sites source)]
      (when (seq raw-sites)
        (compile-branch-rewritten-site-specs schema source raw-sites)))))

(defn make-branch-rewritten-simulate
  "Build a compiled simulate for models with rewritable branches (L1-M4).

   Detects if/if-not where both branches trace the same address with the
   same dist-type, and rewrites dist args using mx/where.

   Returns (fn [key args-vec] -> {:values :score :retval :key}) or nil.

   Reuses M2 infrastructure: build-binding-env, compile-expr, build-site-step,
   noise-transforms, mx/compile-fn."
  [schema source]
  (when-let [{:keys [site-specs retval-fn addrs]}
             (prepare-branch-sites schema source)]
    (let [step-fns (mapv build-site-step site-specs)]
      (when (every? some? step-fns)
        (let [n-sites (count site-specs)
              inner-fn
              (fn [key args-vec]
                (let [result
                      (reduce
                       (fn [state step-fn]
                         (step-fn state args-vec))
                       {:values {} :score (mx/scalar 0.0) :key key}
                       step-fns)
                      vals (mapv #(get (:values result) %) addrs)]
                  (to-array (conj vals (:score result)))))
              ;; Skip mx/compile-fn for M4 — mx/where args vary per call
              ;; and compile-fn traces with fixed values on first call.
              ;; Raw noise transforms still bypass multimethod dispatch.
              compiled-inner
              (fn [key args] (inner-fn key args))]
          (fn compiled-branch-simulate [key args-vec]
            (let [mlx-args (ensure-mlx-args args-vec)
                  result (compiled-inner key mlx-args)
                  values (loop [i 0 m {}]
                           (if (= i n-sites)
                             m
                             (recur (inc i)
                                    (assoc m (nth addrs i) (aget result i)))))
                  score (aget result n-sites)]
              {:values values
               :score score
               :retval (when retval-fn
                         (retval-fn values args-vec))})))))))

