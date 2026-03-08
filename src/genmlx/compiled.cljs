(ns genmlx.compiled
  "Compiled execution for temporal models (Tier 2a).
   Eliminates handler overhead by compiling T-step unfold loops into single
   Metal dispatches via mx/compile-fn.

   Pattern: user provides a pure MLX step-fn, we pre-generate noise and
   unroll the loop. All randomness injected as input tensors.

   Reference: make-compiled-chain in inference/mcmc.cljs uses the same approach."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]))

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
          dummy-obs   (mx/zeros [n-steps obs-dim])]
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
