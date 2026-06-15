(ns genmlx.inference.steppable
  "Steppable / budgeted / introspectable SMC (genmlx-rfal).

   A thin, PURE wrapper over the existing SMCP3 kernels (smcp3-init /
   smcp3-step) that exposes inference as a resumable, introspectable VALUE:

     init-state -> step -> step -> ... -> done?
                   |          |
                   peek       peek   (posterior / ESS / log-ML, eval-bounded)

   The whole point is that a scheduler (genmlx.world.proc) can advance SMC one
   chunk at a time and decide — between chunks — whether to spend more compute.
   This file is pure inference: it imports NOTHING from genmlx.world or
   genmlx.control, preserving the one-way core -> {agents} -> control arrow.

   smcp3.cljs is UNCHANGED — this wrapper is additive. `step` reproduces the
   smcp3 driver loop body exactly (same kernels, same order, same log-ml
   accumulation graph), so a stepped run equals the batch driver to float32
   noise (see steppable_test.cljs)."
  (:require [genmlx.inference.smcp3 :as smcp3]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.util :as u]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

;; All fields immutable; `step` returns a NEW SMCState.
;;   model    - the ALREADY auto-keyed + analytical-stripped model (set once)
;;   kernels  - {:forward :backward :init-proposal} (auto-keyed gfs or nil)
;;   t        - number of COMPLETED steps (0 = uninitialized; init is step 0)
;;   log-ml   - MLX scalar accumulator of the per-step log-ML increments
;;   key      - PRNG key carried into the NEXT step (nil => auto-key entropy)
(defrecord SMCState
  [model args kernels obs-seq particles ess-threshold rejuvenation-fn
   t traces log-weights log-ml key])

(defn init-state
  "Build the initial SMCState. O(1): does NOT run inference (the first `step`
   runs smcp3-init). Replicates the smcp3 driver's one-time setup EXACTLY
   (smcp3.cljs:201-204): auto-key + strip-analytical the model ONCE, and
   auto-key each non-nil kernel. The model MUST be auto-keyed here — smcp3-init
   only re-strips, it does not auto-key — else particles carry no PRNG key
   (or, with a fixed key, collapse to identical draws).

   opts: {:particles :ess-threshold :forward-kernel :backward-kernel
          :init-proposal :rejuvenation-fn :key}"
  [model args obs-seq {:keys [particles ess-threshold forward-kernel
                              backward-kernel init-proposal rejuvenation-fn key]
                       :or {particles 100 ess-threshold 0.5}}]
  (->SMCState
    (-> model dyn/auto-key smc/strip-analytical)
    (vec args)
    {:forward (when forward-kernel (dyn/auto-key forward-kernel))
     :backward (when backward-kernel (dyn/auto-key backward-kernel))
     :init-proposal (when init-proposal (dyn/auto-key init-proposal))}
    (vec obs-seq) particles ess-threshold rejuvenation-fn
    0 nil nil (mx/scalar 0.0) key))

(defn done?
  "True once every observation has been folded in. `t` counts COMPLETED steps."
  [state]
  (>= (:t state) (count (:obs-seq state))))

(defn step
  "Advance one SMCP3 step over the next observation, returning a NEW SMCState.
   Mirrors the smcp3 driver loop body (smcp3.cljs:216-231) verbatim — including
   the every-10-completed-steps GC sweep the driver owns at smcp3.cljs:218 —
   so a stepped run is semantically identical to the batch driver. Throws on a
   completed state (an explicit oracle for a scheduler; check `done?` first)."
  [state]
  (when (done? state)
    (throw (ex-info "step called on a completed SMCState" {:t (:t state)})))
  (let [{:keys [model args kernels obs-seq particles ess-threshold
                rejuvenation-fn t traces log-weights log-ml key]} state
        obs-t (nth obs-seq t)
        [step-key next-key] (rng/split-or-nils key)
        ;; The driver owns the periodic sweep (smcp3-step does not). Keyed off
        ;; the current completed-step count t, invariant to chunking, mirroring
        ;; smcp3.cljs:218 exactly so a long stepped run never leaks per-step
        ;; graphs (genmlx-q6lh class).
        _ (when (and (pos? t) (zero? (mod t 10)))
            (mx/sweep-dead-arrays!)
            (mx/clear-cache!))
        {new-traces :traces new-weights :log-weights :keys [log-ml-increment]}
        (if (zero? t)
          (smcp3/smcp3-init model args obs-t (:init-proposal kernels) particles step-key)
          (smcp3/smcp3-step traces log-weights model obs-t
                            (:forward kernels) (:backward kernels)
                            particles ess-threshold rejuvenation-fn step-key))]
    (assoc state
           :t (inc t)
           :traces new-traces
           :log-weights new-weights
           ;; Same accumulation shape as the driver (smcp3.cljs:230) so the
           ;; log-ml graph is ULP-identical.
           :log-ml (mx/add log-ml log-ml-increment)
           :key next-key)))

(defn peek
  "Cheap, eval-bounded progress probe — a SYMMETRIC map (identical key set for
   a fresh, mid-run, or completed state, filling the init/step payload
   asymmetry at smcp3.cljs:70 vs :169). The single eval! is realizing log-ml.
   :ess is recomputed from the canonical log-weights (full ESS before the first
   step). :resampled? is intentionally absent — it is a per-step transient, not
   a state property."
  [state]
  {:t (:t state)
   :n-steps (count (:obs-seq state))
   :done? (done? state)
   :particles (:particles state)
   :log-ml-estimate (mx/realize (:log-ml state))
   :ess (if (:log-weights state)
          (u/compute-ess (:log-weights state))
          (:particles state))})

(defn run
  "Drive `step` to completion, returning the final SMCState. The ORACLE harness
   (a real scheduler calls step/peek/done? itself); equals the smcp3 driver
   under a fixed seed."
  [state]
  (if (done? state)
    state
    (recur (step state))))
