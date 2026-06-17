(ns genmlx.world.proc
  "The PROCESS / SCHEDULER face of the Bun WORLD membrane (bean genmlx-gsoi) —
   FACE 2, sibling to `genmlx.world.net` (the network face). This is CONTROL's
   eval!-equivalent: the ONE side effect of the control layer (advancing compute
   under a wall-clock deadline). The scheduler is NEVER a generative function.

   GRADE vs net.cljs: net is ASYNC to an autonomous OTHER (a promise, a remote
   peer). This face is SYNCHRONOUS over our OWN compute under a clock — closer to
   mlx eval! than to net. The per-chunk advance MUST stay synchronous: a promesa
   promise returned from a Bun timer callback is silently NOT awaited (the timer
   twin of net.cljs's silent-HTTP-200 gotcha), and an async loop would lose the
   materialize!/tidy boundaries and corrupt deadline accounting. So the floor is
   a plain loop/recur — no setInterval, no promesa.

   The scheduler drives a STEPPABLE interface passed as plain fns
   {:init :step :done? :best} (NOT a protocol — proc stays decoupled; it treats
   `state` as OPAQUE and never inspects substrate internals, respecting the
   smcp3-init/smcp3-step payload asymmetry). genmlx.inference.steppable (genmlx-rfal)
   supplies exactly this surface; proc imports nothing from inference/agents/control.

   The only effectful primitives here are the monotonic clock (Bun.nanoseconds)
   and the periodic GC sweep that re-homes the smcp3 driver's housekeeping
   (smcp3.cljs:218) now that this loop replaces the driver loop.

   Worker / Bun.spawn carrying live MLX state across a Bun realm is UNPROVEN
   (cross-realm Metal) and is fenced as a spike-gated placeholder below — it
   must not silently ship as a v1.0 promise."
  (:require [genmlx.mlx :as mx]))

(def ^:private Bun (.-Bun js/globalThis))

(defn available?
  "True when the Bun high-res monotonic clock backing this face is present."
  []
  (boolean (and Bun (fn? (.-nanoseconds Bun)))))

(defn- now-ns
  "The sole clock read: Bun.nanoseconds as a JS number (0 off-Bun)."
  []
  (if (available?)
    (js/Number (.nanoseconds Bun))
    0))

;; ===========================================================================
;; with-deadline — the synchronous anytime scheduler (control's eval!)
;; ===========================================================================

(defn with-deadline
  "Advance a steppable substrate synchronously until it is `done?` or a wall-clock
   budget elapses, whichever comes first — the anytime 'one cycle, more if time
   allows'. The substrate is four plain fns:

     init   : () -> state         (build the initial value)
     step   : state -> state'     (advance one step; pure value -> value)
     done?  : state -> boolean    (no more work)
     best   : state -> result     (extract the current best answer — always valid)

   opts: {:budget-ms  wall-clock budget in ms (default 1000)
          :chunk      steps advanced between clock polls (default 256); sized so a
                      chunk's compute dwarfs a Bun.nanoseconds read
          :min-steps  minimum steps before the deadline can stop the run
                      (default 1 — guarantees at least one cycle even at budget 0)
          :gc-every   sweep dead MLX arrays every Nth chunk boundary (default 1;
                      re-homes smcp3.cljs:218). 0 disables proc's sweep.
          :now-fn     clock override (default Bun.nanoseconds); injected in tests}

   Returns {:best (best final-state) :state final-state :steps n
            :elapsed-ns e :stopped-by :done|:deadline}. SYNCHRONOUS — returns a
   plain map, never a promise."
  [init step done? best {:keys [budget-ms chunk min-steps gc-every now-fn]
                         :or {budget-ms 1000 chunk 256 min-steps 1 gc-every 1
                              now-fn now-ns}}]
  (let [start (now-fn)
        deadline (+ start (* budget-ms 1e6))
        finish (fn [state steps why]
                 {:best (best state) :state state :steps steps
                  :elapsed-ns (- (now-fn) start) :stopped-by why})]
    (loop [state (init), steps 0, chunks 0]
      (cond
        (done? state) (finish state steps :done)

        ;; Poll the clock only at chunk boundaries, and only once min-steps is met
        ;; (the anytime guarantee). `and` short-circuits so now-fn is not even
        ;; read until at least one chunk has run.
        (and (>= steps min-steps) (>= (now-fn) deadline))
        (finish state steps :deadline)

        :else
        ;; Advance one chunk synchronously (or until done? mid-chunk).
        (let [[state' n] (loop [s state, i 0]
                           (if (or (>= i chunk) (done? s))
                             [s i]
                             (recur (step s) (inc i))))
              chunks' (inc chunks)]
          ;; GC sweep at the chunk boundary — the housekeeping the smcp3 driver
          ;; loop owned (smcp3.cljs:218), now that proc replaces it. Never affects
          ;; results; outside the deadline-accounting hot read.
          (when (and (pos? gc-every) (zero? (mod chunks' gc-every)))
            (mx/sweep-dead-arrays!)
            (mx/clear-cache!))
          (recur state' (+ steps n) chunks'))))))

;; ===========================================================================
;; SPIKE-GATED (NOT v1.0 floor) — parallel/worker scheduling
;; ===========================================================================
;; A Worker / Bun.spawn scheduler carrying live MLX state across a Bun realm is
;; UNPROVEN: MxArray handles do not cross postMessage, and cross-realm Metal
;; collides with the "never parallel GPU" Metal-wedge hazard. This is a post-floor
;; increment, gated on a spike proving Metal state survives the realm boundary.
;; The v1.0 floor is single-realm + synchronous. The placeholder throws so it
;; cannot silently ship as a promise.
(defn worker-pool-UNPROVEN
  [& _]
  (throw (ex-info "Worker/Bun.spawn scheduling is spike-gated, not in the v1.0 floor (cross-realm Metal is unproven)."
                  {:status :not-implemented})))
