(ns genmlx.control.synth-steppable
  "Synthesis steppable (genmlx-yd7c): resource-rational program synthesis as a
   resumable, introspectable VALUE — the {:propose :deepen :stop} substrate the RRPS
   wake-phase controller allocates over (docs/rrps-design.md §3.3).

   This is the synthesis sibling of inference/steppable.cljs (SMC) and the IBIS base in
   the anytime-control bench: a base steppable a scheduler (genmlx.world.proc) drives
   one action at a time, and a metareasoner (genmlx.control.meta-mdp) gates with a
   value-of-computation stop. It is PURE Layer-A flow (the only effects are the injected
   scorer's GPU evals + the host cost it reports); it imports nothing from world/llm.

   GENERIC by construction — the task (the candidate programs, the conjugacy classes,
   the exact/IS scorer, the held-out decision-value) is INJECTED via `config`, so this
   namespace carries no demonstration specifics. The RRPS bench (bench/rrps.cljs) wires
   the P0 task in; a real frozen LLM stream (genmlx-7f99) wires the same surface.

   The base exposes TWO faces:
     • the plain {:init :step :done? :best} a bare scheduler drives (its :step runs a
       default greedy policy — propose until the stream is dry, then stop — so
       proc/with-deadline drives it to a committed stop with no controller);
     • the richer {:actions :apply-action} a K-action VOC controller consumes
       (per-action metered trial-advance: meta-mdp/controlled-steppable-k).

   `apply-action` returns {:state state' :cost cost-meter}, where the cost-meter carries
   the HONEST synthesis cost of the action (:llm-tokens/:sci-evals for a proposal,
   :particles for an IS scoring) folded with any GPU :forced-evals the scorer incurred —
   so the controller meters real compute, never a fiction (cost.cljs/synth-compute is
   the scalar reduction)."
  (:require [genmlx.inference.cost :as cost]))

;; ---------------------------------------------------------------------------
;; State
;;   :pool       vector of {:id :cand :log-ml :depth :method :conjugate?}
;;   :stream-idx index of the next un-revealed stream candidate
;;   :stopped?   the controller's committed stop (flips done?)
;; ---------------------------------------------------------------------------

(defn- score-candidate
  "Score one stream candidate at `depth` via the injected scorer, metering host +
   GPU cost. Returns {:entry pool-entry :cost cost-meter}. The scorer returns
   {:log-ml :method :conjugate?}; a conjugate candidate ignores depth (exact)."
  [{:keys [score proposal-cost init-depth]} cand]
  (let [{result :result gpu :cost}
        (cost/measure (fn [] (score cand init-depth)))
        {:keys [log-ml method conjugate?]} result
        host (cost/cost+ (proposal-cost cand)
                         (if conjugate? cost/zero {:particles init-depth}))
        entry {:id (:id cand) :cand cand :log-ml log-ml :depth (if conjugate? 0 init-depth)
               :method method :conjugate? conjugate?}]
    {:entry entry :cost (cost/cost+ gpu host)}))

(defn- deepen-target
  "The pool index of the IS (non-conjugate) entry most worth deepening — the leading
   IS contender below max-depth (deepening the contender is what could change the
   ranking). nil when none qualifies."
  [pool max-depth]
  (let [cands (->> (map-indexed vector pool)
                   (filter (fn [[_ e]] (and (not (:conjugate? e)) (< (:depth e) max-depth)))))]
    (when (seq cands)
      (first (apply max-key (fn [[_ e]] (:log-ml e)) cands)))))

(defn synth-steppable
  "Build a synthesis steppable. config:
     :stream         vector of opaque candidate descriptors (the frozen proposer order)
     :score          (fn [cand depth] -> {:log-ml :method :conjugate?})  REQUIRED
     :init-depth     starting IS depth for a non-conjugate candidate (default 64)
     :deepen-factor  multiplier applied to depth on :deepen (default 8)
     :max-depth      cap on IS depth (default 4096)
     :proposal-cost  (fn [cand] -> cost-meter) host cost of one proposal
                     (default {:llm-tokens 120 :sci-evals 1})
   Returns {:init :actions :apply-action :done? :best :step :best-entry}."
  [{:keys [stream score init-depth deepen-factor max-depth proposal-cost]
    :or {init-depth 64 deepen-factor 8 max-depth 4096
         proposal-cost (fn [_] {:llm-tokens 120 :sci-evals 1})}
    :as config}]
  (let [cfg (assoc config :init-depth init-depth :proposal-cost proposal-cost)
        n-stream (count stream)
        can-propose? (fn [s] (< (:stream-idx s) n-stream))
        can-deepen?  (fn [s] (some? (deepen-target (:pool s) max-depth)))
        best-entry   (fn [s] (when (seq (:pool s)) (apply max-key :log-ml (:pool s))))]
    {:init (fn [] {:pool [] :stream-idx 0 :stopped? false})

     :actions (fn [s]
                (cond-> [:stop]
                  (can-propose? s) (conj :propose)
                  (can-deepen? s)  (conj :deepen)))

     :apply-action
     (fn [s action]
       (case action
         :propose
         (let [cand (nth stream (:stream-idx s))
               {:keys [entry cost]} (score-candidate cfg cand)]
           {:state (-> s (update :pool conj entry) (update :stream-idx inc))
            :cost cost})
         :deepen
         (let [i (deepen-target (:pool s) max-depth)
               e (nth (:pool s) i)
               new-depth (min max-depth (* deepen-factor (:depth e)))
               {result :result gpu :cost} (cost/measure (fn [] (score (:cand e) new-depth)))
               {:keys [log-ml method]} result]
           {:state (assoc-in s [:pool i] (assoc e :log-ml log-ml :method method :depth new-depth))
            :cost (cost/cost+ gpu {:particles (- new-depth (:depth e))})})
         :stop
         {:state (assoc s :stopped? true) :cost cost/zero}))

     :done?  (fn [s] (or (:stopped? s)
                         (and (not (can-propose? s)) (not (can-deepen? s)))))
     :best   (fn [s] (:id (best-entry s)))
     :best-entry best-entry

     ;; Default greedy policy for a bare scheduler / self-test: propose until the
     ;; stream is dry, then stop. Drives proc/with-deadline to a committed stop.
     :step (fn [s]
             (if (can-propose? s)
               (let [cand (nth stream (:stream-idx s))
                     {:keys [entry]} (score-candidate cfg cand)]
                 (-> s (update :pool conj entry) (update :stream-idx inc)))
               (assoc s :stopped? true)))}))
