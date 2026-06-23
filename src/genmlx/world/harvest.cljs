(ns genmlx.world.harvest
  "Phase 3 (genmlx-oexl), the RUN side of the REPL-trace harvest. The curriculum ->
   loop -> trajectories -> build-corpus GLUE already exists and is tested
   (scripts/curriculum_probe.cljs + curriculum_test + repl_corpus_test); what was
   missing is running that EXISTING pipeline at scale with the REAL LLM loop-proposer
   (vs the native-free family-proposer used for plumbing validation). This ns owns the
   two reusable pieces that scaling needs, so the scaled harvest is a thin script over a
   TESTED core rather than ad-hoc glue:

     1. loop-proposer — the SHARED LLM proposer: `lp/make-proposer` (the propose ->
        check -> revise mini-REPL) UNIONED with a deterministic shared-σ grid refiner.
        This is the resource-rational split the north star rests on: the LLM proposes
        STRUCTURE, the cheap deterministic grid tunes the nuisance scale, and the exact
        oracle accepts a σ only when evidence improves. MOVED here (not duplicated) from
        scripts/synth_llm_probe.cljs so the harvested corpus reflects the EXACT loop the
        verdict probe validated — one source of truth for `propose`.

     2. harvest-task / harvest-tasks — run a task (or a seq) through the greedy
        (`syn/synthesize`) or beam (`se/search`) driver with an injected proposer,
        returning run maps `{:task {:id :task-desc :observations} :trajectory ...}` in
        exactly the shape `genmlx.world.repl-corpus/build-corpus` consumes. `on-run`
        streams each completed run so a long harvest persists incrementally (a worker
        death mid-run never loses the runs already finished).

   NATIVE-FREE: the policy LLM is INJECTED (`:call-llm`) and lives out-of-process; this
   ns loads no model. The caller supplies `init-spec-for` (the crude covering model per
   task) so harvest stays free of a crude-spec copy (curriculum/crude-spec or the probe's
   init-spec are the existing ones) and free of a curriculum dependency."
  (:require [genmlx.world.synth :as syn]
            [genmlx.world.search :as se]
            [genmlx.world.llm-proposer :as lp]))

;; ===========================================================================
;; 1. The shared LLM loop-proposer (LLM structure ∪ deterministic σ grid).
;; ===========================================================================

(def noise-grid
  "The shared observation-σ refinement grid. The LLM neglects the nuisance scale (it
   proposes structure but leaves σ ~1); this cheap deterministic grid is the type-II ML
   move over ONE observation scale, oracle-selected per step (accepted only when it
   improves exact evidence)."
  [0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.5])

(defn noise-refiner
  "A pure proposer offering each grid σ as a shared-noise edit over all obs sites (the σ
   already in use is skipped). Only fires once every obs site carries a noise arg."
  [spec _feedback]
  (let [obs (:obs spec)
        cur (last (:args (first obs)))]
    (if (and (seq obs) (every? #(>= (count (:args %)) 2) obs))
      (for [g noise-grid :when (not= g cur)]
        {:edit :set-noise :desc (str "shared obs σ -> " g)
         :spec' (reduce #(syn/set-noise %1 %2 g) spec (map :addr obs))})
      [])))

(defn loop-proposer
  "Build the north-star inner-loop proposer: the LLM mini-REPL (`lp/make-proposer`)
   UNIONED with `noise-refiner`. Returns `(fn [spec feedback] -> [candidate ...])`.

   opts (the LLM half is forwarded verbatim to `lp/make-proposer`):
     :call-llm      REQUIRED — the injected `(fn [req] -> {:completions [...] ...})`
     :task-desc     natural-language task (no structure given away)
     :observations  the {:addr value} data
     :k             samples per generation call (default 4)
     :temperature   (default 0.7)
     :max-tokens    per-sample cap (default 384)
     :revise        max self-correction re-prompts when no candidate scores (default 0)
     :n-particles   IS particles for the internal revise-decision check (default 2000)
     :seed          worker RNG seed (default 1)"
  [{:keys [call-llm task-desc observations k temperature max-tokens revise n-particles seed system]
    :or   {k 4 temperature 0.7 max-tokens 384 revise 0 n-particles 2000 seed 1}}]
  (syn/union-proposer
   (lp/make-proposer (cond-> {:call-llm call-llm :task-desc task-desc :observations observations
                              :k k :temperature temperature :max-tokens max-tokens
                              :revise revise :n-particles n-particles :seed seed}
                       ;; an optional system override (e.g. the SFT student's short system,
                       ;; to keep train==inference); omitted -> make-proposer's default-system.
                       system (assoc :system system)))
   noise-refiner))

;; ===========================================================================
;; 2. Run a task (or a seq) through the driver, collecting trajectories.
;; ===========================================================================

(defn harvest-task
  "Run ONE task through the REPL driver with an injected `propose`, returning a run map
   in exactly the shape `genmlx.world.repl-corpus/build-corpus` consumes:
     {:task {:id :task-desc :observations}   ; the canonical fields build-corpus needs
      :trajectory [...]                        ; the RAW accepted-state trajectory (each
                                               ;   step carries :code; step 0 = init crude)
      :steps n :final evidence :stop-reason kw :solved? bool}

   `task` is any map with at least {:id :task-desc :observations} (a curriculum record or
   the probe's task map both qualify; :solve-bar, when present, sets :solved?).

   opts:
     :propose     REQUIRED — `(fn [spec feedback] -> [candidate ...])`
     :init-spec   REQUIRED — the crude covering model to start from
     :strategy    :greedy (`syn/synthesize`) | :beam (`se/search`)   default :greedy
     :max-steps :plateau-eps :n-particles                            driver knobs
     :beam-width :adaptive? :seed                                    beam-only knobs"
  [{:keys [id task-desc observations solve-bar]}
   {:keys [propose init-spec strategy max-steps plateau-eps n-particles beam-width adaptive? seed]
    :or   {strategy :greedy max-steps 6 plateau-eps 0.05 n-particles 2000
           beam-width 4 adaptive? true seed 1}}]
  (when-not propose   (throw (js/Error. "harvest-task: :propose is required")))
  (when-not init-spec (throw (js/Error. "harvest-task: :init-spec is required")))
  (let [res   (case strategy
                :greedy (syn/synthesize {:init-spec init-spec :observations observations
                                         :propose propose :max-steps max-steps
                                         :plateau-eps plateau-eps :n-particles n-particles})
                :beam   (se/search {:init-spec init-spec :observations observations
                                    :propose propose :beam-width beam-width :adaptive? adaptive?
                                    :strategy :beam :max-steps max-steps :plateau-eps plateau-eps
                                    :n-particles n-particles :seed seed}))
        beam? (= strategy :beam)
        traj  (if beam? (:trajectory (:best res)) (:trajectory res))
        final (if beam? (:evidence (:best res)) (:evidence (:feedback res)))]
    {:task       {:id id :task-desc task-desc :observations observations}
     :trajectory traj
     :steps      (count traj)
     :final      final
     :stop-reason (:stop-reason res)
     :solved?    (boolean (and solve-bar final (js/isFinite final) (>= final solve-bar)))}))

(defn harvest-tasks
  "Map a task seq -> run maps (via `harvest-task`). `proposer-for` is `(fn [task] ->
   propose)` so each task gets a task-conditioned proposer (the LLM prompt is per-task;
   for a static proposer pass `(constantly p)`).

   opts:
     :init-spec-for  REQUIRED — `(fn [task] -> crude-spec)` (e.g. cur/crude-spec ∘ :observations)
     :run-opts       extra `harvest-task` opts merged into every run (strategy, knobs)
     :on-run         optional `(fn [run idx task] ...)` called after each run (stream/persist)"
  [tasks proposer-for {:keys [init-spec-for run-opts on-run] :or {run-opts {}}}]
  (when-not init-spec-for (throw (js/Error. "harvest-tasks: :init-spec-for is required")))
  (vec
   (map-indexed
    (fn [idx task]
      (let [run (harvest-task task (assoc run-opts
                                          :propose (proposer-for task)
                                          :init-spec (init-spec-for task)))]
        (when on-run (on-run run idx task))
        run))
    tasks)))
