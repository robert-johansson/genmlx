(ns genmlx.world.search
  "Phase 2 of the north star (milestone genmlx-77vv, bean genmlx-47zx): turn the greedy
   Phase-1 kernel into a PARTICLE SEARCH over construction steps. The best-of-K wrapper
   moves to the STEP level, where the oracle signal is dense (experiment A): instead of
   one greedy state that commits to the single best edit, keep a POPULATION of partial
   models and resample/beam them by INTERMEDIATE model evidence — SMC over construction
   steps, not whole programs.

   WHY (the Phase-1 limitation it fixes). Greedy commits to the first improving
   STRUCTURAL move and cannot reconsider it (Phase-1 linreg locked into the no-intercept
   branch). A population explores competing branches in parallel; a wider beam keeps both
   the no-intercept and full-intercept linreg branches alive and lets the exact oracle
   pick the global winner. It is also the right tool for the NOISY-evidence regime (the
   IS-scored GMM): a single greedy importance-sampling draw oscillates, a population
   averages it out.

   Builds entirely on genmlx.world.synth — same model-spec, same edit ops, the same
   four-level `check` node and exact oracle, the same INJECTED proposer
   `(fn [spec feedback] -> [candidate ...])`. Native-free; the proposer here is the
   structured move-vocabulary (native particles); an LLM proposer is a drop-in (Phase 3).

   Sections:
     1. Particles — a partial-model trajectory as a search state
     2. Expansion — propose K edits per particle, render + check + keep the improving
     3. Selection — :beam (deterministic top-B) | :smc (resample by evidence) + dedup
     4. Adaptive allocation — wider beam on ambiguous (close-evidence) steps
     5. search — the population loop (stop when the whole population plateaus)
     6. backtrack-refine — deterministic backtracking (best-first re-search over a trajectory)"
  (:require [genmlx.world.synth :as syn]))

;; ===========================================================================
;; 1. Particles — a partial-model trajectory as a search state
;;
;; A particle carries its current spec, the check feedback, its evidence (nil until
;; scored), a :done? flag (set once no proposed edit improves it), and its trajectory
;; (one row per accepted state — each row keeps its :spec so backtrack can re-open it).
;; ===========================================================================

(defn- init-particle
  [spec observations opts]
  (let [code (syn/render spec)
        fb   (syn/check code observations opts)]
    {:spec spec :feedback fb :evidence (:evidence fb) :done? false
     :trajectory [{:step 0 :edit :init :desc "crude covering model" :code code
                   :evidence (:evidence fb) :method (:method fb) :delta nil :spec spec}]}))

(defn- deterministic-method?
  [fb] (contains? #{:exact :kalman} (:method fb)))

(defn- particle-rank
  "Ascending sort key: deterministically-scored particles first (an exact marginal
   never loses to a noisy IS estimate), then higher evidence, then — among genuine
   evidence ties (e.g. two routes to the same program in dedup) — a LIVE particle
   before a :done? one, then shorter trajectory (Occam tie-break)."
  [p]
  [(if (deterministic-method? (:feedback p)) 0 1)
   (- (or (:evidence p) ##-Inf))
   (if (:done? p) 1 0)
   (count (:trajectory p))])

(defn- particle-code [p] (syn/render (:spec p)))

;; ===========================================================================
;; 2. Expansion — propose K edits per particle, render + check, keep the improving
;; ===========================================================================

(defn- expand
  "Expand one particle: render+check up to `expand-k` proposed candidate edits and keep
   the ones that are valid AND improve the particle's evidence beyond :plateau-eps. If
   none improves, the particle STAYS (returned once, marked :done?) — so a plateaued
   particle persists in the beam as a terminal until the whole population is done."
  [p propose observations {:keys [plateau-eps expand-k] :or {plateau-eps 0.05} :as opts}]
  (if (:done? p)
    [p]
    (let [cands    (let [cs (propose (:spec p) (:feedback p))]
                     (if expand-k (take expand-k cs) cs))
          cur-ev   (or (:evidence p) ##-Inf)
          children (for [c cands
                         ;; a candidate may carry raw `:code` (a real-LLM proposer emits
                         ;; the program text directly — genmlx.world.llm-proposer); check
                         ;; it VERBATIM so DSL slips reach the check node. Else render :spec'.
                         :let [code (or (:code c) (syn/render (:spec' c)))
                               fb   (syn/check code observations opts)]
                         :when (syn/scored? fb)]
                     {:spec (:spec' c) :feedback fb :evidence (:evidence fb) :done? false
                      :trajectory (conj (:trajectory p)
                                        {:step (count (:trajectory p)) :edit (:edit c) :desc (:desc c)
                                         :code code :evidence (:evidence fb) :method (:method fb)
                                         :delta (when (:evidence p) (- (:evidence fb) (:evidence p)))
                                         :spec (:spec' c)})})
          improving (filter #(> (:evidence %) (+ cur-ev plateau-eps)) children)]
      (if (seq improving)
        improving
        [(assoc p :done? true)]))))

;; ===========================================================================
;; 3. Selection — :beam (deterministic top-B) | :smc (resample) + dedup
;; ===========================================================================

(defn- dedup-pool
  "Collapse particles whose current program renders identically to one (best-ranked)
   particle — the same partial model is the same search state, however it was reached."
  [pool]
  (->> pool
       (group-by particle-code)
       vals
       (mapv #(first (sort-by particle-rank %)))))

(defn- select-beam
  "Deterministic top-`width` particles by particle-rank."
  [pool width]
  (vec (take width (sort-by particle-rank pool))))

;; A tiny pure LCG so :smc resampling (and the stochastic-proposer experiment) is
;; reproducible without touching js/Math.random (which nbb can't seed in required nses).
(defn- lcg-next [s] (mod (+ (* 1103515245 s) 12345) 2147483648))
(defn- uniforms [seed n]
  (loop [s (lcg-next (+ seed 1)), i 0, acc []]
    (if (>= i n) acc (recur (lcg-next s) (inc i) (conj acc (/ s 2147483648.0))))))

(defn- softmax [xs temp]
  (let [t  (max temp 1e-6)
        m  (apply max xs)
        es (map #(js/Math.exp (/ (- % m) t)) xs)
        z  (reduce + es)]
    (map #(/ % z) es)))

(defn- weighted-pick [items weights u]
  (loop [is items, ws weights, acc 0.0]
    (let [acc' (+ acc (first ws))]
      (if (or (empty? (rest is)) (<= u acc')) (first is) (recur (rest is) (rest ws) acc')))))

(defn- select-smc
  "Resample `width` UNIQUE particles from the pool with replacement, proportional to
   softmax(evidence / temperature), then dedup — SMC's stochastic exploration biased to
   high-evidence branches. Deterministically-scored particles get a small weight bonus
   so a noisy IS draw cannot crowd out an exact one. Reproducible via `seed`."
  [pool width seed temperature]
  (let [pool (dedup-pool pool)]
    (if (<= (count pool) width)
      (vec pool)
      (let [evs  (mapv (fn [p] (+ (or (:evidence p) ##-Inf)
                                  (if (deterministic-method? (:feedback p)) 1.0 0.0))) pool)]
        (if (every? #(= ##-Inf %) evs)
          ;; no particle scored — softmax would be all-NaN; fall back to a deterministic
          ;; top-width by rank rather than feed NaN weights to weighted-pick.
          (vec (take width (sort-by particle-rank pool)))
          (let [ws    (softmax evs temperature)
                us    (uniforms seed (* 3 width))
                picks (->> us (map #(weighted-pick pool ws %)) distinct (take width))]
            (vec (if (seq picks) picks (take width (sort-by particle-rank pool))))))))))

;; ===========================================================================
;; 4. Adaptive allocation — wider beam on ambiguous (close-evidence) steps
;; ===========================================================================

(defn- pool-spread
  "Evidence gap between the best and 2nd-best DISTINCT-program particle in the pool. A
   small gap = an ambiguous step (competing structures score alike -> widen the beam to
   keep them); a large gap = one move dominates (narrow). ##Inf when <2 distinct."
  [pool]
  (let [evs (->> (dedup-pool pool) (keep :evidence) sort reverse)]
    (if (>= (count evs) 2) (- (first evs) (second evs)) ##Inf)))

(defn- adapt-width
  "Resource-rational beam width for this step: spend MORE particles when the top
   candidates are within :spread-margin (uncertain/structural), fewer when one dominates."
  [base spread {:keys [min-width max-width spread-margin]
                :or {min-width 1 max-width 6 spread-margin 1.0}}]
  (cond
    (< spread spread-margin) (max base max-width)
    (> spread (* 3 spread-margin)) (max min-width (min base 2))
    :else base))

;; ===========================================================================
;; 5. search — the population loop
;; ===========================================================================

(declare backtrack-refine)

(defn search
  "Particle/beam search over construction steps. Maintains a population of partial-model
   particles; each step expands every non-done particle by its proposed edits, dedups,
   adaptively allocates the beam width, and selects the next population (:beam top-B or
   :smc resampling). Stops when the WHOLE population has plateaued (no particle improves)
   or :max-steps is reached.

   opts:
     :init-spec     the crude covering model to start from (or :init-specs for many)
     :observations  {:addr value ...} the data the oracle scores against
     :propose       (fn [spec feedback] -> [{:edit :desc :spec'} ...])  (injected)
     :beam-width    base population size B (default 4)
     :expand-k      max candidate edits considered per particle per step (default all)
     :strategy      :beam (deterministic top-B, default) | :smc (resample by evidence)
     :adaptive?     adapt the beam width to the per-step evidence spread (default true)
     :backtrack?    after the population plateaus, run backtrack-refine on the best
                    particle to escape a lock-in (default false)
     :min-width :max-width :spread-margin   adaptive-width bounds + the evidence-spread
                    threshold below which a step is treated as ambiguous (widen the beam)
     :max-steps :plateau-eps :n-particles :temperature :seed   (search controls)

   Returns {:best :population :trajectory :steps :stop-reason :diagnostics} where :best
   is the highest-evidence particle ever seen (method-aware) and :trajectory is its
   development trace; :diagnostics is one row per step (width, spread, best-evidence)."
  [{:keys [init-spec init-specs observations propose beam-width expand-k strategy
           adaptive? backtrack? max-steps plateau-eps n-particles temperature seed
           min-width max-width spread-margin]
    :or   {beam-width 4 strategy :beam adaptive? true backtrack? false
           max-steps 16 plateau-eps 0.05 n-particles 2000 temperature 0.5 seed 1}}]
  (let [opts  {:plateau-eps plateau-eps :n-particles n-particles :expand-k expand-k}
        awopts {:min-width (or min-width 1) :max-width (or max-width (* 2 beam-width))
                :spread-margin (or spread-margin 1.0)}
        pop0  (mapv #(init-particle % observations opts)
                    (or init-specs [init-spec]))
        better (fn [a b] (if (neg? (compare (particle-rank a) (particle-rank b))) a b))]
    (loop [pop  pop0
           t    0
           best (reduce better (first pop0) (rest pop0))
           diag []]
      (let [all-done? (every? :done? pop)]
        (if (or all-done? (>= t max-steps))
          (let [final-best best
                refined (when (and backtrack? (syn/scored? (:feedback final-best)))
                          (backtrack-refine final-best observations propose
                                            (assoc opts :max-steps max-steps :plateau-eps plateau-eps)))
                best' (if (:improved? refined) (:result refined) final-best)]
            {:best best' :population pop :trajectory (:trajectory best')
             :steps t :stop-reason (if all-done? :plateau :max-steps)
             :diagnostics diag
             :backtrack (when refined (dissoc refined :result))})
          (let [pool    (dedup-pool (vec (mapcat #(expand % propose observations opts) pop)))
                spread  (pool-spread pool)
                width   (if adaptive? (adapt-width beam-width spread awopts) beam-width)
                pop'    (if (= strategy :smc)
                          (select-smc pool width (+ seed t) temperature)
                          (select-beam pool width))
                ;; track the global best over the FULL pre-selection pool, not just the
                ;; selected pop' — :smc resampling (or a narrow beam) can drop the best
                ;; particle from the active population, but it must never be lost.
                best'   (reduce better best pool)
                row     {:t (inc t) :pool (count pool) :width width
                         :spread (when (js/isFinite spread) spread)
                         :best-evidence (:evidence best') :n-done (count (filter :done? pop'))}]
            (recur pop' (inc t) best' (conj diag row))))))))

;; ===========================================================================
;; 6. backtrack-refine — deterministic backtracking (best-first re-search) over a trajectory
;;
;; The design's "backtracking = reconsider an earlier decision given later feedback"
;; (genmlx-47zx). OPERATIONAL realization: for each earlier decision point in the best
;; particle's trajectory, re-open it — take a ROAD NOT TAKEN (an alternative proposed
;; edit ≠ the one originally accepted) from the spec BEFORE that step, greedy-continue
;; it, and keep the best-evidence result. This lets even a NARROW search (beam-width 1 =
;; greedy) escape a structural lock-in (e.g. the Phase-1 linreg no-intercept commitment).
;; It is DETERMINISTIC best-first re-search, NOT MCMC (no stochastic accept/reject); it
;; operates on the trajectory directly rather than through a materialized programmer-GF +
;; GenMLX p/regenerate (the stochastic regenerate move — the §2 fixpoint, deferred).
;; ===========================================================================

(defn- greedy-from
  "Greedy continuation from `spec`: beam-width 1, no adaptation/backtrack — the Phase-1
   driver expressed in the search loop, used as the re-proposal continuation."
  [spec observations propose opts]
  (:best (search (merge opts {:init-spec spec :observations observations :propose propose
                              :beam-width 1 :adaptive? false :backtrack? false}))))

(defn backtrack-refine
  "Re-open each earlier decision in `particle`'s trajectory: at each step take a road
   NOT taken (an alternative proposed edit ≠ the one originally accepted) from the spec
   BEFORE that step, greedy-continue it, and keep the best result (method-aware rank —
   an exact alternative never loses to a noisy IS one). Taking the alternative — not
   re-running greedy, which would re-commit to the same locally-best move — is what
   escapes a step-1 structural lock-in. The returned :result's :trajectory SPLICES the
   original prefix in front of the re-search continuation (with the re-opened edit
   relabeled :reopened?), so it reads as the full development path, not a fresh run.
   Returns {:improved? :result :from :to :reopened-at}; :result is the better particle,
   or the original when nothing beats it."
  [particle observations propose opts]
  (let [traj  (:trajectory particle)
        alts  (for [t     (range 1 (count traj))
                    :let  [prefix    (:spec (nth traj (dec t)))
                           prefix-fb (syn/check (syn/render prefix) observations opts)
                           taken     (:code (nth traj t))]
                    c     (propose prefix prefix-fb)
                    :let  [alt-spec (:spec' c)
                           alt-code (syn/render alt-spec)]
                    :when (not= alt-code taken)            ; a road not taken at step t
                    :let  [res (greedy-from alt-spec observations propose opts)]
                    :when (syn/scored? (:feedback res))]
                {:res res :reopened-at t :edit (:edit c) :desc (:desc c)
                 :prefix-rows (subvec traj 0 t)})
        best  (first (sort-by (comp particle-rank :res) alts))]
    (if (and best
             (> (or (:evidence (:res best)) ##-Inf)
                (+ (or (:evidence particle) ##-Inf) (or (:plateau-eps opts) 0.05))))
      (let [res      (:res best)
            cont     (:trajectory res)                       ; re-search path (cont[0] = its :init)
            reopened (assoc (first cont) :edit (:edit best) :desc (:desc best) :reopened? true)
            rows     (into (vec (:prefix-rows best)) (cons reopened (rest cont)))
            spliced  (vec (map-indexed (fn [i r] (assoc r :step i)) rows))]
        {:improved? true :result (assoc res :trajectory spliced)
         :from (:evidence particle) :to (:evidence res) :reopened-at (:reopened-at best)})
      {:improved? false :result particle})))
