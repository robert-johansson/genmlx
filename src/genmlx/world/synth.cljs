(ns genmlx.world.synth
  "Phase 1 of the north star (beans genmlx-77vv / genmlx-n74t): the minimal
   REPL-driven program-synthesis KERNEL — *the programmer as a GenMLX model*.

   THE IDEA (docs/programmer-as-genmlx-model.md). Whole-program one-shot best-of-K
   CLIFFS to ~0/16 on advanced models (linreg/hier/gmm/kalman) for BOTH a 0.8B AND a
   35B teacher — it is the one-shot INTERFACE, not capability (each failure is a
   one-eval-away DSL slip on a structurally-correct model). The lever is the closed
   FEEDBACK loop a generic code LLM lacks: GenMLX gives (a) exact per-PARTIAL model
   evidence (a calibrated model-selection gradient — experiment A) and (b) in-process
   SCI eval (parse/eval/error/value per step). So synthesis becomes: build the model
   INCREMENTALLY by small edits, CHECK each partial with the exact oracle, and let the
   oracle SELECT which edit to keep and WHEN to stop (add structure while evidence
   climbs, stop on plateau — self-terminating, resource-rational).

   THIS NAMESPACE IS NATIVE-FREE. Like its siblings genmlx.world.distill and
   genmlx.world.train-reward it reuses the same oracle spine — codegen.eval (edamame
   reader + SCI) and msa-score (Bayesian model evidence) — and never loads the policy
   LLM. The proposer is INJECTED as a plain `(fn [spec feedback] -> [candidate ...])`,
   so the driver is identical whether the proposer is the structured move-vocabulary
   used to isolate the loop (experiment B) or a real LLM wired in by a caller.

   DIVISION OF LABOUR ACROSS THE PHASES (kept honest here):
     - Phase 1 (this) proves the LOOP: the four-level self-check, the form-level edit
       ops, and a greedy oracle-driven driver that climbs evidence and self-terminates.
       The proposer's move VOCABULARY is provided.
     - Phase 2 SEARCHES that vocabulary (SMC/beam over construction steps).
     - Phase 3 LEARNS the proposer (REPL-trace SFT + GRPO on the 0.8B).
     - Phase 4 is the metareasoner (genmlx.control) over the control sites.

   A NOTE ON GFI edit/CompositeEdit. genmlx.edit's edits (Constraint/Selection/
   ArgsUpdate/Proposal/Composite) rewrite a *trace* of a FIXED model and carry a GFI
   weight; they are the inference-time analog. Building the MODEL itself changes the
   program's SOURCE FORM, a different layer — so the construction edits here are
   form-level (the homoiconic 'code is data' path, design-doc §5), not trace edits.

   Sections:
     1. Model spec — a partial GenMLX program as data
     2. Rendering — spec -> a (fn [trace] ...) code string
     3. Edit operations — pure spec -> spec moves (the typed action space)
     4. The check node — the four-level self-check (the programmer's senses)
     5. The greedy REPL driver — propose -> check -> accept / backtrack / stop
     6. Proposer helpers — generic parameter-refinement moves + combinators"
  (:require [genmlx.llm.msa-score :as score]
            [genmlx.codegen.eval :as ce]
            [genmlx.world.train-reward :as reward]
            [genmlx.schema :as schema]
            [clojure.string :as str]))

;; ===========================================================================
;; 1. Model spec — a partial GenMLX program as data
;;
;; A partial model is a VALUE: an ordered list of latent trace sites (rendered as
;; let-bindings) and an ordered list of observation trace sites (rendered as the
;; returned map). A SITE is data: an address keyword, a distribution name, and a
;; vector of argument FORMS (numbers, latent-referencing symbols, or s-expressions
;; — code is data). A latent additionally carries the :sym other sites reference.
;;
;;   latent {:sym 'slope :addr :slope :dist "gaussian" :args [0 3]}
;;   obs    {:addr :y0    :dist "gaussian"
;;           :args [(list 'mx/add (list 'mx/multiply 'slope (list 'mx/scalar 0.0))
;;                        'intercept) 1]}
;;   spec   {:latents [latent ...] :obs [obs ...]}
;; ===========================================================================

(defn latent
  "A latent trace site (a let-binding). `sym` is the binding symbol later sites
   reference; `addr` its trace address (defaults to (keyword sym)); `dist` the
   distribution constructor name (string, e.g. \"gaussian\"); `args` a vector of
   argument forms."
  ([sym dist args] (latent sym (keyword sym) dist args))
  ([sym addr dist args] {:sym sym :addr addr :dist dist :args (vec args)}))

(defn obs
  "An observation trace site (an entry in the returned map). `addr` the observed
   address; `dist` the distribution name; `args` a vector of argument forms (the
   first is the mean expression, the last the noise scale, by GenMLX convention)."
  [addr dist args]
  {:addr addr :dist dist :args (vec args)})

(defn spec
  "Assemble a model spec from a seq of latents and a seq of obs sites."
  [latents obs-sites]
  {:latents (vec latents) :obs (vec obs-sites)})

(defn site-by-addr
  "Find the [collection-key index site] of the trace site with `addr`, searching
   latents then obs, or nil. Used by the edit ops to target a site generically.
   INVARIANT: latent and obs addresses must be DISJOINT (they share one trace-address
   namespace, so a collision would render a duplicate address and shadow the obs here).
   Constructors keep them disjoint by convention (latents :mu/:slope, obs :y*/:a*)."
  [{:keys [latents obs]} addr]
  (or (some (fn [[i s]] (when (= addr (:addr s)) [:latents i s]))
            (map-indexed vector latents))
      (some (fn [[i s]] (when (= addr (:addr s)) [:obs i s]))
            (map-indexed vector obs))))

;; ===========================================================================
;; 2. Rendering — spec -> a (fn [trace] ...) code string
;;
;; The rendered string is exactly the shape the oracle expects (and the shape the
;; advanced-probe reference programs use): (fn [trace] (let [<bindings>] {<obs>})).
;; Arg forms are emitted with pr-str (a number renders as a literal, a symbol as a
;; bare name resolving to a let-binding, an s-expression as itself).
;; ===========================================================================

(defn- render-arg
  "Render one distribution argument FORM to source text."
  [a]
  (pr-str a))

(defn- render-dist
  "Render a (dist/NAME args...) constructor call from a site's :dist and :args."
  [{:keys [dist args]}]
  (str "(dist/" dist
       (when (seq args) (str " " (str/join " " (map render-arg args))))
       ")"))

(defn- render-site
  "Render a (trace :addr (dist/NAME args...)) form for a site."
  [{:keys [addr] :as site}]
  (str "(trace " (pr-str addr) " " (render-dist site) ")"))

(defn render
  "Render a model spec to a complete (fn [trace] (let [...] {...})) code string —
   the input the check node parses, evals, and scores."
  [{:keys [latents obs]}]
  (let [bindings (str/join " " (for [l latents] (str (:sym l) " " (render-site l))))
        body     (str/join " " (for [o obs] (str (pr-str (:addr o)) " " (render-site o))))]
    (str "(fn [trace] (let [" bindings "] {" body "}))")))

;; ===========================================================================
;; 3. Edit operations — pure spec -> spec moves (the typed action space)
;;
;; Each op is a pure spec -> spec function: the homoiconic build-up moves the design
;; doc names. add-latent / add-obs / set-args / set-mean / set-noise are the
;; load-bearing moves for the advanced models. The named `wrap-combinator` MOVE is
;; deferred — folding repeated sites into a Map combinator needs the synthesis SCI
;; sandbox to expose the combinators, and the three exact targets are flat — so this
;; section ships `homogeneous-obs?`, the pure predicate that DETECTS the fold
;; opportunity, instead of a rewrite that would render an un-evaluable form.
;; ===========================================================================

(defn add-latent
  "Append a latent trace site to the spec (a new let-binding)."
  [spec lat]
  (update spec :latents (fnil conj []) lat))

(defn add-obs
  "Append (or, when `addr` already exists, REPLACE) an observation trace site."
  [spec ob]
  (update spec :obs
          (fn [obs] (let [obs (vec obs)
                          i   (first (keep-indexed #(when (= (:addr %2) (:addr ob)) %1) obs))]
                      (if i (assoc obs i ob) (conj obs ob))))))

(defn set-args
  "Replace the full argument vector of the site at `addr` (latent or obs). The
   canonical 'set-prior' move on a latent, and the general parameter edit."
  [spec addr new-args]
  (if-let [[coll i _] (site-by-addr spec addr)]
    (assoc-in spec [coll i :args] (vec new-args))
    spec))

(defn set-mean
  "Replace the MEAN (first argument) of the site at `addr`, keeping its remaining
   args. The move that re-points an observation at a latent-dependent mean (e.g.
   shared-mean -> slope*x+intercept, or pooled -> per-group)."
  [spec addr mean-form]
  (if-let [[coll i s] (site-by-addr spec addr)]
    (assoc-in spec [coll i :args] (assoc (vec (:args s)) 0 mean-form))
    spec))

(defn set-noise
  "Replace the NOISE (last argument) of the site at `addr`, keeping its mean. The
   parameter-refinement move on an observation's scale. Expects a [mean noise] site
   (>= 2 args); a site with fewer args has no distinct noise slot, so it is left
   unchanged rather than clobber the mean."
  [spec addr noise]
  (if-let [[coll i s] (site-by-addr spec addr)]
    (let [args (vec (:args s))]
      (if (< (count args) 2)
        spec
        (assoc-in spec [coll i :args] (assoc args (dec (count args)) noise))))
    spec))

(defn homogeneous-obs?
  "True iff every observation site shares the same distribution, the same noise (last
   arg), and the SAME single LATENT symbol as its mean — the precondition for folding
   them into one Map combinator. A pure QUERY, not a rewrite: the actual fold awaits
   the synthesis SCI sandbox exposing the combinators (the three advanced targets are
   heterogeneous + flat, so it is not needed here). Kept honest — it detects the
   fold opportunity rather than render a combinator form that would not eval."
  [{:keys [obs latents]}]
  (let [lat-syms (set (map :sym latents))
        m0       (first (:args (first obs)))]
    (boolean (and (seq obs)
                  (every? #(>= (count (:args %)) 2) obs)
                  (apply = (map :dist obs))
                  (apply = (map (comp last :args) obs))
                  (symbol? m0) (contains? lat-syms m0)
                  (every? #(= m0 (first (:args %))) obs)))))

;; ===========================================================================
;; 4. The check node — the four-level self-check (the programmer's senses)
;;
;; (partial-program code) -> a feedback map. Four altitudes, each gating the next;
;; the fitness level is EXACT math, never an LLM-judge:
;;   syntax    :parses?    edamame reader      — a complete cljs form?
;;   semantics :schema-ok? GenMLX schema       — a well-formed MODEL? (sites, returns
;;                                                a map, no delta point-mass hack)
;;   coverage  :covered?   schema vs data       — every observed addr is a trace site?
;;   behavior  :evals?/:error  SCI eval         — runs to a DynamicGF / what errored?
;;   fit       :evidence/:method  the oracle    — Bayesian model evidence of the data
;; (:delta is relative to a PRIOR accepted state, so it is computed by the driver,
;; not by a single check.)
;; ===========================================================================

(defn- uses-delta?
  "Does the form CALL a `delta` distribution constructor anywhere — a list whose head
   symbol is `delta` / `dist/delta`? A delta point-mass at an observed site is a
   degenerate reward-hack (log-evidence ~0 beats any honest noisy fit). Matches only a
   constructor in HEAD position, so an honest latent/variable merely NAMED `delta` (a
   common offset/difference name) is NOT rejected — narrower (and more precise) than
   genmlx.world.distill/form-uses-delta?, which flags any mention of the name."
  [form]
  (cond
    (seq? form)  (or (and (symbol? (first form)) (= "delta" (name (first form))))
                     (boolean (some uses-delta? form)))
    (coll? form) (boolean (some uses-delta? form))
    :else        false))

(defn- result-form
  "The expression a model body ultimately RETURNS: unwrap let/let*/do wrappers to
   their tail form (the schema's :return-form is the outer let, whose tail is the
   observation map)."
  [form]
  (if (and (seq? form) (contains? #{'let 'let* 'do} (first form)))
    (recur (last form))
    form))

(defn- eval-gf
  "Eval a model code string to a DynamicGF, CAPTURING any error message (msa-score's
   eval-model swallows the error to nil). {:gf gf} on success, {:error msg} otherwise."
  [code]
  (try
    (let [f (score/eval-model-fn code)]
      (if (fn? f)
        {:gf (score/wrap-model f (score/code->source-form code))}
        {:error "result is not a (fn [trace] ...) form"}))
    (catch :default e {:error (.-message e)})))

(defn check
  "Run the four-level self-check on a candidate partial program against
   `observations`. Returns a uniform feedback map (every key present, nil where a
   level was not reached):

     {:code :parses? :schema-ok? :n-sites :returns-map? :uses-delta?
      :covered? :evals? :error :evidence :method}

   `:evidence` is the exact/IS Bayesian model log-evidence (nil if not scored).
   opts: {:n-particles n}  importance samples for the non-conjugate IS fallback."
  ([code observations] (check code observations {}))
  ([code observations {:keys [n-particles] :or {n-particles 2000}}]
   (let [code       (str/trim (str code))
         parses?    (ce/valid-cljs? code)
         src        (when parses? (score/code->source-form code))
         sm         (when src (schema/extract-schema src))
         sites      (:trace-sites sm)
         site-addrs (set (map :addr sites))
         returns-map? (boolean (and sm (map? (result-form (:return-form sm)))))
         delta?     (boolean (when-let [f (ce/parse-form code)] (uses-delta? f)))
         n-sites    (count sites)
         schema-ok? (boolean (and src (pos? n-sites) returns-map? (not delta?)))
         base       {:code code :parses? parses? :schema-ok? schema-ok?
                     :n-sites n-sites :returns-map? returns-map? :uses-delta? delta?
                     :covered? false :evals? false :error nil :evidence nil :method nil}]
     (cond
       (not parses?)
       (assoc base :error "does not parse as a complete ClojureScript form")

       (not schema-ok?)
       (assoc base :error (cond delta?            "uses a delta (point-mass) distribution at a site"
                                (not returns-map?) "model body does not return an observation map"
                                (zero? n-sites)    "no trace sites — not a model"
                                :else              "not a well-formed model form"))

       ;; No data is a misconfiguration, NOT a perfectly-explained model: scoring an
       ;; empty choicemap returns log p({}) = 0, the best possible evidence, which
       ;; would dominate any real candidate. Leave it unscored (scored? false).
       (empty? observations)
       (assoc base :error "no observations to score against")

       (not (every? site-addrs (keys observations)))
       (assoc base :error "an observed address is not a trace site (uncovered)")

       :else
       (let [{:keys [gf error]} (eval-gf code)]
         (if error
           (assoc base :covered? true :evals? false :error error)
           (let [{:keys [log-ml method]} (score/score-model* gf observations {:n-particles n-particles})
                 ok? (reward/finite? log-ml)]
             ;; only report :method for a verdict that was VALIDLY scored — do not
             ;; leak a method label alongside a nil (non-finite) evidence.
             (assoc base :covered? true :evals? true
                    :evidence (when ok? log-ml) :method (when ok? method)
                    :error (when-not ok? "model evidence is non-finite")))))))))

(defn scored?
  "True iff a check verdict reached a finite model evidence (the only candidates the
   driver compares / accepts)."
  [feedback]
  (and (:evals? feedback) (some? (:evidence feedback))))

;; ===========================================================================
;; 5. The greedy REPL driver — propose -> check -> accept / backtrack / stop
;;
;; A PROPOSER is an injected (fn [spec feedback] -> [candidate ...]) where a
;; candidate is {:edit <kw> :desc <str> :spec' <spec>} (it is conditioned on the
;; previous step's feedback, satisfying 'condition each step on verified feedback').
;; Each step renders + checks every candidate, ACCEPTS the best valid one that
;; improves evidence beyond :plateau-eps (BACKTRACK = a broken/uncovered candidate is
;; simply never the max, so the loop steps past it), and STOPS when none improves
;; (the experiment-A self-terminating rule — 'add while evidence climbs').
;;
;; CANDIDATE RANKING is method-aware (mirrors genmlx.world.distill/select-key): an
;; EXACT/Kalman analytical evidence is reproducible, so it outranks a noisy
;; importance-sampling estimate that happened to draw high — only then by evidence.
;; CAVEAT: when the WHOLE search is non-conjugate (every candidate IS-scored), the
;; greedy accept compares two noisy estimates and can mis-accept / oscillate on a
;; lucky draw; that regime is Phase-2 SMC territory (a particle population averages
;; out the noise). The three experiment-B targets are exact, so this does not bite.
;; ===========================================================================

(defn- score-candidates
  "Render + check every proposed candidate against the observations. A candidate may
   carry a raw `:code` string (a real-LLM proposer emits the program text directly — see
   genmlx.world.llm-proposer); that raw code is checked VERBATIM so the LLM's DSL slips
   reach the check node exactly as written. Otherwise the code is rendered from `:spec'`
   (the structured-proposer path)."
  [candidates observations opts]
  (for [c candidates
        :let [code (or (:code c) (render (:spec' c)))
              fb   (check code observations opts)]]
    (assoc c :code code :feedback fb :evidence (:evidence fb))))

(defn- deterministic-method?
  "True iff a verdict's evidence came from a reproducible analytical method (not IS)."
  [feedback]
  (contains? #{:exact :kalman} (:method feedback)))

(defn- candidate-rank
  "Ascending sort key for VALID candidates: deterministically-scored first (an exact
   marginal never loses to a noisy IS estimate), then higher evidence."
  [c]
  [(if (deterministic-method? (:feedback c)) 0 1) (- (:evidence c))])

(defn step
  "One greedy REPL step from `state` ({:spec :feedback}). Returns
   {:candidates [...] :best <candidate-or-nil> :improved? bool}: the best VALID
   candidate (method-aware rank) whose evidence beats the current by more than
   :plateau-eps, if any."
  [{:keys [feedback]} candidates observations {:keys [plateau-eps] :or {plateau-eps 0.05} :as opts}]
  (let [cur-ev (:evidence feedback)
        scored (score-candidates candidates observations opts)
        valid  (filter (comp scored? :feedback) scored)
        best   (first (sort-by candidate-rank valid))]
    {:candidates scored
     :best best
     :improved? (boolean (and best (or (nil? cur-ev)
                                       (> (:evidence best) (+ cur-ev plateau-eps)))))}))

(defn synthesize
  "Run the greedy REPL synthesis loop.

   opts:
     :init-spec     the starting partial model spec (a crude covering model)
     :observations  {:addr value ...} the data the oracle scores against
     :propose       (fn [spec feedback] -> [{:edit :desc :spec'} ...])
     :max-steps     hard cap on accepted edits (default 12)
     :plateau-eps   min evidence gain to accept an edit / not stop (default 0.05)
     :n-particles   IS samples for the non-conjugate fallback (default 2000)

   Returns {:spec :code :feedback :trajectory :stop-reason :steps}, where :stop-reason
   is :plateau (no edit improves a fitted model), :stuck (never reached a scored model)
   or :max-steps. The trajectory is the development trace: one row per state with the
   edit taken, the chosen code, its exact evidence, the scoring :method, and the :delta
   from the prior state — the reportable artifact AND (design-doc §2) the shape of the
   future SFT corpus."
  [{:keys [init-spec observations propose max-steps plateau-eps n-particles]
    :or   {max-steps 12 plateau-eps 0.05 n-particles 2000}}]
  (let [opts      {:plateau-eps plateau-eps :n-particles n-particles}
        init-code (render init-spec)
        init-fb   (check init-code observations opts)]
    (loop [state {:spec init-spec :feedback init-fb}
           t     0
           traj  [{:step 0 :edit :init :desc "crude covering model"
                   :code init-code :evidence (:evidence init-fb)
                   :method (:method init-fb) :delta nil :accepted? true
                   :n-candidates 0}]]
      (if (>= t max-steps)
        ;; report the VERBATIM checked code (the feedback's :code — what actually scored);
        ;; falls back to render. For the structured proposer these are identical (the spec
        ;; round-trips); for a real-LLM proposer the verbatim code is authoritative even if
        ;; its parsed :spec' is an imperfect round-trip (genmlx.world.llm-proposer).
        {:spec (:spec state) :code (or (:code (:feedback state)) (render (:spec state)))
         :feedback (:feedback state) :trajectory traj :stop-reason :max-steps :steps t}
        (let [cands (vec (propose (:spec state) (:feedback state)))
              {:keys [best improved?]} (step state cands observations opts)]
          (if-not improved?
            ;; :plateau = stopped at a genuinely scored model no edit improves;
            ;; :stuck = the current state never reached a finite evidence AND no valid
            ;; candidate did either (a structurally-incomplete init the proposer could
            ;; not complete) — NOT a fitted plateau, so report it distinctly.
            {:spec (:spec state) :code (or (:code (:feedback state)) (render (:spec state)))
             :feedback (:feedback state) :trajectory traj :steps t
             :stop-reason (if (scored? (:feedback state)) :plateau :stuck)}
            (let [prev-ev (:evidence (:feedback state))
                  delta   (when (and prev-ev (:evidence best)) (- (:evidence best) prev-ev))]
              (recur {:spec (:spec' best) :feedback (:feedback best)}
                     (inc t)
                     (conj traj {:step (inc t) :edit (:edit best) :desc (:desc best)
                                 :code (:code best) :evidence (:evidence best)
                                 :method (:method (:feedback best)) :delta delta
                                 :accepted? true :n-candidates (count cands)})))))))))

;; ===========================================================================
;; 6. Proposer helpers — generic parameter-refinement moves + combinators
;;
;; These are the structure-NEUTRAL moves any proposer can offer: tune an existing
;; site's noise / prior over a small grid. A caller composes them with the
;; (task-shaped, Phase-1-provided) STRUCTURAL moves via `union-proposer`. Phase 2
;; replaces the provided vocabulary with a search; Phase 3 with a learned LLM.
;; ===========================================================================

(defn noise-refinements
  "Candidate edits that set each observation site's noise to each value in `grid` —
   the parameter-refinement vocabulary. Skips the no-op (current value)."
  ([spec] (noise-refinements spec [0.3 0.5 0.7 1 1.5 2.5]))
  ([spec grid]
   (for [o (:obs spec)
         g grid
         :when (not= g (last (:args o)))]
     {:edit :set-noise :desc (str "set " (:addr o) " noise -> " g)
      :spec' (set-noise spec (:addr o) g)})))

(defn prior-std-refinements
  "Candidate edits that set each latent's prior std (its last arg) to each value in
   `grid` — refines how tightly a latent is regularized."
  ([spec] (prior-std-refinements spec [1 2 5]))
  ([spec grid]
   (for [l (:latents spec)
         g grid
         :when (not= g (last (:args l)))]
     {:edit :set-prior :desc (str "set " (:addr l) " prior-std -> " g)
      :spec' (set-args spec (:addr l) (assoc (vec (:args l)) (dec (count (:args l))) g))})))

(defn union-proposer
  "Combine several proposers into one whose candidates are the concatenation of all
   their candidates (deduped WITHIN a step by rendered code so identical moves are not
   re-scored). It does not dedup across steps, but that is harmless: a re-offered
   already-accepted move renders to the current state, so its evidence does not beat
   the current by :plateau-eps and it is never re-accepted."
  [& proposers]
  (fn [spec feedback]
    (let [cands (mapcat (fn [p] (p spec feedback)) proposers)]
      (->> cands
           (reduce (fn [[seen acc] c]
                     (let [k (render (:spec' c))]
                       (if (contains? seen k) [seen acc] [(conj seen k) (conj acc c)])))
                   [#{} []])
           second))))
