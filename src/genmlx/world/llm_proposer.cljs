(ns genmlx.world.llm-proposer
  "A REAL-LLM proposer for the REPL-synthesis loop (beans genmlx-0yv7 / genmlx-wpua):
   the design's mechanism (a) — INSTRUCT-DRIVEN, FEEDBACK-CONDITIONED, STEPWISE.

   This is the piece that puts a real LLM in the loop. `genmlx.world.synth` (Phase 1)
   and `genmlx.world.search` (Phase 2) take an INJECTED proposer
   `(fn [spec feedback] -> [{:edit :desc :spec' ...} ...])`; until now that slot held a
   hand-coded structured move-vocabulary, so feedback-conditioning was plumbed but never
   exercised. `make-proposer` here fills the slot with an LLM:

     render the current partial model + its check feedback  ->  PROMPT
     call the policy LLM K times (temperature)              ->  K completions
     extract the (fn [trace] ...) form from each            ->  candidate :code
     parse it (best-effort) back to a synth spec            ->  candidate :spec'

   The candidates flow into the SAME four-level `check` node and oracle — so a real DSL
   slip (mx/0 unbound, a dist bound without `trace`, (:p trace)->nil, a non-static loop)
   reaches the check node as the LLM actually wrote it (the check node's raison d'être),
   producing real :error feedback that the NEXT step is conditioned on. That closed loop
   is the north-star thesis; this namespace is where the loop meets a real generator.

   NATIVE-FREE, like the oracle spine it complements. The policy LLM lives OUT-OF-PROCESS
   (a resident `scripts/llm_server.py` mlx-lm worker); `call-server` is the one I/O
   boundary (a synchronous curl — the driver is synchronous, and an LLM call is a genuine
   I/O boundary). The pure core — `extract-form`, `parse-spec`, the prompt builders, and
   `make-proposer` itself (which takes an injected `:call-llm`) — needs no model and is
   unit-tested with a mock generator.

   Sections:
     1. The DSL system prompt — teaches the exact skeleton + the anti-cliff rules
     2. Prompt builders — task block / step (feedback-conditioned) / one-shot (control)
     3. extract-form — LLM completion -> a (fn [trace] ...) code string (the slips kept)
     4. parse-spec — code string -> a synth spec (the inverse of synth/render)
     5. make-proposer / one-shot-candidates — the injected proposer + the control
     6. call-server — the out-of-process mlx-lm bridge (the sole I/O)"
  (:require [genmlx.world.synth :as syn]
            [genmlx.codegen.eval :as ce]
            [clojure.string :as str]))

;; ===========================================================================
;; 1. The DSL system prompt
;;
;; The cliff (genmlx-d4q4) was the one-shot INTERFACE, not capability: a model that
;; understands the structure still emits one-eval-away GenMLX slips. This system prompt
;; teaches the EXACT skeleton + the precise rules that each documented slip violates.
;; It is given IDENTICALLY to the loop and the one-shot control, so the only difference
;; the experiment measures is the feedback loop, never the prompt.
;; ===========================================================================

(def default-system
  (str
   "You are an expert ClojureScript programmer writing probabilistic models in GenMLX.\n"
   "A GenMLX model is a single ClojureScript function: a (fn [trace] ...) whose body is a\n"
   "`let` of latent variables and returns a map of observations (see the example below).\n\n"
   "RULES (each one is a real mistake to avoid):\n"
   "1. Every random variable is  (trace :address (dist/NAME arg ...)).  Latents are\n"
   "   let-bindings; observations are entries in the RETURNED MAP (a keyword key whose\n"
   "   value is a trace form). A `dist/...` bound WITHOUT `trace` is NOT a site.\n"
   "2. Reference a latent in a later expression by its binding SYMBOL (e.g. slope) —\n"
   "   NEVER by (:slope trace); `trace` is not a map.\n"
   "3. Numbers are PLAIN literals:  (dist/gaussian 0 2.5).  NEVER write mx/0 or mx/2.5.\n"
   "4. To compute a mean from latents use mx ops:\n"
   "     (mx/add (mx/multiply slope (mx/scalar 2.0)) intercept)\n"
   "   mx/scalar wraps a constant; +,*,-,/ are mx/add, mx/multiply, mx/subtract, mx/divide.\n"
   "5. Distributions: (dist/gaussian mean std) (dist/uniform lo hi) (dist/bernoulli p)\n"
   "   (dist/exponential rate) (dist/beta-dist a b) (dist/gamma-dist shape rate).\n"
   "6. NO loops (doseq/for/map), NO combinators, NO inference calls. Write EVERY site\n"
   "   explicitly and flat. The form must be STATIC: literal keyword addresses, no `if`.\n"
   "7. To be scored EXACTLY (not approximately), an observation must be\n"
   "   (trace :addr (dist/gaussian MEAN NOISE)) where (a) NOISE is a FIXED positive number\n"
   "   (e.g. 1.0) — NOT a latent (trace :sigma ...) — and (b) MEAN is written DIRECTLY\n"
   "   inside the (dist/gaussian ...) call as an mx expression of the latents, NOT bound to\n"
   "   a separate let variable. Example of an exactly-scoreable observation:\n"
   "     (trace :y3 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar 3.0)) intercept) 1.0))\n"
   "   A latent noise or a let-factored mean falls back to noisy approximate scoring, so the\n"
   "   model looks worse than it is. (Try a smaller fixed noise if the data is tightly clustered.)\n\n"
   "Example (a complete, valid 1-latent model — yours will have DIFFERENT structure):\n"
   "(fn [trace]\n"
   "  (let [mu (trace :mu (dist/gaussian 0 5))]\n"
   "    {:o0 (trace :o0 (dist/gaussian mu 1.0))\n"
   "     :o1 (trace :o1 (dist/gaussian mu 1.0))}))\n\n"
   "Respond with ONLY the model: one ```clojure code block with a single (fn [trace] ...) form."))

;; ===========================================================================
;; 2. Prompt builders
;; ===========================================================================

(defn- fmt [x] (when (and x (js/isFinite x)) (.toFixed (js/Number x) 2)))

(defn- render-obs
  "Render the observed data as readable `:addr = value` lines (sorted by address)."
  [observations]
  (->> observations
       (sort-by (comp str key))
       (map (fn [[k v]] (str "  " k " = " v)))
       (str/join "\n")))

(defn- feedback-line
  "Turn a check feedback map into the one instruction the LLM is conditioned on. This is
   the defining feedback-conditioning step: the model's NEXT proposal sees the exact
   verifier verdict on its PREVIOUS one (a parse failure, a coverage gap, an eval error,
   or a model-evidence number to beat)."
  [fb]
  (cond
    (nil? fb)              "(no current model yet — propose an initial model)"
    (not (:parses? fb))    "The current model does NOT parse as a complete ClojureScript form. Return a complete (fn [trace] ...) form."
    (not (:schema-ok? fb)) (str "The current model is not a well-formed GenMLX model: " (:error fb)
                                ". Every latent and observation must be a (trace :addr (dist/... )) site, and the body must return a map of observations.")
    (not (:covered? fb))   (str "Coverage problem: " (:error fb)
                                ". Every observed address below must appear as a (trace :addr (dist/...)) entry in the returned map.")
    (not (:evals? fb))     (str "Evaluating the current model RAISED AN ERROR: " (:error fb)
                                ". Fix exactly this error in your next model (check rule 2/3/4 above).")
    (nil? (:evidence fb))  (str "The model is well-formed and evaluates, but its model evidence is NON-FINITE "
                                "(the data could not be scored). The usual cause is an invalid distribution "
                                "parameter: a standard deviation / scale that can be non-positive — e.g. a "
                                "(dist/gaussian ..) PRIOR on a noise scale, which can go negative. Use a FIXED "
                                "positive number (like 1.0) for observation noise, or a positive-support prior "
                                "(dist/exponential, dist/gamma-dist); and make sure every distribution name is real.")
    (:evidence fb)         (str "The current model is VALID. Its Bayesian model evidence (log marginal likelihood; HIGHER is better) is "
                                (fmt (:evidence fb))
                                ". Propose a model that fits the data BETTER — add or adjust ONE piece of structure (a latent, a coupling, a per-group mean, a noise scale).")
    :else                  (str "Current status: " (pr-str (select-keys fb [:parses? :schema-ok? :covered? :evals? :error])))))

(defn step-prompt
  "The feedback-conditioned step prompt: task + data + the current model + the verifier's
   verdict on it + the instruction to make ONE targeted improvement."
  [task-desc observations cur-code fb]
  (str "TASK\n" task-desc
       "\n\nOBSERVED DATA\n" (render-obs observations)
       "\n\nCURRENT MODEL\n" cur-code
       "\n\nVERIFIER FEEDBACK\n" (feedback-line fb)
       "\n\nReturn an IMPROVED model as a single (fn [trace] (let [...] {...})) form. "
       "Make ONE targeted change (fix the error above, or add/adjust one piece of structure "
       "so the model fits the data better)."))

(defn oneshot-prompt
  "The one-shot CONTROL prompt: task + data, no current model, no feedback — the model
   writes its best program in one shot (this is the best-of-K baseline the loop must beat)."
  [task-desc observations]
  (str "TASK\n" task-desc
       "\n\nOBSERVED DATA\n" (render-obs observations)
       "\n\nWrite the best GenMLX generative model you can for this data, as a single "
       "(fn [trace] (let [...] {...})) form."))

;; ===========================================================================
;; 3. extract-form — pull the (fn [trace] ...) form out of an LLM completion
;;
;; The slips stay IN: this strips chat scaffolding (a stray <think> block, a ```fence,
;; prose) and returns the model form the LLM actually wrote — mx/0, (:p trace) and all —
;; so the check node sees real garbage, not a sanitized version.
;; ===========================================================================

(defn- strip-think
  "Drop a (defensive) <think>...</think> block — closed, or unclosed to end-of-string."
  [s]
  (-> s
      (str/replace #"<think>[\s\S]*?</think>" "")
      (str/replace #"<think>[\s\S]*$" "")))

(defn- strip-fence
  "If a ```...``` code block is present, return its CONTENTS; else the string unchanged."
  [s]
  (if-let [m (re-find #"```[a-zA-Z]*\n?([\s\S]*?)```" s)] (second m) s))

(defn- balanced-from
  "From index `i` (a `(` in `s`) return the balanced-paren substring, or nil if it never
   closes. String literals are tracked so parens inside \"...\" do not miscount."
  [s i]
  (let [n (count s)]
    (loop [j i, depth 0, in-str? false, esc? false]
      (if (>= j n)
        nil
        (let [c (nth s j)]
          (cond
            (and in-str? esc?)    (recur (inc j) depth true false)
            (and in-str? (= c \\)) (recur (inc j) depth true true)
            (and in-str? (= c \")) (recur (inc j) depth false false)
            in-str?               (recur (inc j) depth true false)
            (= c \")              (recur (inc j) depth true false)
            (= c \()              (recur (inc j) (inc depth) false false)
            (= c \))              (if (= depth 1)
                                    (subs s i (inc j))
                                    (recur (inc j) (dec depth) false false))
            :else                 (recur (inc j) depth false false)))))))

(defn extract-form
  "An LLM completion -> the (fn [trace] ...) model code string it contains, or nil.
   Strips a <think> block and a ``` fence, finds the first (fn ...) form and returns the
   balanced substring (validated by the reader). Slip recovery: if the model emitted a
   bare (let ...) body without the (fn [trace] ...) wrapper, wrap it — the only
   normalization done; everything inside the form is left exactly as written."
  [completion]
  (when (string? completion)
    (let [t  (-> completion strip-think strip-fence str/trim)
          fi (.search t (js/RegExp. "\\(fn\\*?[\\s\\[]"))]
      (if (>= fi 0)
        (let [f (balanced-from t fi)]
          (when (and f (ce/valid-cljs? f)) f))
        (let [li (.search t (js/RegExp. "\\(let\\*?[\\s\\[]"))]
          (when (>= li 0)
            (let [b (balanced-from t li)
                  w (when b (str "(fn [trace] " b ")"))]
              (when (and w (ce/valid-cljs? w)) w))))))))

;; ===========================================================================
;; 4. parse-spec — code string -> a synth spec (the inverse of synth/render)
;;
;; Best-effort: it reads exactly the model class the spec represents (flat latents +
;; an observation map of (trace :addr (dist ...)) sites), preserving argument FORMS
;; verbatim (so an mx/add mean expression round-trips). It returns nil for anything
;; off-grammar (a loop, a non-trace binding, a nested helper) — and that is fine: the
;; candidate still carries its raw :code (what the check node scores), and only ACCEPTED
;; candidates need a spec (a scored, well-formed, static model is always in-grammar).
;; ===========================================================================

(defn- dist-name
  "A dist constructor symbol -> its name string: dist/gaussian -> \"gaussian\"."
  [sym]
  (when (symbol? sym) (name sym)))

(defn- parse-site
  "(trace :addr (dist/NAME arg ...)) -> {:addr :dist :args}, or nil."
  [expr]
  (when (and (seq? expr) (= 'trace (first expr)) (keyword? (second expr)))
    (let [dexpr (nth expr 2 nil)]
      (when (and (seq? dexpr) (symbol? (first dexpr)))
        {:addr (second expr) :dist (dist-name (first dexpr)) :args (vec (rest dexpr))}))))

(defn- collect-let-sites
  "A let binding vector [sym1 e1 sym2 e2 ...] -> [{:sym :addr :dist :args} ...] (one per
   binding), or nil if any binding is not `symbol + a trace-site`. (Both latents and —
   in the common 'everything in the let, return a map of references' shape that strong
   instruct models emit — observations live here; the return map classifies which is which.)"
  [binds]
  (when (and (vector? binds) (even? (count binds)))
    (reduce (fn [acc [sym e]]
              (let [s (parse-site e)]
                (if (and (symbol? sym) s) (conj acc (assoc s :sym sym)) (reduced nil))))
            [] (partition 2 binds))))

(defn- classify-return-map
  "Split `let-sites` into [latents obs] using the return map. A map value that is a SYMBOL
   referencing a let site marks that site an OBSERVATION; a map value that is an inline
   (trace :addr (dist ...)) form is an inline observation (the canonical render shape). A
   returned symbol that is ALSO referenced in another site's args is a latent DEPENDENCY,
   not a fresh observation — it must stay let-bound (in scope for that reference), so its
   return entry is dropped (its trace-site already covers the address); otherwise the
   round-tripped spec would render an unbound symbol. Observations are deduped by address
   (a latent returned under multiple keys collapses to one). Latents are the let sites not
   reclassified as observations. Returns [latents obs] or nil if any map value is neither a
   known-site reference nor an inline trace site."
  [let-sites retmap]
  (when (map? retmap)
    (let [by-sym     (into {} (map (juxt :sym identity)) let-sites)
          referenced (into #{} (mapcat #(filter symbol? (tree-seq coll? seq (:args %)))) let-sites)]
      (when-let [tagged (reduce (fn [acc [_ v]]
                                  (let [inline (parse-site v)]
                                    (cond
                                      (and (symbol? v) (referenced v)) acc   ; latent dependency, not a fresh obs
                                      (and (symbol? v) (by-sym v))     (conj acc (assoc (by-sym v) :ref (by-sym v)))
                                      inline                           (conj acc inline)
                                      :else                            (reduced nil))))
                                [] retmap)]
        (let [obs-syms (set (keep #(:sym (:ref %)) tagged))
              latents  (remove #(obs-syms (:sym %)) let-sites)
              obs      (->> tagged
                            (reduce (fn [[seen acc] s]
                                      (if (seen (:addr s))
                                        [seen acc]
                                        [(conj seen (:addr s)) (conj acc (syn/obs (:addr s) (:dist s) (:args s)))]))
                                    [#{} []])
                            second)]
          [(mapv #(syn/latent (:sym %) (:addr %) (:dist %) (:args %)) latents) obs])))))

(defn parse-spec
  "A (fn [trace] (let [...] {...})) code string -> a synth spec, or nil if off-grammar.
   Unwraps let/let*/do; handles the canonical shape (latents in the let, observations
   inline in the return map) AND the 'everything in the let, return a map of references'
   shape; supports the no-latent case (a bare observation-map body). Arg FORMS are
   preserved verbatim, so a mean expression round-trips."
  [code]
  (let [form (ce/parse-form code)]
    (when (and (seq? form) (#{'fn 'fn*} (first form)) (vector? (second form)))
      (loop [b (last (drop 2 form))]
        (cond
          (map? b)
          (when-let [[lat obs] (classify-return-map [] b)] (syn/spec lat obs))

          (and (seq? b) (#{'let 'let*} (first b)) (vector? (second b)))
          (when-let [sites (collect-let-sites (second b))]
            (when-let [[lat obs] (classify-return-map sites (last b))] (syn/spec lat obs)))

          (and (seq? b) (= 'do (first b)))
          (recur (last b))

          :else nil)))))

;; ===========================================================================
;; 5. make-proposer / one-shot-candidates
;; ===========================================================================

(defn- candidate-desc [sp]
  (if sp
    (str "LLM: " (count (:latents sp)) "L/" (count (:obs sp)) "O")
    "LLM: off-grammar (raw)"))

(defn progress-level
  "How far a check verdict got, 0..5 — the depth of the self-check a candidate cleared.
   Used to pick the MOST-PROGRESSED failed candidate to revise from (the one whose error
   is the most actionable): 0 not-parse, 1 not-schema, 2 not-covered, 3 eval-error,
   4 evaluates-but-non-finite-evidence, 5 scored."
  [fb]
  (cond (not (:parses? fb))    0
        (not (:schema-ok? fb)) 1
        (not (:covered? fb))   2
        (not (:evals? fb))     3
        (nil? (:evidence fb))  4
        :else                  5))

(defn- extract-distinct
  "The distinct (fn [trace] ...) code strings a batch of completions contains — the shared
   extract+dedup contract for both candidate builders (slips kept; nils dropped)."
  [completions]
  (->> completions (map extract-form) (filter some?) distinct))

(defn completions->candidates
  "Map raw LLM completions to driver/search candidates: extract the form (slips kept),
   dedup identical forms, attach the raw :code (what the check node scores) and a
   best-effort :spec' (for the trajectory / search / backtrack; falls back to the current
   `spec` when the form is off-grammar, so the loop state stays valid)."
  [completions spec]
  (->> (extract-distinct completions)
       (mapv (fn [code]
               (let [sp (parse-spec code)]
                 {:edit (if sp :llm :llm-raw)
                  :desc (candidate-desc sp)
                  :code code
                  :spec' (or sp spec)})))))

(defn make-proposer
  "Build an injected proposer `(fn [spec feedback] -> [candidate ...])` backed by an LLM —
   a mini-REPL: the proposer PROPOSES, CHECKS its own output against the verifier, and on a
   slip RE-PROMPTS the model with that specific error to fix, up to `:revise` times. This
   is the north-star inner loop (propose → eval → read error → revise); without it a real
   LLM stalls, because a slip that blocks every candidate (a non-positive σ, a hallucinated
   distribution, a coverage miss) never produces a scoring candidate for the driver to
   accept, so the model is never told why. The DRIVER's outer loop still owns FIT
   improvement (accept a candidate only when its exact evidence climbs); this inner loop
   owns SELF-CORRECTION. The returned candidates carry raw `:code` (the driver re-checks +
   selects); the proposer's internal check decides only whether to revise.

   opts:
     :call-llm     (fn [{:system :prompt :n :temperature :max_tokens :seed}]
                       -> {:completions [str ...] ...})   REQUIRED (inject `call-server`,
                   or a mock in tests).
     :task-desc    natural-language description of the data/goal (no structure given away)
     :observations the {:addr value} data, rendered into the prompt + used for the check
     :k            samples per generation call (default 4)
     :temperature  sampling temperature (default 0.7)
     :max-tokens   per-sample cap (default 384)
     :revise       max self-correction re-prompts when no candidate scores (default 0 =
                   no revision — a pure best-of-K-per-step proposer)
     :n-particles  IS particles for the internal revise-decision check (default 2000)
     :system       system prompt (default the DSL reference above)
     :seed         RNG seed for the worker (reproducible; default 1)

   The current model rendered into the prompt is `(:code feedback)` (the check node
   records it) falling back to `(syn/render spec)`; on a revision round it is the
   most-progressed FAILED candidate's own code + the error it hit, so the model fixes
   its own attempt."
  [{:keys [call-llm task-desc observations k temperature max-tokens system seed revise n-particles]
    :or {k 4 temperature 0.7 max-tokens 384 seed 1 revise 0 n-particles 2000}}]
  (let [sys (or system default-system)]
    (fn [spec feedback]
      (loop [r       0
             cur     (or (:code feedback) (syn/render spec))
             prompt-fb feedback
             seen    #{}
             acc     []]
        (let [prompt  (step-prompt task-desc observations cur prompt-fb)
              resp    (call-llm {:system sys :prompt prompt :n k :temperature temperature
                                 :max_tokens max-tokens :seed (+ seed (* 1000 r))})
              fresh   (remove #(seen (:code %)) (completions->candidates (:completions resp) spec))
              checked (mapv (fn [c] (assoc c :feedback (syn/check (:code c) observations {:n-particles n-particles})
                                           :revised? (pos? r)))
                            fresh)
              acc'    (into acc checked)
              scored  (filter #(syn/scored? (:feedback %)) checked)]
          (if (or (seq scored) (>= r revise) (empty? checked))
            ;; done: a candidate scores, budget exhausted, or nothing new to revise.
            ;; Strip the internal :feedback (the driver re-checks + selects).
            (mapv #(dissoc % :feedback) acc')
            ;; revise: re-prompt the most-progressed failed candidate with its own error.
            (let [worst (last (sort-by (comp progress-level :feedback) checked))]
              (recur (inc r) (:code worst) (:feedback worst)
                     (into seen (map :code checked)) acc'))))))))

(defn one-shot-candidates
  "The best-of-K CONTROL: K full programs from the SAME model + DSL prompt with NO
   feedback loop. Returns [{:code ...} ...] for the probe to `check`-score and rank — the
   baseline the LLM-in-the-loop must beat. (`:spec'` defaults to nil; the probe scores
   `:code` directly.)"
  [{:keys [call-llm task-desc observations k temperature max-tokens system seed]
    :or {k 16 temperature 0.8 max-tokens 384 seed 1}}]
  (let [resp (call-llm {:system (or system default-system)
                        :prompt (oneshot-prompt task-desc observations)
                        :n k :temperature temperature :max_tokens max-tokens :seed seed})]
    (mapv (fn [code] {:code code}) (extract-distinct (:completions resp)))))

;; ===========================================================================
;; 6. call-server — the out-of-process mlx-lm bridge (the sole I/O boundary)
;; ===========================================================================

(def ^:private cp (js/require "child_process"))
(def ^:private fs (js/require "fs"))
(def ^:private os (js/require "os"))
(def ^:private node-path (js/require "path"))
(def ^:private req-counter (atom 0))

(defn call-server
  "Synchronous POST <url>/generate with the request map (`{:system :prompt :n
   :temperature :max_tokens :seed}`); returns the parsed worker response
   `{:completions [...] :gen-time-s :prompt-tokens :completion-tokens :model}`.

   Blocking (execSync + curl): the synthesis driver is synchronous, and the resident
   policy LLM is a genuine out-of-process I/O boundary. The body is written to a temp
   file (the prompt is paren/quote/newline-heavy — shell-hostile). On any transport error
   returns `{:completions [] :error msg}` so the loop degrades gracefully (no candidates
   -> the driver plateaus) instead of throwing."
  [url req]
  (let [tmp (.join node-path (.tmpdir os)
                   (str "genmlx-llm-req-" (.-pid js/process) "-" (swap! req-counter inc) ".json"))]
    (try
      (.writeFileSync fs tmp (js/JSON.stringify (clj->js req)))
      (let [cmd (str "curl -s --max-time 900 -X POST " url "/generate "
                     "-H 'content-type: application/json' --data-binary @" tmp)
            out (.execSync cp cmd #js {:encoding "utf8" :maxBuffer (* 64 1024 1024)})]
        (js->clj (js/JSON.parse out) :keywordize-keys true))
      (catch :default e {:completions [] :error (.-message e)})
      (finally (try (.unlinkSync fs tmp) (catch :default _ nil))))))
