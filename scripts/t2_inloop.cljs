(ns t2-inloop
  "T2 bake-off (genmlx-8lm2, Tier-2): the IN-THE-LOOP synthesis comparison. Each
   model arm acts as the PROPOSER inside the genmlx.world.search loop (the Phase-2
   particle/beam search over construction steps), driven fully IN-PROCESS — the
   arm's checkpoint is resident ONCE per invocation (one arm per run, Thor
   discipline), no out-of-process worker (scripts/llm_server.py is Metal-bound and
   unusable on this CUDA box). Metrics are synthesis SUCCESS + COMPUTE-TO-SOLUTION.

   TASK-FORMAT RECONCILIATION (t1-battery -> world.search). The unified T1 battery
   (genmlx.world.t1-battery) carries {:id :kind :system-prompt :prompt} + held-out
   oracle fields; world.search's contract is {:init-spec :observations :propose}
   with the four-level syn/check node scoring Bayesian model evidence. The adapter:
     - Only :kind :program tasks fit (9/17): their :observations feed the evidence
       oracle. The 8 :function tasks (state machines + test-case functions) are
       EXCLUDED — their behavioral oracle (:transitions/:test-cases) has no model
       evidence, so the search loop's accept rule and solve bar are undefined for
       them; they are listed under :excluded in the output metadata.
     - :task-desc for the proposer prompt is the battery :prompt VERBATIM (template
       included — the T1 battery's own design: the prompt embeds a complete loose
       template over the real addresses). All arms see identical prompts, so arms
       differ only by checkpoint. The loop uses lp/default-system for every arm
       (its feedback lines reference that prompt's numbered rules); the battery's
       one-shot :system-prompt is not used here.
     - :init-spec is the crude covering model (cur/crude-spec), as in inloop_eval.
     - SOLVE BAR: the evidence of the task's own reference template — the battery's
       exemplar for the 5 lifted MSA tasks, else the template extracted from the
       prompt text (lp/extract-form) — checked with the same oracle opts. solved? =
       the loop produced a candidate whose checked evidence reaches that bar (the
       templates are deliberately loose, so the bar is reachable and beating it
       means real adaptation). For the one IS-scored task (msa-5) the bar and the
       candidate evidence both carry IS noise; treat its solved? as approximate.
       The bar is model-AGNOSTIC (pure evidence): tasks whose loose reference is
       beatable by a sigma-tuned crude gaussian (the two gaussian-mean tasks,
       whose reference IS the crude structure; and gamma-poisson-counts, whose
       gamma(1,1) reference scores below a sigma-tuned gaussian) can be solved by
       the loop's deterministic sigma-refiner arm alone — those rows still meter
       the arm's tokens (the LLM is always called first) but discriminate arms
       weakly; the structural tasks (beta-bernoulli, msa-1..4) need the LLM.

   SYNC/ASYNC BRIDGE (the load-bearing mechanism). The search loop and the proposer
   stack (se/search -> h/loop-proposer -> lp/make-proposer) call :call-llm
   SYNCHRONOUSLY, but in-process generation (genmlx.llm.backend/generate-text-raw+)
   is a promise (async NAPI tokenizer; no sync path). Bridge: a REPLAY TRAMPOLINE.
   The injected :call-llm consults a per-task response cache; on a miss it throws a
   sentinel carrying the request; the async driver catches it, generates ONCE (the
   model stays resident; each unique request costs GPU exactly once), fills the
   cache, and re-drives se/search from scratch. Re-drives are byte-identical
   because js/Math.random — the single entropy injection point (rng/fresh-key) for
   the oracle's IS scoring — is patched to a seeded mulberry32 for the duration of
   each drive, so the replayed prefix consumes the same entropy stream and reaches
   one cache hit further each time. Oracle work is re-done per drive (O(calls^2)
   small exact/IS graphs); GPU generation is not. MAX_DRIVES caps a runaway.

   PER TASK the driver records {:task_id :solved? :steps-used :candidates
   :completion-tokens :gen-ms :tokens-to-solution :wall-ms} (+ :evidence
   :solve-bar :stop-reason :drives :calls :samples :errors). :tokens-to-solution is
   the cumulative completion tokens generated at the FIRST solving candidate (nil
   if unsolved), detected by a propose-wrapper that re-checks each candidate
   against the bar — proposer-granular and covering both LLM and sigma-refiner
   candidates. With EARLY_STOP=1 (default) the wrapper returns no candidates once
   solved, so the loop self-terminates instead of spending GPU past the solve;
   :steps-used then reads 'steps to solution (+1 terminal)'. EARLY_STOP=0 lets the
   loop run its natural plateau course.

   AGGREGATE: solve-rate with a seeded-LCG bootstrap 95% CI (the r2_bakeoff
   pattern), median tokens-to-solution over solved tasks, and total compute
   (completion tokens, gen-ms, wall) — the compute-to-solution table. Output via
   bench.util/write-json. RESUME-FRIENDLY: task ids already present in OUT are
   skipped and their rows carried forward.

   STUB=1 exercises the ENTIRE code path (trampoline, loop, verification, metering,
   output) with a canned :call-llm generation fn — no model, no checkpoint; the
   oracle still runs its tiny scalar graphs on the native addon. The stub emits a
   mix per call: a scoring-but-not-solving degraded model, an unparseable form, an
   uncovered model, prose with no form — and from the second call onward the task's
   own reference template (the solving candidate). Two tasks (gamma-poisson-counts,
   msa-5) get junk-only completions so make-proposer's revise path (which only
   fires when no candidate scores) runs; msa-5 then stays unsolved (the
   plateau/unsolved path), while gamma-poisson-counts still solves through the
   sigma-refiner alone (the non-LLM solve-detection path — see stub-unsolved).
   Stub :n-tokens is an honest chars/4 proxy (no tokenizer is loaded).

   Run (stub, no model / no GPU checkpoint):
     STUB=1 ARM=stub bunx --bun nbb@1.4.208 scripts/t2_inloop.cljs
   Real arms (Thor discipline: ONE GPU process at a time, via the guarded runner):
     ARM=a MODEL_DIR=/home/robert/code/mlx/models/Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5 \\
       ~/genmlx-guarded-run.sh t2-a bunx --bun nbb@1.4.208 scripts/t2_inloop.cljs
     ARM=b MODEL_DIR=$HOME/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-8bit/snapshots/28d10f45a04981989f5a2e53e38f6473e81e815a \\
       ~/genmlx-guarded-run.sh t2-b bunx --bun nbb@1.4.208 scripts/t2_inloop.cljs
   Side-by-side (model-free):
     COMPARE=a,b bunx --bun nbb@1.4.208 scripts/t2_inloop.cljs
   Env: ARM (label; required unless STUB=1) MODEL_DIR (required unless STUB=1)
        MAX_STEPS (8) K_STEP (4) BEAM_WIDTH (2) REVISE (2) TEMPERATURE (0.8 — NOT
        'TEMP': Bun honors TEMP as its temp dir) MAX_TOKENS (512) SEED (42)
        NP (2000) TASKS (comma-separated id subset) OUT
        (results/t2-inloop/t2-<ARM>.json) EARLY_STOP (1) MAX_DRIVES (200)
        BOOT (2000) STUB COMPARE (a,b[,c]) OUT_DIR (compare reads;
        results/t2-inloop)"
  (:require [genmlx.world.t1-battery :as battery]
            [genmlx.world.curriculum :as cur]
            [genmlx.world.harvest :as h]
            [genmlx.world.search :as se]
            [genmlx.world.synth :as syn]
            [genmlx.world.llm-proposer :as lp]
            [clojure.string :as str]
            [promesa.core :as p]))

;; bench/ is not on the nbb classpath (nbb.edn :paths); these scripts run from
;; the repo root (t1_score.cljs pattern), so load the shared JSON helpers by path.
(require '[nbb.core])
(nbb.core/load-file "bench/util.cljs")
(require '[bench.util :as bu])

(def fs   (js/require "fs"))
(def path (js/require "path"))
(def cp   (js/require "child_process"))

(defn- env  [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))
(defn- envf [k d] (let [v (env k nil)] (if v (js/parseFloat v) d)))

(def compare-spec (env "COMPARE" nil))
(def stub?        (= "1" (env "STUB" "0")))
(def arm          (env "ARM" (when stub? "stub")))
(def model-dir    (env "MODEL_DIR" nil))    ;; required for real runs — no default
(def max-steps    (envi "MAX_STEPS" 8))     ;; inloop_eval convention
(def k-step       (envi "K_STEP" 4))
(def beam-width   (envi "BEAM_WIDTH" 2))
(def revise       (envi "REVISE" 2))
;; NOT "TEMP" — Bun honors TEMP as its temp-dir (a bunx cache landed in ./0.8/ once)
(def temp         (envf "TEMPERATURE" 0.8))
(def max-tokens   (envi "MAX_TOKENS" 512))
(def seed         (envi "SEED" 42))
(def np           (envi "NP" 2000))
(def early-stop?  (= "1" (env "EARLY_STOP" "1")))
(def max-drives   (envi "MAX_DRIVES" 200))
(def boot         (envi "BOOT" 2000))
(def task-subset  (when-let [s (env "TASKS" nil)]
                    (set (map str/trim (str/split s #",")))))
(def out-file     (env "OUT" (str "results/t2-inloop/t2-" arm ".json")))
(def out-dir      (.dirname path out-file))
(def out-name     (.basename path out-file))
(def cmp-dir      (env "OUT_DIR" "results/t2-inloop"))

(defn- fx  [x] (if (and (number? x) (js/isFinite x)) (.toFixed (js/Number x) 2) "--"))
(defn- pct [x] (if (and (number? x) (js/isFinite x)) (str (.toFixed (* 100 x) 0) "%") "--"))
(defn- pad [s n] (.padEnd (str s) n))

;; ---------------------------------------------------------------------------
;; Task selection — the :program subset of the T1 battery (see ns docstring for
;; why :function tasks are excluded), TASKS-filtered, with honest exclusion meta.
;; ---------------------------------------------------------------------------

(def ^:private program-tasks (filterv #(= :program (:kind %)) battery/tasks))
(def ^:private function-tasks (filterv #(not= :program (:kind %)) battery/tasks))
(def ^:private task-index
  "Stable per-task index (battery order) for per-task seeds — resume-invariant."
  (into {} (map-indexed (fn [i t] [(:id t) i])) program-tasks))

(def tasks
  (if task-subset
    (let [ts      (filterv #(task-subset (:id %)) program-tasks)
          req-fn  (filterv #(task-subset (:id %)) function-tasks)
          unknown (remove (set (map :id battery/tasks)) task-subset)]
      (when (seq unknown)
        (println "WARN: unknown task ids in TASKS:" (vec unknown)))
      (when (seq req-fn)
        (println "WARN: requested :function tasks excluded (no evidence oracle):"
                 (mapv :id req-fn)))
      ts)
    program-tasks))

(def ^:private excluded-meta
  (mapv (fn [t] {:task_id (:id t) :kind (name (:kind t))
                 :reason (str "behavioral oracle (:transitions/:test-cases) — no "
                              "Bayesian-evidence solve bar for the world.search "
                              "check node")})
        function-tasks))

(defn- reference-code
  "The task's known-good reference template: the battery exemplar (lifted MSA
   tasks) or the template embedded in the prompt text itself (distill tasks)."
  [task]
  (or (get battery/exemplars (:id task))
      (lp/extract-form (:prompt task))))

;; ---------------------------------------------------------------------------
;; Deterministic re-drives: patch js/Math.random (rng/fresh-key's sole entropy
;; source) to a seeded mulberry32 for the duration of one synchronous drive.
;; ---------------------------------------------------------------------------

(defn- mulberry32
  "The standard mulberry32 PRNG as a 0-arg fn -> [0,1) (the curriculum's seeded
   PRNG family; 32-bit clean via js/Math.imul + unsigned shifts)."
  [seed*]
  (let [st (volatile! (bit-or 0 seed*))]
    (fn []
      (vswap! st (fn [s] (bit-or (+ s 0x6D2B79F5) 0)))
      (let [a  @st
            t  (js/Math.imul (bit-xor a (unsigned-bit-shift-right a 15)) (bit-or 1 a))
            t2 (bit-xor (+ t (js/Math.imul (bit-xor t (unsigned-bit-shift-right t 7))
                                           (bit-or 61 t)))
                        t)]
        (/ (unsigned-bit-shift-right (bit-xor t2 (unsigned-bit-shift-right t2 14)) 0)
           4294967296)))))

(defn- with-seeded-entropy
  "Run (f) with js/Math.random replaced by a seeded mulberry32; always restores."
  [seed* f]
  (let [orig (.-random js/Math)]
    (set! (.-random js/Math) (mulberry32 seed*))
    (try (f) (finally (set! (.-random js/Math) orig)))))

;; ---------------------------------------------------------------------------
;; The replay trampoline: a synchronous :call-llm over a per-task response cache.
;; ---------------------------------------------------------------------------

(defn- req-key [{:keys [system prompt n temperature max_tokens seed]}]
  [system prompt n temperature max_tokens seed])

(defn- make-call-llm
  "The synchronous :call-llm closure injected into the loop proposer (the same
   code path for stub and real arms — only the async generation fn differs). A
   cache hit returns the recorded response and books its tokens ONCE per unique
   request into the drive-local causal ledger; a miss throws the re-drive
   sentinel carrying the request."
  [cache drive]
  (fn [req]
    (let [k (req-key req)]
      (if-let [resp (get @cache k)]
        (do (when-not (contains? (:seen @drive) k)
              (vswap! drive #(-> %
                                 (update :seen conj k)
                                 (update :cum-tokens + (or (:completion-tokens resp) 0)))))
            resp)
        (throw (ex-info "t2 llm cache miss" {:t2/request req}))))))

(def ^:private native-gc!
  "Synchronous MLX sweep, set by real-generate! once genmlx.mlx is loaded (nil
   under STUB=1, which must stay native-free). se/search is SYNCHRONOUS: a long
   drive never yields to the event loop, so Bun's finalizers — the only thing
   that frees dead MxArrays — never run and per-step oracle checks accumulate
   live memory until OOM (observed: the owned-35B T2 arms ran ~100 GB deep and
   were floor-killed). Calling this at every propose step bounds the growth to
   one step's worth of checks."
  (atom nil))

(defn- wrap-propose
  "Wrap the loop proposer for metering + solve detection: count candidates, and —
   until the first solve — re-check each candidate against the solve bar,
   recording the drive-local cumulative completion tokens at the first crossing.
   With early-stop, a solved drive proposes nothing further (the population
   plateaus immediately instead of spending more GPU). All state is drive-local,
   so re-drives replay identically. Also sweeps dead MLX arrays synchronously
   each step (see native-gc!)."
  [inner {:keys [bar observations check-opts drive]}]
  (fn [spec fb]
    (when-let [g @native-gc!] (g))
    (if (and early-stop? (some? (:solved-tokens @drive)))
      []
      (let [cands (vec (inner spec fb))]
        (vswap! drive update :candidates + (count cands))
        (when (and bar (nil? (:solved-tokens @drive)))
          (doseq [c cands]
            (when (nil? (:solved-tokens @drive))
              (let [code (or (:code c) (syn/render (:spec' c)))
                    v    (syn/check code observations check-opts)]
                (when (and (syn/scored? v) (>= (:evidence v) bar))
                  (vswap! drive assoc :solved-tokens (:cum-tokens @drive)))))))
        cands))))

;; ---------------------------------------------------------------------------
;; Generation fns. Both resolve to (fn [task req] -> promise<{:completions
;; [str ...] :completion-tokens n :gen-ms n}>) so the trampoline is one shared
;; code path; only the model call differs.
;; ---------------------------------------------------------------------------

(def ^:private stub-unsolved
  "Task ids whose STUB completions are JUNK-ONLY (no scoring LLM candidate at
   all — a widened prior is not reliably worse for a non-location-scale prior
   like gamma's rate, so even the degraded variant could solve), which also
   exercises make-proposer's revise/self-correction path (it only fires when no
   candidate scores). msa-5 then never solves (validating the plateau/unsolved
   path); gamma-poisson-counts still solves WITHOUT any LLM candidate — its
   loose reference bar is beatable by the deterministic sigma-refiner arm alone
   (see the ns docstring) — validating solve detection on non-LLM candidates."
  #{"gamma-poisson-counts" "msa-5"})

(def ^:private stub-parse-fail "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 1))]")
(def ^:private stub-uncovered  "(fn [trace] {:z (trace :z (dist/gaussian 0 1))})")
(def ^:private stub-uncovered2 "(fn [trace] {:w (trace :w (dist/beta 2 2))})")
(def ^:private stub-prose
  "The model looks reasonable; I would consider a hierarchical prior over the group means.")

(defn- degrade-reference
  "A scoring-but-imperfect variant of the reference: the first latent's prior
   last-arg scaled 5x. For the location-scale tasks (gaussian priors) this widens
   the prior and strictly costs marginal evidence, staying below the bar; it is
   NOT guaranteed worse for every prior family (see stub-unsolved), which is why
   the designated-unsolved tasks never receive it. nil when the reference is
   off-grammar for lp/parse-spec (e.g. msa-5's link-function let-binding)."
  [reference]
  (when reference
    (when-let [sp (lp/parse-spec reference)]
      (when-let [l (first (:latents sp))]
        (let [args (vec (:args l))]
          (when (and (seq args) (number? (last args)))
            (syn/render (syn/set-args sp (:addr l)
                                      (assoc args (dec (count args))
                                             (* 5 (last args)))))))))))

(defn- stub-generate!
  "The canned generation fn: per call a mix of one scoring-not-solving model, an
   unparseable form, an uncovered model, and prose; from the 2nd call per task the
   reference template (fenced + <think>-wrapped, exercising extract-form) leads —
   except for stub-unsolved ids. Goes through the SAME trampoline/metering path
   as the real closure. :completion-tokens is an honest chars/4 proxy."
  []
  (let [counters (atom {})]
    (p/resolved
     (fn [task {:keys [n]}]
       (let [t0        (.now js/Date)
             id        (:id task)
             i         (get (swap! counters update id (fnil inc 0)) id)
             reference (reference-code task)
             degraded  (or (degrade-reference reference)
                           (syn/render (cur/crude-spec (:observations task))))
             solving?  (and (>= i 2) (not (stub-unsolved id)))
             pool      (cond
                         (stub-unsolved id)
                         [stub-parse-fail stub-uncovered stub-prose stub-uncovered2]
                         solving?
                         [(str "<think>adapt the priors to the data</think>\n"
                               "```clojure\n" reference "\n```")
                          degraded stub-uncovered stub-prose]
                         :else
                         [degraded stub-parse-fail stub-uncovered stub-prose])
             comps     (vec (take (max 1 (or n 1)) pool))
             toks      (reduce + 0 (map #(max 1 (js/Math.round (/ (count %) 4))) comps))]
         (p/resolved {:completions comps
                      :completion-tokens toks
                      :gen-ms (- (.now js/Date) t0)}))))))

(defn- real-generate!
  "Dynamically require the LLM backend (which js/require's the native model path —
   this is what keeps STUB=1 runs checkpoint-free) and load the model ONCE; the
   returned fn decodes the request's :n samples sequentially (seeds :seed+i) and
   sweeps dead decode graphs between samples (Tegra dark-pages lesson, genmlx-h3p5)."
  []
  (p/let [_ (require '[genmlx.llm.backend]
                     '[genmlx.mlx])]
    (let [load-model (resolve 'genmlx.llm.backend/load-model)
          gen+       (resolve 'genmlx.llm.backend/generate-text-raw+)
          force-gc!  (resolve 'genmlx.mlx/force-gc!)
          _          (reset! native-gc! force-gc!)
          t0         (.now js/Date)]
      (p/let [m (load-model model-dir)]
        (println (str "  loaded: " (name (:type m)) " in "
                      (js/Math.round (/ (- (.now js/Date) t0) 1000)) " s"))
        (fn [_task {:keys [system prompt n temperature max_tokens] :as req}]
          (p/loop [i 0, comps [], toks 0, ms 0]
            (if (>= i (max 1 (or n 1)))
              (p/resolved {:completions comps :completion-tokens toks :gen-ms ms})
              (p/let [r (gen+ m prompt {:max-tokens  (or max_tokens max-tokens)
                                        :temperature (or temperature temp)
                                        :seed        (+ (or (:seed req) 1) i)
                                        :system-prompt (or system lp/default-system)})]
                (force-gc!)
                (p/recur (inc i) (conj comps (:text r))
                         (+ toks (:n-tokens r)) (+ ms (:gen-ms r)))))))))))

;; ---------------------------------------------------------------------------
;; One task through the search loop (async trampoline around sync se/search).
;; ---------------------------------------------------------------------------

(defn- run-task! [task generate!]
  (let [id         (:id task)
        obs        (:observations task)
        tidx       (get task-index id 0)
        drive-seed (+ seed (* 7919 (inc tidx)))
        check-opts {:n-particles np}
        reference  (reference-code task)
        bar-fb     (when reference
                     (with-seeded-entropy (inc drive-seed)
                       #(syn/check reference obs check-opts)))
        bar        (when (and bar-fb (syn/scored? bar-fb)) (:evidence bar-fb))
        init-spec  (cur/crude-spec obs)
        init-fb    (with-seeded-entropy (+ 2 drive-seed)
                     #(syn/check (syn/render init-spec) obs check-opts))
        presolved? (boolean (and bar (syn/scored? init-fb)
                                 (>= (:evidence init-fb) bar)))
        cache      (atom {})
        drive      (volatile! nil)
        stats      (atom {:calls 0 :samples 0 :completion-tokens 0 :gen-ms 0 :errors 0})
        call-llm   (make-call-llm cache drive)
        prop       (h/loop-proposer {:call-llm call-llm :task-desc (:prompt task)
                                     :observations obs :k k-step :temperature temp
                                     :max-tokens max-tokens :revise revise
                                     :n-particles np :seed seed})
        wrapped    (wrap-propose prop {:bar bar :observations obs
                                       :check-opts check-opts :drive drive})
        sopts      {:init-spec init-spec :observations obs :propose wrapped
                    :strategy :beam :beam-width beam-width :adaptive? true
                    :max-steps max-steps :n-particles np :seed seed}
        drive-once! (fn []
                      (vreset! drive {:seen #{} :cum-tokens 0 :candidates 0
                                      :solved-tokens (when presolved? 0)})
                      (try
                        {:done (with-seeded-entropy drive-seed #(se/search sopts))}
                        (catch :default e
                          (if-let [req (:t2/request (ex-data e))]
                            {:need req}
                            (throw e)))))
        t0         (js/Date.now)
        row-base   (fn [drives]
                     (let [st @stats]
                       {:task_id id :kind (name (:kind task))
                        :solve-bar bar :bar-method (some-> (:method bar-fb) name)
                        :drives drives
                        :calls (:calls st) :samples (:samples st)
                        :completion-tokens (:completion-tokens st)
                        :gen-ms (:gen-ms st) :errors (:errors st)
                        :wall-ms (- (js/Date.now) t0)}))]
    (p/loop [drives 1]
      (if (> drives max-drives)
        (p/resolved (merge (row-base (dec drives))
                           {:solved? false :evidence nil :steps-used nil
                            :stop-reason "aborted" :candidates (:candidates @drive)
                            :tokens-to-solution nil :aborted "max-drives exceeded"}))
        (let [r (drive-once!)]
          (if-let [res (:done r)]
            (let [d  @drive
                  ev (:evidence (:best res))]
              (p/resolved (merge (row-base drives)
                                 {:solved? (some? (:solved-tokens d))
                                  :evidence (when (and ev (js/isFinite ev)) ev)
                                  :steps-used (:steps res)
                                  :stop-reason (name (:stop-reason res))
                                  :candidates (:candidates d)
                                  :tokens-to-solution (:solved-tokens d)})))
            (p/let [resp (-> (generate! task (:need r))
                             (p/catch (fn [e] {:completions []
                                               :completion-tokens 0 :gen-ms 0
                                               :error (.-message e)})))]
              (when (:error resp) (println (str "    [llm error] " (:error resp))))
              (swap! stats #(-> %
                                (update :calls inc)
                                (update :samples + (count (:completions resp)))
                                (update :completion-tokens + (or (:completion-tokens resp) 0))
                                (update :gen-ms + (or (:gen-ms resp) 0))
                                (update :errors + (if (:error resp) 1 0))))
              (swap! cache assoc (req-key (:need r)) resp)
              (p/recur (inc drives)))))))))

;; ---------------------------------------------------------------------------
;; Aggregation — seeded-LCG bootstrap CI (r2_bakeoff pattern), self-seeding per
;; call so intermediate per-task writes cannot drift the final CI.
;; ---------------------------------------------------------------------------

(defn- mean [xs] (when (seq xs) (/ (reduce + xs) (count xs))))

(defn- median [xs]
  (when (seq xs)
    (let [s (vec (sort xs)) n (count s) m (quot n 2)]
      (if (odd? n) (nth s m) (/ (+ (nth s (dec m)) (nth s m)) 2.0)))))

(defn- boot-ci
  "Bootstrap a 95% CI for (mean of `f` over a resample of `rows`)."
  [rows f]
  (let [vals  (vec (keep f rows))
        n     (count vals)
        state (volatile! (bit-or 1 (* seed 2654435761)))
        next-u (fn [] (/ (vswap! state (fn [x] (bit-and (+ (* x 1103515245) 12345)
                                                        0x7fffffff)))
                         0x7fffffff))]
    (when (pos? n)
      (let [samples (sort (for [_ (range boot)]
                            (mean (for [_ (range n)] (nth vals (int (* (next-u) n)))))))
            at (fn [q] (nth samples (min (dec boot) (int (* q boot)))))]
        {:mean (mean vals) :lo (at 0.025) :hi (at 0.975) :n n}))))

(defn- aggregate [rows]
  (let [n      (count rows)
        solved (filterv :solved? rows)
        tts    (keep :tokens-to-solution solved)]
    {:n-tasks n
     :n-solved (count solved)
     :solve-rate (when (pos? n) (/ (count solved) (double n)))
     :solve-ci (boot-ci rows #(if (:solved? %) 1.0 0.0))
     :median-tokens-to-solution (median tts)
     :mean-steps-used (mean (keep :steps-used rows))
     :totals {:completion-tokens (reduce + 0 (keep :completion-tokens rows))
              :gen-ms  (reduce + 0 (keep :gen-ms rows))
              :wall-ms (reduce + 0 (keep :wall-ms rows))
              :calls   (reduce + 0 (keep :calls rows))
              :samples (reduce + 0 (keep :samples rows))
              :drives  (reduce + 0 (keep :drives rows))
              :errors  (reduce + 0 (keep :errors rows))}}))

;; ---------------------------------------------------------------------------
;; Output + resume
;; ---------------------------------------------------------------------------

(defn- read-json [p*]
  (js->clj (js/JSON.parse (.readFileSync fs p* "utf8")) :keywordize-keys true))

(defn- existing-rows []
  (if (.existsSync fs out-file)
    (vec (:tasks (read-json out-file)))
    []))

(defn- git-sha []
  (try (str/trim (.toString (.execSync cp "git rev-parse --short HEAD")))
       (catch :default _ "unknown")))

(def ^:private run-config
  {:max-steps max-steps :k-step k-step :beam-width beam-width :revise revise
   :temperature temp :max-tokens max-tokens :seed seed :np np
   :early-stop early-stop? :max-drives max-drives :boot boot})

(defn- write-out! [rows t0]
  (bu/write-json out-dir out-name
                 {:arm arm :stub stub?
                  :model-dir (if stub? "STUB" model-dir)
                  :config run-config :git-sha (git-sha)
                  :battery-ids (mapv :id tasks)
                  :excluded excluded-meta
                  :tasks rows
                  :aggregate (aggregate rows)
                  :wall-ms (- (js/Date.now) t0)}))

(defn- print-row [row i total]
  (println (str "  [" i "/" total "] " (pad (:task_id row) 24)
                (if (:solved? row) " SOLVED  " " unsolved")
                "  ev=" (fx (:evidence row)) " bar=" (fx (:solve-bar row))
                "  steps=" (or (:steps-used row) "--")
                " drives=" (:drives row)
                " cand=" (:candidates row)
                "  tok=" (:completion-tokens row)
                " tok->soln=" (or (:tokens-to-solution row) "--")
                "  wall=" (fx (/ (or (:wall-ms row) 0) 1000.0)) "s"
                (when (:aborted row) (str "  ABORTED(" (:aborted row) ")")))))

(defn- print-aggregate [rows]
  (let [{:keys [n-tasks n-solved solve-ci median-tokens-to-solution mean-steps-used
                totals]} (aggregate rows)]
    (println (str "\n### COMPUTE-TO-SOLUTION  (arm=" arm (when stub? " [STUB]") ")"))
    (println (str "  solve-rate: " n-solved "/" n-tasks " = " (pct (:mean solve-ci))
                  "  CI[" (pct (:lo solve-ci)) ", " (pct (:hi solve-ci)) "]"))
    (println (str "  median tokens-to-solution (solved): "
                  (or median-tokens-to-solution "--")
                  "   mean steps-used: " (fx mean-steps-used)))
    (println (str "  totals: " (:calls totals) " calls, " (:samples totals) " samples, "
                  (:completion-tokens totals) " completion tokens, "
                  (fx (/ (:gen-ms totals) 1000.0)) "s gen, "
                  (fx (/ (:wall-ms totals) 1000.0)) "s task-wall, "
                  (:drives totals) " drives"
                  (when (pos? (:errors totals)) (str ", " (:errors totals) " errors"))))))

;; ---------------------------------------------------------------------------
;; COMPARE mode — side-by-side over 2-3 arm reports (model-free)
;; ---------------------------------------------------------------------------

(def ^:private cmp-metrics
  [["n-tasks"             #(get-in % [:aggregate :n-tasks]) str]
   ["solve-rate"          #(get-in % [:aggregate :solve-rate]) pct]
   ["solve CI lo"         #(get-in % [:aggregate :solve-ci :lo]) pct]
   ["solve CI hi"         #(get-in % [:aggregate :solve-ci :hi]) pct]
   ["median tok->soln"    #(get-in % [:aggregate :median-tokens-to-solution]) fx]
   ["mean steps-used"     #(get-in % [:aggregate :mean-steps-used]) fx]
   ["completion tokens"   #(get-in % [:aggregate :totals :completion-tokens]) str]
   ["gen-s total"         #(when-let [v (get-in % [:aggregate :totals :gen-ms])]
                             (/ v 1000.0)) fx]
   ["wall-s total"        #(when-let [v (get-in % [:aggregate :totals :wall-ms])]
                             (/ v 1000.0)) fx]
   ["llm calls"           #(get-in % [:aggregate :totals :calls]) str]])

(defn- run-compare! [spec*]
  (let [arms (mapv str/trim (str/split spec* #","))]
    (when-not (<= 2 (count arms) 3)
      (println "COMPARE takes 2 or 3 comma-separated arm labels")
      (js/process.exit 1))
    (let [reports (mapv (fn [a]
                          (let [p* (str cmp-dir "/t2-" a ".json")]
                            (when-not (.existsSync fs p*)
                              (println (str "missing report: " p*
                                            " (run that arm first)"))
                              (js/process.exit 1))
                            [a (read-json p*)]))
                        arms)
          two?    (= 2 (count arms))
          rows    (vec (for [[label f _fmt] cmp-metrics]
                         {:metric label
                          :values (into {} (map (fn [[a r]] [a (f r)])) reports)}))]
      (println (str "\n== t2 in-loop compare: " (str/join " vs " arms) " =="))
      (println (str "  " (pad "metric" 20)
                    (apply str (map #(pad % 14) arms))
                    (when two? (str "delta(" (second arms) "-" (first arms) ")"))))
      (doseq [[[label f fmt] row] (map vector cmp-metrics rows)]
        (let [vs (mapv #(f (second %)) reports)]
          (println (str "  " (pad label 20)
                        (apply str (map #(pad (fmt %) 14) vs))
                        (when (and two? (every? number? vs))
                          (fx (- (second vs) (first vs))))))))
      ;; per-task solved matrix (battery order, union of ids across reports)
      (let [by-arm (into {} (map (fn [[a r]]
                                   [a (into {} (map (juxt :task_id identity))
                                            (:tasks r))]))
                         reports)
            ids    (filterv (fn [id] (some #(get-in by-arm [% id]) arms))
                            (map :id program-tasks))]
        (println (str "\n  " (pad "task" 26)
                      (apply str (map #(pad % 18) arms))))
        (doseq [id ids]
          (println (str "  " (pad id 26)
                        (apply str
                               (for [a arms]
                                 (let [t (get-in by-arm [a id])]
                                   (pad (cond
                                          (nil? t)     "--"
                                          (:solved? t) (str "* tok=" (or (:tokens-to-solution t) "?"))
                                          :else        ". unsolved")
                                        18))))))))
      (bu/write-json cmp-dir (str "compare-" (str/join "-vs-" arms) ".json")
                     {:arms (into {} (map (fn [[a r]]
                                            [a {:file (str cmp-dir "/t2-" a ".json")
                                                :config (:config r)
                                                :aggregate (:aggregate r)}]))
                                  reports)
                      :metrics rows
                      :per-task (into {} (map (fn [[a r]]
                                                [a (mapv #(select-keys % [:task_id :solved?
                                                                          :tokens-to-solution
                                                                          :completion-tokens])
                                                         (:tasks r))]))
                                      reports)}))))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(defn- run-arm! []
  (when-not arm
    (println "usage: ARM=<label> MODEL_DIR=<dir> (or STUB=1, or COMPARE=a,b) bunx --bun nbb@1.4.208 scripts/t2_inloop.cljs")
    (js/process.exit 1))
  (when (and (not stub?) (not model-dir))
    (println "MODEL_DIR is required for a real arm (no default model; STUB=1 for the dry run)")
    (js/process.exit 1))
  (bu/ensure-dir out-dir)
  (let [prior    (existing-rows)
        done-ids (set (map :task_id prior))
        todo     (vec (remove #(done-ids (:id %)) tasks))
        total    (+ (count prior) (count todo))
        t0       (js/Date.now)]
    (println (str "== t2_inloop arm=" arm (when stub? " [STUB]") " =="))
    (println (str "  model : " (if stub? "STUB (canned completions)" model-dir)))
    (println (str "  tasks : " (count tasks) " :program (" (count function-tasks)
                  " :function excluded — no evidence oracle)"))
    (println (str "  loop  : beam=" beam-width " max-steps=" max-steps
                  " K_STEP=" k-step " revise=" revise " temp=" temp
                  " max-tokens=" max-tokens " np=" np " seed=" seed
                  (when early-stop? " early-stop")))
    (when (seq prior)
      (println (str "  resume: " (count prior) " task rows already in " out-file)))
    (p/let [generate! (if stub? (stub-generate!) (real-generate!))]
      (p/loop [ts (seq todo), rows (vec prior), i (inc (count prior))]
        (if-not ts
          (do (print-aggregate rows)
              (write-out! rows t0)
              (p/resolved nil))
          (p/let [row (run-task! (first ts) generate!)]
            (print-row row i total)
            (let [rows' (conj rows row)]
              (write-out! rows' t0)
              (p/recur (next ts) rows' (inc i)))))))))

(defn -main []
  (cond
    compare-spec (p/resolved (run-compare! compare-spec))
    :else        (run-arm!)))

(-> (-main)
    (p/catch (fn [e]
               (println "UNCAUGHT:" (.-message e))
               (println (.-stack e))
               (set! (.-exitCode js/process) 1))))
