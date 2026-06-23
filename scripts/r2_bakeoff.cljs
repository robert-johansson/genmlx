(ns r2-bakeoff
  "R2 (genmlx-xrps): the Phase-4-substrate DECISION GATE — the adversarial, compute-matched,
   σ-ablated bake-off of the LOOP against the strategy the evidence already favors,
   one-shot best-of-K + exact-evidence RERANK. Falsification-shaped: it is built to let the
   loop LOSE if it is going to lose (the de-masked R0 bar is 35B one-shot 25%).

   FOUR CELLS per task (a 2×2 of {loop, one-shot} × {fixed-σ, co-refined-σ}), so a 'loop win'
   can never be deterministic-σ-search laundered as an LLM-loop win (the R1 ablation hazard):

     L-strict  : the legacy strict-greedy loop (LLM ∪ σ-refiner union arm, strict ratchet).
                 The Phase-3 loop — the indicted search operator.
     L-coref   : the R1 (genmlx-8smp) co-refined-accept loop (LLM-ONLY proposer; ALL σ tuning
                 lives in the accept rule, judged at grid-best shared obs σ).
     OS-fixed  : one-shot best-of-K reranked by VERBATIM exact evidence (the de-masked R0 bar).
     OS-coref  : one-shot best-of-K reranked by CO-REFINED exact evidence — the SAME σ line
                 search the loop gets, so the ONLY difference vs L-coref is loop-vs-one-shot.

   The headline contrast is L-coref vs OS-coref (σ held constant on both → isolates the loop's
   value). OS-fixed is the R0 25% bar; L-strict is the Phase-3 baseline. COMPUTE-MATCHED:
   K_OS = K_STEP × MAX_STEPS × (1+REVISE) — the loop's LLM sample budget — so it is not a
   K=4-vs-loop strawman. OS-fixed and OS-coref share ONE set of one-shot candidates (one LLM
   call, two rerank rules), so they are strictly comparable and cost the same.

   FAMILY-SPLIT is the headline (the rrps leakage lesson, project_rrps_titleB_earned): a seed
   split of the same parametric families is within-distribution memorization. Cohorts are
   reported SEPARATELY — :within (held-out instances of trained families) and :family (an
   entirely held-out family). The GATE reads the :family cohort.

   NATURAL difficulty only (no steepening — steepening is a secondary diagnostic, never the
   GO gate). The non-conjugate cliff family (real GMM via RB scoring, R4.5/genmlx-9mos) is the
   remaining piece of the FULL gate; this script runs the natural linear-Gaussian families
   where the R0 bar lives, and is wired to accept a GMM family once rb_mixture lands.

   GATE: GO-LOOP iff a loop variant (L-coref or L-strict) beats OS-coref at bootstrap CI-lo>0
   on the :family cohort. Otherwise GO-RERANK (the evidence-favored default) — the crown is
   best-of-K one-shot + exact-evidence rerank + bounded validity-repair.

   Run a 35B worker (scripts/llm_server.py --model Qwen3.6-35B-A3B-4bit), then:
     SERVER_URL=http://127.0.0.1:8765 TIER=35b ROUND=0 INSTANCES=8 \\
       bun run --bun nbb scripts/r2_bakeoff.cljs
   Native-free DRY RUN of the whole harness (no worker — injects each task's gold):
     MOCK=1 INSTANCES=4 bun run --bun nbb scripts/r2_bakeoff.cljs
   Env: ROUND INSTANCES FAMILIES EVAL_FAMILIES TASKS_LIMIT SERVER_URL TIER K_STEP MAX_STEPS
        REVISE TEMP MAX_TOKENS NP SEED BOOT MOCK SYSTEM OUT."
  (:require [genmlx.world.curriculum :as cur]
            [genmlx.world.harvest :as h]
            [genmlx.world.synth :as syn]
            [genmlx.world.llm-proposer :as lp]
            [clojure.string :as str]))

(def os   (js/require "os"))
(def path (js/require "path"))
(def fs   (js/require "fs"))
(defn home [& xs] (apply (.-join path) (.homedir os) xs))
(defn- env  [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))
(defn- envf [k d] (let [v (env k nil)] (if v (js/parseFloat v) d)))
(defn fx [x] (if (and x (js/isFinite x)) (.toFixed (js/Number x) 2) (str x)))
(defn pct [x] (if (and x (js/isFinite x)) (str (.toFixed (* 100 x) 0) "%") "--"))

(def round       (envi "ROUND" 0))
(def instances   (envi "INSTANCES" 8))
(def families    (when-let [s (env "FAMILIES" nil)] (mapv keyword (str/split s #","))))
(def eval-fams   (if-let [s (env "EVAL_FAMILIES" nil)] (set (map keyword (str/split s #","))) #{:segmented}))
(def tasks-limit (envi "TASKS_LIMIT" 0))
(def server-url  (env "SERVER_URL" "http://127.0.0.1:8765"))
(def tier        (env "TIER" "unknown"))
(def k-step      (envi "K_STEP" 4))
(def max-steps   (envi "MAX_STEPS" 8))
(def revise      (envi "REVISE" 2))
(def k-oneshot   (* k-step max-steps (inc revise)))   ;; compute-matched to the loop budget
(def temp        (envf "TEMP" 0.7))
(def max-toks    (envi "MAX_TOKENS" 320))
(def np          (envi "NP" 2000))
(def seed        (envi "SEED" 1))
(def boot        (envi "BOOT" 2000))
(def mock?       (= "1" (env "MOCK" "0")))
(def out-dir     (env "OUT" (home "genmlx-loop-artifacts" "r2")))
(def grid        syn/default-noise-grid)
(def system-prompt
  (when-let [p (env "SYSTEM" nil)]
    (when (and (not= p "default") (.existsSync fs p)) (str/trim (.readFileSync fs p "utf8")))))

;; ---------------------------------------------------------------------------
;; Per-arm cost-counting bridge to the out-of-process worker (or a per-task gold mock).
;; ---------------------------------------------------------------------------
(def stats (atom {}))
(defn- bump! [arm resp]
  (swap! stats update arm
         (fn [s] (-> (or s {:calls 0 :samples 0 :gen-time 0.0 :completion-tokens 0 :errors 0})
                     (update :calls inc)
                     (update :samples + (count (:completions resp)))
                     (update :gen-time + (or (:gen_time_s resp) 0))
                     (update :completion-tokens + (or (:completion_tokens resp) 0))
                     (update :errors + (if (:error resp) 1 0))))))
(defn- counting-call [arm req]
  (let [resp (lp/call-server server-url req)]
    (when (:error resp) (println "    [llm error/" arm "]" (:error resp)))
    (bump! arm resp) resp))
(defn- mock-call
  "A native-free stand-in: returns the task's GOLD program + its crude, so every arm has a
   covering, scoring candidate. Validates ALL harness plumbing (arms, rerank, co-refine,
   cohort split, CI, cost) with NO worker. Numbers are not meaningful — plumbing is."
  [arm gold-code crude-code _req]
  (let [resp {:completions (vec (distinct (remove nil? [gold-code crude-code])))
              :gen_time_s 0.0 :prompt_tokens 0 :completion_tokens 0}]
    (bump! arm resp) resp))

;; ---------------------------------------------------------------------------
;; Scoring helpers.
;; ---------------------------------------------------------------------------
(defn- check-opts [] {:n-particles np})

(defn- verbatim-ev [code obs]
  (let [fb (syn/check code obs (check-opts))] (when (syn/scored? fb) (:evidence fb))))

(defn- coref-ev
  "Best exact evidence of a candidate program under the shared-σ line search: parse to a spec
   and co-refine σ over the grid; fall back to the verbatim check if it is off-grammar."
  [code obs]
  (if-let [sp (lp/parse-spec code)]
    (if (seq (:obs sp))
      (:evidence (syn/co-refine-spec sp obs grid (check-opts)))
      (verbatim-ev code obs))
    (verbatim-ev code obs)))

(defn- best-over [f cands obs]
  (let [es (keep #(f (:code %) obs) cands)]
    (when (seq es) (reduce max es))))

;; ---------------------------------------------------------------------------
;; The four arms for one task. OS-fixed and OS-coref share ONE candidate set.
;; ---------------------------------------------------------------------------
(defn- one-shot-arm [{:keys [task-desc observations]} call]
  (lp/one-shot-candidates (cond-> {:call-llm call :task-desc task-desc :observations observations
                                   :k k-oneshot :temperature 0.8 :max-tokens max-toks :seed seed}
                            system-prompt (assoc :system system-prompt))))

(defn- loop-arm [{:keys [task-desc observations] :as task} call co-refine?]
  (let [prop (h/loop-proposer (cond-> {:call-llm call :task-desc task-desc :observations observations
                                       :k k-step :temperature temp :max-tokens max-toks
                                       :revise revise :n-particles np :seed seed
                                       :co-refine-sigma? co-refine?}
                                system-prompt (assoc :system system-prompt)))
        run  (h/harvest-task task {:propose prop :init-spec (cur/crude-spec observations)
                                   :strategy :greedy :max-steps max-steps :n-particles np
                                   :co-refine-sigma? co-refine?})]
    (:final run)))

(defn- run-task [{:keys [id family cohort complexity solve-bar observations ground-truth-code] :as task} idx n]
  (let [crude (syn/render (cur/crude-spec observations))
        callf (fn [arm] (if mock? (partial mock-call arm ground-truth-code crude)
                            (partial counting-call arm)))
        ;; one shared one-shot candidate set, reranked two ways:
        os-c  (one-shot-arm task (callf :oneshot))
        os-fixed (best-over verbatim-ev os-c observations)
        os-coref (best-over coref-ev   os-c observations)
        l-strict (loop-arm task (callf :loop-strict) false)
        l-coref  (loop-arm task (callf :loop-coref)  true)
        sv?   (fn [e] (boolean (and e (js/isFinite e) (>= e solve-bar))))
        row   {:id (name id) :family (name family) :cohort (name (or cohort :nil))
               :complexity complexity :solve-bar solve-bar
               :os-fixed os-fixed :os-coref os-coref :l-strict l-strict :l-coref l-coref
               :os-fixed-solved (sv? os-fixed) :os-coref-solved (sv? os-coref)
               :l-strict-solved (sv? l-strict) :l-coref-solved  (sv? l-coref)
               ;; headline paired signal: does the co-refine LOOP beat the co-refine RERANK?
               :loop-beats-rerank (boolean (and l-coref (js/isFinite l-coref)
                                                (or (nil? os-coref) (not (js/isFinite os-coref))
                                                    (> l-coref (+ os-coref 0.1)))))}]
    (println (str "  [" (inc idx) "/" n "] " (:id row) " (" (:family row) "/" (:cohort row) " c" complexity ")"
                  "  OS-fix=" (fx os-fixed) (if (:os-fixed-solved row) "*" "")
                  "  OS-cor=" (fx os-coref) (if (:os-coref-solved row) "*" "")
                  "  L-str=" (fx l-strict) (if (:l-strict-solved row) "*" "")
                  "  L-cor=" (fx l-coref) (if (:l-coref-solved row) "*" "")
                  "  loop>rerank=" (:loop-beats-rerank row)))
    row))

;; ---------------------------------------------------------------------------
;; Bootstrap CI for a paired statistic over a set of rows (seeded LCG → reproducible).
;; ---------------------------------------------------------------------------
(def ^:private rng-state (atom (bit-or 1 (* seed 2654435761))))
(defn- next-u []
  (let [s (swap! rng-state (fn [x] (bit-and (+ (* x 1103515245) 12345) 0x7fffffff)))]
    (/ s 0x7fffffff)))
(defn- mean [xs] (when (seq xs) (/ (reduce + xs) (count xs))))
(defn- boot-ci
  "Bootstrap a 95% CI for (mean of `f` over a resample of `rows`). Returns {:mean :lo :hi}."
  [rows f]
  (let [vals (vec (keep f rows)) n (count vals)]
    (when (pos? n)
      (let [samples (sort (for [_ (range boot)]
                            (mean (for [_ (range n)] (nth vals (int (* (next-u) n)))))))
            at (fn [q] (nth samples (min (dec boot) (int (* q boot)))))]
        {:mean (mean vals) :lo (at 0.025) :hi (at 0.975) :n n}))))

(defn- solved-rate [rows k] (when (seq rows) (/ (count (filter k rows)) (double (count rows)))))

(defn- cohort-report [label rows]
  (when (seq rows)
    (println (str "\n  ── cohort " (str/upper-case label) "  (n=" (count rows) ") ──"))
    (doseq [[lbl sk ek] [["OS-fixed (R0 bar)" :os-fixed-solved :os-fixed]
                         ["OS-coref (rerank)" :os-coref-solved :os-coref]
                         ["L-strict (Phase3)" :l-strict-solved :l-strict]
                         ["L-coref  (R1 fix)" :l-coref-solved  :l-coref]]]
      (println (str "    " (.padEnd lbl 20) " solve=" (pct (solved-rate rows sk))
                    "  mean-ev=" (fx (mean (keep ek rows))))))
    ;; the decisive paired contrast: L-coref − OS-coref (σ held constant on both arms)
    (let [adv  (boot-ci rows (fn [r] (when (and (:l-coref r) (:os-coref r)
                                                (js/isFinite (:l-coref r)) (js/isFinite (:os-coref r)))
                                       (- (:l-coref r) (:os-coref r)))))
          wins (count (filter :loop-beats-rerank rows))
          ties (count (filter #(and (:l-coref %) (:os-coref %)
                                    (<= (js/Math.abs (- (:l-coref %) (:os-coref %))) 0.1)) rows))]
      (println (str "    L-coref − OS-coref  paired Δevidence  mean=" (fx (:mean adv))
                    "  95% CI=[" (fx (:lo adv)) ", " (fx (:hi adv)) "]  (n=" (:n adv) ")"))
      (println (str "    loop beats rerank on " wins "/" (count rows) " tasks (" ties " ties);  "
                    "GATE(this cohort) = "
                    (cond (and (:lo adv) (> (:lo adv) 0)) "GO-LOOP (CI-lo>0)"
                          (and (:hi adv) (< (:hi adv) 0)) "GO-RERANK (CI-hi<0)"
                          :else "GO-RERANK (CI straddles 0 — loop not proven better)")))
      {:cohort label :n (count rows)
       :solve {:os-fixed (solved-rate rows :os-fixed-solved) :os-coref (solved-rate rows :os-coref-solved)
               :l-strict (solved-rate rows :l-strict-solved)  :l-coref (solved-rate rows :l-coref-solved)}
       :loop-minus-rerank adv :loop-wins wins :ties ties})))

;; ---------------------------------------------------------------------------
(def C (cur/generate-curriculum (cond-> {:round round :instances-per-family instances :eval-families eval-fams}
                                  families (assoc :families families))))
(def evals0 (:eval-tasks C))
(def evals  (if (pos? tasks-limit) (vec (take tasks-limit evals0)) evals0))

(.mkdirSync fs out-dir #js {:recursive true})
(println (str "\n###  R2 BAKE-OFF  (tier=" tier (when mock? " [MOCK]") ")  ###"))
(println (str "  curriculum round " round ", " instances "/family; eval-families " (vec eval-fams)
              " -> " (count evals0) " eval tasks" (when (pos? tasks-limit) (str " (capped " (count evals) ")"))))
(println (str "  compute-matched K_OS=" k-oneshot " (= K_STEP " k-step " × MAX_STEPS " max-steps
              " × (1+REVISE " revise "));  natural difficulty;  σ-grid " (count grid) "-pt"))
(println "  arms: OS-fixed | OS-coref(rerank) | L-strict(Phase3) | L-coref(R1 fix)")

(def t0 (js/Date.now))
(def results (vec (map-indexed (fn [i t] (run-task t i (count evals))) evals)))
(def wall (/ (- (js/Date.now) t0) 1000.0))

(println "\n### COHORT SUMMARIES  (FAMILY-SPLIT is the headline — rrps leakage lesson) ###")
(def within (filter #(= "within" (:cohort %)) results))
(def famc   (filter #(= "family" (:cohort %)) results))
(def reports (vec (keep identity [(cohort-report "within" within)
                                  (cohort-report "family" famc)
                                  (cohort-report "ALL"    results)])))

(println "\n### PER-FAMILY (L-coref vs OS-coref solve-rate) ###")
(doseq [[fam rows] (sort-by key (group-by :family results))]
  (println (str "  " (.padEnd fam 18) " n=" (count rows)
                "  L-coref=" (pct (solved-rate rows :l-coref-solved))
                "  OS-coref=" (pct (solved-rate rows :os-coref-solved))
                "  L-strict=" (pct (solved-rate rows :l-strict-solved))
                "  OS-fixed=" (pct (solved-rate rows :os-fixed-solved)))))

(def st @stats)
(println "\n### COST PER ARM ###")
(doseq [arm [:oneshot :loop-strict :loop-coref]]
  (let [s (get st arm)]
    (when s (println (str "  " (.padEnd (name arm) 12) " " (:calls s) " calls, " (:samples s) " samples, "
                          (fx (:gen-time s)) "s gen, " (:completion-tokens s) " tok"
                          (when (pos? (:errors s)) (str ", " (:errors s) " errors")))))))
(println (str "  wall " (fx wall) "s"))

(def famgate (some #(when (= "family" (:cohort %)) %) reports))
(println "\n================  R2 VERDICT  ================")
(if famgate
  (let [adv (:loop-minus-rerank famgate)]
    (println (str "  FAMILY-SPLIT (n=" (:n famgate) "): L-coref − OS-coref Δev 95% CI ["
                  (fx (:lo adv)) ", " (fx (:hi adv)) "]; loop wins " (:loop-wins famgate) "/" (:n famgate)))
    (println (str "  => " (cond (and (:lo adv) (> (:lo adv) 0))
                                "GO-LOOP on the natural families (verify with the real GMM cliff family, R4.5)."
                                :else
                                "GO-RERANK (evidence-favored default): the loop does not beat compute-matched one-shot+rerank.")))
    (println "  NOTE: the FULL gate also requires the real non-conjugate cliff family (GMM via RB"
             "\n  scoring, genmlx-9mos). This run covers the natural linear-Gaussian families where the"
             "\n  de-masked R0 25% bar lives; wire a :gmm eval-family once rb_mixture lands."))
  (println "  (no :family-cohort tasks in this run — set EVAL_FAMILIES to a held-out family.)"))

(def report {:tier tier :mock mock? :round round :instances instances
             :config {:k-oneshot k-oneshot :k-step k-step :max-steps max-steps :revise revise
                      :temp temp :np np :boot boot :eval-families (vec eval-fams)}
             :n-eval (count evals) :reports reports :results results :cost st :wall-s wall})
(def outfile (.join path out-dir (str "r2_bakeoff_" tier (when mock? "_mock") ".json")))
(.writeFileSync fs outfile (js/JSON.stringify (clj->js report) nil 2))
(println (str "\n  wrote " outfile))
