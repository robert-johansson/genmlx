(ns harvest-corpus
  "Phase 3 (genmlx-oexl), done-means #1 AT SCALE: run the EXISTING curriculum -> loop ->
   trajectories -> build-corpus pipeline (genmlx.world.{curriculum,harvest,repl-corpus})
   over the FULL train split with the REAL LLM loop-proposer, and write the leakage-safe
   propose-eval-revise SFT corpus the 0.8B student trains on.

   This is the thin DRIVER over a tested core (genmlx.world.harvest + repl-corpus +
   curriculum, all green): it generates the curriculum, runs each TRAIN task through the
   greedy (or beam) driver with the shared `harvest/loop-proposer`, harvests the accepted
   transitions with `repl-corpus/build-corpus` using the curriculum's :eval-task-ids
   (leakage-safe), and writes:
     - runs.jsonl       (streamed: one {:task :trajectory ...} per completed task; a worker
                         death mid-harvest never loses the runs already finished)
     - train_rows.jsonl (the leakage-safe SFT corpus — exactly sft_prep's --corpus shape:
                         {:task-id :kind :rank-key :messages}; ONLY train-task rows)
     - report.json      (corpus stats, per-task coverage, cost, config, the eval-task-ids
                         the in-loop eval must hold out)

   The LLM lives OUT-OF-PROCESS (scripts/llm_server.py); inject via SERVER_URL. With
   PROPOSER=family the run is NATIVE-FREE (the no-LLM structured family-proposer) — the
   end-to-end plumbing validation before spending a single 35B token.

   Run (native-free validation):  PROPOSER=family INSTANCES=2 TRAIN_LIMIT=6 \\
                                    bun run --bun nbb scripts/harvest_corpus.cljs
   Run (real 35B, via the worker): scripts/run_llm_probe.sh-style worker up, then
                                    PROPOSER=llm SERVER_URL=http://127.0.0.1:8765 \\
                                    bun run --bun nbb scripts/harvest_corpus.cljs
   Env: ROUND INSTANCES FAMILIES(csv) TRAIN_LIMIT PROPOSER(llm|family) STRATEGY(greedy|beam)
        K_STEP TEMP MAX_TOKENS REVISE MAX_STEPS BEAM_WIDTH NP SEED SERVER_URL TAG OUT."
  (:require [genmlx.world.curriculum :as cur]
            [genmlx.world.harvest :as h]
            [genmlx.world.repl-corpus :as rc]
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

(def round      (envi "ROUND" 0))
(def instances  (envi "INSTANCES" 20))
(def families   (when-let [s (env "FAMILIES" nil)] (mapv keyword (str/split s #","))))
(def train-limit (envi "TRAIN_LIMIT" 0))         ;; 0 = no cap (full train split)
(def proposer-kind (env "PROPOSER" "llm"))       ;; "llm" | "family" (native-free validation)
(def strategy   (keyword (env "STRATEGY" "greedy")))
(def k-step     (envi "K_STEP" 4))
(def temp       (envf "TEMP" 0.7))
(def max-toks   (envi "MAX_TOKENS" 320))
(def revise     (envi "REVISE" 2))
(def max-steps  (envi "MAX_STEPS" 6))
(def beam-width (envi "BEAM_WIDTH" 4))
(def np         (envi "NP" 2000))
(def seed       (envi "SEED" 1))
(def server-url (env "SERVER_URL" "http://127.0.0.1:8765"))
(def tag        (env "TAG" (str (name proposer-kind) "-r" round)))
(def out-dir    (env "OUT" (home "genmlx-loop-artifacts" "harvest" tag)))

;; ---------------------------------------------------------------------------
;; Cost-counting wrapper around the out-of-process bridge (LLM proposer only).
;; ---------------------------------------------------------------------------
(def stats (atom {:calls 0 :samples 0 :gen-time 0.0 :prompt-tokens 0 :completion-tokens 0 :errors 0}))
(defn counting-call [req]
  (let [resp (lp/call-server server-url req)]
    (swap! stats #(-> %
                      (update :calls inc)
                      (update :samples + (count (:completions resp)))
                      (update :gen-time + (or (:gen_time_s resp) 0))
                      (update :prompt-tokens + (or (:prompt_tokens resp) 0))
                      (update :completion-tokens + (or (:completion_tokens resp) 0))
                      (update :errors + (if (:error resp) 1 0))))
    (when (:error resp) (println "    [llm error]" (:error resp)))
    resp))

;; ---------------------------------------------------------------------------
;; Proposer factory: per-task LLM loop-proposer, or the native-free family-proposer.
;; ---------------------------------------------------------------------------
(defn proposer-for [task]
  (case proposer-kind
    "family" (cur/family-proposer task)
    "llm"    (h/loop-proposer {:call-llm counting-call
                               :task-desc (:task-desc task) :observations (:observations task)
                               :k k-step :temperature temp :max-tokens max-toks
                               :revise revise :n-particles np :seed seed})
    (throw (js/Error. (str "unknown PROPOSER " proposer-kind " (want llm|family)")))))

;; ---------------------------------------------------------------------------
;; Generate the curriculum + select the train tasks to harvest.
;; ---------------------------------------------------------------------------
(def C (cur/generate-curriculum (cond-> {:round round :instances-per-family instances}
                                  families (assoc :families families))))
(def all-train (:train-tasks C))
(def train (if (pos? train-limit) (vec (take train-limit all-train)) all-train))

(.mkdirSync fs out-dir #js {:recursive true})
(def runs-file (.join path out-dir "runs.jsonl"))
(.writeFileSync fs runs-file "")                  ;; truncate (fresh harvest)

(println (str "\n### HARVEST  tag=" tag "  proposer=" proposer-kind "  strategy=" (name strategy)))
(println (str "  curriculum round " round ", " instances "/family"
              (when families (str ", families " (vec families)))
              " -> " (count all-train) " train tasks"
              (when (pos? train-limit) (str " (capped to " (count train) ")"))
              ", " (count (:eval-task-ids C)) " held-out eval ids"))
(when (= "llm" proposer-kind)
  (println (str "  LLM: " server-url "  K_STEP=" k-step " TEMP=" temp " REVISE=" revise
                " MAX_STEPS=" max-steps " MAX_TOKENS=" max-toks)))
(println (str "  out -> " out-dir))

;; ---------------------------------------------------------------------------
;; Run the harvest (streaming each completed run to runs.jsonl for resilience).
;; ---------------------------------------------------------------------------
(def t0 (js/Date.now))
(defn on-run [run idx task]
  (.appendFileSync fs runs-file
                   (str (js/JSON.stringify (clj->js {:task (:task run) :trajectory (:trajectory run)
                                                     :steps (:steps run) :final (:final run)
                                                     :solved? (:solved? run)
                                                     :stop-reason (:stop-reason run)})) "\n"))
  (let [st @stats]
    (println (str "  [" (inc idx) "/" (count train) "] " (name (:id task))
                  " (" (name (:family task)) " c" (:complexity task) ")"
                  "  steps=" (:steps run) " final=" (fx (:final run))
                  " bar=" (fx (:solve-bar task)) " solved=" (:solved? run)
                  (when (= "llm" proposer-kind)
                    (str "  | calls=" (:calls st) " toks=" (:completion-tokens st)
                         " gen=" (fx (:gen-time st)) "s"))))))

(def runs (h/harvest-tasks train proposer-for
                           {:init-spec-for #(cur/crude-spec (:observations %))
                            :run-opts {:strategy strategy :max-steps max-steps :n-particles np
                                       :beam-width beam-width :adaptive? true :seed seed}
                            :on-run on-run}))
(def wall (/ (- (js/Date.now) t0) 1000.0))

;; ---------------------------------------------------------------------------
;; Harvest the corpus (leakage-safe via the curriculum's eval-task-ids) + report.
;; ---------------------------------------------------------------------------
(def corpus (rc/build-corpus runs {:eval-ids (:eval-task-ids C) :n-particles np}))
(def train-rows (:train-rows corpus))
(def rows-file (.join path out-dir "train_rows.jsonl"))
(.writeFileSync fs rows-file
                (str (str/join "\n" (map #(js/JSON.stringify (clj->js %)) train-rows)) "\n"))

(def n-solved (count (filter :solved? runs)))
(def report {:tag tag :proposer proposer-kind :strategy (name strategy)
             :config {:round round :instances instances :families families
                      :train-limit train-limit :k-step k-step :temp temp :revise revise
                      :max-steps max-steps :max-tokens max-toks :np np :seed seed}
             :n-train-tasks (count train) :n-solved n-solved
             :solve-rate (/ n-solved (max 1 (count train)))
             :corpus {:n-runs (:n-runs corpus) :n-rows (:n-rows corpus)
                      :n-train-rows (count train-rows)
                      :n-dropped-eval (count (:dropped-eval corpus))
                      :per-task (:per-task corpus)
                      :train-task-ids (vec (:train-task-ids corpus))}
             :eval-task-ids (vec (:eval-task-ids C))
             :cost @stats :wall-s wall})
(def report-file (.join path out-dir "report.json"))
(.writeFileSync fs report-file (js/JSON.stringify (clj->js report) nil 2))

(println (str "\n### DONE  tag=" tag))
(println (str "  tasks: " (count train) "  solved (cleared bar): " n-solved
              "  (" (.toFixed (* 100.0 (/ n-solved (max 1 (count train)))) 0) "%)"))
(println (str "  corpus: " (:n-rows corpus) " rows harvested, " (count train-rows)
              " train-rows (leakage-safe), " (count (:dropped-eval corpus)) " dropped-eval"))
(println (str "  per-task rows: " (:per-task corpus)))
(when (seq (:eval-task-ids-present (rc/build-corpus runs {:eval-ids (:eval-task-ids C)})))
  (println "  WARNING: held-out eval ids appeared in the harvested rows (should be empty)"))
(when (= "llm" proposer-kind)
  (let [st @stats]
    (println (str "  cost: " (:calls st) " calls, " (:samples st) " samples, "
                  (fx (:gen-time st)) "s gen, " (:completion-tokens st) " completion tokens"
                  (when (pos? (:errors st)) (str ", " (:errors st) " transport errors"))))))
(println (str "  wall " (fx wall) "s  ->  " out-dir))
(println (str "  next: bun run --bun nbb scripts/sft_prep.cljs --corpus " rows-file
              " --out $TMPDIR/genmlx-sft  &&  bash scripts/sft_train.sh"))
