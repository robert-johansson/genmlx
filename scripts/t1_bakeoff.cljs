(ns t1-bakeoff
  "T1 bake-off GENERATION phase (genmlx-8lm2): run ONE instruction-prompted model
   arm over the unified task battery (genmlx.world.t1-battery — the 12 distill
   seed tasks + 5 lifted MSA program tasks), model resident ONCE per invocation.
   Scoring is a separate, model-free phase (scripts/t1_score.cljs) run after this
   process exits — the GPU never waits on the oracle.

   CHAT-INSTRUCTION PROMPTING ONLY, for every arm. This runner deliberately does
   NOT do FIM: the FIM-vs-instruct asymmetry is a separate 80B-only probe
   (genmlx-ttm8 lineage) and is out of T1 scope, so arms differ ONLY by checkpoint.

   Sampling mirrors scripts/distill_teacher.cljs: sample 0 is a greedy anchor
   (GREEDY_FIRST=1, the default), samples 1..K-1 draw at TEMPERATURE with :seed
   (+ SEED i) — so a killed+resumed run regenerates the SAME candidate set, and
   (task, sample) pairs already present in OUT are always skipped (resume-stable,
   always on). Records append line-buffered JSONL (snake-case keys):
     {:task_id :kind :sample_idx :raw_text :n_tokens :gen_ms :seed :greedy}
   with :n_tokens / :gen_ms from llm/generate-text-raw+ (the decode loop's own
   token count — never re-encoded). Totals + frozen config land in
   <OUT-dir>/gen-<ARM>-meta.json at the end.

   STUB=1 replays canned completions through the SAME record-writing path with
   ZERO native/GPU loading (the LLM backend is required dynamically only for
   real runs): sample 0 = a known-correct completion (for the lifted MSA tasks,
   the battery's own prompt template via t1-battery/exemplars), sample 1 = an
   unbalanced parse-failure, samples 2+ = a parseable-but-wrong completion — so
   a stub run exercises the kept / unparseable / test-fail / uncovered oracle
   paths end-to-end. Stub :n_tokens is an honest chars/4 proxy (no tokenizer is
   loaded); stub :gen_ms is the (trivial) measured wall clock.

   LIFECYCLE (Thor discipline): ONE GPU process at a time; run real arms through
   ~/genmlx-guarded-run.sh (see scripts/run_t1_bakeoff.sh for the launcher).

   Run (from repo root):
     ARM=a MODEL_DIR=<checkpoint dir> K=4 SEED=42 TEMPERATURE=0.8 MAX_TOKENS=512 \\
       bunx --bun nbb@1.4.208 scripts/t1_bakeoff.cljs
   Stub (no GPU):
     STUB=1 ARM=stub K=2 bunx --bun nbb@1.4.208 scripts/t1_bakeoff.cljs
   Env: ARM (label, required) MODEL_DIR (required unless STUB=1 — no default
        model) K (samples/task, 4) SEED (42) TEMPERATURE (0.8) MAX_TOKENS (512)
        TASKS (comma-separated task-id subset) OUT (results/t1-bakeoff/
        gen-<ARM>.jsonl) STUB GREEDY_FIRST (1)"
  (:require [genmlx.world.t1-battery :as battery]
            [clojure.string :as str]
            [promesa.core :as p]))

;; bench/ is not on the nbb classpath (nbb.edn :paths); these scripts run from
;; the repo root (the same cwd assumption the distill sandbox makes), so load
;; the shared JSON/dir helpers by path.
(require '[nbb.core])
(nbb.core/load-file "bench/util.cljs")
(require '[bench.util :as bu])

(def fs   (js/require "fs"))
(def path (js/require "path"))
(def cp   (js/require "child_process"))

(defn- env  [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))
(defn- envf [k d] (let [v (env k nil)] (if v (js/parseFloat v) d)))

(def stub?         (= "1" (env "STUB" "0")))
(def arm           (env "ARM" (when stub? "stub")))
(def model-dir     (env "MODEL_DIR" nil))   ;; required for real runs — deliberately no default
(def k-samples     (envi "K" 4))
(def seed          (envi "SEED" 42))
(def temp          (envf "TEMPERATURE" 0.8)  ;; NOT "TEMP" — Bun honors TEMP as its temp-dir (a bunx cache landed in ./0.8/ once))
(def max-tokens    (envi "MAX_TOKENS" 512))
(def greedy-first? (= "1" (env "GREEDY_FIRST" "1")))
(def task-subset   (when-let [s (env "TASKS" nil)]
                     (set (map str/trim (str/split s #",")))))
(def out-file      (env "OUT" (str "results/t1-bakeoff/gen-" arm ".jsonl")))
(def out-dir       (.dirname path out-file))

(def tasks
  (if task-subset
    (let [ts (filterv #(task-subset (:id %)) battery/tasks)
          unknown (remove (set (map :id ts)) task-subset)]
      (when (seq unknown)
        (println "WARN: unknown task ids in TASKS:" (vec unknown)))
      ts)
    battery/tasks))

;; ---------------------------------------------------------------------------
;; JSONL out + resume
;; ---------------------------------------------------------------------------

(defn- append-jsonl! [p row]
  (.appendFileSync fs p (str (js/JSON.stringify (clj->js row)) "\n")))

(defn- done-keys
  "#{[task_id sample_idx] ...} already present in OUT — those pairs are skipped
   (resume-stable; per-sample seeds make the regenerated remainder identical)."
  []
  (if (.existsSync fs out-file)
    (->> (str/split-lines (.readFileSync fs out-file "utf8"))
         (remove str/blank?)
         (keep (fn [l] (try (js->clj (js/JSON.parse l) :keywordize-keys true)
                            (catch :default _ nil))))
         (into #{} (map (juxt :task_id :sample_idx))))
    #{}))

(defn- git-sha []
  (try (str/trim (.toString (.execSync cp "git rev-parse --short HEAD")))
       (catch :default _ "unknown")))

(def totals (atom {:samples 0 :completion-tokens 0 :gen-ms 0 :errors 0}))

(defn- file-totals
  "Cost totals over EVERYTHING in OUT (not just this invocation) + this run's
   error count — so a resumed run's meta json still describes the whole file
   it sits next to."
  []
  (let [rows (if (.existsSync fs out-file)
               (->> (str/split-lines (.readFileSync fs out-file "utf8"))
                    (remove str/blank?)
                    (keep (fn [l] (try (js->clj (js/JSON.parse l) :keywordize-keys true)
                                       (catch :default _ nil)))))
               [])]
    {:samples (count rows)
     :completion-tokens (reduce + 0 (keep :n_tokens rows))
     :gen-ms (reduce + 0 (keep :gen_ms rows))
     :errors (:errors @totals)}))

;; ---------------------------------------------------------------------------
;; The two generators. Both return {:generate (fn [task sample-opts] ->
;; promise<{:text :n-tokens :gen-ms}>) :after-sample! (fn [])} so the loop
;; below is a single shared record-writing code path.
;; ---------------------------------------------------------------------------

(def ^:private stub-correct
  "task-id -> a canned KNOWN-CORRECT completion. The 12 distill-seed entries are
   hand-written here; the 5 lifted MSA entries come from the battery's own
   prompt templates (t1-battery/exemplars), so prompt and stub can never drift."
  (merge
   {"gaussian-mean-near2"
    (str "(fn [trace] (let [mu (trace :mu (dist/gaussian 2 3))]"
         " {:y0 (trace :y0 (dist/gaussian mu 1)) :y1 (trace :y1 (dist/gaussian mu 1))"
         " :y2 (trace :y2 (dist/gaussian mu 1)) :y3 (trace :y3 (dist/gaussian mu 1))}))")
    "gaussian-mean-negshift"
    (str "(fn [trace] (let [mu (trace :mu (dist/gaussian -3 3))]"
         " {:y0 (trace :y0 (dist/gaussian mu 1)) :y1 (trace :y1 (dist/gaussian mu 1))"
         " :y2 (trace :y2 (dist/gaussian mu 1)) :y3 (trace :y3 (dist/gaussian mu 1))}))")
    "beta-bernoulli-coin"
    (str "(fn [trace] (let [p (trace :p (dist/beta 2 2))]"
         " {:f0 (trace :f0 (dist/bernoulli p)) :f1 (trace :f1 (dist/bernoulli p))"
         " :f2 (trace :f2 (dist/bernoulli p)) :f3 (trace :f3 (dist/bernoulli p))"
         " :f4 (trace :f4 (dist/bernoulli p)) :f5 (trace :f5 (dist/bernoulli p))}))")
    "gamma-poisson-counts"
    (str "(fn [trace] (let [rate (trace :rate (dist/gamma 4 1))]"
         " {:c0 (trace :c0 (dist/poisson rate)) :c1 (trace :c1 (dist/poisson rate))"
         " :c2 (trace :c2 (dist/poisson rate)) :c3 (trace :c3 (dist/poisson rate))"
         " :c4 (trace :c4 (dist/poisson rate))}))")
    "counter-machine"
    (str "(fn [state action] (case action :inc (update state :count inc)"
         " :dec (update state :count dec) :reset (assoc state :count 0) state))")
    "traffic-light"
    (str "(fn [state action] (if (= action :tick)"
         " {:light ({:red :green :green :yellow :yellow :red} (:light state))} state))")
    "toggle-switch"
    "(fn [state action] (if (= action :flip) (update state :on not) state))"
    "factorial"
    "(fn [n] (reduce * 1 (range 1 (inc n))))"
    "fizzbuzz"
    (str "(fn [n] (cond (zero? (mod n 15)) \"FizzBuzz\" (zero? (mod n 3)) \"Fizz\""
         " (zero? (mod n 5)) \"Buzz\" :else (str n)))")
    "gcd"
    "(fn [a b] (if (zero? b) a (recur b (mod a b))))"
    "palindrome?"
    "(fn [s] (= s (apply str (reverse s))))"
    "sum-evens"
    "(fn [coll] (reduce + 0 (filter even? coll)))"}
   battery/exemplars))

(def ^:private stub-parse-fail
  "Unbalanced — must land as :unparseable in the oracle."
  "(fn [state action] (assoc state")

(defn- stub-wrong
  "Parseable + evaluable but WRONG: a :program that ignores every observed site
   (lands :uncovered) / a :function that fails the held-out checks (:test-fail)."
  [kind]
  (if (= :program kind)
    "(fn [trace] {:z (trace :z (dist/gaussian 0 1))})"
    "(fn [& args] :wrong)"))

(defn- stub-completion [task i]
  (cond
    (zero? i) (or (get stub-correct (:id task)) (stub-wrong (:kind task)))
    (= 1 i)   stub-parse-fail
    :else     (stub-wrong (:kind task))))

(defn- stub-generator []
  (p/resolved
   {:generate (fn [task {:keys [sample-idx]}]
                (let [t0   (.now js/Date)
                      text (stub-completion task sample-idx)]
                  (p/resolved {:text text
                               ;; honest proxy: no tokenizer in stub mode
                               :n-tokens (max 1 (js/Math.round (/ (count text) 4)))
                               :gen-ms (- (.now js/Date) t0)})))
    :after-sample! (fn [])}))

(defn- real-generator
  "Dynamically require the LLM backend (which js/require's the native addon —
   this is what keeps STUB=1 runs native-free) and load the model ONCE."
  []
  (p/let [_ (require '[genmlx.llm.backend]
                     '[genmlx.mlx])]
    (let [load-model (resolve 'genmlx.llm.backend/load-model)
          gen+       (resolve 'genmlx.llm.backend/generate-text-raw+)
          force-gc!  (resolve 'genmlx.mlx/force-gc!)
          t0         (.now js/Date)]
      (p/let [m (load-model model-dir)]
        (println (str "  loaded: " (name (:type m)) " in "
                      (js/Math.round (/ (- (.now js/Date) t0) 1000)) " s"))
        {:generate (fn [task {:keys [greedy? sample-seed]}]
                     (gen+ m (:prompt task)
                           {:max-tokens  max-tokens
                            :temperature (if greedy? 0 temp)
                            :seed        sample-seed
                            :system-prompt (or (:system-prompt task)
                                               "You are a ClojureScript code generator.")}))
         ;; dead decode graphs are dark pages on Tegra (R4 lesson, genmlx-h3p5)
         ;; — sweep between samples
         :after-sample! force-gc!}))))

;; ---------------------------------------------------------------------------
;; The generation loop (shared by stub and real arms)
;; ---------------------------------------------------------------------------

(defn- run-samples! [{:keys [generate after-sample!]} done]
  (p/loop [ts (seq tasks), n-gen 0]
    (if-not ts
      (p/resolved n-gen)
      (let [{:keys [id kind] :as task} (first ts)]
        (p/let [made
                (p/loop [i 0, made 0]
                  (if (= i k-samples)
                    (p/resolved made)
                    (if (contains? done [id i])
                      (p/recur (inc i) made)
                      (let [greedy?     (and greedy-first? (zero? i))
                            sample-seed (+ seed i)]
                        (p/let [r (-> (generate task {:sample-idx i
                                                      :greedy? greedy?
                                                      :sample-seed sample-seed})
                                      (p/catch (fn [e] {:error (.-message e)})))]
                          (if (:error r)
                            (do (swap! totals update :errors inc)
                                (println (str "  " id "[" i "] ERROR: " (:error r)))
                                (p/recur (inc i) made))
                            (let [{:keys [text n-tokens gen-ms]} r]
                              (append-jsonl! out-file
                                             {:task_id id :kind (name kind)
                                              :sample_idx i :raw_text text
                                              :n_tokens n-tokens :gen_ms gen-ms
                                              :seed sample-seed :greedy greedy?})
                              (swap! totals #(-> %
                                                 (update :samples inc)
                                                 (update :completion-tokens + n-tokens)
                                                 (update :gen-ms + gen-ms)))
                              (println (str "  " id "[" i "] " gen-ms " ms, "
                                            n-tokens " tok"))
                              (after-sample!)
                              (p/recur (inc i) (inc made)))))))))]
          (p/recur (next ts) (+ n-gen made)))))))

(defn -main []
  (when-not arm
    (println "usage: ARM=<label> MODEL_DIR=<dir> bunx --bun nbb@1.4.208 scripts/t1_bakeoff.cljs")
    (js/process.exit 1))
  (when (and (not stub?) (not model-dir))
    (println "MODEL_DIR is required for a real arm (no default model; STUB=1 for the dry run)")
    (js/process.exit 1))
  (bu/ensure-dir out-dir)
  (let [done (done-keys)
        t0   (.now js/Date)]
    (println (str "== t1_bakeoff arm=" arm (when stub? " [STUB]") " =="))
    (println (str "  model : " (if stub? "STUB (canned completions)" model-dir)))
    (println (str "  tasks : " (count tasks) "  samples/task: " k-samples
                  "  temp: " temp "  max-tokens: " max-tokens
                  "  seed: " seed (when greedy-first? "  greedy-first")))
    (when (seq done) (println (str "  resume: " (count done) " (task, sample) pairs already in " out-file)))
    (p/let [g     (if stub? (stub-generator) (real-generator))
            n-gen (run-samples! g done)]
      (bu/write-json out-dir (str "gen-" arm "-meta.json")
                     {:arm arm
                      :model-dir (if stub? "STUB" model-dir)
                      :k k-samples :seed seed :temp temp :max-tokens max-tokens
                      :git-sha (git-sha)
                      :battery-ids (mapv :id tasks)
                      :stub stub? :greedy-first greedy-first?
                      :totals (file-totals)
                      :wall-ms (- (.now js/Date) t0)})
      (println (str "== done: " n-gen " new candidates -> " out-file " ==")))))

(-> (-main)
    (p/catch (fn [e]
               (println "UNCAUGHT:" (.-message e))
               (println (.-stack e))
               (set! (.-exitCode js/process) 1))))
