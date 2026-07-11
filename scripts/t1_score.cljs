(ns t1-score
  "T1 bake-off SCORING phase (genmlx-8lm2): MODEL-FREE — run after the GPU arm
   (scripts/t1_bakeoff.cljs) has exited. Pipes the arm's gen JSONL through the
   distillation oracle under process isolation (genmlx.world.distill-sandbox
   collect-verdicts -> genmlx.world.distill evaluate-candidate, battery \"t1\"),
   then aggregates:

     - SCI pass-rate over :function candidates (kept?), with a seeded-LCG
       bootstrap 95% CI (the scripts/r2_bakeoff.cljs pattern — deterministic
       under SEED)
     - reader-validity (parse) rate over all candidates, same bootstrap CI on
       the overall keep-rate
     - evidence: mean/max :log-ml over KEPT :program candidates
     - tokens-to-first-valid: per task, cumulative :n_tokens in sample order
       until the first kept candidate (median across tasks that reached one)
     - wall-clock + completion-token totals from gen-<ARM>-meta.json

   Writes results/t1-bakeoff/score-<ARM>.json (plus the raw verdict lines as
   score-<ARM>-verdicts.edn, an audit artifact) and prints a compact table.

   COMPARE=a,b mode reads two score jsons and prints/writes a side-by-side
   delta table (results/t1-bakeoff/compare-<a>-vs-<b>.json). No sandbox runs.

   NOTE: model-free is not native-free — the oracle scores programs with the
   GenMLX core (tiny scalar graphs), so the native addon loads, but no LLM
   checkpoint ever does. The sandbox worker spawns `bun run --bun nbb` from the
   repo root; run this script from the repo root too.

   Run (from repo root):
     ARM=a bunx --bun nbb@1.4.208 scripts/t1_score.cljs
     COMPARE=a,b bunx --bun nbb@1.4.208 scripts/t1_score.cljs
   Env: ARM | COMPARE (one required)  IN (results/t1-bakeoff/gen-<ARM>.jsonl)
        OUT_DIR (results/t1-bakeoff)  NP (50)  SEED (42)  BOOT (2000)
        TIMEOUT_MS (60000 — per-candidate stall budget; must exceed the
        worker's nbb+native cold start, minutes-scale safe on Thor)"
  (:require [genmlx.world.t1-battery :as battery]
            [genmlx.world.distill-sandbox :as sandbox]
            [clojure.string :as str]
            [promesa.core :as p]))

;; bench/ is not on the nbb classpath (nbb.edn :paths); load the shared JSON
;; helpers by path (repo-root cwd, same assumption as the sandbox worker spawn).
(require '[nbb.core])
(nbb.core/load-file "bench/util.cljs")
(require '[bench.util :as bu])

(def fs (js/require "fs"))

(defn- env  [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))

(def compare-spec (env "COMPARE" nil))
(def arm          (env "ARM" nil))
(def out-dir      (env "OUT_DIR" "results/t1-bakeoff"))
(def in-file      (env "IN" (str out-dir "/gen-" arm ".jsonl")))
(def np           (envi "NP" 50))
(def seed         (envi "SEED" 42))
(def boot         (envi "BOOT" 2000))
(def timeout-ms   (envi "TIMEOUT_MS" 60000))

(defn- fx  [x] (if (and (number? x) (js/isFinite x)) (.toFixed (js/Number x) 2) "--"))
(defn- pct [x] (if (and (number? x) (js/isFinite x)) (str (.toFixed (* 100 x) 0) "%") "--"))
(defn- pad [s n] (.padEnd (str s) n))

;; ---------------------------------------------------------------------------
;; Bootstrap CI — the seeded-LCG pattern from scripts/r2_bakeoff.cljs (:181-194),
;; deterministic under SEED.
;; ---------------------------------------------------------------------------
(def ^:private rng-state (atom (bit-or 1 (* seed 2654435761))))
(defn- next-u []
  (let [s (swap! rng-state (fn [x] (bit-and (+ (* x 1103515245) 12345) 0x7fffffff)))]
    (/ s 0x7fffffff)))
(defn- mean [xs] (when (seq xs) (/ (reduce + xs) (count xs))))
(defn- boot-ci
  "Bootstrap a 95% CI for (mean of `f` over a resample of `rows`). Returns {:mean :lo :hi :n}."
  [rows f]
  (let [vals (vec (keep f rows)) n (count vals)]
    (when (pos? n)
      (let [samples (sort (for [_ (range boot)]
                            (mean (for [_ (range n)] (nth vals (int (* (next-u) n)))))))
            at (fn [q] (nth samples (min (dec boot) (int (* q boot)))))]
        {:mean (mean vals) :lo (at 0.025) :hi (at 0.975) :n n}))))

(defn- median [xs]
  (when (seq xs)
    (let [s (vec (sort xs)) n (count s) m (quot n 2)]
      (if (odd? n) (nth s m) (/ (+ (nth s (dec m)) (nth s m)) 2.0)))))

(defn- rate [pred xs]
  (when (seq xs) (/ (count (filter pred xs)) (double (count xs)))))

;; ---------------------------------------------------------------------------
;; Aggregation
;; ---------------------------------------------------------------------------

(defn- tokens-to-first-valid
  "Cumulative :n_tokens over a task's rows in :sample_idx order until (and
   including) the first KEPT candidate; nil when the task never reached one."
  [task-rows vmap]
  (loop [rs (sort-by :sample_idx task-rows), cum 0]
    (when-let [r (first rs)]
      (let [cum' (+ cum (or (:n_tokens r) 0))]
        (if (:kept? (get vmap [(:task_id r) (:sample_idx r)]))
          cum'
          (recur (next rs) cum'))))))

(defn- task-report [tid task-rows vmap]
  (let [vs      (keep #(get vmap [(:task_id %) (:sample_idx %)]) task-rows)
        kind    (:kind (first vs))
        kept    (filter :kept? vs)
        log-mls (keep :log-ml kept)]
    {:task-id tid
     :kind (some-> kind name)
     :n (count vs)
     :parse-rate (rate :parse? vs)
     :pass-rate (rate :kept? vs)
     :log-ml-mean (mean log-mls)
     :log-ml-max (when (seq log-mls) (reduce max log-mls))
     :tokens-to-first-valid (tokens-to-first-valid task-rows vmap)}))

(defn- overall-report [rows vmap meta-json per-task]
  (let [vs       (keep #(get vmap [(:task_id %) (:sample_idx %)]) rows)
        fns      (filter #(= :function (:kind %)) vs)
        progs    (filter #(= :program (:kind %)) vs)
        kept-p   (filter :kept? progs)
        log-mls  (keep :log-ml kept-p)
        ttfv     (keep :tokens-to-first-valid per-task)
        kept01   #(if (:kept? %) 1.0 0.0)]
    {:n-rows (count rows)
     :n-verdicts (count vs)
     :parse-rate (rate :parse? vs)
     :keep-rate (rate :kept? vs)
     :keep-ci (boot-ci vs kept01)
     :function-pass-rate (rate :kept? fns)
     :function-pass-ci (boot-ci fns kept01)
     :program-keep-rate (rate :kept? progs)
     :evidence {:mean (mean log-mls)
                :max (when (seq log-mls) (reduce max log-mls))
                :n-kept (count kept-p)}
     :tokens-to-first-valid {:median (median ttfv)
                             :n-tasks-with-valid (count ttfv)
                             :n-tasks (count per-task)}
     :drop-reasons (into (sorted-map) (frequencies (map :reason vs)))
     :totals (:totals meta-json)
     :wall-ms (:wall-ms meta-json)}))

;; ---------------------------------------------------------------------------
;; Printing
;; ---------------------------------------------------------------------------

(defn- print-report [per-task overall]
  (println (str "\n  " (pad "task" 24) (pad "kind" 10) (pad "n" 4)
                (pad "parse" 7) (pad "pass" 7) (pad "ev-mean" 10)
                (pad "ev-max" 10) "tok->valid"))
  (doseq [{:keys [task-id kind n parse-rate pass-rate log-ml-mean log-ml-max
                  tokens-to-first-valid]} per-task]
    (println (str "  " (pad task-id 24) (pad kind 10) (pad n 4)
                  (pad (pct parse-rate) 7) (pad (pct pass-rate) 7)
                  (pad (fx log-ml-mean) 10) (pad (fx log-ml-max) 10)
                  (or tokens-to-first-valid "--"))))
  (let [{:keys [n-rows n-verdicts parse-rate keep-rate keep-ci function-pass-rate
                function-pass-ci program-keep-rate evidence tokens-to-first-valid
                drop-reasons totals wall-ms]} overall]
    (println (str "\n  overall: " n-verdicts " verdicts / " n-rows " rows"
                  "  parse=" (pct parse-rate)
                  "  keep=" (pct keep-rate)
                  " CI[" (pct (:lo keep-ci)) ", " (pct (:hi keep-ci)) "]"))
    (println (str "  SCI pass (:function): " (pct function-pass-rate)
                  " CI[" (pct (:lo function-pass-ci)) ", " (pct (:hi function-pass-ci)) "]"
                  " (n=" (:n function-pass-ci) ")"
                  "   program keep: " (pct program-keep-rate)))
    (println (str "  evidence (kept programs, n=" (:n-kept evidence) "): mean="
                  (fx (:mean evidence)) " max=" (fx (:max evidence))))
    (println (str "  tokens-to-first-valid: median=" (fx (:median tokens-to-first-valid))
                  " over " (:n-tasks-with-valid tokens-to-first-valid) "/"
                  (:n-tasks tokens-to-first-valid) " tasks"))
    (println (str "  drop reasons: " (pr-str drop-reasons)))
    (when totals
      (println (str "  gen cost: " (:samples totals) " samples, "
                    (:completion-tokens totals) " tok, "
                    (fx (/ (or (:gen-ms totals) 0) 1000.0)) "s gen, "
                    (:errors totals) " errors; wall " (fx (/ (or wall-ms 0) 1000.0)) "s")))))

;; ---------------------------------------------------------------------------
;; COMPARE mode
;; ---------------------------------------------------------------------------

(defn- read-json [p]
  (js->clj (js/JSON.parse (.readFileSync fs p "utf8")) :keywordize-keys true))

(def ^:private compare-metrics
  [["function-pass-rate" #(get-in % [:overall :function-pass-rate]) pct]
   ["keep-rate"          #(get-in % [:overall :keep-rate]) pct]
   ["parse-rate"         #(get-in % [:overall :parse-rate]) pct]
   ["evidence-mean"      #(get-in % [:overall :evidence :mean]) fx]
   ["evidence-max"       #(get-in % [:overall :evidence :max]) fx]
   ["tok->valid median"  #(get-in % [:overall :tokens-to-first-valid :median]) fx]
   ["completion tokens"  #(get-in % [:overall :totals :completion-tokens]) str]
   ["gen-ms total"       #(get-in % [:overall :totals :gen-ms]) str]])

(defn- run-compare! [spec]
  (let [[a b] (map str/trim (str/split spec #","))
        sa (read-json (str out-dir "/score-" a ".json"))
        sb (read-json (str out-dir "/score-" b ".json"))
        rows (for [[label f _] compare-metrics
                   :let [va (f sa) vb (f sb)]]
               {:metric label :a va :b vb
                :delta (when (and (number? va) (number? vb)) (- vb va))})]
    (println (str "\n== t1 compare: " a " vs " b " =="))
    (println (str "  " (pad "metric" 20) (pad a 12) (pad b 12) (str "delta(" b "-" a ")")))
    (doseq [[[label _ fmt] {:keys [a b delta]}] (map vector compare-metrics rows)]
      (println (str "  " (pad label 20) (pad (fmt a) 12) (pad (fmt b) 12)
                    (if (number? delta) (fx delta) "--"))))
    (bu/write-json out-dir (str "compare-" a "-vs-" b ".json")
                   {:a {:arm a :score sa} :b {:arm b :score sb}
                    :delta (vec rows)})))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(defn- run-score! []
  (when-not (.existsSync fs in-file)
    (println (str "no gen file: " in-file " (run scripts/t1_bakeoff.cljs first)"))
    (js/process.exit 1))
  (let [rows      (sandbox/read-candidates in-file)
        meta-path (str out-dir "/gen-" arm "-meta.json")
        meta-json (when (.existsSync fs meta-path) (read-json meta-path))
        unknown   (->> rows (map :task_id) distinct
                       (remove battery/tasks-by-id) vec)]
    (when (seq unknown)
      (println "WARN: rows with task ids unknown to the t1 battery (dropped):" unknown))
    (println (str "== t1_score arm=" arm " =="))
    (println (str "  in : " in-file " (" (count rows) " rows)"))
    (println (str "  oracle: sandboxed evaluate-candidate, np=" np
                  " timeout=" timeout-ms "ms"))
    (p/let [verdicts (sandbox/collect-verdicts
                      in-file
                      {:out-path   (str out-dir "/score-" arm "-verdicts.edn")
                       :eval-opts  {:n-particles np}
                       :battery    "t1"
                       :timeout-ms timeout-ms
                       :verbose?   true})]
      (let [vmap     (into {} (map (fn [v] [[(:task-id v) (:sample-idx v)] v])) verdicts)
            by-task  (group-by :task_id rows)
            ;; battery order, restricted to tasks present in the gen file
            tids     (filter by-task (map :id battery/tasks))
            per-task (mapv #(task-report % (by-task %) vmap) tids)
            overall  (overall-report rows vmap meta-json per-task)]
        (print-report per-task overall)
        (bu/write-json out-dir (str "score-" arm ".json")
                       {:arm arm :in in-file :np np :seed seed :boot boot
                        :timeout-ms timeout-ms
                        :config (dissoc meta-json :totals :wall-ms)
                        :overall overall
                        :per-task per-task})))))

(defn -main []
  (cond
    compare-spec (p/resolved (run-compare! compare-spec))
    arm          (run-score!)
    :else        (do (println "usage: ARM=<label> (or COMPARE=a,b) bunx --bun nbb@1.4.208 scripts/t1_score.cljs")
                     (js/process.exit 1))))

(-> (-main)
    (p/catch (fn [e]
               (println "UNCAUGHT:" (.-message e))
               (println (.-stack e))
               (set! (.-exitCode js/process) 1))))
