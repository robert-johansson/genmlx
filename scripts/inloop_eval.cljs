(ns inloop-eval
  "Phase 3 (genmlx-oexl), done-means #4: the IN-LOOP eval. Run a policy model (via its
   out-of-process worker at SERVER_URL) over the curriculum's HELD-OUT eval split, with a
   one-shot best-of-K baseline, and report the cost/quality frontier — SEPARATELY for the
   two eval cohorts (within-family same-distribution vs the held-out-family
   compositional/OOD), the rrps leakage lesson made operational.

   This is NOT sft_eval.cljs (whole-program best-of-k). Here the SAME loop the harvest used
   (genmlx.world.harvest/loop-proposer + harvest-task) is driven by whichever model's worker
   is up, so the comparison is: SFT(+GRPO)-0.8B-IN-LOOP vs 35B-IN-LOOP vs one-shot. Run once
   per tier (point SERVER_URL at that tier's worker, set TIER); a per-tier report is written
   and the two are compared by reading both.

   IMPORTANT: pass the SAME ROUND + INSTANCES the harvest used, so the eval split is
   byte-identical to the one the corpus held out (no leakage).

   Run:  SERVER_URL=http://127.0.0.1:8765 TIER=sft-0.8b ROUND=0 INSTANCES=12 \\
          bun run --bun nbb scripts/inloop_eval.cljs
   Env:  ROUND INSTANCES FAMILIES TASKS_LIMIT COHORT(within|family|all) SERVER_URL TIER
         K_ONESHOT K_STEP MAX_STEPS REVISE TEMP MAX_TOKENS NP SEED OUT."
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

(def round      (envi "ROUND" 0))
(def instances  (envi "INSTANCES" 12))
(def families   (when-let [s (env "FAMILIES" nil)] (mapv keyword (str/split s #","))))
(def tasks-limit (envi "TASKS_LIMIT" 0))
(def cohort-sel (env "COHORT" "all"))            ;; within | family | all
(def server-url (env "SERVER_URL" "http://127.0.0.1:8765"))
(def tier       (env "TIER" "unknown"))
(def k-step     (envi "K_STEP" 4))
(def max-steps  (envi "MAX_STEPS" 8))
(def revise     (envi "REVISE" 2))
;; COMPUTE_MATCHED=1 (R0, genmlx-2uzj): the one-shot+rerank baseline arm spends the loop's
;; LLM sample budget (K_STEP x MAX_STEPS x (1+REVISE)) so the comparison is fair, not a
;; K=4-vs-loop strawman. The permanent baseline every LLM phase must beat.
(def k-oneshot  (if (= "1" (env "COMPUTE_MATCHED" "0"))
                  (* k-step max-steps (inc revise))
                  (envi "K_ONESHOT" 16)))
(def temp       (envf "TEMP" 0.7))
(def max-toks   (envi "MAX_TOKENS" 320))
(def np         (envi "NP" 2000))
(def seed       (envi "SEED" 1))
(def out-dir    (env "OUT" (home "genmlx-loop-artifacts" "eval")))
;; SYSTEM = a file path whose contents override the proposer system prompt (e.g. the SFT
;; student's short_system.txt, to keep train==inference); "default"/unset -> lp/default-system.
(def system-prompt
  (when-let [p (env "SYSTEM" nil)]
    (when (and (not= p "default") (.existsSync fs p)) (str/trim (.readFileSync fs p "utf8")))))

(def stats (atom {:calls 0 :samples 0 :gen-time 0.0 :completion-tokens 0 :errors 0}))
(defn counting-call [req]
  (let [resp (lp/call-server server-url req)]
    (swap! stats #(-> % (update :calls inc) (update :samples + (count (:completions resp)))
                      (update :gen-time + (or (:gen_time_s resp) 0))
                      (update :completion-tokens + (or (:completion_tokens resp) 0))
                      (update :errors + (if (:error resp) 1 0))))
    resp))

(defn- best-scored
  "Best-evidence valid candidate among [{:code}] (the one-shot baseline scorer)."
  [cands obs]
  (let [scored (for [c cands :let [fb (syn/check (:code c) obs {:n-particles np})]
                     :when (syn/scored? fb)]
                 {:code (:code c) :evidence (:evidence fb)})]
    (when (seq scored) (apply max-key :evidence scored))))

(defn- run-oneshot [{:keys [task-desc observations]}]
  (let [cands (lp/one-shot-candidates (cond-> {:call-llm counting-call :task-desc task-desc
                                               :observations observations :k k-oneshot
                                               :temperature 0.8 :max-tokens max-toks :seed seed}
                                        system-prompt (assoc :system system-prompt)))]
    (:evidence (best-scored cands observations))))

(defn- run-loop [task]
  (let [prop (h/loop-proposer (cond-> {:call-llm counting-call :task-desc (:task-desc task)
                                       :observations (:observations task) :k k-step :temperature temp
                                       :max-tokens max-toks :revise revise :n-particles np :seed seed}
                                system-prompt (assoc :system system-prompt)))
        run  (h/harvest-task task {:propose prop :init-spec (cur/crude-spec (:observations task))
                                   :strategy :greedy :max-steps max-steps :n-particles np})]
    (:final run)))

(defn- solved? [bar e] (boolean (and e (js/isFinite e) (>= e bar))))

;; ---------------------------------------------------------------------------
(def C (cur/generate-curriculum (cond-> {:round round :instances-per-family instances}
                                  families (assoc :families families))))
(def evals0 (:eval-tasks C))
(def evals1 (if (= cohort-sel "all") evals0 (filterv #(= (keyword cohort-sel) (:cohort %)) evals0)))
(def evals  (if (pos? tasks-limit) (vec (take tasks-limit evals1)) evals1))

(.mkdirSync fs out-dir #js {:recursive true})
(println (str "\n### IN-LOOP EVAL  tier=" tier "  url=" server-url))
(println (str "  curriculum round " round ", " instances "/family -> " (count evals0)
              " eval tasks; cohort=" cohort-sel " -> " (count evals) " evaluated"
              (when (pos? tasks-limit) (str " (capped " (count evals) ")"))))
(println (str "  arms: one-shot best-of-" k-oneshot "  |  greedy loop (K_STEP=" k-step
              " MAX_STEPS=" max-steps " REVISE=" revise ")"))

(def t0 (js/Date.now))
(def results
  (vec (map-indexed
        (fn [idx t]
          (let [bar (:solve-bar t)
                os1 (run-oneshot t)
                lp1 (run-loop t)
                row {:id (name (:id t)) :family (name (:family t)) :cohort (name (:cohort t))
                     :complexity (:complexity t) :solve-bar bar
                     :oneshot os1 :oneshot-solved (solved? bar os1)
                     :loop lp1 :loop-solved (solved? bar lp1)
                     :loop-beats-oneshot (boolean (and lp1 (js/isFinite lp1)
                                                       (or (nil? os1) (not (js/isFinite os1))
                                                           (> lp1 (+ os1 0.1)))))}]
            (println (str "  [" (inc idx) "/" (count evals) "] " (:id row)
                          " (" (:family row) "/" (:cohort row) " c" (:complexity row) ")"
                          "  one-shot=" (fx os1) (if (:oneshot-solved row) "*" "")
                          "  loop=" (fx lp1) (if (:loop-solved row) "*" "")
                          "  loop>1shot=" (:loop-beats-oneshot row)))
            row))
        evals)))
(def wall (/ (- (js/Date.now) t0) 1000.0))

(defn- cohort-summary [rows label]
  (when (seq rows)
    (let [n (count rows)
          sr (fn [k] (/ (count (filter k rows)) (double n)))
          me (fn [k] (let [es (keep k rows)] (when (seq es) (/ (reduce + es) (count es)))))]
      {:cohort label :n n
       :oneshot-solve-rate (sr :oneshot-solved) :loop-solve-rate (sr :loop-solved)
       :oneshot-mean-evidence (me :oneshot) :loop-mean-evidence (me :loop)
       :loop-beats-oneshot-rate (sr :loop-beats-oneshot)})))

(def by-cohort (->> [["within" (filter #(= "within" (:cohort %)) results)]
                     ["family" (filter #(= "family" (:cohort %)) results)]
                     ["ALL"    results]]
                    (keep (fn [[lbl rows]] (cohort-summary rows lbl)))
                    vec))

(println "\n### COHORT SUMMARY  (within-family vs held-out-family REPORTED SEPARATELY)")
(doseq [s by-cohort]
  (println (str "  " (.padEnd (:cohort s) 8) " n=" (:n s)
                "  loop-solve=" (fx (* 100 (:loop-solve-rate s)) ) "%"
                "  one-shot-solve=" (fx (* 100 (:oneshot-solve-rate s)) ) "%"
                "  loop-mean=" (fx (:loop-mean-evidence s))
                "  one-shot-mean=" (fx (:oneshot-mean-evidence s))
                "  loop>1shot=" (fx (* 100 (:loop-beats-oneshot-rate s)) ) "%")))

(def st @stats)
(println (str "\n  COST (tier=" tier "): " (:calls st) " calls, " (:samples st) " samples, "
              (fx (:gen-time st)) "s gen, " (:completion-tokens st) " completion tokens"
              (when (pos? (:errors st)) (str ", " (:errors st) " transport errors"))))
(println (str "  wall " (fx wall) "s"))

(def report {:tier tier :round round :instances instances
             :config {:k-oneshot k-oneshot :k-step k-step :max-steps max-steps :revise revise
                      :temp temp :np np}
             :n-eval (count evals) :by-cohort by-cohort :results results
             :cost st :wall-s wall})
(def outfile (.join path out-dir (str "inloop_eval_" tier ".json")))
(.writeFileSync fs outfile (js/JSON.stringify (clj->js report) nil 2))
(println (str "  wrote " outfile))
