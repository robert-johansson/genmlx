(ns curriculum-probe
  "RUN + inspect the REPL-synthesis CURRICULUM (genmlx.world.curriculum, bean genmlx-ilna).

   Generates a graded, oracle-grounded, leakage-safe curriculum of structured-model
   tasks, writes it as a JSON artifact (out-of-tree), and proves end-to-end that it is
   consumable by BOTH downstream consumers:
     - the run loop  (scripts/synth_llm_probe.cljs)  via cur/->probe-task
     - the harvester (genmlx.world.repl-corpus)       via a no-LLM family-proposer loop

   It reports the COMPLEXITY SPREAD (the resource-rational training signal: gap/n-latents
   by complexity band) and the two eval cohorts (within-family same-distribution vs the
   held-out family compositional/OOD).

   Run: bun run --bun nbb scripts/curriculum_probe.cljs
   Env: INSTANCES (per family, default 25), ROUND (default 0)."
  (:require [genmlx.world.curriculum :as cur]
            [genmlx.world.synth :as syn]
            [genmlx.world.repl-corpus :as rc]
            [clojure.string :as str]))

(def os   (js/require "os"))
(def path (js/require "path"))
(def fs   (js/require "fs"))
(defn home [& xs] (apply (.-join path) (.homedir os) xs))
(def out-dir (home "genmlx-loop-artifacts" "curriculum"))
(defn- env [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))
(defn fx [x] (if (and x (js/isFinite x)) (.toFixed (js/Number x) 2) (str x)))

(def instances (envi "INSTANCES" 25))
(def round     (envi "ROUND" 0))

(println (str "\n=== curriculum RUN  (round " round ", " instances " instances/family) ===\n"))
(def C (cur/generate-curriculum {:round round :instances-per-family instances}))
(def tasks (:tasks C))

;; ---------------------------------------------------------------------------
;; Report.
;; ---------------------------------------------------------------------------
(let [s (:summary C)]
  (println "SUMMARY")
  (println "  tasks:" (:n-tasks s) " train:" (:n-train s) " eval:" (:n-eval s)
           " (within-family:" (:eval-within s) " held-out-family:" (:eval-family s) ")")
  (println "  held-out families:" (:held-out-families s) "  total drops:" (:total-drops s)))

(println "\nBY FAMILY")
(doseq [[fam {:keys [n drops complexity]}] (:by-family C)]
  (println (str "  " (name fam)) "  c=" complexity "  n=" n "  drops=" drops))

(println "\nBY COMPLEXITY  (the resource-rational spread)")
(doseq [[c {:keys [n mean-gap mean-struct-gap mean-n-latents]}] (:by-complexity C)]
  (println (str "  complexity " c ":  n=" n
                "  mean-gap=" (fx mean-gap) " nats"
                "  mean-struct-gap=" (fx mean-struct-gap)
                "  mean-n-latents=" (fx mean-n-latents))))
(let [bc (:by-complexity C) cs (sort (keys bc))]
  (println "  -> n-latents monotone:" (apply <= (map #(:mean-n-latents (bc %)) cs))
           " | hardest gap > easiest gap:"
           (> (:mean-gap (bc (last cs))) (:mean-gap (bc (first cs))))))

(println "\nEXAMPLE TASKS (one per family)")
(doseq [fam (map :family cur/family-defs)
        :let [t (first (filter #(= fam (:family %)) tasks))]
        :when t]
  (println (str "  [" (name fam) "] " (:id t) "  exact?=" (:exact? t)
                "  crude=" (fx (:crude t)) " gold=" (fx (:gold t))
                " bar=" (fx (:solve-bar t)) " gap=" (fx (:gap t))))
  (println (str "      desc: " (:task-desc t))))

;; ---------------------------------------------------------------------------
;; End-to-end harvest proof (no LLM): run one task/family through the family-proposer
;; loop, harvest with repl-corpus using the curriculum's eval-ids, show leakage-safety.
;; ---------------------------------------------------------------------------
(println "\nEND-TO-END HARVEST (no LLM; family-proposer loop -> repl-corpus)")
(defn harvest-run [t]
  (let [obs (:observations t)
        res (syn/synthesize {:init-spec (cur/crude-spec obs) :observations obs
                             :propose (cur/family-proposer t) :max-steps 8})]
    {:task t :trajectory (:trajectory res)
     :steps (:steps res) :final (get-in res [:feedback :evidence])
     :solved? (>= (get-in res [:feedback :evidence] -1e9) (:solve-bar t))}))

(def sample-runs
  (vec (for [fam (map :family cur/family-defs)
             :let [t (first (filter #(= fam (:family %)) tasks))] :when t]
         (harvest-run t))))
(doseq [r sample-runs]
  (println (str "  [" (name (:family (:task r))) "] steps=" (:steps r)
                " final=" (fx (:final r)) " bar=" (fx (:solve-bar (:task r)))
                " solved=" (:solved? r))))
(println "  loop solves" (count (filter :solved? sample-runs)) "/" (count sample-runs)
         "sampled tasks via the no-LLM structured proposer")

(let [corpus (rc/build-corpus sample-runs {:eval-ids (:eval-task-ids C)})]
  (println "\nHARVEST CORPUS")
  (println "  rows:" (:n-rows corpus) " train-rows:" (count (:train-rows corpus))
           " dropped-eval-rows:" (count (:dropped-eval corpus)))
  (println "  train task ids:" (:train-task-ids corpus))
  (println "  leakage-safe:" (not-any? #(contains? (:eval-task-ids C) (:task-id %))
                                       (:train-rows corpus))))

;; ---------------------------------------------------------------------------
;; Write the JSON artifact (out-of-tree).
;; ---------------------------------------------------------------------------
(.mkdirSync fs out-dir #js {:recursive true})
(def artifact (.join path out-dir (str "curriculum-r" round ".json")))
(.writeFileSync fs artifact
  (.stringify js/JSON (clj->js {:summary (:summary C)
                                :by-family (:by-family C)
                                :by-complexity (:by-complexity C)
                                :eval-task-ids (vec (:eval-task-ids C))
                                :tasks tasks}) nil 2))
(println (str "\nwrote " (count tasks) " tasks -> " artifact))
