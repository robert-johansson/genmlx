(ns r0-demask
  "R0 (genmlx-2uzj): de-mask the metric + cost the oracle. NATIVE-FREE.

   The reported loop-solve rates UNION a deterministic sigma-grid floor (the noise-refiner,
   no LLM) into 'loop' credit -> they are not a proposer measurement. This script computes,
   for each held-out eval task, the SIGMA-GRID-ONLY floor (synthesize from crude with ONLY
   the noise-refiner, no LLM) and re-reports base/SFT/35B with that floor SUBTRACTED: the
   LLM-ONLY solve rate = solves on tasks the sigma-grid alone does NOT solve. Then it costs
   the exact oracle (wall-clock per check) so the loop's hundreds-of-calls/task budget is a
   measured term, not 'free'.

   Run: bun run --bun nbb scripts/r0_demask.cljs"
  (:require [genmlx.world.curriculum :as cur]
            [genmlx.world.harvest :as h]
            [genmlx.world.synth :as syn]
            [clojure.string :as str]))

(def fs (js/require "fs"))
(def os (js/require "os"))
(def path (js/require "path"))
(defn fx [x] (if (and x (js/isFinite x)) (.toFixed (js/Number x) 1) (str x)))
(defn pct [n d] (str (.toFixed (* 100.0 (/ n (max 1 d))) 0) "%"))

;; ---------------------------------------------------------------------------
;; 1. The sigma-grid-only floor per eval task (NO LLM): synthesize from crude with
;;    ONLY the noise-refiner. Does the best shared-sigma covering model cross the bar?
;; ---------------------------------------------------------------------------
(def C (cur/generate-curriculum {:round 0 :instances-per-family 12}))
(def eval-tasks (:eval-tasks C))

(defn sigma-grid-floor [task]
  (let [obs (:observations task)
        res (syn/synthesize {:init-spec (cur/crude-spec obs) :observations obs
                             :propose h/noise-refiner :max-steps 8 :plateau-eps 0.01})
        ev  (get-in res [:feedback :evidence])]
    {:id (name (:id task)) :family (name (:family task)) :cohort (name (:cohort task))
     :bar (:solve-bar task) :floor-ev ev
     :floor-solved (boolean (and ev (js/isFinite ev) (>= ev (:solve-bar task))))}))

(def floors (mapv sigma-grid-floor eval-tasks))
(def floor-by-id (into {} (map (juxt :id identity)) floors))
(def n-floor (count (filter :floor-solved floors)))

(println "\n### R0: de-masking — the sigma-grid-only floor (NO LLM) ###")
(println (str "  " (count eval-tasks) " held-out eval tasks; the sigma-grid alone (no LLM) solves "
              n-floor " of them:"))
(doseq [[fam fs'] (sort-by key (group-by :family floors))]
  (println (str "    " (.padEnd fam 18) (count (filter :floor-solved fs')) "/" (count fs')
                " sigma-grid-solvable (LLM contributes nothing here)")))

;; ---------------------------------------------------------------------------
;; 2. Re-report each model RAW vs LLM-ONLY (floor subtracted).
;; ---------------------------------------------------------------------------
(defn load-eval [tier]
  (let [p (.join path (.homedir os) "genmlx-loop-artifacts" "eval" (str "inloop_eval_" tier ".json"))]
    (when (.existsSync fs p) (js->clj (js/JSON.parse (.readFileSync fs p "utf8")) :keywordize-keys true))))

(defn demask [tier]
  (when-let [r (load-eval tier)]
    (let [rows (:results r)
          ;; LLM-attributable tasks = those the sigma-grid floor does NOT solve
          llm-rows (filter (fn [row] (not (:floor-solved (floor-by-id (:id row))))) rows)
          raw-loop (count (filter :loop-solved rows))
          raw-os   (count (filter :oneshot-solved rows))
          llm-loop (count (filter :loop-solved llm-rows))
          llm-os   (count (filter :oneshot-solved llm-rows))]
      {:tier tier :n (count rows) :n-llm (count llm-rows)
       :raw-loop raw-loop :raw-os raw-os :llm-loop llm-loop :llm-os llm-os})))

(def tiers ["base-0.8b" "sft-i100" "qwen35b"])
(def demasked (keep demask tiers))

(println "\n### RAW vs LLM-ONLY (sigma-grid floor subtracted) ###")
(println (str "  " (.padEnd "tier" 12) (.padEnd "raw loop" 12) (.padEnd "LLM-only loop" 16)
              (.padEnd "raw 1shot" 12) "LLM-only 1shot"))
(doseq [d demasked]
  (println (str "  " (.padEnd (:tier d) 12)
                (.padEnd (str (:raw-loop d) "/" (:n d) " " (pct (:raw-loop d) (:n d))) 12)
                (.padEnd (str (:llm-loop d) "/" (:n-llm d) " " (pct (:llm-loop d) (:n-llm d))) 16)
                (.padEnd (str (:raw-os d) "/" (:n d) " " (pct (:raw-os d) (:n d))) 12)
                (str (:llm-os d) "/" (:n-llm d) " " (pct (:llm-os d) (:n-llm d))))))
(println (str "\n  Reading: the sigma-grid floor solves " n-floor "/" (count eval-tasks)
              " tasks with ZERO LLM. The LLM-only columns are the real proposer measurement;"))
(println "  the loop-vs-one-shot comparison must be made on the LLM-only tasks, not the raw mix.")

;; ---------------------------------------------------------------------------
;; 3. Cost the oracle: wall-clock per exact check (the loop runs hundreds/task).
;; ---------------------------------------------------------------------------
(println "\n### Oracle cost (exact path) ###")
(let [t (first (filter #(= :varying-slopes (:family %)) eval-tasks))
      obs (:observations t)
      gp (:true-params t)
      groups (:groups gp)
      pp (:pp gp)
      model (syn/render
             (syn/spec (mapcat (fn [gn] [(syn/latent (symbol (str "s-" gn)) "gaussian" [0 3])
                                         (syn/latent (symbol (str "i-" gn)) "gaussian" [0 5])]) groups)
                       (for [gn groups i (range pp)]
                         (syn/obs (keyword (str gn i)) "gaussian"
                                  [(list 'mx/add (list 'mx/multiply (symbol (str "s-" gn)) (list 'mx/scalar i)) (symbol (str "i-" gn))) 1.0]))))
      _ (syn/check model obs {:n-particles 2000})
      n 200
      t0 (js/Date.now)
      _ (dotimes [_ n] (syn/check model obs {:n-particles 2000}))
      ms (/ (- (js/Date.now) t0) n)
      per-task (* 4 6 3)]
  (println (str "  exact check: " (fx ms) " ms/call (varying-slopes, " (count obs) " obs)"))
  (println (str "  a full loop task ~ K*steps*revise = " per-task " checks ~ "
                (fx (* per-task ms)) " ms of oracle alone (exact)."))
  (println "  Non-conjugate (GMM/hierarchical, R4.5/R2) uses 2000-particle IS -> far slower;")
  (println "  that is the speed argument for the RB-exact primitive (R4.5)."))
