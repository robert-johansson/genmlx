(ns synth-llm-probe
  "EXPERIMENT B — WITH A REAL LLM IN THE LOOP (beans genmlx-0yv7 / genmlx-1pan): the
   north-star's load-bearing empirical test.

   The structured-proposer experiments (synth_repl_probe / synth_search_probe) PROVED the
   scaffold — the four-level check, the edit ops, the greedy driver, the particle search.
   They did NOT prove the THESIS, because they never put a real LLM in the loop. This
   probe does: a real policy LLM is the proposer, conditioned each step on the verifier's
   feedback, and we ask the load-bearing question directly —

     does the LLM-in-the-loop, conditioned on feedback, beat one-shot best-of-K?

   THREE ARMS PER TASK, SAME MODEL, SAME DSL PROMPT, SAME ORACLE (the only difference is
   the feedback loop):
     1. ONE-SHOT best-of-K   — K full programs, NO feedback. The control (the cliff:
                                whole-program best-of-16 got 0/16 on these models).
     2. GREEDY loop          — genmlx.world.synth with the LLM proposer (beam-width 1).
     3. BEAM   loop          — genmlx.world.search with the LLM proposer (population).

   PHASE-2 ROBUSTNESS: greedy vs beam over SEEDS seeds — population search's real
   justification is robustness to a NOISY/stochastic proposer, which is exactly the LLM
   regime (each propose is temperature-sampled). We report per-strategy mean evidence +
   solve-rate over seeds.

   COST: the resident worker reports per-call gen-time + tokens; we accumulate them, so
   each tier's (solve-rate, cost) is the frontier the Phase-4 fast/slow metareasoner
   allocates over.

   The 3 tasks are the EXACT-scoreable advanced models whole-program best-of-16 got 0/16
   on (byte-identical data to synth_repl_probe / particle_advanced_probe), so the
   comparison is apples-to-apples.

   Requires a running worker (scripts/llm_server.py). Run via scripts/run_llm_probe.sh
   (which launches the worker, waits for READY, runs this, tears down). Env:
     SERVER_URL (default http://127.0.0.1:8765), TIER (output label), TASKS (csv subset),
     K_ONESHOT, K_STEP, BEAM_WIDTH, MAX_STEPS, SEEDS, NP (IS particles), TEMP."
  (:require [genmlx.world.llm-proposer :as lp]
            [genmlx.world.synth :as syn]
            [genmlx.world.search :as se]
            [genmlx.world.harvest :as h]
            [clojure.string :as str]))

(def os   (js/require "os"))
(def path (js/require "path"))
(def fs   (js/require "fs"))
(defn home [& xs] (apply (.-join path) (.homedir os) xs))
(def out-dir (home "genmlx-loop-artifacts" "particle"))
(defn fx [x] (when (and x (js/isFinite x)) (.toFixed (js/Number x) 2)))
(defn- env [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))
(defn- envf [k d] (let [v (env k nil)] (if v (js/parseFloat v) d)))

(def server-url (env "SERVER_URL" "http://127.0.0.1:8765"))
(def tier       (env "TIER" "unknown"))
(def k-oneshot  (envi "K_ONESHOT" 16))
(def k-step     (envi "K_STEP" 4))
(def beam-width (envi "BEAM_WIDTH" 4))
(def max-steps  (envi "MAX_STEPS" 6))
(def n-seeds    (envi "SEEDS" 3))
(def temp       (envf "TEMP" 0.7))
(def max-toks   (envi "MAX_TOKENS" 320))
(def revise     (envi "REVISE" 2))

;; ---------------------------------------------------------------------------
;; Cost-counting wrapper around the out-of-process bridge.
;; ---------------------------------------------------------------------------
(def stats (atom {:calls 0 :samples 0 :gen-time 0.0 :prompt-tokens 0 :completion-tokens 0 :errors 0}))
(defn counting-call [req]
  (let [resp (lp/call-server server-url req)]
    ;; NB the worker's JSON keys keywordize with UNDERSCORES (gen_time_s, completion_tokens).
    (swap! stats #(-> %
                      (update :calls inc)
                      (update :samples + (count (:completions resp)))
                      (update :gen-time + (or (:gen_time_s resp) 0))
                      (update :prompt-tokens + (or (:prompt_tokens resp) 0))
                      (update :completion-tokens + (or (:completion_tokens resp) 0))
                      (update :errors + (if (:error resp) 1 0))))
    (when (:error resp) (println "    [llm error]" (:error resp)))
    resp))

;; ===========================================================================
;; Tasks — the 3 exact-scoreable advanced models (byte-identical to synth_repl_probe).
;; :task-desc gives the SEMANTICS (index/time/group), NOT the structure (slope/AR/pool).
;; :solve-bar ≈ HALFWAY from the structureless crude baseline to the data-warranted
;; optimum (gold, from the structured probe). It cleanly separates "found the structure"
;; (a fixed-σ=1 structural model already clears it) from "structureless" (crude cannot),
;; so passing it means the LLM actually built the right structure. The loop typically
;; climbs WELL above the bar toward gold by refining the (fixed) noise scale — that extra
;; is the loop's iterative-refinement value over a one-shot structural model at σ=1.
;; ===========================================================================

(def tasks
  [{:id :linreg
    :obs {:y0 1.1 :y1 2.0 :y2 2.7 :y3 4.2 :y4 4.8 :y5 6.1 :y6 6.9}
    :task-desc (str "You have 7 observations y0..y6. Observation yj is the measured response at "
                    "input x = j (so x = 0,1,2,3,4,5,6). Model how the response y depends on the input x.")
    :np 0 :solve-bar -12.2 :crude -17.5 :gold -6.90}
   {:id :kalman
    :obs {:y0 0.4 :y1 0.9 :y2 1.3 :y3 1.1 :y4 0.6}
    :task-desc (str "You have a time series of 5 observations y0..y4 recorded at successive times "
                    "t = 0,1,2,3,4. The underlying signal varies smoothly over time. Model the series.")
    :np 0 :solve-bar -6.0 :crude -7.0 :gold -4.92}
   {:id :hier
    :obs {:a0 5.2 :a1 4.8 :a2 5.5 :b0 -1.1 :b1 -0.7 :b2 -1.4 :c0 2.1 :c1 2.6 :c2 1.9}
    :task-desc (str "You have 9 observations in three groups: a0,a1,a2 (group A); b0,b1,b2 (group B); "
                    "c0,c1,c2 (group C). The groups differ from one another. Model the group structure.")
    :np 0 :solve-bar -18.0 :crude -24.0 :gold -12.36}
   ;; HARDER (genmlx-0yv7 follow-up): varying-slopes — 3 groups, each a DISTINCT line in x.
   ;; The data-warranted model needs 6 latents (slope+intercept per group) + 3 linear means
   ;; + a tuned noise — far more slip surface than the single-idea models, so one-shot
   ;; best-of-K must get ALL of it right in one program. Exactly scoreable (linear-Gaussian).
   {:id :vslope
    :obs {:a0 1.0 :a1 3.1 :a2 4.9 :a3 7.0 :b0 6.1 :b1 4.9 :b2 4.0 :b3 3.0 :c0 0.0 :c1 0.6 :c2 1.0 :c3 1.4}
    :task-desc (str "You have 12 observations in three groups a, b, c. Each group has 4 measurements "
                    "g0,g1,g2,g3 taken at inputs x = 0,1,2,3 (so a0 is group a at x=0, a3 is group a at "
                    "x=3, etc.). Within each group the response changes with x. Model how y depends on x "
                    "in each of the three groups.")
    :np 0 :solve-bar -23.5 :crude -30.0 :gold -17.0}])

(defn- only-tasks []
  (if-let [sel (env "TASKS" nil)]
    (let [want (set (map keyword (str/split sel #",")))]
      (filterv #(want (:id %)) tasks))
    tasks))

;; ===========================================================================
;; Arms.
;; ===========================================================================

(defn- best-scored
  "Among candidates [{:code ...}], check each and return the highest-evidence valid one
   (exact methods are reproducible; for these exact tasks all valid candidates are exact)."
  [cands obs np]
  (let [scored (for [c cands
                     :let [fb (syn/check (:code c) obs {:n-particles np})]
                     :when (syn/scored? fb)]
                 {:code (:code c) :evidence (:evidence fb) :method (:method fb)})]
    {:n (count cands) :n-valid (count scored)
     :best (when (seq scored) (apply max-key :evidence scored))}))

(defn run-oneshot [{:keys [task-desc obs np]}]
  (let [cands (lp/one-shot-candidates {:call-llm counting-call :task-desc task-desc
                                       :observations obs :k k-oneshot :temperature 0.8 :max-tokens max-toks :seed 1})
        r     (best-scored cands obs np)]
    {:arm :oneshot :n-extracted (:n r) :n-valid (:n-valid r)
     :evidence (get-in r [:best :evidence]) :code (get-in r [:best :code])}))

(defn- init-spec
  "A crude covering model the loop starts from: one shared mean, every observed key an
   obs site. Identical role to the structured probe's crude rung."
  [obs]
  (syn/spec [(syn/latent 'mu "gaussian" [0 10])]
            (for [k (keys obs)] (syn/obs k "gaussian" ['mu 3.0]))))

;; The loop proposer = the SHARED LLM ∪ σ-grid proposer (genmlx.world.harvest/loop-proposer):
;; the LLM for the hard structural moves, the deterministic shared-σ grid for the nuisance
;; scale (the resource-rational split; oracle-gated per step). The Phase-3 harvest reuses
;; this SAME proposer, so the SFT corpus reflects exactly the loop this probe validates.
(defn- loop-proposer [task-desc obs np seed]
  (h/loop-proposer {:call-llm counting-call :task-desc task-desc :observations obs
                    :k k-step :temperature temp :max-tokens max-toks
                    :revise revise :n-particles (if (pos? np) np 2000) :seed seed}))

(defn run-greedy [{:keys [task-desc obs np]} seed]
  (let [prop (loop-proposer task-desc obs np seed)
        res  (syn/synthesize {:init-spec (init-spec obs) :observations obs :propose prop
                              :max-steps max-steps :plateau-eps 0.05 :n-particles (if (pos? np) np 2000)})]
    {:arm :greedy :seed seed :evidence (:evidence (:feedback res))
     :stop-reason (:stop-reason res) :steps (:steps res)
     :code (get-in res [:feedback :code])
     :trajectory (mapv #(select-keys % [:step :edit :evidence :delta :method :code]) (:trajectory res))}))

(defn run-beam [{:keys [task-desc obs np]} seed]
  (let [prop (loop-proposer task-desc obs np seed)
        res  (se/search {:init-spec (init-spec obs) :observations obs :propose prop
                         :beam-width beam-width :adaptive? true :strategy :beam
                         :max-steps max-steps :plateau-eps 0.05 :n-particles (if (pos? np) np 2000) :seed seed})]
    {:arm :beam :seed seed :evidence (:evidence (:best res))
     :stop-reason (:stop-reason res) :steps (:steps res)
     :code (:code (:best res))
     :trajectory (mapv #(select-keys % [:step :edit :evidence :delta :method :code]) (:trajectory (:best res)))}))

(defn- solved? [bar evidence] (boolean (and evidence (js/isFinite evidence) (>= evidence bar))))

;; ===========================================================================
;; Run one task: one-shot control + greedy/beam over seeds.
;; ===========================================================================

(defn run-task [{:keys [id solve-bar crude gold] :as t}]
  (println (str "\n================  " (name id) "  (tier=" tier ")  ================"))
  (println (str "  solve-bar " solve-bar "  (structureless crude ~" crude
                "; structured-proposer gold ~" gold ")"))
  (let [_ (println "  [one-shot best-of-" k-oneshot " — the control]")
        os1 (run-oneshot t)
        _ (println (str "    extracted " (:n-extracted os1) "/" k-oneshot " parseable, "
                        (:n-valid os1) " valid; best evidence " (fx (:evidence os1))
                        "  SOLVED=" (solved? solve-bar (:evidence os1))))
        seeds (range 1 (inc n-seeds))
        greedy (doall (for [s seeds]
                        (let [r (run-greedy t s)]
                          (println (str "  [greedy seed " s "] -> " (fx (:evidence r))
                                        "  steps=" (:steps r) " stop=" (name (:stop-reason r))
                                        "  SOLVED=" (solved? solve-bar (:evidence r))))
                          r)))
        beam (doall (for [s seeds]
                      (let [r (run-beam t s)]
                        (println (str "  [beam   seed " s "] -> " (fx (:evidence r))
                                      "  steps=" (:steps r) " stop=" (name (:stop-reason r))
                                      "  SOLVED=" (solved? solve-bar (:evidence r))))
                        r)))
        ev    (fn [rs] (keep :evidence rs))
        mean  (fn [xs] (when (seq xs) (/ (reduce + xs) (count xs))))
        rate  (fn [rs] (/ (count (filter #(solved? solve-bar (:evidence %)) rs)) (max 1 (count rs))))]
    {:id id :solve-bar solve-bar :crude crude :gold gold
     :oneshot os1 :oneshot-solved (solved? solve-bar (:evidence os1))
     :greedy greedy :beam beam
     :greedy-mean (mean (ev greedy)) :beam-mean (mean (ev beam))
     :greedy-best (when (seq (ev greedy)) (apply max (ev greedy)))
     :beam-best (when (seq (ev beam)) (apply max (ev beam)))
     :greedy-solve-rate (rate greedy) :beam-solve-rate (rate beam)}))

;; ===========================================================================

(println (str "\n###  EXPERIMENT B WITH A REAL LLM  (tier=" tier ", url=" server-url ")  ###"))
(println (str "  K_ONESHOT=" k-oneshot " K_STEP=" k-step " BEAM_WIDTH=" beam-width
              " MAX_STEPS=" max-steps " SEEDS=" n-seeds " TEMP=" temp))
(let [t0 (js/Date.now)
      results (mapv run-task (only-tasks))
      wall (/ (- (js/Date.now) t0) 1000.0)
      st @stats]
  (println "\n================  VERDICT (tier=" tier ")  ================")
  (println (str (.padEnd "model" 10) (.padEnd "one-shot" 11) (.padEnd "greedy(best)" 14)
                (.padEnd "beam(best)" 12) (.padEnd "g-rate" 9) (.padEnd "b-rate" 9) "loop>one-shot?"))
  (doseq [r results]
    (let [os1 (get-in r [:oneshot :evidence])
          gb  (:greedy-best r) bb (:beam-best r)
          loopbest (apply max (keep identity [(or gb ##-Inf) (or bb ##-Inf)]))
          beats (boolean (and (js/isFinite loopbest)
                              (or (not (and os1 (js/isFinite os1)))
                                  (> loopbest (+ os1 0.1)))))]
      (println (str (.padEnd (name (:id r)) 10)
                    (.padEnd (str (fx os1) (if (:oneshot-solved r) "*" "")) 11)
                    (.padEnd (str (fx gb)) 14)
                    (.padEnd (str (fx bb)) 12)
                    (.padEnd (str (.toFixed (* 100 (:greedy-solve-rate r)) 0) "%") 9)
                    (.padEnd (str (.toFixed (* 100 (:beam-solve-rate r)) 0) "%") 9)
                    (str beats)))))
  (let [n-loop-solve (count (filter #(or (pos? (:greedy-solve-rate %)) (pos? (:beam-solve-rate %))) results))
        n-os-solve   (count (filter :oneshot-solved results))
        beam>=greedy (every? #(>= (or (:beam-best %) ##-Inf) (- (or (:greedy-best %) ##-Inf) 0.1)) results)]
    (println (str "\n  THESIS: the LLM-in-the-loop solves " n-loop-solve "/" (count results)
                  " advanced models; one-shot best-of-" k-oneshot " solves " n-os-solve "/" (count results) "."))
    (println (str "  PHASE-2 robustness: beam >= greedy (best evidence) on every model: " beam>=greedy
                  "  (compare per-seed solve-rates above for the stochastic-regime claim)."))
    (println (str "\n  COST (tier=" tier "): " (:calls st) " LLM calls, " (:samples st) " samples, "
                  (fx (:gen-time st)) "s generation, " (:completion-tokens st) " completion tokens"
                  (when (pos? (:errors st)) (str ", " (:errors st) " transport errors")) "."))
    (println (str "  wall-clock " (fx wall) "s; " (fx (/ (:gen-time st) (max 1 (:samples st)))) "s/sample avg."))
    (when-not (.existsSync fs out-dir) (.mkdirSync fs out-dir #js {:recursive true}))
    (let [outfile (home "genmlx-loop-artifacts" "particle" (str "synth_llm_probe_" tier ".json"))]
      (.writeFileSync fs outfile
                      (js/JSON.stringify (clj->js {:tier tier :config {:k-oneshot k-oneshot :k-step k-step
                                                                       :beam-width beam-width :max-steps max-steps
                                                                       :seeds n-seeds :temp temp}
                                                   :results results :cost st :wall-s wall
                                                   :n-loop-solve n-loop-solve :n-oneshot-solve n-os-solve}) nil 2))
      (println (str "  wrote " outfile)))))
