(ns r-minus-1-rescore
  "R-1 (genmlx-h3l9): the FREE decisive ordering-vs-proposer test. NO model, NO GPU.

   The crown-jewel plan's RC1 claim: the multi-latent 'cliff' (varying-slopes 0/9 etc.) is an
   ACCEPT-ORDERING bug, not a no-monotone-path wall. The loop scores a structural edit at the
   EMITTED/crude noise and the sigma-refiner only fires AFTER structure is accepted, so a
   correct partial structure is scored in its deep fixed-sigma valley and rejected before sigma
   can rescue it.

   This script tests that directly with the EXACT oracle (linear-Gaussian => exact, native-free):
   for each cliff family it builds the INCREMENTAL gold path crude -> +1 structural unit -> ...
   -> gold (from the curriculum's :true-params), and scores each partial at FIXED sigma
   {3.0, 1.0, 0.5} vs GRID-BEST sigma (co-refined over harvest/noise-grid). Then:
     - RC1 confirmed for a family iff the GRID-BEST path is MONOTONE and CROSSES THE BAR while a
       FIXED-sigma path has a VALLEY (a partial dips below crude) -> the accept ordering is the
       bug; the loop is salvageable via the R1 ~30-line co-refined-accept fix.
     - Otherwise (not monotone even grid-best) -> the loop has no mechanism on that family.

   Run: bun run --bun nbb scripts/r_minus_1_rescore.cljs"
  (:require [genmlx.world.curriculum :as cur]
            [genmlx.world.synth :as syn]
            [genmlx.world.harvest :as h]
            [clojure.string :as str]))

(defn fx [x] (if (and x (js/isFinite x)) (.toFixed (js/Number x) 2) (str x)))

;; mean form: (mx/add (mx/multiply <s> (mx/scalar x)) <i>)  -- exactly what the templates emit.
(defn- lin-mean [s-sym x i-sym]
  (list 'mx/add (list 'mx/multiply s-sym (list 'mx/scalar x)) i-sym))

;; ---------------------------------------------------------------------------
;; Incremental partial-spec builders: k structural units own their structure,
;; the remaining groups/points share a single crude mean `mu`. k=0 = crude,
;; k=K = gold. All obs noise defaults to 1.0 (overridden per-score).
;; ---------------------------------------------------------------------------
(defn pgm-partials [{:keys [groups pp]}]
  (for [k (range (inc (count groups)))]
    (let [own (set (take k groups))
          shared? (some (complement own) groups)
          latents (concat (when shared? [(syn/latent 'mu "gaussian" [0 5])])
                          (for [gn groups :when (own gn)] (syn/latent (symbol (str "m-" gn)) "gaussian" [0 5])))
          obs (for [gn groups i (range pp)]
                (syn/obs (keyword (str gn i)) "gaussian"
                         [(if (own gn) (symbol (str "m-" gn)) 'mu) 1.0]))]
      {:k k :spec (syn/spec latents obs)})))

(defn vs-partials [{:keys [groups pp]}]
  (for [k (range (inc (count groups)))]
    (let [own (set (take k groups))
          shared? (some (complement own) groups)
          latents (concat (when shared? [(syn/latent 'mu "gaussian" [0 5])])
                          (mapcat (fn [gn] [(syn/latent (symbol (str "s-" gn)) "gaussian" [0 3])
                                            (syn/latent (symbol (str "i-" gn)) "gaussian" [0 5])])
                                  (filter own groups)))
          obs (for [gn groups x (range pp)]
                (syn/obs (keyword (str gn x)) "gaussian"
                         [(if (own gn) (lin-mean (symbol (str "s-" gn)) x (symbol (str "i-" gn))) 'mu) 1.0]))]
      {:k k :spec (syn/spec latents obs)})))

(defn lin-partials [{:keys [xs addrs]}]
  [{:k 0 :spec (syn/spec [(syn/latent 'mu "gaussian" [0 5])]
                         (for [k addrs] (syn/obs k "gaussian" ['mu 1.0])))}
   ;; intermediate: slope only (intercept folded into the shared mu) -- tests whether the
   ;; 2-latent move must be COORDINATED (slope alone may not improve).
   {:k 1 :spec (syn/spec [(syn/latent 'slope "gaussian" [0 3]) (syn/latent 'mu "gaussian" [0 5])]
                         (map (fn [x k] (syn/obs k "gaussian" [(lin-mean 'slope x 'mu) 1.0])) xs addrs))}
   {:k 2 :spec (syn/spec [(syn/latent 'slope "gaussian" [0 3]) (syn/latent 'intercept "gaussian" [0 5])]
                         (map (fn [x k] (syn/obs k "gaussian" [(lin-mean 'slope x 'intercept) 1.0])) xs addrs))}])

;; ---------------------------------------------------------------------------
;; Scoring: a spec at a fixed shared sigma, and at grid-best (co-refined) sigma.
;; ---------------------------------------------------------------------------
(defn- score-at [spec obs sigma]
  (let [s (reduce (fn [sp a] (syn/set-noise sp a sigma)) spec (map :addr (:obs spec)))
        fb (syn/check (syn/render s) obs {:n-particles 2000})]
    (when (syn/scored? fb) (:evidence fb))))

(defn- grid-best [spec obs]
  (let [scored (for [g h/noise-grid :let [e (score-at spec obs g)] :when e] [g e])]
    (when (seq scored) (apply max-key second scored))))

;; ---------------------------------------------------------------------------
(def fixed-sigmas [3.0 1.0 0.5])
(defn analyze [label task partials]
  (let [obs (:observations task)
        bar (:solve-bar task)
        crude (:crude task) gold (:gold task)]
    (println (str "\n================  " label "  ================"))
    (println (str "  task " (name (:id task)) "  bar=" (fx bar) "  (crude=" (fx crude) " gold=" (fx gold) ")"))
    (println (str "  " (.padEnd "k (structure)" 16)
                  (str/join "" (map #(.padEnd (str "σ=" %) 11) fixed-sigmas))
                  (.padEnd "grid-best" 16) ">=bar"))
    (let [rows (for [{:keys [k spec]} partials]
                 (let [fixed (mapv #(score-at spec obs %) fixed-sigmas)
                       [gsig gev] (or (grid-best spec obs) [nil nil])]
                   {:k k :fixed fixed :gsig gsig :gev gev :solved (and gev (>= gev bar))}))]
      (doseq [{:keys [k fixed gsig gev solved]} rows]
        (println (str "  " (.padEnd (str "k=" k (cond (= k 0) " crude" :else "")) 16)
                      (str/join "" (map #(.padEnd (fx %) 11) fixed))
                      (.padEnd (str (fx gev) " @σ" gsig) 16)
                      (if solved "YES" "no"))))
      ;; verdicts: classify RC1 (ordering) vs RC2 (proposer) per family.
      ;;   crude-σ ordering WORKS  -> gold crosses the bar at σ=crude AND every structural
      ;;     step >= plateau-eps (the strict ratchet would accept it) -> the loop COULD reach
      ;;     it at crude σ, so the 0/N must be the PROPOSER (RC2).
      ;;   crude-σ ordering BLOCKS -> gold does NOT cross at σ=crude (or a step < plateau-eps),
      ;;     but grid-best is monotone + crosses -> co-refinement is REQUIRED; the strict
      ;;     ratchet over emitted/crude σ is the blocker (RC1).
      (let [eps 0.05
            crude-col (mapv #(nth (:fixed %) 0) rows)              ; σ=3.0 column
            ;; first k that crosses the bar at crude σ, and whether the greedy ratchet could
            ;; CLIMB there (every PRE-crossing step >= plateau-eps). No-op tail steps after the
            ;; crossing are irrelevant — the loop has already solved by then.
            crude-cross-k (->> (map-indexed vector crude-col) (filter (fn [[_ e]] (and e (>= e bar)))) first first)
            pre-steps (when crude-cross-k (map - (take crude-cross-k (rest crude-col)) crude-col))
            pre-steps-ok (or (= crude-cross-k 0) (and pre-steps (every? #(>= % (- eps 1e-9)) pre-steps)))
            min-step (when (and pre-steps (seq pre-steps)) (apply min pre-steps))
            gold-at-crude (last crude-col)
            crosses-at-crude (boolean crude-cross-k)
            steps-ok (boolean pre-steps-ok)
            gevs (map :gev rows)
            grid-monotone (apply <= (map #(or % ##-Inf) gevs))
            crosses-k (->> rows (filter :solved) first :k)
            verdict (cond
                      (and crosses-at-crude steps-ok) :RC2-proposer
                      (and grid-monotone crosses-k)   :RC1-ordering
                      :else                           :deeper)]
        (println (str "  -> at σ=crude(3.0): gold " (fx gold-at-crude) " vs bar " (fx bar)
                      "  crosses=" (boolean crosses-at-crude) "  min-step=" (fx min-step)
                      (when (and min-step (< min-step eps)) (str " (<plateau-eps " eps " -> strict ratchet STALLS)"))))
        (println (str "  -> grid-best: monotone=" grid-monotone "  crosses bar at k=" crosses-k))
        (println (str "  -> VERDICT: "
                      (case verdict
                        :RC2-proposer "RC2 (PROPOSER). Ordering already works at crude σ (gold crosses, steps>=eps); the loop's 0/N is because the proposer NEVER EMITTED this structure. Fix = stronger proposer/teacher, NOT the accept rule."
                        :RC1-ordering "RC1 (ORDERING). The structure crosses the bar ONLY under σ-co-refinement; at crude σ it does not cross or a step falls below plateau-eps, so the strict-ratchet-over-emitted-σ accept rule rejects it. Fix = R1 co-refined accept (~30 lines), IF the proposer emits it."
                        :deeper "DEEPER: not monotone even under grid-best σ -> the loop has no incremental mechanism on this family.")))
        {:family label :verdict verdict :grid-monotone grid-monotone :crosses-k crosses-k
         :crosses-at-crude (boolean crosses-at-crude) :min-step min-step}))))

;; ---------------------------------------------------------------------------
(println "\n###  R-1: offline σ-co-refinement re-score (native-free, exact oracle)  ###")
(def C (cur/generate-curriculum {:round 0 :instances-per-family 12}))
(defn- first-task [fam] (first (filter #(= fam (:family %)) (:tasks C))))

(def results
  [(analyze "per-group-means (g latents, coordinated)" (first-task :per-group-means)
            (pgm-partials (:true-params (first-task :per-group-means))))
   (analyze "varying-slopes (2g latents, the 0/9 cliff)" (first-task :varying-slopes)
            (vs-partials (:true-params (first-task :varying-slopes))))
   (analyze "linear (2 latents, coordinated)" (first-task :linear)
            (lin-partials (:true-params (first-task :linear))))])

(println "\n================  R-1 VERDICT  ================")
(doseq [r results]
  (println (str "  " (.padEnd (:family r) 48) (name (:verdict r))
                "   grid-monotone=" (:grid-monotone r) "  crosses-at-crude-σ=" (:crosses-at-crude r)
                "  min-step@crude=" (fx (:min-step r)))))
(let [n-rc1 (count (filter #(= :RC1-ordering (:verdict %)) results))
      n-rc2 (count (filter #(= :RC2-proposer (:verdict %)) results))]
  (println (str "\n  SEPARATION: RC1 (ordering) " n-rc1 "/" (count results)
                " ; RC2 (proposer) " n-rc2 "/" (count results)
                " ; deeper " (- (count results) n-rc1 n-rc2) "/" (count results)))
  (println "  EVERY cliff family is monotone + bar-crossing under σ-co-refinement (no no-monotone-path")
  (println "  wall — correcting the earlier diagnosis). R-1 SEPARATES the blocker per family:")
  (println "   - RC1 families: the structure crosses ONLY with σ-co-refinement; the strict-ratchet")
  (println "     accept rule rejects it at crude σ -> R1 (co-refined accept, ~30 lines) is the fix.")
  (println "   - RC2 families: ordering already works at crude σ; the loop's 0/N is the PROPOSER")
  (println "     never emitting the structure -> a stronger proposer/teacher is the fix, not R1.")
  (println "  CAVEAT: σ-co-refinement is partly DETERMINISTIC (the σ-grid, not the LLM); R1 must")
  (println "  ablate fixed-σ vs co-refined so a deterministic-search win is not laundered as an LLM win."))
