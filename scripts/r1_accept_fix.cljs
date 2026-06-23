(ns r1-accept-fix
  "R1 (genmlx-8smp): the accept-rule fix, isolated + ablated. NATIVE-FREE.

   R-1 proved the cliff structures are MONOTONE + bar-crossing under sigma-co-refinement; the
   STRICT-ratchet-over-emitted-sigma accept rule rejects them. R1 tests the fix directly:
   replace 'score the candidate at its EMITTED sigma' with 'score at GRID-BEST sigma'
   (co-refined), and walk the INCREMENTAL structural path (crude -> +1 unit -> ... -> gold) the
   way an LLM-driven loop would. Two accept modes, same path:
     STRICT    : score each partial at its emitted sigma (crude at 3.0, structure at 1.0).
     CO-REFINE : score each partial at its best shared sigma over the 8-pt grid.

   MANDATORY ABLATION: the CO-REFINE win uses the DETERMINISTIC sigma-grid, not an LLM. So a
   CO-REFINE solve is NOT an LLM-loop vindication by itself -- it shows the accept rule was the
   blocker for a CORRECT incremental proposer. The LLM-attributable part is PROPOSING the
   structure; the sigma tuning is free. (The LLM run -- does the 35B emit the structure
   incrementally -- is the R1/R2 follow-on.)

   OVER-ACCEPTANCE GUARD: on data from the SIMPLER model (shared-mean), CO-REFINE must NOT
   accept a spurious extra-latent split (Occam must hold), or the fix would overfit.

   Run: bun run --bun nbb scripts/r1_accept_fix.cljs"
  (:require [genmlx.world.curriculum :as cur]
            [genmlx.world.harvest :as h]
            [genmlx.world.synth :as syn]
            [clojure.string :as str]))

(defn fx [x] (if (and x (js/isFinite x)) (.toFixed (js/Number x) 2) (str x)))
(defn- lin-mean [s-sym x i-sym]
  (list 'mx/add (list 'mx/multiply s-sym (list 'mx/scalar x)) i-sym))

;; ---------------------------------------------------------------------------
;; Incremental partial path: p0 = crude (sigma 3.0, what the loop starts from);
;; p1..pK = the structure added one UNIT at a time (sigma 1.0, what an LLM emits).
;; ---------------------------------------------------------------------------
(defn pgm-path [{:keys [groups pp]}]
  (for [k (range (inc (count groups)))]
    (let [own (set (take k groups)) shared? (some (complement own) groups)
          sig (if (zero? k) 3.0 1.0)]
      (syn/spec (concat (when shared? [(syn/latent 'mu "gaussian" [0 5])])
                        (for [gn groups :when (own gn)] (syn/latent (symbol (str "m-" gn)) "gaussian" [0 5])))
                (for [gn groups i (range pp)]
                  (syn/obs (keyword (str gn i)) "gaussian"
                           [(if (own gn) (symbol (str "m-" gn)) 'mu) sig]))))))

(defn vs-path [{:keys [groups pp]}]
  (for [k (range (inc (count groups)))]
    (let [own (set (take k groups)) shared? (some (complement own) groups)
          sig (if (zero? k) 3.0 1.0)]
      (syn/spec (concat (when shared? [(syn/latent 'mu "gaussian" [0 5])])
                        (mapcat (fn [gn] [(syn/latent (symbol (str "s-" gn)) "gaussian" [0 3])
                                          (syn/latent (symbol (str "i-" gn)) "gaussian" [0 5])])
                                (filter own groups)))
                (for [gn groups x (range pp)]
                  (syn/obs (keyword (str gn x)) "gaussian"
                           [(if (own gn) (lin-mean (symbol (str "s-" gn)) x (symbol (str "i-" gn))) 'mu) sig]))))))

(defn lin-path [{:keys [xs addrs]}]
  [(syn/spec [(syn/latent 'mu "gaussian" [0 5])] (for [k addrs] (syn/obs k "gaussian" ['mu 3.0])))
   (syn/spec [(syn/latent 'slope "gaussian" [0 3]) (syn/latent 'mu "gaussian" [0 5])]
             (map (fn [x k] (syn/obs k "gaussian" [(lin-mean 'slope x 'mu) 1.0])) xs addrs))
   (syn/spec [(syn/latent 'slope "gaussian" [0 3]) (syn/latent 'intercept "gaussian" [0 5])]
             (map (fn [x k] (syn/obs k "gaussian" [(lin-mean 'slope x 'intercept) 1.0])) xs addrs))])

;; ---------------------------------------------------------------------------
;; Scoring + the two accept modes walking the incremental path greedily.
;; ---------------------------------------------------------------------------
(defn- ev-at [spec obs sigma]
  (let [s (reduce (fn [sp a] (syn/set-noise sp a sigma)) spec (map :addr (:obs spec)))
        fb (syn/check (syn/render s) obs {:n-particles 2000})]
    (when (syn/scored? fb) (:evidence fb))))
(defn- ev-emitted [spec obs] (ev-at spec obs (last (:args (first (:obs spec))))))
(defn- ev-corefine [spec obs] (apply max (keep #(ev-at spec obs %) h/noise-grid)))

(def plateau-eps 0.05)
(defn walk [path obs score-fn bar]
  ;; greedy ratchet over the ordered incremental path; stop at the first rejected step.
  (loop [i 0 cur-ev (score-fn (first path) obs) accepted [0]]
    (if (>= (inc i) (count path))
      {:final-ev cur-ev :solved (and cur-ev (>= cur-ev bar)) :accepted accepted :reached-gold true}
      (let [nxt-ev (score-fn (nth path (inc i)) obs)]
        (if (and nxt-ev (> nxt-ev (+ cur-ev plateau-eps)))
          (recur (inc i) nxt-ev (conj accepted (inc i)))
          {:final-ev cur-ev :solved (and cur-ev (>= cur-ev bar)) :accepted accepted
           :reached-gold false :stuck-at (inc i) :rejected-ev nxt-ev})))))

;; ---------------------------------------------------------------------------
(def C (cur/generate-curriculum {:round 0 :instances-per-family 12}))
(defn- a-task [fam] (first (filter #(= fam (:family %)) (:tasks C))))

(println "\n### R1: strict-ratchet vs sigma-co-refined accept (incremental path, native-free) ###")
(def families
  [["per-group-means" (a-task :per-group-means) pgm-path]
   ["varying-slopes (the 0/9 cliff)" (a-task :varying-slopes) vs-path]
   ["linear" (a-task :linear) lin-path]])

(def results
  (vec (for [[label task pathfn] families]
         (let [obs (:observations task) bar (:solve-bar task)
               path (pathfn (:true-params task))
               strict (walk path obs ev-emitted bar)
               coref  (walk path obs ev-corefine bar)]
           (println (str "\n  " label "  (bar=" (fx bar) ", " (count path) " steps on the path)"))
           (println (str "    STRICT (emitted σ)  : final=" (fx (:final-ev strict))
                         " solved=" (:solved strict) " accepted-steps=" (:accepted strict)
                         (when-not (:reached-gold strict) (str " STUCK at step " (:stuck-at strict)
                                                               " (rejected ev " (fx (:rejected-ev strict)) ")"))))
           (println (str "    CO-REFINE (grid σ) : final=" (fx (:final-ev coref))
                         " solved=" (:solved coref) " accepted-steps=" (:accepted coref)
                         (when (:reached-gold coref) " REACHED GOLD")))
           {:label label :strict-solved (:solved strict) :coref-solved (:coref-solved coref)
            :coref-solved2 (:solved coref) :unlocked (and (:solved coref) (not (:solved strict)))}))))

;; ---------------------------------------------------------------------------
;; Over-acceptance guard: shared-mean data + a spurious per-group split. Co-refine must
;; NOT accept the spurious extra latent (Occam / marginal-likelihood must penalize it).
;; ---------------------------------------------------------------------------
(println "\n### Over-acceptance guard (Occam under co-refine) ###")
(let [t (a-task :shared-mean)
      obs (:observations t)
      addrs (vec (keys obs))
      half (quot (count addrs) 2)
      crude (syn/spec [(syn/latent 'mu "gaussian" [0 5])] (for [k addrs] (syn/obs k "gaussian" ['mu 1.0])))
      ;; spurious: split the (structureless) points into two arbitrary groups with separate means
      spurious (syn/spec [(syn/latent 'mu1 "gaussian" [0 5]) (syn/latent 'mu2 "gaussian" [0 5])]
                         (map-indexed (fn [i k] (syn/obs k "gaussian" [(if (< i half) 'mu1 'mu2) 1.0])) addrs))
      ec (ev-corefine crude obs) es (ev-corefine spurious obs)]
  (println (str "  shared-mean data: crude (1 latent) co-refine ev=" (fx ec)
                " ; spurious split (2 latents) co-refine ev=" (fx es)))
  (println (str "  spurious accepted? " (boolean (and es (> es (+ ec plateau-eps))))
                "  -> guard " (if (and es (> es (+ ec plateau-eps))) "FAILS (over-accepts!)" "PASSES (Occam holds)"))))

;; ---------------------------------------------------------------------------
(println "\n================  R1 VERDICT  ================")
(doseq [r results]
  (println (str "  " (.padEnd (:label r) 32) "strict-solved=" (:strict-solved r)
                "  co-refine-solved=" (:coref-solved2 r) "  UNLOCKED-by-fix=" (:unlocked r))))
(let [n-unlocked (count (filter :unlocked results))]
  (println (str "\n  The co-refined accept UNLOCKS " n-unlocked "/" (count results)
                " cliff families that the strict ratchet rejects (incremental path)."))
  (println "  GATE: GO if a structural family moves off the floor under co-refine AND the")
  (println "  over-acceptance guard PASSES (the win is structure the loop now accepts, not a")
  (println "  spurious-latent overfit). ABLATION: the σ-tuning is deterministic; the LLM-")
  (println "  attributable part is emitting the structure -> the LLM run (does the 35B emit it")
  (println "  incrementally) is the next step, with the compute-matched one-shot+rerank arm."))
