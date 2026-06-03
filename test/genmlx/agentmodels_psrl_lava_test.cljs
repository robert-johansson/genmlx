;; Headless test for the goal+lava PSRL exploration world (genmlx.agents.worlds/
;; lava-world) on the agentmodels Ch 3d PSRL driver. Demonstrates that lava + noise
;; make exploration VISIBLY MEANINGFUL: while uncertain, the agent samples goals all
;; over and chasing a bottom goal forces it across the lava gap (slip -> penalty);
;; once it learns the true goal is in the safe top, it stops crossing. So learning
;; both REDUCES lava exposure and cuts regret vs the no-learning baseline.
;;
;; The PSRL run is seed-driven (deterministic LCG for Thompson sampling AND the
;; orthogonal-slip transitions), so these summed-over-seeds results are reproducible.
;; Run: bun run --bun nbb test/genmlx/agentmodels_psrl_lava_test.cljs

(ns genmlx.agentmodels-psrl-lava-test
  (:require [agentmodels.psrl :as ps]
            [genmlx.agents.worlds :as w]
            [genmlx.agents.gridworld :as gw]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))

(defn- cum [r] (last (:cum-regret r)))
(defn- lava-hits [r lava]
  (reduce + (map (fn [e] (count (filter lava (butlast (:states e))))) (:episodes r))))

;; ---------------------------------------------------------------------------
(println "\n-- world structure --")
(let [{:keys [lava true-goal grid]} w/lava-world
      free (set (remove lava (ps/free-cells grid)))]
  (assert-equal "lava flanks the row-2 gap (idx {10 11 13 14})" #{10 11 13 14} lava)
  (assert-equal "the gap (idx 12) is NOT lava" false (contains? lava 12))
  (assert-equal "the true goal (idx 3) is in the safe top region" 3 true-goal)
  (assert-true  "goal candidates exclude the known lava cells" (not (some lava free)))
  (assert-true  "the goal is a candidate; lava cells are not"
                (and (free true-goal) (not (some free lava)))))

;; ---------------------------------------------------------------------------
(println "\n-- learning reduces lava exposure AND regret (vs no-learning) --")
(def seeds [1 2 3 4 5 6])
(def runs
  (vec (for [s seeds]
         {:seed s
          :learn    (ps/psrl (merge w/lava-world {:seed s :n-episodes 12 :learn? true}))
          :baseline (ps/psrl (merge w/lava-world {:seed s :n-episodes 12 :learn? false}))})))

(let [lava (:lava w/lava-world)
      sum  (fn [k f] (reduce + (map (fn [r] (f (k r))) runs)))
      hl   (sum :learn    #(lava-hits % lava))
      hb   (sum :baseline #(lava-hits % lava))
      rl   (sum :learn    cum)
      rb   (sum :baseline cum)
      reach (reduce + (map (fn [r] (count (filter :reached-goal? (:episodes (:learn r))))) runs))]
  (println (str "   summed over " (count seeds) " seeds: lava-hits learn=" hl " baseline=" hb
                " | regret learn=" (.toFixed rl 1) " baseline=" (.toFixed rb 1)
                " | goal-reaches=" reach "/" (* (count seeds) 12)))
  ;; each seed converges to the true goal
  (doseq [{:keys [seed learn]} runs]
    (assert-true (str "seed " seed ": posterior concentrates on the true goal (P >= 0.99)")
                 (>= (get (:final-posterior learn) (:true-goal w/lava-world) 0.0) 0.99)))
  ;; lava genuinely bites (it is not inert geometry) ...
  (assert-true "lava genuinely bites: the no-learning baseline incurs lava penalties" (pos? hb))
  ;; ... and learning REDUCES the lava exposure (stops crossing once it knows the goal)
  (assert-true "learning reduces total lava exposure vs the no-learning baseline" (< hl hb))
  ;; the headline: learning cuts cumulative regret substantially
  (assert-true "learning cuts total cumulative regret vs the baseline" (< rl rb))
  (assert-true "the regret reduction is large (learn < 0.6 * baseline)" (< rl (* 0.6 rb)))
  ;; the agent does reach the (safe) goal in the majority of episodes
  (assert-true "the agent reaches the goal in the majority of episodes" (> reach (* 0.5 (count seeds) 12))))

;; ---------------------------------------------------------------------------
(println "\n-- backward-compat: the bland-grid PSRL is unchanged (defaults) --")
;; no lava / no noise => identical to the original deterministic PSRL: final episode
;; converges to zero regret and reaches the goal.
(let [r (ps/psrl {:seed 1 :n-episodes 10})
      last-ep (last (:episodes r))]
  (assert-true "default PSRL still converges (final-episode regret = 0)" (= 0.0 (:regret last-ep)))
  (assert-true "default PSRL final episode reaches the goal" (:reached-goal? last-ep)))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
