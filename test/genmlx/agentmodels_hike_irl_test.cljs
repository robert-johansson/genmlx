;; Headless test for the Big-Hiking world + hiker inverse planning (agentmodels
;; Ch 3b world + Ch 4 IRL) — a SECOND IRL domain beyond the Restaurant-Choice grid.
;; Observe a hiker's path and infer which PEAK they value (East vs West), reusing
;; genmlx.agents.inverse/{goal-agents,posterior-sequence}. The peaks are close in
;; value (East 10 / West 7) with a steep Hill cliff (-40) both hypotheses avoid, so
;; the destination is informative: the posterior is flat over the shared prefix and
;; sharpens to the true peak as the path commits. RNG-free (argmax-Q observed path).
;; Run: bun run --bun nbb test/genmlx/agentmodels_hike_irl_test.cljs

(ns genmlx.agentmodels-hike-irl-test
  (:require [agentmodels.hike-irl :as hi]
            [genmlx.agents.worlds :as w]
            [genmlx.mlx :as mx]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))
(defn- close? [a b tol] (<= (Math/abs (- a b)) tol))

;; --- Big-Hiking world structure -------------------------------------------
(println "\n-- Big-Hiking gridworld (agentmodels makeBigHikeMDP) --")
(let [m (w/big-hike-mdp {})]
  (assert-equal "big-hike S = 36"          36 (:S m))
  (assert-equal "big-hike walls {14 20 22}" #{14 20 22} (:walls m))
  (assert-equal "big-hike terminals (W=21 E=23 Hill=30..35)"
                {21 :west 23 :east 30 :hill 31 :hill 32 :hill 33 :hill 34 :hill 35 :hill} (:terminals m))
  (assert-equal "big-hike start-idx = 7 ([1,1])" 7 (:start-idx m))
  (assert-equal "big-hike T shape [36 4 36]" [36 4 36] (mx/shape (:T m)))
  (assert-true  "big-hike T rows each sum to 1 (total 144)"
                (close? 144.0 (mx/item (mx/sum (:T m))) 1e-2)))

;; --- hiker inverse planning: infer the preferred peak from the path -------
(println "\n-- hiker IRL: infer East-vs-West preference from a trajectory --")
(def agents (hi/peak-agents))
(assert-equal "two peak hypotheses {:east :west}" #{:east :west} (set (keys agents)))

(let [e-obs (hi/greedy-observations (:east agents))
      e-seq (hi/infer-peak agents e-obs)
      e-ps  (mapv :east e-seq)
      w-obs (hi/greedy-observations (:west agents))
      w-seq (hi/infer-peak agents w-obs)
      w-ps  (mapv :east w-seq)]
  (println "   East-hiker path :" (mapv first e-obs) "  P(east):" (mapv #(.toFixed % 2) e-ps))
  (println "   West-hiker path :" (mapv first w-obs) "  P(east):" (mapv #(.toFixed % 2) w-ps))
  ;; prior is uniform over the two peaks
  (assert-true "prior over peaks is uniform (P(east) = 0.5)" (close? 0.5 (first e-ps) 1e-6))
  ;; the shared prefix is UNINFORMATIVE — the posterior only moves once paths diverge
  (assert-true "after the (shared) 2nd action the posterior is still ~uniform" (close? 0.5 (nth e-ps 2) 0.05))
  ;; observing an East-preferring hiker concentrates the posterior on :east; its
  ;; evidence never meaningfully supports West (any shared-prefix wobble is < 0.05)
  (assert-true "East-hiker: P(east) never swings toward West (stays >= 0.45)" (every? #(>= % 0.45) e-ps))
  (assert-true "East-hiker: final P(east) > 0.99 (preference identified)" (> (last e-ps) 0.99))
  ;; ...and a West-preferring hiker concentrates it on :west (symmetric, identifiable)
  (assert-true "West-hiker: P(east) never swings toward East (stays <= 0.55)" (every? #(<= % 0.55) w-ps))
  (assert-true "West-hiker: final P(east) < 0.01 (i.e. P(west) > 0.99)" (< (last w-ps) 0.01))
  ;; the two hikers' paths share a prefix then diverge (what makes it informative)
  (assert-true "the two paths share their first action but reach different states"
               (and (= (ffirst e-obs) (ffirst w-obs))
                    (not= (first (last e-obs)) (first (last w-obs))))))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
