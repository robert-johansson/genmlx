;; @tier slow
;; Headless tests for agentmodels Ch 3d — Posterior Sampling RL (PSRL) on a gridworld.
;; (The bandit Thompson/softmax-greedy + bandit regret are shipped & tested in
;; bandit_test.cljs; this covers the remaining model-based gridworld PSRL.)
;; Run: bunx nbb@1.4.206 test/genmlx/agentmodels_psrl_test.cljs

(ns genmlx.agentmodels-psrl-test
  (:require [agentmodels.psrl :as ps]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))

;; ===========================================================================
(println "\n== Phase 1: the reward-grid MDP + optimal value ==")
(let [free (ps/free-cells ps/grid)
      v*   (ps/optimal-return ps/grid (#'agentmodels.psrl/one-hot 16 ps/TRUE-GOAL) ps/START-IDX ps/HORIZON)]
  (println "  free cells:" free "  true-goal:" ps/TRUE-GOAL "  V*(start)=" v*)
  (assert-equal "12 free (non-wall) cells" 12 (count free))
  (assert-true  "true goal (idx 15) is a free cell" (some #(= ps/TRUE-GOAL %) free))
  ;; the goal is 6 steps from the start; over a 9-step horizon the optimal agent
  ;; reaches it and dwells (stay action), collecting reward on 3 of the 9 steps.
  (assert-true  "V*(start) = 3 (reach in 6, dwell for 3 of 9 steps)" (= 3.0 v*)))

;; ===========================================================================
(println "\n== Phase 2: exact reward-posterior update (Bayes) ==")
(let [post (ps/uniform-posterior (ps/free-cells ps/grid))]
  (assert-true "uniform prior over the 12 free cells (each ≈ 1/12)"
               (< (Math/abs (- (get post 0) (/ 1.0 12))) 1e-9))
  ;; observing reward 0 at a cell rules that cell out
  (let [p1 (ps/update-posterior post 0 0)]
    (assert-true "observing reward 0 at cell 0 removes it from the posterior" (nil? (get p1 0)))
    (assert-true "...and renormalizes the survivors" (< (Math/abs (- 1.0 (reduce + (vals p1)))) 1e-9)))
  ;; observing reward 1 at a cell collapses the posterior to it
  (let [p2 (ps/update-posterior post ps/TRUE-GOAL 1)]
    (assert-equal "observing reward 1 at the goal collapses the posterior to it" 1 (count p2))
    (assert-true  "...with all mass on the goal" (> (get p2 ps/TRUE-GOAL) 0.999))))

;; ===========================================================================
(println "\n== Phase 3: PSRL learns the reward — concentration, regret, beats baseline ==")
;; over several seeds the posterior must concentrate on the true goal, cumulative
;; regret must stay sublinear (plateau) and beat the no-learning baseline.
(let [seeds [1 2 3 4 5]]
  (doseq [seed seeds]
    (let [r (ps/psrl {:seed seed :n-episodes 10})
          b (ps/psrl {:seed seed :n-episodes 10 :learn? false})
          p-true (get (:final-posterior r) ps/TRUE-GOAL 0.0)
          tot    (last (:cum-regret r))
          base   (last (:cum-regret b))
          last-ep (last (:episodes r))]
      (println "  seed" seed " P(true)=" (.toFixed p-true 3) " regret=" (.toFixed tot 1)
               " baseline=" (.toFixed base 1) " final-ep regret=" (.toFixed (:regret last-ep) 1))
      (assert-true (str "seed " seed ": posterior concentrates on the true goal (P ≥ 0.99)") (>= p-true 0.99))
      (assert-true (str "seed " seed ": final episode has zero regret (converged to optimal)") (= 0.0 (:regret last-ep)))
      (assert-true (str "seed " seed ": final episode reaches the goal") (:reached-goal? last-ep))
      (assert-true (str "seed " seed ": PSRL beats the no-learning baseline") (< tot base))
      (assert-true (str "seed " seed ": cumulative regret is sublinear (< n·V* = 30)") (< tot 30.0)))))

;; ===========================================================================
(println "\n== Phase 4: regret plateaus — the agent stops paying once it has learned ==")
(let [r (ps/psrl {:seed 1 :n-episodes 12})
      cr (:cum-regret r)
      first-half  (- (nth cr 5) (nth cr 0))          ; regret accrued over episodes 1..6
      second-half (- (last cr) (nth cr 5))]          ; regret accrued over episodes 7..12
  (println "  cum-regret:" (mapv #(.toFixed % 1) cr))
  (assert-true "regret accrual in the 2nd half ≪ 1st half (it plateaus once learned)"
               (< second-half (inc first-half)))
  (assert-true "no further regret once the posterior has collapsed (2nd-half accrual = 0)"
               (= 0.0 second-half)))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
