;; Headless tests for the multi-armed bandit POMDP (agentmodels Ch 3c/3d):
;; host-side Beta-Bernoulli belief filter + Thompson posterior sampling. The
;; belief update and arm-values are pure; the rollout is made reproducible by
;; passing a seeded key (rng/fresh-key 42) to simulate-bandit.
;;
;; Run: bun run --bun nbb test/genmlx/bandit_test.cljs

(ns genmlx.bandit-test
  (:require [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as env]
            [genmlx.agents.presentation :as pres]
            [genmlx.mlx.random :as rng]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))

(def ag (pomdp/make-bandit-agent {:strategy :thompson}))

(println "\n== Section 1: Beta-Bernoulli belief update (pure) ==")
(let [ub (:update-belief ag)
      b0 {:arms [[1.0 1.0] [1.0 1.0] [1.0 1.0]]}]
  (assert-equal "success on arm 0 -> Beta(2,1)"     [2.0 1.0] (get-in (ub b0 0 1) [:arms 0]))
  (assert-equal "unpulled arm 1 unchanged"          [1.0 1.0] (get-in (ub b0 0 1) [:arms 1]))
  (assert-equal "unpulled arm 2 unchanged"          [1.0 1.0] (get-in (ub b0 0 1) [:arms 2]))
  (assert-equal "failure on arm 2 -> Beta(1,2)"     [1.0 2.0] (get-in (ub b0 2 0) [:arms 2])))

(println "\n== Section 2: arm-values = posterior mean alpha/(alpha+beta) (pure) ==")
(let [av (:arm-values ag)]
  (assert-close "mean Beta(8,2) = 0.8" 0.8 (nth (av {:arms [[8.0 2.0]]}) 0) 1e-9)
  (assert-close "mean Beta(1,1) = 0.5" 0.5 (nth (av {:arms [[1.0 1.0]]}) 0) 1e-9))

(println "\n== Section 3: belief converges + pulls concentrate on the best arm ==")
;; 3 Bernoulli arms; arm 2 (theta=0.8) is best. Seeded rollout, horizon 60.
(def bandit (env/bandit-pomdp {:thetas [0.25 0.50 0.80] :horizon 60}))
(def roll   (pomdp/simulate-bandit ag bandit (rng/fresh-key 42)))
(let [means  ((:arm-values ag) (last (:beliefs roll)))
      counts (frequencies (:arms roll))]
  (println "  pull counts:" (into (sorted-map) counts) " | final means:" (mapv #(.toFixed % 2) means))
  (assert-equal "true-best arm is index 2" 2 (:true-best bandit))
  (assert-close "best-arm posterior mean -> ~0.8" 0.80 (nth means 2) 0.15)
  (assert-true  "best arm (2) is the modal pull" (= 2 (apply max-key #(get counts % 0) (range 3))))
  (assert-true  "best arm pulled a majority of the time" (> (get counts 2 0) (/ 60 2))))

(println "\n== Section 4: cumulative regret is sublinear ==")
(let [reg (:regret roll)
      early (/ (nth reg 9) 10.0)                         ; avg regret/step, first 10
      late  (/ (- (last reg) (nth reg 49)) 10.0)]        ; avg regret/step, last 10
  (println "  regret: early-slope" (.toFixed early 3) " late-slope" (.toFixed late 3)
           " total" (.toFixed (last reg) 2))
  (assert-true "regret accrues early"               (> early 0.0))
  (assert-true "regret slope flattens (sublinear)"  (< late early))
  (assert-true "total regret well below the worst-arm bound"
               (< (last reg) (* 60 (- 0.80 0.25)))))

(println "\n== Section 5: the seam — bandit belief -> PosteriorBars ==")
(let [bars (pres/bandit-bars {:arms [[2.0 6.0] [3.0 3.0] [9.0 2.0]]} 2)]
  (assert-equal "one bar per arm" 3 (count (:bars bars)))
  (assert-equal "labels" ["arm0" "arm1" "arm2"] (mapv :label (:bars bars)))
  (assert-close "arm2 weight = posterior mean 9/11" (/ 9.0 11.0) (:weight (nth (:bars bars) 2)) 1e-9)
  (assert-true  "true-best arm highlighted"        (:highlight (nth (:bars bars) 2)))
  (assert-true  "other arms not highlighted"       (not (:highlight (nth (:bars bars) 0)))))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
