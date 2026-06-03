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
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
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

;; ===========================================================================
;; Tensor bandit (bean genmlx-4ifp): [K] alpha/beta tensors + masked bb-update
;; ===========================================================================
(println "\n== Section 6: tensor masked bb-increment == host update-arm ==")
;; The one-hot-masked tensor increment must produce EXACTLY the same counts as the
;; host update-arm for every (arm, reward) pair (increments are exact in float32).
(let [K 3
      a0 (mx/array (clj->js [1.0 1.0 1.0]) mx/float32)
      b0 (mx/array (clj->js [1.0 1.0 1.0]) mx/float32)
      host0 {:arms [[1.0 1.0] [1.0 1.0] [1.0 1.0]]}]
  (doseq [i (range K), r [0 1]]
    (let [[a' b'] (pomdp/tensor-bb-increment a0 b0 i r K)
          host    (:arms (pomdp/update-arm host0 i r))]
      (assert-equal (str "bb-increment arm " i " reward " r ": alpha matches update-arm")
                    (mapv first host)  (vec (mx/->clj a')))
      (assert-equal (str "bb-increment arm " i " reward " r ": beta matches update-arm")
                    (mapv second host) (vec (mx/->clj b'))))))

(println "\n== Section 7: [K] Thompson sampler — moments + selection agreement ==")
;; (a) per-arm Beta moments: column means of N [K] draws match alpha/(alpha+beta).
(let [N 20000
      av (mx/array (clj->js (vec (repeat N [2.0 5.0 8.0]))) mx/float32)   ; [N,K]
      bv (mx/array (clj->js (vec (repeat N [8.0 5.0 2.0]))) mx/float32)
      theta (dist/beta-sample-vec av bv (rng/fresh-key 7))
      means (vec (mx/->clj (mx/mean theta [0])))]
  (println "  [K] Beta column means:" (mapv #(.toFixed % 3) means) "(expect 0.2/0.5/0.8)")
  (doseq [[m e] (map vector means [0.2 0.5 0.8])]
    (assert-close "[K] Beta column mean matches alpha/(alpha+beta)" e m 0.01)))
;; (b) argmax-selection frequencies from the [K] sampler match the OLD per-arm
;;     dist/beta-dist-then-argmax method to small tolerance (same posteriors).
(let [M 4000, K 3
      a [2.0 4.0 5.0], b [6.0 4.0 3.0]                                    ; distinct Beta arms
      av (mx/array (clj->js (vec (repeat M a))) mx/float32)               ; [M,K]
      bv (mx/array (clj->js (vec (repeat M b))) mx/float32)
      ;; new [K] (batched as [M,K]) sampler: argmax along arm axis per row
      new-sel (vec (mx/->clj (mx/argmax (dist/beta-sample-vec av bv (rng/fresh-key 123)) 1)))
      new-freq (mapv (fn [k] (/ (count (filter #(= k %) new-sel)) (double M))) (range K))
      ;; old reference: K scalar dist/beta-dist draws per trial, argmax
      old-sel (mapv (fn [t]
                      (let [ks (rng/split-n (rng/fresh-key (+ 9000 t)) K)
                            ss (mapv (fn [ki [aa bb]] (mx/item (dist/sample (dist/beta-dist aa bb) ki)))
                                     ks (map vector a b))]
                        (apply max-key #(nth ss %) (range K))))
                    (range M))
      old-freq (mapv (fn [k] (/ (count (filter #(= k %) old-sel)) (double M))) (range K))]
  (println "  selection freq  new:" (mapv #(.toFixed % 3) new-freq) " old:" (mapv #(.toFixed % 3) old-freq))
  (assert-true "[K] sampler arm-selection freq matches per-arm sampler (max diff < 0.05)"
               (< (apply max (map (fn [n o] (Math/abs (- n o))) new-freq old-freq)) 0.05)))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
