;; @tier slow
;; Headless tests for differentiable MDP utility/alpha learning (bean genmlx-j5um):
;; genmlx.agents.differentiable. Recover planted agent params by gradient through
;; the planner (lazy value iteration) + the policy log-likelihood at fixed observed
;; (s,a) — exactly the inverse/action-loglik quantity. MDP-only scope (no belief /
;; no stochastic-rollout gradient). Identifiability: recovery is to LIKELIHOOD-
;; equivalence (loss at recovered ≈ loss at plant), with fix-one-learn-the-other
;; sub-tests; the data is a deterministic expected-counts dataset of a finite-alpha
;; policy, so the test is seed-free.
;;
;; Run: bunx nbb@1.4.206 test/genmlx/agentmodels_diff_learn_test.cljs

(ns genmlx.agentmodels-diff-learn-test
  (:require [genmlx.agents.differentiable :as diff]
            [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.inverse :as inv]
            [genmlx.mlx :as mx]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))

(defn maxerr [a b] (apply max 0.0 (map (fn [x y] (Math/abs (- x y))) (flatten a) (flatten b))))

;; Deterministic dataset: for each non-terminal state, emit (s,a) round(K·π[s,a])
;; times — an "expected counts" dataset whose MLE is exactly the planted policy
;; (so loss→loss-at-plant and a finite alpha is identifiable). Seed-free.
(defn policy-dataset [mdp alpha k n-iters]
  (let [{:keys [Q]} (agent/value-iteration mdp alpha n-iters)
        S (:S mdp) A (:A mdp) terms (set (keys (:terminals mdp)))
        pol (mx/->clj (mx/softmax (mx/multiply (mx/scalar alpha) Q) 1))]
    (vec (for [s (range S) :when (not (terms s))
               a (range A)
               _ (range (Math/round (* k (double (nth (nth pol s) a)))))]
           [s a]))))

(def N 40)              ; VI sweeps (finite horizon; same for plant + learner)
(def TC -0.1)

;; ---------------------------------------------------------------------------
(println "\n== Section 1: diff-reward reconstructs build-mdp's R ==")
(doseq [[label grid goals utils]
        [["1x5 corridor" [[:A :empty :empty :empty :B]] [:A :B] {:A 3.0 :B 1.0}]
         ["3x3 hike"      [[:A :empty :B] [:empty :empty :empty] [:empty :empty :C]] [:A :B :C] {:A 5.0 :B 2.0 :C -1.0}]]]
  (let [dmdp   (diff/build-diff-mdp {:grid grid :goals goals})
        theta  (mx/array (clj->js (mapv utils goals)) mx/float32)
        diff-R (gw/diff-reward (:goal-onehot dmdp) theta TC (:S dmdp) (:A dmdp) (:G dmdp))
        host-R (:R (gw/build-mdp {:grid grid :utilities (assoc utils :timeCost TC)}))]
    (assert-true (str label ": diff-reward == build-mdp R (1e-5)")
                 (< (maxerr (mx/->clj diff-R) (mx/->clj host-R)) 1e-5))))

(println "\n== Section 2: value-iteration-lazy Q == eager value-iteration Q ==")
(doseq [[label grid goals utils]
        [["corridor" [[:A :empty :empty :empty :B]] [:A :B] {:A 3.0 :B 1.0}]
         ["hike"     [[:A :empty :B] [:empty :empty :empty] [:empty :empty :C]] [:A :B :C] {:A 5.0 :B 2.0 :C -1.0}]]]
  (let [dmdp (diff/build-diff-mdp {:grid grid :goals goals})
        R    (gw/diff-reward (:goal-onehot dmdp) (mx/array (clj->js (mapv utils goals)) mx/float32) TC (:S dmdp) (:A dmdp) (:G dmdp))
        mdp  (assoc dmdp :R R)
        ql   (:Q (agent/value-iteration-lazy mdp 2.0 N))
        qe   (:Q (agent/value-iteration mdp 2.0 N))]
    (assert-true (str label ": lazy Q == eager Q (1e-4)") (< (maxerr (mx/->clj ql) (mx/->clj qe)) 1e-4))))

(println "\n== Section 3: tensor loss == host Σ action-loglik at the plant ==")
;; The differentiable loss at the planted params must equal -Σ inverse/action-loglik
;; over the same observed (s,a) (the inverse posterior's own quantity).
(let [grid  [[:A :empty :empty :empty :B]]
      goals [:A :B]
      utils {:A 3.0 :B 1.0}
      alpha 2.0
      dmdp  (diff/build-diff-mdp {:grid grid :goals goals})
      mdp   (gw/build-mdp {:grid grid :utilities (assoc utils :timeCost TC)})
      hostagent (agent/make-mdp-agent {:mdp mdp :alpha alpha :n-iters N})
      ;; a goal-agent style wrapper so inverse/action-loglik can score it
      obs   [[1 0] [2 1] [3 1] [2 0] [1 0]]
      tensor-loss (diff/loss-at dmdp TC [3.0 1.0] (Math/log alpha) N obs)
      host-loss   (- (reduce + (map (fn [[s a]] (inv/action-loglik {:policy (:policy hostagent)} s a)) obs)))]
  (println "  tensor-loss" (.toFixed tensor-loss 4) " host-loss" (.toFixed host-loss 4))
  (assert-close "tensor loss == -Σ action-loglik at plant (1e-3)" host-loss tensor-loss 1e-3))

(println "\n== Section 4: gradient is finite and non-zero at a wrong init ==")
(let [grid  [[:A :empty :empty :empty :B]]
      goals [:A :B]
      dmdp  (diff/build-diff-mdp {:grid grid :goals goals})
      obs   [[1 0] [2 0] [3 1]]
      states (mapv first obs) actions (mapv second obs)
      raw   (fn [p _k]
              (let [tu (mx/take-idx p (mx/arange 2) 0)
                    la (mx/idx p 2)]
                (diff/action-loglik-loss dmdp tu TC la N states actions)))
      vg    (mx/value-and-grad raw [0])
      [_loss grad] (vg (mx/array (clj->js [0.0 0.0 0.0]) mx/float32) (genmlx.mlx.random/fresh-key 1))
      g     (vec (mx/->clj grad))]
  (println "  grad =" (mapv #(.toFixed % 4) g))
  (assert-true "gradient is all finite" (every? #(js/isFinite %) g))
  (assert-true "gradient is non-zero (backward pass intact)" (some #(> (Math/abs %) 1e-3) g)))

(println "\n== Section 5: recover RELATIVE utilities (alpha fixed) ==")
(let [grid  [[:A :empty :empty :empty :B]]
      goals [:A :B]
      alpha 2.0
      dmdp  (diff/build-diff-mdp {:grid grid :goals goals})
      mdp   (gw/build-mdp {:grid grid :utilities {:A 4.0 :B 0.0 :timeCost TC}})   ; plant A≫B
      data  (policy-dataset mdp alpha 30 N)
      plant-loss (diff/loss-at dmdp TC [4.0 0.0] (Math/log alpha) N data)
      res   (diff/recover-params dmdp TC N data
                                 {:iterations 250 :lr 0.1 :init-utils [0.0 0.0]
                                  :fixed-log-alpha (Math/log alpha)})
      tu    (vec (mx/->clj (:theta-u res)))
      rec-loss (diff/loss-at dmdp TC tu (Math/log alpha) N data)]
  (println "  planted A≫B | recovered theta-u" (mapv #(.toFixed % 3) tu)
           "| plant-loss" (.toFixed plant-loss 3) " recovered-loss" (.toFixed rec-loss 3))
  (assert-true "recovered A-utility > B-utility (correct ordering)" (> (first tu) (second tu)))
  (assert-true "final loss <= plant loss + 1e-2 (likelihood-equivalent)" (<= rec-loss (+ plant-loss 1e-2)))
  (assert-true "loss decreased from init" (< (last (:loss-history res)) (first (:loss-history res)))))

(println "\n== Section 6: recover ALPHA (utilities fixed) ==")
(let [grid  [[:A :empty :empty :empty :B]]
      goals [:A :B]
      plant-alpha 2.0
      dmdp  (diff/build-diff-mdp {:grid grid :goals goals})
      mdp   (gw/build-mdp {:grid grid :utilities {:A 3.0 :B 1.0 :timeCost TC}})
      data  (policy-dataset mdp plant-alpha 50 N)
      ;; fix utilities at the plant via init-utils + (no fixed-log-alpha so alpha learns)
      ;; — but to isolate alpha we hold utils by setting lr low won't fix them; instead
      ;; learn the full vector from a utils-correct init and check alpha lands near 2.
      res   (diff/recover-params dmdp TC N data
                                 {:iterations 300 :lr 0.08 :init-utils [3.0 1.0] :init-log-alpha (Math/log 0.5)})]
  (println "  plant alpha 2.0 | recovered alpha" (.toFixed (:alpha res) 3)
           "| recovered theta-u" (mapv #(.toFixed % 3) (mx/->clj (:theta-u res))))
  (assert-true "recovered alpha is in the right ballpark [1.0, 4.0]" (<= 1.0 (:alpha res) 4.0)))

(println "\n== Section 7: alpha=##Inf is rejected (non-differentiable) ==")
(let [dmdp (diff/build-diff-mdp {:grid [[:A :empty :B]] :goals [:A :B]})
      R    (gw/diff-reward (:goal-onehot dmdp) (mx/array (clj->js [1.0 0.0]) mx/float32) TC (:S dmdp) (:A dmdp) (:G dmdp))]
  (assert-true "value-iteration-lazy rejects alpha=##Inf"
               (try (agent/value-iteration-lazy (assoc dmdp :R R) ##Inf 10) false
                    (catch :default _ true))))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
