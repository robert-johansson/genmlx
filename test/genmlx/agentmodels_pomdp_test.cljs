;; @tier slow
;; Headless tests for the POMDP agent (agentmodels Ch 3c) — belief filtering +
;; belief-space (QMDP) action on the hidden-goal restaurant gridworld. Pure data,
;; no terminal. The belief filter and belief-Q are pure; the convergence rollout
;; is made deterministic by using the alpha = Inf (argmax) agent on a noise-0 env.
;;
;; Run: bun run --bun nbb test/genmlx/agentmodels_pomdp_test.cljs

(ns genmlx.agentmodels-pomdp-test
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as env]
            [genmlx.agents.presentation :as pres]
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
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))

;; A corridor world: the only path from the start (bottom-centre, idx 13) to the
;; two goals (A top-left idx 0, B top-right idx 2) runs UP through the signpost
;; (idx 7), so the agent must pass it and learn the latent before the fork.
;;   A . B
;;   # . #
;;   # P #     P = signpost (idx 7)
;;   # . #
;;   # @ #     start (idx 13)
(def grid [[:A    :empty :B]
           [:wall :empty :wall]
           [:wall :empty :wall]
           [:wall :empty :wall]
           [:wall :empty :wall]])
(def signpost 7)
(def goals [:A :B])
(defn build [true-world alpha]
  (let [e  (env/restaurant-gridworld {:grid grid :goals goals :signpost signpost
                                      :true-world true-world :start [1 4]})
        pa (pomdp/make-pomdp-agent (assoc e :alpha alpha :gamma 1.0 :n-iters 40))]
    [e pa]))

(println "\n== Section 1: Bayesian belief filter (pure) ==")
(let [[_ pa] (build :A 2.0)
      ub     (:update-belief pa)
      prior  (:prior pa)]
  (assert-close "obs=nil leaves belief unchanged (:A)" 0.5 (:A (ub prior 10 nil)) 1e-9)
  (let [snap-a (ub prior signpost :A)]
    (assert-close "reveal :A at signpost -> P(:A)=1" 1.0 (:A snap-a) 1e-9)
    (assert-close "reveal :A at signpost -> P(:B)=0" 0.0 (:B snap-a) 1e-9))
  (let [snap-b (ub prior signpost :B)]
    (assert-close "reveal :B at signpost -> P(:B)=1" 1.0 (:B snap-b) 1e-9))
  (assert-close "every filtered belief is normalized"
                1.0 (reduce + (vals (ub prior signpost :A))) 1e-9))

(println "\n== Section 2: belief converges + QMDP reaches the right goal ==")
(doseq [[tw goal-idx] [[:A 0] [:B 2]]]
  (let [[e pa] (build tw ##Inf)            ; alpha=Inf + noise 0 => deterministic rollout
        roll   (pomdp/simulate-pomdp pa e (:start-idx e) 12)
        {:keys [states beliefs]} roll]
    (println (str "  true world " (name tw) " | path " states
                  " | P(" (name tw) ") " (mapv #(.toFixed (get % tw) 2) beliefs)))
    (assert-equal (str "true world " (name tw) ": reaches its goal idx") goal-idx (last states))
    (assert-equal "#beliefs == #states" (count states) (count beliefs))
    (assert-close "belief flat (prior) at the start"      0.5 (get (first beliefs) tw) 1e-9)
    (assert-close "belief still flat one step before signpost" 0.5 (get (nth beliefs 1) tw) 1e-9)
    (assert-close "belief snaps to truth at the signpost" 1.0 (get (nth beliefs 2) tw) 1e-9)
    (assert-true  "belief is monotone toward the truth"
                  (>= (get (last beliefs) tw) (get (first beliefs) tw)))))

(println "\n== Section 3: belief-space Q is a genuine QMDP mixture (pure) ==")
(let [[_ pa] (build :A 2.0)
      wa     (:world-agents pa)
      bq     (:belief-Q pa)
      s      1                                  ; the fork
      mixed  (vec (mx/->clj (bq {:A 0.5 :B 0.5} s)))
      qa     (vec (mx/->clj (mx/idx (:Q (wa :A)) s)))
      qb     (vec (mx/->clj (mx/idx (:Q (wa :B)) s)))]
  (assert-true "belief-Q(0.5/0.5) == 0.5*Q_A + 0.5*Q_B elementwise"
               (every? (fn [[m a b]] (< (Math/abs (- m (+ (* 0.5 a) (* 0.5 b)))) 1e-4))
                       (map vector mixed qa qb))))

(println "\n== Section 4: point-mass collapse to the underlying MDP (anchor) ==")
(let [[_ pa] (build :A ##Inf)
      wa     (:world-agents pa)
      bq     (:belief-Q pa)
      s      1
      pm     (vec (mx/->clj (bq {:A 1.0 :B 0.0} s)))
      qa     (vec (mx/->clj (mx/idx (:Q (wa :A)) s)))]
  (assert-true "point-mass belief-Q == Q_A exactly"
               (every? (fn [[m a]] (< (Math/abs (- m a)) 1e-6)) (map vector pm qa)))
  (assert-equal "POMDP act under certain belief == the MDP agent's action"
                ((:act (wa :A)) s) ((:act pa) {:A 1.0 :B 0.0} s)))

(println "\n== Section 4b: tensorized belief-Q == host reduce over a belief sweep (genmlx-4ifp) ==")
;; belief-Q is now a [W,S,A] Qstack contraction; assert it agrees with the old
;; host reduce (Σ_w b(w)·Q_w[s]) to 1e-5 across a sweep of beliefs and states.
(let [[_ pa] (build :A 2.0)
      wa     (:world-agents pa)
      bq     (:belief-Q pa)
      A      (:A (:mdp (val (first wa))))
      host-bq (fn [belief s]
                (reduce (fn [acc [w b]]
                          (mx/add acc (mx/multiply (mx/scalar b) (mx/idx (:Q (wa w)) s))))
                        (mx/zeros #js [A]) belief))
      beliefs [{:A 1.0 :B 0.0} {:A 0.0 :B 1.0} {:A 0.5 :B 0.5} {:A 0.7 :B 0.3} {:A 0.1 :B 0.9}]]
  (assert-true "tensor belief-Q == host reduce to 1e-5 over beliefs×states"
               (every? identity
                 (for [b beliefs, s [0 1 4 7 10]]
                   (every? (fn [[t h]] (< (Math/abs (- t h)) 1e-5))
                           (map vector (mx/->clj (bq b s)) (mx/->clj (host-bq b s))))))))

(println "\n== Section 5: the seam — belief -> PosteriorBars ==")
(let [bars (pres/dist->bars "P(world)" {:A 1.0 :B 0.0} :A)]
  (assert-close "bars sum to 1" 1.0 (reduce + (map :weight (:bars bars))) 1e-9)
  (assert-true  "true world bar is highlighted"
                (:highlight (first (filter #(= "A" (:label %)) (:bars bars))))))

(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
