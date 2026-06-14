(ns agentmodels.ch08-library-guide
  "agentmodels.org Chapter 8 — \"WebPPL agents library\" ported to GenMLX as a
   guided tour of the genmlx.agents foundation. Ch8 in the textbook is a quick-
   start over the reusable env/agent surface; this file is the same tour over the
   GenMLX library spine (src/genmlx/agents/*), showing that the constructors
   compose and that you can drop in your OWN policy without leaving the GFI.

   The tour:
   1. make-line-mdp        — worlds/line-mdp: a 1-D corridor MDP.
   2. make-mdp-agent       — agent/make-mdp-agent, OPTIMAL (alpha = ##Inf, hard
                             argmax) and SOFT-RATIONAL (finite alpha, Boltzmann).
   3. make-gridworld-mdp   — worlds/hike-mdp: the 5×5 hiking gridworld; an agent
                             plans a route to the high peak.
   4. hyperbolic discounting— biased_planners/make-biased-mdp-agent: a present-
                             biased (Naive hyperbolic) planner, the genuine
                             time-inconsistent agent, alongside the optimal one.
   5. make-line-pomdp      — pomdp_env/restaurant-gridworld on a corridor +
                             pomdp/make-pomdp-agent: belief filtering over a hidden
                             goal (manifest action trajectory ⟂ latent world); the
                             belief snaps to truth at the signpost (QMDP).
   6. custom agents        — a RANDOM agent (uniform-draw policy) and an
                             EPSILON-GREEDY agent (a mix of a uniform explore branch
                             and exact/categorical-argmax exploitation) — both are
                             ordinary policy generative functions.

   Reuse, zero engine change: genmlx.agents.{worlds,agent,gridworld,biased-planners,
   pomdp,pomdp-env,helpers}, genmlx.inference.exact (categorical-argmax), the GFI.
   Self-checking: run

     bun run --bun nbb examples/agentmodels/ch08_library_guide.cljs

   prints each demo's output and asserts the headline behaviors; exits non-zero on
   any failure. Deterministic behaviors (optimal first action, POMDP belief snap,
   epsilon=0 greedy) are asserted by value; stochastic ones structurally."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.exact :as exact]
            [genmlx.protocols :as p]
            [genmlx.gen :refer [gen]]
            [genmlx.agents.worlds :as worlds]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as env]
            [genmlx.agents.helpers :as h]))

;; ---------------------------------------------------------------------------
;; tiny self-check harness
;; ---------------------------------------------------------------------------

(def ^:private fails (atom 0))

(defn- ->num [v] (if (mx/array? v) (mx/item v) v))

(defn- check-true [label ok]
  (println (str (if ok "  ✓ " "  ✗ FAIL ") label))
  (when-not ok (swap! fails inc))
  ok)

(defn- check-equal [label expected actual]
  (let [ok (= expected actual)]
    (println (str (if ok "  ✓ " "  ✗ FAIL ") label " — expected " expected ", got " actual))
    (when-not ok (swap! fails inc))
    ok))

(defn- check-close [label expected actual tol]
  (let [ok (<= (abs (- expected actual)) tol)]
    (println (str (if ok "  ✓ " "  ✗ FAIL ") label " — expected " expected ", got " actual))
    (when-not ok (swap! fails inc))
    ok))

;; gridworld action indices (from genmlx.agents.gridworld): [:left :right :up :down]
(def RIGHT 1)

;; ---------------------------------------------------------------------------
;; 1 + 2. line MDP + make-mdp-agent (optimal and soft-rational)
;; ---------------------------------------------------------------------------

(defn demo-line-mdp []
  (println "\n-- 1+2. make-line-mdp + make-mdp-agent (optimal vs soft) --")
  (let [mdp        (worlds/line-mdp {:n 7 :reward 1.0 :time-cost -0.1})
        start      (:start-idx mdp)
        optimal    (agent/make-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24})
        soft       (agent/make-mdp-agent {:mdp mdp :alpha 3.0   :gamma 1.0 :n-iters 24})
        a-opt      ((:act optimal) start)
        roll       (agent/simulate-mdp optimal start 12)]
    (println (str "  start=" start "  optimal first action=" a-opt
                  " (1=right)  optimal path states=" (:states roll)))
    (check-equal "optimal agent steps RIGHT toward the goal first" RIGHT a-opt)
    (check-equal "optimal rollout ends at the goal cell (idx 6)" 6 (last (:states roll)))
    ;; soft-rational agent is a valid policy; its EU favors the goal direction
    (let [eu-row (mapv (fn [a] (->num ((:expected-utility soft) start a))) (range 4))]
      (println (str "  soft EU(start, ·) = " (mapv #(.toFixed % 3) eu-row)))
      (check-true "soft agent: EU(right) >= EU(left)" (>= (nth eu-row 1) (nth eu-row 0))))))

;; ---------------------------------------------------------------------------
;; 3. gridworld MDP — the hiking world; plan a route to the high (East) peak
;; ---------------------------------------------------------------------------

(defn demo-gridworld-mdp []
  (println "\n-- 3. make-gridworld-mdp (hiking 5×5) + planned route --")
  (let [mdp     (worlds/hike-mdp {:noise 0.0})
        start   (:start-idx mdp)
        ag      (agent/make-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24})
        roll    (agent/simulate-mdp ag start 16)
        end     (last (:states roll))
        east-idx 14] ; the prize peak (utility 10)
    (println (str "  start=" start "  path=" (:states roll) "  ends at idx " end))
    (check-equal "optimal hiker reaches the East peak (idx 14)" east-idx end)))

;; ---------------------------------------------------------------------------
;; 4. hyperbolic discounting — a present-biased (Naive) planner
;; ---------------------------------------------------------------------------

(defn demo-hyperbolic []
  (println "\n-- 4. make-biased-mdp-agent (Naive hyperbolic) vs optimal --")
  (let [mdp      (worlds/line-mdp {:n 7 :reward 1.0 :time-cost -0.1})
        start    (:start-idx mdp)
        optimal  (agent/make-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24})
        ;; k=0 reproduces the optimal plan; k>0 is genuinely present-biased
        naive-k0 (bp/make-biased-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24}
                                           {:discount 0.0 :bias :naive})
        naive-k1 (bp/make-biased-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24}
                                           {:discount 1.0 :bias :naive})
        a-opt    ((:act optimal) start)
        a-k0     ((:act naive-k0) start)
        a-k1     ((:act naive-k1) start)]
    (println (str "  optimal first action=" a-opt
                  "  naive(k=0)=" a-k0 "  naive(k=1)=" a-k1))
    (check-equal "biased agent with k=0 matches the optimal first action" a-opt a-k0)
    (check-true "naive hyperbolic agent yields a valid action (0..3)"
                (contains? #{0 1 2 3} a-k1))
    (check-true "make-biased-mdp-agent exposes a :policy GF + :expected-utility"
                (and (some? (:policy naive-k1)) (fn? (:expected-utility naive-k1))))))

;; ---------------------------------------------------------------------------
;; 5. line POMDP — belief filtering over a hidden goal (manifest ⟂ latent)
;; ---------------------------------------------------------------------------
;;
;; A corridor whose only route to the two goals (A left, B right) passes the
;; signpost (P), which reveals the latent world. The MANIFEST is the action
;; trajectory; the LATENT is which goal is real. QMDP plans on the belief; the
;; belief snaps to truth once the signpost is observed.

(def pomdp-grid
  [[:A    :empty :B]
   [:wall :empty :wall]
   [:wall :empty :wall]
   [:wall :empty :wall]
   [:wall :empty :wall]])
(def pomdp-signpost 7)

(defn demo-line-pomdp []
  (println "\n-- 5. make-line-pomdp + make-pomdp-agent (belief filtering) --")
  (doseq [[tw goal-idx] [[:A 0] [:B 2]]]
    (let [e    (env/restaurant-gridworld {:grid pomdp-grid :goals [:A :B]
                                          :signpost pomdp-signpost
                                          :true-world tw :start [1 4]})
          pa   (pomdp/make-pomdp-agent (assoc e :alpha ##Inf :gamma 1.0 :n-iters 40))
          roll (pomdp/simulate-pomdp pa e (:start-idx e) 12)
          {:keys [states beliefs]} roll]
      (println (str "  true world " (name tw) " | path " states
                    " | P(" (name tw) ")=" (mapv #(.toFixed (get % tw) 2) beliefs)))
      (check-close (str (name tw) ": latent prior is flat at the start") 0.5
                   (get (first beliefs) tw) 1e-9)
      (check-close (str (name tw) ": belief snaps to truth at the signpost") 1.0
                   (get (nth beliefs 2) tw) 1e-9)
      (check-equal (str (name tw) ": QMDP agent reaches the true goal") goal-idx
                   (last states)))))

;; ---------------------------------------------------------------------------
;; 6. custom agents — drop-in policies that are ordinary generative functions
;; ---------------------------------------------------------------------------

(defn random-policy
  "A custom RANDOM agent: act uniformly at random over `n-actions`, via the
   value-carrying uniform-draw helper."
  [n-actions]
  (gen [_s] (trace :action (:dist (h/uniform-draw (vec (range n-actions)))))))

(defn epsilon-greedy-policy
  "A custom EPSILON-GREEDY agent: with prob epsilon explore uniformly, else exploit
   the greedy argmax of the Q-row via exact/categorical-argmax. `q-rows` is the
   host-side [S][A] value table (mx/->clj of an agent's :Q)."
  [q-rows n-actions epsilon]
  (gen [s]
    (let [explore (trace :explore (dist/flip epsilon))]
      (trace :action
             (if (pos? (int (mx/item explore)))
               (:dist (h/uniform-draw (vec (range n-actions))))
               (exact/categorical-argmax (mx/array (nth q-rows s) mx/float32)))))))

(defn- policy-act [policy s]
  (int (mx/item (:retval (p/simulate (dyn/auto-key policy) [s])))))

(defn demo-custom-agents []
  (println "\n-- 6. custom agents: random + epsilon-greedy --")
  (let [mdp     (worlds/line-mdp {:n 7 :reward 1.0 :time-cost -0.1})
        start   (:start-idx mdp)
        n-act   4
        optimal (agent/make-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24})
        q-rows  (vec (map vec (mx/->clj (:Q optimal))))
        greedy-a (let [row (nth q-rows start)]
                   (first (apply max-key second (map-indexed vector row))))
        ;; random agent: 40 draws, all in range
        rnd      (random-policy n-act)
        rnd-acts (mapv (fn [_] (policy-act rnd start)) (range 40))
        ;; epsilon-greedy with epsilon=0 is pure greedy (deterministic)
        eg0      (epsilon-greedy-policy q-rows n-act 0.0)
        eg0-acts (mapv (fn [_] (policy-act eg0 start)) (range 10))
        ;; epsilon-greedy with epsilon=1 is pure exploration
        eg1      (epsilon-greedy-policy q-rows n-act 1.0)
        eg1-acts (set (mapv (fn [_] (policy-act eg1 start)) (range 60)))]
    (println (str "  random agent actions (sample): " (take 8 rnd-acts)))
    (check-true "random agent: every action is a legal index 0..3"
                (every? #(contains? #{0 1 2 3} %) rnd-acts))
    (println (str "  greedy argmax action at start = " greedy-a))
    (check-true "epsilon-greedy(0.0) always takes the greedy argmax action"
                (every? #(= greedy-a %) eg0-acts))
    (println (str "  epsilon-greedy(1.0) explored actions = " (sort eg1-acts)))
    (check-true "epsilon-greedy(1.0) explores more than one action"
                (> (count eg1-acts) 1))))

;; ---------------------------------------------------------------------------
;; run + self-check
;; ---------------------------------------------------------------------------

(defn -main []
  (println "\n=== agentmodels.org Ch 8 — genmlx.agents library quick-start ===")
  (demo-line-mdp)
  (demo-gridworld-mdp)
  (demo-hyperbolic)
  (demo-line-pomdp)
  (demo-custom-agents)
  (println (str "\n" (if (zero? @fails)
                       "ALL CHECKS PASSED ✓"
                       (str @fails " CHECK(S) FAILED ✗"))))
  (when (pos? @fails) (js/process.exit 1)))

(-main)
