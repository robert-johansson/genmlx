;; @tier slow
;; Headless tests for the canonical opening worlds (genmlx.agents.worlds):
;; the Ch 3a integer line-world and the Ch 3b Hiking gridworld. Pure data +
;; asserts (no Ink/terminal). The hiking world reproduces agentmodels' canonical
;; noise-induced detour: the DETERMINISTIC optimal agent cuts along the
;; cliff-adjacent bottom row, while the STOCHASTIC (orthogonal-slip) agent detours
;; along the top, away from the Hill — both reach the East peak. Value iteration
;; is RNG-free, so the routes below are reproducible.
;; Run: bun run --bun nbb test/genmlx/agentmodels_worlds_test.cljs

(ns genmlx.agentmodels-worlds-test
  (:require [genmlx.agents.worlds :as w]
            [genmlx.agents.agent :as agent]
            [genmlx.mlx :as mx]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))

(defn- vvec [V n] (mapv #(mx/item (mx/idx V %)) (range n)))
(defn- greedy-path
  "Follow the agent's argmax-Q action under deterministic geometry (ns-fn) from
   `start` to a terminal — the agent's INTENDED route, independent of env noise."
  [ag start]
  (let [Qm (mx/->clj (:Q ag)) A (:A (:mdp ag)) nsf (:ns-fn (:mdp ag))
        terms (set (keys (:terminals (:mdp ag))))]
    (loop [s start path [start] n 0]
      (if (or (terms s) (> n 30)) path
          (let [a (apply max-key #(nth (nth Qm s) %) (range A)) s' (nsf s a)]
            (recur s' (conj path s') (inc n)))))))

;; --- Ch 3a: integer line-world ----------------------------------------------
(println "\n-- Ch 3a: integer line-world --")
(let [m  (w/line-mdp {})
      ag (agent/make-mdp-agent {:mdp m :alpha ##Inf :gamma 1.0 :n-iters 10})
      {:keys [states actions]} (agent/simulate-mdp ag (:start-idx m) 10)
      V  (vvec (:V ag) (:S m))]
  (assert-equal "line S = 7"               7 (:S m))
  (assert-equal "line terminal {6 :goal}"  {6 :goal} (:terminals m))
  (assert-equal "line start-idx = 0"       0 (:start-idx m))
  (assert-equal "line rollout walks 0 -> 6" [0 1 2 3 4 5 6] states)
  (assert-true  "line rollout is all :right" (every? #{:right} (mapv (:action-kw m) actions)))
  (assert-equal "line rollout ends at goal" 6 (last states))
  (assert-true  "line V is maximal at the goal (idx 6)" (= 6 (apply max-key #(nth V %) (range 7)))))

;; --- Ch 3b: Hiking gridworld — structure + deterministic optimal ------------
(println "\n-- Ch 3b: hiking gridworld (deterministic) --")
(def hike-det (w/hike-mdp {:noise 0.0}))
(let [m hike-det]
  (assert-equal "hike S = 25"            25 (:S m))
  (assert-equal "hike walls {6 11 13}"   #{6 11 13} (:walls m))
  (assert-equal "hike terminals (W=12 E=14 Hill=20..24)"
                {12 :west 14 :east 20 :hill 21 :hill 22 :hill 23 :hill 24 :hill} (:terminals m))
  (assert-equal "hike start-idx = 5 ([0,1])" 5 (:start-idx m))
  (assert-equal "hike T shape [25 4 25]" [25 4 25] (mx/shape (:T m)))
  (assert-true  "hike T rows each sum to 1 (total 100)"
                (< (Math/abs (- 100.0 (mx/item (mx/sum (:T m))))) 1e-3)))

(def ag-det (agent/make-mdp-agent {:mdp hike-det :alpha ##Inf :gamma 1.0 :n-iters 15}))
(let [V (vvec (:V ag-det) 25)
      path (greedy-path ag-det (:start-idx hike-det))
      hills #{20 21 22 23 24}]
  (assert-true  "hike det: V is maximal at East (idx 14)" (= 14 (apply max-key #(nth V %) (range 25))))
  (assert-equal "hike det: greedy route reaches East" 14 (last path))
  (assert-true  "hike det: route never steps on a Hill" (not (some hills path))))

;; --- Ch 3b: the noise-induced detour (the canonical demonstration) ----------
(println "\n-- Ch 3b: noise-induced detour --")
(def hike-noisy (w/hike-mdp {:noise 0.1}))
(def ag-noisy (agent/make-mdp-agent {:mdp hike-noisy :alpha ##Inf :gamma 1.0 :n-iters 15}))
(let [vd (vvec (:V ag-det) 25)
      vn (vvec (:V ag-noisy) 25)
      cliff-adjacent #{15 16 17 18 19}]   ; row 3, directly above the Hill cliff
  (assert-true "cliff cells worth less under noise (V_noisy < V_det across row 3)"
               (every? (fn [i] (< (nth vn i) (nth vd i))) cliff-adjacent))
  (assert-true "the value drop is large mid-cliff (idx 16,17 lose > 1.0)"
               (and (> (- (nth vd 16) (nth vn 16)) 1.0)
                    (> (- (nth vd 17) (nth vn 17)) 1.0))))

(let [row3 #{15 16 17 18 19}
      pd (greedy-path ag-det (:start-idx hike-det))      ; cuts along the cliff
      pn (greedy-path ag-noisy (:start-idx hike-noisy))   ; detours along the top
      cliff-d (count (filter row3 pd))
      cliff-n (count (filter row3 pn))]
  (println "   det route  :" pd "  cliff-adjacent cells:" cliff-d)
  (println "   noisy route:" pn "  cliff-adjacent cells:" cliff-n)
  (assert-true  "deterministic route hugs the cliff (uses row-3 cells)" (pos? cliff-d))
  (assert-equal "noisy route avoids the cliff entirely (0 row-3 cells)" 0 cliff-n)
  (assert-true  "noisy agent DETOURS: strictly fewer cliff-adjacent cells" (< cliff-n cliff-d))
  (assert-equal "noisy route still reaches East" 14 (last pn)))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
