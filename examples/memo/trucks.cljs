(ns trucks
  "Food Trucks — inverse planning under partial observability.

   A grad student navigates a 4x4 grid to reach food trucks.
   Truck K (Korean) is at (3,3) — always visible.
   Truck X is at (0,0) — hidden behind a wall. Today it has Lebanese food.
   The student can see truck X only from the bottom two rows (y < 2).

   The student has a cuisine preference (Korean, Lebanese, Mexican) and
   plans rationally under partial observability. An observer watches
   the student's actions and infers their cuisine preference using
   exact enumeration and inverse planning (theory of mind).

   Inspired by Baker, Jara-Ettinger, Saxe & Tenenbaum (2017).

   Architecture:
     1. Grid MDP with wall, deterministic transitions
     2. Belief-space value iteration (scalar DP)
     3. Inverse planning via exact enumeration (gen functions)
     4. Sequential Bayesian updating over multi-step trajectories"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact]
            [genmlx.protocols :as p]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Grid: 4x4 with wall
;; =========================================================================
;;
;;   y=3: [ .  S  .  K ]   K = Korean truck (3,3)  S = start (1,3)
;;   y=2: [ .  #  #  # ]   # = wall (x=1..3)
;;   y=1: [ .  .  .  . ]   open column at x=0 goes around wall
;;   y=0: [ X  .  .  . ]   X = hidden truck (0,0)
;;
;;         x=0 x=1 x=2 x=3
;;
;; Path to K: start (1,3) -> right, right = (3,3) = K  [2 steps]
;; Path to X: start (1,3) -> left (0,3) -> down (0,2) -> down (0,1) -> down (0,0) = X  [4 steps]
;; Student can see what's at X only when y < 2.

(def W 4) (def H 4)
(def n-states (* W H))

(def wall? #{[1 2] [2 2] [3 2]})

(defn xy->s [x y] (+ x (* W y)))
(defn s->x [s] (mod s W))
(defn s->y [s] (quot s W))

(def truck-K (xy->s 3 3))  ;; Korean truck — visible, close
(def truck-X (xy->s 0 0))  ;; Hidden truck — far, behind wall
(def start (xy->s 1 3))    ;; Start position

;; Actions: 0=left 1=right 2=up 3=down 4=stay
(def n-actions 5)
(def action-names ["left" "right" "up" "down" "stay"])

(defn next-state [s a]
  (let [x (s->x s) y (s->y s)
        [dx dy] (nth [[-1 0] [1 0] [0 1] [0 -1] [0 0]] a)
        nx (+ x dx) ny (+ y dy)]
    (if (or (< nx 0) (>= nx W) (< ny 0) (>= ny H) (wall? [nx ny]))
      s (xy->s nx ny))))

(defn terminal? [s] (or (= s truck-K) (= s truck-X)))
(defn can-see-X? [s] (< (s->y s) 2))

;; =========================================================================
;; Rewards and parameters
;; =========================================================================

;; Reward at each truck for each cuisine preference
(def reward-K [80.0 -20.0 -20.0])      ;; [Korean-lover, Leb-lover, Mex-lover]
(def reward-X-leb [-20.0 80.0 -20.0])  ;; truck X has Lebanese
(def reward-X-mex [-20.0 -20.0 80.0])  ;; truck X has Mexican

(def step-cost 1.0)
(def gamma 0.9)
(def beta 3.0)

;; Beliefs: discretized P(truck-X = Lebanese)
(def n-beliefs 5)
(def belief-grid [0.0 0.25 0.5 0.75 1.0])

;; =========================================================================
;; Belief update and observation model
;; =========================================================================

(defn belief-update [b s o]
  ;; o: 0=see-Lebanese 1=see-Mexican 2=nothing
  (if-not (can-see-X? s) b
    (case o
      0 (let [n (* 0.99 b)] (/ n (max (+ n (* 0.01 (- 1 b))) 1e-10)))
      1 (let [n (* 0.01 b)] (/ n (max (+ n (* 0.99 (- 1 b))) 1e-10)))
      2 b)))

(defn nearest-belief [b]
  (apply min-key #(abs (- b (nth belief-grid %))) (range n-beliefs)))

(defn obs-prob [b s o]
  (if-not (can-see-X? s)
    (if (= o 2) 1.0 0.0)
    (case o
      0 (+ (* b 0.99) (* (- 1 b) 0.01))
      1 (+ (* b 0.01) (* (- 1 b) 0.99))
      2 0.0)))

;; =========================================================================
;; Value iteration (scalar DP on small grid)
;; =========================================================================

(defn expected-reward [pref bi s]
  (let [b (nth belief-grid bi)]
    (cond
      (= s truck-K) (nth reward-K pref)
      (= s truck-X) (+ (* b (nth reward-X-leb pref))
                        (* (- 1 b) (nth reward-X-mex pref)))
      :else 0.0)))

(defn solve-Q [horizon]
  (let [v-idx (fn [pref bi s] (+ (* pref n-beliefs n-states) (* bi n-states) s))
        qa-idx (fn [pref bi s a]
                 (+ (* pref n-beliefs n-states n-actions)
                    (* bi n-states n-actions) (* s n-actions) a))]
    (loop [V (vec (repeat (* 3 n-beliefs n-states) 0.0)), t 0]
      (let [Q (vec (for [pref (range 3) bi (range n-beliefs)
                         s (range n-states) a (range n-actions)]
                     (if (terminal? s) 0.0
                       (let [sp (next-state s a)
                             r (expected-reward pref bi sp)
                             c (if (= a 4) 0.0 step-cost)
                             ev (reduce + (map (fn [o]
                                                 (let [po (obs-prob (nth belief-grid bi) sp o)
                                                       bi2 (nearest-belief
                                                             (belief-update (nth belief-grid bi) sp o))]
                                                   (* po (nth V (v-idx pref bi2 sp)))))
                                               [0 1 2]))]
                         (+ r (- c) (* gamma ev))))))]
        (if (>= t horizon) Q
          (recur (vec (for [pref (range 3) bi (range n-beliefs) s (range n-states)]
                        (apply max (map #(nth Q (qa-idx pref bi s %)) (range n-actions)))))
                 (inc t)))))))

;; =========================================================================
;; Solve
;; =========================================================================

(println "\n===== Food Trucks POMDP =====")
(println "  4x4 grid, wall at y=2 (x=1..3)")
(println "  Truck K (Korean) at (3,3), always visible")
(println "  Truck X at (0,0), hidden behind wall")
(println "  Start at (1,3)")
(println "  Cuisines: Korean, Lebanese, Mexican\n")

(println "-- Solving POMDP (horizon=10) --")
(def Q-flat (solve-Q 10))

(def cuisine-names {0 "Korean" 1 "Lebanese" 2 "Mexican"})

(defn q-val [pref bi s a]
  (nth Q-flat (+ (* pref n-beliefs n-states n-actions)
                  (* bi n-states n-actions) (* s n-actions) a)))

(defn q-row [pref bi s]
  (mapv #(q-val pref bi s %) (range n-actions)))

(defn best-action [pref bi s]
  (apply max-key #(q-val pref bi s %) (range n-actions)))

(println "\nOptimal actions at start (1,3), belief=0.5:")
(doseq [pref (range 3)]
  (let [ba (best-action pref 2 start)]
    (println (str "  " (cuisine-names pref) ": " (nth action-names ba)
                  "  Q=" (mapv #(.toFixed % 1) (q-row pref 2 start))))))

;; =========================================================================
;; Inverse planning via exact enumeration
;; =========================================================================

(println "\n-- Inverse planning via exact enumeration --")

(defn plan-model
  "Gen function: student chooses action at (belief, state).
   Traces :pref (cuisine) and :action."
  [bi s]
  (let [logits (mx/reshape
                 (mx/array (clj->js (for [p (range 3) a (range n-actions)]
                                      (* beta (q-val p bi s a))))
                           mx/float32)
                 #js [3 5])
        _ (mx/eval! logits)]
    (gen []
      (let [pref (trace :pref (dist/weighted [1.0 1.0 1.0]))
            action (trace :action (dist/categorical (mx/take-idx logits pref 0)))]
        pref))))

(println "\nSingle step at start, belief=0.5:")
(let [model (plan-model 2 start)]
  (doseq [a (range n-actions)]
    (let [p (exact/observes model :action a :pref)
          _ (mx/eval! p)
          max-p (apply max (map #(mx/item (mx/idx p %)) (range 3)))]
      (when (> max-p 0.01)
        (println (str "  " (nth action-names a) ": "
                      (apply str (interpose "  "
                        (map #(str (cuisine-names %) "=" (.toFixed (mx/item (mx/idx p %)) 3))
                             (range 3))))))))))

;; =========================================================================
;; Sequential trajectory inference
;; =========================================================================

(println "\n-- Sequential trajectory inference --")
(println "  Observer watches the student's path and updates P(cuisine).\n")

(defn infer-trajectory
  "Bayesian update over cuisine preference given (state, action) pairs.
   Uniform prior. Each step multiplies by softmax likelihood from Q-values."
  [trajectory]
  (reduce
    (fn [{:keys [probs bi]} [s a]]
      (let [likelihoods (mapv (fn [pref]
                                (let [qr (q-row pref bi s)
                                      logits (mapv #(* beta %) qr)
                                      max-l (apply max logits)
                                      exps (mapv #(js/Math.exp (- % max-l)) logits)
                                      Z (reduce + exps)]
                                  (/ (nth exps a) Z)))
                              (range 3))
            unnormed (mapv * probs likelihoods)
            Z (reduce + unnormed)
            normed (mapv #(/ % Z) unnormed)
            sp (next-state s a)
            bi-new (if (can-see-X? sp) 4 bi)]
        {:probs normed :bi bi-new}))
    {:probs [1/3 1/3 1/3] :bi 2}
    trajectory))

;; Trajectory A: RIGHT, RIGHT -> beeline to Korean truck K
(println "Trajectory A: RIGHT, RIGHT (beeline to K at (3,3))")
(let [{:keys [probs]} (infer-trajectory [[start 1] [(xy->s 2 3) 1]])]
  (doseq [i (range 3)]
    (println (str "  P(" (cuisine-names i) ") = " (.toFixed (nth probs i) 4)))))

;; Trajectory B: LEFT, DOWN -> heading toward X via open column
(println "\nTrajectory B: LEFT, DOWN (heading around wall toward X)")
(let [{:keys [probs]} (infer-trajectory [[start 0] [(xy->s 0 3) 3]])]
  (doseq [i (range 3)]
    (println (str "  P(" (cuisine-names i) ") = " (.toFixed (nth probs i) 4)))))

;; Trajectory C: LEFT then back RIGHT (indecisive)
(println "\nTrajectory C: LEFT, RIGHT (indecisive)")
(let [{:keys [probs]} (infer-trajectory [[start 0] [(xy->s 0 3) 1]])]
  (doseq [i (range 3)]
    (println (str "  P(" (cuisine-names i) ") = " (.toFixed (nth probs i) 4)))))

;; =========================================================================
;; Baker scenario: student goes around wall, discovers X, comes back
;; =========================================================================

(println "\n-- Baker scenario --")
(println "  Student goes around wall toward X, discovers it's Lebanese,")
(println "  then turns back toward K. Inference: Mexican lover!")
(println "  (Going left ruled out Korean; turning back ruled out Lebanese.)\n")

;; Path: (1,3) -L-> (0,3) -D-> (0,2) -D-> (0,1): can now see X!
;;        (0,1) -U-> (0,2): turns around (didn't want Lebanese)
;;        (0,2) -U-> (0,3): heading back toward K

(let [trajectory [[start 0]           ;; (1,3) -> left -> (0,3)
                   [(xy->s 0 3) 3]    ;; (0,3) -> down -> (0,2)
                   [(xy->s 0 2) 3]    ;; (0,2) -> down -> (0,1): sees X is Lebanese
                   [(xy->s 0 1) 2]    ;; (0,1) -> up -> (0,2): turns around!
                   [(xy->s 0 2) 2]    ;; (0,2) -> up -> (0,3)
                   [(xy->s 0 3) 1]]   ;; (0,3) -> right -> (1,3): heading to K
      {:keys [probs]} (infer-trajectory trajectory)]
  (println "After 6-step trajectory (explore X, see Lebanese, return):")
  (doseq [i (range 3)]
    (println (str "  P(" (cuisine-names i) ") = " (.toFixed (nth probs i) 4)))))

;; =========================================================================
;; Theory of mind: exact/thinks + exact/pr
;; =========================================================================

(println "\n-- Theory of mind with exact/pr --")

(def student-model (plan-model 2 start))

(let [p-k-right (exact/pr student-model :pref 0 :given :action 1)
      p-k-left  (exact/pr student-model :pref 0 :given :action 0)
      p-l-right (exact/pr student-model :pref 1 :given :action 1)
      p-l-left  (exact/pr student-model :pref 1 :given :action 0)]
  (println (str "  P(Korean  | right) = " (.toFixed p-k-right 4)
               "  (toward K)"))
  (println (str "  P(Korean  | left)  = " (.toFixed p-k-left 4)
               "  (toward X)"))
  (println (str "  P(Lebanese| right) = " (.toFixed p-l-right 4)))
  (println (str "  P(Lebanese| left)  = " (.toFixed p-l-left 4))))

;; exact/thinks: observer splices student model via exact enumeration
(println "\n-- exact/thinks: observer reasons about student --")
(let [obs-model (vary-meta
                  (gen []
                    (let [probs (splice :student (exact/thinks student-model) [])]
                      probs))
                  assoc :genmlx.dynamic/key (rng/fresh-key))
      tr (p/simulate obs-model [])
      retval (:retval tr)]
  (mx/eval! retval)
  (println (str "  enumerate retval shape: " (vec (mx/shape retval))))
  (println "  (Joint probability table over student's discrete choices)"))

;; =========================================================================
;; Verification
;; =========================================================================

(println "\n-- Verification --")

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn check [name pred]
  (if pred
    (do (swap! pass-count inc) (println (str "  PASS: " name)))
    (do (swap! fail-count inc) (println (str "  FAIL: " name)))))

;; 1. Korean lover goes RIGHT at start (toward K at (3,3))
(check "Korean lover: best action is RIGHT"
       (= 1 (best-action 0 2 start)))

;; 2. Q=0 at terminal states
(check "Q=0 at truck K" (< (abs (q-val 0 2 truck-K 0)) 0.01))
(check "Q=0 at truck X" (< (abs (q-val 0 2 truck-X 0)) 0.01))

;; 3. Korean: Q(right) > Q(left) at start
(check "Korean: Q(right) > Q(left)"
       (> (q-val 0 2 start 1) (q-val 0 2 start 0)))

;; 4. Lebanese lover (certain X=Lebanese) prefers LEFT (toward X via open column)
(let [q-leb-left (q-val 1 4 start 0)
      q-leb-right (q-val 1 4 start 1)]
  (check "Lebanese (b=1.0): prefers LEFT over RIGHT"
         (> q-leb-left q-leb-right)))

;; 5. Belief update
(check "belief: see Lebanese -> b > 0.9" (> (belief-update 0.5 0 0) 0.9))
(check "belief: see Mexican -> b < 0.1" (< (belief-update 0.5 0 1) 0.1))
(check "belief: no observation -> unchanged" (= (belief-update 0.5 8 2) 0.5))

;; 6. Inverse planning: right -> Korean, left -> X-cuisine
(let [p-k-right (exact/pr student-model :pref 0 :given :action 1)
      p-k-left  (exact/pr student-model :pref 0 :given :action 0)]
  (check "P(Korean | right) > P(Korean | left)" (> p-k-right p-k-left))
  (check "P(Korean | right) > 1/3" (> p-k-right (/ 1.0 3.0))))

;; 7. Sequential: RIGHT, RIGHT -> Korean most likely
(let [{:keys [probs]} (infer-trajectory [[start 1] [(xy->s 2 3) 1]])]
  (check "trajectory RIGHT,RIGHT: Korean most likely"
         (and (> (nth probs 0) (nth probs 1))
              (> (nth probs 0) (nth probs 2)))))

;; 8. Sequential: LEFT, DOWN -> Korean NOT most likely
(let [{:keys [probs]} (infer-trajectory [[start 0] [(xy->s 0 3) 3]])]
  (check "trajectory LEFT,DOWN: Korean NOT most likely"
         (< (nth probs 0) (+ (nth probs 1) (nth probs 2)))))

;; 9. Baker scenario: explore X, see Lebanese, return -> Mexican
;;    Going left eliminated Korean; seeing Lebanese and turning back eliminated Lebanese.
(let [trajectory [[start 0] [(xy->s 0 3) 3] [(xy->s 0 2) 3]
                   [(xy->s 0 1) 2] [(xy->s 0 2) 2] [(xy->s 0 3) 1]]
      {:keys [probs]} (infer-trajectory trajectory)]
  (check "Baker: explore-and-return -> Mexican most likely"
         (and (> (nth probs 2) (nth probs 0))
              (> (nth probs 2) (nth probs 1)))))

;; 10. exact/thinks produces valid trace
(let [obs-model (vary-meta
                  (gen [] (splice :student (exact/thinks student-model) []))
                  assoc :genmlx.dynamic/key (rng/fresh-key))
      tr (p/simulate obs-model [])]
  (mx/eval! (:retval tr) (:score tr))
  (check "exact/thinks: simulate produces trace" (some? (:retval tr)))
  (check "exact/thinks: score is finite" (js/isFinite (mx/item (:score tr)))))

;; Summary
(println (str "\n===== Results: " @pass-count " passed, " @fail-count " failed ====="))
(when (pos? @fail-count) (js/process.exit 1))
