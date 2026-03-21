(ns mdp
  "MDP Gridworld — Q-value iteration + inverse planning via exact enumeration.

   5x5 grid, 4 actions (left/right/up/down), 2 goals (corners 0 and 24).
   Deterministic transitions, reward = 1 at goal, -0.1 step cost.

   Phase 1: Bellman backup (tensor DP) computes Q-values Q(s, a, g).
   Phase 2: Inverse planning — observe an agent's action, infer their goal
            using exact enumeration over a softmax-rational agent model."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Grid setup
;; ---------------------------------------------------------------------------

(def H 5)
(def W 5)
(def n-states (* H W))

;; Scalar next-state function using MLX ops (vmap-friendly).
(defn make-next-state-fn
  "Returns a function (fn [s]) that computes the next state for a fixed (dx, dy)."
  [dx dy]
  (let [w  (mx/array W mx/int32)
        lo (mx/array 0 mx/int32)
        hi-x (mx/array (dec W) mx/int32)
        hi-y (mx/array (dec H) mx/int32)
        dx-arr (mx/array dx mx/int32)
        dy-arr (mx/array dy mx/int32)]
    (fn [s]
      (let [x  (mx/remainder s w)
            y  (mx/floor-divide s w)
            xp (mx/clip (mx/add x dx-arr) lo hi-x)
            yp (mx/clip (mx/add y dy-arr) lo hi-y)]
        (mx/add xp (mx/multiply w yp))))))

;; Action deltas: left=[-1,0] right=[1,0] up=[0,-1] down=[0,1]
(def action-deltas [[-1 0] [1 0] [0 -1] [0 1]])

(def states-vec (mx/astype (mx/arange n-states) mx/int32))  ;; [25]

;; For each action, vmap a scalar next-state fn over all states -> [25] per action.
;; Stack to get ns-table [4, 25], transpose to [25, 4].
(def ns-table
  (let [per-action (mapv (fn [[dx dy]]
                           (let [f (make-next-state-fn dx dy)]
                             ((mx/vmap f) states-vec)))
                         action-deltas)]
    (mx/transpose (mx/stack per-action))))  ;; [25, 4]

;; Transition tensor [25, 4, 25] — one-hot via broadcasting:
;; T[s, a, sp] = 1 iff sp == ns-table[s, a]
(def T
  (let [sp-axis (mx/reshape (mx/astype (mx/arange n-states) mx/int32)
                            #js [1 1 n-states])          ;; [1, 1, 25]
        ns-exp  (mx/reshape ns-table #js [n-states 4 1])] ;; [25, 4, 1]
    (mx/eq? sp-axis ns-exp)))                              ;; [25, 4, 25] float32

;; Goals: top-left (0) and bottom-right (24)
(def goals [0 24])
(def goals-arr (mx/array #js [0 24] mx/int32))  ;; [2]

;; Reward [25, 4, 2]: R(s,a,g) = (s==g) - 0.1
;; Broadcast: states [25,1] vs goals [1,2] -> [25,2], expand for actions -> [25,1,2]
(def R
  (let [s-col   (mx/reshape states-vec #js [n-states 1])  ;; [25, 1]
        g-row   (mx/reshape goals-arr  #js [1 2])         ;; [1, 2]
        at-goal (mx/eq? s-col g-row)                       ;; [25, 2] float32
        r-sa    (mx/subtract at-goal (mx/scalar 0.1))]     ;; [25, 2]
    ;; Expand to [25, 4, 2] — reward is the same for all actions
    (mx/broadcast-to (mx/reshape r-sa #js [n-states 1 2])
                     #js [n-states 4 2])))

;; Terminal [25, 2]
(def term
  (let [s-col (mx/reshape states-vec #js [n-states 1])
        g-row (mx/reshape goals-arr  #js [1 2])]
    (mx/eq? s-col g-row)))

(mx/eval! T R term)

;; ---------------------------------------------------------------------------
;; Bellman backup: Q = R + (1-term) * T . V
;; ---------------------------------------------------------------------------

(defn bellman [Q-prev]
  (let [V (mx/reshape (mx/amax Q-prev [1]) #js [n-states 2])
        T-flat (mx/reshape T #js [(* n-states 4) n-states])
        next-V (mx/reshape (mx/matmul T-flat V) #js [n-states 4 2])
        cont (mx/reshape (mx/subtract (mx/scalar 1.0) term) #js [n-states 1 2])]
    (mx/add R (mx/multiply cont next-V))))

;; 7 iterations of value iteration
(def Q
  (loop [Q (mx/zeros #js [n-states 4 2]), t 0]
    (if (>= t 7) Q
      (let [Q-new (bellman Q) _ (mx/eval! Q-new)]
        (recur Q-new (inc t))))))

;; Helper: extract Q[s, :, g] as JS list
(defn q-row [s g]
  (.tolist (mx/idx (mx/idx Q s) g 1)))

;; ---------------------------------------------------------------------------
;; Display Q-values
;; ---------------------------------------------------------------------------

(println "MDP Gridworld (5x5)")
(println "===================")
(println)
(println "Goals: state 0 (top-left) and state 24 (bottom-right)")
(println "Actions: 0=left, 1=right, 2=up, 3=down")
(println)

;; Center state
(println "Q-values at center (state 12):")
(let [q12-g0 (q-row 12 0)
      q12-g1 (q-row 12 1)]
  (println (str "  Goal 0: [L=" (.toFixed (aget q12-g0 0) 2)
                " R=" (.toFixed (aget q12-g0 1) 2)
                " U=" (.toFixed (aget q12-g0 2) 2)
                " D=" (.toFixed (aget q12-g0 3) 2) "]"
                "  (best: left or up)"))
  (println (str "  Goal 1: [L=" (.toFixed (aget q12-g1 0) 2)
                " R=" (.toFixed (aget q12-g1 1) 2)
                " U=" (.toFixed (aget q12-g1 2) 2)
                " D=" (.toFixed (aget q12-g1 3) 2) "]"
                "  (best: right or down)")))

;; Near-goal states
(println)
(println "Q-values near goals:")
(let [q1-g0 (q-row 1 0)
      q23-g1 (q-row 23 1)]
  (println (str "  State 1,  Goal 0: best=left,  Q=" (.toFixed (aget q1-g0 0) 2)))
  (println (str "  State 23, Goal 1: best=right, Q=" (.toFixed (aget q23-g1 1) 2))))

;; V-values at goals
(println)
(println "V-values at goals:")
(let [v0 (mx/item (mx/amax (mx/idx (mx/idx Q 0) 0 1)))
      v24 (mx/item (mx/amax (mx/idx (mx/idx Q 24) 1 1)))]
  (println (str "  V(state 0,  goal 0) = " (.toFixed v0 2) " (at goal: reward 1 - cost 0.1)"))
  (println (str "  V(state 24, goal 1) = " (.toFixed v24 2))))

;; ---------------------------------------------------------------------------
;; Inverse planning via exact enumeration
;; ---------------------------------------------------------------------------

(println)
(println "Inverse planning: observe action -> infer goal")
(println "-----------------------------------------------")

(defn inverse-plan
  "Given state s, compute joint P(goal, action) for a softmax-rational agent."
  [s-val]
  (let [q-sa (mx/idx Q s-val)  ;; [4, 2]
        logits (mx/multiply (mx/scalar 2.0) (mx/transpose q-sa))  ;; [2, 4]
        _ (mx/eval! logits)
        model (gen []
                (let [g (trace :g (dist/weighted (vec (repeat 2 1.0))))
                      a (trace :a (dist/categorical logits))]
                  g))]
    (exact/exact-joint model [] nil)))

;; Center (s=12): observe left -> infer goal
(let [j12 (inverse-plan 12)
      ;; Observe left (action=2 since actions: left=0,right=1,up=2,down=3)
      ;; Wait - in the test, left=2 and right=3. Let's match the test.
      c12-left (exact/condition-on (:log-probs j12) (:axes j12) :a 2)
      p12-left (mx/exp (:log-probs c12-left))
      c12-right (exact/condition-on (:log-probs j12) (:axes j12) :a 3)
      p12-right (mx/exp (:log-probs c12-right))
      _ (mx/eval! p12-left p12-right)]
  (println)
  (println "At center (state 12):")
  (println (str "  Observe up (a=2):   P(goal=0) = " (.toFixed (mx/item (mx/idx p12-left 0)) 4)
                "  P(goal=1) = " (.toFixed (mx/item (mx/idx p12-left 1)) 4)))
  (println (str "  Observe down (a=3): P(goal=0) = " (.toFixed (mx/item (mx/idx p12-right 0)) 4)
                "  P(goal=1) = " (.toFixed (mx/item (mx/idx p12-right 1)) 4)))
  (println "  -> Up favors goal 0 (top-left), down favors goal 1 (bottom-right)"))

;; Near goal 0 (s=1): observe left
(let [j1 (inverse-plan 1)
      c1-left (exact/condition-on (:log-probs j1) (:axes j1) :a 2)
      p1-left (mx/exp (:log-probs c1-left))
      _ (mx/eval! p1-left)]
  (println)
  (println "Near goal 0 (state 1):")
  (println (str "  Observe up (a=2): P(goal=0) = " (.toFixed (mx/item (mx/idx p1-left 0)) 4)
                "  P(goal=1) = " (.toFixed (mx/item (mx/idx p1-left 1)) 4))))

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println)
(println "Verification:")

(let [pass? (atom true)
      close (fn [name expected actual tol]
              (let [ok (< (abs (- expected actual)) tol)]
                (when-not ok (reset! pass? false))
                (println (str "  " (if ok "PASS" "FAIL") ": " name
                              " (expected " (.toFixed expected 4)
                              ", got " (.toFixed actual 4) ")"))))
      check (fn [name ok]
              (when-not ok (reset! pass? false))
              (println (str "  " (if ok "PASS" "FAIL") ": " name)))]
  ;; Q-value shape
  (check "Q shape [25, 4, 2]" (= [25 4 2] (mx/shape Q)))
  ;; Center Q-values
  (let [q12-g0 (q-row 12 0)]
    (close "Q(12,left,g=0)=0.5" 0.5 (aget q12-g0 0) 1e-4)
    (close "Q(12,right,g=0)=0.3" 0.3 (aget q12-g0 1) 1e-4)
    (close "Q(12,up,g=0)=0.5" 0.5 (aget q12-g0 2) 1e-4)
    (close "Q(12,down,g=0)=0.3" 0.3 (aget q12-g0 3) 1e-4))
  ;; Near-goal Q-values
  (let [q1-g0 (q-row 1 0)
        q23-g1 (q-row 23 1)]
    (close "Q(1,left,g=0)=0.8" 0.8 (aget q1-g0 0) 1e-4)
    (close "Q(23,right,g=1)=0.8" 0.8 (aget q23-g1 1) 1e-4))
  ;; V-values at goals
  (close "V(0,g=0)=0.9" 0.9
         (mx/item (mx/amax (mx/idx (mx/idx Q 0) 0 1))) 1e-4)
  (close "V(24,g=1)=0.9" 0.9
         (mx/item (mx/amax (mx/idx (mx/idx Q 24) 1 1))) 1e-4)
  ;; Inverse planning
  (let [j12 (inverse-plan 12)
        c12-left (exact/condition-on (:log-probs j12) (:axes j12) :a 2)
        p12-left (mx/exp (:log-probs c12-left))
        c12-right (exact/condition-on (:log-probs j12) (:axes j12) :a 3)
        p12-right (mx/exp (:log-probs c12-right))
        j1 (inverse-plan 1)
        c1-left (exact/condition-on (:log-probs j1) (:axes j1) :a 2)
        p1-left (mx/exp (:log-probs c1-left))
        _ (mx/eval! p12-left p12-right p1-left)]
    (close "P(g=0|s=12,a=up)" 0.5987 (mx/item (mx/idx p12-left 0)) 1e-3)
    (close "P(g=1|s=12,a=up)" 0.4013 (mx/item (mx/idx p12-left 1)) 1e-3)
    (close "P(g=1|s=12,a=down)" 0.5987 (mx/item (mx/idx p12-right 1)) 1e-3)
    (close "P(g=0|s=1,a=up)" 0.5090 (mx/item (mx/idx p1-left 0)) 1e-3)
    (check "up at center -> g=0 more likely"
           (> (mx/item (mx/idx p12-left 0)) (mx/item (mx/idx p12-left 1))))
    (check "down at center -> g=1 more likely"
           (> (mx/item (mx/idx p12-right 1)) (mx/item (mx/idx p12-right 0)))))
  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
