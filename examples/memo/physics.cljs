(ns physics
  "Intuitive psychology meets intuitive physics.

   A baby observes an agent launch itself toward a goal. The agent
   may have to arc over a wall to reach it. From the observed
   trajectory (angle + impulse), the baby infers whether a wall
   was present — using theory of mind about the agent's rationality.

   Inspired by Csibra et al. (2003) and Liu, Outa & Akbiyik (2024).

   Architecture (100% GenMLX):
     projectile-final-x — differentiable MLX physics (Layer 0)
     agent-utility      — reward - cost (Layer 0)
     agent-model        — gen fn, traces :wall + :action (Layer 3)
     baby-inference     — exact enumeration + condition-on (Layer 6)
     baby-inference-gpu — vectorized GPU version for benchmarks

   The agent model IS a gen function: :wall and :action are traced,
   physics determines the utility, exact enumeration gives the joint.
   Conditioning on observed action yields P(wall) — the baby's inference."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Verification helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- check [name pred]
  (if pred
    (do (swap! pass-count inc) (println (str "  PASS: " name)))
    (do (swap! fail-count inc) (println (str "  FAIL: " name)))))

(defn- check-close [name expected actual tol]
  (if (<= (abs (- expected actual)) tol)
    (do (swap! pass-count inc) (println (str "  PASS: " name)))
    (do (swap! fail-count inc) (println (str "  FAIL: " name " (expected " expected " got " actual ")")))))

;; ---------------------------------------------------------------------------
;; Physics parameters
;; ---------------------------------------------------------------------------

(def gravity 0.5)
(def wall-x 1.0)
(def wall-height 0.5)
(def goal-x 2.0)

;; ---------------------------------------------------------------------------
;; Layer 0: Differentiable projectile physics
;; ---------------------------------------------------------------------------

(defn projectile-final-x
  "Final x position of a projectile launched at (theta, iota).
   If wall is present and trajectory passes through at y < wall-height,
   the ball stops. All MLX operations — differentiable and GPU-native."
  [theta iota wall]
  (let [vx (mx/multiply iota (mx/cos theta))
        vy (mx/multiply iota (mx/sin theta))
        t-land (mx/maximum (mx/divide (mx/multiply (mx/scalar 2.0) vy)
                                      (mx/scalar gravity))
                           (mx/scalar 0.01))
        x-free (mx/multiply vx t-land)
        t-wall (mx/divide (mx/scalar wall-x) (mx/maximum vx (mx/scalar 0.001)))
        y-at-wall (mx/subtract (mx/multiply vy t-wall)
                    (mx/multiply (mx/scalar (* 0.5 gravity))
                                 (mx/multiply t-wall t-wall)))
        hits (mx/multiply wall
               (mx/and* (mx/lt? t-wall t-land)
                        (mx/lt? y-at-wall (mx/scalar wall-height))))]
    (mx/where (mx/gt? hits 0.5)
              (mx/scalar wall-x)
              x-free)))

(defn agent-utility
  "Agent's utility for a trajectory. Reward for reaching the goal,
   cost proportional to impulse and angle (arcing is costly)."
  [theta iota wall weight]
  (let [xf (projectile-final-x theta iota wall)
        reward (mx/gt? xf (mx/scalar goal-x))
        cost (mx/multiply iota (mx/add (mx/scalar 1.0)
                                       (mx/multiply (mx/scalar 0.5) theta)))]
    (mx/subtract (mx/multiply weight reward) cost)))

;; ---------------------------------------------------------------------------
;; Layer 3: Agent model — gen function with physics-based rationality
;; ---------------------------------------------------------------------------

(defn agent-model
  "Rational agent: chooses trajectory to maximize utility.
   Wall presence is traced (hidden from the baby).
   Action logits come from physics simulation.

   This IS a gen function: :wall and :action are traced choices.
   Exact enumeration gives the joint P(wall, action).
   Conditioning on observed action gives P(wall | action).

   Parameters:
     angle-values   — MLX array of discretized angles
     impulse-values — MLX array of discretized impulses
     beta           — agent rationality (softmax temperature)
     weight         — goal reward weight"
  [angle-values impulse-values beta weight]
  (let [n-angles (first (mx/shape angle-values))
        n-impulses (first (mx/shape impulse-values))
        ;; Precompute utility for ALL (wall, angle, impulse) via broadcast
        th (mx/reshape angle-values #js [1 n-angles 1])
        io (mx/reshape impulse-values #js [1 1 n-impulses])
        w  (mx/reshape (mx/array #js [0.0 1.0]) #js [2 1 1])
        ;; Physics: [2, n-angles, n-impulses] — all trajectories at once
        xf (projectile-final-x th io w)
        reward (mx/gt? xf (mx/scalar goal-x))
        cost (mx/multiply io (mx/add (mx/scalar 1.0)
                                     (mx/multiply (mx/scalar 0.5) th)))
        utility (mx/subtract (mx/multiply (mx/scalar weight) reward) cost)
        ;; Flatten to [wall=2, n-actions] as categorical logits
        action-logits (mx/multiply (mx/scalar beta)
                        (mx/reshape utility #js [2 (* n-angles n-impulses)]))
        _ (mx/eval! action-logits)]
    (gen []
      (let [wall (trace :wall (dist/weighted [1.0 1.0]))
            action (trace :action (dist/categorical
                                    (mx/take-idx action-logits wall 0)))]
        wall))))

;; ---------------------------------------------------------------------------
;; Layer 6: Baby's inference — exact enumeration + conditioning
;; ---------------------------------------------------------------------------

(defn baby-inference
  "Baby observes agent's action, infers wall presence.

   Uses exact enumeration over the agent model's joint P(wall, action),
   then conditions on the observed action to get P(wall | action).

   Returns {:joint exact-joint-result
            :p-wall-given-action fn  — (fn [action-idx] -> P(wall=1))}."
  [angle-values impulse-values beta weight]
  (let [model (agent-model angle-values impulse-values beta weight)]
    {:joint (exact/exact-joint model [] nil)
     :p-wall-given-action
     (fn [action-idx]
       (exact/pr model :wall 1 :given :action action-idx))}))

;; ---------------------------------------------------------------------------
;; GPU-vectorized version for benchmarks
;; ---------------------------------------------------------------------------

(defn baby-inference-gpu
  "GPU-vectorized inference — all trajectories in one MLX broadcast kernel.
   Returns [n-angles, n-impulses] tensor of P(wall=1 | theta, iota)."
  [{:keys [n-angles n-impulses max-impulse weights beta]
    :or {n-angles 50 n-impulses 50 max-impulse 1.5
         weights [1.0 1.5 2.0] beta 8.0}}]
  (let [n-wts (count weights)
        th-grid (mx/reshape
                  (.astype (mx/array (clj->js (mapv #(* (/ % (dec n-angles)) (/ js/Math.PI 2))
                                                    (range n-angles)))) mx/float32)
                  #js [1 1 n-angles 1])
        io-grid (mx/reshape
                  (.astype (mx/array (clj->js (mapv #(* (/ % (dec n-impulses)) max-impulse)
                                                    (range n-impulses)))) mx/float32)
                  #js [1 1 1 n-impulses])
        w-grid (mx/reshape (mx/array #js [0.0 1.0]) #js [2 1 1 1])
        wt-grid (mx/reshape (.astype (mx/array (clj->js weights)) mx/float32) #js [1 n-wts 1 1])
        _ (mx/eval! th-grid io-grid w-grid wt-grid)
        xf (projectile-final-x th-grid io-grid w-grid)
        reward (.astype (mx/greater xf (mx/scalar goal-x)) mx/float32)
        cost (mx/multiply io-grid (mx/add (mx/scalar 1.0)
                                          (mx/multiply (mx/scalar 0.5) th-grid)))
        logits (mx/multiply (mx/scalar beta) (mx/subtract (mx/multiply wt-grid reward) cost))
        flat (mx/reshape logits #js [2 n-wts (* n-angles n-impulses)])
        log-Z (mx/logsumexp flat [-1] true)
        log-p (mx/reshape (mx/subtract flat log-Z) #js [2 n-wts n-angles n-impulses])
        log-p-wall (mx/subtract (mx/logsumexp log-p [1])
                                 (mx/scalar (js/Math.log n-wts)))
        result (mx/sigmoid (mx/subtract (mx/idx log-p-wall 1) (mx/idx log-p-wall 0)))]
    (mx/eval! result)
    result))

;; ===========================================================================
;; Demonstration
;; ===========================================================================

(println "\n===== Intuitive Physics + Psychology (GenMLX) =====\n")

;; Shared discretization
(def angle-vals (mx/array #js [0.15 0.35 0.8 1.2]))
(def impulse-vals (mx/array #js [0.3 0.8 1.5]))
(mx/eval! angle-vals impulse-vals)

;; ---------------------------------------------------------------------------
;; 1. Projectile physics
;; ---------------------------------------------------------------------------
(println "-- 1. Projectile physics --")
(let [x1 (projectile-final-x (mx/scalar 0.3) (mx/scalar 1.5) (mx/scalar 0.0))
      x2 (projectile-final-x (mx/scalar 0.3) (mx/scalar 1.5) (mx/scalar 1.0))
      x3 (projectile-final-x (mx/scalar 1.2) (mx/scalar 1.5) (mx/scalar 1.0))
      _ (mx/eval! x1 x2 x3)]
  (println (str "  No wall, direct shot: x = " (.toFixed (mx/item x1) 3) " (should reach goal > 2.0)"))
  (println (str "  Wall, direct shot:    x = " (.toFixed (mx/item x2) 3) " (blocked at wall ~ 1.0)"))
  (println (str "  Wall, arcing shot:    x = " (.toFixed (mx/item x3) 3) " (clears wall > 2.0)"))
  (check "no wall, direct -> reaches goal" (> (mx/item x1) 2.0))
  (check "wall, direct -> blocked" (< (mx/item x2) 1.1))
  (check "wall, arcing -> clears wall" (> (mx/item x3) 2.0)))

;; ---------------------------------------------------------------------------
;; 2. Baby inference via gen fn + exact enumeration
;; ---------------------------------------------------------------------------
(println "\n-- 2. Baby inference via gen fn --")
(println "  The baby observes the agent's action and infers wall presence.")
(println "  Action = (angle, impulse) pair. 4 angles x 3 impulses = 12 actions.")
(println "  Action 5 = angle[1]=0.35, impulse[2]=1.5 -> direct path")
(println "  Action 8 = angle[2]=0.80, impulse[2]=1.5 -> arcing path")
(println "  Action 0 = angle[0]=0.15, impulse[0]=0.3 -> barely moves\n")

(let [{:keys [p-wall-given-action]} (baby-inference angle-vals impulse-vals 5.0 2.0)
      p-direct (p-wall-given-action 5)
      p-arcing (p-wall-given-action 8)
      p-ambig  (p-wall-given-action 0)]
  (println (str "  P(wall | direct path):  " (.toFixed p-direct 4)))
  (println (str "  P(wall | arcing path):  " (.toFixed p-arcing 4)))
  (println (str "  P(wall | barely moves): " (.toFixed p-ambig 4)))
  (check "direct path -> no wall (< 0.01)" (< p-direct 0.01))
  (check "arcing -> wall likely (> 0.5)" (> p-arcing 0.5))
  (check "arcing > direct" (> p-arcing p-direct)))

;; ---------------------------------------------------------------------------
;; 3. Differentiability — gradients through physics
;; ---------------------------------------------------------------------------
(println "\n-- 3. Differentiability --")
(let [grad-fn (mx/grad (fn [theta]
                         (agent-utility theta (mx/scalar 1.5) (mx/scalar 0.0) (mx/scalar 2.0))))
      g (grad-fn (mx/scalar 0.5))
      _ (mx/eval! g)]
  (println (str "  d(utility)/d(theta) at theta=0.5: " (.toFixed (mx/item g) 4)))
  (check "utility gradient w.r.t. theta is finite" (js/isFinite (mx/item g))))

;; ---------------------------------------------------------------------------
;; 4. GPU vectorization benchmark
;; ---------------------------------------------------------------------------
(println "\n-- 4. GPU vectorization benchmark --")
(println "  50x50 grid = 2500 (angle, impulse) pairs x 2 walls x 3 weights")
(println "  = 15,000 trajectories per call, all in one MLX broadcast kernel.\n")

(let [;; Warmup
      _ (baby-inference-gpu {})
      ;; Benchmark
      t0 (js/Date.now)
      _ (dotimes [_ 50] (baby-inference-gpu {}))
      t1 (js/Date.now)
      ms-per-run (/ (- t1 t0) 50.0)]
  (println (str "  50 runs of 15K trajectories: " (.toFixed ms-per-run 2) " ms/run"))
  (check "vectorized inference < 10ms" (< ms-per-run 10.0)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n===== Results: " @pass-count " passed, " @fail-count " failed ====="))
(when (pos? @fail-count) (js/process.exit 1))
