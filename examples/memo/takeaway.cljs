(ns takeaway
  "1-2 Takeaway: game theory + theory of mind.

   Two players take turns removing 1 or 2 matchsticks from a pile.
   The player who takes the last matchstick wins. The 'trick' is to
   leave a multiple of 3 for your opponent.

   Part 1: Solve the game via value iteration using `iterate`.
   Part 2: Observe an opponent's move and infer whether they know
   the trick (theory of mind via exact enumeration).

   Implements memo's demo-takeaway.ipynb."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Game parameters
;; =========================================================================

(def max-n 21)  ;; pile sizes 0..20

;; =========================================================================
;; Part 1: Solve the game via iterate
;; =========================================================================

(defn game-step
  "One step of value iteration. Pure function: state → state'.

   For each pile size n, compute:
   - EU(n): expected utility of optimal play
   - π(n, m): probability of taking m matchsticks (softmax over EU)

   EU(0) = -1 (you lost — no matchsticks to take).
   EU(n) = max_m -EU(n-m) (optimal play negates opponent's value).
   π(n, m) ∝ exp(10 * -EU(n-m)) for valid moves (n >= m)."
  [{:keys [eu]}]
  (let [beta 10.0
        ;; For each pile size, compute EU and policy
        new-eu (mx/array
                 (clj->js
                   (mapv (fn [n]
                           (if (zero? n)
                             -1.0  ;; lost
                             (let [;; EU of taking 1: -EU(n-1)
                                   eu-take-1 (- (mx/item (mx/idx eu (dec n))))
                                   ;; EU of taking 2: -EU(n-2) if n >= 2
                                   eu-take-2 (if (>= n 2)
                                               (- (mx/item (mx/idx eu (- n 2))))
                                               -1000.0)]  ;; invalid
                               (max eu-take-1 eu-take-2))))
                         (range max-n))))
        ;; Policy: softmax over valid moves
        new-pi (mx/array
                 (clj->js
                   (mapv (fn [n]
                           (if (zero? n)
                             [0.5 0.5]  ;; doesn't matter
                             (let [eu-1 (- (mx/item (mx/idx eu (dec n))))
                                   eu-2 (if (>= n 2)
                                           (- (mx/item (mx/idx eu (- n 2))))
                                           -1000.0)
                                   ;; Softmax
                                   logits [(* beta eu-1) (* beta eu-2)]
                                   max-l (apply max logits)
                                   exps (mapv #(js/Math.exp (- % max-l)) logits)
                                   z (reduce + exps)]
                               (mapv #(/ % z) exps))))
                         (range max-n))))
        _ (mx/eval! new-eu new-pi)]
    {:eu new-eu :pi new-pi}))

(defn solve-game
  "Solve 1-2 Takeaway via iterate. Returns converged {eu, pi}."
  [n-iters]
  (let [initial {:eu (mx/zeros #js [max-n])}
        _ (mx/eval! (:eu initial))]
    (->> initial
         (iterate game-step)
         (drop n-iters)
         first)))

;; =========================================================================
;; Part 2: Theory of mind — does the opponent know the trick?
;; =========================================================================

(defn cognizant-model
  "Model for inferring if opponent knows the trick.
   Traces :strategy (0=knows, 1=guessing) and :move (0=take1, 1=take2).
   A knowledgeable player follows π; a guesser plays uniformly."
  [n pi]
  (let [;; Extract policy for this pile size
        p-take-1 (mx/item (mx/idx (mx/idx pi n) 0))
        p-take-2 (mx/item (mx/idx (mx/idx pi n) 1))]
    (gen []
      (let [strategy (trace :strategy (dist/weighted [1.0 1.0]))  ;; uniform prior
            ;; Knowledgeable: follows optimal policy
            ;; Guesser: uniform over valid moves
            w1 (mx/where (mx/eq? strategy 0)
                  (mx/scalar p-take-1)
                  (mx/scalar (if (>= n 2) 0.5 1.0)))
            w2 (mx/where (mx/eq? strategy 0)
                  (mx/scalar p-take-2)
                  (mx/scalar (if (>= n 2) 0.5 0.0)))
            move (trace :move (dist/weighted [w1 w2]))]
        strategy))))

;; =========================================================================
;; Demo
;; =========================================================================

(println "=============================================")
(println " 1-2 Takeaway: Game Theory + Theory of Mind")
(println "=============================================\n")

;; Solve the game
(println "-- Part 1: Optimal strategy (via iterate) --\n")

(let [{:keys [eu pi]} (solve-game 20)]
  (println "  The trick: leave a multiple of 3 for your opponent.\n")
  (println "  n   EU    P(take 1)  P(take 2)  Optimal")
  (println "  --  ----  ---------  ---------  -------")
  (doseq [n (range 1 13)]
    (let [p1 (mx/item (mx/idx (mx/idx pi n) 0))
          p2 (mx/item (mx/idx (mx/idx pi n) 1))
          ev (mx/item (mx/idx eu n))
          optimal (cond
                    (> p1 0.9) "take 1"
                    (> p2 0.9) "take 2"
                    :else "either (losing)")]
      (println (str "  " (if (< n 10) (str " " n) n)
                    "  " (.toFixed ev 1)
                    "    " (.toFixed p1 3)
                    "      " (.toFixed p2 3)
                    "      " optimal))))

  ;; Verification
  (println "\nVerification:")

  ;; n=1: must take 1 (win)
  (let [p1 (mx/item (mx/idx (mx/idx pi 1) 0))]
    (assert (> p1 0.9) "n=1: take 1")
    (println "  PASS: n=1 → take 1"))

  ;; n=2: must take 2 (win)
  (let [p2 (mx/item (mx/idx (mx/idx pi 2) 1))]
    (assert (> p2 0.9) "n=2: take 2")
    (println "  PASS: n=2 → take 2"))

  ;; n=3: losing position (multiple of 3)
  (let [p1 (mx/item (mx/idx (mx/idx pi 3) 0))
        p2 (mx/item (mx/idx (mx/idx pi 3) 1))]
    (assert (< (abs (- p1 0.5)) 0.1) "n=3: indifferent")
    (println "  PASS: n=3 → indifferent (losing position)"))

  ;; n=4: take 1 (leave 3 for opponent)
  (let [p1 (mx/item (mx/idx (mx/idx pi 4) 0))]
    (assert (> p1 0.9) "n=4: take 1")
    (println "  PASS: n=4 → take 1 (leave 3)"))

  ;; n=5: take 2 (leave 3 for opponent)
  (let [p2 (mx/item (mx/idx (mx/idx pi 5) 1))]
    (assert (> p2 0.9) "n=5: take 2")
    (println "  PASS: n=5 → take 2 (leave 3)"))

  ;; Pattern: multiples of 3 are losing positions
  (let [losing (every? (fn [n]
                         (let [ev (mx/item (mx/idx eu n))]
                           (< ev 0)))
                       [3 6 9 12 15 18])]
    (assert losing "multiples of 3 are losing")
    (println "  PASS: all multiples of 3 are losing positions"))

  ;; Part 2: Theory of mind
  (println "\n-- Part 2: Does the opponent know the trick? --\n")

  (println "  Observe optimal move → infer if opponent is cognizant.\n")
  (doseq [n [4 5 6 7 10 12 20]]
    (let [model (cognizant-model n pi)
          ;; What move would the optimal player make?
          optimal-move (if (> (mx/item (mx/idx (mx/idx pi n) 0)) 0.5) 0 1)
          ;; P(cognizant | optimal move)
          p-cog (exact/pr model :strategy 0 :given :move optimal-move)]
      (println (str "  n=" (if (< n 10) (str " " n) n)
                    " optimal=take" (inc optimal-move)
                    " → P(cognizant)=" (.toFixed p-cog 3)
                    (when (zero? (mod n 3)) " (multiple of 3: uninformative)")))))

  ;; Verify: non-multiples of 3 should give high P(cognizant)
  (let [p-cog-4 (exact/pr (cognizant-model 4 pi) :strategy 0 :given :move 0)
        p-cog-7 (exact/pr (cognizant-model 7 pi) :strategy 0 :given :move 0)]
    (assert (> p-cog-4 0.6) "n=4: cognizant likely")
    (println "\n  PASS: n=4, optimal move → P(cognizant) > 0.6")
    (assert (> p-cog-7 0.6) "n=7: cognizant likely")
    (println "  PASS: n=7, optimal move → P(cognizant) > 0.6"))

  ;; Multiple of 3: both strategies look the same
  (let [p-cog-6 (exact/pr (cognizant-model 6 pi) :strategy 0 :given :move 0)]
    (assert (< (abs (- p-cog-6 0.5)) 0.15) "n=6: uninformative")
    (println "  PASS: n=6, any move → P(cognizant) ≈ 0.5 (uninformative)")))

(println "\nAll checks passed.")
