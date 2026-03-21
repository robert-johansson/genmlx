(ns guess-23
  "Guess 2/3 of the average — level-k cognitive hierarchy.

   A newspaper runs a contest: readers guess an integer 0-100. The winner
   is whoever is closest to 2/3 of the average guess. The Nash equilibrium
   is 0, but real players reason at finite depth.

   Level-0: guesses uniformly (expected value 50)
   Level-k: best-responds to level-(k-1) via softmax over
            exp(beta * -|n - (2/3) * E_{k-1}|)

   Shows convergence toward 0 with increasing depth.

   The level-k hierarchy is computed via Clojure's `iterate` — a pure
   functional sequence where each level best-responds to the previous.
   No mutation, no cache atoms, just lazy iteration.

   Reference: Nagel (1995), 'Unraveling in guessing games'"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Domain: N = 0..100
;; ---------------------------------------------------------------------------

(def N 101)
(def ns-arr (.astype (mx/arange 0 N) mx/float32))
(mx/eval! ns-arr)

;; ---------------------------------------------------------------------------
;; Level-k iteration (pure functional — no mutation)
;; ---------------------------------------------------------------------------

(defn level-step
  "One level of best-response reasoning. Pure function: probs -> probs.

   Given the previous level's probability distribution over guesses 0-100,
   computes the best-response distribution via softmax:
     target = (2/3) * E[guess | prev]
     utility(n) = -|n - target|
     probs(n) ∝ exp(beta * utility(n))

   Returns the new level's probability array [101]."
  [beta probs]
  (let [;; E[n] under previous level's distribution
        expected (mx/sum (mx/multiply ns-arr probs))
        ;; Target: 2/3 of the expected value
        target (mx/multiply (mx/scalar (/ 2.0 3.0)) expected)
        ;; Utility: -|n - target|
        utility (mx/negative (mx/abs (mx/subtract ns-arr target)))
        ;; Softmax with temperature beta
        logits (mx/multiply (mx/scalar beta) utility)
        log-z (mx/logsumexp logits)
        new-probs (mx/exp (mx/subtract logits log-z))
        _ (mx/eval! new-probs)]
    new-probs))

(defn level-hierarchy
  "Compute probability distributions for levels 0 through max-level.
   Returns a vector of [101]-shaped probability arrays.

   Pure functional — uses Clojure's `iterate` over level-step."
  [beta max-level]
  (let [uniform (mx/full #js [N] (/ 1.0 N))
        _ (mx/eval! uniform)]
    (->> uniform
         (iterate (partial level-step beta))
         (take (inc max-level))
         vec)))

;; ---------------------------------------------------------------------------
;; Gen function model (for exact enumeration via GFI)
;; ---------------------------------------------------------------------------

(defn player-model
  "Gen function: a player chooses a guess from the given distribution.
   Uses dist/weighted with pre-computed level-k probabilities."
  [probs]
  (let [weights (mapv #(mx/idx probs %) (range N))]
    (gen []
      (let [guess (trace :guess (dist/weighted weights))]
        guess))))

;; ---------------------------------------------------------------------------
;; Compute and display results
;; ---------------------------------------------------------------------------

(println "Guess 2/3 of the Average")
(println "========================")
(println)
(println "Players guess 0-100, trying to be closest to 2/3 of the average.")
(println "Level-0 guesses uniformly; level-k best-responds to level-(k-1).")
(println)

(def beta 1.0)
(def levels (level-hierarchy beta 5))

;; Show expected values and peaks for levels 0-5
(doseq [k (range 6)]
  (let [probs (nth levels k)
        ev (mx/item (mx/sum (mx/multiply ns-arr probs)))
        peak (mx/item (mx/argmax probs))]
    (println (str "Level " k ": E[guess] = " (.toFixed ev 2)
                  (if (zero? k)
                    "  (uniform)"
                    (str "  mode = " peak))))))

(println)

;; Show P(guess) for select values using exact/pr via the gen model
(println "Exact probabilities via gen model + exact enumeration:")
(println)

(doseq [k (range 4)]
  (let [probs (nth levels k)
        model (player-model probs)
        ev (mx/item (mx/sum (mx/multiply ns-arr probs)))
        peak (long (mx/item (mx/argmax probs)))
        p-peak (exact/pr model :guess peak)
        ;; Also show probability at 0 and 50
        p-0 (exact/pr model :guess 0)
        p-50 (exact/pr model :guess 50)]
    (println (str "Level " k " (E=" (.toFixed ev 1) "):"))
    (println (str "  P(guess=" peak ") = " (.toFixed p-peak 4) " (mode)"))
    (println (str "  P(guess=0)  = " (.toFixed p-0 4)))
    (println (str "  P(guess=50) = " (.toFixed p-50 4)))
    (println)))

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println "Verification:")

(defn check [label pred]
  (println (str "  " (if pred "PASS" "FAIL") " " label))
  (when-not pred
    (throw (js/Error. (str "FAIL: " label)))))

(defn check-close [label expected actual tol]
  (check (str label " (" (.toFixed actual 4) " ~ " (.toFixed expected 4) ")")
         (< (abs (- expected actual)) tol)))

;; Level 0: uniform => E[guess] = 50
(let [probs (nth levels 0)
      ev (mx/item (mx/sum (mx/multiply ns-arr probs)))]
  (check-close "L0 E[guess] = 50" 50.0 ev 0.1))

;; Level 1: best-responds to E=50 => target = 33.33, mode ~ 33
(let [probs (nth levels 1)
      ev (mx/item (mx/sum (mx/multiply ns-arr probs)))
      peak (mx/item (mx/argmax probs))]
  (check-close "L1 E[guess] ~ 33.3" 33.33 ev 1.0)
  (check "L1 mode = 33" (= peak 33)))

;; Level 2: best-responds to L1 => target ~ 22.2
(let [probs (nth levels 2)
      ev (mx/item (mx/sum (mx/multiply ns-arr probs)))]
  (check-close "L2 E[guess] ~ 22.2" 22.2 ev 1.0))

;; Level 3: target ~ 14.8
(let [probs (nth levels 3)
      ev (mx/item (mx/sum (mx/multiply ns-arr probs)))]
  (check-close "L3 E[guess] ~ 14.8" 14.8 ev 1.0))

;; Monotone convergence: E[k] > E[k+1] for all k
(let [evs (mapv (fn [probs]
                  (mx/item (mx/sum (mx/multiply ns-arr probs))))
                levels)]
  (doseq [k (range 5)]
    (check (str "E[L" k "] > E[L" (inc k) "]")
           (> (nth evs k) (nth evs (inc k))))))

;; Convergence toward 0: Level 5 E[guess] < 10
(let [probs (nth levels 5)
      ev (mx/item (mx/sum (mx/multiply ns-arr probs)))]
  (check (str "L5 E[guess] < 10 (got " (.toFixed ev 2) ")")
         (< ev 10)))

;; exact/pr matches distribution
(let [probs (nth levels 1)
      model (player-model probs)
      p33 (exact/pr model :guess 33)
      p33-direct (mx/item (mx/idx probs 33))]
  (check-close "exact/pr matches direct" p33-direct p33 1e-5))

(println "\nAll checks passed. Deeper reasoning -> lower guesses -> convergence to 0.")
