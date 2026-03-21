(ns schelling
  "Schelling coordination game with recursive reasoning.

   Two agents (Alice and Bob) choose between two bars. Both prefer to
   meet at the same bar. The prior favors Bar0 (0.55 vs 0.45).

   At each depth of reasoning, each agent models the other's choice
   using exact enumeration, then combines their prior preference with
   the predicted distribution of the other agent. Mutual recursion
   creates a feedback loop that amplifies the prior asymmetry:

     depth 0: Alice P(Bar0) ~ 0.60
     depth 4: Alice P(Bar0) ~ 0.83+

   This is the focal-point amplification effect from game theory.

   The recursion is expressed as a pure functional iteration:
   state -> state' via Clojure's `iterate`. No atoms, no cache —
   just a lazy sequence of improving beliefs."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.gen :refer [gen]]
            [genmlx.inference.exact :as exact]))

;; ---------------------------------------------------------------------------
;; Prior
;; ---------------------------------------------------------------------------

(def prior (mx/array #js [0.55 0.45]))
(def uniform (mx/array #js [0.5 0.5]))

;; ---------------------------------------------------------------------------
;; Pure step function: one depth of mutual reasoning
;; ---------------------------------------------------------------------------

(defn agent-probs
  "Compute one agent's choice distribution given the other agent's probs.
   The agent believes the other chose according to other-dist, then
   picks a bar weighted by prior * other-dist (wanting to coordinate).
   Returns the marginal over the agent's own choice."
  [other-dist self-addr other-addr]
  (let [model (gen []
                (let [_other (trace other-addr (dist/categorical (mx/log other-dist)))
                      self   (trace self-addr (dist/categorical
                                               (mx/log (mx/multiply prior other-dist))))]
                  self))
        r (exact/exact-posterior model [] nil)
        table (mx/array #js [(get-in r [:marginals self-addr 0])
                              (get-in r [:marginals self-addr 1])])
        _ (mx/eval! table)]
    table))

(defn schelling-step
  "One depth of mutual reasoning. Pure function: state -> state'.

   Given Bob's previous probs:
   1. Alice models Bob (using his previous probs) -> alice-probs
   2. Bob models Alice (using her new probs) -> bob-probs

   Returns updated {:alice-probs :bob-probs}."
  [{:keys [bob-probs]}]
  (let [alice-probs (agent-probs bob-probs :alice-b :bob-b)
        bob-probs'  (agent-probs alice-probs :bob-b :alice-b)]
    {:alice-probs alice-probs
     :bob-probs   bob-probs'}))

;; ---------------------------------------------------------------------------
;; Build the full reasoning sequence via iterate
;; ---------------------------------------------------------------------------

(def max-depth 4)

;; Depth 0: Bob reasons with uniform alice, no alice model yet
(def depth-0
  (let [bob-probs (agent-probs uniform :bob-b :alice-b)]
    {:alice-probs nil
     :bob-probs   bob-probs}))

;; Depths 1..max-depth via iterate
(def depths
  (->> depth-0
       (iterate schelling-step)
       (take (inc max-depth))
       vec))

;; ---------------------------------------------------------------------------
;; Display
;; ---------------------------------------------------------------------------

(defn val-at [table idx]
  (mx/item (mx/slice table idx (inc idx))))

(println "Schelling Coordination Game")
(println "===========================")
(println)
(println "Two agents choose between Bar0 and Bar1.")
(println "Prior: Bar0 = 0.55, Bar1 = 0.45")
(println "Goal: coordinate (meet at the same bar).")
(println)

(doseq [d (range (inc max-depth))]
  (let [{:keys [alice-probs bob-probs]} (nth depths d)]
    (println (str "Depth " d ":"))
    (println (str "  Bob   P(Bar0) = " (.toFixed (val-at bob-probs 0) 4)
                  "  P(Bar1) = " (.toFixed (val-at bob-probs 1) 4)))
    (when alice-probs
      (println (str "  Alice P(Bar0) = " (.toFixed (val-at alice-probs 0) 4)
                    "  P(Bar1) = " (.toFixed (val-at alice-probs 1) 4))))
    (println)))

;; ---------------------------------------------------------------------------
;; Verify against memo reference values
;; ---------------------------------------------------------------------------

(println "Verification against reference values:")

(defn check [label expected actual tol]
  (let [ok (< (abs (- expected actual)) tol)]
    (println (str "  " (if ok "PASS" "FAIL") " " label
                  " expected=" (.toFixed expected 4)
                  " got=" (.toFixed actual 4)))
    (when-not ok
      (throw (js/Error. (str "FAIL: " label))))))

;; Depth 0
(check "bob(0)[Bar0]"   0.55   (val-at (:bob-probs (nth depths 0)) 0) 1e-4)
(check "alice(0)[Bar0]" 0.5990 (val-at (:alice-probs (nth depths 1)) 0) 1e-3)

;; Depth 1
(check "bob(1)[Bar0]"   0.6461 (val-at (:bob-probs (nth depths 1)) 0) 1e-3)

;; Depth 2
(check "bob(2)[Bar0]"   0.7317 (val-at (:bob-probs (nth depths 2)) 0) 1e-3)
(check "alice(2)[Bar0]" 0.6905 (val-at (:alice-probs (nth depths 2)) 0) 1e-3)

;; Depth 4: convergence — Alice strongly prefers Bar0
(let [a4 (val-at (:alice-probs (nth depths 4)) 0)]
  (when (< a4 0.83)
    (throw (js/Error. (str "FAIL: alice(4)[Bar0] = " a4 " < 0.83"))))
  (println (str "  PASS alice(4)[Bar0] >= 0.83 (got " (.toFixed a4 4) ")"))
  (println (str "\nConvergence: Alice depth-4 P(Bar0) = " (.toFixed a4 4)
               " (amplified from prior 0.55)")))

(println "\nAll checks passed.")
