(ns empowerment
  "Empowerment — channel capacity via Blahut-Arimoto iteration.

   Empowerment measures how much control an agent has over its future:
   the channel capacity I(action; outcome) = max_q I(X;Y) where X is
   the action, Y is the observation, and q(x) is the input distribution.

   The Blahut-Arimoto algorithm finds the capacity-achieving q(x) by
   iterating a pure function: state → state'. This is a perfect fit
   for Clojure's `iterate` — no mutation, no cache atoms, just a lazy
   sequence of improving distributions.

   Inspired by: Klyubin, Polani & Nehaniv (2005).
   Implements memo's demo-empowerment.py."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Channel definition
;; =========================================================================

;; Symmetric channel: p(y|x) — high probability of y=x, small noise
(def symmetric-channel
  (mx/array #js [#js [0.9  0.05 0.05]
                 #js [0.05 0.9  0.05]
                 #js [0.05 0.05 0.9]]))

;; Asymmetric channel: different noise in each row
(def asymmetric-channel
  (mx/array #js [#js [0.8  0.15 0.05]
                 #js [0.1  0.7  0.2]
                 #js [0.05 0.15 0.8]]))

(mx/eval! symmetric-channel asymmetric-channel)

;; =========================================================================
;; Blahut-Arimoto as a pure functional iteration
;; =========================================================================

(defn ba-step
  "One Blahut-Arimoto step. Pure function: state → state'.

   Given the current input distribution q(x), computes:
   1. Joint p(x,y) = q(x) * p(y|x)
   2. Marginal p(y) = sum_x p(x,y)
   3. New q(x) ∝ exp(D_KL(p(y|x) || p(y))) — the capacity-achieving update
   4. Capacity I(X;Y) = H(Y) - H(Y|X)

   Returns updated state with {:q, :capacity}."
  [channel {:keys [q]}]
  (let [n (first (mx/shape channel))
        ;; Joint: p(x,y) = q(x) * p(y|x)
        joint (mx/multiply (mx/reshape q #js [n 1]) channel)
        ;; Marginal: p(y) = sum_x p(x,y)
        p-y (mx/sum joint [0])
        ;; KL per input: D_KL(p(y|x) || p(y)) for each x
        ;; = sum_y p(y|x) * log(p(y|x) / p(y))
        log-channel (mx/log (mx/maximum channel (mx/scalar 1e-30)))
        log-p-y (mx/log (mx/maximum (mx/reshape p-y #js [1 n]) (mx/scalar 1e-30)))
        kl-per-x (mx/sum (mx/multiply channel (mx/subtract log-channel log-p-y)) [-1])
        ;; New q: softmax of KL divergences
        new-q (mx/softmax kl-per-x)
        ;; Capacity: I(X;Y) = H(Y) - H(Y|X)
        h-y (mx/negative (mx/sum (mx/multiply p-y (mx/log (mx/maximum p-y (mx/scalar 1e-30))))))
        h-y-given-x (mx/negative (mx/sum (mx/multiply joint
                      (mx/log (mx/maximum channel (mx/scalar 1e-30))))))
        capacity (mx/divide (mx/subtract h-y h-y-given-x) (mx/scalar (js/Math.log 2)))
        _ (mx/eval! new-q capacity)]
    {:q new-q
     :capacity (mx/item capacity)}))

(defn blahut-arimoto
  "Run Blahut-Arimoto for n iterations. Returns lazy sequence of states.
   Each state has {:q input-distribution, :capacity channel-capacity-bits}.

   Pure functional — no mutation, just Clojure's `iterate`."
  [channel n-iterations]
  (let [n (first (mx/shape channel))
        initial {:q (mx/divide (mx/ones #js [n]) (mx/scalar n))}
        _ (mx/eval! (:q initial))]
    (->> initial
         (iterate (partial ba-step channel))
         (take (inc n-iterations))
         vec)))

;; =========================================================================
;; GenMLX integration: gen functions + exact/mutual-info
;; =========================================================================

(defn channel-model
  "Gen function for a communication channel.
   Traces :x (action) and :y (observation).
   Uses the given input distribution q(x) as action weights."
  [channel q]
  (gen []
    (let [x (trace :x (dist/categorical (mx/log (mx/maximum q (mx/scalar 1e-30)))))
          y (trace :y (dist/categorical (mx/take-idx (mx/log (mx/maximum channel (mx/scalar 1e-30))) x 0)))]
      [x y])))

;; =========================================================================
;; Demo: symmetric channel
;; =========================================================================

(println "=========================================")
(println " Empowerment via Blahut-Arimoto")
(println "=========================================\n")

(println "-- Symmetric channel (0.9 on diagonal) --\n")

(let [states (blahut-arimoto symmetric-channel 5)]
  (doseq [{:keys [q capacity]} (rest states)]
    (let [qs (.tolist q)]
      (println (str "  C = " (.toFixed capacity 4) " bits"
                    "  q = [" (.toFixed (aget qs 0) 4) ", "
                    (.toFixed (aget qs 1) 4) ", "
                    (.toFixed (aget qs 2) 4) "]"))))

  ;; Verify: symmetric channel → uniform q, capacity ≈ 1.016
  (let [final (last states)]
    (assert (< (abs (- (:capacity final) 1.016)) 0.01)
            "symmetric capacity ≈ 1.016")
    (assert (< (abs (- (mx/item (mx/idx (:q final) 0)) 1/3)) 0.01)
            "symmetric q is uniform")
    (println "\n  PASS: symmetric capacity ≈ 1.016 bits")
    (println "  PASS: optimal input is uniform")))

;; =========================================================================
;; Demo: asymmetric channel
;; =========================================================================

(println "\n-- Asymmetric channel --\n")

(let [states (blahut-arimoto asymmetric-channel 10)]
  (doseq [[i {:keys [q capacity]}] (map-indexed vector (rest states))]
    (let [qs (.tolist q)]
      (println (str "  t=" i
                    "  C = " (.toFixed capacity 4) " bits"
                    "  q = [" (.toFixed (aget qs 0) 4) ", "
                    (.toFixed (aget qs 1) 4) ", "
                    (.toFixed (aget qs 2) 4) "]"))))

  (let [final (last states)]
    ;; Verify convergence: capacity ≈ 0.617
    (assert (< (abs (- (:capacity final) 0.617)) 0.01)
            "asymmetric capacity ≈ 0.617")
    ;; Verify non-uniform: q is NOT [1/3, 1/3, 1/3]
    (assert (> (abs (- (mx/item (mx/idx (:q final) 1)) 1/3)) 0.01)
            "asymmetric q is non-uniform")
    (println "\n  PASS: asymmetric capacity ≈ 0.617 bits")
    (println "  PASS: optimal input is non-uniform")))

;; =========================================================================
;; Verify: exact/mutual-info matches Blahut-Arimoto
;; =========================================================================

(println "\n-- Cross-verify: exact/mutual-info matches BA --\n")

(let [;; BA result for asymmetric channel
      ba-states (blahut-arimoto asymmetric-channel 10)
      ba-capacity (:capacity (last ba-states))
      ba-q (:q (last ba-states))
      ;; GenMLX: build gen function with optimal q, compute MI
      model (channel-model asymmetric-channel ba-q)
      mi-nats (exact/mutual-info model #{:x} #{:y})
      mi-bits (/ mi-nats (js/Math.log 2))]
  (println (str "  Blahut-Arimoto capacity: " (.toFixed ba-capacity 4) " bits"))
  (println (str "  exact/mutual-info:       " (.toFixed mi-bits 4) " bits"))
  (assert (< (abs (- ba-capacity mi-bits)) 0.01)
          "BA and mutual-info agree")
  (println "  PASS: BA capacity = exact/mutual-info"))

;; =========================================================================
;; Bonus: iterate is lazy — compute on demand
;; =========================================================================

(println "\n-- Functional elegance: iterate is lazy --\n")

(let [;; The entire algorithm in one expression
      capacity-at-t10
      (->> {:q (mx/divide (mx/ones #js [3]) (mx/scalar 3))}
           (iterate (partial ba-step asymmetric-channel))
           (drop 10)
           first
           :capacity)]
  (println (str "  One-liner capacity: " (.toFixed capacity-at-t10 4) " bits"))
  (println "  (No intermediate storage — lazy evaluation)\n"))

(println "All checks passed.")
