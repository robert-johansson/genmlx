(ns dining-crypto
  "Dining Cryptographers — anonymous payment protocol.

   Inspired by: Chaum, D. (1988). The dining cryptographers problem:
   Unconditional sender and recipient untraceability.

   Three cryptographers dine together. Either one of them paid or the
   NSA paid. Each adjacent pair flips a coin. Each cryptographer
   announces XOR of their two coins, flipped if they paid. The XOR of
   all announcements reveals whether the NSA paid, without revealing
   which cryptographer paid (if any).

   We model this from cryptographer A's perspective. A knows they did
   not pay, their own coin flip, and B's coin flip (shared neighbor
   coin). A observes B's and C's announcements. We show that A can
   never distinguish B-paid from C-paid — the protocol is private.

   Who = {A_PAID=0, B_PAID=1, C_PAID=2, NSA_PAID=3}
   Bit = {0, 1}

   For every combination of (a-coin, b-coin, b-announce, c-announce),
   A's posterior is always either [0 0 0 1] (NSA paid) or
   [0 0.5 0.5 0] (B or C paid, equally likely). A can never tell
   which of B or C paid — unconditional sender untraceability."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.exact :as exact]
            [clojure.string :as str])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Belief model: A's reasoning about who paid
;; ---------------------------------------------------------------------------
;; A knows: a-coin, b-coin (shared between A and B).
;; A observes: B's announcement (bx), C's announcement (cx).
;; A reasons over: who paid (w), C's coin (c-coin).
;;
;; Protocol constraints (deterministic):
;;   B announces: b-coin XOR c-coin XOR (w == 1)   [flipped if B paid]
;;   C announces: c-coin XOR a-coin XOR (w == 2)   [flipped if C paid]
;;
;; XOR of two binary values = neq? (a XOR b = 1 iff a != b).

(defn belief-model
  "Build A's inner belief model for given known values.
   a-coin, b-coin are 0 or 1 (JS numbers)."
  [a-coin b-coin]
  (gen []
    ;; Who paid? A knows it wasn't them (w != 0), so uniform over {1,2,3}
    (let [w (trace :w (dist/weighted [0 1 1 1]))
          ;; C's coin: uniform {0, 1}
          c-coin (trace :c-coin (dist/weighted [1 1]))
          ;; B's announcement: b-coin XOR c-coin XOR (w == 1)
          ;; Chain XOR: neq?(neq?(a, b), c) = a XOR b XOR c
          bx-val (mx/neq? (mx/neq? b-coin c-coin) (mx/eq? w 1))
          ;; C's announcement: c-coin XOR a-coin XOR (w == 2)
          cx-val (mx/neq? (mx/neq? c-coin a-coin) (mx/eq? w 2))
          ;; Trace announcements as categorical with log-indicator logits.
          ;; When bx-val=1: logits [-inf, 0] -> must be 1
          ;; When bx-val=0: logits [0, -inf] -> must be 0
          ;; This gives exact conditioning (zero probability for wrong value).
          bx-logits (mx/stack [(mx/log (mx/subtract 1 bx-val))
                               (mx/log bx-val)]
                              -1)
          cx-logits (mx/stack [(mx/log (mx/subtract 1 cx-val))
                               (mx/log cx-val)]
                              -1)]
      (trace :bx (dist/categorical bx-logits))
      (trace :cx (dist/categorical cx-logits))
      w)))

;; ---------------------------------------------------------------------------
;; Run all 16 scenarios and display results
;; ---------------------------------------------------------------------------

(println "Dining Cryptographers — exact enumeration")
(println "==========================================")
(println)
(println "Protocol: each pair flips a coin, each cryptographer announces")
(println "XOR of their two coins (flipped if they paid).")
(println)
(println "From A's perspective (A did not pay):")
(println "  Who: A_PAID=0, B_PAID=1, C_PAID=2, NSA_PAID=3")
(println)

(let [pass? (atom true)
      results (atom [])]
  (doseq [a-coin [0 1]
          b-coin [0 1]
          bx     [0 1]
          cx     [0 1]]
    (let [model (belief-model a-coin b-coin)
          obs (cm/merge-cm
                (cm/choicemap :bx (mx/scalar bx mx/int32))
                (cm/choicemap :cx (mx/scalar cx mx/int32)))
          r (exact/exact-posterior model [] obs)
          probs (mapv #(get-in r [:marginals :w %] 0.0) (range 4))
          ;; Expected: either [0 0 0 1] or [0 0.5 0.5 0]
          ;; P(A_PAID) is always 0 (A knows they didn't pay: prior weight 0)
          ;; Either NSA paid (all 3 XORs cancel) or B/C paid (indistinguishable)
          nsa? (> (nth probs 3) 0.9)
          expected (if nsa? [0.0 0.0 0.0 1.0] [0.0 0.5 0.5 0.0])
          ok (every? true? (map #(< (abs (- %1 %2)) 1e-4) probs expected))]
      (when-not ok (reset! pass? false))
      (swap! results conj {:a a-coin :b b-coin :bx bx :cx cx
                           :probs probs :expected expected :ok ok})
      (println (str "  A flips " a-coin ", B flips " b-coin
                    ", B says " bx ", C says " cx
                    " -> [" (str/join " "
                              (map #(.toFixed % 1) probs)) "]"
                    (when-not ok " FAIL")))))

  ;; ---------------------------------------------------------------------------
  ;; Verification
  ;; ---------------------------------------------------------------------------

  (println)
  (println "Verification:")

  ;; Key property: P(B_PAID) always equals P(C_PAID)
  (let [all-equal (every? #(< (abs (- (nth (:probs %) 1)
                                       (nth (:probs %) 2)))
                               1e-6)
                          @results)]
    (println (str "  " (if all-equal "PASS" "FAIL")
                  ": P(B_PAID) = P(C_PAID) in all 16 scenarios"
                  " (unconditional untraceability)"))
    (when-not all-equal (reset! pass? false)))

  ;; Key property: P(A_PAID) is always 0
  (let [a-zero (every? #(< (nth (:probs %) 0) 1e-6) @results)]
    (println (str "  " (if a-zero "PASS" "FAIL")
                  ": P(A_PAID) = 0 in all scenarios"
                  " (A knows they didn't pay)"))
    (when-not a-zero (reset! pass? false)))

  ;; Key property: posterior is always one of two values
  (let [binary (every? (fn [{:keys [probs]}]
                         (let [p-nsa (nth probs 3)]
                           (or (< (abs (- p-nsa 1.0)) 1e-4)
                               (< (abs p-nsa) 1e-4))))
                       @results)]
    (println (str "  " (if binary "PASS" "FAIL")
                  ": posterior is always [0 0 0 1] or [0 0.5 0.5 0]"))
    (when-not binary (reset! pass? false)))

  ;; Count: exactly 8 of 16 scenarios should be NSA, 8 should be B/C
  (let [nsa-count (count (filter #(> (nth (:probs %) 3) 0.9) @results))]
    (println (str "  " (if (= nsa-count 8) "PASS" "FAIL")
                  ": " nsa-count "/16 scenarios point to NSA"
                  " (expected 8)"))
    (when (not= nsa-count 8) (reset! pass? false)))

  (println)
  (if @pass?
    (println "All checks passed. The protocol provides unconditional untraceability.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
