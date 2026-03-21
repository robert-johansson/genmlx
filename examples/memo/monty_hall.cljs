(ns monty-hall
  "The Monty Hall problem solved by exact enumeration.

   You pick one of three doors (A=0, B=1, C=2). Behind one is a prize.
   The host, who knows where the prize is, opens a door that (a) you
   didn't pick and (b) doesn't have the prize, then asks if you want
   to switch. Exact enumeration shows switching wins 2/3 of the time."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Model
;; ---------------------------------------------------------------------------

(def model
  (gen [pick-idx]
    (let [prize (trace :prize (dist/weighted [1/3 1/3 1/3]))
          pick  (mx/scalar pick-idx mx/int32)
          ;; Host can reveal door i only if prize != i AND pick != i
          v0     (mx/and* (mx/neq? prize 0) (mx/neq? pick 0))
          v1     (mx/and* (mx/neq? prize 1) (mx/neq? pick 1))
          v2     (mx/and* (mx/neq? prize 2) (mx/neq? pick 2))
          logits (mx/transpose (mx/log (mx/stack [v0 v1 v2])))
          reveal (trace :reveal (dist/categorical logits))]
      prize)))

;; ---------------------------------------------------------------------------
;; Inference — exact posterior via enumeration
;; ---------------------------------------------------------------------------

(println "Monty Hall — exact enumeration\n")

;; Scenario 1: pick door A (0), host reveals door C (2)
(let [r (exact/exact-posterior model [0] (cm/choicemap :reveal (mx/scalar 2 mx/int32)))
      p0 (get-in r [:marginals :prize 0])
      p1 (get-in r [:marginals :prize 1])
      p2 (get-in r [:marginals :prize 2])]
  (println "Pick A, host reveals C:")
  (println "  P(prize=A) =" (.toFixed p0 4) " (stay)")
  (println "  P(prize=B) =" (.toFixed p1 4) " (switch)")
  (println "  P(prize=C) =" (.toFixed p2 4) " (opened)")
  (assert (< (abs (- p0 (/ 1.0 3))) 1e-5) "P(prize=A) should be 1/3")
  (assert (< (abs (- p1 (/ 2.0 3))) 1e-5) "P(prize=B) should be 2/3")
  (assert (< (abs (- p2 0.0)) 1e-5) "P(prize=C) should be 0")
  (println "  => Switch wins 2/3 of the time.\n"))

;; Scenario 2: pick door A (0), host reveals door B (1)
(let [r (exact/exact-posterior model [0] (cm/choicemap :reveal (mx/scalar 1 mx/int32)))
      p2 (get-in r [:marginals :prize 2])]
  (println "Pick A, host reveals B:")
  (println "  P(prize=C) =" (.toFixed p2 4) " (switch)")
  (assert (< (abs (- p2 (/ 2.0 3))) 1e-5) "P(prize=C) should be 2/3")
  (println "  => Switching to C wins 2/3.\n"))

;; Scenario 3: pick door B (1), host reveals door C (2)
(let [r (exact/exact-posterior model [1] (cm/choicemap :reveal (mx/scalar 2 mx/int32)))
      p0 (get-in r [:marginals :prize 0])]
  (println "Pick B, host reveals C:")
  (println "  P(prize=A) =" (.toFixed p0 4) " (switch)")
  (assert (< (abs (- p0 (/ 2.0 3))) 1e-5) "P(prize=A) should be 2/3")
  (println "  => Switching to A wins 2/3.\n"))

(println "All checks passed. Always switch!")
