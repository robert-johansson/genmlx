(ns chimp
  "Rational belief revision in chimpanzees.

   Inspired by Schleihauf et al., 'Chimpanzees rationally revise their
   beliefs.' Science 390, 521-526 (2025).

   A chimp chooses between two boxes that may or may not contain food.
   Evidence of varying strength (crumbs, noises, visual) is presented
   for one box, and the chimp makes a choice. Then stronger or weaker
   evidence is presented for the other box, and the chimp revises.

   Exact enumeration shows:
     - First choice follows evidence strength
     - Second choice flips if and only if the new evidence is stronger

   Evidence types (encoded as integers):
     0 = CRUMBS  (very weak)
     1 = NOISES  (moderate)
     2 = VISUAL  (strong)

   Food status: 0 = EMPTY, 1 = FOODY
   Box choice:  0 = BOX1,  1 = BOX2"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Evidence model
;; ---------------------------------------------------------------------------
;; P(foody evidence | evidence-type, food-status)
;; Rows: [crumbs, noises, visual], Columns: [empty, foody]

(def evidence-table
  (mx/array #js [#js [0.50 0.60]    ;; crumbs: very weak
                 #js [0.30 0.60]    ;; noises: moderate
                 #js [0.01 0.90]])) ;; visual: strong

(mx/eval! evidence-table)

(defn p-evidence
  "P(foody evidence | evidence-type, food-status).
   ev-type and food-status are MLX int32 scalars or enumerated tensors."
  [ev-type food-status]
  (mx/idx (mx/idx evidence-table ev-type) food-status))

;; ---------------------------------------------------------------------------
;; World models: food status + evidence generation
;; ---------------------------------------------------------------------------

(defn world-model-1
  "World generates food for both boxes + evidence e1 for box 1.
   Traces: :box1-fs, :box2-fs, :e1."
  [et1-val]
  (gen []
    (let [box1-fs (trace :box1-fs (dist/weighted [1 1]))
          box2-fs (trace :box2-fs (dist/weighted [1 1]))
          p-ev (p-evidence (mx/scalar et1-val mx/int32) box1-fs)
          e1 (trace :e1 (dist/bernoulli p-ev))]
      box1-fs)))

(defn world-model-2
  "World generates food for both boxes + evidence for each.
   Traces: :box1-fs, :box2-fs, :e1, :e2."
  [et1-val et2-val]
  (gen []
    (let [box1-fs (trace :box1-fs (dist/weighted [1 1]))
          box2-fs (trace :box2-fs (dist/weighted [1 1]))
          p-ev1 (p-evidence (mx/scalar et1-val mx/int32) box1-fs)
          e1 (trace :e1 (dist/bernoulli p-ev1))
          p-ev2 (p-evidence (mx/scalar et2-val mx/int32) box2-fs)
          e2 (trace :e2 (dist/bernoulli p-ev2))]
      box1-fs)))

;; ---------------------------------------------------------------------------
;; Chimp's decision: thinks about world, observes evidence, chooses
;; ---------------------------------------------------------------------------
;; Architecture matches memo: the chimp runs exact enumeration on
;; the world model (thinks), conditions on observed evidence (observes),
;; reads posterior food probabilities (pr), then makes a softmax choice.

(defn first-choice
  "P(chimp chooses box b | evidence type et1, observes foody evidence).

   The chimp:
     1. THINKS about the world model (exact enumeration)
     2. OBSERVES e1=FOODY (conditions on evidence)
     3. Reads P(food in each box) via exact/pr
     4. Makes a softmax choice weighted by expected utility

   Returns {:b {0 prob, 1 prob}}."
  [et1-val]
  (let [model (world-model-1 et1-val)
        ;; P(box has food | evidence) via exact/pr with conditioning
        p-food-box1 (exact/pr model :box1-fs 1 :given :e1 1)
        p-food-box2 (exact/pr model :box2-fs 1 :given :e1 1)
        ;; Softmax choice: exp(10 * EU) where EU = P(food)
        logits (mx/multiply (mx/scalar 10.0)
                 (mx/array #js [p-food-box1 p-food-box2]))
        ;; Enumerate the choice
        choice-model (gen [] (trace :b (dist/categorical logits)))
        cr (exact/exact-posterior choice-model [] nil)]
    (:marginals cr)))

(defn second-choice
  "P(chimp chooses box b | et1 for box1, et2 for box2, both foody).

   Same as first-choice but the chimp observes two pieces of evidence.
   Conditions on both e1=1 and e2=1 via exact-posterior, then reads
   posterior food probabilities for each box.

   Returns {:b {0 prob, 1 prob}}."
  [et1-val et2-val]
  (let [model (world-model-2 et1-val et2-val)
        ;; Chimp observes both pieces of foody evidence
        r (exact/exact-posterior model []
            (cm/choicemap :e1 (mx/scalar 1.0)
                          :e2 (mx/scalar 1.0)))
        p-food-box1 (get-in r [:marginals :box1-fs 1])
        p-food-box2 (get-in r [:marginals :box2-fs 1])
        ;; Softmax choice
        logits (mx/multiply (mx/scalar 10.0)
                 (mx/array #js [p-food-box1 p-food-box2]))
        choice-model (gen [] (trace :b (dist/categorical logits)))
        cr (exact/exact-posterior choice-model [] nil)]
    (:marginals cr)))

;; ---------------------------------------------------------------------------
;; Evidence type labels
;; ---------------------------------------------------------------------------

(def ev-names {0 "CRUMBS" 1 "NOISES" 2 "VISUAL"})

;; ---------------------------------------------------------------------------
;; Run: First choice
;; ---------------------------------------------------------------------------

(println "Chimp Belief Revision — Exact Enumeration")
(println "==========================================\n")

(println "First choice: one piece of evidence for box 1")
(println "  The chimp should follow the evidence, more strongly for stronger evidence.\n")

(def first-results
  (into {}
    (for [et [0 1 2]]
      (let [m (first-choice et)
            p-box1 (get-in m [:b 0])]
        (println (str "  " (ev-names et) ": P(box1) = " (.toFixed p-box1 4)
                      "  P(box2) = " (.toFixed (get-in m [:b 1]) 4)))
        [et p-box1]))))

(println)

;; ---------------------------------------------------------------------------
;; Run: Second choice
;; ---------------------------------------------------------------------------

(println "Second choice: evidence for box 1, then evidence for box 2")
(println "  The chimp revises iff the second evidence is stronger.\n")

(def second-results
  (into {}
    (for [et1 [0 1 2]
          et2 [0 1 2]]
      (let [m (second-choice et1 et2)
            p-box1 (get-in m [:b 0])
            p-box2 (get-in m [:b 1])]
        (println (str "  " (ev-names et1) " -> " (ev-names et2)
                      ":  P(box1) = " (.toFixed p-box1 4)
                      "  P(box2) = " (.toFixed p-box2 4)))
        [[et1 et2] {:box1 p-box1 :box2 p-box2}]))))

(println)

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println "Verification")
(println "------------")

(def pass? (atom true))

(defn check [name pred]
  (if pred
    (println (str "  PASS: " name))
    (do (reset! pass? false)
        (println (str "  FAIL: " name)))))

(defn check-close [name expected actual tol]
  (if (<= (abs (- expected actual)) tol)
    (println (str "  PASS: " name))
    (do (reset! pass? false)
        (println (str "  FAIL: " name " expected=" (.toFixed expected 4)
                      " got=" (.toFixed actual 4))))))

;; First choice: stronger evidence -> stronger preference
(check "CRUMBS: prefers box1 (> 0.5)"
       (> (first-results 0) 0.5))
(check "NOISES: prefers box1 more than CRUMBS"
       (> (first-results 1) (first-results 0)))
(check "VISUAL: prefers box1 most (> 0.99)"
       (> (first-results 2) 0.99))

;; Reference values from memo (Schleihauf et al. model)
(check-close "CRUMBS first-choice P(box1)" 0.6117 (first-results 0) 0.01)
(check-close "NOISES first-choice P(box1)" 0.8411 (first-results 1) 0.01)
(check-close "VISUAL first-choice P(box1)" 0.9925 (first-results 2) 0.01)

;; Second choice: same evidence -> 50/50
(check-close "CRUMBS/CRUMBS -> 0.5" 0.5 (:box1 (second-results [0 0])) 0.01)
(check-close "NOISES/NOISES -> 0.5" 0.5 (:box1 (second-results [1 1])) 0.01)
(check-close "VISUAL/VISUAL -> 0.5" 0.5 (:box1 (second-results [2 2])) 0.01)

;; Second choice: stronger second evidence -> revise to box 2
(check "CRUMBS then NOISES -> revise (box2 > 0.5)"
       (> (:box2 (second-results [0 1])) 0.5))
(check "NOISES then VISUAL -> revise strongly (box2 > 0.9)"
       (> (:box2 (second-results [1 2])) 0.9))

;; Second choice: weaker second evidence -> stay with box 1
(check "NOISES then CRUMBS -> stay (box1 > 0.5)"
       (> (:box1 (second-results [1 0])) 0.5))
(check "VISUAL then NOISES -> stay (box1 > 0.9)"
       (> (:box1 (second-results [2 1])) 0.9))

;; Key reference values from memo
(check-close "VISUAL/NOISES P(box2)" 0.0383 (:box2 (second-results [2 1])) 0.01)
(check-close "NOISES/VISUAL P(box2)" 0.9617 (:box2 (second-results [1 2])) 0.01)

;; Demonstrate exact/thinks and exact/observes directly
;; exact/thinks: wrap world model as ExactGF for use with splice
(let [world-gf (exact/thinks (world-model-1 1))]
  (check "exact/thinks produces ExactGF"
         (instance? genmlx.inference.exact/ExactGF world-gf)))

;; exact/observes: P(box1-fs | e1=1) for NOISES — should match exact/pr
(let [posterior (exact/observes (world-model-1 1) :e1 1 :box1-fs)
      p-food (mx/item (mx/idx posterior 1))]
  (check-close "exact/observes matches exact/pr"
               (exact/pr (world-model-1 1) :box1-fs 1 :given :e1 1)
               p-food 1e-5))

(println)
(if @pass?
  (println "All checks passed.")
  (do (println "SOME CHECKS FAILED.")
      (js/process.exit 1)))
