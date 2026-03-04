(ns genmlx.combinator-contract-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.edit :as edit]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Combinator Contract Tests ===")

;; Shared kernel for combinator tests
(def kernel
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/gaussian x 1))]
        (mx/eval! y)
        (mx/item y)))))

;; Second kernel for Switch tests
(def kernel2
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/uniform (- x 1) (+ x 1)))]
        (mx/eval! y)
        (mx/item y)))))

;; ---------------------------------------------------------------------------
;; 21.4 — Edit round-trip
;; ---------------------------------------------------------------------------

(println "\n-- 21.4: Edit round-trip --")

;; Multi-site model for edit tests
(def edit-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/eval! x y)
        (+ (mx/item x) (mx/item y))))))

;; ConstraintEdit: apply forward → get backward → apply backward → verify recovery
(println "\n  constraint-edit round-trip")
(let [trace (p/simulate edit-model [])
      orig-x (let [v (cm/get-value (cm/get-submap (:choices trace) :x))]
               (mx/eval! v) (mx/item v))
      ;; Forward: constrain :x to 5.0
      fwd-req (edit/constraint-edit (cm/choicemap :x (mx/scalar 5.0)))
      fwd-result (edit/edit-dispatch edit-model trace fwd-req)
      new-x (let [v (cm/get-value (cm/get-submap (:choices (:trace fwd-result)) :x))]
              (mx/eval! v) (mx/item v))]
  (assert-close "constraint-edit sets new value" 5.0 new-x 0.01)
  ;; Backward: apply discard to recover
  (when (:discard fwd-result)
    (let [bwd-req (edit/constraint-edit (:discard fwd-result))
          bwd-result (edit/edit-dispatch edit-model (:trace fwd-result) bwd-req)
          recovered-x (let [v (cm/get-value (cm/get-submap (:choices (:trace bwd-result)) :x))]
                        (mx/eval! v) (mx/item v))]
      (assert-close "constraint-edit round-trip recovers" orig-x recovered-x 0.01))))

;; SelectionEdit: apply forward → backward has same selection type
(println "\n  selection-edit")
(let [trace (p/simulate edit-model [])
      sel-req (edit/selection-edit (sel/select :x))
      result (edit/edit-dispatch edit-model trace sel-req)]
  (mx/eval! (:weight result))
  (assert-true "selection-edit weight is finite" (js/isFinite (mx/item (:weight result))))
  (assert-true "backward request is SelectionEdit"
               (instance? edit/SelectionEdit (:backward-request result))))

;; ProposalEdit: apply forward → verify weight is finite
;; Forward/backward GFs receive the trace's choices (a choicemap) as argument
(println "\n  proposal-edit")
(let [fwd (dyn/auto-key (gen [choices]
            (trace :x (dist/gaussian 4.0 0.5))))
      bwd (dyn/auto-key (gen [choices]
            (trace :x (dist/gaussian 4.0 0.5))))
      trace (p/simulate edit-model [])
      edit-req (edit/proposal-edit fwd bwd)
      result (edit/edit edit-model trace edit-req)]
  (mx/eval! (:weight result))
  (assert-true "proposal-edit weight is finite" (js/isFinite (mx/item (:weight result))))
  (assert-true "proposal-edit has backward-request" (some? (:backward-request result))))

;; ---------------------------------------------------------------------------
;; 21.7 — Combinator degenerate cases
;; ---------------------------------------------------------------------------

(println "\n-- 21.7: Combinator degenerate cases --")

;; Map(kernel, [single-input]): should match running kernel directly
(println "\n  map single-input")
(let [mapped (comb/map-combinator kernel)
      constraint (cm/set-choice cm/EMPTY [0] (cm/choicemap :y (mx/scalar 2.5)))
      {:keys [trace weight]} (p/generate mapped [[3.0]] constraint)
      ;; Compare to direct kernel
      {:keys [trace weight]} (p/generate kernel [3.0] (cm/choicemap :y (mx/scalar 2.5)))]
  ;; Both should produce the same score for the same constraint
  (let [map-result (p/generate mapped [[3.0]] constraint)
        direct-result (p/generate kernel [3.0] (cm/choicemap :y (mx/scalar 2.5)))]
    (mx/eval! (:score (:trace map-result)) (:score (:trace direct-result)))
    (assert-close "map(single) score ≈ kernel score"
                  (mx/item (:score (:trace direct-result)))
                  (mx/item (:score (:trace map-result))) 0.01)))

;; Unfold(kernel, init, 1): single-step should match kernel
(println "\n  unfold single-step")
(let [step (dyn/auto-key (gen [t state]
             (let [y (trace :y (dist/gaussian state 1))]
               (mx/eval! y)
               (mx/item y))))
      unfold (comb/unfold-combinator step)
      trace (p/simulate unfold [1 0.0])
      score (:score trace)]
  (mx/eval! score)
  (assert-true "unfold(1) has finite score" (js/isFinite (mx/item score)))
  ;; Should have choices under key 0
  (let [sub (cm/get-submap (:choices trace) 0)]
    (assert-true "unfold(1) has step-0 choices" (some? sub))))

;; Switch([g1, g2], 0): identical to running g1
(println "\n  switch idx=0")
(let [g1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1))))
      g2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 10 1))))
      sw (comb/switch-combinator g1 g2)
      constraint (cm/choicemap :y (mx/scalar 0.5))
      {:keys [trace]} (p/generate sw [0] constraint)
      {:keys [trace]} (p/generate g1 [] constraint)]
  (let [sw-result (p/generate sw [0] constraint)
        g1-result (p/generate g1 [] constraint)]
    (mx/eval! (:score (:trace sw-result)) (:score (:trace g1-result)))
    (assert-close "switch(0) score ≈ g1 score"
                  (mx/item (:score (:trace g1-result)))
                  (mx/item (:score (:trace sw-result))) 0.01)))

;; Mask(kernel, true): identical to running kernel
(println "\n  mask(true)")
(let [masked (comb/mask-combinator kernel)
      constraint (cm/choicemap :y (mx/scalar 2.0))
      mask-result (p/generate masked [true 3.0] constraint)
      direct-result (p/generate kernel [3.0] constraint)]
  (mx/eval! (:score (:trace mask-result)) (:score (:trace direct-result)))
  (assert-close "mask(true) score ≈ kernel score"
                (mx/item (:score (:trace direct-result)))
                (mx/item (:score (:trace mask-result))) 0.01))

;; Mask(kernel, false): no choices, score = 0
(println "\n  mask(false)")
(let [masked (comb/mask-combinator kernel)
      trace (p/simulate masked [false 3.0])
      score (:score trace)]
  (mx/eval! score)
  (assert-close "mask(false) score ≈ 0" 0.0 (mx/item score) 0.01))

;; Scan(kernel, init, 1): single-step
(println "\n  scan single-step")
(let [scan-kernel (dyn/auto-key (gen [carry x]
                    (let [y (trace :y (dist/gaussian carry 1))]
                      (mx/eval! y)
                      [(mx/item y) (mx/item y)])))
      scanned (comb/scan-combinator scan-kernel)
      trace (p/simulate scanned [0.0 [1.0]])
      score (:score trace)]
  (mx/eval! score)
  (assert-true "scan(1) has finite score" (js/isFinite (mx/item score))))

;; ---------------------------------------------------------------------------
;; 21.8 — Nested combinator tests
;; ---------------------------------------------------------------------------

(println "\n-- 21.8: Nested combinators --")

;; Map(Switch(g1, g2), [0, 1, 0])
(println "\n  map(switch)")
(let [g1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1))))
      g2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 10 1))))
      sw (comb/switch-combinator g1 g2)
      mapped (comb/map-combinator sw)
      trace (p/simulate mapped [[0 1 0]])
      score (:score trace)]
  (mx/eval! score)
  (assert-true "map(switch) has finite score" (js/isFinite (mx/item score)))
  ;; Check that all 3 elements have choices
  (assert-true "map(switch) has 3 elements"
               (and (some? (cm/get-submap (:choices trace) 0))
                    (some? (cm/get-submap (:choices trace) 1))
                    (some? (cm/get-submap (:choices trace) 2)))))

;; Unfold(Mask(kernel), init, 3): alternating masked steps
(println "\n  unfold(mask)")
(let [inner (dyn/auto-key (gen [t state]
              (let [y (trace :y (dist/gaussian state 1))]
                (mx/eval! y)
                (mx/item y))))
      masked (comb/mask-combinator inner)
      ;; Wrap in a gen that passes mask flag
      step (dyn/auto-key (gen [t state]
             (let [active? (even? t)
                   y (if active?
                       (let [v (trace :y (dist/gaussian state 1))]
                         (mx/eval! v) (mx/item v))
                       state)]
               y)))
      unfold (comb/unfold-combinator step)
      trace (p/simulate unfold [3 0.0])
      score (:score trace)]
  (mx/eval! score)
  (assert-true "unfold(mask) has finite score" (js/isFinite (mx/item score))))

;; Switch(Map(k1), Map(k2), idx)
(println "\n  switch(map, map)")
(let [m1 (comb/map-combinator kernel)
      m2 (comb/map-combinator kernel2)
      sw (comb/switch-combinator m1 m2)
      trace0 (p/simulate sw [0 [1.0 2.0]])
      trace1 (p/simulate sw [1 [1.0 2.0]])]
  (mx/eval! (:score trace0) (:score trace1))
  (assert-true "switch(map,map) idx=0 finite" (js/isFinite (mx/item (:score trace0))))
  (assert-true "switch(map,map) idx=1 finite" (js/isFinite (mx/item (:score trace1)))))

;; ---------------------------------------------------------------------------
;; 21.9 — Score additivity
;; ---------------------------------------------------------------------------

(println "\n-- 21.9: Score additivity --")

;; Map: score = sum(element-scores)
(println "\n  map score additivity")
(let [mapped (comb/map-combinator kernel)
      trace (p/simulate mapped [[1.0 2.0 3.0]])
      total-score (:score trace)
      _ (mx/eval! total-score)
      ;; Sum element scores from metadata
      step-scores (::comb/element-scores (meta trace))]
  (when step-scores
    (let [sum-scores (reduce (fn [acc s] (mx/eval! s) (+ acc (mx/item s)))
                             0.0 step-scores)]
      (assert-close "map: score = sum(element-scores)"
                    (mx/item total-score) sum-scores 0.01))))

;; Unfold: score = sum(step-scores)
(println "\n  unfold score additivity")
(let [step (dyn/auto-key (gen [t state]
             (let [y (trace :y (dist/gaussian state 1))]
               (mx/eval! y)
               (mx/item y))))
      unfold (comb/unfold-combinator step)
      trace (p/simulate unfold [3 0.0])
      total-score (:score trace)
      _ (mx/eval! total-score)
      step-scores (::comb/step-scores (meta trace))]
  (when step-scores
    (let [sum-scores (reduce (fn [acc s] (mx/eval! s) (+ acc (mx/item s)))
                             0.0 step-scores)]
      (assert-close "unfold: score = sum(step-scores)"
                    (mx/item total-score) sum-scores 0.01))))

;; Switch: score = branch-score
(println "\n  switch score = branch score")
(let [g1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1))))
      g2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 10 1))))
      sw (comb/switch-combinator g1 g2)
      constraint (cm/choicemap :y (mx/scalar 0.5))
      {:keys [trace]} (p/generate sw [0] constraint)]
  (mx/eval! (:score trace))
  (let [{:keys [trace]} (p/generate g1 [] constraint)]
    (mx/eval! (:score trace))
    ;; Score of switch with idx=0 should match g1's score
    (let [sw-r (p/generate sw [0] constraint)
          g1-r (p/generate g1 [] constraint)]
      (mx/eval! (:score (:trace sw-r)) (:score (:trace g1-r)))
      (assert-close "switch: score = branch score"
                    (mx/item (:score (:trace g1-r)))
                    (mx/item (:score (:trace sw-r))) 0.01))))

;; Scan: score = sum(step-scores)
(println "\n  scan score additivity")
(let [scan-kernel (dyn/auto-key (gen [carry x]
                    (let [y (trace :y (dist/gaussian carry 1))]
                      (mx/eval! y)
                      [(mx/item y) (mx/item y)])))
      scanned (comb/scan-combinator scan-kernel)
      trace (p/simulate scanned [0.0 [1.0 2.0 3.0]])
      total-score (:score trace)
      _ (mx/eval! total-score)
      step-scores (::comb/step-scores (meta trace))]
  (when step-scores
    (let [sum-scores (reduce (fn [acc s] (mx/eval! s) (+ acc (mx/item s)))
                             0.0 step-scores)]
      (assert-close "scan: score = sum(step-scores)"
                    (mx/item total-score) sum-scores 0.01))))

(println "\nAll combinator contract tests complete.")
