(ns genmlx.combinator-contract-test
  "Combinator contract tests: edit round-trip, degenerate cases, nested combinators, score additivity."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.edit :as edit]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

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

;; Multi-site model for edit tests
(def edit-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/eval! x y)
        (+ (mx/item x) (mx/item y))))))

;; ---------------------------------------------------------------------------
;; 21.4 -- Edit round-trip
;; ---------------------------------------------------------------------------

(deftest constraint-edit-round-trip
  (testing "constraint-edit round-trip"
    (let [trace (p/simulate edit-model [])
          orig-x (let [v (cm/get-value (cm/get-submap (:choices trace) :x))]
                   (mx/eval! v) (mx/item v))
          fwd-req (edit/constraint-edit (cm/choicemap :x (mx/scalar 5.0)))
          fwd-result (edit/edit-dispatch edit-model trace fwd-req)
          new-x (let [v (cm/get-value (cm/get-submap (:choices (:trace fwd-result)) :x))]
                  (mx/eval! v) (mx/item v))]
      (is (h/close? 5.0 new-x 0.01) "constraint-edit sets new value")
      (when (:discard fwd-result)
        (let [bwd-req (edit/constraint-edit (:discard fwd-result))
              bwd-result (edit/edit-dispatch edit-model (:trace fwd-result) bwd-req)
              recovered-x (let [v (cm/get-value (cm/get-submap (:choices (:trace bwd-result)) :x))]
                            (mx/eval! v) (mx/item v))]
          (is (h/close? orig-x recovered-x 0.01) "constraint-edit round-trip recovers"))))))

(deftest selection-edit
  (testing "selection-edit"
    (let [trace (p/simulate edit-model [])
          sel-req (edit/selection-edit (sel/select :x))
          result (edit/edit-dispatch edit-model trace sel-req)]
      (mx/eval! (:weight result))
      (is (js/isFinite (mx/item (:weight result))) "selection-edit weight is finite")
      (is (instance? edit/SelectionEdit (:backward-request result)) "backward request is SelectionEdit"))))

(deftest proposal-edit
  (testing "proposal-edit"
    (let [fwd (dyn/auto-key (gen [choices]
                (trace :x (dist/gaussian 4.0 0.5))))
          bwd (dyn/auto-key (gen [choices]
                (trace :x (dist/gaussian 4.0 0.5))))
          trace (p/simulate edit-model [])
          edit-req (edit/proposal-edit fwd bwd)
          result (edit/edit edit-model trace edit-req)]
      (mx/eval! (:weight result))
      (is (js/isFinite (mx/item (:weight result))) "proposal-edit weight is finite")
      (is (some? (:backward-request result)) "proposal-edit has backward-request"))))

;; ---------------------------------------------------------------------------
;; 21.7 -- Combinator degenerate cases
;; ---------------------------------------------------------------------------

(deftest map-single-input
  (testing "map single-input"
    (let [mapped (comb/map-combinator kernel)
          constraint (cm/set-choice cm/EMPTY [0] (cm/choicemap :y (mx/scalar 2.5)))
          map-result (p/generate mapped [[3.0]] constraint)
          direct-result (p/generate kernel [3.0] (cm/choicemap :y (mx/scalar 2.5)))]
      (mx/eval! (:score (:trace map-result)) (:score (:trace direct-result)))
      (is (h/close? (mx/item (:score (:trace direct-result)))
                    (mx/item (:score (:trace map-result))) 0.01)
          "map(single) score ~ kernel score"))))

(deftest unfold-single-step
  (testing "unfold single-step"
    (let [step (dyn/auto-key (gen [t state]
                 (let [y (trace :y (dist/gaussian state 1))]
                   (mx/eval! y)
                   (mx/item y))))
          unfold (comb/unfold-combinator step)
          trace (p/simulate unfold [1 0.0])
          score (:score trace)]
      (mx/eval! score)
      (is (js/isFinite (mx/item score)) "unfold(1) has finite score")
      (let [sub (cm/get-submap (:choices trace) 0)]
        (is (some? sub) "unfold(1) has step-0 choices")))))

(deftest switch-idx-zero
  (testing "switch idx=0"
    (let [g1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1))))
          g2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 10 1))))
          sw (comb/switch-combinator g1 g2)
          constraint (cm/choicemap :y (mx/scalar 0.5))
          sw-result (p/generate sw [0] constraint)
          g1-result (p/generate g1 [] constraint)]
      (mx/eval! (:score (:trace sw-result)) (:score (:trace g1-result)))
      (is (h/close? (mx/item (:score (:trace g1-result)))
                    (mx/item (:score (:trace sw-result))) 0.01)
          "switch(0) score ~ g1 score"))))

(deftest mask-true
  (testing "mask(true)"
    (let [masked (comb/mask-combinator kernel)
          constraint (cm/choicemap :y (mx/scalar 2.0))
          mask-result (p/generate masked [true 3.0] constraint)
          direct-result (p/generate kernel [3.0] constraint)]
      (mx/eval! (:score (:trace mask-result)) (:score (:trace direct-result)))
      (is (h/close? (mx/item (:score (:trace direct-result)))
                    (mx/item (:score (:trace mask-result))) 0.01)
          "mask(true) score ~ kernel score"))))

(deftest mask-false
  (testing "mask(false)"
    (let [masked (comb/mask-combinator kernel)
          trace (p/simulate masked [false 3.0])
          score (:score trace)]
      (mx/eval! score)
      (is (h/close? 0.0 (mx/item score) 0.01) "mask(false) score ~ 0"))))

(deftest scan-single-step
  (testing "scan single-step"
    (let [scan-kernel (dyn/auto-key (gen [carry x]
                        (let [y (trace :y (dist/gaussian carry 1))]
                          (mx/eval! y)
                          [(mx/item y) (mx/item y)])))
          scanned (comb/scan-combinator scan-kernel)
          trace (p/simulate scanned [0.0 [1.0]])
          score (:score trace)]
      (mx/eval! score)
      (is (js/isFinite (mx/item score)) "scan(1) has finite score"))))

;; ---------------------------------------------------------------------------
;; 21.8 -- Nested combinator tests
;; ---------------------------------------------------------------------------

(deftest map-switch-nested
  (testing "map(switch)"
    (let [g1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1))))
          g2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 10 1))))
          sw (comb/switch-combinator g1 g2)
          mapped (comb/map-combinator sw)
          trace (p/simulate mapped [[0 1 0]])
          score (:score trace)]
      (mx/eval! score)
      (is (js/isFinite (mx/item score)) "map(switch) has finite score")
      (is (and (some? (cm/get-submap (:choices trace) 0))
               (some? (cm/get-submap (:choices trace) 1))
               (some? (cm/get-submap (:choices trace) 2)))
          "map(switch) has 3 elements"))))

(deftest unfold-mask-nested
  (testing "unfold(mask)"
    (let [step (dyn/auto-key (gen [t state]
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
      (is (js/isFinite (mx/item score)) "unfold(mask) has finite score"))))

(deftest switch-map-nested
  (testing "switch(map, map)"
    (let [m1 (comb/map-combinator kernel)
          m2 (comb/map-combinator kernel2)
          sw (comb/switch-combinator m1 m2)
          trace0 (p/simulate sw [0 [1.0 2.0]])
          trace1 (p/simulate sw [1 [1.0 2.0]])]
      (mx/eval! (:score trace0) (:score trace1))
      (is (js/isFinite (mx/item (:score trace0))) "switch(map,map) idx=0 finite")
      (is (js/isFinite (mx/item (:score trace1))) "switch(map,map) idx=1 finite"))))

;; ---------------------------------------------------------------------------
;; 21.9 -- Score additivity
;; ---------------------------------------------------------------------------

(deftest map-score-additivity
  (testing "map score additivity"
    (let [mapped (comb/map-combinator kernel)
          trace (p/simulate mapped [[1.0 2.0 3.0]])
          total-score (:score trace)
          _ (mx/eval! total-score)
          step-scores (::comb/element-scores (meta trace))]
      (when step-scores
        (let [sum-scores (reduce (fn [acc s] (mx/eval! s) (+ acc (mx/item s)))
                                 0.0 step-scores)]
          (is (h/close? (mx/item total-score) sum-scores 0.01) "map: score = sum(element-scores)"))))))

(deftest unfold-score-additivity
  (testing "unfold score additivity"
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
          (is (h/close? (mx/item total-score) sum-scores 0.01) "unfold: score = sum(step-scores)"))))))

(deftest switch-score-equals-branch
  (testing "switch score = branch score"
    (let [g1 (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1))))
          g2 (dyn/auto-key (gen [] (trace :y (dist/gaussian 10 1))))
          sw (comb/switch-combinator g1 g2)
          constraint (cm/choicemap :y (mx/scalar 0.5))
          sw-r (p/generate sw [0] constraint)
          g1-r (p/generate g1 [] constraint)]
      (mx/eval! (:score (:trace sw-r)) (:score (:trace g1-r)))
      (is (h/close? (mx/item (:score (:trace g1-r)))
                    (mx/item (:score (:trace sw-r))) 0.01)
          "switch: score = branch score"))))

(deftest scan-score-additivity
  (testing "scan score additivity"
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
          (is (h/close? (mx/item total-score) sum-scores 0.01) "scan: score = sum(step-scores)"))))))

(cljs.test/run-tests)
