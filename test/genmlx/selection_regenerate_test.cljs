;; @tier fast core
(ns genmlx.selection-regenerate-test
  "End-to-end counterexample/regression tests for genmlx-yey5: selection
   descent must resample ONLY selected addresses, keeping unselected splice
   subtrees and combinator elements byte-for-byte unchanged.

   Oracle: value preservation. An UNSELECTED address must retain its exact
   prior value after regenerate; a SELECTED continuous address must change.
   This is an independent invariant (not the function under test) that directly
   falsifies the old bug, where get-subselection returned `all` unconditionally
   and resampled entire unselected subtrees / every combinator element."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.gfi :as gfi]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- leaf
  "Realized numeric value at a path in a trace's choices."
  [trace path]
  (h/realize (cm/get-choice (:choices trace) path)))

;; ---------------------------------------------------------------------------
;; Splice case (scalar executor descent — runtime.cljs splice-fn)
;; ---------------------------------------------------------------------------

(def sub-model
  (dyn/auto-key
    (gen [mu]
      (let [z (trace :z (dist/gaussian mu 1))
            w (trace :w (dist/gaussian mu 1))]
        (mx/add z w)))))

(def splice-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 10))]
        (splice :sub sub-model x)
        x))))

(deftest splice-unselected-subtree-preserved
  (testing "regenerate (select :x) must NOT resample the unselected :sub subtree"
    (let [t0   (p/simulate splice-model [])
          x0   (leaf t0 [:x])
          z0   (leaf t0 [:sub :z])
          w0   (leaf t0 [:sub :w])
          t1   (:trace (p/regenerate splice-model t0 (sel/select :x)))
          x1   (leaf t1 [:x])
          z1   (leaf t1 [:sub :z])
          w1   (leaf t1 [:sub :w])]
      ;; THE FIX: unselected splice subtree is preserved exactly.
      (is (h/close? z0 z1 0.0) ":sub/:z preserved (was resampled before the fix)")
      (is (h/close? w0 w1 0.0) ":sub/:w preserved (was resampled before the fix)")
      ;; The selected continuous address is resampled (changes w.p. 1).
      (is (not (h/close? x0 x1 1e-9)) ":x resampled"))))

(deftest splice-descend-partial-select
  (testing "regenerate (hierarchical :sub (select :z)) resamples only :sub/:z"
    (let [t0   (p/simulate splice-model [])
          x0   (leaf t0 [:x])
          z0   (leaf t0 [:sub :z])
          w0   (leaf t0 [:sub :w])
          t1   (:trace (p/regenerate splice-model t0
                                     (sel/hierarchical :sub (sel/select :z))))
          x1   (leaf t1 [:x])
          z1   (leaf t1 [:sub :z])
          w1   (leaf t1 [:sub :w])]
      (is (h/close? x0 x1 0.0) ":x preserved (not selected)")
      (is (h/close? w0 w1 0.0) ":sub/:w preserved (sibling not selected)")
      (is (not (h/close? z0 z1 1e-9)) ":sub/:z resampled (selected via descent)"))))

(deftest splice-empty-selection-preserves-all
  (testing "regenerate (select :nonexistent) preserves the whole trace"
    (let [t0 (p/simulate splice-model [])
          t1 (:trace (p/regenerate splice-model t0 (sel/select :nonexistent)))]
      (is (h/close? (leaf t0 [:x])      (leaf t1 [:x])      0.0) ":x preserved")
      (is (h/close? (leaf t0 [:sub :z]) (leaf t1 [:sub :z]) 0.0) ":sub/:z preserved")
      (is (h/close? (leaf t0 [:sub :w]) (leaf t1 [:sub :w]) 0.0) ":sub/:w preserved"))))

;; ---------------------------------------------------------------------------
;; Map combinator case (element descent — combinators.cljs)
;; ---------------------------------------------------------------------------

;; Two-site kernel so we can test partial selection WITHIN an element.
(def kernel2
  (dyn/auto-key
    (gen [x]
      (let [a (trace :a (dist/gaussian x 1))
            b (trace :b (dist/gaussian x 1))]
        (mx/add a b)))))

(defn- map-trace [kernel inputs]
  (p/simulate (comb/map-combinator kernel) [inputs]))

(deftest map-select-one-element
  (testing "regenerate (select 1) resamples only element 1; others preserved"
    (let [inputs [0.0 1.0 2.0 3.0]
          mapped (comb/map-combinator kernel2)
          t0     (p/simulate mapped [inputs])
          olds   (mapv (fn [i] [(leaf t0 [i :a]) (leaf t0 [i :b])]) (range 4))
          t1     (:trace (p/regenerate mapped t0 (sel/select 1)))
          news   (mapv (fn [i] [(leaf t1 [i :a]) (leaf t1 [i :b])]) (range 4))]
      ;; THE FIX: only element 1 changes; before the fix EVERY element was
      ;; resampled because get-subselection handed `all` to every index.
      (doseq [i [0 2 3]]
        (is (h/close? (first (olds i)) (first (news i)) 0.0)
            (str "element " i " :a preserved"))
        (is (h/close? (second (olds i)) (second (news i)) 0.0)
            (str "element " i " :b preserved")))
      (is (not (h/close? (first (olds 1)) (first (news 1)) 1e-9)) "element 1 :a resampled")
      (is (not (h/close? (second (olds 1)) (second (news 1)) 1e-9)) "element 1 :b resampled"))))

(deftest map-descend-into-element
  (testing "regenerate (hierarchical 1 (select :a)) resamples only element 1's :a"
    (let [inputs [0.0 1.0 2.0 3.0]
          mapped (comb/map-combinator kernel2)
          t0     (p/simulate mapped [inputs])
          olds   (mapv (fn [i] [(leaf t0 [i :a]) (leaf t0 [i :b])]) (range 4))
          t1     (:trace (p/regenerate mapped t0
                                       (sel/hierarchical 1 (sel/select :a))))
          news   (mapv (fn [i] [(leaf t1 [i :a]) (leaf t1 [i :b])]) (range 4))]
      (doseq [i [0 2 3]]
        (is (h/close? (first (olds i)) (first (news i)) 0.0)
            (str "element " i " :a preserved"))
        (is (h/close? (second (olds i)) (second (news i)) 0.0)
            (str "element " i " :b preserved")))
      (is (not (h/close? (first (olds 1)) (first (news 1)) 1e-9)) "element 1 :a resampled")
      (is (h/close? (second (olds 1)) (second (news 1)) 0.0) "element 1 :b preserved"))))

;; ---------------------------------------------------------------------------
;; Compiled-vs-handler parity: the fix must behave identically on both paths.
;; strip-compiled forces the handler (interpreter) regenerate for the kernel.
;; ---------------------------------------------------------------------------

(deftest map-compiled-handler-parity
  (testing "select-one-element preservation holds on the stripped (handler) path too"
    (let [inputs  [0.0 1.0 2.0 3.0]
          stripped (dyn/auto-key (gfi/strip-compiled kernel2))
          mapped  (comb/map-combinator stripped)
          t0      (p/simulate mapped [inputs])
          olds    (mapv (fn [i] [(leaf t0 [i :a]) (leaf t0 [i :b])]) (range 4))
          t1      (:trace (p/regenerate mapped t0 (sel/select 2)))
          news    (mapv (fn [i] [(leaf t1 [i :a]) (leaf t1 [i :b])]) (range 4))]
      (doseq [i [0 1 3]]
        (is (h/close? (first (olds i)) (first (news i)) 0.0)
            (str "[handler] element " i " :a preserved"))
        (is (h/close? (second (olds i)) (second (news i)) 0.0)
            (str "[handler] element " i " :b preserved")))
      (is (not (h/close? (first (olds 2)) (first (news 2)) 1e-9)) "[handler] element 2 :a resampled")
      (is (not (h/close? (second (olds 2)) (second (news 2)) 1e-9)) "[handler] element 2 :b resampled"))))

(cljs.test/run-tests)
