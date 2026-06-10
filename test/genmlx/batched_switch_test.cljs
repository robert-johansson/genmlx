;; @tier fast
(ns genmlx.batched-switch-test
  "Batched switch combinator tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(def branch-low
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      x)))

(def branch-high
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 10.0) (mx/scalar 1.0)))]
      x)))

(deftest scalar-switch-sanity
  (testing "scalar switch sanity check"
    (let [sw (comb/switch-combinator (dyn/auto-key branch-low) (dyn/auto-key branch-high))
          trace0 (p/simulate sw [0])
          trace1 (p/simulate sw [1])]
      (mx/eval! (:retval trace0))
      (mx/eval! (:retval trace1))
      (is (< (js/Math.abs (mx/item (:retval trace0))) 5) "branch 0 returns value near 0")
      (is (< (js/Math.abs (- 10 (mx/item (:retval trace1)))) 5) "branch 1 returns value near 10"))))

(def model-switch
  (gen [index]
    (let [sw (comb/switch-combinator branch-low branch-high)
          result (splice :choice sw index)]
      result)))

(deftest batched-switch-vsimulate
  (testing "batched switch via vsimulate"
    (let [key (rng/fresh-key)
          index (mx/array (vec (concat (repeat 50 0) (repeat 50 1))) mx/int32)
          vtrace (dyn/vsimulate model-switch [index] 100 key)]
      (is (some? vtrace) "batched switch returns vtrace")
      (mx/eval! (:score vtrace))
      (is (= [100] (mx/shape (:score vtrace))) "score is [100]-shaped")
      (let [choices (:choices vtrace)
            x-vals (cm/get-value (cm/get-submap (cm/get-submap choices :choice) :x))]
        (mx/eval! x-vals)
        (is (= [100] (mx/shape x-vals)) ":x is [100]-shaped")
        (let [first-half-mean (mx/item (mx/mean (mx/slice x-vals 0 50)))
              second-half-mean (mx/item (mx/mean (mx/slice x-vals 50 100)))]
          (is (< (js/Math.abs first-half-mean) 3) "branch 0 particles near 0")
          (is (< (js/Math.abs (- 10 second-half-mean)) 3) "branch 1 particles near 10"))))))

(deftest batched-switch-vgenerate
  (testing "batched switch with constraints (vgenerate)"
    ;; cm/set-choice, not cm/set-value: set-value takes a SINGLE address,
    ;; so the path vector landed as a literal map key and the constraint
    ;; was silently ignored — the test asserted only shapes (genmlx-wurv).
    (let [key (rng/fresh-key)
          index (mx/array (vec (concat (repeat 50 0) (repeat 50 1))) mx/int32)
          obs (cm/set-choice cm/EMPTY [:choice :x] (mx/scalar 5.0))
          vtrace (dyn/vgenerate model-switch [index] obs 100 key)]
      (is (some? vtrace) "vgenerate returns vtrace")
      (mx/eval! (:score vtrace))
      (mx/eval! (:weight vtrace))
      (is (= [100] (mx/shape (:score vtrace))) "score is [100]-shaped")
      (is (= [100] (mx/shape (:weight vtrace))) "weight is [100]-shaped")
      ;; The constraint must actually apply...
      (let [x-vals (h/realize-vec (cm/get-value (cm/get-submap
                                                 (cm/get-submap (:choices vtrace) :choice)
                                                 :x)))]
        (is (every? #(h/close? 5.0 % 1e-6) x-vals)
            "constrained :x is 5.0 for every particle"))
      ;; ...and the weight must equal the selected branch's closed-form
      ;; log-density at the constrained value: N(5; 0,1) for branch 0
      ;; particles, N(5; 10,1) for branch 1 — equal by symmetry.
      (let [w-vals (h/realize-vec (:weight vtrace))
            expected (h/gaussian-lp 5.0 0.0 1.0)]
        (is (every? #(h/close? expected % 1e-4) w-vals)
            "weight = selected branch lp of the constrained value")))))

(def model-mixture
  (gen []
    (let [z (trace :z (dist/bernoulli (mx/scalar 0.5)))
          idx (mx/multiply z (mx/scalar 1 mx/int32))
          sw (comb/switch-combinator branch-low branch-high)
          result (splice :comp sw idx)]
      result)))

(deftest random-per-particle-branch
  (testing "random per-particle branch selection"
    (let [key (rng/fresh-key)
          vtrace (dyn/vsimulate model-mixture [] 200 key)]
      (is (some? vtrace) "random branch vtrace exists")
      (mx/eval! (:score vtrace))
      (is (= [200] (mx/shape (:score vtrace))) "score is [200]-shaped")
      (let [choices (:choices vtrace)
            x-vals (cm/get-value (cm/get-submap (cm/get-submap choices :comp) :x))]
        (mx/eval! x-vals)
        (let [overall-mean (mx/item (mx/mean x-vals))]
          (is (< (js/Math.abs (- 5 overall-mean)) 3) "mixture mean near 5"))))))

;; ---------------------------------------------------------------------------
;; Heterogeneous branch addresses + non-array retvals (genmlx-v740 item 3)
;; ---------------------------------------------------------------------------

(def branch-a
  (gen []
    (trace :a (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    nil))

(def branch-b
  (gen []
    (trace :b (dist/gaussian (mx/scalar 5.0) (mx/scalar 2.0)))
    nil))

(def het-model
  (gen [index]
    (splice :sw (comb/switch-combinator branch-a branch-b) index)))

(deftest batched-switch-heterogeneous-branch-addresses
  (testing "branch-only addresses survive the mask combine; per-particle
            score equals the selected branch's closed-form log-density"
    ;; Pre-fix, where-select-choicemap iterated only the later branch's
    ;; addresses: :a was dropped from the combined choices and the missing
    ;; :b on the accumulator side hit mx/where with nil (NAPI type error).
    (let [n 6
          index (mx/array [0 1 0 1 0 1] mx/int32)
          vt (dyn/vsimulate het-model [index] n (rng/fresh-key 11))
          sw-choices (cm/get-submap (:choices vt) :sw)
          a-vals (h/realize-vec (cm/get-value (cm/get-submap sw-choices :a)))
          b-vals (h/realize-vec (cm/get-value (cm/get-submap sw-choices :b)))
          scores (h/realize-vec (:score vt))]
      (is (= n (count a-vals)) ":a present and [n]-shaped")
      (is (= n (count b-vals)) ":b present and [n]-shaped")
      (doseq [i (range n)]
        (let [expected (if (even? i)
                         (h/gaussian-lp (nth a-vals i) 0.0 1.0)
                         (h/gaussian-lp (nth b-vals i) 5.0 2.0))]
          (is (h/close? expected (nth scores i) 1e-4)
              (str "particle " i " score = selected branch closed-form lp")))))))

(def branch-ret-42
  (gen []
    (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    42))

(def branch-ret-7
  (gen []
    (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    7))

(deftest batched-switch-number-retvals
  (testing "numeric branch retvals mask-select per particle (pre-fix: nil)"
    (let [n 4
          index (mx/array [0 1 1 0] mx/int32)
          model (gen [idx]
                  (splice :sw (comb/switch-combinator branch-ret-42 branch-ret-7) idx))
          vt (dyn/vsimulate model [index] n (rng/fresh-key 12))]
      (is (= [42 7 7 42] (h/realize-vec (:retval vt)))
          "retval is the selected branch's constant per particle"))))

(def branch-done-a
  (gen []
    (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    :done))

(def branch-done-b
  (gen []
    (trace :x (dist/gaussian (mx/scalar 1.0) (mx/scalar 1.0)))
    :done))

(deftest batched-switch-identical-value-retvals
  (testing "identical non-numeric retvals pass through (pre-fix: nil)"
    (let [index (mx/array [0 1] mx/int32)
          model (gen [idx]
                  (splice :sw (comb/switch-combinator branch-done-a branch-done-b) idx))
          vt (dyn/vsimulate model [index] 2 (rng/fresh-key 13))]
      (is (= :done (:retval vt)) "particle-invariant retval survives"))))

(def branch-map-a
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      {:v x})))

(def branch-map-b
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)))]
      {:v x})))

(deftest batched-switch-map-retvals
  (testing "map retvals combine recursively, pairing with the combined choices"
    (let [n 4
          index (mx/array [0 1 0 1] mx/int32)
          model (gen [idx]
                  (splice :sw (comb/switch-combinator branch-map-a branch-map-b) idx))
          vt (dyn/vsimulate model [index] n (rng/fresh-key 14))
          x-comb (cm/get-value (cm/get-submap (cm/get-submap (:choices vt) :sw) :x))]
      (is (map? (:retval vt)) "retval keeps its map structure")
      (is (= (h/realize-vec x-comb) (h/realize-vec (:v (:retval vt))))
          "retval :v mask-selects identically to the :x choices"))))

(def branch-left
  (gen []
    (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    :left))

(def branch-right
  (gen []
    (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    :right))

(deftest batched-switch-uncombinable-retvals-throw
  (testing "heterogeneous non-numeric retvals throw honestly (pre-fix: silent nil)"
    (let [index (mx/array [0 1] mx/int32)
          model (gen [idx]
                  (splice :sw (comb/switch-combinator branch-left branch-right) idx))]
      (is (thrown? js/Error (dyn/vsimulate model [index] 2 (rng/fresh-key 15)))
          "uncombinable retvals are a loud contract violation"))))

;; ---------------------------------------------------------------------------
;; Batched Mix: heterogeneous components + retvals (genmlx-v740 item 3)
;; ---------------------------------------------------------------------------

(def comp-het-a
  (gen []
    (trace :a (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    nil))

(def comp-het-b
  (gen []
    (trace :b (dist/gaussian (mx/scalar 3.0) (mx/scalar 1.0)))
    nil))

(deftest batched-mix-heterogeneous-components
  (testing "component-only addresses survive; per-particle score is the
            selected component's closed-form lp + categorical lp; constrained
            component-idx contributes exactly the categorical lp to weight"
    (let [n 4
          idx (mx/array [0 1 0 1] mx/int32)
          log-w (mx/array [(js/Math.log 0.3) (js/Math.log 0.7)])
          model (gen []
                  (splice :mix (comb/mix-combinator [comp-het-a comp-het-b] log-w)))
          obs (cm/set-choice cm/EMPTY [:mix :component-idx] idx)
          vt (dyn/vgenerate model [] obs n (rng/fresh-key 16))
          mix-choices (cm/get-submap (:choices vt) :mix)
          a-vals (h/realize-vec (cm/get-value (cm/get-submap mix-choices :a)))
          b-vals (h/realize-vec (cm/get-value (cm/get-submap mix-choices :b)))
          scores (h/realize-vec (:score vt))
          weights (h/realize-vec (:weight vt))
          lw0 (js/Math.log 0.3)
          lw1 (js/Math.log 0.7)]
      (is (= n (count a-vals)) ":a present and [n]-shaped")
      (is (= n (count b-vals)) ":b present and [n]-shaped")
      (doseq [i (range n)]
        (let [sel0? (even? i)
              comp-lp (if sel0?
                        (h/gaussian-lp (nth a-vals i) 0.0 1.0)
                        (h/gaussian-lp (nth b-vals i) 3.0 1.0))
              idx-lp (if sel0? lw0 lw1)]
          (is (h/close? (+ comp-lp idx-lp) (nth scores i) 1e-4)
              (str "particle " i " score = component lp + categorical lp"))
          (is (h/close? idx-lp (nth weights i) 1e-4)
              (str "particle " i " weight = categorical lp of constrained idx")))))))

(def comp-ret-1
  (gen []
    (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    1))

(def comp-ret-2
  (gen []
    (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
    2))

(deftest batched-mix-number-retvals
  (testing "numeric component retvals mask-select per particle (pre-fix: nil)"
    (let [idx (mx/array [1 0 1] mx/int32)
          log-w (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)])
          model (gen []
                  (splice :mix (comb/mix-combinator [comp-ret-1 comp-ret-2] log-w)))
          obs (cm/set-choice cm/EMPTY [:mix :component-idx] idx)
          vt (dyn/vgenerate model [] obs 3 (rng/fresh-key 17))]
      (is (= [2 1 2] (h/realize-vec (:retval vt)))
          "retval is the selected component's constant per particle"))))

(cljs.test/run-tests)
