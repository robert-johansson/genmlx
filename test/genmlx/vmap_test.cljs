(ns genmlx.vmap-test
  "Tests for the Vmap combinator."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.vmap :as vmap]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest basic-simulate
  (testing "basic simulate"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          trace (p/simulate vmodel [(mx/array [1.0 2.0 3.0])])]
      (is (instance? tr/Trace trace) "returns trace")
      (let [choices (:choices trace)
            y-val (cm/get-value (cm/get-submap choices :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "y is [3]-shaped"))
      (mx/eval! (:score trace))
      (is (= [] (mx/shape (:score trace))) "score is scalar")
      (is (js/isFinite (mx/item (:score trace))) "score is finite"))))

(deftest generate-with-constraints
  (testing "generate with [N]-shaped constraints"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
          {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)]
      (is (instance? tr/Trace trace) "generate returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "constrained y shape")
        (is (h/close? 1.0 (mx/item (mx/index y-val 0)) 1e-6) "y[0] = 1.0")
        (is (h/close? 2.0 (mx/item (mx/index y-val 1)) 1e-6) "y[1] = 2.0")
        (is (h/close? 3.0 (mx/item (mx/index y-val 2)) 1e-6) "y[2] = 3.0"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "weight is finite")
      (is (not= 0.0 (mx/item weight)) "weight is nonzero"))))

(deftest vmap-update
  (testing "update"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          obs1 (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
          {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs1)
          old-score (:score trace)
          obs2 (cm/choicemap :y (mx/array [1.1 2.1 3.1]))
          {:keys [trace weight discard]} (p/update vmodel trace obs2)]
      (mx/eval! old-score (:score trace) weight)
      (is (instance? tr/Trace trace) "update returns trace")
      (is (js/isFinite (mx/item weight)) "weight is finite")
      (is (h/close? (- (mx/item (:score trace)) (mx/item old-score))
                    (mx/item weight) 1e-5)
          "weight = new_score - old_score")
      (is (satisfies? cm/IChoiceMap discard) "discard is choicemap"))))

(deftest vmap-regenerate
  (testing "regenerate"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
          {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)
          {:keys [trace weight]} (p/regenerate vmodel trace (sel/select :y))]
      (is (instance? tr/Trace trace) "regenerate returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "regenerated y shape")
        (is (let [v0 (mx/item (mx/index y-val 0))]
              (or (not= v0 1.0) true))
            "y changed after regenerate"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "regenerate weight is finite"))))

(deftest in-axes-broadcast
  (testing "in-axes [0 nil] — broadcast second arg"
    (let [kernel (gen [x shared]
                   (let [y (trace :y (dist/gaussian (mx/add x shared) 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel) :in-axes [0 nil])
          xs (mx/array [1.0 2.0 3.0])
          shared (mx/scalar 10.0)
          trace (p/simulate vmodel [xs shared])]
      (is (instance? tr/Trace trace) "in-axes simulate returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "y is [3]-shaped with in-axes")
        (is (< (js/Math.abs (- (mx/item (mx/index y-val 0)) 11)) 2) "y[0] near 11")
        (is (< (js/Math.abs (- (mx/item (mx/index y-val 1)) 12)) 2) "y[1] near 12")
        (is (< (js/Math.abs (- (mx/item (mx/index y-val 2)) 13)) 2) "y[2] near 13")))))

(deftest repeat-gf-iid
  (testing "repeat-gf IID sampling"
    (let [kernel (gen []
                   (let [z (trace :z (dist/gaussian 0 1))]
                     z))
          iid (vmap/repeat-gf (dyn/auto-key kernel) 50)
          trace (p/simulate iid [])]
      (is (instance? tr/Trace trace) "repeat-gf returns trace")
      (let [z-val (cm/get-value (cm/get-submap (:choices trace) :z))]
        (mx/eval! z-val)
        (is (= [50] (mx/shape z-val)) "z is [50]-shaped"))
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "repeat-gf score is finite"))))

(deftest nested-splice
  (testing "Vmap inside gen body"
    (let [obs-kernel (gen [x]
                       (let [y (trace :y (dist/gaussian x 0.1))]
                         y))
          outer (gen [xs]
                  (let [slope (trace :slope (dist/gaussian 0 10))]
                    (splice :ys (vmap/vmap-gf (dyn/auto-key obs-kernel)) xs)
                    slope))
          trace (p/simulate (dyn/auto-key outer) [(mx/array [1.0 2.0 3.0])])]
      (is (instance? tr/Trace trace) "nested splice returns trace")
      (let [slope-val (cm/get-value (cm/get-submap (:choices trace) :slope))]
        (mx/eval! slope-val)
        (is (some? slope-val) "slope exists"))
      (let [ys-sub (cm/get-submap (:choices trace) :ys)
            y-val (cm/get-value (cm/get-submap ys-sub :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "nested ys :y is [3]-shaped")))))

(deftest vmap-assess
  (testing "assess"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          choices (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
          args [(mx/array [1.0 2.0 3.0])]
          {:keys [weight]} (p/assess vmodel args choices)
          gen-result (p/generate vmodel args choices)]
      (mx/eval! weight (:weight gen-result))
      (is (h/close? (mx/item (:weight gen-result))
                    (mx/item weight) 1e-5)
          "assess weight matches generate weight"))))

(deftest vmap-propose
  (testing "propose"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          {:keys [choices weight retval]} (p/propose vmodel [(mx/array [1.0 2.0 3.0])])]
      (is (satisfies? cm/IChoiceMap choices) "propose returns choices")
      (let [y-val (cm/get-value (cm/get-submap choices :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "propose y is [3]-shaped"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "propose weight is finite"))))

(deftest vmap-project
  (testing "project"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
          {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)
          proj (p/project vmodel trace (sel/select :y))]
      (mx/eval! proj (:score trace))
      (is (h/close? (mx/item (:score trace)) (mx/item proj) 1e-5) "project all = score"))))

(deftest dist-kernel-repeat-gf
  (testing "distribution kernel with repeat-gf"
    (let [iid (vmap/repeat-gf (dist/gaussian 0 1) 50)
          trace (p/simulate iid [])]
      (is (instance? tr/Trace trace) "dist kernel returns trace")
      (let [val (cm/get-value (:choices trace))]
        (mx/eval! val)
        (is (= [50] (mx/shape val)) "dist kernel choices are [50]-shaped"))
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "dist kernel score is finite"))))

(deftest dist-kernel-generate
  (testing "generate with distribution kernel"
    (let [iid (vmap/repeat-gf (dist/gaussian 0 1) 5)
          obs (cm/->Value (mx/array [0.5 -0.5 1.0 -1.0 0.0]))
          {:keys [trace weight]} (p/generate iid [] obs)]
      (let [val (cm/get-value (:choices trace))]
        (mx/eval! val)
        (is (h/close? 0.5 (mx/item (mx/index val 0)) 1e-6) "constrained val[0]")
        (is (h/close? 0.0 (mx/item (mx/index val 4)) 1e-6) "constrained val[4]"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "dist kernel generate weight finite"))))

(deftest sequence-args
  (testing "sequence args (not just MLX arrays)"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          trace (p/simulate vmodel [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]])]
      (is (instance? tr/Trace trace) "seq args returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "seq args y is [3]-shaped")))))

(deftest scalar-constraint-broadcast
  (testing "scalar constraint broadcast"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          scalar-obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] scalar-obs)]
      (is (instance? tr/Trace trace) "scalar broadcast returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "y is [3]-shaped")
        (is (h/close? 5.0 (mx/item (mx/index y-val 0)) 1e-6) "y[0] = 5.0")
        (is (h/close? 5.0 (mx/item (mx/index y-val 1)) 1e-6) "y[1] = 5.0")
        (is (h/close? 5.0 (mx/item (mx/index y-val 2)) 1e-6) "y[2] = 5.0"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "scalar broadcast weight finite")))

  (testing "EMPTY constraints still work"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] cm/EMPTY)]
      (is (instance? tr/Trace trace) "EMPTY constraint returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "EMPTY constraint y is [3]-shaped"))
      (mx/eval! weight)
      (is (< (js/Math.abs (mx/item weight)) 1e-6) "EMPTY constraint weight is 0")))

  (testing "scalar constraint in assess"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          scalar-obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [weight]} (p/assess vmodel [(mx/array [1.0 2.0 3.0])] scalar-obs)]
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "scalar assess weight finite"))))

(deftest per-element-selection
  (testing "shared selection (backward compat)"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
          {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)
          {:keys [trace weight]} (p/regenerate vmodel trace (sel/select :y))]
      (is (instance? tr/Trace trace) "shared selection returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "shared selection y is [3]-shaped"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "shared selection weight finite")))

  (testing "per-element selection"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          obs (cm/choicemap :y (mx/array [100.0 200.0 300.0]))
          {:keys [trace]} (p/generate vmodel [(mx/array [100.0 200.0 300.0])] obs)
          per-sel (sel/hierarchical 0 (sel/select :y) 2 (sel/select :y))
          {:keys [trace weight]} (p/regenerate vmodel trace per-sel)]
      (is (instance? tr/Trace trace) "per-element regen returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "per-element regen y is [3]-shaped")
        (is (h/close? 200.0 (mx/item (mx/index y-val 1)) 1e-6) "y[1] kept at 200.0"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "per-element regen weight finite")))

  (testing "per-element project"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
          {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)
          per-sel (sel/hierarchical 1 (sel/select :y))
          proj (p/project vmodel trace per-sel)
          full-proj (p/project vmodel trace (sel/select :y))]
      (mx/eval! proj full-proj)
      (is (= [] (mx/shape proj)) "per-element project is scalar")
      (is (<= (mx/item proj) (+ (mx/item full-proj) 1e-6)) "per-element project < full project"))))

(deftest nested-vmap-of-vmap
  (testing "nested simulate [N,M]-shaped leaves"
    (let [inner-kernel (gen []
                         (let [z (trace :z (dist/gaussian 0 1))]
                           z))
          inner-vmap (vmap/repeat-gf (dyn/auto-key inner-kernel) 3)
          outer-vmap (vmap/repeat-gf inner-vmap 4)
          trace (p/simulate outer-vmap [])]
      (is (instance? tr/Trace trace) "nested vmap returns trace")
      (let [z-val (cm/get-value (cm/get-submap (:choices trace) :z))]
        (mx/eval! z-val)
        (is (= [4 3] (mx/shape z-val)) "nested z is [4,3]-shaped"))
      (mx/eval! (:score trace))
      (is (= [] (mx/shape (:score trace))) "nested score is scalar")
      (is (js/isFinite (mx/item (:score trace))) "nested score is finite")))

  (testing "nested generate with [N,M]-shaped constraints"
    (let [inner-kernel (gen []
                         (let [z (trace :z (dist/gaussian 0 1))]
                           z))
          inner-vmap (vmap/repeat-gf (dyn/auto-key inner-kernel) 3)
          outer-vmap (vmap/repeat-gf inner-vmap 4)
          obs (cm/choicemap :z (mx/reshape (mx/array [1 2 3 4 5 6 7 8 9 10 11 12]) [4 3]))
          {:keys [trace weight]} (p/generate outer-vmap [] obs)]
      (is (instance? tr/Trace trace) "nested generate returns trace")
      (let [z-val (cm/get-value (cm/get-submap (:choices trace) :z))]
        (mx/eval! z-val)
        (is (= [4 3] (mx/shape z-val)) "nested generated z is [4,3]-shaped")
        (is (h/close? 1.0 (mx/item (mx/index (mx/index z-val 0) 0)) 1e-6) "z[0,0] = 1"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "nested generate weight finite")))

  (testing "nested update"
    (let [inner-kernel (gen []
                         (let [z (trace :z (dist/gaussian 0 1))]
                           z))
          inner-vmap (vmap/repeat-gf (dyn/auto-key inner-kernel) 3)
          outer-vmap (vmap/repeat-gf inner-vmap 4)
          obs1 (cm/choicemap :z (mx/reshape (mx/array [1 2 3 4 5 6 7 8 9 10 11 12]) [4 3]))
          {:keys [trace]} (p/generate outer-vmap [] obs1)
          old-score (:score trace)
          obs2 (cm/choicemap :z (mx/reshape (mx/array [0 0 0 0 0 0 0 0 0 0 0 0]) [4 3]))
          {:keys [trace weight]} (p/update outer-vmap trace obs2)]
      (mx/eval! old-score (:score trace) weight)
      (is (instance? tr/Trace trace) "nested update trace")
      (is (h/close? (- (mx/item (:score trace)) (mx/item old-score))
                    (mx/item weight) 1e-4)
          "nested weight = new - old")))

  (testing "nested regenerate"
    (let [inner-kernel (gen []
                         (let [z (trace :z (dist/gaussian 0 1))]
                           z))
          inner-vmap (vmap/repeat-gf (dyn/auto-key inner-kernel) 3)
          outer-vmap (vmap/repeat-gf inner-vmap 4)
          obs (cm/choicemap :z (mx/reshape (mx/array [1 2 3 4 5 6 7 8 9 10 11 12]) [4 3]))
          {:keys [trace]} (p/generate outer-vmap [] obs)
          {:keys [trace weight]} (p/regenerate outer-vmap trace (sel/select :z))]
      (is (instance? tr/Trace trace) "nested regenerate trace")
      (let [z-val (cm/get-value (cm/get-submap (:choices trace) :z))]
        (mx/eval! z-val)
        (is (= [4 3] (mx/shape z-val)) "nested regenerated z is [4,3]-shaped"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "nested regenerate weight finite"))))

(deftest fast-path
  (testing "fast-path simulate"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          trace (p/simulate vmodel [(mx/array [1.0 2.0 3.0])])]
      (is (instance? tr/Trace trace) "fast simulate returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "fast simulate y is [3]-shaped"))
      (mx/eval! (:score trace))
      (is (= [] (mx/shape (:score trace))) "fast simulate score is scalar")
      (is (js/isFinite (mx/item (:score trace))) "fast simulate score finite")))

  (testing "generate with scalar constraints (slow path)"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)]
      (is (instance? tr/Trace trace) "scalar constraint generate returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "y is [3]-shaped")
        (is (h/close? 5.0 (mx/item (mx/index y-val 0)) 1e-6) "y[0] = 5.0"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "scalar constraint generate weight finite")))

  (testing "fast-path with EMPTY constraints"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] cm/EMPTY)]
      (is (instance? tr/Trace trace) "fast generate EMPTY returns trace")
      (mx/eval! weight)
      (is (h/close? 0.0 (mx/item weight) 1e-6) "fast generate EMPTY weight is 0")))

  (testing "fast-path in-axes [0 nil]"
    (let [kernel (gen [x shared]
                   (let [y (trace :y (dist/gaussian (mx/add x shared) 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel) :in-axes [0 nil])
          xs (mx/array [1.0 2.0 3.0])
          shared (mx/scalar 10.0)
          trace (p/simulate vmodel [xs shared])]
      (is (instance? tr/Trace trace) "fast in-axes simulate returns trace")
      (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! y-val)
        (is (= [3] (mx/shape y-val)) "fast in-axes y is [3]-shaped")
        (is (< (js/Math.abs (- (mx/item (mx/index y-val 0)) 11)) 3) "fast in-axes y[0] near 11")
        (is (< (js/Math.abs (- (mx/item (mx/index y-val 2)) 13)) 3) "fast in-axes y[2] near 13"))))

  (testing "non-DynamicGF falls back to slow path"
    (let [iid (vmap/repeat-gf (dist/gaussian 0 1) 10)
          trace (p/simulate iid [])]
      (is (instance? tr/Trace trace) "dist kernel slow path works")
      (let [val (cm/get-value (:choices trace))]
        (mx/eval! val)
        (is (= [10] (mx/shape val)) "dist kernel choices are [10]-shaped"))))

  (testing "update after fast-path simulate"
    (let [kernel (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     y))
          vmodel (vmap/vmap-gf (dyn/auto-key kernel))
          trace (p/simulate vmodel [(mx/array [1.0 2.0 3.0])])
          old-score (:score trace)
          obs2 (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
          {:keys [trace weight]} (p/update vmodel trace obs2)]
      (mx/eval! old-score (:score trace) weight)
      (is (instance? tr/Trace trace) "update after fast simulate trace")
      (is (h/close? (- (mx/item (:score trace)) (mx/item old-score))
                    (mx/item weight) 1e-4)
          "update weight = new - old"))))

(deftest splice-of-vmap-batched
  (testing "vsimulate with vmap splice"
    (let [obs-kernel (gen [x]
                       (let [y (trace :y (dist/gaussian x 0.1))]
                         y))
          outer (gen [xs]
                  (let [slope (trace :slope (dist/gaussian 0 10))]
                    (splice :obs (vmap/vmap-gf (dyn/auto-key obs-kernel)) xs)
                    slope))
          vtrace (dyn/vsimulate outer [(mx/array [1.0 2.0 3.0])] 5 nil)]
      (is (instance? genmlx.vectorized/VectorizedTrace vtrace) "vsimulate with vmap splice returns vtrace")
      (let [slope-val (cm/get-value (cm/get-submap (:choices vtrace) :slope))]
        (mx/eval! slope-val)
        (is (= [5] (mx/shape slope-val)) "slope is [5]-shaped (N particles)"))
      (let [obs-sub (cm/get-submap (:choices vtrace) :obs)
            y-val (cm/get-value (cm/get-submap obs-sub :y))]
        (mx/eval! y-val)
        (is (= [5 3] (mx/shape y-val)) "vmap y is [5,3]-shaped"))
      (mx/eval! (:score vtrace))
      (is (= [5] (mx/shape (:score vtrace))) "vsimulate score is [5]-shaped")))

  (testing "vgenerate with vmap splice and constraints"
    (let [obs-kernel (gen [x]
                       (let [y (trace :y (dist/gaussian x 0.1))]
                         y))
          outer (gen [xs]
                  (let [slope (trace :slope (dist/gaussian 0 10))]
                    (splice :obs (vmap/vmap-gf (dyn/auto-key obs-kernel)) xs)
                    slope))
          obs (cm/choicemap :slope (mx/scalar 2.0))
          vtrace (dyn/vgenerate outer [(mx/array [1.0 2.0 3.0])] obs 5 nil)]
      (is (instance? genmlx.vectorized/VectorizedTrace vtrace) "vgenerate with vmap splice returns vtrace")
      (let [slope-val (cm/get-value (cm/get-submap (:choices vtrace) :slope))]
        (mx/eval! slope-val)
        (is (h/close? 2.0 (mx/item slope-val) 1e-6) "slope = 2.0"))
      (let [obs-sub (cm/get-submap (:choices vtrace) :obs)
            y-val (cm/get-value (cm/get-submap obs-sub :y))]
        (mx/eval! y-val)
        (is (= [5 3] (mx/shape y-val)) "vgenerate vmap y is [5,3]-shaped"))
      (mx/eval! (:weight vtrace))
      (is (js/isFinite (mx/item (:weight vtrace))) "vgenerate weight is finite"))))

(cljs.test/run-tests)
