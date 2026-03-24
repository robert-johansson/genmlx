(ns genmlx.combinators-test
  "Tests for Map, Unfold, Switch, Mask, Scan, Contramap, and MapRetval combinators."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.diff :as diff]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; -------------------------------------------------------------------------
;; Map combinator
;; -------------------------------------------------------------------------

(deftest map-combinator-simulate-test
  (testing "Map combinator simulate"
    (let [kernel (dyn/auto-key (gen [x]
                   (let [y (trace :y (dist/gaussian x 1))]
                     (mx/eval! y)
                     (mx/item y))))
          mapped (comb/map-combinator kernel)
          trace (p/simulate mapped [[1.0 2.0 3.0]])]
      (is (instance? tr/Trace trace) "map returns trace")
      (is (= 3 (count (:retval trace))) "map returns 3 values")
      (is (every? number? (:retval trace)) "map retvals are numbers"))))

(deftest map-combinator-generate-test
  (testing "Map combinator generate"
    (let [kernel (dyn/auto-key (gen [x]
                   (let [y (trace :y (dist/gaussian x 0.1))]
                     (mx/eval! y)
                     (mx/item y))))
          mapped (comb/map-combinator kernel)
          constraints (cm/set-choice cm/EMPTY [0] (cm/choicemap :y (mx/scalar 1.5)))
          {:keys [trace weight]} (p/generate mapped [[1.0 2.0]] constraints)]
      (is (instance? tr/Trace trace) "map generate returns trace")
      (is (number? (h/realize weight)) "map generate has weight"))))

;; -------------------------------------------------------------------------
;; Unfold combinator
;; -------------------------------------------------------------------------

(deftest unfold-combinator-simulate-test
  (testing "Unfold combinator simulate"
    (let [step (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   (mx/eval! next)
                   (mx/item next))))
          unfold (comb/unfold-combinator step)
          trace (p/simulate unfold [5 0.0])]
      (is (instance? tr/Trace trace) "unfold returns trace")
      (is (= 5 (count (:retval trace))) "unfold returns 5 states")
      (is (every? number? (:retval trace)) "unfold retvals are numbers"))))

;; -------------------------------------------------------------------------
;; Switch combinator
;; -------------------------------------------------------------------------

(deftest switch-combinator-simulate-test
  (testing "Switch combinator simulate"
    (let [branch0 (dyn/auto-key (gen []
                    (let [x (trace :x (dist/gaussian 0 1))]
                      (mx/eval! x)
                      (mx/item x))))
          branch1 (dyn/auto-key (gen []
                    (let [x (trace :x (dist/gaussian 10 1))]
                      (mx/eval! x)
                      (mx/item x))))
          sw (comb/switch-combinator branch0 branch1)
          trace0 (p/simulate sw [0])
          trace1 (p/simulate sw [1])]
      (is (instance? tr/Trace trace0) "switch branch 0 returns trace")
      (is (instance? tr/Trace trace1) "switch branch 1 returns trace")
      (is (< (js/Math.abs (:retval trace0)) 5) "branch 0 value near 0")
      (is (< (js/Math.abs (- (:retval trace1) 10)) 5) "branch 1 value near 10"))))

;; -------------------------------------------------------------------------
;; Mask combinator
;; -------------------------------------------------------------------------

(deftest mask-combinator-simulate-test
  (testing "Mask combinator simulate"
    (let [inner (dyn/auto-key (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.1))]
                    (mx/eval! y)
                    (mx/item y))))
          masked (comb/mask-combinator inner)
          trace-active (p/simulate masked [true 5.0])
          trace-inactive (p/simulate masked [false 5.0])]
      (is (instance? tr/Trace trace-active) "mask active returns trace")
      (is (not= (:choices trace-active) cm/EMPTY) "mask active has choices")
      (is (= (:choices trace-inactive) cm/EMPTY) "mask inactive has empty choices")
      (is (nil? (:retval trace-inactive)) "mask inactive retval is nil"))))

(deftest mask-combinator-update-active-test
  (testing "Mask update (active)"
    (let [inner (dyn/auto-key (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.1))]
                    (mx/eval! y)
                    (mx/item y))))
          masked (comb/mask-combinator inner)
          obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace]} (p/generate masked [true 5.0] obs)
          new-obs (cm/choicemap :y (mx/scalar 5.5))
          {:keys [trace weight discard]} (p/update masked trace new-obs)]
      (mx/eval! weight)
      (is (instance? tr/Trace trace) "mask update active returns trace")
      (is (number? (mx/item weight)) "mask update active has weight")
      (is (not= 0.0 (mx/item weight)) "mask update active has non-zero weight")
      (is (not (nil? discard)) "mask update active has discard")
      (let [new-val (cm/get-choice (:choices trace) [:y])]
        (is (< (js/Math.abs (- (h/realize new-val) 5.5)) 0.01) "mask update active updated value")))))

(deftest mask-combinator-update-inactive-test
  (testing "Mask update (inactive)"
    (let [inner (dyn/auto-key (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.1))]
                    (mx/eval! y)
                    (mx/item y))))
          masked (comb/mask-combinator inner)
          trace (p/simulate masked [false 5.0])
          {:keys [trace weight discard]} (p/update masked trace cm/EMPTY)]
      (mx/eval! weight)
      (is (instance? tr/Trace trace) "mask update inactive returns same trace")
      (is (= 0.0 (mx/item weight)) "mask update inactive zero weight")
      (is (= discard cm/EMPTY) "mask update inactive empty discard"))))

(deftest mask-combinator-regenerate-active-test
  (testing "Mask regenerate (active)"
    (let [inner (dyn/auto-key (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.1))]
                    (mx/eval! y)
                    (mx/item y))))
          masked (comb/mask-combinator inner)
          obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace]} (p/generate masked [true 5.0] obs)
          {:keys [trace weight]} (p/regenerate masked trace (sel/select :y))]
      (mx/eval! weight)
      (is (instance? tr/Trace trace) "mask regenerate active returns trace")
      (is (number? (mx/item weight)) "mask regenerate active has weight")
      (is (not= (:choices trace) cm/EMPTY) "mask regenerate active has choices"))))

(deftest mask-combinator-regenerate-inactive-test
  (testing "Mask regenerate (inactive)"
    (let [inner (dyn/auto-key (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.1))]
                    (mx/eval! y)
                    (mx/item y))))
          masked (comb/mask-combinator inner)
          trace (p/simulate masked [false 5.0])
          {:keys [trace weight]} (p/regenerate masked trace (sel/select :y))]
      (mx/eval! weight)
      (is (instance? tr/Trace trace) "mask regenerate inactive returns same trace")
      (is (= 0.0 (mx/item weight)) "mask regenerate inactive zero weight"))))

;; -------------------------------------------------------------------------
;; update-with-diffs fast paths
;; -------------------------------------------------------------------------

(deftest mask-update-with-diffs-test
  (testing "Mask update-with-diffs no-change fast path"
    (let [inner (dyn/auto-key (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.1))]
                    (mx/eval! y)
                    (mx/item y))))
          masked (comb/mask-combinator inner)
          obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace]} (p/generate masked [true 5.0] obs)
          {:keys [trace weight discard]} (p/update-with-diffs masked trace cm/EMPTY diff/no-change)]
      (mx/eval! weight)
      (is (identical? trace trace) "mask update-with-diffs no-change returns same trace")
      (is (= 0.0 (mx/item weight)) "mask update-with-diffs no-change zero weight")
      (is (= discard cm/EMPTY) "mask update-with-diffs no-change empty discard"))))

(deftest contramap-update-with-diffs-test
  (testing "Contramap update-with-diffs no-change fast path"
    (let [inner (dyn/auto-key (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.1))]
                    (mx/eval! y)
                    (mx/item y))))
          cmapped (comb/contramap-gf inner identity)
          obs (cm/choicemap :y (mx/scalar 3.0))
          {:keys [trace]} (p/generate cmapped [3.0] obs)
          {:keys [trace weight discard]} (p/update-with-diffs cmapped trace cm/EMPTY diff/no-change)]
      (mx/eval! weight)
      (is (= 0.0 (mx/item weight)) "contramap update-with-diffs no-change zero weight")
      (is (= discard cm/EMPTY) "contramap update-with-diffs no-change empty discard"))))

(deftest map-retval-update-with-diffs-test
  (testing "MapRetval update-with-diffs no-change fast path"
    (let [inner (dyn/auto-key (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.1))]
                    (mx/eval! y)
                    (mx/item y))))
          mr (comb/map-retval inner (fn [v] (* v 2)))
          obs (cm/choicemap :y (mx/scalar 3.0))
          {:keys [trace]} (p/generate mr [3.0] obs)
          {:keys [trace weight discard]} (p/update-with-diffs mr trace cm/EMPTY diff/no-change)]
      (mx/eval! weight)
      (is (= 0.0 (mx/item weight)) "map-retval update-with-diffs no-change zero weight")
      (is (= discard cm/EMPTY) "map-retval update-with-diffs no-change empty discard"))))

;; -------------------------------------------------------------------------
;; Unfold prefix-skip tests
;; -------------------------------------------------------------------------

(deftest unfold-step-scores-metadata-test
  (testing "Unfold step-scores metadata"
    (let [step (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   (mx/eval! next)
                   (mx/item next))))
          unfold (comb/unfold-combinator step)
          trace-sim (p/simulate unfold [5 0.0])
          sim-meta (meta trace-sim)
          obs (-> cm/EMPTY
                  (cm/set-choice [3] (cm/choicemap :x (mx/scalar 1.0)))
                  (cm/set-choice [4] (cm/choicemap :x (mx/scalar 2.0))))
          {:keys [trace]} (p/generate unfold [5 0.0] obs)
          gen-meta (meta trace)]
      (is (some? (::comb/step-scores sim-meta)) "simulate has ::step-scores metadata")
      (is (= 5 (count (::comb/step-scores sim-meta))) "simulate step-scores count")
      (is (some? (::comb/step-scores gen-meta)) "generate has ::step-scores metadata")
      (is (= 5 (count (::comb/step-scores gen-meta))) "generate step-scores count"))))

(deftest unfold-prefix-skip-test
  (testing "Unfold prefix skip optimization"
    (let [step (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   (mx/eval! next)
                   (mx/item next))))
          unfold (comb/unfold-combinator step)
          init-obs (-> cm/EMPTY
                       (cm/set-choice [3] (cm/choicemap :x (mx/scalar 1.0)))
                       (cm/set-choice [4] (cm/choicemap :x (mx/scalar 2.0))))
          {:keys [trace]} (p/generate unfold [5 0.0] init-obs)
          new-obs (cm/set-choice cm/EMPTY [4] (cm/choicemap :x (mx/scalar 3.0)))
          result-opt (p/update unfold trace new-obs)
          trace-no-meta (with-meta trace (dissoc (meta trace) ::comb/step-scores))
          result-full (p/update unfold trace-no-meta new-obs)]
      (is (some? (::comb/step-scores (meta (:trace result-opt))))
          "update result has ::step-scores")
      (mx/eval! (:weight result-opt))
      (mx/eval! (:weight result-full))
      (let [w-opt (mx/item (:weight result-opt))
            w-full (mx/item (:weight result-full))]
        (is (< (js/Math.abs (- w-opt w-full)) 1e-5)
            "optimized weight matches full update weight"))
      (let [new-val (cm/get-choice (:choices (:trace result-opt)) [4 :x])]
        (is (< (js/Math.abs (- (h/realize new-val) 3.0)) 0.01)
            "step 4 choice updated"))
      (let [old-val (cm/get-choice (:choices trace) [3 :x])
            new-val (cm/get-choice (:choices (:trace result-opt)) [3 :x])]
        (mx/eval! old-val)
        (mx/eval! new-val)
        (is (< (js/Math.abs (- (mx/item old-val) (mx/item new-val))) 1e-6)
            "step 3 choice preserved in prefix")))))

(deftest unfold-update-empty-constraints-test
  (testing "Unfold update with empty constraints"
    (let [step (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   (mx/eval! next)
                   (mx/item next))))
          unfold (comb/unfold-combinator step)
          trace (p/simulate unfold [5 0.0])
          result (p/update unfold trace cm/EMPTY)]
      (mx/eval! (:weight result))
      (is (= 0.0 (mx/item (:weight result)))
          "update with empty constraints returns zero weight")
      (is (identical? (:trace result) trace)
          "update with empty constraints returns same trace"))))

;; -------------------------------------------------------------------------
;; Scan prefix-skip tests
;; -------------------------------------------------------------------------

(deftest scan-step-metadata-test
  (testing "Scan step metadata"
    (let [kernel (dyn/auto-key (gen [carry input]
                   (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                     (mx/eval! x)
                     [(mx/item x) (mx/item x)])))
          scan (comb/scan-combinator kernel)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                  (mx/scalar 4.0) (mx/scalar 5.0)]
          trace-sim (p/simulate scan [(mx/scalar 0.0) inputs])
          sim-meta (meta trace-sim)
          obs (cm/set-choice cm/EMPTY [3] (cm/choicemap :x (mx/scalar 10.0)))
          {:keys [trace]} (p/generate scan [(mx/scalar 0.0) inputs] obs)
          gen-meta (meta trace)]
      (is (some? (::comb/step-scores sim-meta)) "scan simulate has ::step-scores")
      (is (some? (::comb/step-carries sim-meta)) "scan simulate has ::step-carries")
      (is (= 5 (count (::comb/step-scores sim-meta))) "scan simulate step-scores count")
      (is (= 5 (count (::comb/step-carries sim-meta))) "scan simulate step-carries count")
      (is (some? (::comb/step-scores gen-meta)) "scan generate has ::step-scores")
      (is (some? (::comb/step-carries gen-meta)) "scan generate has ::step-carries"))))

(deftest scan-prefix-skip-test
  (testing "Scan prefix skip optimization"
    (let [kernel (dyn/auto-key (gen [carry input]
                   (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                     (mx/eval! x)
                     [(mx/item x) (mx/item x)])))
          scan (comb/scan-combinator kernel)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                  (mx/scalar 4.0) (mx/scalar 5.0)]
          init-obs (cm/set-choice cm/EMPTY [3] (cm/choicemap :x (mx/scalar 10.0)))
          {:keys [trace]} (p/generate scan [(mx/scalar 0.0) inputs] init-obs)
          new-obs (cm/set-choice cm/EMPTY [4] (cm/choicemap :x (mx/scalar 20.0)))
          result-opt (p/update scan trace new-obs)
          trace-no-meta (with-meta trace (dissoc (meta trace) ::comb/step-scores ::comb/step-carries))
          result-full (p/update scan trace-no-meta new-obs)]
      (is (some? (::comb/step-scores (meta (:trace result-opt))))
          "scan update result has ::step-scores")
      (is (some? (::comb/step-carries (meta (:trace result-opt))))
          "scan update result has ::step-carries")
      (mx/eval! (:weight result-opt))
      (mx/eval! (:weight result-full))
      (let [w-opt (mx/item (:weight result-opt))
            w-full (mx/item (:weight result-full))]
        (is (< (js/Math.abs (- w-opt w-full)) 1e-5)
            "scan optimized weight matches full update"))
      (let [new-val (cm/get-choice (:choices (:trace result-opt)) [4 :x])]
        (is (< (js/Math.abs (- (h/realize new-val) 20.0)) 0.01)
            "scan step 4 choice updated")))))

(deftest scan-update-empty-constraints-test
  (testing "Scan update with empty constraints"
    (let [kernel (dyn/auto-key (gen [carry input]
                   (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                     (mx/eval! x)
                     [(mx/item x) (mx/item x)])))
          scan (comb/scan-combinator kernel)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          trace (p/simulate scan [(mx/scalar 0.0) inputs])
          result (p/update scan trace cm/EMPTY)]
      (mx/eval! (:weight result))
      (is (= 0.0 (mx/item (:weight result)))
          "scan update with empty constraints returns zero weight")
      (is (identical? (:trace result) trace)
          "scan update with empty constraints returns same trace"))))

;; -------------------------------------------------------------------------
;; Switch branch-switching tests
;; -------------------------------------------------------------------------

(deftest switch-update-same-branch-test
  (testing "Switch update same branch"
    (let [branch0 (dyn/auto-key (gen []
                    (let [x (trace :x (dist/gaussian 0 1))]
                      (mx/eval! x)
                      (mx/item x))))
          branch1 (dyn/auto-key (gen []
                    (let [x (trace :x (dist/gaussian 10 1))]
                      (mx/eval! x)
                      (mx/item x))))
          sw (comb/switch-combinator branch0 branch1)
          obs (cm/choicemap :x (mx/scalar 1.0))
          {:keys [trace]} (p/generate sw [0] obs)
          new-obs (cm/choicemap :x (mx/scalar 2.0))
          updated-trace (tr/make-trace {:gen-fn sw :args [0]
                                        :choices (:choices trace)
                                        :retval (:retval trace)
                                        :score (:score trace)})
          updated-trace (with-meta updated-trace (meta trace))
          {:keys [trace weight discard]} (p/update sw updated-trace new-obs)]
      (mx/eval! weight)
      (let [new-val (cm/get-choice (:choices trace) [:x])]
        (is (< (js/Math.abs (- (h/realize new-val) 2.0)) 0.01)
            "switch same-branch update has new value"))
      (is (number? (mx/item weight)) "switch same-branch update has weight")
      (is (some? discard) "switch same-branch update has discard"))))

(deftest switch-update-branch-change-test
  (testing "Switch update branch change"
    (let [branch0 (dyn/auto-key (gen []
                    (let [x (trace :x (dist/gaussian 0 1))]
                      (mx/eval! x)
                      (mx/item x))))
          branch1 (dyn/auto-key (gen []
                    (let [x (trace :x (dist/gaussian 10 1))]
                      (mx/eval! x)
                      (mx/item x))))
          sw (comb/switch-combinator branch0 branch1)
          obs (cm/choicemap :x (mx/scalar 1.0))
          {:keys [trace]} (p/generate sw [0] obs)
          old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          switched-trace (with-meta
                           (tr/make-trace {:gen-fn sw :args [1]
                                           :choices (:choices trace)
                                           :retval (:retval trace)
                                           :score (:score trace)})
                           (meta trace))
          new-obs (cm/choicemap :x (mx/scalar 10.5))
          {:keys [trace weight discard]} (p/update sw switched-trace new-obs)]
      (mx/eval! weight)
      (mx/eval! (:score trace))
      (let [new-val (cm/get-choice (:choices trace) [:x])]
        (is (< (js/Math.abs (- (h/realize new-val) 10.5)) 0.01)
            "switch branch-change has new value"))
      (let [expected-weight (- (mx/item (:score trace)) old-score)]
        (is (< (js/Math.abs (- (mx/item weight) expected-weight)) 1e-5)
            "switch branch-change weight = new_score - old_score"))
      (is (some? discard) "switch branch-change discard is old choices"))))

(deftest switch-metadata-preserved-test
  (testing "Switch metadata preserved"
    (let [branch0 (dyn/auto-key (gen []
                    (let [x (trace :x (dist/gaussian 0 1))]
                      (mx/eval! x)
                      (mx/item x))))
          sw (comb/switch-combinator branch0)
          trace (p/simulate sw [0])]
      (is (= 0 (::comb/switch-idx (meta trace)))
          "simulate has ::switch-idx metadata")
      (let [{:keys [trace]} (p/generate sw [0] (cm/choicemap :x (mx/scalar 1.0)))]
        (is (= 0 (::comb/switch-idx (meta trace)))
            "generate has ::switch-idx metadata"))
      (let [{:keys [trace]} (p/regenerate sw trace (sel/select :x))]
        (is (= 0 (::comb/switch-idx (meta trace)))
            "regenerate has ::switch-idx metadata")))))

(cljs.test/run-tests)
