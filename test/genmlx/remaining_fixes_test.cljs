(ns genmlx.remaining-fixes-test
  "Tests for three remaining audit fixes:
   1. Edit interface — IEdit implemented on all GF records
   2. Argdiffs — diff-aware update on combinators
   3. Trainable parameters — dyn/param with param store"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.edit :as edit]
            [genmlx.diff :as diff]
            [genmlx.combinators :as comb]
            [genmlx.learning :as learn]))

;; =========================================================================
;; Test 1: Edit Interface — IEdit on DynamicGF
;; =========================================================================

(deftest constraint-edit-on-dynamic-gf
  (testing "ConstraintEdit on DynamicGF"
    (let [model (gen [obs-val]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian mu 1))
                    mu))
          obs-val 5.0
          init-cm (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar obs-val))
          {:keys [trace]} (p/generate (dyn/auto-key model) [obs-val] init-cm)
          edit-req (edit/constraint-edit (cm/choicemap :mu (mx/scalar 4.0)))
          edit-result (edit/edit (dyn/auto-key (:gen-fn trace)) trace edit-req)
          update-result (p/update (dyn/auto-key (:gen-fn trace)) trace (cm/choicemap :mu (mx/scalar 4.0)))
          edit-weight (mx/realize (:weight edit-result))
          update-weight (mx/realize (:weight update-result))]
      (is (h/close? update-weight edit-weight 0.001) "ConstraintEdit weight matches update weight")
      (is (some? (:backward-request edit-result)) "ConstraintEdit produces backward-request")
      (is (instance? edit/ConstraintEdit (:backward-request edit-result)) "backward-request is ConstraintEdit"))))

(deftest selection-edit-on-dynamic-gf
  (testing "SelectionEdit on DynamicGF"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    [x y]))
          init-cm (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace]} (p/generate (dyn/auto-key model) [] init-cm)
          sel (sel/select :x)
          edit-req (edit/selection-edit sel)
          edit-result (edit/edit (dyn/auto-key (:gen-fn trace)) trace edit-req)]
      (is (some? (:trace edit-result)) "SelectionEdit produces a trace")
      (is (number? (mx/realize (:weight edit-result))) "SelectionEdit weight is a number")
      (is (instance? edit/SelectionEdit (:backward-request edit-result)) "SelectionEdit backward-request is SelectionEdit")
      (let [new-y (mx/realize (cm/get-choice (:choices (:trace edit-result)) [:y]))]
        (is (h/close? 2.0 new-y 0.001) "SelectionEdit: unselected address y unchanged")))))

(deftest constraint-edit-on-map-combinator
  (testing "ConstraintEdit on MapCombinator"
    (let [kernel (gen [x]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian mu 1))
                    mu))
          model (comb/map-combinator (dyn/auto-key kernel))
          args [[0.0 0.0]]
          constraints (cm/choicemap
                        0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                        1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0)))
          {:keys [trace]} (p/generate model args constraints)
          edit-req (edit/constraint-edit (cm/choicemap 0 (cm/choicemap :mu (mx/scalar 5.0))))
          edit-result (edit/edit model trace edit-req)]
      (is (some? (:trace edit-result)) "MapCombinator edit produces trace")
      (is (some? (:backward-request edit-result)) "MapCombinator edit produces backward-request"))))

;; =========================================================================
;; Test 2: Argdiffs — IUpdateWithDiffs
;; =========================================================================

(deftest argdiff-no-change-shortcut
  (testing "DynamicGF no-change shortcut"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    x))
          init-cm (cm/choicemap :x (mx/scalar 3.0))
          {:keys [trace]} (p/generate (dyn/auto-key model) [] init-cm)
          old-x (mx/realize (cm/get-choice (:choices trace) [:x]))
          result (p/update-with-diffs (dyn/auto-key model) trace cm/EMPTY diff/no-change)]
      (is (h/close? 0.0 (mx/realize (:weight result)) 0.001) "no-change: weight = 0")
      (let [new-x (mx/realize (cm/get-choice (:choices (:trace result)) [:x]))]
        (is (h/close? old-x new-x 0.001) "no-change: trace unchanged")))))

(deftest argdiff-map-combinator-vector-diff
  (testing "MapCombinator vector-diff optimization"
    (let [kernel (gen [x]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian mu 1))
                    mu))
          model (comb/map-combinator (dyn/auto-key kernel))
          args [[0.0 0.0 0.0]]
          constraints (cm/choicemap
                        0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                        1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0))
                        2 (cm/choicemap :mu (mx/scalar 3.0) :obs (mx/scalar 5.0)))
          {:keys [trace]} (p/generate model args constraints)
          old-mu-1 (mx/realize (cm/get-choice (:choices trace) [1 :mu]))
          old-mu-2 (mx/realize (cm/get-choice (:choices trace) [2 :mu]))
          new-constraints (cm/choicemap 0 (cm/choicemap :mu (mx/scalar 5.0)))
          vdiff {:diff-type :vector-diff :changed #{0}}
          result (p/update-with-diffs model trace new-constraints vdiff)
          new-trace (:trace result)
          full-result (p/update model trace new-constraints)]
      (let [new-mu-0 (mx/realize (cm/get-choice (:choices new-trace) [0 :mu]))]
        (is (h/close? 5.0 new-mu-0 0.001) "vector-diff: element 0 updated"))
      (let [new-mu-1 (mx/realize (cm/get-choice (:choices new-trace) [1 :mu]))
            new-mu-2 (mx/realize (cm/get-choice (:choices new-trace) [2 :mu]))]
        (is (h/close? old-mu-1 new-mu-1 0.001) "vector-diff: element 1 unchanged")
        (is (h/close? old-mu-2 new-mu-2 0.001) "vector-diff: element 2 unchanged"))
      (let [diff-weight (mx/realize (:weight result))
            full-weight (mx/realize (:weight full-result))]
        (is (h/close? full-weight diff-weight 0.01) "vector-diff: weight matches full update")))))

(deftest argdiff-map-combinator-no-change-with-constraints
  (testing "MapCombinator no-change with constraints"
    (let [kernel (gen [x]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian mu 1))
                    mu))
          model (comb/map-combinator (dyn/auto-key kernel))
          args [[0.0 0.0]]
          constraints (cm/choicemap
                        0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                        1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0)))
          {:keys [trace]} (p/generate model args constraints)
          new-constraints (cm/choicemap 1 (cm/choicemap :mu (mx/scalar 5.0)))
          result (p/update-with-diffs model trace new-constraints diff/no-change)
          new-trace (:trace result)]
      (let [old-mu-0 (mx/realize (cm/get-choice (:choices trace) [0 :mu]))
            new-mu-0 (mx/realize (cm/get-choice (:choices new-trace) [0 :mu]))]
        (is (h/close? old-mu-0 new-mu-0 0.001) "no-change+constraints: element 0 unchanged"))
      (let [new-mu-1 (mx/realize (cm/get-choice (:choices new-trace) [1 :mu]))]
        (is (h/close? 5.0 new-mu-1 0.001) "no-change+constraints: element 1 updated")))))

;; =========================================================================
;; Test 3: Trainable Parameters — dyn/param
;; =========================================================================

(deftest param-default-value
  (testing "param with default value"
    (let [val (dyn/param :theta 3.0)]
      (is (h/close? 3.0 (mx/realize val) 0.001) "param outside handler returns default"))))

(deftest param-with-store
  (testing "param with param store via simulate-with-params"
    (let [param-model (gen []
                        (let [v (param :theta 0.0)]
                          v))
          store (learn/make-param-store {:theta 7.0 :sigma 2.0})
          trace (learn/simulate-with-params param-model [] store)]
      (is (h/close? 7.0 (mx/realize (:retval trace)) 0.001) "param with store reads stored value")))

  (testing "missing param falls back to default"
    (let [param-model (gen []
                        (let [v (param :missing-param 99.0)]
                          v))
          store (learn/make-param-store {:theta 7.0})
          trace (learn/simulate-with-params param-model [] store)]
      (is (h/close? 99.0 (mx/realize (:retval trace)) 0.001) "param missing from store returns default"))))

(deftest param-inside-gen-body
  (testing "dyn/param inside gen body"
    (let [model (gen [obs-val]
                  (let [mu (param :mu 0.0)]
                    (trace :obs (dist/gaussian mu 1))
                    mu))
          trace-no-store (p/simulate (dyn/auto-key model) [5.0])
          retval-default (mx/realize (:retval trace-no-store))]
      (is (h/close? 0.0 retval-default 0.001) "gen body: param default = 0.0")
      (let [store (learn/make-param-store {:mu 10.0})
            trace-with-store (learn/simulate-with-params model [5.0] store)
            retval-stored (mx/realize (:retval trace-with-store))]
        (is (h/close? 10.0 retval-stored 0.001) "gen body: param from store = 10.0")))))

(deftest generate-with-params-test
  (testing "generate-with-params"
    (let [model (gen [obs-val]
                  (let [mu (param :mu 0.0)]
                    (trace :obs (dist/gaussian mu 1))
                    mu))
          store (learn/make-param-store {:mu 5.0})
          obs (cm/choicemap :obs (mx/scalar 5.0))
          {:keys [trace weight]} (learn/generate-with-params model [5.0] obs store)]
      (is (some? trace) "generate-with-params produces a trace")
      (is (number? (mx/realize weight)) "generate-with-params weight is a number"))))

(deftest make-param-loss-fn-test
  (testing "make-param-loss-fn"
    (let [model (gen [obs-val]
                  (let [mu (param :mu 0.0)]
                    (trace :obs (dist/gaussian mu 0.1))
                    mu))
          obs (cm/choicemap :obs (mx/scalar 5.0))
          loss-grad-fn (learn/make-param-loss-fn model [5.0] obs [:mu])
          init-params (mx/array [0.0])
          {:keys [loss grad]} (loss-grad-fn init-params nil)]
      (is (js/isFinite (mx/realize loss)) "loss is finite")
      (is (js/isFinite (mx/realize (mx/index grad 0))) "gradient is finite")
      (let [g (mx/realize (mx/index grad 0))]
        (is (< g 0) "gradient points toward obs (negative for mu < obs)")))))

(cljs.test/run-tests)
