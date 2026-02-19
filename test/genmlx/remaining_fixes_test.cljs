(ns genmlx.remaining-fixes-test
  "Tests for three remaining audit fixes:
   1. Edit interface — IEdit implemented on all GF records
   2. Argdiffs — diff-aware update on combinators
   3. Trainable parameters — dyn/param with param store"
  (:require [genmlx.protocols :as p]
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
            [genmlx.handler :as h]
            [genmlx.learning :as learn]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:dynamic *failures* (atom 0))

(defn assert-true [msg pred]
  (if pred
    (println (str "  PASS: " msg))
    (do (println (str "  FAIL: " msg))
        (swap! *failures* inc))))

(defn assert-close [msg expected actual tol]
  (let [ok (< (js/Math.abs (- actual expected)) tol)]
    (if ok
      (println (str "  PASS: " msg " (" actual " ≈ " expected ")"))
      (do (println (str "  FAIL: " msg " (got " actual ", expected " expected ", tol " tol ")"))
          (swap! *failures* inc)))))

(println "\n=== Remaining Fixes Tests ===\n")

;; =========================================================================
;; Test 1: Edit Interface — IEdit on DynamicGF
;; =========================================================================

(println "-- Edit interface: ConstraintEdit on DynamicGF --")

(let [model (gen [obs-val]
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian mu 1))
                mu))
      obs-val 5.0
      init-cm (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar obs-val))
      {:keys [trace]} (p/generate model [obs-val] init-cm)

      ;; Use edit with ConstraintEdit (should work like update)
      edit-req (edit/constraint-edit (cm/choicemap :mu (mx/scalar 4.0)))
      edit-result (edit/edit (:gen-fn trace) trace edit-req)

      ;; Also do regular update for comparison
      update-result (p/update (:gen-fn trace) trace (cm/choicemap :mu (mx/scalar 4.0)))

      ;; Edit weight should match update weight
      edit-weight (mx/realize (:weight edit-result))
      update-weight (mx/realize (:weight update-result))]

  (assert-close "ConstraintEdit weight matches update weight"
                update-weight edit-weight 0.001)

  ;; Edit should produce a backward request
  (assert-true "ConstraintEdit produces backward-request"
               (some? (:backward-request edit-result)))

  ;; Backward request should be a ConstraintEdit containing discarded values
  (assert-true "backward-request is ConstraintEdit"
               (instance? edit/ConstraintEdit (:backward-request edit-result))))

;; SelectionEdit on DynamicGF
(println "\n-- Edit interface: SelectionEdit on DynamicGF --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                [x y]))
      init-cm (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
      {:keys [trace]} (p/generate model [] init-cm)

      ;; Edit with SelectionEdit (should work like regenerate)
      sel (sel/select :x)
      edit-req (edit/selection-edit sel)
      edit-result (edit/edit (:gen-fn trace) trace edit-req)]

  (assert-true "SelectionEdit produces a trace"
               (some? (:trace edit-result)))
  (assert-true "SelectionEdit weight is a number"
               (number? (mx/realize (:weight edit-result))))
  (assert-true "SelectionEdit backward-request is SelectionEdit"
               (instance? edit/SelectionEdit (:backward-request edit-result)))
  ;; y should be unchanged
  (let [new-y (mx/realize (cm/get-choice (:choices (:trace edit-result)) [:y]))]
    (assert-close "SelectionEdit: unselected address y unchanged"
                  2.0 new-y 0.001)))

;; IEdit on MapCombinator
(println "\n-- Edit interface: ConstraintEdit on MapCombinator --")

(let [kernel (gen [x]
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian mu 1))
                mu))
      model (comb/map-combinator kernel)
      args [[0.0 0.0]]
      constraints (cm/choicemap
                    0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                    1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0)))
      {:keys [trace]} (p/generate model args constraints)

      edit-req (edit/constraint-edit (cm/choicemap 0 (cm/choicemap :mu (mx/scalar 5.0))))
      edit-result (edit/edit model trace edit-req)]

  (assert-true "MapCombinator edit produces trace"
               (some? (:trace edit-result)))
  (assert-true "MapCombinator edit produces backward-request"
               (some? (:backward-request edit-result))))

;; =========================================================================
;; Test 2: Argdiffs — IUpdateWithDiffs
;; =========================================================================

(println "\n-- Argdiffs: DynamicGF no-change shortcut --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                x))
      init-cm (cm/choicemap :x (mx/scalar 3.0))
      {:keys [trace]} (p/generate model [] init-cm)
      old-x (mx/realize (cm/get-choice (:choices trace) [:x]))

      ;; Update with no-change argdiff and no constraints: trace should be unchanged
      result (p/update-with-diffs model trace cm/EMPTY diff/no-change)]

  (assert-close "no-change: weight = 0"
                0.0 (mx/realize (:weight result)) 0.001)
  ;; Trace should be identical
  (let [new-x (mx/realize (cm/get-choice (:choices (:trace result)) [:x]))]
    (assert-close "no-change: trace unchanged"
                  old-x new-x 0.001)))

(println "\n-- Argdiffs: MapCombinator vector-diff optimization --")

(let [kernel (gen [x]
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian mu 1))
                mu))
      model (comb/map-combinator kernel)
      args [[0.0 0.0 0.0]]  ;; 3 elements
      constraints (cm/choicemap
                    0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                    1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0))
                    2 (cm/choicemap :mu (mx/scalar 3.0) :obs (mx/scalar 5.0)))
      {:keys [trace]} (p/generate model args constraints)
      old-score (mx/realize (:score trace))

      ;; Get values before update
      old-mu-0 (mx/realize (cm/get-choice (:choices trace) [0 :mu]))
      old-mu-1 (mx/realize (cm/get-choice (:choices trace) [1 :mu]))
      old-mu-2 (mx/realize (cm/get-choice (:choices trace) [2 :mu]))

      ;; Update only element 0 with vector-diff
      new-constraints (cm/choicemap 0 (cm/choicemap :mu (mx/scalar 5.0)))
      vdiff (diff/vector-diff #{0})
      result (p/update-with-diffs model trace new-constraints vdiff)
      new-trace (:trace result)

      ;; Also do full update for comparison
      full-result (p/update model trace new-constraints)
      full-trace (:trace full-result)]

  ;; Element 0 should be updated
  (let [new-mu-0 (mx/realize (cm/get-choice (:choices new-trace) [0 :mu]))]
    (assert-close "vector-diff: element 0 updated"
                  5.0 new-mu-0 0.001))

  ;; Elements 1 and 2 should be unchanged
  (let [new-mu-1 (mx/realize (cm/get-choice (:choices new-trace) [1 :mu]))
        new-mu-2 (mx/realize (cm/get-choice (:choices new-trace) [2 :mu]))]
    (assert-close "vector-diff: element 1 unchanged"
                  old-mu-1 new-mu-1 0.001)
    (assert-close "vector-diff: element 2 unchanged"
                  old-mu-2 new-mu-2 0.001))

  ;; Weight should match full update
  (let [diff-weight (mx/realize (:weight result))
        full-weight (mx/realize (:weight full-result))]
    (assert-close "vector-diff: weight matches full update"
                  full-weight diff-weight 0.01)))

(println "\n-- Argdiffs: MapCombinator no-change with constraints --")

(let [kernel (gen [x]
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian mu 1))
                mu))
      model (comb/map-combinator kernel)
      args [[0.0 0.0]]
      constraints (cm/choicemap
                    0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                    1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0)))
      {:keys [trace]} (p/generate model args constraints)

      ;; no-change argdiff but with constraints: should still update constrained elements
      new-constraints (cm/choicemap 1 (cm/choicemap :mu (mx/scalar 5.0)))
      result (p/update-with-diffs model trace new-constraints diff/no-change)
      new-trace (:trace result)]

  ;; Element 0 should be unchanged (no constraint, no arg change)
  (let [old-mu-0 (mx/realize (cm/get-choice (:choices trace) [0 :mu]))
        new-mu-0 (mx/realize (cm/get-choice (:choices new-trace) [0 :mu]))]
    (assert-close "no-change+constraints: element 0 unchanged"
                  old-mu-0 new-mu-0 0.001))

  ;; Element 1 should be updated
  (let [new-mu-1 (mx/realize (cm/get-choice (:choices new-trace) [1 :mu]))]
    (assert-close "no-change+constraints: element 1 updated"
                  5.0 new-mu-1 0.001)))

;; =========================================================================
;; Test 3: Trainable Parameters — dyn/param
;; =========================================================================

(println "\n-- Trainable params: dyn/param with default value --")

;; Outside any handler, param returns default
(let [val (dyn/param :theta 3.0)]
  (assert-close "param outside handler returns default"
                3.0 (mx/realize val) 0.001))

(println "\n-- Trainable params: dyn/param with param store --")

;; With param store bound
(let [store (learn/make-param-store {:theta 7.0 :sigma 2.0})
      val (binding [h/*param-store* store]
            (dyn/param :theta 0.0))]
  (assert-close "param with store reads stored value"
                7.0 (mx/realize val) 0.001))

;; Missing param falls back to default
(let [store (learn/make-param-store {:theta 7.0})
      val (binding [h/*param-store* store]
            (dyn/param :missing-param 99.0))]
  (assert-close "param missing from store returns default"
                99.0 (mx/realize val) 0.001))

(println "\n-- Trainable params: dyn/param inside gen body --")

;; Model that uses dyn/param for its mean
(let [model (gen [obs-val]
              (let [mu (dyn/param :mu 0.0)]
                (dyn/trace :obs (dist/gaussian mu 1))
                mu))

      ;; Without param store: mu = 0.0 (default)
      trace-no-store (p/simulate model [5.0])
      retval-default (mx/realize (:retval trace-no-store))]

  (assert-close "gen body: param default = 0.0"
                0.0 retval-default 0.001)

  ;; With param store: mu = 10.0
  (let [store (learn/make-param-store {:mu 10.0})
        trace-with-store (learn/simulate-with-params model [5.0] store)
        retval-stored (mx/realize (:retval trace-with-store))]
    (assert-close "gen body: param from store = 10.0"
                  10.0 retval-stored 0.001)))

(println "\n-- Trainable params: generate-with-params --")

(let [model (gen [obs-val]
              (let [mu (dyn/param :mu 0.0)]
                (dyn/trace :obs (dist/gaussian mu 1))
                mu))
      store (learn/make-param-store {:mu 5.0})
      obs (cm/choicemap :obs (mx/scalar 5.0))
      {:keys [trace weight]} (learn/generate-with-params model [5.0] obs store)]

  (assert-true "generate-with-params produces a trace"
               (some? trace))
  ;; mu param = 5.0, obs = 5.0, so obs ~ N(5, 1) observed at 5 → high likelihood
  (assert-true "generate-with-params weight is a number"
               (number? (mx/realize weight))))

(println "\n-- Trainable params: make-param-loss-fn --")

;; Test that gradient computation works with params
(let [model (gen [obs-val]
              (let [mu (dyn/param :mu 0.0)]
                (dyn/trace :obs (dist/gaussian mu 0.1))
                mu))
      obs (cm/choicemap :obs (mx/scalar 5.0))
      loss-grad-fn (learn/make-param-loss-fn model [5.0] obs [:mu])

      ;; Initial params: mu = 0.0 (far from obs = 5.0)
      init-params (mx/array [0.0])
      {:keys [loss grad]} (loss-grad-fn init-params nil)]

  (assert-true "loss is finite"
               (js/isFinite (mx/realize loss)))
  (assert-true "gradient is finite"
               (js/isFinite (mx/realize (mx/index grad 0))))
  ;; Gradient should point toward obs=5 (positive gradient reduces loss
  ;; because loss = -weight, and moving mu toward obs increases weight)
  ;; Actually: loss = -log p(obs|mu). d/dmu -log N(5; mu, 0.1) = -(5-mu)/0.01
  ;; At mu=0: gradient = -5/0.01 = -500. Loss is high.
  ;; The gradient of the loss w.r.t. mu should be negative (decrease mu makes loss worse,
  ;; increase mu toward 5 makes loss better). Actually d/dmu = (obs-mu)/sigma^2 for log p.
  ;; For -log p: d/dmu = -(obs-mu)/sigma^2 = negative when mu < obs. So gradient is negative.
  (let [g (mx/realize (mx/index grad 0))]
    (assert-true "gradient points toward obs (negative for mu < obs)"
                 (< g 0))))

;; =========================================================================
;; Test 4: diff.cljs utilities are functional
;; =========================================================================

(println "\n-- Diff utilities: basic operations --")

(assert-true "no-change predicate"
             (diff/no-change? diff/no-change))
(assert-true "unknown-change predicate"
             (diff/unknown-change? diff/unknown-change))
(assert-true "nil is unknown"
             (diff/unknown-change? nil))
(assert-true "no-change is not changed"
             (not (diff/changed? diff/no-change)))
(assert-true "unknown-change is changed"
             (diff/changed? diff/unknown-change))

(let [d (diff/value-change 1 2)]
  (assert-true "value-change is changed" (diff/changed? d))
  (assert-true "value-change old" (= (:old d) 1))
  (assert-true "value-change new" (= (:new d) 2)))

(let [vd (diff/compute-vector-diff [1 2 3] [1 2 4])]
  (assert-true "vector-diff detects index 2"
               (= (:changed vd) #{2})))

(let [md (diff/compute-map-diff {:a 1 :b 2} {:a 1 :b 3 :c 4})]
  (assert-true "map-diff: b changed" (contains? (:changed md) :b))
  (assert-true "map-diff: c added" (contains? (:added md) :c))
  (assert-true "map-diff: nothing removed" (empty? (:removed md))))

(assert-true "should-recompute? false for no-change"
             (not (diff/should-recompute? diff/no-change :any)))
(assert-true "should-recompute? true for unknown-change"
             (diff/should-recompute? diff/unknown-change :any))
(assert-true "should-recompute? true for changed index in vector-diff"
             (diff/should-recompute? (diff/vector-diff #{2}) 2))
(assert-true "should-recompute? false for unchanged index"
             (not (diff/should-recompute? (diff/vector-diff #{2}) 1)))

;; =========================================================================
;; Summary
;; =========================================================================

(println (str "\n=== Remaining Fixes Test Results ==="))
(println (str "  Failures: " @*failures*))
(when (pos? @*failures*)
  (js/process.exit 1))
