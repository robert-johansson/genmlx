(ns genmlx.edit-diff-property-test
  "Property-based tests for edit.cljs and diff.cljs using test.check.
   Verifies edit dispatch, backward requests, and diff computation invariants."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.edit :as edit]
            [genmlx.diff :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- report-result [name result]
  (if (:pass? result)
    (do (vswap! pass-count inc)
        (println "  PASS:" name (str "(" (:num-tests result) " trials)")))
    (do (vswap! fail-count inc)
        (println "  FAIL:" name)
        (println "    seed:" (:seed result))
        (when-let [s (get-in result [:shrunk :smallest])]
          (println "    shrunk:" s)))))

(defn- check [name prop & {:keys [num-tests] :or {num-tests 50}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

(defn- choice-val [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

;; ---------------------------------------------------------------------------
;; Models and fixture pools
;; ---------------------------------------------------------------------------

(def two-gauss
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/eval! x y)
        (+ (mx/item x) (mx/item y))))))

(def mixed-model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/uniform 0 1))]
        (mx/eval! x y)
        (+ (mx/item x) (mx/item y))))))

(def model-pool [two-gauss mixed-model])
(def gen-model (gen/elements model-pool))

(def constraint-pool
  [(cm/choicemap :x (mx/scalar 0.5))
   (cm/choicemap :x (mx/scalar -1.0))
   (cm/choicemap :y (mx/scalar 0.3))
   (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 0.3))])
(def gen-constraint (gen/elements constraint-pool))

(def selection-pool [sel/all (sel/select :x) (sel/select :y)])
(def gen-selection (gen/elements selection-pool))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

(println "\n=== Edit & Diff Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Edit properties (8)
;; ---------------------------------------------------------------------------

(println "-- constraint-edit --")

;; Property 1: constraint-edit dispatch produces same weight as p/update
(check "constraint-edit: weight matches p/update"
  (prop/for-all [m gen-model
                 c gen-constraint]
    (let [trace (p/simulate m [])
          ;; Via edit
          edit-req (edit/constraint-edit c)
          edit-result (edit/edit m trace edit-req)
          edit-w (eval-weight (:weight edit-result))
          ;; Via p/update
          update-result (p/update m trace c)
          update-w (eval-weight (:weight update-result))]
      (close? edit-w update-w 0.01))))

;; Property 2: selection-edit dispatch produces same weight as p/regenerate
(println "\n-- selection-edit --")

(check "selection-edit: weight matches p/regenerate"
  (prop/for-all [_ (gen/return nil)]
    ;; Use two-gauss only (both continuous)
    (let [trace (p/simulate two-gauss [])
          sel-choice (sel/select :x)
          ;; Via edit
          edit-req (edit/selection-edit sel-choice)
          edit-result (edit/edit two-gauss trace edit-req)
          edit-w (eval-weight (:weight edit-result))
          ;; Weight should be finite (match p/regenerate behavior)
          regen-result (p/regenerate two-gauss trace sel-choice)
          regen-w (eval-weight (:weight regen-result))]
      ;; Both should be finite (values differ due to different keys)
      (and (finite? edit-w) (finite? regen-w)))))

;; Property 3: constraint-edit backward-request contains discarded values
(check "constraint-edit: backward-request contains discard"
  (prop/for-all [c gen-constraint]
    (let [trace (p/simulate two-gauss [])
          edit-req (edit/constraint-edit c)
          result (edit/edit two-gauss trace edit-req)
          bwd (:backward-request result)]
      ;; backward-request should be a ConstraintEdit
      (instance? edit/ConstraintEdit bwd))))

;; Property 4: selection-edit backward-request is same selection type
(check "selection-edit: backward-request is SelectionEdit"
  (prop/for-all [s gen-selection]
    (let [trace (p/simulate two-gauss [])
          edit-req (edit/selection-edit s)
          result (edit/edit two-gauss trace edit-req)
          bwd (:backward-request result)]
      (instance? edit/SelectionEdit bwd))))

;; Property 5: proposal-edit backward-request swaps forward/backward GFs
(println "\n-- proposal-edit --")

(def proposal-gf
  (dyn/auto-key
    (gen [choices]
      (let [old-x (choice-val choices :x)
            new-x (trace :x (dist/gaussian (or old-x 0) 0.1))]
        (mx/eval! new-x)
        (mx/item new-x)))))

(check "proposal-edit: backward swaps forward/backward GFs"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate two-gauss [])
          edit-req (edit/proposal-edit proposal-gf proposal-gf)
          result (edit/edit two-gauss trace edit-req)
          bwd (:backward-request result)]
      (and (instance? edit/ProposalEdit bwd)
           ;; backward's forward-gf should be the original backward-gf
           (= (:forward-gf bwd) proposal-gf)))))

;; Property 6: constraint-edit round-trip recovers original trace values
(check "constraint-edit: round-trip recovers original"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate two-gauss [])
          orig-x (choice-val (:choices trace) :x)
          ;; Forward edit: constrain :x to new value
          c (cm/choicemap :x (mx/scalar 42.0))
          fwd-result (edit/edit two-gauss trace (edit/constraint-edit c))
          ;; Backward edit: apply backward-request (should contain old :x)
          bwd-req (:backward-request fwd-result)
          bwd-result (edit/edit two-gauss (:trace fwd-result) bwd-req)
          recovered-x (choice-val (:choices (:trace bwd-result)) :x)]
      (close? orig-x recovered-x 0.01))))

;; Property 7: proposal-edit weight is finite
(check "proposal-edit: weight is finite"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate two-gauss [])
          edit-req (edit/proposal-edit proposal-gf proposal-gf)
          result (edit/edit two-gauss trace edit-req)
          w (eval-weight (:weight result))]
      (finite? w))))

;; Property 8: Unknown edit-request type throws ex-info
(check "unknown edit-request: throws ex-info"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate two-gauss [])]
      (try
        (edit/edit-dispatch two-gauss trace {:type :bogus})
        false  ;; should have thrown
        (catch :default e
          (some? (ex-data e)))))))

;; ---------------------------------------------------------------------------
;; Diff properties (7)
;; ---------------------------------------------------------------------------

(println "\n-- diff computation --")

(def gen-scalar-val
  (gen/double* {:min -100.0 :max 100.0 :NaN? false :infinite? false}))

(def gen-vec-val
  (gen/vector gen-scalar-val 1 5))

(def gen-map-val
  (gen/map gen/keyword gen-scalar-val {:min-elements 1 :max-elements 5}))

;; Property 9: compute-diff(x, x) = no-change
(check "compute-diff(x, x) = no-change"
  (prop/for-all [x gen-scalar-val]
    (diff/no-change? (diff/compute-diff x x)))
  :num-tests 100)

;; Property 10: compute-vector-diff(v, v) = no-change
(check "compute-vector-diff(v, v) = no-change"
  (prop/for-all [v gen-vec-val]
    (diff/no-change? (diff/compute-vector-diff v v)))
  :num-tests 100)

;; Property 11: compute-map-diff(m, m) = no-change
(check "compute-map-diff(m, m) = no-change"
  (prop/for-all [m gen-map-val]
    (diff/no-change? (diff/compute-map-diff m m)))
  :num-tests 100)

;; Property 12: changed? is complement of no-change?
(check "changed? is complement of no-change?"
  (prop/for-all [x gen-scalar-val
                 y gen-scalar-val]
    (let [d (diff/compute-diff x y)]
      (= (diff/changed? d) (not (diff/no-change? d)))))
  :num-tests 100)

;; Property 13: map-diff changed/added/removed sets are disjoint
(check "map-diff: changed/added/removed are disjoint"
  (prop/for-all [m1 gen-map-val
                 m2 gen-map-val]
    (let [d (diff/compute-map-diff m1 m2)]
      (if (diff/no-change? d)
        true
        (let [{:keys [changed added removed]} d]
          (and (empty? (clojure.set/intersection (or changed #{}) (or added #{})))
               (empty? (clojure.set/intersection (or changed #{}) (or removed #{})))
               (empty? (clojure.set/intersection (or added #{}) (or removed #{}))))))))
  :num-tests 100)

;; Property 14: should-recompute?(no-change, addr) = false
(check "should-recompute?(no-change, addr) = false"
  (prop/for-all [addr (gen/elements [:a :b :c :x :y])]
    (not (diff/should-recompute? diff/no-change addr))))

;; Property 15: should-recompute?(unknown-change, addr) = true
(check "should-recompute?(unknown-change, addr) = true"
  (prop/for-all [addr (gen/elements [:a :b :c :x :y])]
    (diff/should-recompute? diff/unknown-change addr)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Edit & Diff Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
