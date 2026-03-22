(ns genmlx.edit-property-test
  "Property-based tests for the edit interface and its algebraic laws.
   Every test verifies a genuine mathematical invariant of trace mutation:
   invertibility via discard, identity elements, and equivalence with
   the underlying GFI protocol operations."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.edit :as edit])
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

(defn- check [name prop & {:keys [num-tests] :or {num-tests 30}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- choice-val
  "Extract a JS number from a choicemap at addr."
  [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

(defn- trace-score [trace]
  (mx/eval! (:score trace))
  (mx/item (:score trace)))

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Model pool
;; ---------------------------------------------------------------------------

(def two-site
  {:model (dyn/auto-key
            (gen []
              (let [x (trace :x (dist/gaussian 0 1))]
                (trace :y (dist/gaussian x 1)))))
   :addrs #{:x :y}
   :first-addr :x
   :label "two-site"})

(def three-site
  {:model (dyn/auto-key
            (gen []
              (let [a (trace :a (dist/gaussian 0 2))
                    b (trace :b (dist/gaussian a 1))]
                (trace :c (dist/gaussian b 0.5)))))
   :addrs #{:a :b :c}
   :first-addr :a
   :label "three-site"})

;; Independent model: sites have no dependencies between them
(def two-independent
  {:model (dyn/auto-key
            (gen []
              (let [x (trace :x (dist/gaussian 0 1))
                    y (trace :y (dist/gaussian 5 2))]
                (mx/eval! x y) (+ (mx/item x) (mx/item y)))))
   :addrs #{:x :y}
   :first-addr :x
   :label "two-independent"})

(def model-pool [two-site three-site])
(def gen-model (gen/elements model-pool))
(def gen-indep-model (gen/elements [two-independent]))
(def gen-constraint-val (gen/elements [-3.0 -1.0 0.0 1.0 3.0]))

(println "\n=== Edit Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; E14.1: ConstraintEdit round-trip — update then undo via discard
;; Law: constraint edit is invertible via discard
;; ---------------------------------------------------------------------------

(println "-- edit invertibility --")

(check "ConstraintEdit round-trip: update then undo via discard"
  (prop/for-all [m gen-model
                 v gen-constraint-val]
    (let [model (:model m)
          trace0 (p/simulate model [])
          score0 (trace-score trace0)
          ;; Save original choice values
          orig-vals (into {} (map (fn [a] [a (choice-val (:choices trace0) a)])
                                 (:addrs m)))
          ;; Forward edit: constrain first-addr to new value
          constraint (cm/choicemap (:first-addr m) (mx/scalar v))
          fwd (edit/edit-dispatch model trace0 (edit/constraint-edit constraint))
          trace1 (:trace fwd)
          discard (:discard fwd)
          ;; Backward edit: apply discard to recover original
          bwd (edit/edit-dispatch model trace1 (edit/constraint-edit discard))
          trace2 (:trace bwd)
          score2 (trace-score trace2)]
      ;; Score should be recovered
      (and (close? score0 score2 0.01)
           ;; Choice values at first-addr should be recovered
           (close? (get orig-vals (:first-addr m))
                   (choice-val (:choices trace2) (:first-addr m))
                   1e-6)))))

;; ---------------------------------------------------------------------------
;; E14.2: SelectionEdit weight = score difference
;; Law: selection edit weight is the MH acceptance ratio exponent
;; ---------------------------------------------------------------------------

;; For independent sites, the prior IS the proposal, so the MH acceptance
;; ratio exponent (weight) is exactly zero: the change in joint score from
;; resampling equals the proposal ratio, and they cancel.
(check "SelectionEdit on independent site: weight = 0"
  (prop/for-all [m gen-indep-model]
    (let [model (:model m)
          trace0 (p/simulate model [])
          result (edit/edit-dispatch model trace0
                                     (edit/selection-edit (sel/select (:first-addr m))))
          w (eval-weight (:weight result))]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; E14.3: ConstraintEdit with empty constraint = identity
;; Law: the empty constraint is the identity element for edit
;; ---------------------------------------------------------------------------

(println "\n-- edit identity elements --")

(check "ConstraintEdit with empty constraint = identity"
  (prop/for-all [m gen-model]
    (let [model (:model m)
          trace0 (p/simulate model [])
          orig-vals (into {} (map (fn [a] [a (choice-val (:choices trace0) a)])
                                 (:addrs m)))
          result (edit/edit-dispatch model trace0 (edit/constraint-edit cm/EMPTY))
          trace1 (:trace result)
          w (eval-weight (:weight result))]
      (and (close? 0.0 w 0.01)
           (every? (fn [a]
                     (close? (get orig-vals a)
                             (choice-val (:choices trace1) a)
                             1e-6))
                   (:addrs m))))))

;; ---------------------------------------------------------------------------
;; E14.4: SelectionEdit with sel/none = identity
;; Law: the empty selection is the identity element for regenerate-style edit
;; ---------------------------------------------------------------------------

(check "SelectionEdit with sel/none = identity"
  (prop/for-all [m gen-model]
    (let [model (:model m)
          trace0 (p/simulate model [])
          orig-vals (into {} (map (fn [a] [a (choice-val (:choices trace0) a)])
                                 (:addrs m)))
          result (edit/edit-dispatch model trace0 (edit/selection-edit sel/none))
          trace1 (:trace result)
          w (eval-weight (:weight result))]
      (and (close? 0.0 w 0.01)
           (every? (fn [a]
                     (close? (get orig-vals a)
                             (choice-val (:choices trace1) a)
                             1e-6))
                   (:addrs m))))))

;; ---------------------------------------------------------------------------
;; E14.5: Edit weight matches equivalent protocol call
;; Law: edit and update are equivalent for constraint edits
;; ---------------------------------------------------------------------------

(println "\n-- edit/protocol equivalence --")

(check "Edit weight matches equivalent protocol call"
  (prop/for-all [m gen-model
                 v gen-constraint-val]
    (let [model (:model m)
          trace0 (p/simulate model [])
          constraint (cm/choicemap (:first-addr m) (mx/scalar v))
          ;; Edit path
          edit-result (edit/edit-dispatch model trace0 (edit/constraint-edit constraint))
          edit-w (eval-weight (:weight edit-result))
          ;; Protocol path
          update-result (p/update model trace0 constraint)
          update-w (eval-weight (:weight update-result))]
      (close? edit-w update-w 1e-6))))

;; ---------------------------------------------------------------------------
;; E14.6: ProposalEdit forward/backward symmetry
;; Law: a symmetric proposal is approximately involutive
;; ---------------------------------------------------------------------------

(println "\n-- proposal symmetry --")

(check "ProposalEdit forward/backward symmetry"
  (prop/for-all [m gen-model]
    (let [model (:model m)
          trace0 (p/simulate model [])
          score0 (trace-score trace0)
          ;; Forward proposal: small gaussian perturbation of first address
          fwd-proposal (dyn/auto-key
                         (gen [choices]
                           (let [old-val (cm/get-value
                                           (cm/get-submap choices (:first-addr m)))]
                             (trace (:first-addr m) (dist/gaussian old-val 0.1)))))
          bwd-proposal fwd-proposal  ;; symmetric proposal
          ;; Forward edit
          fwd-result (edit/edit-dispatch model trace0
                       (edit/proposal-edit fwd-proposal bwd-proposal))
          fwd-w (eval-weight (:weight fwd-result))
          trace1 (:trace fwd-result)
          ;; Backward edit
          bwd-result (edit/edit-dispatch model trace1
                       (edit/proposal-edit bwd-proposal fwd-proposal))
          bwd-w (eval-weight (:weight bwd-result))
          trace2 (:trace bwd-result)
          score2 (trace-score trace2)]
      ;; Both weights must be finite
      (and (finite? fwd-w)
           (finite? bwd-w)
           ;; Score should approximately recover (small perturbation, so close)
           (close? score0 score2 2.0))))
  :num-tests 20)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Edit Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
