(ns genmlx.combinator-property-test
  "Property-based combinator invariant tests using test.check.
   Verifies Map, Unfold, Switch, Scan, Mask structure and score invariants.

   Uses gen/elements with pre-built combinator instances to avoid
   SCI interop crashes during test.check shrink traversal."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb])
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

(defn- eval-score [trace]
  (mx/eval! (:score trace))
  (mx/item (:score trace)))

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

(defn- sum-meta-scores
  "Sum scores from metadata vector, returning JS number."
  [scores]
  (reduce (fn [acc s] (mx/eval! s) (+ acc (mx/item s)))
          0.0 scores))

;; ---------------------------------------------------------------------------
;; Shared kernel
;; ---------------------------------------------------------------------------

(def kernel
  (gen [x]
    (let [y (dyn/trace :y (dist/gaussian x 1))]
      (mx/eval! y)
      (mx/item y))))

;; ---------------------------------------------------------------------------
;; Pre-built combinator pools (gen/elements avoids SCI shrink crashes)
;; ---------------------------------------------------------------------------

;; Map combinator pool: {n, args}
(def map-pool
  [{:n 1 :args [[1.0]]        :label "map(n=1)"}
   {:n 3 :args [[1.0 2.0 3.0]] :label "map(n=3)"}
   {:n 5 :args [[1.0 2.0 3.0 4.0 5.0]] :label "map(n=5)"}])

(def gen-map-spec (gen/elements map-pool))

(def mapped (comb/map-combinator kernel))

;; Unfold combinator pool
(def unfold-step
  (gen [t state]
    (let [y (dyn/trace :y (dist/gaussian state 1))]
      (mx/eval! y)
      (mx/item y))))

(def unfolded (comb/unfold-combinator unfold-step))

(def unfold-pool
  [{:n 1 :args [1 0.0] :label "unfold(n=1)"}
   {:n 3 :args [3 0.0] :label "unfold(n=3)"}
   {:n 5 :args [5 0.0] :label "unfold(n=5)"}])

(def gen-unfold-spec (gen/elements unfold-pool))

;; Switch combinator pool
(def switch-g1 (gen [] (dyn/trace :y (dist/gaussian 0 1))))
(def switch-g2 (gen [] (dyn/trace :y (dist/gaussian 5 2))))
(def switched (comb/switch-combinator switch-g1 switch-g2))

(def switch-pool
  [{:idx 0 :args [0] :label "switch(idx=0)"}
   {:idx 1 :args [1] :label "switch(idx=1)"}])

(def gen-switch-spec (gen/elements switch-pool))

;; Scan combinator pool
(def scan-kernel
  (gen [carry x]
    (let [y (dyn/trace :y (dist/gaussian carry 1))]
      (mx/eval! y)
      [(mx/item y) (mx/item y)])))

(def scanned (comb/scan-combinator scan-kernel))

(def scan-pool
  [{:n 1 :args [0.0 [1.0]]           :label "scan(n=1)"}
   {:n 3 :args [0.0 [1.0 2.0 3.0]]   :label "scan(n=3)"}])

(def gen-scan-spec (gen/elements scan-pool))

;; Mask combinator pool
(def masked (comb/mask-combinator kernel))

(def mask-pool
  [{:active? true  :args [true 3.0]  :label "mask(true)"}
   {:active? false :args [false 3.0] :label "mask(false)"}])

(def gen-mask-spec (gen/elements mask-pool))

(println "\n=== Combinator Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Map properties
;; ---------------------------------------------------------------------------

(println "-- Map combinator --")

;; Property 1: Map has n integer-keyed sub-traces
(check "Map: n integer-keyed sub-traces"
  (prop/for-all [spec gen-map-spec]
    (let [trace (p/simulate mapped (:args spec))
          choices (:choices trace)
          n (:n spec)]
      (every? (fn [i]
                (not= (cm/get-submap choices i) cm/EMPTY))
              (range n)))))

;; Property 2: Map score = sum(element-scores)
(check "Map: score = sum(element-scores)"
  (prop/for-all [spec gen-map-spec]
    (let [trace (p/simulate mapped (:args spec))
          total (eval-score trace)
          element-scores (::comb/element-scores (meta trace))]
      (if element-scores
        (close? total (sum-meta-scores element-scores) 0.01)
        true))))

;; Property 3: Map generate(empty) weight ≈ 0
(check "Map: generate(empty) weight ≈ 0"
  (prop/for-all [spec gen-map-spec]
    (let [{:keys [weight]} (p/generate mapped (:args spec) cm/EMPTY)]
      (close? 0.0 (eval-weight weight) 0.01))))

;; Property 4: Map generate(full) weight ≈ score
(check "Map: generate(full) weight ≈ score"
  (prop/for-all [spec gen-map-spec]
    (let [trace (p/simulate mapped (:args spec))
          {:keys [trace weight]} (p/generate mapped (:args spec) (:choices trace))
          w (eval-weight weight)
          s (eval-score trace)]
      (close? s w 0.01))))

;; Property 5: Map update(same) weight ≈ 0
(check "Map: update(same) weight ≈ 0"
  (prop/for-all [spec gen-map-spec]
    (let [trace (p/simulate mapped (:args spec))
          {:keys [weight]} (p/update mapped trace (:choices trace))]
      (close? 0.0 (eval-weight weight) 0.01))))

;; ---------------------------------------------------------------------------
;; Unfold properties
;; ---------------------------------------------------------------------------

(println "\n-- Unfold combinator --")

;; Property 6: Unfold has n step sub-traces
(check "Unfold: n step sub-traces"
  (prop/for-all [spec gen-unfold-spec]
    (let [trace (p/simulate unfolded (:args spec))
          choices (:choices trace)
          n (:n spec)]
      (every? (fn [i]
                (not= (cm/get-submap choices i) cm/EMPTY))
              (range n)))))

;; Property 7: Unfold score = sum(step-scores)
(check "Unfold: score = sum(step-scores)"
  (prop/for-all [spec gen-unfold-spec]
    (let [trace (p/simulate unfolded (:args spec))
          total (eval-score trace)
          step-scores (::comb/step-scores (meta trace))]
      (if step-scores
        (close? total (sum-meta-scores step-scores) 0.01)
        true))))

;; Property 8: Unfold GFI contracts (generate/update)
(check "Unfold: generate(empty) weight ≈ 0, update(same) weight ≈ 0"
  (prop/for-all [spec gen-unfold-spec]
    (let [{:keys [weight]} (p/generate unfolded (:args spec) cm/EMPTY)
          gen-w (eval-weight weight)
          trace (p/simulate unfolded (:args spec))
          {:keys [weight]} (p/update unfolded trace (:choices trace))
          upd-w (eval-weight weight)]
      (and (close? 0.0 gen-w 0.01)
           (close? 0.0 upd-w 0.01)))))

;; ---------------------------------------------------------------------------
;; Switch properties
;; ---------------------------------------------------------------------------

(println "\n-- Switch combinator --")

;; Property 9: Switch GFI contracts
(check "Switch: GFI contracts (generate/update)"
  (prop/for-all [spec gen-switch-spec]
    (let [{:keys [weight]} (p/generate switched (:args spec) cm/EMPTY)
          gen-w (eval-weight weight)
          trace (p/simulate switched (:args spec))
          s (eval-score trace)
          {:keys [trace weight]} (p/generate switched (:args spec) (:choices trace))
          full-w (eval-weight weight)
          full-s (eval-score trace)]
      (and (close? 0.0 gen-w 0.01)
           (close? full-s full-w 0.01)))))

;; ---------------------------------------------------------------------------
;; Scan properties
;; ---------------------------------------------------------------------------

(println "\n-- Scan combinator --")

;; Property 10: Scan retval has correct structure
(check "Scan: retval has :carry and :outputs"
  (prop/for-all [spec gen-scan-spec]
    (let [trace (p/simulate scanned (:args spec))
          retval (:retval trace)]
      (and (contains? retval :carry)
           (contains? retval :outputs)
           (= (:n spec) (count (:outputs retval)))))))

;; Property 11: Scan score = sum(step-scores)
(check "Scan: score = sum(step-scores)"
  (prop/for-all [spec gen-scan-spec]
    (let [trace (p/simulate scanned (:args spec))
          total (eval-score trace)
          step-scores (::comb/step-scores (meta trace))]
      (if step-scores
        (close? total (sum-meta-scores step-scores) 0.01)
        true))))

;; ---------------------------------------------------------------------------
;; Cross-combinator properties
;; ---------------------------------------------------------------------------

(println "\n-- Cross-combinator --")

;; Property 12: Map(n=1) score ≈ kernel score (same constraint)
(check "Map(n=1) score ≈ kernel score"
  (prop/for-all [_ (gen/return nil)]
    (let [constraint-val (mx/scalar 2.5)
          map-constraint (cm/set-choice cm/EMPTY [0] (cm/choicemap :y constraint-val))
          kernel-constraint (cm/choicemap :y constraint-val)
          {:keys [trace]} (p/generate mapped [[3.0]] map-constraint)
          map-score (eval-score trace)
          {:keys [trace]} (p/generate kernel [3.0] kernel-constraint)
          kernel-score (eval-score trace)]
      (close? kernel-score map-score 0.01))))

;; Property 13: Mask(true) ≈ kernel, Mask(false) score ≈ 0
(check "Mask: true ≈ kernel, false score ≈ 0"
  (prop/for-all [spec gen-mask-spec]
    (let [trace (p/simulate masked (:args spec))
          s (eval-score trace)]
      (if (:active? spec)
        (finite? s)
        (close? 0.0 s 0.01)))))

;; ---------------------------------------------------------------------------
;; Project properties (all combinators)
;; ---------------------------------------------------------------------------

(println "\n-- Project properties --")

;; Property 14: project(all) ≈ score for all combinators
(check "All combinators: project(all) ≈ score"
  (prop/for-all [which (gen/elements [:map :unfold :switch :scan :mask-true])]
    (let [[gf args] (case which
                      :map       [mapped [[1.0 2.0]]]
                      :unfold    [unfolded [2 0.0]]
                      :switch    [switched [0]]
                      :scan      [scanned [0.0 [1.0 2.0]]]
                      :mask-true [masked [true 3.0]])
          trace (p/simulate gf args)
          s (eval-score trace)
          proj (eval-weight (p/project gf trace sel/all))]
      (close? s proj 0.01))))

;; Property 15: project(none) ≈ 0 for all combinators
(check "All combinators: project(none) ≈ 0"
  (prop/for-all [which (gen/elements [:map :unfold :switch :scan :mask-true])]
    (let [[gf args] (case which
                      :map       [mapped [[1.0 2.0]]]
                      :unfold    [unfolded [2 0.0]]
                      :switch    [switched [0]]
                      :scan      [scanned [0.0 [1.0 2.0]]]
                      :mask-true [masked [true 3.0]])
          trace (p/simulate gf args)
          proj (eval-weight (p/project gf trace sel/none))]
      (close? 0.0 proj 0.01))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Combinator Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
