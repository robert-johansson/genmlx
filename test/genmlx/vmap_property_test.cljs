(ns genmlx.vmap-property-test
  "Property-based tests for the Vmap combinator's GFI contract.
   Verifies that vmap-gf and repeat-gf satisfy the same algebraic laws
   as any generative function: simulate liveness, importance weighting
   identity, update invertibility, score additivity, and projection
   decomposition."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.vmap :as vmap])
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

;; ---------------------------------------------------------------------------
;; Shared kernel and vmap instances
;; ---------------------------------------------------------------------------

(def kernel-gf
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/gaussian x 1))]
        y))))

(def vmap-gf-3 (vmap/vmap-gf kernel-gf))
(def vmap-gf-1 (vmap/vmap-gf kernel-gf))

;; Multi-address kernel for project decomposition test
(def multi-kernel-gf
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/gaussian x 1))
            z (trace :z (dist/gaussian x 2))]
        (mx/add y z)))))

(def multi-vmap-3 (vmap/vmap-gf multi-kernel-gf))

;; ---------------------------------------------------------------------------
;; Pools (gen/elements for SCI shrink safety)
;; ---------------------------------------------------------------------------

(def args-pool-3
  [{:args [(mx/array [1.0 2.0 3.0])]   :label "args=[1,2,3]"}
   {:args [(mx/array [0.0 0.0 0.0])]   :label "args=[0,0,0]"}
   {:args [(mx/array [-1.0 5.0 2.0])]  :label "args=[-1,5,2]"}])

(def gen-args-3 (gen/elements args-pool-3))

(def args-pool-1
  [{:args [(mx/array [1.0])]  :label "args=[1]"}
   {:args [(mx/array [0.0])]  :label "args=[0]"}
   {:args [(mx/array [-2.0])] :label "args=[-2]"}])

(def gen-args-1 (gen/elements args-pool-1))

(def multi-args-pool
  [{:args [(mx/array [1.0 2.0 3.0])]  :label "args=[1,2,3]"}
   {:args [(mx/array [0.0 0.0 0.0])]  :label "args=[0,0,0]"}])

(def gen-multi-args (gen/elements multi-args-pool))

(def gen-constraint-val (gen/elements [-3.0 -1.0 0.0 1.0 3.0]))

(println "\n=== Vmap Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; E15.1: simulate produces valid trace with finite score
;; Law: GFI liveness — simulate always returns a well-formed trace
;; ---------------------------------------------------------------------------

(println "-- simulate --")

(check "simulate: finite score, addresses present"
  (prop/for-all [spec gen-args-3]
    (let [trace (p/simulate vmap-gf-3 (:args spec))
          s (eval-score trace)
          choices (:choices trace)
          y-sub (cm/get-submap choices :y)]
      (and (finite? s)
           (cm/has-value? y-sub)
           (let [v (cm/get-value y-sub)]
             (mx/eval! v)
             (= [3] (mx/shape v)))))))

;; ---------------------------------------------------------------------------
;; E15.2: generate(empty) weight = 0
;; Law: importance weighting identity — unconstrained generate has unit weight
;; ---------------------------------------------------------------------------

(println "\n-- generate --")

(check "generate(empty): weight = 0"
  (prop/for-all [spec gen-args-3]
    (let [{:keys [weight]} (p/generate vmap-gf-3 (:args spec) cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; E15.3: generate(full) weight = score
;; Law: fully constrained generate has weight equal to joint log-probability
;; ---------------------------------------------------------------------------

(check "generate(full): weight = score"
  (prop/for-all [spec gen-args-3]
    (let [trace (p/simulate vmap-gf-3 (:args spec))
          {:keys [trace weight]} (p/generate vmap-gf-3 (:args spec) (:choices trace))
          w (eval-weight weight)
          s (eval-score trace)]
      (close? s w 0.01))))

;; ---------------------------------------------------------------------------
;; E15.4: update(same) weight = 0
;; Law: updating a trace with its own choices is the identity
;; ---------------------------------------------------------------------------

(println "\n-- update --")

(check "update(same): weight = 0"
  (prop/for-all [spec gen-args-3]
    (let [trace (p/simulate vmap-gf-3 (:args spec))
          {:keys [weight]} (p/update vmap-gf-3 trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

;; ---------------------------------------------------------------------------
;; E15.5: update round-trip via discard
;; Law: update is invertible — applying the discard recovers the original trace
;; ---------------------------------------------------------------------------

(check "update round-trip via discard"
  (prop/for-all [spec gen-args-3
                 v gen-constraint-val]
    (let [trace0 (p/simulate vmap-gf-3 (:args spec))
          score0 (eval-score trace0)
          ;; Update with new [3]-shaped constraint
          constraint (cm/choicemap :y (mx/array [v v v]))
          {:keys [trace discard]} (p/update vmap-gf-3 trace0 constraint)
          ;; Round-trip: apply discard
          {:keys [trace]} (p/update vmap-gf-3 trace discard)
          score2 (eval-score trace)]
      (close? score0 score2 0.01))))

;; ---------------------------------------------------------------------------
;; E15.6: regenerate(none) weight = 0, choices exactly preserved
;; Law: regenerating nothing is the identity on traces
;; ---------------------------------------------------------------------------

(println "\n-- regenerate --")

(check "regenerate(none): weight = 0, choices preserved"
  (prop/for-all [spec gen-args-3]
    (let [trace0 (p/simulate vmap-gf-3 (:args spec))
          y-orig (cm/get-value (cm/get-submap (:choices trace0) :y))
          _ (mx/eval! y-orig)
          orig-vals (mapv #(mx/item (mx/index y-orig %)) (range 3))
          {:keys [trace weight]} (p/regenerate vmap-gf-3 trace0 sel/none)
          w (eval-weight weight)
          y-new (cm/get-value (cm/get-submap (:choices trace) :y))
          _ (mx/eval! y-new)
          new-vals (mapv #(mx/item (mx/index y-new %)) (range 3))]
      (and (close? 0.0 w 0.01)
           (every? true?
                   (map (fn [o n] (close? o n 1e-6)) orig-vals new-vals))))))

;; ---------------------------------------------------------------------------
;; E15.7: project(all) = score
;; Law: projecting all addresses yields the total log-probability
;; ---------------------------------------------------------------------------

(println "\n-- project --")

(check "project(all) = score"
  (prop/for-all [spec gen-args-3]
    (let [trace (p/simulate vmap-gf-3 (:args spec))
          s (eval-score trace)
          proj (eval-weight (p/project vmap-gf-3 trace sel/all))]
      (close? s proj 0.01))))

;; ---------------------------------------------------------------------------
;; E15.8: project(none) = 0
;; Law: projecting no addresses yields zero log-probability
;; ---------------------------------------------------------------------------

(check "project(none) = 0"
  (prop/for-all [spec gen-args-3]
    (let [trace (p/simulate vmap-gf-3 (:args spec))
          proj (eval-weight (p/project vmap-gf-3 trace sel/none))]
      (close? 0.0 proj 0.01))))

;; ---------------------------------------------------------------------------
;; E15.9: project(S) + project(complement(S)) = score
;; Law: projection is additively decomposable over address partitions
;; ---------------------------------------------------------------------------

(check "project(S) + project(complement(S)) = score"
  (prop/for-all [spec gen-multi-args]
    (let [trace (p/simulate multi-vmap-3 (:args spec))
          score (eval-score trace)
          s (sel/select :y)
          proj-s (eval-weight (p/project multi-vmap-3 trace s))
          proj-cs (eval-weight (p/project multi-vmap-3 trace (sel/complement-sel s)))]
      (close? score (+ proj-s proj-cs) 0.1))))

;; ---------------------------------------------------------------------------
;; E15.10: vmap(f, n=1) generate score = f generate score (same constraint)
;; Law: the degenerate vmap of one element has the same score as the kernel
;; ---------------------------------------------------------------------------

(println "\n-- degenerate cases --")

(check "vmap(f, n=1) generate score = f generate score"
  (prop/for-all [spec gen-args-1
                 v gen-constraint-val]
    (let [;; vmap generate with [1]-shaped constraint
          vmap-obs (cm/choicemap :y (mx/array [v]))
          {:keys [trace]} (p/generate vmap-gf-1 (:args spec) vmap-obs)
          vmap-score (eval-score trace)
          ;; kernel generate with scalar constraint
          scalar-arg (mx/index (first (:args spec)) 0)
          kernel-obs (cm/choicemap :y (mx/scalar v))
          {:keys [trace]} (p/generate kernel-gf [scalar-arg] kernel-obs)
          kernel-score (eval-score trace)]
      (close? vmap-score kernel-score 1e-5))))

;; ---------------------------------------------------------------------------
;; E15.11: vmap score = sum(element scores)
;; Law: score additivity — the joint log-prob of independent sub-traces
;;      equals the sum of their individual log-probabilities
;; ---------------------------------------------------------------------------

(println "\n-- score additivity --")

(check "vmap score = sum(element scores)"
  (prop/for-all [spec gen-args-3]
    (let [trace (p/simulate vmap-gf-3 (:args spec))
          total-score (eval-score trace)
          ;; Extract per-element scores from metadata
          element-scores (::vmap/element-scores (meta trace))]
      (if element-scores
        (let [score-sum (reduce (fn [acc s]
                                  (mx/eval! s)
                                  (+ acc (mx/item s)))
                                0.0 element-scores)]
          (close? total-score score-sum 0.01))
        ;; Fallback: generate per-element traces with matching choices
        ;; and sum their scores
        (let [choices (:choices trace)
              y-val (cm/get-value (cm/get-submap choices :y))
              _ (mx/eval! y-val)
              input-arr (first (:args spec))
              elem-scores
              (reduce
                (fn [acc i]
                  (let [yi (mx/index y-val i)
                        xi (mx/index input-arr i)
                        obs (cm/choicemap :y yi)
                        {:keys [trace]} (p/generate kernel-gf [xi] obs)
                        s (eval-score trace)]
                    (+ acc s)))
                0.0
                (range 3))]
          (close? total-score elem-scores 0.01))))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Vmap Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
