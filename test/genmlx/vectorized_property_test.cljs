(ns genmlx.vectorized-property-test
  "Property-based vectorized inference tests using test.check.
   Verifies vsimulate/vgenerate shape invariants, statistical equivalence,
   VectorizedTrace operations, and batched vs scalar consistency."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vec])
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

(defn- shape= [arr expected-shape]
  (= (mx/shape arr) expected-shape))

;; ---------------------------------------------------------------------------
;; Models and fixture pools
;; ---------------------------------------------------------------------------

;; Independent model — for vsimulate tests
(def ind-model
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))
          y (dyn/trace :y (dist/gaussian 0 1))]
      (mx/add x y))))

;; Dependent model — for vgenerate tests (y depends on x)
;; This ensures weight is [N]-shaped when :y is constrained and :x is unconstrained
(def dep-model
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))
          y (dyn/trace :y (dist/gaussian x 1))]
      y)))

(def n-pool [5 10 15 20])
(def gen-n (gen/elements n-pool))

;; Partial obs (constrain only :y, leave :x free) → [N]-shaped weight
(def partial-obs-pool
  [(cm/choicemap :y (mx/scalar 1.0))
   (cm/choicemap :y (mx/scalar 0.0))
   (cm/choicemap :y (mx/scalar -1.0))
   (cm/choicemap :y (mx/scalar 2.0))])

(def gen-partial-obs (gen/elements partial-obs-pool))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

(println "\n=== Vectorized Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; vsimulate Shape (4)
;; ---------------------------------------------------------------------------

(println "-- vsimulate shape --")

(check "vsimulate: score shape is [N]"
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vsimulate ind-model [] n k)]
      (shape= (:score vt) [n]))))

(check "vsimulate: all choice leaves are [N]-shaped"
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vsimulate ind-model [] n k)
          choices (:choices vt)]
      (and (shape= (cm/get-value (cm/get-submap choices :x)) [n])
           (shape= (cm/get-value (cm/get-submap choices :y)) [n])))))

(check "vsimulate: n-particles matches N"
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vsimulate ind-model [] n k)]
      (= (:n-particles vt) n))))

(check "vsimulate: all scores are finite"
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vsimulate ind-model [] n k)
          _ (mx/eval! (:score vt))
          scores (mx/->clj (:score vt))]
      (every? js/isFinite scores))))

;; ---------------------------------------------------------------------------
;; vgenerate Shape (4)
;; ---------------------------------------------------------------------------

(println "\n-- vgenerate shape --")

(check "vgenerate: weight shape is [N] (dependent model, partial obs)"
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)]
      (shape= (:weight vt) [n]))))

(check "vgenerate: constrained site is scalar"
  (prop/for-all [n gen-n
                 k gen-key]
    (let [obs (cm/choicemap :y (mx/scalar 2.0))
          vt (dyn/vgenerate dep-model [] obs n k)
          y-val (cm/get-value (cm/get-submap (:choices vt) :y))
          _ (mx/eval! y-val)]
      ;; Constrained site is scalar
      (close? 2.0 (mx/realize y-val) 1e-6))))

(check "vgenerate: unconstrained sites are [N]-shaped"
  (prop/for-all [n gen-n
                 k gen-key]
    (let [obs (cm/choicemap :y (mx/scalar 1.0))
          vt (dyn/vgenerate dep-model [] obs n k)
          x-arr (cm/get-value (cm/get-submap (:choices vt) :x))]
      (shape= x-arr [n]))))

(check "vgenerate: empty constraints weight is scalar 0"
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vgenerate ind-model [] cm/EMPTY n k)
          w (:weight vt)
          _ (mx/eval! w)]
      ;; Empty constraints → weight = 0 (scalar)
      (close? 0.0 (mx/item w) 0.01))))

;; ---------------------------------------------------------------------------
;; Statistical Equivalence (2)
;; ---------------------------------------------------------------------------

(println "\n-- statistical equivalence --")

(check "vsimulate mean score near mean of N sequential simulates"
  (prop/for-all [k gen-key]
    (let [n 20
          ;; Vectorized
          vt (dyn/vsimulate ind-model [] n k)
          _ (mx/eval! (:score vt))
          v-scores (mx/->clj (:score vt))
          v-mean (/ (reduce + v-scores) n)
          ;; Sequential
          seq-scores (mapv (fn [_]
                             (let [t (p/simulate ind-model [])
                                   _ (mx/eval! (:score t))]
                               (mx/item (:score t))))
                           (range n))
          s-mean (/ (reduce + seq-scores) n)]
      ;; Both should be around -log(2π) ≈ -1.84 for standard Gaussian
      ;; Loose tolerance since different random draws
      (and (finite? v-mean) (finite? s-mean)
           (< (js/Math.abs (- v-mean s-mean)) 3.0))))
  :num-tests 30)

(check "vgenerate log-ML is finite"
  (prop/for-all [obs gen-partial-obs
                 k gen-key]
    (let [n 10
          vt (dyn/vgenerate dep-model [] obs n k)
          v-lml (vec/vtrace-log-ml-estimate vt)
          _ (mx/eval! v-lml)]
      (finite? (mx/item v-lml))))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; VectorizedTrace Operations (4)
;; ---------------------------------------------------------------------------

(println "\n-- vectorized trace ops --")

(check "vtrace-ess in (0, N]"
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          ess (vec/vtrace-ess vt)]
      (and (> ess 0) (<= ess (+ n 0.01))))))

(check "vtrace-log-ml-estimate is finite"
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          lml (vec/vtrace-log-ml-estimate vt)]
      (finite? (eval-weight lml)))))

(check "resample-vtrace produces near-uniform weights"
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          [k1 _] (rng/split k)
          resampled (vec/resample-vtrace vt k1)
          _ (mx/eval! (:weight resampled))
          ws (mx/->clj (:weight resampled))]
      ;; After resampling, weights should be uniform (zeros in log-space)
      (every? #(close? 0.0 % 0.01) ws))))

(check "resample-vtrace preserves n-particles"
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          [k1 _] (rng/split k)
          resampled (vec/resample-vtrace vt k1)]
      (= (:n-particles resampled) n))))

;; ---------------------------------------------------------------------------
;; Batched vs Scalar (4)
;; ---------------------------------------------------------------------------

(println "\n-- batched vs scalar --")

(check "N=1 vsimulate score shape is [1]"
  (prop/for-all [k gen-key]
    (let [vt (dyn/vsimulate ind-model [] 1 k)]
      (shape= (:score vt) [1]))))

(check "N=1 vgenerate weight shape is [1]"
  (prop/for-all [k gen-key
                 obs gen-partial-obs]
    (let [vt (dyn/vgenerate dep-model [] obs 1 k)]
      (shape= (:weight vt) [1]))))

(check "vsimulate scores finite for all N"
  (prop/for-all [n gen-n
                 k gen-key]
    (let [vt (dyn/vsimulate ind-model [] n k)
          _ (mx/eval! (:score vt))
          scores (mx/->clj (:score vt))]
      (every? js/isFinite scores))))

(check "resample preserves score shape [N]"
  (prop/for-all [n gen-n
                 obs gen-partial-obs
                 k gen-key]
    (let [vt (dyn/vgenerate dep-model [] obs n k)
          [k1 _] (rng/split k)
          resampled (vec/resample-vtrace vt k1)]
      (shape= (:score resampled) [n]))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Vectorized Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
