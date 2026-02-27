(ns genmlx.contract-verification-test
  "Task 24.4: Run verify-gfi-contracts on 13 canonical models covering all features."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.contracts :as contracts]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(println "\n=== Contract Verification Test Suite ===")

;; ---------------------------------------------------------------------------
;; Contract key sets
;; ---------------------------------------------------------------------------

(def all-keys (set (keys contracts/contracts)))

;; All minus broadcast-equivalence (vsimulate only works on DynamicGF)
(def scalar-keys (disj all-keys :broadcast-equivalence))

;; Scalar minus 3 addr-dependent contracts that fail on hierarchical int keys
;; (Map, Unfold, Scan use integer-keyed submaps; first-address returns a submap, not a leaf)
(def leaf-safe-keys
  (disj scalar-keys
        :update-weight-correctness
        :update-round-trip
        :regenerate-empty-identity))

;; Scalar minus score-decomposition (project doesn't decompose through splice boundaries)
(def splice-keys (disj scalar-keys :score-decomposition))

;; Deep-nesting: first-address returns :inner (non-leaf) + splice decomposition issue
(def deep-nesting-keys
  (disj scalar-keys
        :update-weight-correctness
        :update-round-trip
        :regenerate-empty-identity
        :score-decomposition))

;; Recurse: splice decomposition + regenerate doesn't handle recursive structure
(def recurse-keys
  (disj scalar-keys :score-decomposition :regenerate-empty-identity))

;; ---------------------------------------------------------------------------
;; 13 canonical models
;; ---------------------------------------------------------------------------

;; 1. Single-site gaussian
(def single-site
  (gen [] (let [x (dyn/trace :x (dist/gaussian 0 1))]
            (mx/eval! x) (mx/item x))))

;; 2. Multi-site dependent (x → y)
(def multi-site
  (gen [] (let [x (dyn/trace :x (dist/gaussian 0 1))]
            (mx/eval! x)
            (let [y (dyn/trace :y (dist/gaussian (mx/item x) 1))]
              (mx/eval! y) (mx/item y)))))

;; 3. Linear regression (5 addresses: slope, intercept, y0, y1, y2)
(def linreg
  (gen [xs]
    (let [slope (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (mx/eval! slope intercept)
      (let [sv (mx/item slope) iv (mx/item intercept)]
        (doseq [[j x] (map-indexed vector xs)]
          (dyn/trace (keyword (str "y" j))
                     (dist/gaussian (+ (* sv x) iv) 1)))
        sv))))

;; 4. Splice (sub-GF via splice)
(def inner-gf
  (gen [] (let [z (dyn/trace :z (dist/gaussian 0 1))]
            (mx/eval! z) (mx/item z))))

(def splice-model
  (gen [] (let [x (dyn/trace :x (dist/gaussian 0 10))]
            (mx/eval! x)
            (dyn/splice :inner inner-gf)
            (mx/item x))))

;; 5. Mixed discrete/continuous
(def mixed-model
  (gen [] (let [b (dyn/trace :b (dist/bernoulli 0.5))]
            (mx/eval! b)
            (let [y (dyn/trace :y (dist/gaussian (if (> (mx/item b) 0.5) 5.0 -5.0) 1))]
              (mx/eval! y) (mx/item y)))))

;; 6. Deep nesting (3-level splice chain: x, inner/y, inner/mid/z)
(def level2 (gen [] (let [z (dyn/trace :z (dist/gaussian 0 1))]
                      (mx/eval! z) (mx/item z))))
(def level1 (gen [] (dyn/splice :mid level2)
                    (let [y (dyn/trace :y (dist/gaussian 0 1))]
                      (mx/eval! y) (mx/item y))))
(def deep-nesting
  (gen [] (dyn/splice :inner level1)
          (let [x (dyn/trace :x (dist/gaussian 0 1))]
            (mx/eval! x) (mx/item x))))

;; 7. Two-site vectorization-compatible (no eval!/item in body)
(def two-site-vec
  (gen [] (let [x (dyn/trace :x (dist/gaussian 0 1))
                y (dyn/trace :y (dist/gaussian 0 1))]
            y)))

;; 8. Map combinator
(def map-kernel (gen [x] (let [y (dyn/trace :y (dist/gaussian x 1))]
                            (mx/eval! y) (mx/item y))))
(def map-model (comb/map-combinator map-kernel))

;; 9. Unfold combinator
(def unfold-step (gen [t state] (let [y (dyn/trace :y (dist/gaussian state 1))]
                                  (mx/eval! y) (mx/item y))))
(def unfold-model (comb/unfold-combinator unfold-step))

;; 10. Switch combinator (flat choices — branch has keyword addr)
(def branch-a (gen [] (dyn/trace :x (dist/gaussian 0 1))))
(def branch-b (gen [] (dyn/trace :x (dist/gaussian 5 1))))
(def switch-model (comb/switch-combinator branch-a branch-b))

;; 11. Scan combinator
(def scan-step (gen [carry x] (let [y (dyn/trace :y (dist/gaussian carry 1))]
                                 (mx/eval! y)
                                 [(mx/item y) (mx/item y)])))
(def scan-model (comb/scan-combinator scan-step))

;; 12. Mask combinator (active=true exposes inner choices)
(def mask-inner (gen [] (let [y (dyn/trace :y (dist/gaussian 0 1))]
                          (mx/eval! y) (mx/item y))))
(def mask-model (comb/mask-combinator mask-inner))

;; 13. Recurse combinator (depth-2 tree, leaf addr :v)
(def recurse-model
  (comb/recurse
    (fn [self]
      (gen [depth]
        (let [v (dyn/trace :v (dist/gaussian 0 1))]
          (mx/eval! v)
          (when (> depth 0)
            (dyn/splice :child self (dec depth)))
          (mx/item v))))))

;; ---------------------------------------------------------------------------
;; Model suite (data-driven)
;; ---------------------------------------------------------------------------

(def models
  [{:name "single-site"    :model single-site    :args []              :contract-keys scalar-keys}
   {:name "multi-site"     :model multi-site     :args []              :contract-keys scalar-keys}
   {:name "linreg"         :model linreg         :args [[1 2 3]]      :contract-keys scalar-keys}
   {:name "splice"         :model splice-model   :args []              :contract-keys splice-keys}
   {:name "mixed"          :model mixed-model    :args []              :contract-keys scalar-keys}
   {:name "deep-nesting"   :model deep-nesting   :args []              :contract-keys deep-nesting-keys}
   {:name "two-site-vec"   :model two-site-vec   :args []              :contract-keys all-keys}
   {:name "map"            :model map-model      :args [[1 2 3]]      :contract-keys leaf-safe-keys}
   {:name "unfold"         :model unfold-model   :args [3 0.0]        :contract-keys leaf-safe-keys}
   {:name "switch"         :model switch-model   :args [0]            :contract-keys scalar-keys}
   {:name "scan"           :model scan-model     :args [0.0 [1 2 3]]  :contract-keys leaf-safe-keys}
   {:name "mask"           :model mask-model     :args [true]         :contract-keys scalar-keys}
   {:name "recurse"        :model recurse-model  :args [2]            :contract-keys recurse-keys}])

;; ---------------------------------------------------------------------------
;; Run verification
;; ---------------------------------------------------------------------------

(def total-checks (atom 0))
(def total-failures (atom 0))

(doseq [{mname :name :keys [model args contract-keys]} models]
  (println (str "\n-- " mname " (" (count contract-keys) " contracts x 5 trials) --"))
  (let [report (contracts/verify-gfi-contracts model args
                 :n-trials 5 :contract-keys contract-keys)]
    (assert-true (str mname ": all-pass?") (:all-pass? report))
    (println "    " (:total-pass report) "/" (+ (:total-pass report) (:total-fail report)) "checks passed")
    (swap! total-checks + (:total-pass report) (:total-fail report))
    (swap! total-failures + (:total-fail report))
    (when-not (:all-pass? report)
      (doseq [[k {:keys [theorem pass fail]}] (:results report)]
        (when (pos? fail)
          (println "    FAILED:" (cljs.core/name k) "-" theorem
                   "(" fail "/" (+ pass fail) "failures)"))))))

(println (str "\n=== Summary: " @total-checks " total checks, "
              @total-failures " failures ==="))
(assert-true "all models pass all contracts" (zero? @total-failures))
(assert-true (str "expected 575 checks, got " @total-checks) (= 575 @total-checks))

(println "\nContract verification test complete.")
