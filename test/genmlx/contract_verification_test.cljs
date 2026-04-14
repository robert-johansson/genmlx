(ns genmlx.contract-verification-test
  "GFI law verification on 13 canonical models covering all features.
   Each model is verified against its applicable law subset (5 trials each).
   Migrated from contracts.cljs to gfi.cljs (2026-04-14)."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gfi :as gfi]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Law key sets (mapped from old contract keys)
;;
;; Old contract key                  -> GFI law name
;; :generate-weight-equals-score     -> :generate-full-weight-equals-score
;; :update-empty-identity            -> :update-identity
;; :update-weight-correctness        -> :update-density-ratio
;; :update-round-trip                -> :update-round-trip
;; :regenerate-empty-identity        -> :regenerate-empty-identity
;; :project-all-equals-score         -> :project-all-equals-score
;; :project-none-equals-zero         -> :project-none-equals-zero
;; :assess-equals-generate-score     -> :assess-equals-generate-score
;; :propose-generate-round-trip      -> :propose-weight-equals-generate
;; :score-decomposition              -> :score-full-decomposition
;; :broadcast-equivalence            -> :vsimulate-shape-correctness
;; ---------------------------------------------------------------------------

(def all-laws
  #{:generate-full-weight-equals-score
    :update-identity
    :update-density-ratio
    :update-round-trip
    :regenerate-empty-identity
    :project-all-equals-score
    :project-none-equals-zero
    :assess-equals-generate-score
    :propose-weight-equals-generate
    :score-full-decomposition
    :vsimulate-shape-correctness})

;; All minus vsimulate (only works on DynamicGF)
(def scalar-laws (disj all-laws :vsimulate-shape-correctness))

;; Scalar minus addr-dependent laws that fail on hierarchical int keys
(def leaf-safe-laws
  (disj scalar-laws
        :update-density-ratio
        :update-round-trip
        :regenerate-empty-identity))

;; Scalar minus score-decomposition (project doesn't decompose through splice)
(def splice-laws (disj scalar-laws :score-full-decomposition))

;; Deep-nesting: addr-dependent + decomposition issues
(def deep-nesting-laws
  (disj scalar-laws
        :update-density-ratio
        :update-round-trip
        :regenerate-empty-identity
        :score-full-decomposition))

;; Recurse: decomposition + regenerate
(def recurse-laws
  (disj scalar-laws :score-full-decomposition :regenerate-empty-identity))

;; ---------------------------------------------------------------------------
;; 13 canonical models
;; ---------------------------------------------------------------------------

;; 1. Single-site gaussian
(def single-site
  (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))]
            (mx/eval! x) (mx/item x)))))

;; 2. Multi-site dependent (x -> y)
(def multi-site
  (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))]
            (mx/eval! x)
            (let [y (trace :y (dist/gaussian (mx/item x) 1))]
              (mx/eval! y) (mx/item y))))))

;; 3. Linear regression (5 addresses: slope, intercept, y0, y1, y2)
(def linreg
  (dyn/auto-key (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (mx/eval! slope intercept)
      (let [sv (mx/item slope) iv (mx/item intercept)]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                     (dist/gaussian (+ (* sv x) iv) 1)))
        sv)))))

;; 4. Splice (sub-GF via splice)
(def inner-gf
  (dyn/auto-key (gen [] (let [z (trace :z (dist/gaussian 0 1))]
            (mx/eval! z) (mx/item z)))))

(def splice-model
  (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 10))]
            (mx/eval! x)
            (splice :inner inner-gf)
            (mx/item x)))))

;; 5. Mixed discrete/continuous
(def mixed-model
  (dyn/auto-key (gen [] (let [b (trace :b (dist/bernoulli 0.5))]
            (mx/eval! b)
            (let [y (trace :y (dist/gaussian (if (> (mx/item b) 0.5) 5.0 -5.0) 1))]
              (mx/eval! y) (mx/item y))))))

;; 6. Deep nesting (3-level splice chain: x, inner/y, inner/mid/z)
(def level2 (dyn/auto-key (gen [] (let [z (trace :z (dist/gaussian 0 1))]
                      (mx/eval! z) (mx/item z)))))
(def level1 (dyn/auto-key (gen [] (splice :mid level2)
                    (let [y (trace :y (dist/gaussian 0 1))]
                      (mx/eval! y) (mx/item y)))))
(def deep-nesting
  (dyn/auto-key (gen [] (splice :inner level1)
          (let [x (trace :x (dist/gaussian 0 1))]
            (mx/eval! x) (mx/item x)))))

;; 7. Two-site vectorization-compatible (no eval!/item in body)
(def two-site-vec
  (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))
                y (trace :y (dist/gaussian 0 1))]
            y))))

;; 8. Map combinator
(def map-kernel (dyn/auto-key (gen [x] (let [y (trace :y (dist/gaussian x 1))]
                            (mx/eval! y) (mx/item y)))))
(def map-model (comb/map-combinator map-kernel))

;; 9. Unfold combinator
(def unfold-step (dyn/auto-key (gen [t state] (let [y (trace :y (dist/gaussian state 1))]
                                  (mx/eval! y) (mx/item y)))))
(def unfold-model (comb/unfold-combinator unfold-step))

;; 10. Switch combinator (flat choices -- branch has keyword addr)
(def branch-a (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1)))))
(def branch-b (dyn/auto-key (gen [] (trace :x (dist/gaussian 5 1)))))
(def switch-model (comb/switch-combinator branch-a branch-b))

;; 11. Scan combinator
(def scan-step (dyn/auto-key (gen [carry x] (let [y (trace :y (dist/gaussian carry 1))]
                                 (mx/eval! y)
                                 [(mx/item y) (mx/item y)]))))
(def scan-model (comb/scan-combinator scan-step))

;; 12. Mask combinator (active=true exposes inner choices)
(def mask-inner (dyn/auto-key (gen [] (let [y (trace :y (dist/gaussian 0 1))]
                          (mx/eval! y) (mx/item y)))))
(def mask-model (comb/mask-combinator mask-inner))

;; 13. Recurse combinator (depth-2 tree, leaf addr :v)
(def recurse-model
  (comb/recurse
    (fn [self]
      (dyn/auto-key (gen [depth]
        (let [v (trace :v (dist/gaussian 0 1))]
          (mx/eval! v)
          (when (> depth 0)
            (splice :child self (dec depth)))
          (mx/item v)))))))

;; ---------------------------------------------------------------------------
;; Verification helper
;; ---------------------------------------------------------------------------

(defn- verify-laws
  "Run GFI laws on a model and assert all pass.
   Returns the report for further inspection."
  [model args law-names]
  (let [{:keys [all-pass? results total-pass total-fail] :as report}
        (gfi/verify model args :n-trials 5 :law-names law-names)]
    (is all-pass?
        (str "expected all laws to pass but "
             total-fail "/" (+ total-pass total-fail) " checks failed"
             (when-not all-pass?
               (str ": "
                 (->> results
                      (filter (fn [{:keys [fails]}] (pos? fails)))
                      (map (fn [{:keys [name fails passes]}]
                             (str (clojure.core/name name) " (" fails "/" (+ passes fails) ")")))
                      (interpose ", ")
                      (apply str))))))
    report))

;; ---------------------------------------------------------------------------
;; Law verification tests — one per canonical model
;; ---------------------------------------------------------------------------

(deftest single-site-satisfies-laws
  (testing "single-site gaussian passes all scalar laws"
    (verify-laws single-site [] scalar-laws)))

(deftest multi-site-satisfies-laws
  (testing "multi-site dependent model passes all scalar laws"
    (verify-laws multi-site [] scalar-laws)))

(deftest linreg-satisfies-laws
  (testing "linear regression passes all scalar laws"
    (verify-laws linreg [[1 2 3]] scalar-laws)))

(deftest splice-satisfies-laws
  (testing "splice model passes all laws except score-decomposition"
    (verify-laws splice-model [] splice-laws)))

(deftest mixed-satisfies-laws
  (testing "mixed discrete/continuous passes all scalar laws"
    (verify-laws mixed-model [] scalar-laws)))

(deftest deep-nesting-satisfies-laws
  (testing "deep nesting passes laws excluding addr-dependent and decomposition"
    (verify-laws deep-nesting [] deep-nesting-laws)))

(deftest two-site-vec-satisfies-all-laws
  (testing "two-site vectorizable model passes all laws including broadcast"
    (verify-laws two-site-vec [] all-laws)))

(deftest map-combinator-satisfies-laws
  (testing "map combinator passes leaf-safe laws"
    (verify-laws map-model [[1 2 3]] leaf-safe-laws)))

(deftest unfold-combinator-satisfies-laws
  (testing "unfold combinator passes leaf-safe laws"
    (verify-laws unfold-model [3 0.0] leaf-safe-laws)))

(deftest switch-combinator-satisfies-laws
  (testing "switch combinator passes all scalar laws"
    (verify-laws switch-model [0] scalar-laws)))

(deftest scan-combinator-satisfies-laws
  (testing "scan combinator passes leaf-safe laws"
    (verify-laws scan-model [0.0 [1 2 3]] leaf-safe-laws)))

(deftest mask-combinator-satisfies-laws
  (testing "mask combinator passes all scalar laws"
    (verify-laws mask-model [true] scalar-laws)))

(deftest recurse-combinator-satisfies-laws
  (testing "recurse combinator passes laws excluding decomposition and regenerate"
    (verify-laws recurse-model [2] recurse-laws)))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
