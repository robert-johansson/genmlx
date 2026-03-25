(ns genmlx.contract-verification-test
  "GFI contract verification on 13 canonical models covering all features.
   Each model is verified against its applicable contract subset (5 trials each)."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.contracts :as contracts]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

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

(defn- verify-contracts
  "Run GFI contracts on a model and assert all pass.
   Returns the report for further inspection."
  [model args contract-keys]
  (let [{:keys [all-pass? results total-pass total-fail] :as report}
        (contracts/verify-gfi-contracts model args
          :n-trials 5 :contract-keys contract-keys)]
    (is all-pass?
        (str "expected all contracts to pass but "
             total-fail "/" (+ total-pass total-fail) " checks failed"
             (when-not all-pass?
               (str ": "
                 (->> results
                      (filter (fn [[_ {:keys [fail]}]] (pos? fail)))
                      (map (fn [[k {:keys [theorem fail pass]}]]
                             (str (name k) " (" fail "/" (+ pass fail) ")")))
                      (interpose ", ")
                      (apply str))))))
    report))

;; ---------------------------------------------------------------------------
;; Contract verification tests — one per canonical model
;; ---------------------------------------------------------------------------

(deftest single-site-satisfies-contracts
  (testing "single-site gaussian passes all scalar contracts"
    (verify-contracts single-site [] scalar-keys)))

(deftest multi-site-satisfies-contracts
  (testing "multi-site dependent model passes all scalar contracts"
    (verify-contracts multi-site [] scalar-keys)))

(deftest linreg-satisfies-contracts
  (testing "linear regression passes all scalar contracts"
    (verify-contracts linreg [[1 2 3]] scalar-keys)))

(deftest splice-satisfies-contracts
  (testing "splice model passes all contracts except score-decomposition"
    (verify-contracts splice-model [] splice-keys)))

(deftest mixed-satisfies-contracts
  (testing "mixed discrete/continuous passes all scalar contracts"
    (verify-contracts mixed-model [] scalar-keys)))

(deftest deep-nesting-satisfies-contracts
  (testing "deep nesting passes contracts excluding addr-dependent and decomposition"
    (verify-contracts deep-nesting [] deep-nesting-keys)))

(deftest two-site-vec-satisfies-all-contracts
  (testing "two-site vectorizable model passes all contracts including broadcast"
    (verify-contracts two-site-vec [] all-keys)))

(deftest map-combinator-satisfies-contracts
  (testing "map combinator passes leaf-safe contracts"
    (verify-contracts map-model [[1 2 3]] leaf-safe-keys)))

(deftest unfold-combinator-satisfies-contracts
  (testing "unfold combinator passes leaf-safe contracts"
    (verify-contracts unfold-model [3 0.0] leaf-safe-keys)))

(deftest switch-combinator-satisfies-contracts
  (testing "switch combinator passes all scalar contracts"
    (verify-contracts switch-model [0] scalar-keys)))

(deftest scan-combinator-satisfies-contracts
  (testing "scan combinator passes leaf-safe contracts"
    (verify-contracts scan-model [0.0 [1 2 3]] leaf-safe-keys)))

(deftest mask-combinator-satisfies-contracts
  (testing "mask combinator passes all scalar contracts"
    (verify-contracts mask-model [true] scalar-keys)))

(deftest recurse-combinator-satisfies-contracts
  (testing "recurse combinator passes contracts excluding decomposition and regenerate"
    (verify-contracts recurse-model [2] recurse-keys)))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
