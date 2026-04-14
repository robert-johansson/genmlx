(ns genmlx.non-gaussian-contract-test
  "GFI laws on non-Gaussian canonical models.
   Extends contract_verification_test.cljs to distributions beyond Gaussian:
   Beta-Bernoulli, Gamma-Poisson, Gamma-Exponential, Uniform-Categorical,
   Map-with-Beta, Unfold-with-Poisson, Switch-with-mixed."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gfi :as gfi]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Law name sets
;; ---------------------------------------------------------------------------

(def all-keys
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

;; All minus vsimulate-shape-correctness (vsimulate only works on DynamicGF)
(def scalar-keys (disj all-keys :vsimulate-shape-correctness))

;; For models with constrained-domain distributions (beta, exponential, etc.),
;; update-density-ratio and update-round-trip use hardcoded values (0.0, 42.0)
;; that may fall outside the support. Exclude those.
(def constrained-domain-keys
  (disj scalar-keys
        :update-density-ratio
        :update-round-trip))

;; For combinator models: also exclude addr-dependent laws
;; (integer-keyed submaps; first-address returns a submap, not a leaf)
(def combinator-constrained-keys
  (disj constrained-domain-keys
        :regenerate-empty-identity))

;; For models with score-decomposition issues through splice/combinator boundaries
(def combinator-no-decomp-keys
  (disj combinator-constrained-keys
        :score-full-decomposition))

;; ---------------------------------------------------------------------------
;; 7 non-Gaussian canonical models
;; ---------------------------------------------------------------------------

;; 1. Beta-Bernoulli: conjugate prior pattern
(def beta-bernoulli
  (dyn/auto-key
    (gen []
      (let [p (trace :p (dist/beta-dist 2 5))]
        (mx/eval! p)
        (let [x (trace :x (dist/bernoulli (mx/item p)))]
          (mx/eval! x)
          (mx/item x))))))

;; 2. Gamma-Poisson: rate prior with count data
(def gamma-poisson
  (dyn/auto-key
    (gen []
      (let [rate (trace :rate (dist/gamma-dist 2 1))]
        (mx/eval! rate)
        (let [x (trace :x (dist/poisson (mx/item rate)))]
          (mx/eval! x)
          (mx/item x))))))

;; 3. Gamma-Exponential: rate prior with continuous positive data
(def gamma-exponential
  (dyn/auto-key
    (gen []
      (let [rate (trace :rate (dist/gamma-dist 3 2))]
        (mx/eval! rate)
        (let [obs (trace :obs (dist/exponential (mx/item rate)))]
          (mx/eval! obs)
          (mx/item obs))))))

;; 4. Uniform-Categorical: logit from gaussian, categorical observation
(def uniform-categorical
  (dyn/auto-key
    (gen []
      (let [logit (trace :logit (dist/gaussian 0 1))]
        (mx/eval! logit)
        (let [x (trace :x (dist/categorical (mx/array #js [(float 0) (mx/item logit)])))]
          (mx/eval! x)
          (mx/item x))))))

;; 5. Map combinator with Beta distribution
(def map-beta-kernel
  (dyn/auto-key
    (gen [alpha]
      (let [x (trace :x (dist/beta-dist alpha 2))]
        (mx/eval! x)
        (mx/item x)))))
(def map-beta-model (comb/map-combinator map-beta-kernel))

;; 6. Unfold with Poisson observations
(def unfold-poisson-step
  (dyn/auto-key
    (gen [t state]
      (let [x (trace :x (dist/poisson (+ (js/Math.abs state) 1)))]
        (mx/eval! x)
        (+ state (mx/item x))))))
(def unfold-poisson-model (comb/unfold-combinator unfold-poisson-step))

;; 7. Switch with mixed distributions: branch0 = beta, branch1 = exponential
(def switch-branch-beta
  (dyn/auto-key (gen [] (trace :x (dist/beta-dist 2 3)))))
(def switch-branch-exp
  (dyn/auto-key (gen [] (trace :x (dist/exponential 1)))))
(def switch-mixed-model
  (comb/switch-combinator switch-branch-beta switch-branch-exp))

;; ---------------------------------------------------------------------------
;; Verification helper
;; ---------------------------------------------------------------------------

(defn- verify-contracts
  "Run GFI laws on a model and assert all pass.
   Returns the report for further inspection."
  [model args law-names]
  (let [{:keys [all-pass? results total-pass total-fail] :as report}
        (gfi/verify model args
          :n-trials 5 :law-names law-names)]
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
;; Law verification tests -- one per non-Gaussian model
;; ---------------------------------------------------------------------------

(deftest beta-bernoulli-satisfies-contracts
  (testing "beta-bernoulli passes constrained-domain laws"
    (verify-contracts beta-bernoulli [] constrained-domain-keys)))

(deftest gamma-poisson-satisfies-contracts
  (testing "gamma-poisson passes constrained-domain laws"
    (verify-contracts gamma-poisson [] constrained-domain-keys)))

(deftest gamma-exponential-satisfies-contracts
  (testing "gamma-exponential passes constrained-domain laws"
    (verify-contracts gamma-exponential [] constrained-domain-keys)))

(deftest uniform-categorical-satisfies-contracts
  (testing "uniform-categorical passes constrained-domain laws"
    (verify-contracts uniform-categorical [] constrained-domain-keys)))

(deftest map-beta-satisfies-contracts
  (testing "map combinator with beta passes combinator-constrained laws"
    (verify-contracts map-beta-model [[3 4 5]] combinator-constrained-keys)))

(deftest unfold-poisson-satisfies-contracts
  (testing "unfold with poisson passes combinator-constrained laws"
    (verify-contracts unfold-poisson-model [3 0.0] combinator-constrained-keys)))

(deftest switch-mixed-satisfies-contracts
  (testing "switch with mixed beta/exponential passes constrained-domain laws"
    (verify-contracts switch-mixed-model [0] constrained-domain-keys)))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
