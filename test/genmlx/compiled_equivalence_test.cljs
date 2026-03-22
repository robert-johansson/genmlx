(ns genmlx.compiled-equivalence-test
  "Compiled path equivalence: compiled score = handler score for static models.
   Same PRNG key → same choices and scores."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled :as compiled]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip compiled paths, forcing handler execution."
  [gf]
  (dyn/->DynamicGF (:body-fn gf) (:source gf)
                   (dissoc (:schema gf)
                           :compiled-simulate
                           :compiled-generate
                           :compiled-update
                           :compiled-assess
                           :compiled-project
                           :compiled-regenerate)))

(defn compiled?
  "True if model has a compiled simulate path."
  [gf]
  (some? (:compiled-simulate (:schema gf))))

;; ---------------------------------------------------------------------------
;; Test models (all static → should compile)
;; ---------------------------------------------------------------------------

(def single-site (gen [] (trace :x (dist/gaussian 0 1))))

(def multi-site
  (gen []
       (trace :a (dist/gaussian 0 10))
       (trace :b (dist/gaussian 5 2))))

(def dependent-model
  (gen [x]
       (let [slope (trace :slope (dist/gaussian 0 10))
             intercept (trace :intercept (dist/gaussian 0 5))]
         (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))
         slope)))

(def intermediate-let-model
  (gen [x]
       (let [slope (trace :slope (dist/gaussian 0 10))
             predicted (mx/multiply slope x)]
         (trace :y (dist/gaussian predicted 1))
         predicted)))

(def mixed-dist-model
  (gen []
       (let [p (trace :p (dist/uniform 0 1))]
         (trace :coin (dist/bernoulli p))
         p)))

;; ---------------------------------------------------------------------------
;; Compilation gate: static models compile
;; ---------------------------------------------------------------------------

(deftest static-models-have-compiled-simulate
  (is (compiled? single-site) "single-site compiles")
  (is (compiled? multi-site) "multi-site compiles")
  (is (compiled? dependent-model) "dependent compiles")
  (is (compiled? intermediate-let-model) "intermediate-let compiles")
  (is (compiled? mixed-dist-model) "mixed-dist compiles"))

;; ---------------------------------------------------------------------------
;; Score equivalence: compiled vs handler (same key → same result)
;; ---------------------------------------------------------------------------

(defn assert-equivalence
  "Verify compiled and handler paths produce identical traces."
  [model args seed]
  (let [k1 (rng/fresh-key seed)
        compiled-tr (p/simulate (dyn/with-key model k1) args)
        k2 (rng/fresh-key seed)
        handler-tr (p/simulate (dyn/with-key (force-handler model) k2) args)
        compiled-score (h/realize (:score compiled-tr))
        handler-score (h/realize (:score handler-tr))]
    {:score-match? (h/close? compiled-score handler-score 1e-5)
     :compiled-score compiled-score
     :handler-score handler-score
     :compiled-choices (:choices compiled-tr)
     :handler-choices (:choices handler-tr)}))

(deftest single-site-compiled-equals-handler
  (let [{:keys [score-match?]} (assert-equivalence single-site [] 77)]
    (is score-match? "compiled score = handler score")))

(deftest multi-site-compiled-equals-handler
  (let [{:keys [score-match?]} (assert-equivalence multi-site [] 77)]
    (is score-match? "compiled score = handler score")))

(deftest dependent-model-compiled-equals-handler
  (let [{:keys [score-match?]} (assert-equivalence dependent-model [(mx/scalar 2.5)] 77)]
    (is score-match? "compiled score = handler score")))

(deftest intermediate-let-compiled-equals-handler
  (let [{:keys [score-match?]} (assert-equivalence intermediate-let-model [(mx/scalar 3.0)] 77)]
    (is score-match? "compiled score = handler score")))

(deftest mixed-dist-compiled-equals-handler
  (let [{:keys [score-match?]} (assert-equivalence mixed-dist-model [] 77)]
    (is score-match? "compiled score = handler score")))

;; ---------------------------------------------------------------------------
;; Choice equivalence (same key → same values)
;; ---------------------------------------------------------------------------

(deftest compiled-choices-have-correct-structure
  (testing "compiled path produces same address set as handler"
    (let [k1 (rng/fresh-key 42)
          compiled-tr (p/simulate (dyn/with-key dependent-model k1) [(mx/scalar 2.0)])
          k2 (rng/fresh-key 42)
          handler-tr (p/simulate (dyn/with-key (force-handler dependent-model) k2) [(mx/scalar 2.0)])
          compiled-addrs (set (cm/addresses (:choices compiled-tr)))
          handler-addrs (set (cm/addresses (:choices handler-tr)))]
      (is (= compiled-addrs handler-addrs)
          "compiled and handler have same address set"))))

(deftest compiled-score-is-consistent-with-choices
  (testing "compiled score matches analytical log-prob of sampled values"
    (let [k (rng/fresh-key 42)
          tr (p/simulate (dyn/with-key single-site k) [])
          x-val (h/realize (cm/get-value (cm/get-submap (:choices tr) :x)))
          score (h/realize (:score tr))
          expected (h/gaussian-lp x-val 0 1)]
      (is (h/close? expected score 1e-4)
          "score = log N(x; 0, 1) for compiled path"))))

;; ---------------------------------------------------------------------------
;; Generate weight equivalence
;; ---------------------------------------------------------------------------

(deftest compiled-generate-weight-equals-handler
  (testing "generate weight: compiled = handler"
    (let [constraints (cm/choicemap :x (mx/scalar 0.5))
          k1 (rng/fresh-key 88)
          {:keys [trace weight]} (p/generate (dyn/with-key single-site k1) [] constraints)
          k2 (rng/fresh-key 88)
          h-result (p/generate (dyn/with-key (force-handler single-site) k2) [] constraints)
          compiled-weight (h/realize weight)
          handler-weight (h/realize (:weight h-result))]
      (is (h/close? compiled-weight handler-weight 1e-5)
          "generate weight: compiled = handler"))))

(deftest compiled-generate-weight-dependent-model
  (testing "dependent model generate weight: compiled = handler"
    (let [constraints (-> cm/EMPTY
                          (cm/set-choice [:slope] (mx/scalar 1.0))
                          (cm/set-choice [:intercept] (mx/scalar 0.0))
                          (cm/set-choice [:y] (mx/scalar 2.5)))
          k1 (rng/fresh-key 88)
          {:keys [weight]} (p/generate (dyn/with-key dependent-model k1)
                                       [(mx/scalar 2.0)] constraints)
          k2 (rng/fresh-key 88)
          h-result (p/generate (dyn/with-key (force-handler dependent-model) k2)
                               [(mx/scalar 2.0)] constraints)]
      (is (h/close? (h/realize weight) (h/realize (:weight h-result)) 1e-5)
          "dependent generate weight: compiled = handler"))))

;; ---------------------------------------------------------------------------
;; Non-static models fall back to handler
;; ---------------------------------------------------------------------------

(deftest non-static-model-not-compiled
  (testing "model with splice does not compile"
    (let [sub (gen [] (trace :a (dist/gaussian 0 1)))
          m (gen [] (splice :sub sub))]
      (is (not (compiled? m))
          "splice model has no compiled-simulate"))))

(deftest branching-model-has-branch-rewrite-or-no-compiled
  (testing "branching model: may have branch-rewritten or no compiled"
    (let [m (gen [flag]
                 (if flag
                   (trace :a (dist/gaussian 0 1))
                   (trace :b (dist/gaussian 0 1))))]
      ;; Branch models may or may not have compiled-simulate
      ;; (depends on whether branch rewriting succeeded)
      ;; But they should NOT be classified as static
      (is (not (:static? (:schema m)))
          "branching model is not static"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
