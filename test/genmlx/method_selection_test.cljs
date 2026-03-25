(ns genmlx.method-selection-test
  "Automatic method selection tests.
   Tests select-method decision tree and tune-method-opts across
   8 model categories + edge cases."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.method-selection :as ms])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

;; 1. All-conjugate: Normal-Normal prior + observed -> :exact
(def m-conjugate
  (gen [x]
       (let [mu (trace :mu (dist/gaussian 0 10))]
         (trace :y (dist/gaussian mu 1)))))

;; 2. Non-conjugate, small static (3 latents) -> :hmc
(def m-static-small
  (gen []
       (let [a (trace :a (dist/gaussian 0 1))
             b (trace :b (dist/gaussian 0 1))
             c (trace :c (dist/gaussian 0 1))]
         (mx/add a b))))

;; 3. Non-conjugate, large static (11+ latents) -> :vi
(def m-static-large
  (gen []
       (let [x1 (trace :x1 (dist/gaussian 0 1))
             x2 (trace :x2 (dist/gaussian 0 1))
             x3 (trace :x3 (dist/gaussian 0 1))
             x4 (trace :x4 (dist/gaussian 0 1))
             x5 (trace :x5 (dist/gaussian 0 1))
             x6 (trace :x6 (dist/gaussian 0 1))
             x7 (trace :x7 (dist/gaussian 0 1))
             x8 (trace :x8 (dist/gaussian 0 1))
             x9 (trace :x9 (dist/gaussian 0 1))
             x10 (trace :x10 (dist/gaussian 0 1))
             x11 (trace :x11 (dist/gaussian 0 1))]
         x1)))

;; 4. Mixed conjugate: 2 of 5 sites conjugate, obs on y1/y2 -> :hmc with residual :p
(def m-mixed
  (gen [xs]
       (let [mu1 (trace :mu1 (dist/gaussian 0 10))
             mu2 (trace :mu2 (dist/gaussian 0 10))
             p (trace :p (dist/uniform 0 1))
             y1 (trace :y1 (dist/gaussian mu1 1))
             y2 (trace :y2 (dist/gaussian mu2 1))]
         (mx/add mu1 mu2))))

;; 5. Temporal model with Unfold
(def unfold-kernel
  (gen [t state]
       (let [x (trace :x (dist/gaussian state 1))]
         x)))

(def m-unfold
  (gen [T]
       (let [init (trace :init (dist/gaussian 0 10))
             unfold-gf (comb/unfold-combinator unfold-kernel)]
         (splice :steps unfold-gf T init))))

;; 6. Temporal model with Scan (name contains 'scan')
(def scan-kernel
  (gen [carry input]
       (let [x (trace :x (dist/gaussian carry 1))]
         x)))

(def m-scan
  (gen [inputs]
       (let [init (trace :init (dist/gaussian 0 10))
             scan-gf (comb/scan-combinator scan-kernel)]
         (splice :steps scan-gf init inputs))))

;; 7. Dynamic addresses
(def m-dynamic
  (gen [n]
       (let [k (trace :k (dist/poisson 5))]
         (dotimes [i 3]
           (trace (keyword (str "x" i)) (dist/gaussian 0 1))))))

;; 8. Empty model (no trace sites)
(def m-empty (gen [] 42))

;; 9. All-observed model
(def m-all-obs
  (gen []
       (let [x (trace :x (dist/gaussian 0 1))
             y (trace :y (dist/gaussian 0 1))]
         (mx/add x y))))

;; 10. Splice model without temporal name
(def sub-model
  (gen [x]
       (let [z (trace :z (dist/gaussian x 1))]
         z)))

(def m-splice-generic
  (gen []
       (let [a (trace :a (dist/gaussian 0 1))
             helper sub-model]
         (splice :sub helper a))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest category-1-conjugate-exact
  (testing "All-conjugate model -> :exact"
    (let [obs (cm/choicemap :y 1.0)
          result (ms/select-method m-conjugate obs)]
      (is (= :exact (:method result)) "conjugate model -> :exact")
      (is (contains? (:eliminated result) :mu) "conjugate: :mu eliminated")
      (is (= 0 (:n-residual result)) "conjugate: 0 residual")
      (is (some? (:reason result)) "conjugate: reason mentions analytical")
      (is (map? (:opts result)) "conjugate: opts is map"))))

(deftest category-2-static-small-hmc
  (testing "Static small -> :hmc"
    (let [result (ms/select-method m-static-small nil)]
      (is (= :hmc (:method result)) "static small -> :hmc")
      (is (= 3 (:n-residual result)) "static small: 3 residual")
      (is (some? (:reason result)) "static small: reason mentions static")
      (is (contains? (:opts result) :n-samples) "static small: opts has :n-samples")
      (is (contains? (:opts result) :n-leapfrog) "static small: opts has :n-leapfrog"))))

(deftest category-3-static-large-vi
  (testing "Static large -> :vi"
    (let [result (ms/select-method m-static-large nil)]
      (is (= :vi (:method result)) "static large -> :vi")
      (is (= 11 (:n-residual result)) "static large: 11 residual")
      (is (contains? (:opts result) :n-iters) "static large: opts has :n-iters")
      (is (contains? (:opts result) :learning-rate) "static large: opts has :learning-rate"))))

(deftest category-4-mixed-conjugate-hmc
  (testing "Mixed conjugate -> :hmc with reduced residual"
    (let [obs (cm/choicemap :y1 1.0 :y2 2.0)
          result (ms/select-method m-mixed obs)]
      (is (= :hmc (:method result)) "mixed conjugate -> :hmc")
      (is (contains? (:eliminated result) :mu1) "mixed: :mu1 eliminated")
      (is (contains? (:eliminated result) :mu2) "mixed: :mu2 eliminated")
      (is (contains? (:residual-addrs result) :p) "mixed: :p is residual")
      (is (= 1 (:n-residual result)) "mixed: 1 residual"))))

(deftest category-5-temporal-unfold-smc
  (testing "Temporal (Unfold) -> :smc"
    (let [result (ms/select-method m-unfold nil)]
      (is (= :smc (:method result)) "unfold -> :smc")
      (is (clojure.string/includes? (:reason result) "emporal") "unfold: reason mentions temporal")
      (is (contains? (:opts result) :n-particles) "unfold: opts has :n-particles"))))

(deftest category-6-temporal-scan-smc
  (testing "Temporal (Scan) -> :smc"
    (let [result (ms/select-method m-scan nil)]
      (is (= :smc (:method result)) "scan -> :smc")
      (is (clojure.string/includes? (:reason result) "emporal") "scan: reason mentions temporal"))))

(deftest category-7-dynamic-handler-is
  (testing "Dynamic addresses -> :handler-is"
    (let [result (ms/select-method m-dynamic nil)]
      (is (= :handler-is (:method result)) "dynamic -> :handler-is")
      (is (clojure.string/includes? (:reason result) "ynamic") "dynamic: reason mentions dynamic")
      (is (contains? (:opts result) :n-particles) "dynamic: opts has :n-particles"))))

(deftest category-8-generic-splice-smc
  (testing "Generic splice (non-temporal name) -> :smc"
    (let [result (ms/select-method m-splice-generic nil)]
      (is (= :smc (:method result)) "generic splice -> :smc")
      (is (clojure.string/includes? (:reason result) "plice") "generic splice: reason mentions splice"))))

(deftest edge-cases
  (testing "Empty model -> :exact"
    (let [result (ms/select-method m-empty nil)]
      (is (= :exact (:method result)) "empty model -> :exact")
      (is (= 0 (:n-residual result)) "empty: 0 residual")
      (is (= 0 (:n-latent result)) "empty: 0 latent")))

  (testing "All observed -> :exact"
    (let [obs (cm/choicemap :x 1.0 :y 2.0)
          result (ms/select-method m-all-obs obs)]
      (is (= :exact (:method result)) "all observed -> :exact")
      (is (= 0 (:n-residual result)) "all observed: 0 residual")
      (is (= 0 (:n-latent result)) "all observed: 0 latent")))

  (testing "Conjugate without observations -> :hmc"
    (let [result (ms/select-method m-conjugate nil)]
      (is (= :hmc (:method result)) "conjugate no obs -> :hmc")
      (is (contains? (:residual-addrs result) :y) "conjugate no obs: :y is residual")))

  (testing "nil analytical-plan is safe"
    (let [result (ms/select-method m-static-small nil)]
      (is (some? (:method result)) "nil analytical-plan: safe")
      (is (= #{} (:eliminated result)) "nil plan: eliminated is empty set"))))

(deftest return-structure-validation
  (testing "Return structure has all required keys"
    (let [result (ms/select-method m-static-small nil)]
      (is (contains? result :method) "has :method")
      (is (contains? result :reason) "has :reason")
      (is (contains? result :opts) "has :opts")
      (is (contains? result :eliminated) "has :eliminated")
      (is (contains? result :residual-addrs) "has :residual-addrs")
      (is (contains? result :n-residual) "has :n-residual")
      (is (contains? result :n-latent) "has :n-latent")
      (is (keyword? (:method result)) ":method is keyword")
      (is (string? (:reason result)) ":reason is string")
      (is (map? (:opts result)) ":opts is map")
      (is (set? (:eliminated result)) ":eliminated is set")
      (is (set? (:residual-addrs result)) ":residual-addrs is set")
      (is (number? (:n-residual result)) ":n-residual is number")
      (is (number? (:n-latent result)) ":n-latent is number"))))

(deftest tune-method-opts-hmc
  (testing "HMC tuning (small residual)"
    (let [sel (ms/select-method m-static-small nil)
          tuned (ms/tune-method-opts sel)]
      (is (= 10 (:n-leapfrog tuned)) "hmc tune: n-leapfrog for 3 residual")
      (is (= 0.05 (:step-size tuned)) "hmc tune: step-size for 3 residual")
      (is (= 200 (:n-warmup tuned)) "hmc tune: n-warmup for 3 residual"))))

(deftest tune-method-opts-vi
  (testing "VI tuning (large residual)"
    (let [sel (ms/select-method m-static-large nil)
          tuned (ms/tune-method-opts sel)]
      (is (= 2000 (:n-iters tuned)) "vi tune: n-iters for 11 residual")
      (is (= 10 (:n-samples tuned)) "vi tune: n-samples for 11 residual"))))

(deftest tune-method-opts-smc
  (testing "SMC tuning"
    (let [sel (ms/select-method m-unfold nil)
          tuned (ms/tune-method-opts sel)]
      (is (contains? tuned :n-particles) "smc tune: has :n-particles")
      (is (contains? tuned :ess-threshold) "smc tune: has :ess-threshold"))))

(deftest tune-method-opts-handler-is
  (testing "handler-is tuning"
    (let [sel (ms/select-method m-dynamic nil)
          tuned (ms/tune-method-opts sel)]
      (is (contains? tuned :n-particles) "handler-is tune: has :n-particles"))))

(deftest tune-method-opts-exact
  (testing "exact tuning"
    (let [obs (cm/choicemap :y 1.0)
          sel (ms/select-method m-conjugate obs)
          tuned (ms/tune-method-opts sel)]
      (is (= {} tuned) "exact tune: empty opts"))))

(deftest tune-method-opts-user-overrides
  (testing "User overrides take priority"
    (let [sel (ms/select-method m-static-small nil)
          tuned (ms/tune-method-opts sel {:step-size 0.001 :custom-key 42})]
      (is (= 0.001 (:step-size tuned)) "user override: step-size")
      (is (= 42 (:custom-key tuned)) "user override: custom-key")
      (is (contains? tuned :n-leapfrog) "user override: preserves n-leapfrog"))))

(deftest tune-method-opts-1-arity
  (testing "1-arity version works"
    (let [sel (ms/select-method m-static-small nil)
          tuned (ms/tune-method-opts sel)]
      (is (map? tuned) "1-arity: returns map"))))

(cljs.test/run-tests)
