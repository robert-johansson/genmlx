(ns genmlx.compiled-generate-test
  "WP-1 tests: compiled generate for static DynamicGF models.
   Validates that compiled generate matches handler generate exactly."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.schema :as schema])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-prefix :compiled-prefix-addrs)]
    (assoc gf :schema schema)))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-model
  (dyn/auto-key
    (gen [mu]
      (let [x (trace :x (dist/gaussian mu (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 2.0)))]
        y))))

(def chain-model
  (dyn/auto-key
    (gen [a]
      (let [x (trace :x (dist/gaussian a (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 0.5)))
            z (trace :z (dist/gaussian (mx/add x y) (mx/scalar 1.0)))]
        z))))

(def multi-dist-model
  (dyn/auto-key
    (gen []
      (let [g (trace :g (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            u (trace :u (dist/uniform (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/bernoulli (mx/scalar 0.5)))
            e (trace :e (dist/exponential (mx/scalar 2.0)))]
        g))))

(def retval-model
  (dyn/auto-key
    (gen [s]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) s))
            y (trace :y (dist/gaussian (mx/scalar 0.0) s))]
        (mx/add x y)))))

(def delta-model
  (dyn/auto-key
    (gen [v]
      (let [x (trace :x (dist/delta v))
            y (trace :y (dist/gaussian x (mx/scalar 1.0)))]
        y))))

(def exotic-model
  (dyn/auto-key
    (gen []
      (let [a (trace :a (dist/laplace (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/cauchy (mx/scalar 0.0) (mx/scalar 1.0)))
            c (trace :c (dist/log-normal (mx/scalar 0.0) (mx/scalar 0.5)))]
        a))))

(def dynamic-addr-model
  (dyn/auto-key
    (gen [n]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
        (doseq [i (range (mx/item n))]
          (trace (keyword (str "y" i)) (dist/gaussian x (mx/scalar 1.0))))
        x))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest compilation-presence-test
  (testing "compilation presence"
    (is (some? (:compiled-generate (:schema simple-model))) "simple-model has :compiled-generate")
    (is (some? (:compiled-generate (:schema chain-model))) "chain-model has :compiled-generate")
    (is (some? (:compiled-generate (:schema multi-dist-model))) "multi-dist-model has :compiled-generate")
    (is (some? (:compiled-generate (:schema retval-model))) "retval-model has :compiled-generate")
    (is (some? (:compiled-generate (:schema delta-model))) "delta-model has :compiled-generate")
    (is (some? (:compiled-generate (:schema exotic-model))) "exotic-model has :compiled-generate")
    (is (nil? (:compiled-generate (:schema dynamic-addr-model))) "dynamic-addr-model does NOT have :compiled-generate")))

(deftest all-sites-constrained-test
  (testing "all sites constrained"
    (let [key (rng/fresh-key 42)
          gf (dyn/with-key simple-model key)
          constraints (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 5.0))
          {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
          gf-h (dyn/with-key (force-handler simple-model) key)
          {:keys [trace weight] :as ref} (p/generate gf-h [(mx/scalar 0.0)] constraints)
          ref-trace trace
          ref-weight weight
          gf2 (dyn/with-key simple-model key)
          {:keys [trace weight]} (p/generate gf2 [(mx/scalar 0.0)] constraints)]
      (is (h/close? (mx/item (:score ref-trace)) (mx/item (:score trace)) 1e-5) "all-constrained: score matches handler")
      (is (h/close? (mx/item ref-weight) (mx/item weight) 1e-5) "all-constrained: weight matches handler")
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 1e-5) "all-constrained: weight = score (all constrained)")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6) "all-constrained: x = constraint")
      (is (h/close? 5.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y))) 1e-6) "all-constrained: y = constraint"))))

(deftest no-sites-constrained-test
  (testing "no sites constrained (= simulate)"
    (let [key (rng/fresh-key 77)
          gf (dyn/with-key simple-model key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 1.0)] cm/EMPTY)
          gf-h (dyn/with-key (force-handler simple-model) key)
          ref (p/generate gf-h [(mx/scalar 1.0)] cm/EMPTY)]
      (is (h/close? 0.0 (mx/item weight) 1e-6) "no-constraints: weight = 0")
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5) "no-constraints: score matches handler")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5) "no-constraints: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-5)
          "no-constraints: x matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "no-constraints: y matches handler"))))

(deftest mixed-constraints-test
  (testing "mixed constrained/unconstrained"
    (let [key (rng/fresh-key 99)
          constraints (cm/choicemap :x (mx/scalar 2.5))
          gf (dyn/with-key simple-model key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
          gf-h (dyn/with-key (force-handler simple-model) key)
          ref (p/generate gf-h [(mx/scalar 0.0)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5) "mixed: score matches handler")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5) "mixed: weight matches handler")
      (is (h/close? 2.5 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6) "mixed: x = constraint")
      (is (not= 0.0 (mx/item weight)) "mixed: weight > 0 (one constrained site)")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "mixed: y matches handler")))

  (testing "constrain only y (second site)"
    (let [key (rng/fresh-key 99)
          constraints (cm/choicemap :y (mx/scalar 7.0))
          gf (dyn/with-key simple-model key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
          gf-h (dyn/with-key (force-handler simple-model) key)
          ref (p/generate gf-h [(mx/scalar 0.0)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5) "mixed-y: score matches handler")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5) "mixed-y: weight matches handler")
      (is (h/close? 7.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y))) 1e-6) "mixed-y: y = constraint"))))

(deftest dependency-chain-test
  (testing "constrain x, leave y and z free"
    (let [key (rng/fresh-key 55)
          constraints (cm/choicemap :x (mx/scalar 1.0))
          gf (dyn/with-key chain-model key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
          gf-h (dyn/with-key (force-handler chain-model) key)
          ref (p/generate gf-h [(mx/scalar 0.0)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5) "chain x-only: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5) "chain x-only: weight matches")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6) "chain x-only: x = constraint")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "chain x-only: y matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :z)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z)))
                    1e-5)
          "chain x-only: z matches handler")))

  (testing "constrain x and z, leave y free"
    (let [key (rng/fresh-key 55)
          constraints (cm/choicemap :x (mx/scalar 1.0) :z (mx/scalar 3.0))
          gf (dyn/with-key chain-model key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
          gf-h (dyn/with-key (force-handler chain-model) key)
          ref (p/generate gf-h [(mx/scalar 0.0)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5) "chain x+z: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5) "chain x+z: weight matches"))))

(deftest multi-distribution-types-test
  (testing "constrain gaussian + bernoulli, leave uniform + exponential free"
    (let [key (rng/fresh-key 33)
          constraints (cm/choicemap :g (mx/scalar 1.5) :b (mx/scalar 1.0))
          gf (dyn/with-key multi-dist-model key)
          {:keys [trace weight]} (p/generate gf [] constraints)
          gf-h (dyn/with-key (force-handler multi-dist-model) key)
          ref (p/generate gf-h [] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-4) "multi-dist: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-4) "multi-dist: weight matches")
      (is (h/close? 1.5 (mx/item (cm/get-value (cm/get-submap (:choices trace) :g))) 1e-6) "multi-dist: g = constraint")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :b))) 1e-6) "multi-dist: b = constraint")))

  (testing "all 4 types constrained"
    (let [key (rng/fresh-key 33)
          constraints (cm/choicemap :g (mx/scalar 0.5)
                                    :u (mx/scalar 0.3)
                                    :b (mx/scalar 0.0)
                                    :e (mx/scalar 1.0))
          gf (dyn/with-key multi-dist-model key)
          {:keys [trace weight]} (p/generate gf [] constraints)
          gf-h (dyn/with-key (force-handler multi-dist-model) key)
          ref (p/generate gf-h [] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-4) "multi-dist all: score matches")
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 1e-4) "multi-dist all: weight = score"))))

(deftest retval-correctness-test
  (testing "retval-model returns (mx/add x y)"
    (let [key (rng/fresh-key 11)
          constraints (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
          gf (dyn/with-key retval-model key)
          {:keys [trace]} (p/generate gf [(mx/scalar 1.0)] constraints)]
      (is (h/close? 5.0 (mx/item (:retval trace)) 1e-6) "retval: add(x,y) = 5.0")))

  (testing "simple-model returns y (last trace site)"
    (let [key (rng/fresh-key 11)
          constraints (cm/choicemap :y (mx/scalar 9.0))
          gf (dyn/with-key simple-model key)
          {:keys [trace]} (p/generate gf [(mx/scalar 0.0)] constraints)]
      (is (h/close? 9.0 (mx/item (:retval trace)) 1e-6) "retval: y = constraint"))))

(deftest delta-distribution-test
  (testing "delta distribution"
    (let [key (rng/fresh-key 22)
          constraints (cm/choicemap :y (mx/scalar 4.0))
          gf (dyn/with-key delta-model key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 2.0)] constraints)
          gf-h (dyn/with-key (force-handler delta-model) key)
          ref (p/generate gf-h [(mx/scalar 2.0)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5) "delta: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5) "delta: weight matches")
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6) "delta: x = v (delta deterministic)"))))

(deftest exotic-distributions-test
  (testing "all exotic constrained"
    (let [key (rng/fresh-key 44)
          constraints (cm/choicemap :a (mx/scalar 0.5)
                                    :b (mx/scalar -1.0)
                                    :c (mx/scalar 2.0))
          gf (dyn/with-key exotic-model key)
          {:keys [trace weight]} (p/generate gf [] constraints)
          gf-h (dyn/with-key (force-handler exotic-model) key)
          ref (p/generate gf-h [] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-4) "exotic all: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-4) "exotic all: weight matches")))

  (testing "partial exotic constraints"
    (let [key (rng/fresh-key 44)
          constraints (cm/choicemap :a (mx/scalar 0.5))
          gf (dyn/with-key exotic-model key)
          {:keys [trace weight]} (p/generate gf [] constraints)
          gf-h (dyn/with-key (force-handler exotic-model) key)
          ref (p/generate gf-h [] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-4) "exotic partial: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-4) "exotic partial: weight matches"))))

(deftest non-compilable-fallback-test
  (testing "dynamic-addr-model falls back to handler"
    (let [key (rng/fresh-key 66)
          constraints (cm/choicemap :x (mx/scalar 2.0) :y0 (mx/scalar 3.0))
          gf (dyn/with-key dynamic-addr-model key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 2.0)] constraints)]
      (is (some? trace) "dynamic-addr: generate works via handler")
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6) "dynamic-addr: x = constraint")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y0))) 1e-6) "dynamic-addr: y0 = constraint")))

  (testing "force-handler on a compilable model still works"
    (let [key (rng/fresh-key 66)
          constraints (cm/choicemap :x (mx/scalar 2.0))
          gf (dyn/with-key (force-handler simple-model) key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)]
      (is (some? trace) "force-handler: generate works")
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6) "force-handler: x = constraint"))))

(deftest trace-structure-test
  (testing "trace structure"
    (let [key (rng/fresh-key 88)
          constraints (cm/choicemap :x (mx/scalar 1.0))
          gf (dyn/with-key simple-model key)
          {:keys [trace]} (p/generate gf [(mx/scalar 0.0)] constraints)]
      (is (= simple-model (:gen-fn trace)) "trace: has :gen-fn")
      (is (and (= 1 (count (:args trace)))
               (= 0.0 (mx/item (first (:args trace)))))
          "trace: has :args")
      (is (some? (:choices trace)) "trace: has :choices")
      (is (some? (:score trace)) "trace: has :score")
      (is (some? (:retval trace)) "trace: has :retval")
      (is (cm/has-value? (cm/get-submap (:choices trace) :x)) "trace: choices has :x")
      (is (cm/has-value? (cm/get-submap (:choices trace) :y)) "trace: choices has :y"))))

(deftest prng-key-equivalence-test
  (testing "PRNG key equivalence"
    (let [key (rng/fresh-key 123)
          gf (dyn/with-key simple-model key)
          {:keys [trace]} (p/generate gf [(mx/scalar 5.0)] cm/EMPTY)
          gf-s (dyn/with-key simple-model key)
          sim-trace (p/simulate gf-s [(mx/scalar 5.0)])]
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-5)
          "prng: generate(empty) x = simulate x")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "prng: generate(empty) y = simulate y"))))

(deftest direct-compiled-generate-test
  (testing "direct compiled-generate call with one constraint"
    (let [compiled-gen (:compiled-generate (:schema simple-model))
          key (rng/fresh-key 200)
          constraints (cm/choicemap :x (mx/scalar 4.0))
          result (compiled-gen key [(mx/scalar 1.0)] constraints)]
      (is (map? (:values result)) "direct: returns map with :values")
      (is (some? (:score result)) "direct: returns :score")
      (is (some? (:weight result)) "direct: returns :weight")
      (is (some? (:retval result)) "direct: returns :retval")
      (is (h/close? 4.0 (mx/item (get (:values result) :x)) 1e-6) "direct: constrained x = 4.0")
      (is (some? (get (:values result) :y)) "direct: unconstrained y is sampled")
      (is (not= 0.0 (mx/item (:weight result))) "direct: weight != 0 (one constrained site)")
      (is (not= (mx/item (:weight result)) (mx/item (:score result))) "direct: weight != score (unconstrained site adds to score only)")))

  (testing "all constrained via direct call"
    (let [compiled-gen (:compiled-generate (:schema simple-model))
          key (rng/fresh-key 201)
          constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          result (compiled-gen key [(mx/scalar 0.0)] constraints)]
      (is (h/close? (mx/item (:score result)) (mx/item (:weight result)) 1e-6) "direct all-constrained: weight = score")))

  (testing "no constraints via direct call"
    (let [compiled-gen (:compiled-generate (:schema simple-model))
          key (rng/fresh-key 202)
          result (compiled-gen key [(mx/scalar 0.0)] cm/EMPTY)]
      (is (h/close? 0.0 (mx/item (:weight result)) 1e-6) "direct no-constraints: weight = 0"))))

(cljs.test/run-tests)
