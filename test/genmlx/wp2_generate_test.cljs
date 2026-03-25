(ns genmlx.wp2-generate-test
  "WP-2 tests: compiled generate for M4 (branch-rewritten) and M3 (partial prefix) models."
  (:require-macros [genmlx.gen :refer [gen]])
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
            [genmlx.schema :as schema]))

;; ---------------------------------------------------------------------------
;; Non-assert helpers (kept)
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-prefix :compiled-prefix-addrs
                       :compiled-prefix-generate)]
    (assoc gf :schema schema)))

;; ---------------------------------------------------------------------------
;; M4 test models (branch-rewritten)
;; ---------------------------------------------------------------------------

(def m4-simple
  (dyn/auto-key
    (gen [flag]
      (let [x (if flag
                (trace :x (dist/gaussian (mx/scalar 1.0) (mx/scalar 0.5)))
                (trace :x (dist/gaussian (mx/scalar -1.0) (mx/scalar 2.0))))
            y (trace :y (dist/gaussian x (mx/scalar 1.0)))]
        y))))

(def m4-multi-site
  (dyn/auto-key
    (gen [flag]
      (let [a (if flag
                (trace :a (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)))
                (trace :a (dist/gaussian (mx/scalar 0.0) (mx/scalar 3.0))))
            b (if flag
                (trace :b (dist/gaussian a (mx/scalar 0.5)))
                (trace :b (dist/gaussian (mx/multiply a (mx/scalar 2.0)) (mx/scalar 1.0))))]
        (mx/add a b)))))

;; ---------------------------------------------------------------------------
;; M3 test models (partial prefix with dynamic tail)
;; ---------------------------------------------------------------------------

(def m3-loop
  (dyn/auto-key
    (gen [n]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 2.0)))]
        (doseq [i (range (mx/item n))]
          (trace (keyword (str "z" i)) (dist/gaussian y (mx/scalar 0.5))))
        y))))

(def m3-dep-prefix
  (dyn/auto-key
    (gen []
      (let [a (trace :a (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/gaussian a (mx/scalar 0.5)))]
        (doseq [i (range 2)]
          (trace (keyword (str "c" i)) (dist/gaussian (mx/add a b) (mx/scalar 1.0))))
        (mx/add a b)))))

(def m3-args
  (dyn/auto-key
    (gen [mu sigma]
      (let [x (trace :x (dist/gaussian mu sigma))
            y (trace :y (dist/gaussian x (mx/scalar 1.0)))]
        (doseq [i (range 3)]
          (trace (keyword (str "d" i)) (dist/gaussian y (mx/scalar 0.5))))
        y))))

(def m3-multi-dist
  (dyn/auto-key
    (gen []
      (let [g (trace :g (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            u (trace :u (dist/uniform (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/bernoulli (mx/scalar 0.5)))
            e (trace :e (dist/exponential (mx/scalar 2.0)))]
        (doseq [i (range 2)]
          (trace (keyword (str "f" i)) (dist/gaussian g (mx/scalar 1.0))))
        g))))

;; ===========================================================================
;; PART A: M4 Branch-Rewritten
;; ===========================================================================

(deftest m4-compilation-presence-test
  (testing "M4 compilation presence"
    (is (some? (:compiled-generate (:schema m4-simple)))
        "m4-simple has :compiled-generate")
    (is (some? (:compiled-generate (:schema m4-multi-site)))
        "m4-multi-site has :compiled-generate")))

(deftest m4-all-constrained-test
  (testing "M4 all sites constrained"
    (let [key (rng/fresh-key 42)
          constraints (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 5.0))
          gf (dyn/with-key m4-simple key)
          {:keys [trace weight]} (p/generate gf [true] constraints)
          gf-h (dyn/with-key (force-handler m4-simple) key)
          ref (p/generate gf-h [true] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m4 all-constrained: score matches handler")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m4 all-constrained: weight matches handler")
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 1e-5)
          "m4 all-constrained: weight = score")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
          "m4 all-constrained: x = constraint")
      (is (h/close? 5.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y))) 1e-6)
          "m4 all-constrained: y = constraint"))))

(deftest m4-no-constraints-test
  (testing "M4 no constraints"
    (let [key (rng/fresh-key 77)
          gf (dyn/with-key m4-simple key)
          {:keys [trace weight]} (p/generate gf [true] cm/EMPTY)
          gf-h (dyn/with-key (force-handler m4-simple) key)
          ref (p/generate gf-h [true] cm/EMPTY)]
      (is (h/close? 0.0 (mx/item weight) 1e-6)
          "m4 no-constraints: weight = 0")
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m4 no-constraints: score matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-5)
          "m4 no-constraints: x matches handler"))))

(deftest m4-mixed-constraints-test
  (testing "M4 mixed constraints"
    (let [key (rng/fresh-key 99)
          constraints (cm/choicemap :x (mx/scalar 2.0))
          gf (dyn/with-key m4-simple key)
          {:keys [trace weight]} (p/generate gf [false] constraints)
          gf-h (dyn/with-key (force-handler m4-simple) key)
          ref (p/generate gf-h [false] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m4 mixed: score matches handler")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m4 mixed: weight matches handler")
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
          "m4 mixed: x = constraint"))))

(deftest m4-multi-site-branch-test
  (testing "M4 multi-site all constrained"
    (let [key (rng/fresh-key 55)
          constraints (cm/choicemap :a (mx/scalar 3.0) :b (mx/scalar 4.0))
          gf (dyn/with-key m4-multi-site key)
          {:keys [trace weight]} (p/generate gf [true] constraints)
          gf-h (dyn/with-key (force-handler m4-multi-site) key)
          ref (p/generate gf-h [true] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m4 multi all: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m4 multi all: weight matches")))

  (testing "M4 multi-site partial constraints"
    (let [key (rng/fresh-key 55)
          constraints (cm/choicemap :a (mx/scalar 3.0))
          gf (dyn/with-key m4-multi-site key)
          {:keys [trace weight]} (p/generate gf [false] constraints)
          gf-h (dyn/with-key (force-handler m4-multi-site) key)
          ref (p/generate gf-h [false] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m4 multi partial: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m4 multi partial: weight matches")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :b)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :b)))
                    1e-5)
          "m4 multi partial: b matches handler"))))

(deftest m4-prng-equivalence-test
  (testing "M4 PRNG equivalence"
    (let [key (rng/fresh-key 123)
          gf (dyn/with-key m4-simple key)
          {:keys [trace]} (p/generate gf [true] cm/EMPTY)
          gf-s (dyn/with-key m4-simple key)
          sim-trace (p/simulate gf-s [true])]
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-5)
          "m4 prng: generate(empty) x = simulate x")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "m4 prng: generate(empty) y = simulate y"))))

;; ===========================================================================
;; PART B: M3 Partial Generate
;; ===========================================================================

(deftest m3-compilation-presence-test
  (testing "M3 compilation presence"
    (is (some? (:compiled-prefix-generate (:schema m3-loop)))
        "m3-loop has :compiled-prefix-generate")
    (is (some? (:compiled-prefix-generate (:schema m3-dep-prefix)))
        "m3-dep-prefix has :compiled-prefix-generate")
    (is (some? (:compiled-prefix-generate (:schema m3-args)))
        "m3-args has :compiled-prefix-generate")
    (is (some? (:compiled-prefix-generate (:schema m3-multi-dist)))
        "m3-multi-dist has :compiled-prefix-generate")))

(deftest m3-prefix-only-constraints-test
  (testing "M3 prefix-only constraints"
    (let [key (rng/fresh-key 42)
          constraints (cm/choicemap :x (mx/scalar 2.0))
          gf (dyn/with-key m3-loop key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 3)] constraints)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          ref (p/generate gf-h [(mx/scalar 3)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m3 pfx-only: score matches handler")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m3 pfx-only: weight matches handler")
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
          "m3 pfx-only: x = constraint"))))

(deftest m3-dynamic-only-constraints-test
  (testing "M3 dynamic-only constraints"
    (let [key (rng/fresh-key 77)
          constraints (cm/choicemap :z0 (mx/scalar 1.5))
          gf (dyn/with-key m3-loop key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 2)] constraints)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          ref (p/generate gf-h [(mx/scalar 2)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m3 dyn-only: score matches handler")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m3 dyn-only: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-5)
          "m3 dyn-only: x matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "m3 dyn-only: y matches handler"))))

(deftest m3-mixed-constraints-test
  (testing "M3 mixed prefix + dynamic constraints"
    (let [key (rng/fresh-key 88)
          constraints (cm/choicemap :x (mx/scalar 1.5) :z1 (mx/scalar 3.0))
          gf (dyn/with-key m3-loop key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 3)] constraints)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          ref (p/generate gf-h [(mx/scalar 3)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m3 mixed: score matches handler")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m3 mixed: weight matches handler")
      (is (h/close? 1.5 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
          "m3 mixed: x = constraint")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :z1))) 1e-6)
          "m3 mixed: z1 = constraint"))))

(deftest m3-no-constraints-test
  (testing "M3 no constraints = simulate"
    (let [key (rng/fresh-key 99)
          gf (dyn/with-key m3-loop key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 2)] cm/EMPTY)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          ref (p/generate gf-h [(mx/scalar 2)] cm/EMPTY)]
      (is (h/close? 0.0 (mx/item weight) 1e-6)
          "m3 no-constraints: weight = 0")
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m3 no-constraints: score matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-5)
          "m3 no-constraints: x matches")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "m3 no-constraints: y matches"))))

(deftest m3-all-constrained-test
  (testing "M3 all constrained"
    (let [key (rng/fresh-key 11)
          constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0)
                                    :z0 (mx/scalar 3.0) :z1 (mx/scalar 4.0))
          gf (dyn/with-key m3-loop key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 2)] constraints)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          ref (p/generate gf-h [(mx/scalar 2)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m3 all: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m3 all: weight matches")
      (is (h/close? (mx/item (:score trace)) (mx/item weight) 1e-5)
          "m3 all: weight = score (all constrained)"))))

(deftest m3-dependent-prefix-test
  (testing "M3 dependent prefix"
    (let [key (rng/fresh-key 33)
          constraints (cm/choicemap :a (mx/scalar 1.0) :c0 (mx/scalar 5.0))
          gf (dyn/with-key m3-dep-prefix key)
          {:keys [trace weight]} (p/generate gf [] constraints)
          gf-h (dyn/with-key (force-handler m3-dep-prefix) key)
          ref (p/generate gf-h [] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m3 dep: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m3 dep: weight matches")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :a))) 1e-6)
          "m3 dep: a = constraint")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :b)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :b)))
                    1e-5)
          "m3 dep: b matches handler"))))

(deftest m3-prefix-with-args-test
  (testing "M3 prefix with gen args"
    (let [key (rng/fresh-key 44)
          constraints (cm/choicemap :x (mx/scalar 3.0))
          gf (dyn/with-key m3-args key)
          {:keys [trace weight]} (p/generate gf [(mx/scalar 5.0) (mx/scalar 2.0)] constraints)
          gf-h (dyn/with-key (force-handler m3-args) key)
          ref (p/generate gf-h [(mx/scalar 5.0) (mx/scalar 2.0)] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-5)
          "m3 args: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-5)
          "m3 args: weight matches")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
          "m3 args: x = constraint"))))

(deftest m3-multi-dist-prefix-test
  (testing "M3 multi-dist prefix"
    (let [key (rng/fresh-key 55)
          constraints (cm/choicemap :g (mx/scalar 0.5) :u (mx/scalar 0.3))
          gf (dyn/with-key m3-multi-dist key)
          {:keys [trace weight]} (p/generate gf [] constraints)
          gf-h (dyn/with-key (force-handler m3-multi-dist) key)
          ref (p/generate gf-h [] constraints)]
      (is (h/close? (mx/item (:score (:trace ref))) (mx/item (:score trace)) 1e-4)
          "m3 multi: score matches")
      (is (h/close? (mx/item (:weight ref)) (mx/item weight) 1e-4)
          "m3 multi: weight matches")
      (is (h/close? 0.5 (mx/item (cm/get-value (cm/get-submap (:choices trace) :g))) 1e-6)
          "m3 multi: g = constraint")
      (is (h/close? 0.3 (mx/item (cm/get-value (cm/get-submap (:choices trace) :u))) 1e-6)
          "m3 multi: u = constraint")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :b)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :b)))
                    1e-5)
          "m3 multi: b matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :e)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :e)))
                    1e-5)
          "m3 multi: e matches handler"))))

(deftest m3-prng-consistency-test
  (testing "M3 PRNG: generate(empty) = simulate"
    (let [key (rng/fresh-key 66)
          gf (dyn/with-key m3-loop key)
          {:keys [trace]} (p/generate gf [(mx/scalar 2)] cm/EMPTY)
          gf-s (dyn/with-key m3-loop key)
          sim-trace (p/simulate gf-s [(mx/scalar 2)])]
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-5)
          "m3 prng: generate(empty) x = simulate x")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "m3 prng: generate(empty) y = simulate y")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :z0)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z0)))
                    1e-5)
          "m3 prng: generate(empty) z0 = simulate z0")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :z1)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z1)))
                    1e-5)
          "m3 prng: generate(empty) z1 = simulate z1")))

  (testing "M3 PRNG: constrained prefix, dynamic sites get correct keys"
    (let [key (rng/fresh-key 66)
          constraints (cm/choicemap :x (mx/scalar 1.0))
          gf (dyn/with-key m3-loop key)
          {:keys [trace]} (p/generate gf [(mx/scalar 2)] constraints)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          ref (p/generate gf-h [(mx/scalar 2)] constraints)]
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    1e-5)
          "m3 prng-constrained: y matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :z0)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z0)))
                    1e-5)
          "m3 prng-constrained: z0 matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :z1)))
                    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z1)))
                    1e-5)
          "m3 prng-constrained: z1 matches handler"))))

(deftest m3-trace-structure-test
  (testing "M3 trace structure"
    (let [key (rng/fresh-key 88)
          constraints (cm/choicemap :x (mx/scalar 1.0))
          gf (dyn/with-key m3-loop key)
          {:keys [trace]} (p/generate gf [(mx/scalar 3)] constraints)]
      (is (= m3-loop (:gen-fn trace)) "m3 trace: has :gen-fn")
      (is (cm/has-value? (cm/get-submap (:choices trace) :x)) "m3 trace: has x")
      (is (cm/has-value? (cm/get-submap (:choices trace) :y)) "m3 trace: has y")
      (is (cm/has-value? (cm/get-submap (:choices trace) :z0)) "m3 trace: has z0")
      (is (cm/has-value? (cm/get-submap (:choices trace) :z1)) "m3 trace: has z1")
      (is (cm/has-value? (cm/get-submap (:choices trace) :z2)) "m3 trace: has z2")
      (is (some? (:retval trace)) "m3 trace: has retval")
      (is (some? (:score trace)) "m3 trace: has score"))))

(deftest m3-retval-test
  (testing "M3 retval correctness"
    (let [key (rng/fresh-key 22)
          constraints (cm/choicemap :a (mx/scalar 2.0) :b (mx/scalar 3.0))
          gf (dyn/with-key m3-dep-prefix key)
          {:keys [trace]} (p/generate gf [] constraints)]
      (is (h/close? 5.0 (mx/item (:retval trace)) 1e-6)
          "m3 retval: add(a,b) = 5.0"))))

(cljs.test/run-tests)
