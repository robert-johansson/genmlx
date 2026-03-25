(ns genmlx.compiled-simulate-test
  "L1-M2 tests: compiled simulate for static models.

   Tests cover:
   1. Expression compiler (compile-expr) independently
   2. Binding environment construction
   3. Compiled simulate on multiple model types
   4. PRNG equivalence (compiled vs handler, same key → same values)
   5. Score correctness
   6. GFI contracts on compiled models
   7. Fallback to handler for non-static models
   8. Design constraints (no atoms/volatile! in compiled path)"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.schema :as schema]))

;; ---------------------------------------------------------------------------
;; Test utilities
;; ---------------------------------------------------------------------------

(defn- force-handler
  "Return a copy of gf that always uses the handler path (no compiled-simulate)."
  [gf]
  (dyn/->DynamicGF (:body-fn gf) (:source gf)
                    (dissoc (:schema gf) :compiled-simulate)))

;; ---------------------------------------------------------------------------
;; 1. Binding environment construction
;; ---------------------------------------------------------------------------

(deftest binding-environment-test
  (testing "single param + single trace"
    (let [source '([x] (let [slope (trace :slope (dist/gaussian 0 10))] slope))
          env (compiled/build-binding-env source)]
      (is (= {:kind :param :index 0} (get env "x")) "param x")
      (is (= {:kind :trace :addr :slope} (get env "slope")) "trace slope")))

  (testing "multiple params + multiple traces + expr binding"
    (let [source '([x y] (let [a (trace :a (dist/gaussian 0 1))
                                b (trace :b (dist/gaussian 0 1))
                                c (mx/add a b)]
                            (trace :d (dist/gaussian c 1))))
          env (compiled/build-binding-env source)]
      (is (= {:kind :param :index 0} (get env "x")) "param x idx 0")
      (is (= {:kind :param :index 1} (get env "y")) "param y idx 1")
      (is (= {:kind :trace :addr :a} (get env "a")) "trace a")
      (is (= {:kind :trace :addr :b} (get env "b")) "trace b")
      (is (= {:kind :expr :form '(mx/add a b)} (get env "c")) "expr c"))))

;; ---------------------------------------------------------------------------
;; 2. Expression compiler
;; ---------------------------------------------------------------------------

(deftest expression-compiler-test
  (let [env {"x" {:kind :param :index 0}
             "slope" {:kind :trace :addr :slope}
             "predicted" {:kind :expr :form '(mx/multiply slope x)}}]

    (testing "number literal"
      (let [f (compiled/compile-expr 5 env #{})]
        (is (= 5 (f {} [])) "number literal")))

    (testing "param symbol"
      (let [f (compiled/compile-expr 'x env #{})]
        (is (= 42 (f {} [42])) "param symbol")))

    (testing "trace symbol"
      (let [f (compiled/compile-expr 'slope env #{})]
        (is (= :val (f {:slope :val} [])) "trace symbol")))

    (testing "expr symbol (intermediate let binding)"
      (let [f (compiled/compile-expr 'predicted env #{})
            v {:slope (mx/scalar 3.0)}
            a [(mx/scalar 2.0)]]
        (is (some? f) "expr symbol compiled")
        (is (h/close? 6.0 (mx/item (f v a)) 1e-6) "expr symbol value")))

    (testing "function call: mx/add"
      (let [f (compiled/compile-expr '(mx/add slope x) env #{})]
        (is (some? f) "mx/add compiled")
        (is (h/close? 5.0 (mx/item (f {:slope (mx/scalar 3.0)} [(mx/scalar 2.0)])) 1e-6)
            "mx/add value")))

    (testing "nested expression"
      (let [f (compiled/compile-expr '(mx/add (mx/multiply slope x) slope) env #{})]
        (is (some? f) "nested expr compiled")
        (is (h/close? 9.0 (mx/item (f {:slope (mx/scalar 3.0)} [(mx/scalar 2.0)])) 1e-6)
            "nested expr value")))

    (testing "unsupported: unknown symbol"
      (let [f (compiled/compile-expr 'unknown env #{})]
        (is (nil? f) "unknown symbol -> nil")))

    (testing "unsupported: unknown function"
      (let [f (compiled/compile-expr '(custom/fn 1 2) env #{})]
        (is (nil? f) "unknown fn -> nil")))

    (testing "keyword literal"
      (let [f (compiled/compile-expr :foo env #{})]
        (is (= :foo (f {} [])) "keyword literal")))

    (testing "boolean literal"
      (let [f (compiled/compile-expr true env #{})]
        (is (= true (f {} [])) "boolean literal")))))

;; ---------------------------------------------------------------------------
;; 3. Compilation gate tests
;; ---------------------------------------------------------------------------

(deftest compilation-gates-test
  (testing "static model compiles"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)))]
      (is (some? (:compiled-simulate (:schema m))) "static model compiles")))

  (testing "splice model not compiled"
    (let [sub (gen [] (trace :a (dist/gaussian 0 1)))
          m (gen [] (splice :sub sub))]
      (is (nil? (:compiled-simulate (:schema m))) "splice model not compiled")))

  (testing "dynamic addr schema not static"
    (let [source '([n] (doseq [i (range n)]
                         (trace (keyword (str "x" i)) (dist/gaussian 0 1))))
          schema (schema/extract-schema source)]
      (is (not (:static? schema)) "dynamic addr schema not static")))

  (testing "branching model not compiled"
    (let [m (gen [flag]
              (if flag
                (trace :a (dist/gaussian 0 1))
                (trace :b (dist/gaussian 0 1))))]
      (is (nil? (:compiled-simulate (:schema m))) "branching model not compiled"))))

;; ---------------------------------------------------------------------------
;; 4. Model tests: compiled simulate correctness
;; ---------------------------------------------------------------------------

(deftest compiled-simulate-correctness-test
  (testing "single site, constant args"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)))
          k (rng/fresh-key 42)
          t (p/simulate (dyn/with-key m k) [])]
      (is (cm/has-value? (cm/get-submap (:choices t) :x)) "single site: has :x")
      (is (js/isFinite (mx/item (:score t))) "single site: finite score")))

  (testing "multi-site, independent"
    (let [m (gen []
              (trace :a (dist/gaussian 0 10))
              (trace :b (dist/uniform 0 1))
              (trace :c (dist/bernoulli 0.5)))
          k (rng/fresh-key 42)
          t (p/simulate (dyn/with-key m k) [])]
      (is (cm/has-value? (cm/get-submap (:choices t) :a)) "multi-site: has :a")
      (is (cm/has-value? (cm/get-submap (:choices t) :b)) "multi-site: has :b")
      (is (cm/has-value? (cm/get-submap (:choices t) :c)) "multi-site: has :c")
      (is (js/isFinite (mx/item (:score t))) "multi-site: finite score")))

  (testing "dependent sites with gen arg"
    (let [m (gen [x]
              (let [slope (trace :slope (dist/gaussian 0 10))
                    intercept (trace :intercept (dist/gaussian 0 5))]
                (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))
                slope))
          k (rng/fresh-key 42)
          t (p/simulate (dyn/with-key m k) [(mx/scalar 2.0)])]
      (is (cm/has-value? (cm/get-submap (:choices t) :slope)) "dependent: has :slope")
      (is (cm/has-value? (cm/get-submap (:choices t) :intercept)) "dependent: has :intercept")
      (is (cm/has-value? (cm/get-submap (:choices t) :y)) "dependent: has :y")
      (is (js/isFinite (mx/item (:score t))) "dependent: finite score")
      (is (mx/array? (:retval t)) "dependent: retval is slope")))

  (testing "return expression (not just a symbol)"
    (let [m (gen [x]
              (let [a (trace :a (dist/gaussian 0 1))]
                (mx/multiply a x)))
          k (rng/fresh-key 42)
          t (p/simulate (dyn/with-key m k) [(mx/scalar 5.0)])
          a-val (mx/item (cm/get-value (cm/get-submap (:choices t) :a)))
          retval (mx/item (:retval t))]
      (is (h/close? (* a-val 5.0) retval 1e-5) "return expr: retval = a * x")))

  (testing "intermediate let binding (expr, not trace)"
    (let [m (gen [x]
              (let [slope (trace :slope (dist/gaussian 0 10))
                    predicted (mx/multiply slope x)]
                (trace :y (dist/gaussian predicted 1))
                predicted))
          k (rng/fresh-key 42)
          t (p/simulate (dyn/with-key m k) [(mx/scalar 3.0)])
          slope-val (mx/item (cm/get-value (cm/get-submap (:choices t) :slope)))
          retval (mx/item (:retval t))]
      (is (h/close? (* slope-val 3.0) retval 1e-5) "intermediate let: retval = slope * x")))

  (testing "multiple distributions"
    (let [m (gen []
              (let [mu (trace :mu (dist/gaussian 0 5))
                    p  (trace :p (dist/uniform 0 1))]
                (trace :obs (dist/gaussian mu 1))
                (trace :coin (dist/bernoulli p))
                mu))
          k (rng/fresh-key 42)
          t (p/simulate (dyn/with-key m k) [])]
      (is (cm/has-value? (cm/get-submap (:choices t) :mu)) "multi-dist: has :mu")
      (is (cm/has-value? (cm/get-submap (:choices t) :p)) "multi-dist: has :p")
      (is (cm/has-value? (cm/get-submap (:choices t) :obs)) "multi-dist: has :obs")
      (is (cm/has-value? (cm/get-submap (:choices t) :coin)) "multi-dist: has :coin")
      (is (js/isFinite (mx/item (:score t))) "multi-dist: finite score"))))

;; ---------------------------------------------------------------------------
;; 5. PRNG equivalence: compiled vs handler
;; ---------------------------------------------------------------------------

(deftest prng-equivalence-test
  (letfn [(test-equivalence [desc model args]
            (let [k (rng/fresh-key 77)
                  compiled-trace (p/simulate (dyn/with-key model k) args)
                  k2 (rng/fresh-key 77)
                  handler-trace (p/simulate (dyn/with-key (force-handler model) k2) args)]
              (is (h/close? (mx/item (:score compiled-trace))
                            (mx/item (:score handler-trace))
                            1e-5)
                  (str desc " score"))
              (doseq [ts (:trace-sites (:schema model))]
                (let [addr (:addr ts)
                      cv (mx/item (cm/get-value (cm/get-submap (:choices compiled-trace) addr)))
                      hv (mx/item (cm/get-value (cm/get-submap (:choices handler-trace) addr)))]
                  (is (h/close? cv hv 1e-5) (str desc " " addr))))))]

    (testing "single-site"
      (test-equivalence "single-site"
        (gen [] (trace :x (dist/gaussian 0 1)))
        []))

    (testing "multi-site"
      (test-equivalence "multi-site"
        (gen []
          (trace :a (dist/gaussian 0 10))
          (trace :b (dist/gaussian 5 2)))
        []))

    (testing "dependent-linreg"
      (test-equivalence "dependent-linreg"
        (gen [x]
          (let [slope (trace :slope (dist/gaussian 0 10))
                intercept (trace :intercept (dist/gaussian 0 5))]
            (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))
            slope))
        [(mx/scalar 2.5)]))

    (testing "intermediate-let"
      (test-equivalence "intermediate-let"
        (gen [x]
          (let [slope (trace :slope (dist/gaussian 0 10))
                predicted (mx/multiply slope x)]
            (trace :y (dist/gaussian predicted 1))
            predicted))
        [(mx/scalar 3.0)]))

    (testing "uniform+bernoulli"
      (test-equivalence "uniform+bernoulli"
        (gen []
          (let [p (trace :p (dist/uniform 0 1))]
            (trace :coin (dist/bernoulli p))
            p))
        []))))

;; ---------------------------------------------------------------------------
;; 6. Score correctness (manual verification)
;; ---------------------------------------------------------------------------

(deftest score-correctness-test
  (testing "single site: score = manual log-prob"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)))
          k (rng/fresh-key 42)
          t (p/simulate (dyn/with-key m k) [])
          x-val (cm/get-value (cm/get-submap (:choices t) :x))
          manual-lp (mx/item (dc/dist-log-prob (dist/gaussian 0 1) x-val))]
      (is (h/close? manual-lp (mx/item (:score t)) 1e-6) "score = manual log-prob")))

  (testing "multi-site: score = sum of log-probs"
    (let [m (gen []
              (let [a (trace :a (dist/gaussian 0 10))]
                (trace :b (dist/gaussian a 1))
                a))
          k (rng/fresh-key 42)
          t (p/simulate (dyn/with-key m k) [])
          a-val (cm/get-value (cm/get-submap (:choices t) :a))
          b-val (cm/get-value (cm/get-submap (:choices t) :b))
          manual-score (+ (mx/item (dc/dist-log-prob (dist/gaussian 0 10) a-val))
                          (mx/item (dc/dist-log-prob (dist/gaussian a-val 1) b-val)))]
      (is (h/close? manual-score (mx/item (:score t)) 1e-5)
          "multi-site score = sum of log-probs"))))

;; ---------------------------------------------------------------------------
;; 7. GFI operations still work on compiled models
;; ---------------------------------------------------------------------------

(deftest gfi-operations-test
  (let [m (gen [x]
            (let [slope (trace :slope (dist/gaussian 0 10))]
              (trace :y (dist/gaussian (mx/multiply slope x) 1))
              slope))
        k (rng/fresh-key 42)]

    (testing "simulate works (compiled)"
      (let [t (p/simulate (dyn/with-key m k) [(mx/scalar 2.0)])]
        (is (some? t) "simulate works")
        (is (cm/has-value? (cm/get-submap (:choices t) :slope)) "simulate has choices")))

    (testing "generate works"
      (let [obs (cm/set-value cm/EMPTY :y (mx/scalar 5.0))
            {:keys [trace weight]} (p/generate (dyn/with-key m (rng/fresh-key 42))
                                               [(mx/scalar 2.0)] obs)]
        (is (some? trace) "generate works")
        (is (js/isFinite (mx/item weight)) "generate has weight")))

    (testing "update works (handler path)"
      (let [t1 (p/simulate (dyn/with-key m (rng/fresh-key 42)) [(mx/scalar 2.0)])
            new-obs (cm/set-value cm/EMPTY :y (mx/scalar 3.0))
            {:keys [trace weight]} (p/update (dyn/with-key m (rng/fresh-key 43)) t1 new-obs)]
        (is (some? trace) "update works")
        (is (js/isFinite (mx/item weight)) "update has weight")))))

;; ---------------------------------------------------------------------------
;; 8. Handler fallback for non-compilable models
;; ---------------------------------------------------------------------------

(deftest handler-fallback-test
  (testing "splice model simulates via handler"
    (let [sub (gen [] (trace :a (dist/gaussian 0 1)))
          m (gen [] (splice :sub sub))
          t (p/simulate (dyn/auto-key m) [])]
      (is (some? t) "splice model simulates")))

  (testing "loop model: not compiled, not static"
    (let [m (gen [n]
              (doseq [i (range n)]
                (trace (keyword (str "x" i)) (dist/gaussian 0 1))))]
      (is (nil? (:compiled-simulate (:schema m))) "loop model: not compiled")
      (is (not (:static? (:schema m))) "loop model: not static"))))

;; ---------------------------------------------------------------------------
;; 9. Design constraint verification
;; ---------------------------------------------------------------------------

(deftest design-constraints-test
  (testing "DC-3: compiled-simulate is a pure function"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)))
          csim (:compiled-simulate (:schema m))
          k (rng/fresh-key 42)
          _ (rng/seed! k)
          r1 (csim k [])
          _ (mx/materialize! (:score r1) (get (:values r1) :x))
          _ (rng/seed! k)
          r2 (csim k [])]
      (is (h/close? (mx/item (:score r1)) (mx/item (:score r2)) 1e-6)
          "DC-3: same key -> same score")
      (is (h/close? (mx/item (get (:values r1) :x))
                    (mx/item (get (:values r2) :x))
                    1e-6)
          "DC-3: same key -> same value")))

  (testing "DC-4: DynamicGF dispatches compiled vs handler"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)))]
      (is (some? (:compiled-simulate (:schema m)))
          "DC-4: static model has compiled-simulate")
      (is (fn? (:compiled-simulate (:schema m)))
          "DC-4: compiled-simulate is a fn")))

  (testing "DC-6: compiled fn returns raw values, not choicemaps"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)))
          csim (:compiled-simulate (:schema m))
          result (csim (rng/fresh-key 42) [])]
      (is (map? (:values result)) "DC-6: values is a map")
      (is (mx/array? (get (:values result) :x)) "DC-6: values :x is MLX array")
      (is (mx/array? (:score result)) "DC-6: score is MLX array"))))

;; ---------------------------------------------------------------------------
;; 10. Edge cases
;; ---------------------------------------------------------------------------

(deftest edge-cases-test
  (testing "model with nil return value"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)) nil)
          t (p/simulate (dyn/auto-key m) [])]
      (is (nil? (:retval t)) "nil return")))

  (testing "model returning a number"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)) 42)
          t (p/simulate (dyn/auto-key m) [])]
      (is (= 42 (:retval t)) "number return")))

  (testing "single-site model where return IS the trace call"
    (let [m (gen [] (trace :x (dist/gaussian 0 1)))
          t (p/simulate (dyn/auto-key m) [])]
      (is (mx/array? (:retval t)) "trace-as-return: retval is MLX array")
      (is (h/close? (mx/item (:retval t))
                    (mx/item (cm/get-value (cm/get-submap (:choices t) :x)))
                    1e-6)
          "trace-as-return: retval = :x value")))

  (testing "model with multiple gen params"
    (let [m (gen [a b c]
              (trace :x (dist/gaussian a b))
              (trace :y (dist/gaussian b c)))
          t (p/simulate (dyn/auto-key m) [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)])]
      (is (some? (:compiled-simulate (:schema m))) "multi-param: compiles")
      (is (cm/has-value? (cm/get-submap (:choices t) :x)) "multi-param: has :x")
      (is (cm/has-value? (cm/get-submap (:choices t) :y)) "multi-param: has :y"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
