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
  (:require [genmlx.gen :refer [gen]]
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
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toFixed diff 6) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " expected=" expected " actual=" actual " diff=" diff))))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " expected=" expected " actual=" actual)))))

(defn- force-handler
  "Return a copy of gf that always uses the handler path (no compiled-simulate)."
  [gf]
  (dyn/->DynamicGF (:body-fn gf) (:source gf)
                    (dissoc (:schema gf) :compiled-simulate)))

;; ---------------------------------------------------------------------------
;; 1. Binding environment construction
;; ---------------------------------------------------------------------------

(println "\n== 1. Binding environment ==")

(let [source '([x] (let [slope (trace :slope (dist/gaussian 0 10))] slope))]
  (let [env (compiled/build-binding-env source)]
    (assert-equal "param x" {:kind :param :index 0} (get env "x"))
    (assert-equal "trace slope" {:kind :trace :addr :slope} (get env "slope"))))

(let [source '([x y] (let [a (trace :a (dist/gaussian 0 1))
                            b (trace :b (dist/gaussian 0 1))
                            c (mx/add a b)]
                        (trace :d (dist/gaussian c 1))))]
  (let [env (compiled/build-binding-env source)]
    (assert-equal "param x idx 0" {:kind :param :index 0} (get env "x"))
    (assert-equal "param y idx 1" {:kind :param :index 1} (get env "y"))
    (assert-equal "trace a" {:kind :trace :addr :a} (get env "a"))
    (assert-equal "trace b" {:kind :trace :addr :b} (get env "b"))
    (assert-equal "expr c" {:kind :expr :form '(mx/add a b)} (get env "c"))))

;; ---------------------------------------------------------------------------
;; 2. Expression compiler
;; ---------------------------------------------------------------------------

(println "\n== 2. Expression compiler ==")

(let [env {"x" {:kind :param :index 0}
           "slope" {:kind :trace :addr :slope}
           "predicted" {:kind :expr :form '(mx/multiply slope x)}}]

  ;; Number literal
  (let [f (compiled/compile-expr 5 env #{})]
    (assert-true "number literal" (= 5 (f {} []))))

  ;; Param symbol
  (let [f (compiled/compile-expr 'x env #{})]
    (assert-true "param symbol" (= 42 (f {} [42]))))

  ;; Trace symbol
  (let [f (compiled/compile-expr 'slope env #{})]
    (assert-true "trace symbol" (= :val (f {:slope :val} []))))

  ;; Expr symbol (intermediate let binding)
  (let [f (compiled/compile-expr 'predicted env #{})
        v {:slope (mx/scalar 3.0)}
        a [(mx/scalar 2.0)]]
    (assert-true "expr symbol compiled" (some? f))
    (assert-close "expr symbol value" 6.0 (mx/item (f v a)) 1e-6))

  ;; Function call: mx/add
  (let [f (compiled/compile-expr '(mx/add slope x) env #{})]
    (assert-true "mx/add compiled" (some? f))
    (assert-close "mx/add value" 5.0
                  (mx/item (f {:slope (mx/scalar 3.0)} [(mx/scalar 2.0)])) 1e-6))

  ;; Nested expression
  (let [f (compiled/compile-expr '(mx/add (mx/multiply slope x) slope) env #{})]
    (assert-true "nested expr compiled" (some? f))
    (assert-close "nested expr value" 9.0
                  (mx/item (f {:slope (mx/scalar 3.0)} [(mx/scalar 2.0)])) 1e-6))

  ;; Unsupported: unknown symbol
  (let [f (compiled/compile-expr 'unknown env #{})]
    (assert-true "unknown symbol → nil" (nil? f)))

  ;; Unsupported: unknown function
  (let [f (compiled/compile-expr '(custom/fn 1 2) env #{})]
    (assert-true "unknown fn → nil" (nil? f)))

  ;; Keyword literal
  (let [f (compiled/compile-expr :foo env #{})]
    (assert-true "keyword literal" (= :foo (f {} []))))

  ;; Boolean literal
  (let [f (compiled/compile-expr true env #{})]
    (assert-true "boolean literal" (= true (f {} [])))))

;; ---------------------------------------------------------------------------
;; 3. Compilation gate tests
;; ---------------------------------------------------------------------------

(println "\n== 3. Compilation gates ==")

;; Static model → compiled
(let [m (gen [] (trace :x (dist/gaussian 0 1)))]
  (assert-true "static model compiles" (some? (:compiled-simulate (:schema m)))))

;; Model with splice → not compiled
(let [sub (gen [] (trace :a (dist/gaussian 0 1)))
      m (gen [] (splice :sub sub))]
  (assert-true "splice model not compiled" (nil? (:compiled-simulate (:schema m)))))

;; Model with dynamic addresses → not compiled
(let [source '([n] (doseq [i (range n)]
                     (trace (keyword (str "x" i)) (dist/gaussian 0 1))))
      schema (schema/extract-schema source)]
  (assert-true "dynamic addr schema not static" (not (:static? schema))))

;; Model with branches → not compiled
(let [m (gen [flag]
          (if flag
            (trace :a (dist/gaussian 0 1))
            (trace :b (dist/gaussian 0 1))))]
  (assert-true "branching model not compiled" (nil? (:compiled-simulate (:schema m)))))

;; ---------------------------------------------------------------------------
;; 4. Model tests: compiled simulate correctness
;; ---------------------------------------------------------------------------

(println "\n== 4. Compiled simulate correctness ==")

;; Model 1: Single site, constant args
(let [m (gen [] (trace :x (dist/gaussian 0 1)))
      k (rng/fresh-key 42)
      t (p/simulate (dyn/with-key m k) [])]
  (assert-true "single site: has :x" (cm/has-value? (cm/get-submap (:choices t) :x)))
  (assert-true "single site: finite score" (js/isFinite (mx/item (:score t)))))

;; Model 2: Multi-site, independent
(let [m (gen []
          (trace :a (dist/gaussian 0 10))
          (trace :b (dist/uniform 0 1))
          (trace :c (dist/bernoulli 0.5)))
      k (rng/fresh-key 42)
      t (p/simulate (dyn/with-key m k) [])]
  (assert-true "multi-site: has :a" (cm/has-value? (cm/get-submap (:choices t) :a)))
  (assert-true "multi-site: has :b" (cm/has-value? (cm/get-submap (:choices t) :b)))
  (assert-true "multi-site: has :c" (cm/has-value? (cm/get-submap (:choices t) :c)))
  (assert-true "multi-site: finite score" (js/isFinite (mx/item (:score t)))))

;; Model 3: Dependent sites with gen arg
(let [m (gen [x]
          (let [slope (trace :slope (dist/gaussian 0 10))
                intercept (trace :intercept (dist/gaussian 0 5))]
            (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))
            slope))
      k (rng/fresh-key 42)
      t (p/simulate (dyn/with-key m k) [(mx/scalar 2.0)])]
  (assert-true "dependent: has :slope" (cm/has-value? (cm/get-submap (:choices t) :slope)))
  (assert-true "dependent: has :intercept" (cm/has-value? (cm/get-submap (:choices t) :intercept)))
  (assert-true "dependent: has :y" (cm/has-value? (cm/get-submap (:choices t) :y)))
  (assert-true "dependent: finite score" (js/isFinite (mx/item (:score t))))
  (assert-true "dependent: retval is slope" (mx/array? (:retval t))))

;; Model 4: Return expression (not just a symbol)
(let [m (gen [x]
          (let [a (trace :a (dist/gaussian 0 1))]
            (mx/multiply a x)))
      k (rng/fresh-key 42)
      t (p/simulate (dyn/with-key m k) [(mx/scalar 5.0)])]
  (let [a-val (mx/item (cm/get-value (cm/get-submap (:choices t) :a)))
        retval (mx/item (:retval t))]
    (assert-close "return expr: retval = a * x" (* a-val 5.0) retval 1e-5)))

;; Model 5: Intermediate let binding (expr, not trace)
(let [m (gen [x]
          (let [slope (trace :slope (dist/gaussian 0 10))
                predicted (mx/multiply slope x)]
            (trace :y (dist/gaussian predicted 1))
            predicted))
      k (rng/fresh-key 42)
      t (p/simulate (dyn/with-key m k) [(mx/scalar 3.0)])]
  (let [slope-val (mx/item (cm/get-value (cm/get-submap (:choices t) :slope)))
        retval (mx/item (:retval t))]
    (assert-close "intermediate let: retval = slope * x" (* slope-val 3.0) retval 1e-5)))

;; Model 6: Multiple distributions
(let [m (gen []
          (let [mu (trace :mu (dist/gaussian 0 5))
                p  (trace :p (dist/uniform 0 1))]
            (trace :obs (dist/gaussian mu 1))
            (trace :coin (dist/bernoulli p))
            mu))
      k (rng/fresh-key 42)
      t (p/simulate (dyn/with-key m k) [])]
  (assert-true "multi-dist: has :mu" (cm/has-value? (cm/get-submap (:choices t) :mu)))
  (assert-true "multi-dist: has :p" (cm/has-value? (cm/get-submap (:choices t) :p)))
  (assert-true "multi-dist: has :obs" (cm/has-value? (cm/get-submap (:choices t) :obs)))
  (assert-true "multi-dist: has :coin" (cm/has-value? (cm/get-submap (:choices t) :coin)))
  (assert-true "multi-dist: finite score" (js/isFinite (mx/item (:score t)))))

;; ---------------------------------------------------------------------------
;; 5. PRNG equivalence: compiled vs handler
;; ---------------------------------------------------------------------------

(println "\n== 5. PRNG equivalence ==")

(defn- test-equivalence [desc model args]
  (let [k (rng/fresh-key 77)
        compiled-trace (p/simulate (dyn/with-key model k) args)
        k2 (rng/fresh-key 77)
        handler-trace (p/simulate (dyn/with-key (force-handler model) k2) args)]
    ;; Compare scores — mx/compile-fn graph fusion may introduce ~1e-6 diffs
    (assert-close (str desc " score")
                  (mx/item (:score compiled-trace))
                  (mx/item (:score handler-trace))
                  1e-5)
    ;; Compare each choice
    (doseq [ts (:trace-sites (:schema model))]
      (let [addr (:addr ts)
            cv (mx/item (cm/get-value (cm/get-submap (:choices compiled-trace) addr)))
            hv (mx/item (cm/get-value (cm/get-submap (:choices handler-trace) addr)))]
        (assert-close (str desc " " addr) cv hv 1e-5)))))

(test-equivalence "single-site"
  (gen [] (trace :x (dist/gaussian 0 1)))
  [])

(test-equivalence "multi-site"
  (gen []
    (trace :a (dist/gaussian 0 10))
    (trace :b (dist/gaussian 5 2)))
  [])

(test-equivalence "dependent-linreg"
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))]
      (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))
      slope))
  [(mx/scalar 2.5)])

(test-equivalence "intermediate-let"
  (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))
          predicted (mx/multiply slope x)]
      (trace :y (dist/gaussian predicted 1))
      predicted))
  [(mx/scalar 3.0)])

(test-equivalence "uniform+bernoulli"
  (gen []
    (let [p (trace :p (dist/uniform 0 1))]
      (trace :coin (dist/bernoulli p))
      p))
  [])

;; ---------------------------------------------------------------------------
;; 6. Score correctness (manual verification)
;; ---------------------------------------------------------------------------

(println "\n== 6. Score correctness ==")

(let [m (gen [] (trace :x (dist/gaussian 0 1)))
      k (rng/fresh-key 42)
      t (p/simulate (dyn/with-key m k) [])
      x-val (cm/get-value (cm/get-submap (:choices t) :x))
      ;; Manual: log-prob of gaussian(0,1) at x
      manual-lp (mx/item (dc/dist-log-prob (dist/gaussian 0 1) x-val))]
  (assert-close "score = manual log-prob" manual-lp (mx/item (:score t)) 1e-6))

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
  (assert-close "multi-site score = sum of log-probs" manual-score (mx/item (:score t)) 1e-5))

;; ---------------------------------------------------------------------------
;; 7. GFI operations still work on compiled models
;; ---------------------------------------------------------------------------

(println "\n== 7. GFI operations on compiled models ==")

(let [m (gen [x]
          (let [slope (trace :slope (dist/gaussian 0 10))]
            (trace :y (dist/gaussian (mx/multiply slope x) 1))
            slope))
      k (rng/fresh-key 42)]

  ;; simulate works (compiled)
  (let [t (p/simulate (dyn/with-key m k) [(mx/scalar 2.0)])]
    (assert-true "simulate works" (some? t))
    (assert-true "simulate has choices" (cm/has-value? (cm/get-submap (:choices t) :slope))))

  ;; generate works (falls back to handler since only simulate is compiled)
  (let [obs (cm/set-value cm/EMPTY :y (mx/scalar 5.0))
        {:keys [trace weight]} (p/generate (dyn/with-key m (rng/fresh-key 42))
                                           [(mx/scalar 2.0)] obs)]
    (assert-true "generate works" (some? trace))
    (assert-true "generate has weight" (js/isFinite (mx/item weight))))

  ;; update works (handler path)
  (let [t1 (p/simulate (dyn/with-key m (rng/fresh-key 42)) [(mx/scalar 2.0)])
        new-obs (cm/set-value cm/EMPTY :y (mx/scalar 3.0))
        {:keys [trace weight]} (p/update (dyn/with-key m (rng/fresh-key 43)) t1 new-obs)]
    (assert-true "update works" (some? trace))
    (assert-true "update has weight" (js/isFinite (mx/item weight)))))

;; ---------------------------------------------------------------------------
;; 8. Handler fallback for non-compilable models
;; ---------------------------------------------------------------------------

(println "\n== 8. Handler fallback ==")

;; Model with splice → handler
(let [sub (gen [] (trace :a (dist/gaussian 0 1)))
      m (gen [] (splice :sub sub))]
  (let [t (p/simulate (dyn/auto-key m) [])]
    (assert-true "splice model simulates" (some? t))))

;; Model with loop → handler
(let [m (gen [n]
          (doseq [i (range n)]
            (trace (keyword (str "x" i)) (dist/gaussian 0 1))))]
  (assert-true "loop model: not compiled" (nil? (:compiled-simulate (:schema m))))
  (assert-true "loop model: not static" (not (:static? (:schema m)))))

;; ---------------------------------------------------------------------------
;; 9. Design constraint verification
;; ---------------------------------------------------------------------------

(println "\n== 9. Design constraints ==")

;; DC-3: compiled-simulate is a pure function
;; Calling it twice with same inputs and same seed gives same results
(let [m (gen [] (trace :x (dist/gaussian 0 1)))
      csim (:compiled-simulate (:schema m))
      k (rng/fresh-key 42)
      _ (rng/seed! k)
      r1 (csim k [])
      _ (mx/materialize! (:score r1) (get (:values r1) :x))
      _ (rng/seed! k)
      r2 (csim k [])]
  (assert-close "DC-3: same key → same score"
                (mx/item (:score r1)) (mx/item (:score r2)) 1e-6)
  (assert-close "DC-3: same key → same value"
                (mx/item (get (:values r1) :x))
                (mx/item (get (:values r2) :x))
                1e-6))

;; DC-4: DynamicGF dispatches compiled vs handler
(let [m (gen [] (trace :x (dist/gaussian 0 1)))]
  (assert-true "DC-4: static model has compiled-simulate"
               (some? (:compiled-simulate (:schema m))))
  (assert-true "DC-4: compiled-simulate is a fn"
               (fn? (:compiled-simulate (:schema m)))))

;; DC-6: compiled fn returns raw values, not choicemaps
(let [m (gen [] (trace :x (dist/gaussian 0 1)))
      csim (:compiled-simulate (:schema m))
      result (csim (rng/fresh-key 42) [])]
  (assert-true "DC-6: values is a map" (map? (:values result)))
  (assert-true "DC-6: values :x is MLX array" (mx/array? (get (:values result) :x)))
  (assert-true "DC-6: score is MLX array" (mx/array? (:score result))))

;; ---------------------------------------------------------------------------
;; 10. Edge cases
;; ---------------------------------------------------------------------------

(println "\n== 10. Edge cases ==")

;; Model with no return value (returns nil)
(let [m (gen [] (trace :x (dist/gaussian 0 1)) nil)
      t (p/simulate (dyn/auto-key m) [])]
  (assert-true "nil return" (nil? (:retval t))))

;; Model returning a number
(let [m (gen [] (trace :x (dist/gaussian 0 1)) 42)
      t (p/simulate (dyn/auto-key m) [])]
  (assert-equal "number return" 42 (:retval t)))

;; Single-site model where return IS the trace call
(let [m (gen [] (trace :x (dist/gaussian 0 1)))
      t (p/simulate (dyn/auto-key m) [])]
  (assert-true "trace-as-return: retval is MLX array" (mx/array? (:retval t)))
  (assert-close "trace-as-return: retval = :x value"
                (mx/item (:retval t))
                (mx/item (cm/get-value (cm/get-submap (:choices t) :x)))
                1e-6))

;; Model with multiple gen params
(let [m (gen [a b c]
          (trace :x (dist/gaussian a b))
          (trace :y (dist/gaussian b c)))
      t (p/simulate (dyn/auto-key m) [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)])]
  (assert-true "multi-param: compiles" (some? (:compiled-simulate (:schema m))))
  (assert-true "multi-param: has :x" (cm/has-value? (cm/get-submap (:choices t) :x)))
  (assert-true "multi-param: has :y" (cm/has-value? (cm/get-submap (:choices t) :y))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== L1-M2 Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed =="))
(when (pos? @fail-count)
  (println (str "FAILURES: " @fail-count)))
