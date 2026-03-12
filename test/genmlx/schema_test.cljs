(ns genmlx.schema-test
  "L1-M1: Schema extraction tests.
   Verify that extract-schema correctly analyzes gen body source forms
   to extract trace sites, splice sites, param sites, and classify
   models as static vs dynamic."
  (:require [genmlx.schema :as schema]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.combinators :as comb]
            [clojure.set])
  (:require-macros [genmlx.gen :refer [gen]]))

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-true [msg actual]
  (if actual
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg "- expected truthy, got" actual))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg "\n    expected:" expected "\n    actual:  " actual))))

(defn assert-contains [msg coll item]
  (if (some #(= % item) coll)
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg "- expected" item "in" coll))))

(println "\n=== L1-M1: Schema Extraction Tests ===\n")

;; ============================================================
;; Test 1: Single trace site
;; ============================================================
(println "-- 1. Single trace site --")
(let [model (gen []
              (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
      s (:schema model)]
  (assert-true "schema exists" (some? s))
  (assert-equal "one trace site" 1 (count (:trace-sites s)))
  (assert-equal "trace addr is :x" :x (-> s :trace-sites first :addr))
  (assert-true "dist recognized as gaussian" (= :gaussian (-> s :trace-sites first :dist-type)))
  (assert-equal "no splice sites" 0 (count (:splice-sites s)))
  (assert-equal "no param sites" 0 (count (:param-sites s)))
  (assert-true "model is static" (:static? s))
  (assert-true "no dynamic addresses" (not (:dynamic-addresses? s)))
  (assert-true "no branches" (not (:has-branches? s))))

;; ============================================================
;; Test 2: Multiple trace sites with let bindings
;; ============================================================
(println "\n-- 2. Multiple traces with let bindings --")
(let [model (gen [x]
              (let [slope (trace :slope (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
                    intercept (trace :intercept (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept)
                                         (mx/scalar 1)))))
      s (:schema model)]
  (assert-equal "three trace sites" 3 (count (:trace-sites s)))
  (let [addrs (set (map :addr (:trace-sites s)))]
    (assert-true ":slope present" (contains? addrs :slope))
    (assert-true ":intercept present" (contains? addrs :intercept))
    (assert-true ":y present" (contains? addrs :y)))
  (assert-true "all static addresses" (every? :static? (:trace-sites s)))
  (assert-true "model is static" (:static? s)))

;; ============================================================
;; Test 3: Distribution type recognition
;; ============================================================
(println "\n-- 3. Distribution type recognition --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/bernoulli (mx/scalar 0.5)))
                    c (trace :c (dist/uniform (mx/scalar 0) (mx/scalar 1)))
                    d (trace :d (dist/exponential (mx/scalar 1.0)))
                    e (trace :e (dist/categorical (mx/array [0.2 0.3 0.5])))]
                [a b c d e]))
      s (:schema model)
      types (into {} (map (fn [t] [(:addr t) (:dist-type t)]) (:trace-sites s)))]
  (assert-equal "5 trace sites" 5 (count (:trace-sites s)))
  (assert-equal ":a is gaussian" :gaussian (get types :a))
  (assert-equal ":b is bernoulli" :bernoulli (get types :b))
  (assert-equal ":c is uniform" :uniform (get types :c))
  (assert-equal ":d is exponential" :exponential (get types :d))
  (assert-equal ":e is categorical" :categorical (get types :e)))

;; ============================================================
;; Test 4: Computed addresses → dynamic
;; ============================================================
(println "\n-- 4. Computed (dynamic) addresses --")
(let [model (gen [data]
              (doseq [[i x] (map-indexed vector data)]
                (trace (keyword (str "y" i))
                       (dist/gaussian x (mx/scalar 1)))))
      s (:schema model)]
  (assert-true "has dynamic addresses" (:dynamic-addresses? s))
  (assert-true "model is NOT static" (not (:static? s)))
  ;; Should still find the trace site, but marked as dynamic
  (assert-true "has trace sites" (pos? (count (:trace-sites s))))
  (assert-true "trace site marked dynamic" (some #(not (:static? %)) (:trace-sites s))))

;; ============================================================
;; Test 5: Data-dependent branching
;; ============================================================
(println "\n-- 5. Data-dependent branching --")
(let [model (gen [use-prior?]
              (if use-prior?
                (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                (trace :x (dist/gaussian (mx/scalar 10) (mx/scalar 1)))))
      s (:schema model)]
  (assert-true "has branches" (:has-branches? s))
  ;; Both branches have trace :x, so it should appear
  (assert-true "trace :x found" (some #(= :x (:addr %)) (:trace-sites s))))

;; ============================================================
;; Test 6: Loop with traces (doseq)
;; ============================================================
(println "\n-- 6. Loop with traces --")
(let [model (gen [n]
              (doseq [i (range n)]
                (trace (keyword (str "x" i))
                       (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
      s (:schema model)]
  (assert-true "has loops" (:has-loops? s))
  (assert-true "model is not static" (not (:static? s))))

;; ============================================================
;; Test 7: Splice site detection
;; ============================================================
(println "\n-- 7. Splice sites --")
(let [inner (gen [mu]
              (trace :x (dist/gaussian mu (mx/scalar 1))))
      outer (gen [mu]
              (let [z (trace :z (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                (splice :obs inner z)))
      s (:schema outer)]
  (assert-equal "one trace site" 1 (count (:trace-sites s)))
  (assert-equal "one splice site" 1 (count (:splice-sites s)))
  (assert-equal "splice addr is :obs" :obs (-> s :splice-sites first :addr))
  (assert-true "splice is static" (-> s :splice-sites first :static?)))

;; ============================================================
;; Test 8: Param site detection
;; ============================================================
(println "\n-- 8. Param sites --")
(let [model (gen [x]
              (let [w (param :weight (mx/scalar 1.0))
                    b (param :bias (mx/scalar 0.0))]
                (trace :y (dist/gaussian (mx/add (mx/multiply w x) b)
                                         (mx/scalar 1)))))
      s (:schema model)]
  (assert-equal "one trace site" 1 (count (:trace-sites s)))
  (assert-equal "two param sites" 2 (count (:param-sites s)))
  (let [param-names (set (map :name (:param-sites s)))]
    (assert-true ":weight param" (contains? param-names :weight))
    (assert-true ":bias param" (contains? param-names :bias))))

;; ============================================================
;; Test 9: Empty body (no trace/splice/param)
;; ============================================================
(println "\n-- 9. Empty body (no trace sites) --")
(let [model (gen [x]
              (mx/add x (mx/scalar 1)))
      s (:schema model)]
  (assert-true "schema exists" (some? s))
  (assert-equal "zero trace sites" 0 (count (:trace-sites s)))
  (assert-equal "zero splice sites" 0 (count (:splice-sites s)))
  (assert-equal "zero param sites" 0 (count (:param-sites s)))
  (assert-true "model is static" (:static? s)))

;; ============================================================
;; Test 10: Nested let bindings
;; ============================================================
(println "\n-- 10. Nested let bindings --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                (let [b (trace :b (dist/gaussian a (mx/scalar 1)))]
                  (let [c (trace :c (dist/gaussian b (mx/scalar 1)))]
                    c))))
      s (:schema model)]
  (assert-equal "three trace sites" 3 (count (:trace-sites s)))
  (assert-true "all static" (:static? s))
  (let [addrs (mapv :addr (:trace-sites s))]
    (assert-true ":a found" (some #(= :a %) addrs))
    (assert-true ":b found" (some #(= :b %) addrs))
    (assert-true ":c found" (some #(= :c %) addrs))))

;; ============================================================
;; Test 11: cond/when with traces → branches
;; ============================================================
(println "\n-- 11. cond/when with traces --")
(let [model (gen [mode]
              (cond
                (= mode 0) (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                (= mode 1) (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                :else (trace :c (dist/gaussian (mx/scalar 10) (mx/scalar 1)))))
      s (:schema model)]
  (assert-true "has branches" (:has-branches? s))
  (assert-equal "three trace sites" 3 (count (:trace-sites s)))
  (let [addrs (set (map :addr (:trace-sites s)))]
    (assert-true ":a found" (contains? addrs :a))
    (assert-true ":b found" (contains? addrs :b))
    (assert-true ":c found" (contains? addrs :c))))

;; ============================================================
;; Test 12: Traces inside map/for (HOF patterns)
;; ============================================================
(println "\n-- 12. Traces inside higher-order forms --")
(let [model (gen [xs]
              (let [slope (trace :slope (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                (doall
                  (map-indexed
                    (fn [i x]
                      (trace (keyword (str "y" i))
                             (dist/gaussian (mx/multiply slope x) (mx/scalar 1))))
                    xs))))
      s (:schema model)]
  ;; :slope is static, but the map-indexed traces are dynamic
  (assert-true "has trace sites" (pos? (count (:trace-sites s))))
  (assert-true "has dynamic addresses" (:dynamic-addresses? s))
  ;; :slope should be found as static
  (assert-true ":slope is static" (some #(and (= :slope (:addr %)) (:static? %))
                                        (:trace-sites s))))

;; ============================================================
;; Test 13: Real-world linear regression model
;; ============================================================
(println "\n-- 13. Linear regression model --")
(let [model (gen [xs ys]
              (let [slope (trace :slope (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
                    intercept (trace :intercept (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
                    noise (trace :noise (dist/exponential (mx/scalar 1)))]
                (doseq [[j [x y]] (map-indexed vector (map vector xs ys))]
                  (trace (keyword (str "obs" j))
                         (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                                intercept)
                                        noise)))
                slope))
      s (:schema model)]
  ;; Static params + dynamic observations
  (assert-true "has both static and dynamic traces"
               (and (some :static? (:trace-sites s))
                    (some #(not (:static? %)) (:trace-sites s))))
  (assert-true "has dynamic addresses" (:dynamic-addresses? s))
  (assert-true "not fully static" (not (:static? s)))
  ;; Known static addresses
  (let [static-addrs (set (map :addr (filter :static? (:trace-sites s))))]
    (assert-true ":slope is static" (contains? static-addrs :slope))
    (assert-true ":intercept is static" (contains? static-addrs :intercept))
    (assert-true ":noise is static" (contains? static-addrs :noise))))

;; ============================================================
;; Test 14: Schema stored on DynamicGF record — all fields
;; ============================================================
(println "\n-- 14. Schema stored on DynamicGF --")
(let [model (gen [x]
              (trace :a (dist/gaussian x (mx/scalar 1))))
      s (:schema model)]
  (assert-true "schema is on record" (some? s))
  (assert-true "schema is a map" (map? s))
  (assert-true "has :trace-sites key" (contains? s :trace-sites))
  (assert-true "has :splice-sites key" (contains? s :splice-sites))
  (assert-true "has :param-sites key" (contains? s :param-sites))
  (assert-true "has :static? key" (contains? s :static?))
  (assert-true "has :dynamic-addresses? key" (contains? s :dynamic-addresses?))
  (assert-true "has :has-branches? key" (contains? s :has-branches?))
  (assert-true "has :has-loops? key" (contains? s :has-loops?))
  (assert-true "has :params key" (contains? s :params))
  ;; New fields
  (assert-true "has :return-form key" (contains? s :return-form))
  (assert-true "has :dep-order key" (contains? s :dep-order))
  ;; Trace site has deps and dist-args
  (let [ts (first (:trace-sites s))]
    (assert-true "trace site has :deps" (contains? ts :deps))
    (assert-true "trace site has :dist-args" (contains? ts :dist-args))
    (assert-true "deps is a set" (set? (:deps ts)))
    (assert-true "dist-args is a vector" (vector? (:dist-args ts)))))

;; ============================================================
;; Test 15: Formal parameters captured
;; ============================================================
(println "\n-- 15. Formal parameters captured --")
(let [m1 (gen [] (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
      m2 (gen [a] (trace :x (dist/gaussian a (mx/scalar 1))))
      m3 (gen [a b c] (trace :x (dist/gaussian a b)))]
  (assert-equal "zero params" [] (:params (:schema m1)))
  (assert-equal "one param" '[a] (:params (:schema m2)))
  (assert-equal "three params" '[a b c] (:params (:schema m3))))

;; ============================================================
;; Test 16: Multiple splices
;; ============================================================
(println "\n-- 16. Multiple splices --")
(let [sub-a (gen [] (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
      sub-b (gen [] (trace :x (dist/gaussian (mx/scalar 10) (mx/scalar 1))))
      outer (gen []
              (let [a (splice :part-a sub-a)
                    b (splice :part-b sub-b)]
                (mx/add a b)))
      s (:schema outer)]
  (assert-equal "zero trace sites" 0 (count (:trace-sites s)))
  (assert-equal "two splice sites" 2 (count (:splice-sites s)))
  (let [addrs (set (map :addr (:splice-sites s)))]
    (assert-true ":part-a found" (contains? addrs :part-a))
    (assert-true ":part-b found" (contains? addrs :part-b))))

;; ============================================================
;; Test 17: Mixed trace + splice
;; ============================================================
(println "\n-- 17. Mixed trace and splice --")
(let [sub (gen [mu] (trace :x (dist/gaussian mu (mx/scalar 1))))
      model (gen []
              (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                (splice :obs sub mu)))
      s (:schema model)]
  (assert-equal "one trace site" 1 (count (:trace-sites s)))
  (assert-equal "one splice site" 1 (count (:splice-sites s)))
  (assert-equal "trace is :mu" :mu (-> s :trace-sites first :addr))
  (assert-equal "splice is :obs" :obs (-> s :splice-sites first :addr)))

;; ============================================================
;; Test 18: when/when-not with traces → branches
;; ============================================================
(println "\n-- 18. when/when-not with traces --")
(let [model (gen [flag]
              (when flag
                (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
      s (:schema model)]
  (assert-true "has branches" (:has-branches? s))
  (assert-true "trace :x found" (some #(= :x (:addr %)) (:trace-sites s))))

;; ============================================================
;; Test 19: case with traces → branches
;; ============================================================
(println "\n-- 19. case with traces --")
(let [model (gen [k]
              (case k
                0 (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                1 (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                (trace :c (dist/gaussian (mx/scalar 10) (mx/scalar 1)))))
      s (:schema model)]
  (assert-true "has branches" (:has-branches? s))
  (assert-equal "three trace sites" 3 (count (:trace-sites s))))

;; ============================================================
;; Test 20: Deeply nested structure
;; ============================================================
(println "\n-- 20. Deeply nested structure --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                (let [b (if true
                          (let [c (trace :c (dist/gaussian a (mx/scalar 1)))]
                            (trace :d (dist/gaussian c (mx/scalar 1))))
                          (trace :e (dist/gaussian a (mx/scalar 1))))]
                  b)))
      s (:schema model)]
  (assert-true "has branches" (:has-branches? s))
  ;; Should find all trace sites regardless of nesting
  (let [addrs (set (map :addr (:trace-sites s)))]
    (assert-true ":a found" (contains? addrs :a))
    (assert-true ":c found" (contains? addrs :c))
    (assert-true ":d found" (contains? addrs :d))
    (assert-true ":e found" (contains? addrs :e))))

;; ============================================================
;; Test 21: Behavior preservation — schema doesn't change execution
;; ============================================================
(println "\n-- 21. Behavior preservation --")
(let [model (dyn/auto-key
              (gen []
                (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                  x)))
      ;; Must still work identically to before
      t1 (genmlx.protocols/simulate model [])
      t2 (genmlx.protocols/simulate model [])]
  (assert-true "simulate still works" (some? t1))
  (assert-true "has retval" (some? (:retval t1)))
  (assert-true "has score" (some? (:score t1)))
  (assert-true "has choices" (some? (:choices t1)))
  (mx/eval! (:retval t1))
  (mx/eval! (:score t1))
  (assert-true "score is scalar" (= [] (mx/shape (:score t1)))))

;; ============================================================
;; Test 22: Behavior preservation — generate with constraints
;; ============================================================
(println "\n-- 22. Behavior preservation — generate --")
(let [model (dyn/auto-key
              (gen []
                (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                  (trace :y (dist/gaussian x (mx/scalar 1))))))
      obs (genmlx.choicemap/set-value genmlx.choicemap/EMPTY :y (mx/scalar 3.0))
      result (genmlx.protocols/generate model [] obs)]
  (assert-true "generate still works" (some? result))
  (assert-true "has trace" (some? (:trace result)))
  (assert-true "has weight" (some? (:weight result)))
  (mx/eval! (:weight result))
  (assert-true "weight is scalar" (= [] (mx/shape (:weight result)))))

;; ============================================================
;; Test 23: Behavior preservation — vsimulate
;; ============================================================
(println "\n-- 23. Behavior preservation — vsimulate --")
(let [model (gen []
              (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
      key (genmlx.mlx.random/fresh-key)
      vt (dyn/vsimulate model [] 50 key)]
  (assert-true "vsimulate still works" (some? vt))
  (mx/eval! (:score vt))
  (assert-equal "score shape [50]" [50] (mx/shape (:score vt))))

;; ============================================================
;; Test 24: dotimes → has-loops
;; ============================================================
(println "\n-- 24. dotimes with traces --")
(let [model (gen []
              (dotimes [i 5]
                (trace (keyword (str "x" i))
                       (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
      s (:schema model)]
  (assert-true "has loops" (:has-loops? s))
  (assert-true "has dynamic addresses" (:dynamic-addresses? s)))

;; ============================================================
;; Test 25: for loop → has-loops
;; ============================================================
(println "\n-- 25. for with traces --")
(let [model (gen [items]
              (doall
                (for [item items]
                  (trace (keyword (str "obs-" item))
                         (dist/gaussian (mx/scalar 0) (mx/scalar 1))))))
      s (:schema model)]
  (assert-true "has loops" (:has-loops? s)))

;; ============================================================
;; Test 26: Static model with many distributions
;; ============================================================
(println "\n-- 26. Many distributions, all static --")
(let [model (gen []
              (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    y (trace :y (dist/bernoulli (mx/scalar 0.5)))
                    z (trace :z (dist/uniform (mx/scalar -1) (mx/scalar 1)))]
                [x y z]))
      s (:schema model)]
  (assert-true "static model" (:static? s))
  (assert-true "no dynamic addresses" (not (:dynamic-addresses? s)))
  (assert-true "no branches" (not (:has-branches? s)))
  (assert-true "no loops" (not (:has-loops? s)))
  (assert-equal "three traces" 3 (count (:trace-sites s))))

;; ============================================================
;; Test 27: Splice with dynamic address
;; ============================================================
(println "\n-- 27. Splice with dynamic address --")
(let [sub (gen [] (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
      model (gen [n]
              (doseq [i (range n)]
                (splice (keyword (str "part" i)) sub)))
      s (:schema model)]
  (assert-true "has splice sites" (pos? (count (:splice-sites s))))
  (assert-true "splice is dynamic" (some #(not (:static? %)) (:splice-sites s)))
  (assert-true "has dynamic addresses" (:dynamic-addresses? s)))

;; ============================================================
;; DEPENDENCY TRACKING TESTS
;; ============================================================

;; ============================================================
;; Test 28: Simple dependency chain a → b → c
;; ============================================================
(println "\n-- 28. Dependency chain a → b → c --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian a (mx/scalar 1)))
                    c (trace :c (dist/gaussian b (mx/scalar 1)))]
                c))
      s (:schema model)
      deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
  (assert-equal ":a has no deps" #{} (deps-of :a))
  (assert-true ":b depends on :a" (contains? (deps-of :b) :a))
  (assert-true ":c depends on :b" (contains? (deps-of :c) :b))
  (assert-true ":c transitively depends on :a" (contains? (deps-of :c) :a)))

;; ============================================================
;; Test 29: Fan-out — a feeds both b and c
;; ============================================================
(println "\n-- 29. Fan-out: a → b, a → c --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian a (mx/scalar 1)))
                    c (trace :c (dist/gaussian a (mx/scalar 2)))]
                [b c]))
      s (:schema model)
      deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
  (assert-equal ":a has no deps" #{} (deps-of :a))
  (assert-equal ":b depends on :a" #{:a} (deps-of :b))
  (assert-equal ":c depends on :a" #{:a} (deps-of :c)))

;; ============================================================
;; Test 30: Fan-in — a, b both feed c
;; ============================================================
(println "\n-- 30. Fan-in: a, b → c --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                    c (trace :c (dist/gaussian (mx/add a b) (mx/scalar 1)))]
                c))
      s (:schema model)
      deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
  (assert-equal ":a has no deps" #{} (deps-of :a))
  (assert-equal ":b has no deps" #{} (deps-of :b))
  (assert-equal ":c depends on :a and :b" #{:a :b} (deps-of :c)))

;; ============================================================
;; Test 31: Transitive deps through non-trace binding
;; ============================================================
(println "\n-- 31. Transitive deps through computed binding --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                    sum (mx/add a b)  ;; non-trace binding
                    c (trace :c (dist/gaussian sum (mx/scalar 1)))]
                c))
      s (:schema model)
      deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
  (assert-equal ":c depends on :a and :b via sum" #{:a :b} (deps-of :c)))

;; ============================================================
;; Test 32: Independent traces (no deps)
;; ============================================================
(println "\n-- 32. Independent traces --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                    c (trace :c (dist/uniform (mx/scalar -1) (mx/scalar 1)))]
                [a b c]))
      s (:schema model)
      deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
  (assert-equal ":a no deps" #{} (deps-of :a))
  (assert-equal ":b no deps" #{} (deps-of :b))
  (assert-equal ":c no deps" #{} (deps-of :c)))

;; ============================================================
;; Test 33: and/or with traces → branches
;; ============================================================
(println "\n-- 33. and/or with traces → branches --")
(let [model-and (gen [flag]
                  (and flag (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
      model-or (gen [flag]
                 (or flag (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
      s-and (:schema model-and)
      s-or (:schema model-or)]
  (assert-true "and with trace → has-branches" (:has-branches? s-and))
  (assert-true "or with trace → has-branches" (:has-branches? s-or))
  (assert-true "and: trace :x found" (some #(= :x (:addr %)) (:trace-sites s-and)))
  (assert-true "or: trace :x found" (some #(= :x (:addr %)) (:trace-sites s-or))))

;; ============================================================
;; Test 34: Splice with dependency on trace
;; ============================================================
(println "\n-- 34. Splice dependencies --")
(let [sub (gen [mu] (trace :x (dist/gaussian mu (mx/scalar 1))))
      model (gen []
              (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                (splice :obs sub mu)))
      s (:schema model)
      splice-site (first (:splice-sites s))]
  (assert-true "splice :obs depends on :mu" (contains? (:deps splice-site) :mu)))

;; ============================================================
;; Test 35: Distribution args captured
;; ============================================================
(println "\n-- 35. Distribution args captured --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                (trace :b (dist/gaussian a (mx/scalar 2)))))
      s (:schema model)
      site-a (first (filter #(= :a (:addr %)) (:trace-sites s)))
      site-b (first (filter #(= :b (:addr %)) (:trace-sites s)))]
  (assert-equal ":a has 2 dist args" 2 (count (:dist-args site-a)))
  (assert-equal ":b has 2 dist args" 2 (count (:dist-args site-b))))

;; ============================================================
;; Test 36: Return form captured
;; ============================================================
(println "\n-- 36. Return form --")
(let [m1 (gen [] (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
      m2 (gen []
            (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
              a))
      m3 (gen []
            (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
            (trace :b (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
      s1 (:schema m1)
      s2 (:schema m2)
      s3 (:schema m3)]
  (assert-true "m1 return-form is a trace call" (and (seq? (:return-form s1))
                                                      (= 'trace (first (:return-form s1)))))
  (assert-true "m2 return-form is a let" (and (seq? (:return-form s2))
                                               (= 'let (first (:return-form s2)))))
  ;; m3 has two body forms; return-form is the last one
  (assert-true "m3 return-form is second trace" (and (seq? (:return-form s3))
                                                      (= 'trace (first (:return-form s3))))))

;; ============================================================
;; Test 37: Topological order — linear chain
;; ============================================================
(println "\n-- 37. Dep-order: linear chain --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian a (mx/scalar 1)))
                    c (trace :c (dist/gaussian b (mx/scalar 1)))]
                c))
      s (:schema model)
      order (:dep-order s)
      idx (fn [addr] (.indexOf order addr))]
  (assert-true "dep-order has 3 elements" (= 3 (count order)))
  (assert-true ":a before :b" (< (idx :a) (idx :b)))
  (assert-true ":b before :c" (< (idx :b) (idx :c))))

;; ============================================================
;; Test 38: Topological order — fan-in
;; ============================================================
(println "\n-- 38. Dep-order: fan-in --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                    c (trace :c (dist/gaussian (mx/add a b) (mx/scalar 1)))]
                c))
      s (:schema model)
      order (:dep-order s)
      idx (fn [addr] (.indexOf order addr))]
  (assert-true "dep-order has 3 elements" (= 3 (count order)))
  (assert-true ":a before :c" (< (idx :a) (idx :c)))
  (assert-true ":b before :c" (< (idx :b) (idx :c))))

;; ============================================================
;; Test 39: Topological order — independent traces
;; ============================================================
(println "\n-- 39. Dep-order: independent traces --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                    c (trace :c (dist/uniform (mx/scalar -1) (mx/scalar 1)))]
                [a b c]))
      s (:schema model)
      order (:dep-order s)]
  ;; All should be present; order doesn't matter
  (assert-equal "dep-order has 3 elements" 3 (count order))
  (assert-true ":a in order" (some #(= :a %) order))
  (assert-true ":b in order" (some #(= :b %) order))
  (assert-true ":c in order" (some #(= :c %) order)))

;; ============================================================
;; Test 40: Parameter shadowing in nested fn
;; ============================================================
(println "\n-- 40. Parameter shadowing in fn --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                ;; Inside the fn, 'a' is a parameter, NOT the trace result
                (map (fn [a]
                       (trace :b (dist/gaussian a (mx/scalar 1))))
                     [1 2 3])))
      s (:schema model)
      site-b (first (filter #(= :b (:addr %)) (:trace-sites s)))]
  ;; :b should NOT depend on :a because 'a' is shadowed by fn param
  (assert-equal ":b has no trace deps (a is shadowed)" #{} (:deps site-b)))

;; ============================================================
;; Test 41: No shadowing — fn with different param name
;; ============================================================
(println "\n-- 41. No shadowing — deps preserved --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                (map (fn [x]
                       (trace :b (dist/gaussian a (mx/scalar 1))))
                     [1 2 3])))
      s (:schema model)
      site-b (first (filter #(= :b (:addr %)) (:trace-sites s)))]
  ;; :b SHOULD depend on :a because 'a' is NOT shadowed
  (assert-true ":b depends on :a" (contains? (:deps site-b) :a)))

;; ============================================================
;; Test 42: letfn with traces
;; ============================================================
(println "\n-- 42. letfn with traces --")
(let [model (gen []
              (letfn [(helper [x]
                        (trace :inner (dist/gaussian x (mx/scalar 1))))]
                (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                  (helper a))))
      s (:schema model)]
  (assert-true "trace :inner found" (some #(= :inner (:addr %)) (:trace-sites s)))
  (assert-true "trace :a found" (some #(= :a (:addr %)) (:trace-sites s)))
  (assert-equal "two trace sites" 2 (count (:trace-sites s))))

;; ============================================================
;; Test 43: Dep-order matches full dep graph
;; ============================================================
(println "\n-- 43. Dep-order: complex diamond --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian a (mx/scalar 1)))
                    c (trace :c (dist/gaussian a (mx/scalar 1)))
                    d (trace :d (dist/gaussian (mx/add b c) (mx/scalar 1)))]
                d))
      s (:schema model)
      order (:dep-order s)
      idx (fn [addr] (.indexOf order addr))]
  (assert-equal "4 in dep-order" 4 (count order))
  (assert-true ":a before :b" (< (idx :a) (idx :b)))
  (assert-true ":a before :c" (< (idx :a) (idx :c)))
  (assert-true ":b before :d" (< (idx :b) (idx :d)))
  (assert-true ":c before :d" (< (idx :c) (idx :d))))

;; ============================================================
;; Test 44: and/or WITHOUT traces → no branches
;; ============================================================
(println "\n-- 44. and/or without traces → no branches --")
(let [model (gen [a b]
              (let [flag (and a b)]
                (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
      s (:schema model)]
  (assert-true "no branches (and doesn't contain trace)" (not (:has-branches? s))))

;; ============================================================
;; Test 45: Deps through multiple intermediate bindings
;; ============================================================
(println "\n-- 45. Deps through chain of intermediates --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    x (mx/multiply a (mx/scalar 2))
                    y (mx/add x (mx/scalar 1))
                    z (mx/exp y)
                    b (trace :b (dist/gaussian z (mx/scalar 1)))]
                b))
      s (:schema model)
      deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
  (assert-true ":b depends on :a through x→y→z" (contains? (deps-of :b) :a)))

;; ============================================================
;; Test 46: Nested let deps are scoped correctly
;; ============================================================
(println "\n-- 46. Let scoping: inner bindings don't leak --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                (let [inner (mx/multiply a (mx/scalar 2))]
                  (trace :b (dist/gaussian inner (mx/scalar 1)))))
              ;; If we had another form here, 'inner' shouldn't be in scope
              ;; but since gen bodies execute sequentially, this is fine
              )
      s (:schema model)
      deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
  (assert-true ":b depends on :a through inner" (contains? (deps-of :b) :a)))

;; ============================================================
;; Test 47: Verify dep-order valid for all entries
;; ============================================================
(println "\n-- 47. Dep-order valid: all deps precede dependents --")
(let [model (gen []
              (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    b (trace :b (dist/gaussian a (mx/scalar 1)))
                    c (trace :c (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                    d (trace :d (dist/gaussian (mx/add b c) (mx/scalar 1)))
                    e (trace :e (dist/gaussian d (mx/scalar 1)))]
                e))
      s (:schema model)
      order (:dep-order s)
      ;; Build addr→deps map
      dep-map (into {} (map (fn [ts] [(:addr ts) (:deps ts)])
                            (filter :static? (:trace-sites s))))
      ;; Verify: for each addr in order, all its deps (within addr-set) appear earlier
      addr-set (set (keys dep-map))
      valid? (loop [seen #{}
                    [addr & rest] order]
               (if addr
                 (let [needed (clojure.set/intersection (get dep-map addr #{}) addr-set)]
                   (if (clojure.set/subset? needed seen)
                     (recur (conj seen addr) rest)
                     false))
                 true))]
  (assert-equal "5 in dep-order" 5 (count order))
  (assert-true "dep-order is valid topological order" valid?))

;; ============================================================
;; Test 48: Behavior preservation — update
;; ============================================================
(println "\n-- 48. Behavior preservation — update --")
(let [model (dyn/auto-key
              (gen []
                (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                  (trace :y (dist/gaussian x (mx/scalar 1))))))
      obs (genmlx.choicemap/set-value genmlx.choicemap/EMPTY :y (mx/scalar 3.0))
      {:keys [trace]} (genmlx.protocols/generate model [] obs)
      new-obs (genmlx.choicemap/set-value genmlx.choicemap/EMPTY :x (mx/scalar 2.0))
      result (genmlx.protocols/update model trace new-obs)]
  (assert-true "update still works" (some? result))
  (assert-true "update has trace" (some? (:trace result)))
  (assert-true "update has weight" (some? (:weight result)))
  (mx/eval! (:weight result))
  (assert-true "update weight is scalar" (= [] (mx/shape (:weight result)))))

;; ============================================================
;; Test 49: Schema with quoted forms — should NOT walk into quotes
;; ============================================================
(println "\n-- 49. Quoted forms not walked --")
(let [model (gen []
              (let [code '(trace :phantom (dist/gaussian 0 1))]
                (trace :real (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
      s (:schema model)]
  (assert-equal "only one trace site (quoted ignored)" 1 (count (:trace-sites s)))
  (assert-equal "trace is :real" :real (-> s :trace-sites first :addr)))

;; ============================================================
;; Summary
;; ============================================================
(println "\n=== Schema Test Results ===")
(println "  Passed:" @pass-count)
(println "  Failed:" @fail-count)
(println (if (zero? @fail-count) "  ALL PASS" "  *** FAILURES ***"))
