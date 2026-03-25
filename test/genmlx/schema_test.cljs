(ns genmlx.schema-test
  "L1-M1: Schema extraction tests.
   Verify that extract-schema correctly analyzes gen body source forms
   to extract trace sites, splice sites, param sites, and classify
   models as static vs dynamic."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.schema :as schema]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.combinators :as comb]
            [clojure.set])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================
;; Test 1: Single trace site
;; ============================================================
(deftest test-1-single-trace-site
  (testing "Single trace site"
    (let [model (gen []
                  (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
          s (:schema model)]
      (is (some? s) "schema exists")
      (is (= 1 (count (:trace-sites s))) "one trace site")
      (is (= :x (-> s :trace-sites first :addr)) "trace addr is :x")
      (is (= :gaussian (-> s :trace-sites first :dist-type)) "dist recognized as gaussian")
      (is (= 0 (count (:splice-sites s))) "no splice sites")
      (is (= 0 (count (:param-sites s))) "no param sites")
      (is (:static? s) "model is static")
      (is (not (:dynamic-addresses? s)) "no dynamic addresses")
      (is (not (:has-branches? s)) "no branches"))))

;; ============================================================
;; Test 2: Multiple trace sites with let bindings
;; ============================================================
(deftest test-2-multiple-traces-with-let
  (testing "Multiple traces with let bindings"
    (let [model (gen [x]
                  (let [slope (trace :slope (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
                        intercept (trace :intercept (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                    (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept)
                                             (mx/scalar 1)))))
          s (:schema model)]
      (is (= 3 (count (:trace-sites s))) "three trace sites")
      (let [addrs (set (map :addr (:trace-sites s)))]
        (is (contains? addrs :slope) ":slope present")
        (is (contains? addrs :intercept) ":intercept present")
        (is (contains? addrs :y) ":y present"))
      (is (every? :static? (:trace-sites s)) "all static addresses")
      (is (:static? s) "model is static"))))

;; ============================================================
;; Test 3: Distribution type recognition
;; ============================================================
(deftest test-3-distribution-type-recognition
  (testing "Distribution type recognition"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/bernoulli (mx/scalar 0.5)))
                        c (trace :c (dist/uniform (mx/scalar 0) (mx/scalar 1)))
                        d (trace :d (dist/exponential (mx/scalar 1.0)))
                        e (trace :e (dist/categorical (mx/array [0.2 0.3 0.5])))]
                    [a b c d e]))
          s (:schema model)
          types (into {} (map (fn [t] [(:addr t) (:dist-type t)]) (:trace-sites s)))]
      (is (= 5 (count (:trace-sites s))) "5 trace sites")
      (is (= :gaussian (get types :a)) ":a is gaussian")
      (is (= :bernoulli (get types :b)) ":b is bernoulli")
      (is (= :uniform (get types :c)) ":c is uniform")
      (is (= :exponential (get types :d)) ":d is exponential")
      (is (= :categorical (get types :e)) ":e is categorical"))))

;; ============================================================
;; Test 4: Computed addresses → dynamic
;; ============================================================
(deftest test-4-computed-dynamic-addresses
  (testing "Computed (dynamic) addresses"
    (let [model (gen [data]
                  (doseq [[i x] (map-indexed vector data)]
                    (trace (keyword (str "y" i))
                           (dist/gaussian x (mx/scalar 1)))))
          s (:schema model)]
      (is (:dynamic-addresses? s) "has dynamic addresses")
      (is (not (:static? s)) "model is NOT static")
      (is (pos? (count (:trace-sites s))) "has trace sites")
      (is (some #(not (:static? %)) (:trace-sites s)) "trace site marked dynamic"))))

;; ============================================================
;; Test 5: Data-dependent branching
;; ============================================================
(deftest test-5-data-dependent-branching
  (testing "Data-dependent branching"
    (let [model (gen [use-prior?]
                  (if use-prior?
                    (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    (trace :x (dist/gaussian (mx/scalar 10) (mx/scalar 1)))))
          s (:schema model)]
      (is (:has-branches? s) "has branches")
      (is (some #(= :x (:addr %)) (:trace-sites s)) "trace :x found"))))

;; ============================================================
;; Test 6: Loop with traces (doseq)
;; ============================================================
(deftest test-6-loop-with-traces
  (testing "Loop with traces"
    (let [model (gen [n]
                  (doseq [i (range n)]
                    (trace (keyword (str "x" i))
                           (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s (:schema model)]
      (is (:has-loops? s) "has loops")
      (is (not (:static? s)) "model is not static"))))

;; ============================================================
;; Test 7: Splice site detection
;; ============================================================
(deftest test-7-splice-site-detection
  (testing "Splice sites"
    (let [inner (gen [mu]
                  (trace :x (dist/gaussian mu (mx/scalar 1))))
          outer (gen [mu]
                  (let [z (trace :z (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                    (splice :obs inner z)))
          s (:schema outer)]
      (is (= 1 (count (:trace-sites s))) "one trace site")
      (is (= 1 (count (:splice-sites s))) "one splice site")
      (is (= :obs (-> s :splice-sites first :addr)) "splice addr is :obs")
      (is (-> s :splice-sites first :static?) "splice is static"))))

;; ============================================================
;; Test 8: Param site detection
;; ============================================================
(deftest test-8-param-site-detection
  (testing "Param sites"
    (let [model (gen [x]
                  (let [w (param :weight (mx/scalar 1.0))
                        b (param :bias (mx/scalar 0.0))]
                    (trace :y (dist/gaussian (mx/add (mx/multiply w x) b)
                                             (mx/scalar 1)))))
          s (:schema model)]
      (is (= 1 (count (:trace-sites s))) "one trace site")
      (is (= 2 (count (:param-sites s))) "two param sites")
      (let [param-names (set (map :name (:param-sites s)))]
        (is (contains? param-names :weight) ":weight param")
        (is (contains? param-names :bias) ":bias param")))))

;; ============================================================
;; Test 9: Empty body (no trace/splice/param)
;; ============================================================
(deftest test-9-empty-body
  (testing "Empty body (no trace sites)"
    (let [model (gen [x]
                  (mx/add x (mx/scalar 1)))
          s (:schema model)]
      (is (some? s) "schema exists")
      (is (= 0 (count (:trace-sites s))) "zero trace sites")
      (is (= 0 (count (:splice-sites s))) "zero splice sites")
      (is (= 0 (count (:param-sites s))) "zero param sites")
      (is (:static? s) "model is static"))))

;; ============================================================
;; Test 10: Nested let bindings
;; ============================================================
(deftest test-10-nested-let-bindings
  (testing "Nested let bindings"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (let [b (trace :b (dist/gaussian a (mx/scalar 1)))]
                      (let [c (trace :c (dist/gaussian b (mx/scalar 1)))]
                        c))))
          s (:schema model)]
      (is (= 3 (count (:trace-sites s))) "three trace sites")
      (is (:static? s) "all static")
      (let [addrs (mapv :addr (:trace-sites s))]
        (is (some #(= :a %) addrs) ":a found")
        (is (some #(= :b %) addrs) ":b found")
        (is (some #(= :c %) addrs) ":c found")))))

;; ============================================================
;; Test 11: cond/when with traces → branches
;; ============================================================
(deftest test-11-cond-when-with-traces
  (testing "cond/when with traces"
    (let [model (gen [mode]
                  (cond
                    (= mode 0) (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    (= mode 1) (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                    :else (trace :c (dist/gaussian (mx/scalar 10) (mx/scalar 1)))))
          s (:schema model)]
      (is (:has-branches? s) "has branches")
      (is (= 3 (count (:trace-sites s))) "three trace sites")
      (let [addrs (set (map :addr (:trace-sites s)))]
        (is (contains? addrs :a) ":a found")
        (is (contains? addrs :b) ":b found")
        (is (contains? addrs :c) ":c found")))))

;; ============================================================
;; Test 12: Traces inside map/for (HOF patterns)
;; ============================================================
(deftest test-12-traces-inside-hof
  (testing "Traces inside higher-order forms"
    (let [model (gen [xs]
                  (let [slope (trace :slope (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                    (doall
                      (map-indexed
                        (fn [i x]
                          (trace (keyword (str "y" i))
                                 (dist/gaussian (mx/multiply slope x) (mx/scalar 1))))
                        xs))))
          s (:schema model)]
      (is (pos? (count (:trace-sites s))) "has trace sites")
      (is (:dynamic-addresses? s) "has dynamic addresses")
      (is (some #(and (= :slope (:addr %)) (:static? %))
                (:trace-sites s))
          ":slope is static"))))

;; ============================================================
;; Test 13: Real-world linear regression model
;; ============================================================
(deftest test-13-linear-regression
  (testing "Linear regression model"
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
      (is (and (some :static? (:trace-sites s))
               (some #(not (:static? %)) (:trace-sites s)))
          "has both static and dynamic traces")
      (is (:dynamic-addresses? s) "has dynamic addresses")
      (is (not (:static? s)) "not fully static")
      (let [static-addrs (set (map :addr (filter :static? (:trace-sites s))))]
        (is (contains? static-addrs :slope) ":slope is static")
        (is (contains? static-addrs :intercept) ":intercept is static")
        (is (contains? static-addrs :noise) ":noise is static")))))

;; ============================================================
;; Test 14: Schema stored on DynamicGF record — all fields
;; ============================================================
(deftest test-14-schema-on-dynamic-gf
  (testing "Schema stored on DynamicGF"
    (let [model (gen [x]
                  (trace :a (dist/gaussian x (mx/scalar 1))))
          s (:schema model)]
      (is (some? s) "schema is on record")
      (is (map? s) "schema is a map")
      (is (contains? s :trace-sites) "has :trace-sites key")
      (is (contains? s :splice-sites) "has :splice-sites key")
      (is (contains? s :param-sites) "has :param-sites key")
      (is (contains? s :static?) "has :static? key")
      (is (contains? s :dynamic-addresses?) "has :dynamic-addresses? key")
      (is (contains? s :has-branches?) "has :has-branches? key")
      (is (contains? s :has-loops?) "has :has-loops? key")
      (is (contains? s :params) "has :params key")
      (is (contains? s :return-form) "has :return-form key")
      (is (contains? s :dep-order) "has :dep-order key")
      (let [ts (first (:trace-sites s))]
        (is (contains? ts :deps) "trace site has :deps")
        (is (contains? ts :dist-args) "trace site has :dist-args")
        (is (set? (:deps ts)) "deps is a set")
        (is (vector? (:dist-args ts)) "dist-args is a vector")))))

;; ============================================================
;; Test 15: Formal parameters captured
;; ============================================================
(deftest test-15-formal-parameters
  (testing "Formal parameters captured"
    (let [m1 (gen [] (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
          m2 (gen [a] (trace :x (dist/gaussian a (mx/scalar 1))))
          m3 (gen [a b c] (trace :x (dist/gaussian a b)))]
      (is (= [] (:params (:schema m1))) "zero params")
      (is (= '[a] (:params (:schema m2))) "one param")
      (is (= '[a b c] (:params (:schema m3))) "three params"))))

;; ============================================================
;; Test 16: Multiple splices
;; ============================================================
(deftest test-16-multiple-splices
  (testing "Multiple splices"
    (let [sub-a (gen [] (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
          sub-b (gen [] (trace :x (dist/gaussian (mx/scalar 10) (mx/scalar 1))))
          outer (gen []
                  (let [a (splice :part-a sub-a)
                        b (splice :part-b sub-b)]
                    (mx/add a b)))
          s (:schema outer)]
      (is (= 0 (count (:trace-sites s))) "zero trace sites")
      (is (= 2 (count (:splice-sites s))) "two splice sites")
      (let [addrs (set (map :addr (:splice-sites s)))]
        (is (contains? addrs :part-a) ":part-a found")
        (is (contains? addrs :part-b) ":part-b found")))))

;; ============================================================
;; Test 17: Mixed trace + splice
;; ============================================================
(deftest test-17-mixed-trace-splice
  (testing "Mixed trace and splice"
    (let [sub (gen [mu] (trace :x (dist/gaussian mu (mx/scalar 1))))
          model (gen []
                  (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                    (splice :obs sub mu)))
          s (:schema model)]
      (is (= 1 (count (:trace-sites s))) "one trace site")
      (is (= 1 (count (:splice-sites s))) "one splice site")
      (is (= :mu (-> s :trace-sites first :addr)) "trace is :mu")
      (is (= :obs (-> s :splice-sites first :addr)) "splice is :obs"))))

;; ============================================================
;; Test 18: when/when-not with traces → branches
;; ============================================================
(deftest test-18-when-with-traces
  (testing "when/when-not with traces"
    (let [model (gen [flag]
                  (when flag
                    (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s (:schema model)]
      (is (:has-branches? s) "has branches")
      (is (some #(= :x (:addr %)) (:trace-sites s)) "trace :x found"))))

;; ============================================================
;; Test 19: case with traces → branches
;; ============================================================
(deftest test-19-case-with-traces
  (testing "case with traces"
    (let [model (gen [k]
                  (case k
                    0 (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                    1 (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                    (trace :c (dist/gaussian (mx/scalar 10) (mx/scalar 1)))))
          s (:schema model)]
      (is (:has-branches? s) "has branches")
      (is (= 3 (count (:trace-sites s))) "three trace sites"))))

;; ============================================================
;; Test 20: Deeply nested structure
;; ============================================================
(deftest test-20-deeply-nested
  (testing "Deeply nested structure"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (let [b (if true
                              (let [c (trace :c (dist/gaussian a (mx/scalar 1)))]
                                (trace :d (dist/gaussian c (mx/scalar 1))))
                              (trace :e (dist/gaussian a (mx/scalar 1))))]
                      b)))
          s (:schema model)]
      (is (:has-branches? s) "has branches")
      (let [addrs (set (map :addr (:trace-sites s)))]
        (is (contains? addrs :a) ":a found")
        (is (contains? addrs :c) ":c found")
        (is (contains? addrs :d) ":d found")
        (is (contains? addrs :e) ":e found")))))

;; ============================================================
;; Test 21: Behavior preservation — schema doesn't change execution
;; ============================================================
(deftest test-21-behavior-preservation
  (testing "Behavior preservation"
    (let [model (dyn/auto-key
                  (gen []
                    (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                      x)))
          t1 (genmlx.protocols/simulate model [])
          t2 (genmlx.protocols/simulate model [])]
      (is (some? t1) "simulate still works")
      (is (some? (:retval t1)) "has retval")
      (is (some? (:score t1)) "has score")
      (is (some? (:choices t1)) "has choices")
      (mx/eval! (:retval t1))
      (mx/eval! (:score t1))
      (is (= [] (mx/shape (:score t1))) "score is scalar"))))

;; ============================================================
;; Test 22: Behavior preservation — generate with constraints
;; ============================================================
(deftest test-22-behavior-preservation-generate
  (testing "Behavior preservation — generate"
    (let [model (dyn/auto-key
                  (gen []
                    (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                      (trace :y (dist/gaussian x (mx/scalar 1))))))
          obs (genmlx.choicemap/set-value genmlx.choicemap/EMPTY :y (mx/scalar 3.0))
          result (genmlx.protocols/generate model [] obs)]
      (is (some? result) "generate still works")
      (is (some? (:trace result)) "has trace")
      (is (some? (:weight result)) "has weight")
      (mx/eval! (:weight result))
      (is (= [] (mx/shape (:weight result))) "weight is scalar"))))

;; ============================================================
;; Test 23: Behavior preservation — vsimulate
;; ============================================================
(deftest test-23-behavior-preservation-vsimulate
  (testing "Behavior preservation — vsimulate"
    (let [model (gen []
                  (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
          key (genmlx.mlx.random/fresh-key)
          vt (dyn/vsimulate model [] 50 key)]
      (is (some? vt) "vsimulate still works")
      (mx/eval! (:score vt))
      (is (= [50] (mx/shape (:score vt))) "score shape [50]"))))

;; ============================================================
;; Test 24: dotimes → has-loops
;; ============================================================
(deftest test-24-dotimes-with-traces
  (testing "dotimes with traces"
    (let [model (gen []
                  (dotimes [i 5]
                    (trace (keyword (str "x" i))
                           (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s (:schema model)]
      (is (:has-loops? s) "has loops")
      (is (:dynamic-addresses? s) "has dynamic addresses"))))

;; ============================================================
;; Test 25: for loop → has-loops
;; ============================================================
(deftest test-25-for-with-traces
  (testing "for with traces"
    (let [model (gen [items]
                  (doall
                    (for [item items]
                      (trace (keyword (str "obs-" item))
                             (dist/gaussian (mx/scalar 0) (mx/scalar 1))))))
          s (:schema model)]
      (is (:has-loops? s) "has loops"))))

;; ============================================================
;; Test 26: Static model with many distributions
;; ============================================================
(deftest test-26-static-many-dists
  (testing "Many distributions, all static"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        y (trace :y (dist/bernoulli (mx/scalar 0.5)))
                        z (trace :z (dist/uniform (mx/scalar -1) (mx/scalar 1)))]
                    [x y z]))
          s (:schema model)]
      (is (:static? s) "static model")
      (is (not (:dynamic-addresses? s)) "no dynamic addresses")
      (is (not (:has-branches? s)) "no branches")
      (is (not (:has-loops? s)) "no loops")
      (is (= 3 (count (:trace-sites s))) "three traces"))))

;; ============================================================
;; Test 27: Splice with dynamic address
;; ============================================================
(deftest test-27-splice-dynamic-address
  (testing "Splice with dynamic address"
    (let [sub (gen [] (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))
          model (gen [n]
                  (doseq [i (range n)]
                    (splice (keyword (str "part" i)) sub)))
          s (:schema model)]
      (is (pos? (count (:splice-sites s))) "has splice sites")
      (is (some #(not (:static? %)) (:splice-sites s)) "splice is dynamic")
      (is (:dynamic-addresses? s) "has dynamic addresses"))))

;; ============================================================
;; DEPENDENCY TRACKING TESTS
;; ============================================================

;; ============================================================
;; Test 28: Simple dependency chain a → b → c
;; ============================================================
(deftest test-28-dependency-chain
  (testing "Dependency chain a -> b -> c"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian a (mx/scalar 1)))
                        c (trace :c (dist/gaussian b (mx/scalar 1)))]
                    c))
          s (:schema model)
          deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
      (is (= #{} (deps-of :a)) ":a has no deps")
      (is (contains? (deps-of :b) :a) ":b depends on :a")
      (is (contains? (deps-of :c) :b) ":c depends on :b")
      (is (contains? (deps-of :c) :a) ":c transitively depends on :a"))))

;; ============================================================
;; Test 29: Fan-out — a feeds both b and c
;; ============================================================
(deftest test-29-fan-out
  (testing "Fan-out: a -> b, a -> c"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian a (mx/scalar 1)))
                        c (trace :c (dist/gaussian a (mx/scalar 2)))]
                    [b c]))
          s (:schema model)
          deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
      (is (= #{} (deps-of :a)) ":a has no deps")
      (is (= #{:a} (deps-of :b)) ":b depends on :a")
      (is (= #{:a} (deps-of :c)) ":c depends on :a"))))

;; ============================================================
;; Test 30: Fan-in — a, b both feed c
;; ============================================================
(deftest test-30-fan-in
  (testing "Fan-in: a, b -> c"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                        c (trace :c (dist/gaussian (mx/add a b) (mx/scalar 1)))]
                    c))
          s (:schema model)
          deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
      (is (= #{} (deps-of :a)) ":a has no deps")
      (is (= #{} (deps-of :b)) ":b has no deps")
      (is (= #{:a :b} (deps-of :c)) ":c depends on :a and :b"))))

;; ============================================================
;; Test 31: Transitive deps through non-trace binding
;; ============================================================
(deftest test-31-transitive-deps
  (testing "Transitive deps through computed binding"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                        sum (mx/add a b)
                        c (trace :c (dist/gaussian sum (mx/scalar 1)))]
                    c))
          s (:schema model)
          deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
      (is (= #{:a :b} (deps-of :c)) ":c depends on :a and :b via sum"))))

;; ============================================================
;; Test 32: Independent traces (no deps)
;; ============================================================
(deftest test-32-independent-traces
  (testing "Independent traces"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                        c (trace :c (dist/uniform (mx/scalar -1) (mx/scalar 1)))]
                    [a b c]))
          s (:schema model)
          deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
      (is (= #{} (deps-of :a)) ":a no deps")
      (is (= #{} (deps-of :b)) ":b no deps")
      (is (= #{} (deps-of :c)) ":c no deps"))))

;; ============================================================
;; Test 33: and/or with traces → branches
;; ============================================================
(deftest test-33-and-or-with-traces
  (testing "and/or with traces -> branches"
    (let [model-and (gen [flag]
                      (and flag (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          model-or (gen [flag]
                     (or flag (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s-and (:schema model-and)
          s-or (:schema model-or)]
      (is (:has-branches? s-and) "and with trace -> has-branches")
      (is (:has-branches? s-or) "or with trace -> has-branches")
      (is (some #(= :x (:addr %)) (:trace-sites s-and)) "and: trace :x found")
      (is (some #(= :x (:addr %)) (:trace-sites s-or)) "or: trace :x found"))))

;; ============================================================
;; Test 34: Splice with dependency on trace
;; ============================================================
(deftest test-34-splice-dependencies
  (testing "Splice dependencies"
    (let [sub (gen [mu] (trace :x (dist/gaussian mu (mx/scalar 1))))
          model (gen []
                  (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                    (splice :obs sub mu)))
          s (:schema model)
          splice-site (first (:splice-sites s))]
      (is (contains? (:deps splice-site) :mu) "splice :obs depends on :mu"))))

;; ============================================================
;; Test 35: Distribution args captured
;; ============================================================
(deftest test-35-dist-args-captured
  (testing "Distribution args captured"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (trace :b (dist/gaussian a (mx/scalar 2)))))
          s (:schema model)
          site-a (first (filter #(= :a (:addr %)) (:trace-sites s)))
          site-b (first (filter #(= :b (:addr %)) (:trace-sites s)))]
      (is (= 2 (count (:dist-args site-a))) ":a has 2 dist args")
      (is (= 2 (count (:dist-args site-b))) ":b has 2 dist args"))))

;; ============================================================
;; Test 36: Return form captured
;; ============================================================
(deftest test-36-return-form
  (testing "Return form"
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
      (is (and (seq? (:return-form s1))
               (= 'trace (first (:return-form s1))))
          "m1 return-form is a trace call")
      (is (and (seq? (:return-form s2))
               (= 'let (first (:return-form s2))))
          "m2 return-form is a let")
      (is (and (seq? (:return-form s3))
               (= 'trace (first (:return-form s3))))
          "m3 return-form is second trace"))))

;; ============================================================
;; Test 37: Topological order — linear chain
;; ============================================================
(deftest test-37-dep-order-linear-chain
  (testing "Dep-order: linear chain"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian a (mx/scalar 1)))
                        c (trace :c (dist/gaussian b (mx/scalar 1)))]
                    c))
          s (:schema model)
          order (:dep-order s)
          idx (fn [addr] (.indexOf order addr))]
      (is (= 3 (count order)) "dep-order has 3 elements")
      (is (< (idx :a) (idx :b)) ":a before :b")
      (is (< (idx :b) (idx :c)) ":b before :c"))))

;; ============================================================
;; Test 38: Topological order — fan-in
;; ============================================================
(deftest test-38-dep-order-fan-in
  (testing "Dep-order: fan-in"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                        c (trace :c (dist/gaussian (mx/add a b) (mx/scalar 1)))]
                    c))
          s (:schema model)
          order (:dep-order s)
          idx (fn [addr] (.indexOf order addr))]
      (is (= 3 (count order)) "dep-order has 3 elements")
      (is (< (idx :a) (idx :c)) ":a before :c")
      (is (< (idx :b) (idx :c)) ":b before :c"))))

;; ============================================================
;; Test 39: Topological order — independent traces
;; ============================================================
(deftest test-39-dep-order-independent
  (testing "Dep-order: independent traces"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                        c (trace :c (dist/uniform (mx/scalar -1) (mx/scalar 1)))]
                    [a b c]))
          s (:schema model)
          order (:dep-order s)]
      (is (= 3 (count order)) "dep-order has 3 elements")
      (is (some #(= :a %) order) ":a in order")
      (is (some #(= :b %) order) ":b in order")
      (is (some #(= :c %) order) ":c in order"))))

;; ============================================================
;; Test 40: Parameter shadowing in nested fn
;; ============================================================
(deftest test-40-parameter-shadowing
  (testing "Parameter shadowing in fn"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (map (fn [a]
                           (trace :b (dist/gaussian a (mx/scalar 1))))
                         [1 2 3])))
          s (:schema model)
          site-b (first (filter #(= :b (:addr %)) (:trace-sites s)))]
      (is (= #{} (:deps site-b)) ":b has no trace deps (a is shadowed)"))))

;; ============================================================
;; Test 41: No shadowing — fn with different param name
;; ============================================================
(deftest test-41-no-shadowing
  (testing "No shadowing — deps preserved"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (map (fn [x]
                           (trace :b (dist/gaussian a (mx/scalar 1))))
                         [1 2 3])))
          s (:schema model)
          site-b (first (filter #(= :b (:addr %)) (:trace-sites s)))]
      (is (contains? (:deps site-b) :a) ":b depends on :a"))))

;; ============================================================
;; Test 42: letfn with traces
;; ============================================================
(deftest test-42-letfn-with-traces
  (testing "letfn with traces"
    (let [model (gen []
                  (letfn [(helper [x]
                            (trace :inner (dist/gaussian x (mx/scalar 1))))]
                    (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                      (helper a))))
          s (:schema model)]
      (is (some #(= :inner (:addr %)) (:trace-sites s)) "trace :inner found")
      (is (some #(= :a (:addr %)) (:trace-sites s)) "trace :a found")
      (is (= 2 (count (:trace-sites s))) "two trace sites"))))

;; ============================================================
;; Test 43: Dep-order matches full dep graph
;; ============================================================
(deftest test-43-dep-order-complex-diamond
  (testing "Dep-order: complex diamond"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian a (mx/scalar 1)))
                        c (trace :c (dist/gaussian a (mx/scalar 1)))
                        d (trace :d (dist/gaussian (mx/add b c) (mx/scalar 1)))]
                    d))
          s (:schema model)
          order (:dep-order s)
          idx (fn [addr] (.indexOf order addr))]
      (is (= 4 (count order)) "4 in dep-order")
      (is (< (idx :a) (idx :b)) ":a before :b")
      (is (< (idx :a) (idx :c)) ":a before :c")
      (is (< (idx :b) (idx :d)) ":b before :d")
      (is (< (idx :c) (idx :d)) ":c before :d"))))

;; ============================================================
;; Test 44: and/or WITHOUT traces → no branches
;; ============================================================
(deftest test-44-and-or-without-traces
  (testing "and/or without traces -> no branches"
    (let [model (gen [a b]
                  (let [flag (and a b)]
                    (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s (:schema model)]
      (is (not (:has-branches? s)) "no branches (and doesn't contain trace)"))))

;; ============================================================
;; Test 45: Deps through multiple intermediate bindings
;; ============================================================
(deftest test-45-deps-through-intermediates
  (testing "Deps through chain of intermediates"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        x (mx/multiply a (mx/scalar 2))
                        y (mx/add x (mx/scalar 1))
                        z (mx/exp y)
                        b (trace :b (dist/gaussian z (mx/scalar 1)))]
                    b))
          s (:schema model)
          deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
      (is (contains? (deps-of :b) :a) ":b depends on :a through x->y->z"))))

;; ============================================================
;; Test 46: Nested let deps are scoped correctly
;; ============================================================
(deftest test-46-let-scoping
  (testing "Let scoping: inner bindings don't leak"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (let [inner (mx/multiply a (mx/scalar 2))]
                      (trace :b (dist/gaussian inner (mx/scalar 1))))))
          s (:schema model)
          deps-of (fn [addr] (:deps (first (filter #(= addr (:addr %)) (:trace-sites s)))))]
      (is (contains? (deps-of :b) :a) ":b depends on :a through inner"))))

;; ============================================================
;; Test 47: Verify dep-order valid for all entries
;; ============================================================
(deftest test-47-dep-order-valid
  (testing "Dep-order valid: all deps precede dependents"
    (let [model (gen []
                  (let [a (trace :a (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                        b (trace :b (dist/gaussian a (mx/scalar 1)))
                        c (trace :c (dist/gaussian (mx/scalar 5) (mx/scalar 1)))
                        d (trace :d (dist/gaussian (mx/add b c) (mx/scalar 1)))
                        e (trace :e (dist/gaussian d (mx/scalar 1)))]
                    e))
          s (:schema model)
          order (:dep-order s)
          dep-map (into {} (map (fn [ts] [(:addr ts) (:deps ts)])
                                (filter :static? (:trace-sites s))))
          addr-set (set (keys dep-map))
          valid? (loop [seen #{}
                        [addr & rest] order]
                   (if addr
                     (let [needed (clojure.set/intersection (get dep-map addr #{}) addr-set)]
                       (if (clojure.set/subset? needed seen)
                         (recur (conj seen addr) rest)
                         false))
                     true))]
      (is (= 5 (count order)) "5 in dep-order")
      (is valid? "dep-order is valid topological order"))))

;; ============================================================
;; Test 48: Behavior preservation — update
;; ============================================================
(deftest test-48-behavior-preservation-update
  (testing "Behavior preservation — update"
    (let [model (dyn/auto-key
                  (gen []
                    (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                      (trace :y (dist/gaussian x (mx/scalar 1))))))
          obs (genmlx.choicemap/set-value genmlx.choicemap/EMPTY :y (mx/scalar 3.0))
          {:keys [trace]} (genmlx.protocols/generate model [] obs)
          new-obs (genmlx.choicemap/set-value genmlx.choicemap/EMPTY :x (mx/scalar 2.0))
          result (genmlx.protocols/update model trace new-obs)]
      (is (some? result) "update still works")
      (is (some? (:trace result)) "update has trace")
      (is (some? (:weight result)) "update has weight")
      (mx/eval! (:weight result))
      (is (= [] (mx/shape (:weight result))) "update weight is scalar"))))

;; ============================================================
;; Test 49: Schema with quoted forms — should NOT walk into quotes
;; ============================================================
(deftest test-49-quoted-forms-not-walked
  (testing "Quoted forms not walked"
    (let [model (gen []
                  (let [code '(trace :phantom (dist/gaussian 0 1))]
                    (trace :real (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s (:schema model)]
      (is (= 1 (count (:trace-sites s))) "only one trace site (quoted ignored)")
      (is (= :real (-> s :trace-sites first :addr)) "trace is :real"))))

;; ============================================================
;; M3: Loop analysis tests
;; ============================================================

;; Test 50: Simple doseq with keyword-str address pattern
(deftest test-50-doseq-keyword-str-pattern
  (testing "M3: doseq with keyword-str pattern"
    (let [model (gen [xs]
                  (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                    (doseq [[j x] (map-indexed vector xs)]
                      (trace (keyword (str "y" j))
                             (dist/gaussian mu (mx/scalar 1))))
                    mu))
          s (:schema model)
          ls (first (:loop-sites s))]
      (is (= 1 (count (:loop-sites s))) "one loop-site")
      (is (= :doseq (:type ls)) "loop type is :doseq")
      (is (= 'j (:index-sym ls)) "index-sym is j")
      (is (= 'x (:element-sym ls)) "element-sym is x")
      (is (some? (:collection-form ls)) "has collection-form")
      (is (:homogeneous? ls) "homogeneous")
      (is (:rewritable? ls) "rewritable")
      (is (= [] (:rewrite-blockers ls)) "no rewrite blockers")
      (let [ts (first (:trace-sites ls))]
        (is (= :keyword-str (:type (:addr-pattern ts))) "addr-pattern type")
        (is (= "y" (:prefix (:addr-pattern ts))) "addr prefix")
        (is (= 'j (:index-sym (:addr-pattern ts))) "addr index-sym")
        (is (= :gaussian (:dist-type ts)) "dist-type gaussian")
        (is (contains? (:outer-deps ts) :mu) "outer-deps includes :mu"))
      (is (:has-loops? s) "has-loops? still true")
      (is (:dynamic-addresses? s) "dynamic-addresses? still true")
      (is (not (:static? s)) "not static"))))

;; Test 51: Linear regression (3 static prefix + 1 loop)
(deftest test-51-linreg-prefix-sites
  (testing "M3: linreg with prefix sites"
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
          s (:schema model)
          ls (first (:loop-sites s))
          static-addrs (set (map :addr (filter :static? (:trace-sites s))))]
      (is (= 1 (count (:loop-sites s))) "one loop-site")
      (is (contains? static-addrs :slope) "slope is static prefix")
      (is (contains? static-addrs :intercept) "intercept is static prefix")
      (is (contains? static-addrs :noise) "noise is static prefix")
      (is (:homogeneous? ls) "homogeneous")
      (is (:rewritable? ls) "rewritable")
      (is (= "obs" (-> ls :trace-sites first :addr-pattern :prefix)) "addr prefix is obs")
      (let [deps (-> ls :trace-sites first :outer-deps)]
        (is (contains? deps :slope) "depends on :slope")
        (is (contains? deps :intercept) "depends on :intercept")
        (is (contains? deps :noise) "depends on :noise")))))

;; Test 52: dotimes with literal count
(deftest test-52-dotimes-literal-count
  (testing "M3: dotimes with literal count"
    (let [model (gen []
                  (dotimes [i 5]
                    (trace (keyword (str "x" i))
                           (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s (:schema model)
          ls (first (:loop-sites s))]
      (is (= :dotimes (:type ls)) "loop type :dotimes")
      (is (= 5 (:count-form ls)) "count-form is 5")
      (is (= -1 (:count-arg-idx ls)) "count-arg-idx is -1")
      (is (:rewritable? ls) "rewritable"))))

;; Test 53: for loop
(deftest test-53-for-loop
  (testing "M3: for loop"
    (let [model (gen [items]
                  (doall (for [item items]
                    (trace (keyword (str "obs-" item))
                           (dist/gaussian (mx/scalar 0) (mx/scalar 1))))))
          s (:schema model)
          ls (first (:loop-sites s))]
      (is (= :for (:type ls)) "loop type :for")
      (is (= 'item (:element-sym ls)) "element-sym is item")
      (is (:rewritable? ls) "rewritable"))))

;; Test 54: Non-rewritable — mixed distribution types
(deftest test-54-mixed-dist-types
  (testing "M3: mixed dist types -> not rewritable"
    (let [model (gen [xs]
                  (doseq [[j x] (map-indexed vector xs)]
                    (trace (keyword (str "y" j)) (dist/gaussian (mx/scalar x) (mx/scalar 1)))
                    (trace (keyword (str "z" j)) (dist/bernoulli (mx/scalar 0.5)))))
          s (:schema model)
          ls (first (:loop-sites s))]
      (is (not (:homogeneous? ls)) "not homogeneous")
      (is (not (:rewritable? ls)) "not rewritable")
      (is (some #(re-find #"heterogeneous" %) (:rewrite-blockers ls))
          "blocker mentions heterogeneous"))))

;; Test 55: Non-rewritable — branch with trace in loop
(deftest test-55-branch-in-loop
  (testing "M3: branch in loop -> not rewritable"
    (let [model (gen [xs]
                  (doseq [[j x] (map-indexed vector xs)]
                    (if (pos? x)
                      (trace (keyword (str "y" j)) (dist/gaussian (mx/scalar x) (mx/scalar 1)))
                      (trace (keyword (str "y" j)) (dist/gaussian (mx/scalar 0) (mx/scalar 1))))))
          s (:schema model)
          ls (first (:loop-sites s))]
      (is (not (:rewritable? ls)) "not rewritable")
      (is (some #(re-find #"branch" %) (:rewrite-blockers ls))
          "blocker mentions branch"))))

;; Test 56: Non-rewritable — unknown address pattern
(deftest test-56-unknown-addr-pattern
  (testing "M3: unknown addr pattern -> not rewritable"
    (let [model (gen [xs]
                  (doseq [[j x] (map-indexed vector xs)]
                    (trace (nth [:a :b :c] j) (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s (:schema model)
          ls (first (:loop-sites s))]
      (is (not (:rewritable? ls)) "not rewritable")
      (is (some #(re-find #"address pattern" %) (:rewrite-blockers ls))
          "blocker mentions address pattern"))))

;; Test 57: Static model — empty loop-sites
(deftest test-57-static-no-loop-sites
  (testing "M3: static model has no loop-sites"
    (let [model (gen [x]
                  (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (trace :y (dist/gaussian mu (mx/scalar 1)))))
          s (:schema model)]
      (is (= [] (:loop-sites s)) "loop-sites is empty")
      (is (not (:has-loops? s)) "not has-loops")
      (is (:static? s) "static"))))

;; Test 58: loop/recur — has-loops but no loop-sites
(deftest test-58-loop-recur
  (testing "M3: loop/recur not analyzed"
    (let [model (gen [n]
                  (loop [i 0]
                    (when (< i n)
                      (trace (keyword (str "x" i)) (dist/gaussian (mx/scalar 0) (mx/scalar 1)))
                      (recur (inc i)))))
          s (:schema model)]
      (is (:has-loops? s) "has-loops")
      (is (= [] (:loop-sites s)) "loop-sites empty"))))

;; Test 59: Multiple loops in one model
(deftest test-59-multiple-loops
  (testing "M3: multiple loops"
    (let [model (gen [xs ys]
                  (let [mu-x (trace :mu-x (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
                        mu-y (trace :mu-y (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                    (doseq [[j x] (map-indexed vector xs)]
                      (trace (keyword (str "x" j)) (dist/gaussian mu-x (mx/scalar 1))))
                    (doseq [[j y] (map-indexed vector ys)]
                      (trace (keyword (str "y" j)) (dist/gaussian mu-y (mx/scalar 1))))
                    [mu-x mu-y]))
          s (:schema model)]
      (is (= 2 (count (:loop-sites s))) "two loop-sites")
      (let [ls0 (first (:loop-sites s))
            ls1 (second (:loop-sites s))]
        (is (= "x" (-> ls0 :trace-sites first :addr-pattern :prefix)) "first prefix")
        (is (= "y" (-> ls1 :trace-sites first :addr-pattern :prefix)) "second prefix")
        (is (contains? (-> ls0 :trace-sites first :outer-deps) :mu-x) "first deps on :mu-x")
        (is (contains? (-> ls1 :trace-sites first :outer-deps) :mu-y) "second deps on :mu-y")
        (is (and (:rewritable? ls0) (:rewritable? ls1)) "both rewritable")))))

;; Test 60: dotimes with param-derived count
(deftest test-60-dotimes-param-derived-count
  (testing "M3: dotimes count from param"
    (let [model (gen [n]
                  (dotimes [i n]
                    (trace (keyword (str "x" i)) (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
          s (:schema model)
          ls (first (:loop-sites s))]
      (is (= 'n (:count-form ls)) "count-form is n")
      (is (= 0 (:count-arg-idx ls)) "count-arg-idx is 0"))))

(cljs.test/run-tests)
