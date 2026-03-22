(ns genmlx.compiled-schema-test2
  "Schema extraction: trace sites, splice sites, param sites, classification.
   Verifies extract-schema produces correct structural metadata."
  (:require [cljs.test :refer [deftest is testing are]]
            [genmlx.schema :as schema]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helper: extract schema from a model
;; ---------------------------------------------------------------------------

(defn schema-of [model] (:schema model))

;; ---------------------------------------------------------------------------
;; Single trace site
;; ---------------------------------------------------------------------------

(deftest single-trace-site-schema
  (let [{:keys [trace-sites splice-sites param-sites static?
                dynamic-addresses? has-branches?]}
        (schema-of (gen [] (trace :x (dist/gaussian 0 1))))]
    (is (= 1 (count trace-sites)))
    (is (= :x (-> trace-sites first :addr)))
    (is (= :gaussian (-> trace-sites first :dist-type)))
    (is (zero? (count splice-sites)))
    (is (zero? (count param-sites)))
    (is static?)
    (is (not dynamic-addresses?))
    (is (not has-branches?))))

;; ---------------------------------------------------------------------------
;; Multiple trace sites with dependencies
;; ---------------------------------------------------------------------------

(deftest multi-trace-with-dependencies
  (let [{:keys [trace-sites static?]}
        (schema-of (gen [x]
                        (let [slope (trace :slope (dist/gaussian 0 10))
                              intercept (trace :intercept (dist/gaussian 0 5))]
                          (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))))]
    (is (= 3 (count trace-sites)))
    (let [addrs (set (map :addr trace-sites))]
      (is (= #{:slope :intercept :y} addrs)))
    (is (every? :static? trace-sites))
    (is static?)))

;; ---------------------------------------------------------------------------
;; Distribution type recognition
;; ---------------------------------------------------------------------------

(deftest distribution-type-recognition
  (let [{:keys [trace-sites]}
        (schema-of (gen []
                        (let [a (trace :a (dist/gaussian 0 1))
                              b (trace :b (dist/bernoulli 0.5))
                              c (trace :c (dist/uniform 0 1))
                              d (trace :d (dist/exponential 1.0))]
                          [a b c d])))
        by-addr (into {} (map (juxt :addr :dist-type) trace-sites))]
    (are [addr expected-type] (= expected-type (get by-addr addr))
      :a :gaussian
      :b :bernoulli
      :c :uniform
      :d :exponential)))

;; ---------------------------------------------------------------------------
;; Splice sites
;; ---------------------------------------------------------------------------

(deftest splice-site-detection
  (let [sub (gen [] (trace :a (dist/gaussian 0 1)))
        {:keys [splice-sites]}
        (schema-of (gen [] (splice :sub sub)))]
    (is (= 1 (count splice-sites)))
    (is (= :sub (-> splice-sites first :addr)))))

;; ---------------------------------------------------------------------------
;; Param sites
;; ---------------------------------------------------------------------------

(deftest param-site-detection
  (let [{:keys [param-sites]}
        (schema-of (gen []
                        (let [w (param :weight (mx/zeros [3]))]
                          (trace :x (dist/gaussian w 1)))))]
    (is (= 1 (count param-sites)))
    (is (= :weight (-> param-sites first :name)))))

;; ---------------------------------------------------------------------------
;; Dynamic addresses (non-static)
;; ---------------------------------------------------------------------------

(deftest dynamic-address-model-not-static
  (let [source '([n] (doseq [i (range n)]
                       (trace (keyword (str "x" i)) (dist/gaussian 0 1))))
        schema (schema/extract-schema source)]
    (is (:dynamic-addresses? schema))
    (is (not (:static? schema)))))

;; ---------------------------------------------------------------------------
;; Branch detection
;; ---------------------------------------------------------------------------

(deftest branching-model-detected
  (let [{:keys [has-branches? static?]}
        (schema-of (gen [flag]
                        (if flag
                          (trace :a (dist/gaussian 0 1))
                          (trace :b (dist/gaussian 0 1)))))]
    (is has-branches?)
    (is (not static?))))

;; ---------------------------------------------------------------------------
;; Params extraction
;; ---------------------------------------------------------------------------

(deftest params-vector-captured
  (let [{:keys [params]} (schema-of (gen [x y z] (trace :a (dist/gaussian x y))))]
    (is (= '[x y z] params))))

(deftest no-params-is-empty
  (let [{:keys [params]} (schema-of (gen [] (trace :x (dist/gaussian 0 1))))]
    (is (= '[] params))))

;; ---------------------------------------------------------------------------
;; Dependency tracking
;; ---------------------------------------------------------------------------

(deftest dependency-set-tracks-trace-deps
  (let [{:keys [trace-sites]}
        (schema-of (gen []
                        (let [a (trace :a (dist/gaussian 0 1))]
                          (trace :b (dist/gaussian a 1)))))
        b-site (first (filter #(= :b (:addr %)) trace-sites))]
    (is (contains? (:deps b-site) :a)
        ":b depends on :a")))

;; ---------------------------------------------------------------------------
;; Dep-order (topological sort)
;; ---------------------------------------------------------------------------

(deftest dep-order-respects-dependencies
  (let [{:keys [dep-order]}
        (schema-of (gen []
                        (let [a (trace :a (dist/gaussian 0 1))
                              b (trace :b (dist/gaussian a 1))]
                          (trace :c (dist/gaussian b 1)))))]
    ;; :a must come before :b, :b before :c
    (let [idx (into {} (map-indexed (fn [i addr] [addr i]) dep-order))]
      (is (< (idx :a) (idx :b)))
      (is (< (idx :b) (idx :c))))))

;; ---------------------------------------------------------------------------
;; Return form captured
;; ---------------------------------------------------------------------------

(deftest return-form-is-last-body-expression
  (let [{:keys [return-form]}
        (schema-of (gen [x]
                        (let [a (trace :a (dist/gaussian 0 1))]
                          (mx/multiply a x))))]
    (is (some? return-form))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
