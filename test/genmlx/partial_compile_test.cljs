(ns genmlx.partial-compile-test
  "L1-M3 tests: partial compilation for dynamic models.

   Tests cover:
   1. Prefix extraction (extract-prefix-sites)
   2. Prefix gates (splice/param -> no prefix; static -> L1-M2 not M3)
   3. Single-prefix-site models
   4. Multi-prefix-site models
   5. Dependent prefix sites (b depends on a)
   6. Prefix with gen args (dist-args reference gen params)
   7. Value equivalence (compiled vs handler, same key)
   8. Score equivalence
   9. PRNG consistency (dynamic site values match handler)
   10. GFI operations (generate/update/regenerate work via handler)
   11. Non-compilable cutoff (unsupported dist truncates prefix)
   12. All distributions in prefix (each noise transform type)
   13. Edge cases"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.schema :as schema]
            [genmlx.handler :as handler]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Test utilities
;; ---------------------------------------------------------------------------

(defn- force-handler
  "Return a copy of gf that always uses the handler path (no compiled paths)."
  [gf]
  (dyn/->DynamicGF (:body-fn gf) (:source gf)
                    (dissoc (:schema gf) :compiled-simulate :compiled-prefix
                                         :compiled-prefix-addrs)))

(defn- cv
  "Get choice value at addr as JS number."
  [trace addr]
  (mx/item (cm/get-value (cm/get-submap (:choices trace) addr))))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

;; M1: Loop model — 2 static priors + doseq loop
(def loop-model
  (gen [xs]
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/exponential 1))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j)) (dist/gaussian mu sigma)))
      mu)))

;; M2: Branch model — 1 static prior + if branch
(def branch-model
  (gen [flag]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (if flag
        (trace :a (dist/gaussian mu 1))
        (trace :b (dist/gaussian mu 2))))))

;; M3: Single prefix site + loop
(def single-prefix-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      (doseq [i (range 3)]
        (trace (keyword (str "y" i)) (dist/gaussian x 1)))
      x)))

;; M4: Multi-prefix with dependencies — b depends on a
(def dep-prefix-model
  (gen [x]
    (let [a (trace :a (dist/gaussian x 1))
          b (trace :b (dist/gaussian a 2))]
      (doseq [i (range 2)]
        (trace (keyword (str "y" i)) (dist/gaussian b 1)))
      b)))

;; M5: Fully static model (should get L1-M2 not L1-M3)
(def static-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))
          y (trace :y (dist/gaussian x 1))]
      y)))

;; M6: Cutoff model — gaussian prefix, then beta-dist (unsupported), then loop
(def cutoff-model
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/beta-dist 2 3))]
      (doseq [i (range 2)]
        (trace (keyword (str "y" i)) (dist/gaussian a 1)))
      a)))

;; M7: No compilable prefix (beta-dist first)
(def no-compilable-prefix-model
  (gen []
    (let [a (trace :a (dist/beta-dist 2 3))]
      (doseq [i (range 2)]
        (trace (keyword (str "y" i)) (dist/gaussian a 1)))
      a)))

;; M8: All-dynamic model (no prefix at all)
(def all-dynamic-model
  (gen [n]
    (let [result (atom 0)]
      (doseq [i (range n)]
        (let [v (trace (keyword (str "x" i)) (dist/gaussian 0 1))]
          (reset! result v)))
      @result)))

;; M9: Multi-distribution prefix (tests all noise transform types)
(def multi-dist-prefix
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/uniform 0 1))
          c (trace :c (dist/bernoulli 0.5))
          d (trace :d (dist/exponential 2))
          e (trace :e (dist/laplace 0 1))
          f (trace :f (dist/cauchy 0 1))]
      (doseq [i (range 2)]
        (trace (keyword (str "y" i)) (dist/gaussian a 1)))
      a)))

;; M10: Prefix with gen args referenced in dist-args
(def args-prefix-model
  (gen [mu sigma]
    (let [x (trace :x (dist/gaussian mu sigma))
          y (trace :y (dist/gaussian x 1))]
      (doseq [i (range 2)]
        (trace (keyword (str "z" i)) (dist/gaussian y 1)))
      y)))

;; M11: Log-normal + delta in prefix
(def lognormal-delta-prefix
  (gen []
    (let [a (trace :a (dist/log-normal 0 1))
          b (trace :b (dist/delta (mx/scalar 42.0)))]
      (doseq [i (range 2)]
        (trace (keyword (str "y" i)) (dist/gaussian a 1)))
      a)))

;; ---------------------------------------------------------------------------
;; 1. Prefix extraction
;; ---------------------------------------------------------------------------

(deftest prefix-extraction-test
  (testing "loop model: 2 prefix sites before doseq"
    (let [sites (compiled/extract-prefix-sites (:source loop-model))]
      (is (= 2 (count sites)) "loop-model has 2 prefix sites")
      (is (= :mu (:addr (first sites))) "first prefix addr is :mu")
      (is (= :sigma (:addr (second sites))) "second prefix addr is :sigma")
      (is (= :gaussian (:dist-type (first sites))) "mu dist-type is gaussian")
      (is (= :exponential (:dist-type (second sites))) "sigma dist-type is exponential")))

  (testing "branch model: 1 prefix site before if"
    (let [sites (compiled/extract-prefix-sites (:source branch-model))]
      (is (= 1 (count sites)) "branch-model has 1 prefix site")
      (is (= :mu (:addr (first sites))) "prefix addr is :mu")))

  (testing "single prefix model: 1 prefix site before doseq"
    (let [sites (compiled/extract-prefix-sites (:source single-prefix-model))]
      (is (= 1 (count sites)) "single-prefix has 1 site")
      (is (= :x (:addr (first sites))) "prefix addr is :x")))

  (testing "all-dynamic: 0 prefix sites"
    (let [sites (compiled/extract-prefix-sites (:source all-dynamic-model))]
      (is (= 0 (count sites)) "all-dynamic has 0 prefix sites")))

  (testing "dependent prefix: 2 prefix sites"
    (let [sites (compiled/extract-prefix-sites (:source dep-prefix-model))]
      (is (= 2 (count sites)) "dep-prefix has 2 prefix sites")
      (is (= :a (:addr (first sites))) "first is :a")
      (is (= :b (:addr (second sites))) "second is :b")))

  (testing "multi-dist prefix: 6 prefix sites"
    (let [sites (compiled/extract-prefix-sites (:source multi-dist-prefix))]
      (is (= 6 (count sites)) "multi-dist has 6 prefix sites"))))

;; ---------------------------------------------------------------------------
;; 2. Prefix gates
;; ---------------------------------------------------------------------------

(deftest prefix-gates-test
  (testing "static model gets compiled-simulate (L1-M2), NOT compiled-prefix"
    (is (some? (:compiled-simulate (:schema static-model)))
        "static model has compiled-simulate")
    (is (nil? (:compiled-prefix (:schema static-model)))
        "static model has NO compiled-prefix"))

  (testing "loop model gets compiled-prefix (non-static with prefix sites)"
    (is (some? (:compiled-prefix (:schema loop-model)))
        "loop model has compiled-prefix")
    (is (nil? (:compiled-simulate (:schema loop-model)))
        "loop model has NO compiled-simulate"))

  (testing "all-dynamic model: no compiled paths"
    (is (nil? (:compiled-prefix (:schema all-dynamic-model)))
        "all-dynamic has no compiled-prefix")
    (is (nil? (:compiled-simulate (:schema all-dynamic-model)))
        "all-dynamic has no compiled-simulate"))

  (testing "no-compilable-prefix model: beta-dist not compilable"
    (is (nil? (:compiled-prefix (:schema no-compilable-prefix-model)))
        "no-compilable-prefix has no compiled-prefix")))

;; ---------------------------------------------------------------------------
;; 3. Single prefix site: value + score + choices
;; ---------------------------------------------------------------------------

(deftest single-prefix-site-test
  (let [k (rng/fresh-key 42)
        model-c (dyn/with-key single-prefix-model k)
        model-h (dyn/with-key (force-handler single-prefix-model) k)
        trace-c (p/simulate model-c [])
        trace-h (p/simulate model-h [])]
    (is (h/close? (cv trace-h :x) (cv trace-c :x) 1e-5)
        "single-prefix: :x value matches")
    (is (h/close? (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
        "single-prefix: score matches")
    (is (cm/has-value? (cm/get-submap (:choices trace-c) :x))
        "single-prefix: has :x choice")
    (is (cm/has-value? (cm/get-submap (:choices trace-c) :y0))
        "single-prefix: has :y0 choice")))

;; ---------------------------------------------------------------------------
;; 4. Multi prefix site: loop model
;; ---------------------------------------------------------------------------

(deftest multi-prefix-site-test
  (let [k (rng/fresh-key 55)
        xs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
        model-c (dyn/with-key loop-model k)
        model-h (dyn/with-key (force-handler loop-model) k)
        trace-c (p/simulate model-c [xs])
        trace-h (p/simulate model-h [xs])]
    (is (h/close? (cv trace-h :mu) (cv trace-c :mu) 1e-5)
        "loop-model: :mu matches")
    (is (h/close? (cv trace-h :sigma) (cv trace-c :sigma) 1e-5)
        "loop-model: :sigma matches")
    (is (h/close? (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
        "loop-model: score matches")
    (is (cm/has-value? (cm/get-submap (:choices trace-c) :y0))
        "loop-model: has :y0")
    (is (cm/has-value? (cm/get-submap (:choices trace-c) :y1))
        "loop-model: has :y1")
    (is (cm/has-value? (cm/get-submap (:choices trace-c) :y2))
        "loop-model: has :y2")))

;; ---------------------------------------------------------------------------
;; 5. Dependent prefix sites
;; ---------------------------------------------------------------------------

(deftest dependent-prefix-sites-test
  (let [k (rng/fresh-key 66)
        model-c (dyn/with-key dep-prefix-model k)
        model-h (dyn/with-key (force-handler dep-prefix-model) k)
        trace-c (p/simulate model-c [(mx/scalar 5.0)])
        trace-h (p/simulate model-h [(mx/scalar 5.0)])]
    (is (h/close? (cv trace-h :a) (cv trace-c :a) 1e-5)
        "dep-prefix: :a matches")
    (is (h/close? (cv trace-h :b) (cv trace-c :b) 1e-5)
        "dep-prefix: :b matches (depends on :a)")
    (is (h/close? (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
        "dep-prefix: score matches")
    (is (h/close? (mx/item (:retval trace-h)) (mx/item (:retval trace-c)) 1e-5)
        "dep-prefix: retval matches")))

;; ---------------------------------------------------------------------------
;; 6. Prefix with gen args
;; ---------------------------------------------------------------------------

(deftest prefix-with-gen-args-test
  (let [k (rng/fresh-key 77)
        model-c (dyn/with-key args-prefix-model k)
        model-h (dyn/with-key (force-handler args-prefix-model) k)
        trace-c (p/simulate model-c [(mx/scalar 5.0) (mx/scalar 2.0)])
        trace-h (p/simulate model-h [(mx/scalar 5.0) (mx/scalar 2.0)])]
    (is (h/close? (cv trace-h :x) (cv trace-c :x) 1e-5)
        "args-prefix: :x matches (uses gen args)")
    (is (h/close? (cv trace-h :y) (cv trace-c :y) 1e-5)
        "args-prefix: :y matches")
    (is (h/close? (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
        "args-prefix: score matches")
    (is (cm/has-value? (cm/get-submap (:choices trace-c) :z0))
        "args-prefix: has :z0")))

;; ---------------------------------------------------------------------------
;; 7. Value equivalence across multiple models
;; ---------------------------------------------------------------------------

(deftest value-equivalence-test
  (testing "branch model (flag=true path)"
    (let [k (rng/fresh-key 88)
          model-c (dyn/with-key branch-model k)
          model-h (dyn/with-key (force-handler branch-model) k)
          trace-c (p/simulate model-c [true])
          trace-h (p/simulate model-h [true])]
      (is (h/close? (cv trace-h :mu) (cv trace-c :mu) 1e-5)
          "branch(true): :mu matches")
      (is (h/close? (cv trace-h :a) (cv trace-c :a) 1e-5)
          "branch(true): :a matches")))

  (testing "branch model (flag=false path)"
    (let [k (rng/fresh-key 88)
          model-c (dyn/with-key branch-model k)
          model-h (dyn/with-key (force-handler branch-model) k)
          trace-c (p/simulate model-c [false])
          trace-h (p/simulate model-h [false])]
      (is (h/close? (cv trace-h :mu) (cv trace-c :mu) 1e-5)
          "branch(false): :mu matches")
      (is (h/close? (cv trace-h :b) (cv trace-c :b) 1e-5)
          "branch(false): :b matches"))))

;; ---------------------------------------------------------------------------
;; 8. Score equivalence
;; ---------------------------------------------------------------------------

(deftest score-equivalence-test
  (doseq [[label model args-fn]
          [["loop-model" loop-model
            (fn [] [[(mx/scalar 1.0) (mx/scalar 2.0)]])]
           ["branch-model" branch-model
            (fn [] [true])]
           ["dep-prefix" dep-prefix-model
            (fn [] [(mx/scalar 3.0)])]
           ["args-prefix" args-prefix-model
            (fn [] [(mx/scalar 1.0) (mx/scalar 0.5)])]]]
    (doseq [seed [10 20 30]]
      (let [k (rng/fresh-key seed)
            tc (p/simulate (dyn/with-key model k) (args-fn))
            th (p/simulate (dyn/with-key (force-handler model) k) (args-fn))]
        (is (h/close? (mx/item (:score th)) (mx/item (:score tc)) 1e-4)
            (str label " score@seed=" seed))))))

;; ---------------------------------------------------------------------------
;; 9. PRNG consistency — dynamic sites get correct keys
;; ---------------------------------------------------------------------------

(deftest prng-consistency-test
  (testing "loop model: dynamic sites match handler"
    (let [k (rng/fresh-key 99)
          xs [(mx/scalar 1.0) (mx/scalar 2.0)]
          trace-c (p/simulate (dyn/with-key loop-model k) [xs])
          trace-h (p/simulate (dyn/with-key (force-handler loop-model) k) [xs])]
      (doseq [addr [:y0 :y1]]
        (is (h/close? (cv trace-h addr) (cv trace-c addr) 1e-5)
            (str "PRNG: " addr " matches handler")))))

  (testing "dep-prefix model: dynamic sites match handler"
    (let [k (rng/fresh-key 99)
          trace-c (p/simulate (dyn/with-key dep-prefix-model k) [(mx/scalar 2.0)])
          trace-h (p/simulate (dyn/with-key (force-handler dep-prefix-model) k) [(mx/scalar 2.0)])]
      (doseq [addr [:y0 :y1]]
        (is (h/close? (cv trace-h addr) (cv trace-c addr) 1e-5)
            (str "PRNG dep: " addr " matches handler"))))))

;; ---------------------------------------------------------------------------
;; 10. GFI operations on partially-compiled models
;; ---------------------------------------------------------------------------

(deftest gfi-operations-test
  (testing "generate with constraints"
    (let [k (rng/fresh-key 42)
          obs (cm/choicemap :y0 (mx/scalar 5.0) :y1 (mx/scalar 6.0) :y2 (mx/scalar 7.0))
          {:keys [trace weight]} (p/generate (dyn/with-key single-prefix-model k) [] obs)]
      (is (some? trace) "generate: trace exists")
      (is (js/isFinite (mx/item weight)) "generate: weight is finite")
      (is (h/close? 5.0 (cv trace :y0) 1e-6) "generate: :y0 constrained")))

  (testing "update from a trace"
    (let [k1 (rng/fresh-key 42)
          t1 (p/simulate (dyn/with-key single-prefix-model k1) [])
          new-obs (cm/choicemap :y0 (mx/scalar 10.0))
          {:keys [trace weight]} (p/update (dyn/with-key single-prefix-model (rng/fresh-key 43)) t1 new-obs)]
      (is (some? trace) "update: trace exists")
      (is (js/isFinite (mx/item weight)) "update: weight is finite")))

  (testing "regenerate with selection"
    (let [k (rng/fresh-key 42)
          t1 (p/simulate (dyn/with-key loop-model k) [[(mx/scalar 1.0)]])
          sel (sel/select :mu)
          {:keys [trace weight]} (p/regenerate (dyn/with-key loop-model (rng/fresh-key 43)) t1 sel)]
      (is (some? trace) "regenerate: trace exists")
      (is (js/isFinite (mx/item weight)) "regenerate: weight is finite")))

  (testing "assess"
    (let [k (rng/fresh-key 42)
          t1 (p/simulate (dyn/with-key single-prefix-model k) [])
          {:keys [weight]} (p/assess (dyn/with-key single-prefix-model (rng/fresh-key 42))
                                      [] (:choices t1))]
      (is (js/isFinite (mx/item weight)) "assess: weight is finite"))))

;; ---------------------------------------------------------------------------
;; 11. Non-compilable cutoff
;; ---------------------------------------------------------------------------

(deftest non-compilable-cutoff-test
  (testing "cutoff model: gaussian prefix stops at beta-dist"
    (let [sites (compiled/extract-prefix-sites (:source cutoff-model))]
      (is (= 2 (count sites)) "cutoff: 2 raw prefix sites (before doseq)")
      (is (= :a (:addr (first sites))) "cutoff: first is :a")
      (is (= :b (:addr (second sites))) "cutoff: second is :b (beta-dist)")))

  (testing "schema has compiled-prefix with only :a"
    (is (some? (:compiled-prefix (:schema cutoff-model)))
        "cutoff model has compiled-prefix")
    (is (= [:a] (:compiled-prefix-addrs (:schema cutoff-model)))
        "cutoff: prefix-addrs is [:a]"))

  (testing "values still correct despite truncation"
    (let [k (rng/fresh-key 42)
          trace-c (p/simulate (dyn/with-key cutoff-model k) [])
          trace-h (p/simulate (dyn/with-key (force-handler cutoff-model) k) [])]
      (is (h/close? (cv trace-h :a) (cv trace-c :a) 1e-5)
          "cutoff: :a matches handler")
      (is (h/close? (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
          "cutoff: score matches"))))

;; ---------------------------------------------------------------------------
;; 12. All distributions in prefix
;; ---------------------------------------------------------------------------

(deftest all-distributions-in-prefix-test
  (testing "multi-dist model: gaussian, uniform, bernoulli, exponential, laplace, cauchy"
    (is (some? (:compiled-prefix (:schema multi-dist-prefix)))
        "multi-dist has compiled-prefix")
    (let [k (rng/fresh-key 42)
          trace-c (p/simulate (dyn/with-key multi-dist-prefix k) [])
          trace-h (p/simulate (dyn/with-key (force-handler multi-dist-prefix) k) [])]
      (doseq [addr [:a :b :c :d :e :f]]
        (let [tol (if (= addr :f) 1e-3 1e-4)]
          (is (h/close? (cv trace-h addr) (cv trace-c addr) tol)
              (str "multi-dist: " addr " matches"))))
      (is (h/close? (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-3)
          "multi-dist: score matches")))

  (testing "log-normal + delta prefix"
    (is (some? (:compiled-prefix (:schema lognormal-delta-prefix)))
        "lognormal-delta has compiled-prefix")
    (let [k (rng/fresh-key 42)
          trace-c (p/simulate (dyn/with-key lognormal-delta-prefix k) [])
          trace-h (p/simulate (dyn/with-key (force-handler lognormal-delta-prefix) k) [])]
      (is (h/close? (cv trace-h :a) (cv trace-c :a) 1e-4)
          "log-normal prefix: :a matches")
      (is (h/close? (cv trace-h :b) (cv trace-c :b) 1e-6)
          "delta prefix: :b matches")
      (is (h/close? (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-3)
          "lognormal-delta: score matches"))))

;; ---------------------------------------------------------------------------
;; 13. Edge cases
;; ---------------------------------------------------------------------------

(deftest edge-cases-test
  (testing "prefix-only model: static prefix site + branch with no traces"
    (let [k (rng/fresh-key 42)
          m (gen []
              (let [x (trace :x (dist/gaussian 0 1))]
                (if (> (mx/item x) 0)
                  (mx/scalar 1.0)
                  (mx/scalar -1.0))))
          trace-c (p/simulate (dyn/with-key m k) [])
          trace-h (p/simulate (dyn/with-key (force-handler m) k) [])]
      (is (h/close? (cv trace-h :x) (cv trace-c :x) 1e-5)
          "edge: branch-no-trace prefix :x matches")
      (is (h/close? (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
          "edge: branch-no-trace score matches")))

  (testing "prefix with 0-arg gen fn"
    (let [k (rng/fresh-key 42)
          m (gen []
              (let [x (trace :x (dist/gaussian 0 1))]
                (doseq [i (range 2)]
                  (trace (keyword (str "y" i)) (dist/gaussian x 1)))
                x))
          trace-c (p/simulate (dyn/with-key m k) [])
          trace-h (p/simulate (dyn/with-key (force-handler m) k) [])]
      (is (h/close? (cv trace-h :x) (cv trace-c :x) 1e-5)
          "edge: 0-arg model :x matches")))

  (testing "reproducibility: same key -> same result twice"
    (let [k (rng/fresh-key 42)
          t1 (p/simulate (dyn/with-key loop-model k) [[(mx/scalar 1.0)]])
          t2 (p/simulate (dyn/with-key loop-model k) [[(mx/scalar 1.0)]])]
      (is (h/close? (cv t1 :mu) (cv t2 :mu) 1e-6)
          "edge: reproducible :mu")
      (is (h/close? (mx/item (:score t1)) (mx/item (:score t2)) 1e-6)
          "edge: reproducible score")))

  (testing "return value correct for partially-compiled model"
    (let [k (rng/fresh-key 42)
          trace-c (p/simulate (dyn/with-key loop-model k) [[(mx/scalar 1.0)]])
          trace-h (p/simulate (dyn/with-key (force-handler loop-model) k) [[(mx/scalar 1.0)]])]
      (is (h/close? (mx/item (:retval trace-h)) (mx/item (:retval trace-c)) 1e-5)
          "edge: retval matches handler"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
