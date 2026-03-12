(ns genmlx.partial-compile-test
  "L1-M3 tests: partial compilation for dynamic models.

   Tests cover:
   1. Prefix extraction (extract-prefix-sites)
   2. Prefix gates (splice/param → no prefix; static → L1-M2 not M3)
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
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.schema :as schema]
            [genmlx.handler :as h]
            [genmlx.selection :as sel]))

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

(println "\n========================================")
(println "L1-M3: Partial Compilation for Dynamic Models")
(println "========================================")

;; ---------------------------------------------------------------------------
;; 1. Prefix extraction
;; ---------------------------------------------------------------------------

(println "\n== 1. Prefix extraction ==")

;; Loop model: 2 prefix sites before doseq
(let [sites (compiled/extract-prefix-sites (:source loop-model))]
  (assert-true "loop-model has 2 prefix sites" (= 2 (count sites)))
  (assert-equal "first prefix addr is :mu" :mu (:addr (first sites)))
  (assert-equal "second prefix addr is :sigma" :sigma (:addr (second sites)))
  (assert-equal "mu dist-type is gaussian" :gaussian (:dist-type (first sites)))
  (assert-equal "sigma dist-type is exponential" :exponential (:dist-type (second sites))))

;; Branch model: 1 prefix site before if
(let [sites (compiled/extract-prefix-sites (:source branch-model))]
  (assert-true "branch-model has 1 prefix site" (= 1 (count sites)))
  (assert-equal "prefix addr is :mu" :mu (:addr (first sites))))

;; Single prefix model: 1 prefix site before doseq
(let [sites (compiled/extract-prefix-sites (:source single-prefix-model))]
  (assert-true "single-prefix has 1 site" (= 1 (count sites)))
  (assert-equal "prefix addr is :x" :x (:addr (first sites))))

;; All-dynamic: 0 prefix sites (doseq immediately)
(let [sites (compiled/extract-prefix-sites (:source all-dynamic-model))]
  (assert-true "all-dynamic has 0 prefix sites" (= 0 (count sites))))

;; Dependent prefix: 2 prefix sites
(let [sites (compiled/extract-prefix-sites (:source dep-prefix-model))]
  (assert-true "dep-prefix has 2 prefix sites" (= 2 (count sites)))
  (assert-equal "first is :a" :a (:addr (first sites)))
  (assert-equal "second is :b" :b (:addr (second sites))))

;; Multi-dist prefix: 6 prefix sites
(let [sites (compiled/extract-prefix-sites (:source multi-dist-prefix))]
  (assert-true "multi-dist has 6 prefix sites" (= 6 (count sites))))

;; ---------------------------------------------------------------------------
;; 2. Prefix gates
;; ---------------------------------------------------------------------------

(println "\n== 2. Prefix gates ==")

;; Static model should get compiled-simulate (L1-M2), NOT compiled-prefix
(assert-true "static model has compiled-simulate"
  (some? (:compiled-simulate (:schema static-model))))
(assert-true "static model has NO compiled-prefix"
  (nil? (:compiled-prefix (:schema static-model))))

;; Loop model should get compiled-prefix (non-static with prefix sites)
(assert-true "loop model has compiled-prefix"
  (some? (:compiled-prefix (:schema loop-model))))
(assert-true "loop model has NO compiled-simulate"
  (nil? (:compiled-simulate (:schema loop-model))))

;; All-dynamic model: no compiled paths (empty prefix)
(assert-true "all-dynamic has no compiled-prefix"
  (nil? (:compiled-prefix (:schema all-dynamic-model))))
(assert-true "all-dynamic has no compiled-simulate"
  (nil? (:compiled-simulate (:schema all-dynamic-model))))

;; No-compilable-prefix model: prefix extraction finds :a but beta-dist
;; isn't compilable, so no compiled-prefix
(assert-true "no-compilable-prefix has no compiled-prefix"
  (nil? (:compiled-prefix (:schema no-compilable-prefix-model))))

;; ---------------------------------------------------------------------------
;; 3. Single-prefix-site: value + score + choices
;; ---------------------------------------------------------------------------

(println "\n== 3. Single prefix site ==")

(let [k (rng/fresh-key 42)
      model-c (dyn/with-key single-prefix-model k)
      model-h (dyn/with-key (force-handler single-prefix-model) k)
      trace-c (p/simulate model-c [])
      trace-h (p/simulate model-h [])]
  (assert-close "single-prefix: :x value matches"
    (cv trace-h :x) (cv trace-c :x) 1e-5)
  (assert-close "single-prefix: score matches"
    (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
  ;; All 4 choices present (x + y0 y1 y2)
  (assert-true "single-prefix: has :x choice"
    (cm/has-value? (cm/get-submap (:choices trace-c) :x)))
  (assert-true "single-prefix: has :y0 choice"
    (cm/has-value? (cm/get-submap (:choices trace-c) :y0))))

;; ---------------------------------------------------------------------------
;; 4. Multi-prefix-site: loop model
;; ---------------------------------------------------------------------------

(println "\n== 4. Multi-prefix sites ==")

(let [k (rng/fresh-key 55)
      xs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      model-c (dyn/with-key loop-model k)
      model-h (dyn/with-key (force-handler loop-model) k)
      trace-c (p/simulate model-c [xs])
      trace-h (p/simulate model-h [xs])]
  (assert-close "loop-model: :mu matches"
    (cv trace-h :mu) (cv trace-c :mu) 1e-5)
  (assert-close "loop-model: :sigma matches"
    (cv trace-h :sigma) (cv trace-c :sigma) 1e-5)
  (assert-close "loop-model: score matches"
    (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
  ;; Dynamic choices present
  (assert-true "loop-model: has :y0"
    (cm/has-value? (cm/get-submap (:choices trace-c) :y0)))
  (assert-true "loop-model: has :y1"
    (cm/has-value? (cm/get-submap (:choices trace-c) :y1)))
  (assert-true "loop-model: has :y2"
    (cm/has-value? (cm/get-submap (:choices trace-c) :y2))))

;; ---------------------------------------------------------------------------
;; 5. Dependent prefix sites
;; ---------------------------------------------------------------------------

(println "\n== 5. Dependent prefix sites ==")

(let [k (rng/fresh-key 66)
      model-c (dyn/with-key dep-prefix-model k)
      model-h (dyn/with-key (force-handler dep-prefix-model) k)
      trace-c (p/simulate model-c [(mx/scalar 5.0)])
      trace-h (p/simulate model-h [(mx/scalar 5.0)])]
  (assert-close "dep-prefix: :a matches"
    (cv trace-h :a) (cv trace-c :a) 1e-5)
  (assert-close "dep-prefix: :b matches (depends on :a)"
    (cv trace-h :b) (cv trace-c :b) 1e-5)
  (assert-close "dep-prefix: score matches"
    (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
  (assert-close "dep-prefix: retval matches"
    (mx/item (:retval trace-h)) (mx/item (:retval trace-c)) 1e-5))

;; ---------------------------------------------------------------------------
;; 6. Prefix with gen args
;; ---------------------------------------------------------------------------

(println "\n== 6. Prefix with gen args ==")

(let [k (rng/fresh-key 77)
      model-c (dyn/with-key args-prefix-model k)
      model-h (dyn/with-key (force-handler args-prefix-model) k)
      trace-c (p/simulate model-c [(mx/scalar 5.0) (mx/scalar 2.0)])
      trace-h (p/simulate model-h [(mx/scalar 5.0) (mx/scalar 2.0)])]
  (assert-close "args-prefix: :x matches (uses gen args)"
    (cv trace-h :x) (cv trace-c :x) 1e-5)
  (assert-close "args-prefix: :y matches"
    (cv trace-h :y) (cv trace-c :y) 1e-5)
  (assert-close "args-prefix: score matches"
    (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4)
  ;; Dynamic choices present
  (assert-true "args-prefix: has :z0"
    (cm/has-value? (cm/get-submap (:choices trace-c) :z0))))

;; ---------------------------------------------------------------------------
;; 7. Value equivalence across multiple models
;; ---------------------------------------------------------------------------

(println "\n== 7. Value equivalence ==")

;; Branch model (flag=true path)
(let [k (rng/fresh-key 88)
      model-c (dyn/with-key branch-model k)
      model-h (dyn/with-key (force-handler branch-model) k)
      trace-c (p/simulate model-c [true])
      trace-h (p/simulate model-h [true])]
  (assert-close "branch(true): :mu matches"
    (cv trace-h :mu) (cv trace-c :mu) 1e-5)
  (assert-close "branch(true): :a matches"
    (cv trace-h :a) (cv trace-c :a) 1e-5))

;; Branch model (flag=false path)
(let [k (rng/fresh-key 88)
      model-c (dyn/with-key branch-model k)
      model-h (dyn/with-key (force-handler branch-model) k)
      trace-c (p/simulate model-c [false])
      trace-h (p/simulate model-h [false])]
  (assert-close "branch(false): :mu matches"
    (cv trace-h :mu) (cv trace-c :mu) 1e-5)
  (assert-close "branch(false): :b matches"
    (cv trace-h :b) (cv trace-c :b) 1e-5))

;; ---------------------------------------------------------------------------
;; 8. Score equivalence
;; ---------------------------------------------------------------------------

(println "\n== 8. Score equivalence ==")

;; Run each model 3 times with different keys, verify score matches
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
      (assert-close (str label " score@seed=" seed)
        (mx/item (:score th)) (mx/item (:score tc)) 1e-4))))

;; ---------------------------------------------------------------------------
;; 9. PRNG consistency — dynamic sites get correct keys
;; ---------------------------------------------------------------------------

(println "\n== 9. PRNG consistency ==")

;; Loop model: dynamic sites :y0 :y1 must match handler
(let [k (rng/fresh-key 99)
      xs [(mx/scalar 1.0) (mx/scalar 2.0)]
      trace-c (p/simulate (dyn/with-key loop-model k) [xs])
      trace-h (p/simulate (dyn/with-key (force-handler loop-model) k) [xs])]
  (doseq [addr [:y0 :y1]]
    (assert-close (str "PRNG: " addr " matches handler")
      (cv trace-h addr) (cv trace-c addr) 1e-5)))

;; Dep-prefix model: dynamic sites :y0 :y1 must match
(let [k (rng/fresh-key 99)
      trace-c (p/simulate (dyn/with-key dep-prefix-model k) [(mx/scalar 2.0)])
      trace-h (p/simulate (dyn/with-key (force-handler dep-prefix-model) k) [(mx/scalar 2.0)])]
  (doseq [addr [:y0 :y1]]
    (assert-close (str "PRNG dep: " addr " matches handler")
      (cv trace-h addr) (cv trace-c addr) 1e-5)))

;; ---------------------------------------------------------------------------
;; 10. GFI operations on partially-compiled models
;; ---------------------------------------------------------------------------

(println "\n== 10. GFI operations ==")

;; generate with constraints
(let [k (rng/fresh-key 42)
      obs (cm/choicemap :y0 (mx/scalar 5.0) :y1 (mx/scalar 6.0) :y2 (mx/scalar 7.0))
      {:keys [trace weight]} (p/generate (dyn/with-key single-prefix-model k) [] obs)]
  (assert-true "generate: trace exists" (some? trace))
  (assert-true "generate: weight is finite"
    (js/isFinite (mx/item weight)))
  (assert-close "generate: :y0 constrained"
    5.0 (cv trace :y0) 1e-6))

;; update from a trace
(let [k1 (rng/fresh-key 42)
      t1 (p/simulate (dyn/with-key single-prefix-model k1) [])
      new-obs (cm/choicemap :y0 (mx/scalar 10.0))
      {:keys [trace weight]} (p/update (dyn/with-key single-prefix-model (rng/fresh-key 43)) t1 new-obs)]
  (assert-true "update: trace exists" (some? trace))
  (assert-true "update: weight is finite"
    (js/isFinite (mx/item weight))))

;; regenerate with selection
(let [k (rng/fresh-key 42)
      t1 (p/simulate (dyn/with-key loop-model k) [[(mx/scalar 1.0)]])
      sel (sel/select :mu)
      {:keys [trace weight]} (p/regenerate (dyn/with-key loop-model (rng/fresh-key 43)) t1 sel)]
  (assert-true "regenerate: trace exists" (some? trace))
  (assert-true "regenerate: weight is finite"
    (js/isFinite (mx/item weight))))

;; assess
(let [k (rng/fresh-key 42)
      t1 (p/simulate (dyn/with-key single-prefix-model k) [])
      {:keys [weight]} (p/assess (dyn/with-key single-prefix-model (rng/fresh-key 42))
                                  [] (:choices t1))]
  (assert-true "assess: weight is finite"
    (js/isFinite (mx/item weight))))

;; ---------------------------------------------------------------------------
;; 11. Non-compilable cutoff
;; ---------------------------------------------------------------------------

(println "\n== 11. Non-compilable cutoff ==")

;; Cutoff model: gaussian prefix stops at beta-dist
(let [sites (compiled/extract-prefix-sites (:source cutoff-model))]
  (assert-true "cutoff: 2 raw prefix sites (before doseq)"
    (= 2 (count sites)))
  (assert-equal "cutoff: first is :a" :a (:addr (first sites)))
  (assert-equal "cutoff: second is :b (beta-dist)" :b (:addr (second sites))))

;; Schema should have compiled-prefix with only :a (beta-dist truncated)
(assert-true "cutoff model has compiled-prefix"
  (some? (:compiled-prefix (:schema cutoff-model))))
(assert-equal "cutoff: prefix-addrs is [:a]"
  [:a] (:compiled-prefix-addrs (:schema cutoff-model)))

;; Values still correct despite truncation
(let [k (rng/fresh-key 42)
      trace-c (p/simulate (dyn/with-key cutoff-model k) [])
      trace-h (p/simulate (dyn/with-key (force-handler cutoff-model) k) [])]
  (assert-close "cutoff: :a matches handler"
    (cv trace-h :a) (cv trace-c :a) 1e-5)
  (assert-close "cutoff: score matches"
    (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4))

;; ---------------------------------------------------------------------------
;; 12. All distributions in prefix
;; ---------------------------------------------------------------------------

(println "\n== 12. All distributions in prefix ==")

;; Multi-dist model: gaussian, uniform, bernoulli, exponential, laplace, cauchy
(assert-true "multi-dist has compiled-prefix"
  (some? (:compiled-prefix (:schema multi-dist-prefix))))

(let [k (rng/fresh-key 42)
      trace-c (p/simulate (dyn/with-key multi-dist-prefix k) [])
      trace-h (p/simulate (dyn/with-key (force-handler multi-dist-prefix) k) [])]
  (doseq [addr [:a :b :c :d :e :f]]
    ;; cauchy has wider tolerance: tan() amplifies float32 precision diffs
    (let [tol (if (= addr :f) 1e-3 1e-4)]
      (assert-close (str "multi-dist: " addr " matches")
        (cv trace-h addr) (cv trace-c addr) tol)))
  (assert-close "multi-dist: score matches"
    (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-3))

;; Log-normal + delta prefix
(assert-true "lognormal-delta has compiled-prefix"
  (some? (:compiled-prefix (:schema lognormal-delta-prefix))))

(let [k (rng/fresh-key 42)
      trace-c (p/simulate (dyn/with-key lognormal-delta-prefix k) [])
      trace-h (p/simulate (dyn/with-key (force-handler lognormal-delta-prefix) k) [])]
  (assert-close "log-normal prefix: :a matches"
    (cv trace-h :a) (cv trace-c :a) 1e-4)
  (assert-close "delta prefix: :b matches"
    (cv trace-h :b) (cv trace-c :b) 1e-6)
  (assert-close "lognormal-delta: score matches"
    (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-3))

;; ---------------------------------------------------------------------------
;; 13. Edge cases
;; ---------------------------------------------------------------------------

(println "\n== 13. Edge cases ==")

;; Prefix-only model: static prefix site + branch with no traces (non-static)
(let [k (rng/fresh-key 42)
      m (gen []
          (let [x (trace :x (dist/gaussian 0 1))]
            (if (> (mx/item x) 0)
              (mx/scalar 1.0)
              (mx/scalar -1.0))))
      trace-c (p/simulate (dyn/with-key m k) [])
      trace-h (p/simulate (dyn/with-key (force-handler m) k) [])]
  (assert-close "edge: branch-no-trace prefix :x matches"
    (cv trace-h :x) (cv trace-c :x) 1e-5)
  (assert-close "edge: branch-no-trace score matches"
    (mx/item (:score trace-h)) (mx/item (:score trace-c)) 1e-4))

;; Prefix with 0-arg gen fn
(let [k (rng/fresh-key 42)
      m (gen []
          (let [x (trace :x (dist/gaussian 0 1))]
            (doseq [i (range 2)]
              (trace (keyword (str "y" i)) (dist/gaussian x 1)))
            x))
      trace-c (p/simulate (dyn/with-key m k) [])
      trace-h (p/simulate (dyn/with-key (force-handler m) k) [])]
  (assert-close "edge: 0-arg model :x matches"
    (cv trace-h :x) (cv trace-c :x) 1e-5))

;; Reproducibility: same key → same result twice
(let [k (rng/fresh-key 42)
      t1 (p/simulate (dyn/with-key loop-model k) [[(mx/scalar 1.0)]])
      t2 (p/simulate (dyn/with-key loop-model k) [[(mx/scalar 1.0)]])]
  (assert-close "edge: reproducible :mu"
    (cv t1 :mu) (cv t2 :mu) 1e-6)
  (assert-close "edge: reproducible score"
    (mx/item (:score t1)) (mx/item (:score t2)) 1e-6))

;; Return value correct for partially-compiled model
(let [k (rng/fresh-key 42)
      trace-c (p/simulate (dyn/with-key loop-model k) [[(mx/scalar 1.0)]])
      trace-h (p/simulate (dyn/with-key (force-handler loop-model) k) [[(mx/scalar 1.0)]])]
  (assert-close "edge: retval matches handler"
    (mx/item (:retval trace-h)) (mx/item (:retval trace-c)) 1e-5))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n========================================")
(println (str "L1-M3 Partial Compilation: " @pass-count " passed, " @fail-count " failed"))
(println "========================================")

(when (> @fail-count 0)
  (js/process.exit 1))
