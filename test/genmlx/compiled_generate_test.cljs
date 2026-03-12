(ns genmlx.compiled-generate-test
  "WP-1 tests: compiled generate for static DynamicGF models.
   Validates that compiled generate matches handler generate exactly."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.schema :as schema]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc) (println (str "  FAIL: " msg)))))

(defn assert-close [msg expected actual tol]
  (let [e (if (number? expected) expected (mx/item expected))
        a (if (number? actual) actual (mx/item actual))
        diff (js/Math.abs (- e a))]
    (if (<= diff tol)
      (do (swap! pass-count inc) (println (str "  PASS: " msg)))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " msg " expected=" e " actual=" a " diff=" diff))))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " msg " expected=" expected " actual=" actual)))))

(defn force-handler
  "Strip compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-prefix :compiled-prefix-addrs)]
    (assoc gf :schema schema)))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

;; Simple 2-site gaussian model
(def simple-model
  (dyn/auto-key
    (gen [mu]
      (let [x (trace :x (dist/gaussian mu (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 2.0)))]
        y))))

;; 3-site model with dependency chain
(def chain-model
  (dyn/auto-key
    (gen [a]
      (let [x (trace :x (dist/gaussian a (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 0.5)))
            z (trace :z (dist/gaussian (mx/add x y) (mx/scalar 1.0)))]
        z))))

;; Multi-distribution model
(def multi-dist-model
  (dyn/auto-key
    (gen []
      (let [g (trace :g (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            u (trace :u (dist/uniform (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/bernoulli (mx/scalar 0.5)))
            e (trace :e (dist/exponential (mx/scalar 2.0)))]
        g))))

;; Model with computed return value (not a trace site)
(def retval-model
  (dyn/auto-key
    (gen [s]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) s))
            y (trace :y (dist/gaussian (mx/scalar 0.0) s))]
        (mx/add x y)))))

;; Delta distribution model
(def delta-model
  (dyn/auto-key
    (gen [v]
      (let [x (trace :x (dist/delta v))
            y (trace :y (dist/gaussian x (mx/scalar 1.0)))]
        y))))

;; Model with laplace + cauchy + log-normal
(def exotic-model
  (dyn/auto-key
    (gen []
      (let [a (trace :a (dist/laplace (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/cauchy (mx/scalar 0.0) (mx/scalar 1.0)))
            c (trace :c (dist/log-normal (mx/scalar 0.0) (mx/scalar 0.5)))]
        a))))

;; Non-compilable model (has dynamic addresses) — should fall back to handler
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

(println "\n== WP-1: Compiled Generate Tests ==")

;; ---- Section 1: Compilation check ----
(println "\n-- 1. compilation presence --")

(assert-true "simple-model has :compiled-generate"
  (some? (:compiled-generate (:schema simple-model))))

(assert-true "chain-model has :compiled-generate"
  (some? (:compiled-generate (:schema chain-model))))

(assert-true "multi-dist-model has :compiled-generate"
  (some? (:compiled-generate (:schema multi-dist-model))))

(assert-true "retval-model has :compiled-generate"
  (some? (:compiled-generate (:schema retval-model))))

(assert-true "delta-model has :compiled-generate"
  (some? (:compiled-generate (:schema delta-model))))

(assert-true "exotic-model has :compiled-generate"
  (some? (:compiled-generate (:schema exotic-model))))

(assert-true "dynamic-addr-model does NOT have :compiled-generate"
  (nil? (:compiled-generate (:schema dynamic-addr-model))))

;; ---- Section 2: All sites constrained ----
(println "\n-- 2. all sites constrained --")

(let [key (rng/fresh-key 42)
      gf (dyn/with-key simple-model key)
      constraints (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 5.0))
      {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
      ;; Handler reference
      gf-h (dyn/with-key (force-handler simple-model) key)
      {:keys [trace weight] :as ref} (p/generate gf-h [(mx/scalar 0.0)] constraints)
      ref-trace trace
      ref-weight weight
      ;; Re-run compiled
      gf2 (dyn/with-key simple-model key)
      {:keys [trace weight]} (p/generate gf2 [(mx/scalar 0.0)] constraints)]
  (assert-close "all-constrained: score matches handler"
    (:score ref-trace) (:score trace) 1e-5)
  (assert-close "all-constrained: weight matches handler"
    ref-weight weight 1e-5)
  (assert-close "all-constrained: weight = score (all constrained)"
    (:score trace) weight 1e-5)
  (assert-close "all-constrained: x = constraint"
    3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
  (assert-close "all-constrained: y = constraint"
    5.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y))) 1e-6))

;; ---- Section 3: No sites constrained (= simulate) ----
(println "\n-- 3. no sites constrained --")

(let [key (rng/fresh-key 77)
      gf (dyn/with-key simple-model key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 1.0)] cm/EMPTY)
      ;; Handler reference
      gf-h (dyn/with-key (force-handler simple-model) key)
      ref (p/generate gf-h [(mx/scalar 1.0)] cm/EMPTY)]
  (assert-close "no-constraints: weight = 0"
    0.0 weight 1e-6)
  (assert-close "no-constraints: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "no-constraints: weight matches handler"
    (:weight ref) weight 1e-5)
  ;; Values should match (same PRNG key)
  (assert-close "no-constraints: x matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :x)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
    1e-5)
  (assert-close "no-constraints: y matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5))

;; ---- Section 4: Mixed constrained/unconstrained ----
(println "\n-- 4. mixed constraints --")

(let [key (rng/fresh-key 99)
      constraints (cm/choicemap :x (mx/scalar 2.5))
      gf (dyn/with-key simple-model key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
      ;; Handler reference
      gf-h (dyn/with-key (force-handler simple-model) key)
      ref (p/generate gf-h [(mx/scalar 0.0)] constraints)]
  (assert-close "mixed: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "mixed: weight matches handler"
    (:weight ref) weight 1e-5)
  (assert-close "mixed: x = constraint"
    2.5 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
  (assert-true "mixed: weight > 0 (one constrained site)"
    (not= 0.0 (mx/item weight)))
  ;; y should be sampled (not constrained), same as handler
  (assert-close "mixed: y matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5))

;; Constrain only y (second site)
(let [key (rng/fresh-key 99)
      constraints (cm/choicemap :y (mx/scalar 7.0))
      gf (dyn/with-key simple-model key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
      gf-h (dyn/with-key (force-handler simple-model) key)
      ref (p/generate gf-h [(mx/scalar 0.0)] constraints)]
  (assert-close "mixed-y: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "mixed-y: weight matches handler"
    (:weight ref) weight 1e-5)
  (assert-close "mixed-y: y = constraint"
    7.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y))) 1e-6))

;; ---- Section 5: Dependency chain model ----
(println "\n-- 5. dependency chains --")

(let [key (rng/fresh-key 55)
      ;; Constrain x, leave y and z free
      constraints (cm/choicemap :x (mx/scalar 1.0))
      gf (dyn/with-key chain-model key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
      gf-h (dyn/with-key (force-handler chain-model) key)
      ref (p/generate gf-h [(mx/scalar 0.0)] constraints)]
  (assert-close "chain x-only: score matches"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "chain x-only: weight matches"
    (:weight ref) weight 1e-5)
  (assert-close "chain x-only: x = constraint"
    1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
  (assert-close "chain x-only: y matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5)
  (assert-close "chain x-only: z matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :z)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z)))
    1e-5))

;; Constrain x and z, leave y free (z depends on x+y)
(let [key (rng/fresh-key 55)
      constraints (cm/choicemap :x (mx/scalar 1.0) :z (mx/scalar 3.0))
      gf (dyn/with-key chain-model key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)
      gf-h (dyn/with-key (force-handler chain-model) key)
      ref (p/generate gf-h [(mx/scalar 0.0)] constraints)]
  (assert-close "chain x+z: score matches"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "chain x+z: weight matches"
    (:weight ref) weight 1e-5))

;; ---- Section 6: Multi-distribution model ----
(println "\n-- 6. multi-distribution types --")

(let [key (rng/fresh-key 33)
      ;; Constrain gaussian + bernoulli, leave uniform + exponential free
      constraints (cm/choicemap :g (mx/scalar 1.5) :b (mx/scalar 1.0))
      gf (dyn/with-key multi-dist-model key)
      {:keys [trace weight]} (p/generate gf [] constraints)
      gf-h (dyn/with-key (force-handler multi-dist-model) key)
      ref (p/generate gf-h [] constraints)]
  (assert-close "multi-dist: score matches"
    (:score (:trace ref)) (:score trace) 1e-4)
  (assert-close "multi-dist: weight matches"
    (:weight ref) weight 1e-4)
  (assert-close "multi-dist: g = constraint"
    1.5 (mx/item (cm/get-value (cm/get-submap (:choices trace) :g))) 1e-6)
  (assert-close "multi-dist: b = constraint"
    1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :b))) 1e-6))

;; All 4 types constrained
(let [key (rng/fresh-key 33)
      constraints (cm/choicemap :g (mx/scalar 0.5)
                                :u (mx/scalar 0.3)
                                :b (mx/scalar 0.0)
                                :e (mx/scalar 1.0))
      gf (dyn/with-key multi-dist-model key)
      {:keys [trace weight]} (p/generate gf [] constraints)
      gf-h (dyn/with-key (force-handler multi-dist-model) key)
      ref (p/generate gf-h [] constraints)]
  (assert-close "multi-dist all: score matches"
    (:score (:trace ref)) (:score trace) 1e-4)
  (assert-close "multi-dist all: weight = score"
    (:score trace) weight 1e-4))

;; ---- Section 7: Retval correctness ----
(println "\n-- 7. retval correctness --")

;; retval-model returns (mx/add x y), not a trace site
(let [key (rng/fresh-key 11)
      constraints (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
      gf (dyn/with-key retval-model key)
      {:keys [trace]} (p/generate gf [(mx/scalar 1.0)] constraints)]
  (assert-close "retval: add(x,y) = 5.0"
    5.0 (mx/item (:retval trace)) 1e-6))

;; simple-model returns y (last trace site)
(let [key (rng/fresh-key 11)
      constraints (cm/choicemap :y (mx/scalar 9.0))
      gf (dyn/with-key simple-model key)
      {:keys [trace]} (p/generate gf [(mx/scalar 0.0)] constraints)]
  (assert-close "retval: y = constraint"
    9.0 (mx/item (:retval trace)) 1e-6))

;; ---- Section 8: Delta distribution ----
(println "\n-- 8. delta distribution --")

(let [key (rng/fresh-key 22)
      constraints (cm/choicemap :y (mx/scalar 4.0))
      gf (dyn/with-key delta-model key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 2.0)] constraints)
      gf-h (dyn/with-key (force-handler delta-model) key)
      ref (p/generate gf-h [(mx/scalar 2.0)] constraints)]
  (assert-close "delta: score matches"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "delta: weight matches"
    (:weight ref) weight 1e-5)
  (assert-close "delta: x = v (delta deterministic)"
    2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6))

;; ---- Section 9: Exotic distributions (laplace, cauchy, log-normal) ----
(println "\n-- 9. exotic distributions --")

(let [key (rng/fresh-key 44)
      constraints (cm/choicemap :a (mx/scalar 0.5)
                                :b (mx/scalar -1.0)
                                :c (mx/scalar 2.0))
      gf (dyn/with-key exotic-model key)
      {:keys [trace weight]} (p/generate gf [] constraints)
      gf-h (dyn/with-key (force-handler exotic-model) key)
      ref (p/generate gf-h [] constraints)]
  (assert-close "exotic all: score matches"
    (:score (:trace ref)) (:score trace) 1e-4)
  (assert-close "exotic all: weight matches"
    (:weight ref) weight 1e-4))

(let [key (rng/fresh-key 44)
      constraints (cm/choicemap :a (mx/scalar 0.5))
      gf (dyn/with-key exotic-model key)
      {:keys [trace weight]} (p/generate gf [] constraints)
      gf-h (dyn/with-key (force-handler exotic-model) key)
      ref (p/generate gf-h [] constraints)]
  (assert-close "exotic partial: score matches"
    (:score (:trace ref)) (:score trace) 1e-4)
  (assert-close "exotic partial: weight matches"
    (:weight ref) weight 1e-4))

;; ---- Section 10: Non-compilable fallback ----
(println "\n-- 10. non-compilable fallback --")

;; dynamic-addr-model: not compilable (has loops), falls back to handler
(let [key (rng/fresh-key 66)
      constraints (cm/choicemap :x (mx/scalar 2.0) :y0 (mx/scalar 3.0))
      gf (dyn/with-key dynamic-addr-model key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 2.0)] constraints)]
  (assert-true "dynamic-addr: generate works via handler"
    (some? trace))
  (assert-close "dynamic-addr: x = constraint"
    2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
  (assert-close "dynamic-addr: y0 = constraint"
    3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y0))) 1e-6))

;; force-handler on a compilable model → still works
(let [key (rng/fresh-key 66)
      constraints (cm/choicemap :x (mx/scalar 2.0))
      gf (dyn/with-key (force-handler simple-model) key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 0.0)] constraints)]
  (assert-true "force-handler: generate works"
    (some? trace))
  (assert-close "force-handler: x = constraint"
    2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6))

;; ---- Section 11: Trace structure ----
(println "\n-- 11. trace structure --")

(let [key (rng/fresh-key 88)
      constraints (cm/choicemap :x (mx/scalar 1.0))
      gf (dyn/with-key simple-model key)
      {:keys [trace]} (p/generate gf [(mx/scalar 0.0)] constraints)]
  (assert-true "trace: has :gen-fn"
    (= simple-model (:gen-fn trace)))
  (assert-true "trace: has :args"
    (and (= 1 (count (:args trace)))
         (= 0.0 (mx/item (first (:args trace))))))
  (assert-true "trace: has :choices"
    (some? (:choices trace)))
  (assert-true "trace: has :score"
    (some? (:score trace)))
  (assert-true "trace: has :retval"
    (some? (:retval trace)))
  (assert-true "trace: choices has :x"
    (cm/has-value? (cm/get-submap (:choices trace) :x)))
  (assert-true "trace: choices has :y"
    (cm/has-value? (cm/get-submap (:choices trace) :y))))

;; ---- Section 12: PRNG key equivalence ----
(println "\n-- 12. PRNG key equivalence --")

;; Same key, no constraints → compiled and handler give same samples
(let [key (rng/fresh-key 123)
      gf (dyn/with-key simple-model key)
      {:keys [trace]} (p/generate gf [(mx/scalar 5.0)] cm/EMPTY)
      ;; Simulate with same key
      gf-s (dyn/with-key simple-model key)
      sim-trace (p/simulate gf-s [(mx/scalar 5.0)])]
  (assert-close "prng: generate(empty) x = simulate x"
    (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :x)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
    1e-5)
  (assert-close "prng: generate(empty) y = simulate y"
    (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5))

;; ---- Section 13: Direct compiled function call ----
(println "\n-- 13. direct compiled-generate call --")

;; Prove the compiled function exists and produces correct results directly,
;; independent of DynamicGF.generate dispatch
(let [compiled-gen (:compiled-generate (:schema simple-model))
      key (rng/fresh-key 200)
      _ (rng/seed! key)
      constraints (cm/choicemap :x (mx/scalar 4.0))
      result (compiled-gen key [(mx/scalar 1.0)] constraints)]
  (assert-true "direct: returns map with :values"
    (map? (:values result)))
  (assert-true "direct: returns :score"
    (some? (:score result)))
  (assert-true "direct: returns :weight"
    (some? (:weight result)))
  (assert-true "direct: returns :retval"
    (some? (:retval result)))
  (assert-close "direct: constrained x = 4.0"
    4.0 (mx/item (get (:values result) :x)) 1e-6)
  (assert-true "direct: unconstrained y is sampled"
    (some? (get (:values result) :y)))
  (assert-true "direct: weight ≠ 0 (one constrained site)"
    (not= 0.0 (mx/item (:weight result))))
  ;; Weight should equal the log-prob of x=4.0 under gaussian(mu=1.0, sigma=1.0)
  ;; i.e., only the constrained site contributes to weight
  (assert-true "direct: weight ≠ score (unconstrained site adds to score only)"
    (not= (mx/item (:weight result)) (mx/item (:score result)))))

;; All constrained via direct call
(let [compiled-gen (:compiled-generate (:schema simple-model))
      key (rng/fresh-key 201)
      _ (rng/seed! key)
      constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
      result (compiled-gen key [(mx/scalar 0.0)] constraints)]
  (assert-close "direct all-constrained: weight = score"
    (mx/item (:score result)) (mx/item (:weight result)) 1e-6))

;; No constraints via direct call
(let [compiled-gen (:compiled-generate (:schema simple-model))
      key (rng/fresh-key 202)
      _ (rng/seed! key)
      result (compiled-gen key [(mx/scalar 0.0)] cm/EMPTY)]
  (assert-close "direct no-constraints: weight = 0"
    0.0 (mx/item (:weight result)) 1e-6))

;; ---- Summary ----
(println (str "\n== WP-1 Results: " @pass-count " passed, " @fail-count " failed"
              " (total " (+ @pass-count @fail-count) ") =="))
