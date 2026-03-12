(ns genmlx.wp2-generate-test
  "WP-2 tests: compiled generate for M4 (branch-rewritten) and M3 (partial prefix) models.
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
                       :compiled-prefix :compiled-prefix-addrs
                       :compiled-prefix-generate)]
    (assoc gf :schema schema)))

;; ---------------------------------------------------------------------------
;; M4 test models (branch-rewritten)
;; ---------------------------------------------------------------------------

;; Simple if-model: boolean param selects different gaussian args
(def m4-simple
  (dyn/auto-key
    (gen [flag]
      (let [x (if flag
                (trace :x (dist/gaussian (mx/scalar 1.0) (mx/scalar 0.5)))
                (trace :x (dist/gaussian (mx/scalar -1.0) (mx/scalar 2.0))))
            y (trace :y (dist/gaussian x (mx/scalar 1.0)))]
        y))))

;; Multi-dist branch model: gaussian in true branch, gaussian in false (different params)
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

;; Loop model: static prefix + doseq dynamic tail
(def m3-loop
  (dyn/auto-key
    (gen [n]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 2.0)))]
        (doseq [i (range (mx/item n))]
          (trace (keyword (str "z" i)) (dist/gaussian y (mx/scalar 0.5))))
        y))))

;; Dependent prefix model: b depends on a
(def m3-dep-prefix
  (dyn/auto-key
    (gen []
      (let [a (trace :a (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/gaussian a (mx/scalar 0.5)))]
        (doseq [i (range 2)]
          (trace (keyword (str "c" i)) (dist/gaussian (mx/add a b) (mx/scalar 1.0))))
        (mx/add a b)))))

;; Prefix with gen args
(def m3-args
  (dyn/auto-key
    (gen [mu sigma]
      (let [x (trace :x (dist/gaussian mu sigma))
            y (trace :y (dist/gaussian x (mx/scalar 1.0)))]
        (doseq [i (range 3)]
          (trace (keyword (str "d" i)) (dist/gaussian y (mx/scalar 0.5))))
        y))))

;; Multi-dist prefix: gaussian + uniform + bernoulli + exponential before dynamic
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
(println "\n== WP-2: Compiled Generate for M4 + M3 Models ==")
;; ===========================================================================

;; ============================
;; PART A: M4 Branch-Rewritten
;; ============================

;; ---- Section 1: M4 compilation presence ----
(println "\n-- 1. M4 compilation presence --")

(assert-true "m4-simple has :compiled-generate"
  (some? (:compiled-generate (:schema m4-simple))))

(assert-true "m4-multi-site has :compiled-generate"
  (some? (:compiled-generate (:schema m4-multi-site))))

;; ---- Section 2: M4 all constrained ----
(println "\n-- 2. M4 all sites constrained --")

(let [key (rng/fresh-key 42)
      constraints (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 5.0))
      gf (dyn/with-key m4-simple key)
      {:keys [trace weight]} (p/generate gf [true] constraints)
      ;; Handler reference
      gf-h (dyn/with-key (force-handler m4-simple) key)
      ref (p/generate gf-h [true] constraints)]
  (assert-close "m4 all-constrained: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m4 all-constrained: weight matches handler"
    (:weight ref) weight 1e-5)
  (assert-close "m4 all-constrained: weight = score"
    (:score trace) weight 1e-5)
  (assert-close "m4 all-constrained: x = constraint"
    3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
  (assert-close "m4 all-constrained: y = constraint"
    5.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y))) 1e-6))

;; ---- Section 3: M4 no constraints ----
(println "\n-- 3. M4 no constraints --")

(let [key (rng/fresh-key 77)
      gf (dyn/with-key m4-simple key)
      {:keys [trace weight]} (p/generate gf [true] cm/EMPTY)
      gf-h (dyn/with-key (force-handler m4-simple) key)
      ref (p/generate gf-h [true] cm/EMPTY)]
  (assert-close "m4 no-constraints: weight = 0" 0.0 weight 1e-6)
  (assert-close "m4 no-constraints: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m4 no-constraints: x matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :x)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
    1e-5))

;; ---- Section 4: M4 mixed constraints ----
(println "\n-- 4. M4 mixed constraints --")

(let [key (rng/fresh-key 99)
      constraints (cm/choicemap :x (mx/scalar 2.0))
      gf (dyn/with-key m4-simple key)
      {:keys [trace weight]} (p/generate gf [false] constraints)
      gf-h (dyn/with-key (force-handler m4-simple) key)
      ref (p/generate gf-h [false] constraints)]
  (assert-close "m4 mixed: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m4 mixed: weight matches handler"
    (:weight ref) weight 1e-5)
  (assert-close "m4 mixed: x = constraint"
    2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6))

;; ---- Section 5: M4 multi-site branch model ----
(println "\n-- 5. M4 multi-site branch --")

(let [key (rng/fresh-key 55)
      constraints (cm/choicemap :a (mx/scalar 3.0) :b (mx/scalar 4.0))
      gf (dyn/with-key m4-multi-site key)
      {:keys [trace weight]} (p/generate gf [true] constraints)
      gf-h (dyn/with-key (force-handler m4-multi-site) key)
      ref (p/generate gf-h [true] constraints)]
  (assert-close "m4 multi all: score matches" (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m4 multi all: weight matches" (:weight ref) weight 1e-5))

;; Partial constraints on multi-site
(let [key (rng/fresh-key 55)
      constraints (cm/choicemap :a (mx/scalar 3.0))
      gf (dyn/with-key m4-multi-site key)
      {:keys [trace weight]} (p/generate gf [false] constraints)
      gf-h (dyn/with-key (force-handler m4-multi-site) key)
      ref (p/generate gf-h [false] constraints)]
  (assert-close "m4 multi partial: score matches" (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m4 multi partial: weight matches" (:weight ref) weight 1e-5)
  (assert-close "m4 multi partial: b matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :b)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :b)))
    1e-5))

;; ---- Section 6: M4 PRNG equivalence ----
(println "\n-- 6. M4 PRNG equivalence --")

(let [key (rng/fresh-key 123)
      gf (dyn/with-key m4-simple key)
      {:keys [trace]} (p/generate gf [true] cm/EMPTY)
      gf-s (dyn/with-key m4-simple key)
      sim-trace (p/simulate gf-s [true])]
  (assert-close "m4 prng: generate(empty) x = simulate x"
    (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :x)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
    1e-5)
  (assert-close "m4 prng: generate(empty) y = simulate y"
    (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5))

;; ============================
;; PART B: M3 Partial Generate
;; ============================

;; ---- Section 7: M3 compilation presence ----
(println "\n-- 7. M3 compilation presence --")

(assert-true "m3-loop has :compiled-prefix-generate"
  (some? (:compiled-prefix-generate (:schema m3-loop))))

(assert-true "m3-dep-prefix has :compiled-prefix-generate"
  (some? (:compiled-prefix-generate (:schema m3-dep-prefix))))

(assert-true "m3-args has :compiled-prefix-generate"
  (some? (:compiled-prefix-generate (:schema m3-args))))

(assert-true "m3-multi-dist has :compiled-prefix-generate"
  (some? (:compiled-prefix-generate (:schema m3-multi-dist))))

;; ---- Section 8: M3 prefix-only constraints ----
(println "\n-- 8. M3 prefix-only constraints --")

(let [key (rng/fresh-key 42)
      constraints (cm/choicemap :x (mx/scalar 2.0))
      gf (dyn/with-key m3-loop key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 3)] constraints)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      ref (p/generate gf-h [(mx/scalar 3)] constraints)]
  (assert-close "m3 pfx-only: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m3 pfx-only: weight matches handler"
    (:weight ref) weight 1e-5)
  (assert-close "m3 pfx-only: x = constraint"
    2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6))

;; ---- Section 9: M3 dynamic-only constraints ----
(println "\n-- 9. M3 dynamic-only constraints --")

(let [key (rng/fresh-key 77)
      constraints (cm/choicemap :z0 (mx/scalar 1.5))
      gf (dyn/with-key m3-loop key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 2)] constraints)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      ref (p/generate gf-h [(mx/scalar 2)] constraints)]
  (assert-close "m3 dyn-only: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m3 dyn-only: weight matches handler"
    (:weight ref) weight 1e-5)
  ;; Prefix sites should match handler (same PRNG, unconstrained)
  (assert-close "m3 dyn-only: x matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :x)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
    1e-5)
  (assert-close "m3 dyn-only: y matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5))

;; ---- Section 10: M3 mixed prefix + dynamic constraints ----
(println "\n-- 10. M3 mixed constraints --")

(let [key (rng/fresh-key 88)
      constraints (cm/choicemap :x (mx/scalar 1.5) :z1 (mx/scalar 3.0))
      gf (dyn/with-key m3-loop key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 3)] constraints)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      ref (p/generate gf-h [(mx/scalar 3)] constraints)]
  (assert-close "m3 mixed: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m3 mixed: weight matches handler"
    (:weight ref) weight 1e-5)
  (assert-close "m3 mixed: x = constraint"
    1.5 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6)
  (assert-close "m3 mixed: z1 = constraint"
    3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :z1))) 1e-6))

;; ---- Section 11: M3 no constraints = simulate ----
(println "\n-- 11. M3 no constraints = simulate --")

(let [key (rng/fresh-key 99)
      gf (dyn/with-key m3-loop key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 2)] cm/EMPTY)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      ref (p/generate gf-h [(mx/scalar 2)] cm/EMPTY)]
  (assert-close "m3 no-constraints: weight = 0" 0.0 weight 1e-6)
  (assert-close "m3 no-constraints: score matches handler"
    (:score (:trace ref)) (:score trace) 1e-5)
  ;; Check all values match handler
  (assert-close "m3 no-constraints: x matches"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :x)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
    1e-5)
  (assert-close "m3 no-constraints: y matches"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5))

;; ---- Section 12: M3 all constrained ----
(println "\n-- 12. M3 all constrained --")

(let [key (rng/fresh-key 11)
      constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0)
                                :z0 (mx/scalar 3.0) :z1 (mx/scalar 4.0))
      gf (dyn/with-key m3-loop key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 2)] constraints)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      ref (p/generate gf-h [(mx/scalar 2)] constraints)]
  (assert-close "m3 all: score matches" (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m3 all: weight matches" (:weight ref) weight 1e-5)
  (assert-close "m3 all: weight = score (all constrained)"
    (:score trace) weight 1e-5))

;; ---- Section 13: M3 dependent prefix ----
(println "\n-- 13. M3 dependent prefix --")

(let [key (rng/fresh-key 33)
      constraints (cm/choicemap :a (mx/scalar 1.0) :c0 (mx/scalar 5.0))
      gf (dyn/with-key m3-dep-prefix key)
      {:keys [trace weight]} (p/generate gf [] constraints)
      gf-h (dyn/with-key (force-handler m3-dep-prefix) key)
      ref (p/generate gf-h [] constraints)]
  (assert-close "m3 dep: score matches" (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m3 dep: weight matches" (:weight ref) weight 1e-5)
  (assert-close "m3 dep: a = constraint"
    1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :a))) 1e-6)
  ;; b is in prefix (depends on a), should be sampled and match handler
  (assert-close "m3 dep: b matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :b)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :b)))
    1e-5))

;; ---- Section 14: M3 prefix with gen args ----
(println "\n-- 14. M3 prefix with gen args --")

(let [key (rng/fresh-key 44)
      constraints (cm/choicemap :x (mx/scalar 3.0))
      gf (dyn/with-key m3-args key)
      {:keys [trace weight]} (p/generate gf [(mx/scalar 5.0) (mx/scalar 2.0)] constraints)
      gf-h (dyn/with-key (force-handler m3-args) key)
      ref (p/generate gf-h [(mx/scalar 5.0) (mx/scalar 2.0)] constraints)]
  (assert-close "m3 args: score matches" (:score (:trace ref)) (:score trace) 1e-5)
  (assert-close "m3 args: weight matches" (:weight ref) weight 1e-5)
  (assert-close "m3 args: x = constraint"
    3.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6))

;; ---- Section 15: M3 multi-dist prefix ----
(println "\n-- 15. M3 multi-dist prefix --")

(let [key (rng/fresh-key 55)
      constraints (cm/choicemap :g (mx/scalar 0.5) :u (mx/scalar 0.3))
      gf (dyn/with-key m3-multi-dist key)
      {:keys [trace weight]} (p/generate gf [] constraints)
      gf-h (dyn/with-key (force-handler m3-multi-dist) key)
      ref (p/generate gf-h [] constraints)]
  (assert-close "m3 multi: score matches" (:score (:trace ref)) (:score trace) 1e-4)
  (assert-close "m3 multi: weight matches" (:weight ref) weight 1e-4)
  (assert-close "m3 multi: g = constraint"
    0.5 (mx/item (cm/get-value (cm/get-submap (:choices trace) :g))) 1e-6)
  (assert-close "m3 multi: u = constraint"
    0.3 (mx/item (cm/get-value (cm/get-submap (:choices trace) :u))) 1e-6)
  ;; b and e are unconstrained prefix sites
  (assert-close "m3 multi: b matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :b)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :b)))
    1e-5)
  (assert-close "m3 multi: e matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :e)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :e)))
    1e-5))

;; ---- Section 16: M3 PRNG consistency ----
(println "\n-- 16. M3 PRNG consistency --")

;; generate(empty) = simulate
(let [key (rng/fresh-key 66)
      gf (dyn/with-key m3-loop key)
      {:keys [trace]} (p/generate gf [(mx/scalar 2)] cm/EMPTY)
      gf-s (dyn/with-key m3-loop key)
      sim-trace (p/simulate gf-s [(mx/scalar 2)])]
  (assert-close "m3 prng: generate(empty) x = simulate x"
    (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :x)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
    1e-5)
  (assert-close "m3 prng: generate(empty) y = simulate y"
    (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5)
  (assert-close "m3 prng: generate(empty) z0 = simulate z0"
    (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :z0)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z0)))
    1e-5)
  (assert-close "m3 prng: generate(empty) z1 = simulate z1"
    (mx/item (cm/get-value (cm/get-submap (:choices sim-trace) :z1)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z1)))
    1e-5))

;; Constrained prefix → dynamic sites get correct keys
(let [key (rng/fresh-key 66)
      constraints (cm/choicemap :x (mx/scalar 1.0))
      gf (dyn/with-key m3-loop key)
      {:keys [trace]} (p/generate gf [(mx/scalar 2)] constraints)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      ref (p/generate gf-h [(mx/scalar 2)] constraints)]
  (assert-close "m3 prng-constrained: y matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :y)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
    1e-5)
  (assert-close "m3 prng-constrained: z0 matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :z0)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z0)))
    1e-5)
  (assert-close "m3 prng-constrained: z1 matches handler"
    (mx/item (cm/get-value (cm/get-submap (:choices (:trace ref)) :z1)))
    (mx/item (cm/get-value (cm/get-submap (:choices trace) :z1)))
    1e-5))

;; ---- Section 17: M3 trace structure ----
(println "\n-- 17. M3 trace structure --")

(let [key (rng/fresh-key 88)
      constraints (cm/choicemap :x (mx/scalar 1.0))
      gf (dyn/with-key m3-loop key)
      {:keys [trace]} (p/generate gf [(mx/scalar 3)] constraints)]
  (assert-true "m3 trace: has :gen-fn" (= m3-loop (:gen-fn trace)))
  (assert-true "m3 trace: has x" (cm/has-value? (cm/get-submap (:choices trace) :x)))
  (assert-true "m3 trace: has y" (cm/has-value? (cm/get-submap (:choices trace) :y)))
  (assert-true "m3 trace: has z0" (cm/has-value? (cm/get-submap (:choices trace) :z0)))
  (assert-true "m3 trace: has z1" (cm/has-value? (cm/get-submap (:choices trace) :z1)))
  (assert-true "m3 trace: has z2" (cm/has-value? (cm/get-submap (:choices trace) :z2)))
  (assert-true "m3 trace: has retval" (some? (:retval trace)))
  (assert-true "m3 trace: has score" (some? (:score trace))))

;; ---- Section 18: M3 retval correctness ----
(println "\n-- 18. M3 retval correctness --")

(let [key (rng/fresh-key 22)
      constraints (cm/choicemap :a (mx/scalar 2.0) :b (mx/scalar 3.0))
      gf (dyn/with-key m3-dep-prefix key)
      {:keys [trace]} (p/generate gf [] constraints)]
  ;; m3-dep-prefix returns (mx/add a b)
  (assert-close "m3 retval: add(a,b) = 5.0"
    5.0 (mx/item (:retval trace)) 1e-6))

;; ---- Summary ----
(println (str "\n== WP-2 Results: " @pass-count " passed, " @fail-count " failed"
              " (total " (+ @pass-count @fail-count) ") =="))
