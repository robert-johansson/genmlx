(ns genmlx.wp4-update-test
  "WP-4 tests: compiled update for M4 (branch-rewritten) and M3 (partial prefix) models.
   Validates that compiled update matches handler update exactly."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]))

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

(defn force-handler
  "Strip all compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-update
                       :compiled-prefix :compiled-prefix-addrs
                       :compiled-prefix-generate :compiled-prefix-update)]
    (assoc gf :schema schema)))

(defn make-trace-via-generate
  "Create a trace by calling generate with all sites constrained via handler path."
  [gf args constraints]
  (let [gf-h (force-handler gf)
        {:keys [trace]} (p/generate (dyn/with-key gf-h (rng/fresh-key 1)) args constraints)]
    trace))

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

;; Multi-branch model: two branch sites + computed retval
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

;; Dependent prefix: b depends on a, dynamic tail depends on both
(def m3-dep-prefix
  (dyn/auto-key
    (gen []
      (let [a (trace :a (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/gaussian a (mx/scalar 0.5)))]
        (doseq [i (range 2)]
          (trace (keyword (str "c" i)) (dist/gaussian (mx/add a b) (mx/scalar 1.0))))
        (mx/add a b)))))

;; ===========================================================================
(println "\n== WP-4: Compiled Update for M4 + M3 Models ==")
;; ===========================================================================

;; ============================
;; PART A: M4 Branch-Rewritten
;; ============================

;; ---- Section 1: Compilation check ----
(println "\n-- 1. M4 compilation check --")

(assert-true "m4-simple has :compiled-update"
  (some? (:compiled-update (:schema m4-simple))))
(assert-true "m4-multi-site has :compiled-update"
  (some? (:compiled-update (:schema m4-multi-site))))
(assert-true "m3-loop has :compiled-prefix-update (not :compiled-update)"
  (and (some? (:compiled-prefix-update (:schema m3-loop)))
       (nil? (:compiled-update (:schema m3-loop)))))

;; ---- Section 2: M4 no constraints ----
(println "\n-- 2. M4 no constraints --")

(let [trace (make-trace-via-generate m4-simple [true]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      gf m4-simple
      gf-h (force-handler m4-simple)
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace cm/EMPTY)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace cm/EMPTY)]
  (assert-close "m4 no-cst: weight = 0"
    0.0 (:weight result-c) 1e-6)
  (assert-close "m4 no-cst: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m4 no-cst: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; ---- Section 3: M4 single site constrained ----
(println "\n-- 3. M4 single site constrained --")

(let [trace (make-trace-via-generate m4-simple [true]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      gf m4-simple
      gf-h (force-handler m4-simple)
      constraints (cm/choicemap :x (mx/scalar 2.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m4 single: x = 2.0"
    2.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-10)
  (assert-close "m4 single: y kept"
    (cm/get-value (cm/get-submap (:choices trace) :y))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))
    1e-10)
  (assert-close "m4 single: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m4 single: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6)
  (assert-close "m4 single: discard has old x"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:discard result-c) :x))
    1e-10))

;; ---- Section 4: M4 all sites constrained ----
(println "\n-- 4. M4 all sites constrained --")

(let [trace (make-trace-via-generate m4-simple [true]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      gf m4-simple
      gf-h (force-handler m4-simple)
      constraints (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 4.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m4 all: x = 3.0"
    3.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-10)
  (assert-close "m4 all: y = 4.0"
    4.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)) 1e-10)
  (assert-close "m4 all: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m4 all: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; ---- Section 5: M4 branch condition true vs false ----
(println "\n-- 5. M4 branch conditions --")

;; True branch
(let [trace (make-trace-via-generate m4-simple [true]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      gf m4-simple
      gf-h (force-handler m4-simple)
      constraints (cm/choicemap :x (mx/scalar 0.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m4 true-branch: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m4 true-branch: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; False branch
(let [trace (make-trace-via-generate m4-simple [false]
              (cm/choicemap :x (mx/scalar -0.5) :y (mx/scalar 0.0)))
      gf m4-simple
      gf-h (force-handler m4-simple)
      constraints (cm/choicemap :x (mx/scalar 1.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m4 false-branch: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m4 false-branch: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; ---- Section 6: M4 retval ----
(println "\n-- 6. M4 retval --")

(let [trace (make-trace-via-generate m4-multi-site [true]
              (cm/choicemap :a (mx/scalar 2.0) :b (mx/scalar 3.0)))
      gf m4-multi-site
      gf-h (force-handler m4-multi-site)
      constraints (cm/choicemap :a (mx/scalar 4.0) :b (mx/scalar 5.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m4 retval: a+b = 9.0"
    9.0 (:retval (:trace result-c)) 1e-10)
  (assert-close "m4 retval: matches handler"
    (:retval (:trace result-h)) (:retval (:trace result-c)) 1e-10))

;; ---- Section 7: M4 multi-site with dependency chain ----
(println "\n-- 7. M4 multi-site dependency --")

(let [trace (make-trace-via-generate m4-multi-site [true]
              (cm/choicemap :a (mx/scalar 5.0) :b (mx/scalar 6.0)))
      gf m4-multi-site
      gf-h (force-handler m4-multi-site)
      ;; Constrain :a → changes dist params for :b (depends on a)
      constraints (cm/choicemap :a (mx/scalar 10.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m4 dep: a = 10.0"
    10.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :a)) 1e-10)
  (assert-close "m4 dep: b kept"
    (cm/get-value (cm/get-submap (:choices trace) :b))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :b))
    1e-10)
  (assert-close "m4 dep: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m4 dep: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; ---- Section 8: M4 idempotent ----
(println "\n-- 8. M4 idempotent --")

(let [trace (make-trace-via-generate m4-simple [true]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      gf m4-simple
      constraints (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0))
      result (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)]
  (assert-close "m4 idempotent: weight = 0"
    0.0 (:weight result) 1e-6)
  (assert-close "m4 idempotent: score unchanged"
    (:score trace) (:score (:trace result)) 1e-6))

;; ============================
;; PART B: M3 Partial Prefix
;; ============================

;; ---- Section 9: M3 compilation check ----
(println "\n-- 9. M3 compilation check --")

(assert-true "m3-loop has :compiled-prefix-update"
  (some? (:compiled-prefix-update (:schema m3-loop))))
(assert-true "m3-dep-prefix has :compiled-prefix-update"
  (some? (:compiled-prefix-update (:schema m3-dep-prefix))))

;; ---- Section 10: M3 no constraints ----
(println "\n-- 10. M3 no constraints --")

(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                            :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
      gf m3-loop
      gf-h (force-handler m3-loop)
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace cm/EMPTY)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace cm/EMPTY)]
  (assert-close "m3 no-cst: weight = 0"
    0.0 (:weight result-c) 1e-6)
  (assert-close "m3 no-cst: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m3 no-cst: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6)
  (assert-close "m3 no-cst: x unchanged"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))
    1e-10))

;; ---- Section 11: M3 prefix site constrained ----
(println "\n-- 11. M3 prefix site constrained --")

(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                            :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
      gf m3-loop
      gf-h (force-handler m3-loop)
      constraints (cm/choicemap :x (mx/scalar 3.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m3 prefix-cst: x = 3.0"
    3.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-10)
  (assert-close "m3 prefix-cst: z0 kept"
    (cm/get-value (cm/get-submap (:choices trace) :z0))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z0))
    1e-10)
  (assert-close "m3 prefix-cst: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m3 prefix-cst: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6)
  (assert-close "m3 prefix-cst: discard has old x"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:discard result-c) :x))
    1e-10))

;; ---- Section 12: M3 dynamic site constrained ----
(println "\n-- 12. M3 dynamic site constrained --")

(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                            :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
      gf m3-loop
      gf-h (force-handler m3-loop)
      constraints (cm/choicemap :z0 (mx/scalar 5.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m3 dyn-cst: z0 = 5.0"
    5.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z0)) 1e-10)
  (assert-close "m3 dyn-cst: x kept"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))
    1e-10)
  (assert-close "m3 dyn-cst: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m3 dyn-cst: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; ---- Section 13: M3 both prefix + dynamic constrained ----
(println "\n-- 13. M3 both constrained --")

(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                            :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
      gf m3-loop
      gf-h (force-handler m3-loop)
      constraints (cm/choicemap :x (mx/scalar 2.0) :z1 (mx/scalar 3.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m3 both: x = 2.0"
    2.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-10)
  (assert-close "m3 both: z1 = 3.0"
    3.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z1)) 1e-10)
  (assert-close "m3 both: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m3 both: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6)
  ;; Discard should have both prefix and dynamic old values
  (assert-close "m3 both: discard has old x"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:discard result-c) :x))
    1e-10)
  (assert-close "m3 both: discard has old z1"
    (cm/get-value (cm/get-submap (:choices trace) :z1))
    (cm/get-value (cm/get-submap (:discard result-c) :z1))
    1e-10))

;; ---- Section 14: M3 idempotent ----
(println "\n-- 14. M3 idempotent --")

(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                            :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
      gf m3-loop
      constraints (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                                :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2))
      result (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)]
  (assert-close "m3 idempotent: weight = 0"
    0.0 (:weight result) 1e-6)
  (assert-close "m3 idempotent: score unchanged"
    (:score trace) (:score (:trace result)) 1e-6))

;; ---- Section 15: M3 chained updates ----
(println "\n-- 15. M3 chained updates --")

(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                            :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
      gf m3-loop
      gf-h (force-handler m3-loop)
      ;; Update 1: constrain prefix
      c1 (cm/choicemap :x (mx/scalar 2.0))
      r1-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace c1)
      r1-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace c1)
      ;; Update 2: constrain dynamic
      c2 (cm/choicemap :z0 (mx/scalar 4.0))
      r2-c (p/update (dyn/with-key gf (rng/fresh-key 88)) (:trace r1-c) c2)
      r2-h (p/update (dyn/with-key gf-h (rng/fresh-key 88)) (:trace r1-h) c2)]
  (assert-close "m3 chain-1: score matches handler"
    (:score (:trace r1-h)) (:score (:trace r1-c)) 1e-6)
  (assert-close "m3 chain-1: weight matches handler"
    (:weight r1-h) (:weight r1-c) 1e-6)
  (assert-close "m3 chain-2: score matches handler"
    (:score (:trace r2-h)) (:score (:trace r2-c)) 1e-6)
  (assert-close "m3 chain-2: weight matches handler"
    (:weight r2-h) (:weight r2-c) 1e-6))

;; ---- Section 16: M3 retval ----
(println "\n-- 16. M3 retval --")

(let [trace (make-trace-via-generate m3-dep-prefix []
              (cm/choicemap :a (mx/scalar 1.0) :b (mx/scalar 2.0)
                            :c0 (mx/scalar 3.0) :c1 (mx/scalar 4.0)))
      gf m3-dep-prefix
      gf-h (force-handler m3-dep-prefix)
      constraints (cm/choicemap :a (mx/scalar 2.0) :b (mx/scalar 3.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m3 retval: a+b = 5.0"
    5.0 (:retval (:trace result-c)) 1e-10)
  (assert-close "m3 retval: matches handler"
    (:retval (:trace result-h)) (:retval (:trace result-c)) 1e-10))

;; ---- Section 17: M3 dep-prefix upstream change ----
(println "\n-- 17. M3 dep-prefix upstream --")

(let [trace (make-trace-via-generate m3-dep-prefix []
              (cm/choicemap :a (mx/scalar 1.0) :b (mx/scalar 2.0)
                            :c0 (mx/scalar 3.0) :c1 (mx/scalar 4.0)))
      gf m3-dep-prefix
      gf-h (force-handler m3-dep-prefix)
      ;; Constrain :a → changes dist params for :b (depends on a) and :c0/:c1 (depend on a+b)
      constraints (cm/choicemap :a (mx/scalar 5.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "m3 dep: a = 5.0"
    5.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :a)) 1e-10)
  (assert-close "m3 dep: b kept"
    (cm/get-value (cm/get-submap (:choices trace) :b))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :b))
    1e-10)
  (assert-close "m3 dep: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "m3 dep: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; ---- Summary ----
(println (str "\n== WP-4 Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed =="))
(when (pos? @fail-count)
  (println (str "FAILURES: " @fail-count)))
