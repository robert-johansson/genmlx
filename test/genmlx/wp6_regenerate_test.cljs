(ns genmlx.wp6-regenerate-test
  "WP-6 tests: compiled regenerate for DynamicGF.
   Validates that compiled paths produce identical scores/weights as handler.
   Key subtlety: regenerate samples, so dyn/with-key fixes the PRNG key
   for compiled vs handler comparison."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.selection :as sel]))

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
  "Strip ALL compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf)
                 :compiled-simulate :compiled-generate
                 :compiled-update :compiled-assess :compiled-project
                 :compiled-regenerate
                 :compiled-prefix :compiled-prefix-addrs
                 :compiled-prefix-generate :compiled-prefix-update
                 :compiled-prefix-assess :compiled-prefix-project
                 :compiled-prefix-regenerate)]
    (assoc gf :schema schema)))

(defn make-trace-via-generate
  "Create a deterministic trace by calling generate with all sites constrained."
  [gf args constraints]
  (let [gf-h (force-handler gf)
        {:keys [trace]} (p/generate (dyn/with-key gf-h (rng/fresh-key 1)) args constraints)]
    trace))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

;; M2: 2-site independent (y does NOT depend on x)
(def model-indep
  (gen [mu]
    (let [x (trace :x (dist/gaussian mu 1))
          y (trace :y (dist/gaussian 0 2))]
      (mx/add x y))))

;; M2: 2-site dependent (y depends on x)
(def model-dep
  (gen [mu]
    (let [x (trace :x (dist/gaussian mu 1))
          y (trace :y (dist/gaussian x 0.5))]
      y)))

;; M2: 3-site chain
(def model-chain
  (gen [a]
    (let [x (trace :x (dist/gaussian a 1))
          y (trace :y (dist/gaussian x 0.5))
          z (trace :z (dist/gaussian (mx/add x y) 1))]
      z)))

;; M2: multi-distribution
(def model-multi
  (gen []
    (let [g (trace :g (dist/gaussian 0 1))
          u (trace :u (dist/uniform 0 1))
          e (trace :e (dist/exponential 2))]
      g)))

;; Non-compilable (beta-dist has no noise transform)
(def model-beta
  (dyn/auto-key
    (gen [] (trace :x (dist/beta-dist 2 5)))))

;; M4: branch model
(def m4-branch
  (gen [flag]
    (let [x (if flag
              (trace :x (dist/gaussian 0 1))
              (trace :x (dist/gaussian 5 2)))
          y (trace :y (dist/gaussian x 0.5))]
      y)))

;; M3: partial model (prefix + loop)
(def m3-loop
  (gen [n]
    (let [x (trace :x (dist/gaussian 0 1))]
      (doseq [i (range (mx/item n))]
        (trace (keyword (str "y" i)) (dist/gaussian x 0.5)))
      x)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(println "\n== WP-6: Compiled Regenerate Tests ==")

;; ---- Section 0: Prerequisites ----
(println "\n-- 0. Prerequisites (compilation check) --")

(assert-true "model-indep has :compiled-regenerate"
  (some? (:compiled-regenerate (:schema model-indep))))

(assert-true "model-dep has :compiled-regenerate"
  (some? (:compiled-regenerate (:schema model-dep))))

(assert-true "model-chain has :compiled-regenerate"
  (some? (:compiled-regenerate (:schema model-chain))))

(assert-true "model-beta has NO :compiled-regenerate"
  (nil? (:compiled-regenerate (:schema model-beta))))

(assert-true "m4-branch has :compiled-regenerate"
  (some? (:compiled-regenerate (:schema m4-branch))))

(assert-true "m3-loop has :compiled-prefix-regenerate"
  (some? (:compiled-prefix-regenerate (:schema m3-loop))))

;; ---- Section 1: Select All ----
(println "\n-- 1. Select all --")

(let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 42)
      gf-c (dyn/with-key model-indep key)
      gf-h (dyn/with-key (force-handler model-indep) key)
      result-c (p/regenerate gf-c trace sel/all)
      result-h (p/regenerate gf-h trace sel/all)]
  (assert-close "select-all, indep: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "select-all, indep: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5)
  (assert-close "select-all, indep: weight = 0 (independent sites)"
    0.0 (:weight result-c) 1e-5))

(let [trace (make-trace-via-generate model-dep [(mx/scalar 2.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 43)
      gf-c (dyn/with-key model-dep key)
      gf-h (dyn/with-key (force-handler model-dep) key)
      result-c (p/regenerate gf-c trace sel/all)
      result-h (p/regenerate gf-h trace sel/all)]
  (assert-close "select-all, dep: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "select-all, dep: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5)
  ;; Retval should be computed
  (assert-true "select-all, dep: retval is finite"
    (js/isFinite (mx/item (:retval (:trace result-c))))))

;; ---- Section 2: Select None ----
(println "\n-- 2. Select none --")

(let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 44)
      gf-c (dyn/with-key model-indep key)
      gf-h (dyn/with-key (force-handler model-indep) key)
      result-c (p/regenerate gf-c trace sel/none)
      result-h (p/regenerate gf-h trace sel/none)]
  (assert-close "select-none: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "select-none: weight = 0"
    0.0 (:weight result-c) 1e-5)
  (assert-close "select-none: choices :x unchanged"
    1.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-6)
  (assert-close "select-none: choices :y unchanged"
    0.5 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)) 1e-6)
  ;; Retval: x + y = 1.0 + 0.5 = 1.5
  (assert-close "select-none: retval = x + y = 1.5"
    1.5 (:retval (:trace result-c)) 1e-5))

;; ---- Section 3: Partial Selection ----
(println "\n-- 3. Partial selection --")

;; Independent model: select :x only → weight = 0 (y doesn't depend on x)
(let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 50)
      gf-c (dyn/with-key model-indep key)
      gf-h (dyn/with-key (force-handler model-indep) key)
      result-c (p/regenerate gf-c trace (sel/select :x))
      result-h (p/regenerate gf-h trace (sel/select :x))]
  (assert-close "indep, select :x: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "indep, select :x: weight = 0 (independent)"
    0.0 (:weight result-c) 1e-5)
  (assert-close "indep, select :x: :y unchanged"
    0.5 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)) 1e-6))

;; Independent model: select :y only → weight = 0
(let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 51)
      gf-c (dyn/with-key model-indep key)
      gf-h (dyn/with-key (force-handler model-indep) key)
      result-c (p/regenerate gf-c trace (sel/select :y))
      result-h (p/regenerate gf-h trace (sel/select :y))]
  (assert-close "indep, select :y: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "indep, select :y: weight = 0 (independent)"
    0.0 (:weight result-c) 1e-5)
  (assert-close "indep, select :y: :x unchanged"
    1.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-6))

;; Dependent model: select :x → weight != 0 (y's lp changes when x changes)
(let [trace (make-trace-via-generate model-dep [(mx/scalar 2.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 52)
      gf-c (dyn/with-key model-dep key)
      gf-h (dyn/with-key (force-handler model-dep) key)
      result-c (p/regenerate gf-c trace (sel/select :x))
      result-h (p/regenerate gf-h trace (sel/select :x))]
  (assert-close "dep, select :x: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "dep, select :x: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5)
  ;; y keeps old value
  (assert-close "dep, select :x: :y unchanged"
    0.5 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)) 1e-6)
  ;; x gets resampled (with this key it should differ from 1.0)
  (assert-true "dep, select :x: :x resampled (finite)"
    (js/isFinite (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))))))

;; Chain model: select :x → both :y and :z log-probs change
(let [trace (make-trace-via-generate model-chain [(mx/scalar 0.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5) :z (mx/scalar 2.0)))
      key (rng/fresh-key 53)
      gf-c (dyn/with-key model-chain key)
      gf-h (dyn/with-key (force-handler model-chain) key)
      result-c (p/regenerate gf-c trace (sel/select :x))
      result-h (p/regenerate gf-h trace (sel/select :x))]
  (assert-close "chain, select :x: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "chain, select :x: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5)
  ;; y and z keep old values
  (assert-close "chain, select :x: :y unchanged"
    0.5 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)) 1e-6)
  (assert-close "chain, select :x: :z unchanged"
    2.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z)) 1e-6))

;; ---- Section 4: Multi-Distribution ----
(println "\n-- 4. Multi-distribution --")

(let [trace (make-trace-via-generate model-multi []
              (cm/choicemap :g (mx/scalar 0.5) :u (mx/scalar 0.3) :e (mx/scalar 0.1)))
      key (rng/fresh-key 60)]
  ;; Select :g only
  (let [gf-c (dyn/with-key model-multi key)
        gf-h (dyn/with-key (force-handler model-multi) key)
        result-c (p/regenerate gf-c trace (sel/select :g))
        result-h (p/regenerate gf-h trace (sel/select :g))]
    (assert-close "multi, select :g: score matches handler"
      (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
    (assert-close "multi, select :g: weight matches handler"
      (:weight result-h) (:weight result-c) 1e-5))

  ;; Select :u only
  (let [gf-c (dyn/with-key model-multi key)
        gf-h (dyn/with-key (force-handler model-multi) key)
        result-c (p/regenerate gf-c trace (sel/select :u))
        result-h (p/regenerate gf-h trace (sel/select :u))]
    (assert-close "multi, select :u: score matches handler"
      (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
    (assert-close "multi, select :u: weight matches handler"
      (:weight result-h) (:weight result-c) 1e-5))

  ;; Select all
  (let [gf-c (dyn/with-key model-multi key)
        gf-h (dyn/with-key (force-handler model-multi) key)
        result-c (p/regenerate gf-c trace sel/all)
        result-h (p/regenerate gf-h trace sel/all)]
    (assert-close "multi, select all: score matches handler"
      (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
    (assert-close "multi, select all: weight matches handler"
      (:weight result-h) (:weight result-c) 1e-5)))

;; ---- Section 5: M4 Branch Models ----
(println "\n-- 5. M4 branch models --")

;; flag=true, select :x
(let [trace (make-trace-via-generate m4-branch [true]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 70)
      gf-c (dyn/with-key m4-branch key)
      gf-h (dyn/with-key (force-handler m4-branch) key)
      result-c (p/regenerate gf-c trace (sel/select :x))
      result-h (p/regenerate gf-h trace (sel/select :x))]
  (assert-close "m4, flag=true, select :x: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "m4, flag=true, select :x: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5))

;; flag=false, select :x
(let [trace (make-trace-via-generate m4-branch [false]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 71)
      gf-c (dyn/with-key m4-branch key)
      gf-h (dyn/with-key (force-handler m4-branch) key)
      result-c (p/regenerate gf-c trace (sel/select :x))
      result-h (p/regenerate gf-h trace (sel/select :x))]
  (assert-close "m4, flag=false, select :x: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "m4, flag=false, select :x: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5))

;; flag=true, select all
(let [trace (make-trace-via-generate m4-branch [true]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 72)
      gf-c (dyn/with-key m4-branch key)
      gf-h (dyn/with-key (force-handler m4-branch) key)
      result-c (p/regenerate gf-c trace sel/all)
      result-h (p/regenerate gf-h trace sel/all)]
  (assert-close "m4, select all: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "m4, select all: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5))

;; flag=true, select none
(let [trace (make-trace-via-generate m4-branch [true]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 73)
      gf-c (dyn/with-key m4-branch key)
      gf-h (dyn/with-key (force-handler m4-branch) key)
      result-c (p/regenerate gf-c trace sel/none)
      result-h (p/regenerate gf-h trace sel/none)]
  (assert-close "m4, select none: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "m4, select none: weight = 0"
    0.0 (:weight result-c) 1e-5))

;; ---- Section 6: M3 Partial Models ----
(println "\n-- 6. M3 partial models --")

;; Select :x (prefix site)
(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 1.0) :y0 (mx/scalar 0.5) :y1 (mx/scalar -0.5)))
      key (rng/fresh-key 80)
      gf-c (dyn/with-key m3-loop key)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      result-c (p/regenerate gf-c trace (sel/select :x))
      result-h (p/regenerate gf-h trace (sel/select :x))]
  (assert-close "m3, select :x: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "m3, select :x: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5))

;; Select :y0 (dynamic site)
(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 1.0) :y0 (mx/scalar 0.5) :y1 (mx/scalar -0.5)))
      key (rng/fresh-key 81)
      gf-c (dyn/with-key m3-loop key)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      result-c (p/regenerate gf-c trace (sel/select :y0))
      result-h (p/regenerate gf-h trace (sel/select :y0))]
  (assert-close "m3, select :y0: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "m3, select :y0: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5))

;; Select all
(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 1.0) :y0 (mx/scalar 0.5) :y1 (mx/scalar -0.5)))
      key (rng/fresh-key 82)
      gf-c (dyn/with-key m3-loop key)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      result-c (p/regenerate gf-c trace sel/all)
      result-h (p/regenerate gf-h trace sel/all)]
  (assert-close "m3, select all: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "m3, select all: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5))

;; Select none
(let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
              (cm/choicemap :x (mx/scalar 1.0) :y0 (mx/scalar 0.5) :y1 (mx/scalar -0.5)))
      key (rng/fresh-key 83)
      gf-c (dyn/with-key m3-loop key)
      gf-h (dyn/with-key (force-handler m3-loop) key)
      result-c (p/regenerate gf-c trace sel/none)
      result-h (p/regenerate gf-h trace sel/none)]
  (assert-close "m3, select none: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "m3, select none: weight = 0"
    0.0 (:weight result-c) 1e-5))

;; ---- Section 7: Non-compilable Fallback ----
(println "\n-- 7. Non-compilable fallback --")

(let [trace (p/simulate model-beta [])
      key (rng/fresh-key 90)
      gf-c (dyn/with-key model-beta key)
      gf-h (dyn/with-key (force-handler model-beta) key)
      result-c (p/regenerate gf-c trace sel/all)
      result-h (p/regenerate gf-h trace sel/all)]
  (assert-close "beta: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-5)
  (assert-close "beta: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-5)
  (assert-true "beta: uses handler path (no compiled-regenerate)"
    (nil? (:compiled-regenerate (:schema model-beta)))))

;; ---- Section 8: Cross-Operation Consistency ----
(println "\n-- 8. Cross-operation consistency --")

(let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      key (rng/fresh-key 100)]
  ;; regenerate(none).trace.score = old trace.score
  (let [result (p/regenerate (dyn/with-key model-indep key) trace sel/none)]
    (assert-close "regen(none).score = old score"
      (:score trace) (:score (:trace result)) 1e-5)
    (assert-close "regen(none).weight = 0"
      0.0 (:weight result) 1e-5)
    ;; Choices unchanged
    (assert-close "regen(none).choices :x = old :x"
      (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
      (mx/item (cm/get-value (cm/get-submap (:choices (:trace result)) :x)))
      1e-6)
    (assert-close "regen(none).choices :y = old :y"
      (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
      (mx/item (cm/get-value (cm/get-submap (:choices (:trace result)) :y)))
      1e-6)))

;; regenerate(all) produces valid trace with finite score
(let [result (p/regenerate (dyn/with-key model-indep (rng/fresh-key 101))
               (make-trace-via-generate model-indep [(mx/scalar 2.0)]
                 (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
               sel/all)]
  (assert-true "regen(all): finite score"
    (js/isFinite (mx/item (:score (:trace result))))))

;; After regenerate, project(all) = new trace.score
(let [trace (make-trace-via-generate model-dep [(mx/scalar 2.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
      result (p/regenerate (dyn/with-key model-dep (rng/fresh-key 102)) trace (sel/select :x))
      new-trace (:trace result)
      proj (p/project (dyn/with-key model-dep (rng/fresh-key 103)) new-trace sel/all)]
  (assert-close "project(all) = new trace score"
    (:score new-trace) proj 1e-5))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------
(println (str "\n== WP-6 Results: " @pass-count "/" (+ @pass-count @fail-count) " passed =="))
