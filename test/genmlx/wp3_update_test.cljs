(ns genmlx.wp3-update-test
  "WP-3 tests: compiled update for static DynamicGF models.
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

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " msg " expected=" expected " actual=" actual)))))

(defn force-handler
  "Strip compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-update
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

;; Model with computed return value
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

;; Single-site model
(def single-site-model
  (dyn/auto-key
    (gen [mu sigma]
      (let [x (trace :x (dist/gaussian mu sigma))]
        x))))

;; Non-compilable model (has dynamic addresses) — should fall back to handler
(def dynamic-addr-model
  (dyn/auto-key
    (gen [n]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
        (doseq [i (range (mx/item n))]
          (trace (keyword (str "y" i)) (dist/gaussian x (mx/scalar 1.0))))
        x))))

;; ---------------------------------------------------------------------------
;; Helper: generate a trace via handler (ground truth), then test update paths
;; ---------------------------------------------------------------------------
;; Strategy: generate trace via handler to get deterministic starting point,
;; then update same trace via compiled and handler paths for fair comparison.

(defn make-trace-via-generate
  "Create a trace by calling generate with all sites constrained via handler path.
   This gives a deterministic trace regardless of compiled/handler simulate."
  [gf args constraints]
  (let [gf-h (force-handler gf)
        {:keys [trace]} (p/generate (dyn/with-key gf-h (rng/fresh-key 1)) args constraints)]
    trace))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(println "\n== WP-3: Compiled Update Tests ==")

;; ---- Section 1: Compilation check ----
(println "\n-- 1. Compilation check --")

(assert-true "static model has :compiled-update"
  (some? (:compiled-update (:schema simple-model))))

(assert-true "chain model has :compiled-update"
  (some? (:compiled-update (:schema chain-model))))

(assert-true "multi-dist model has :compiled-update"
  (some? (:compiled-update (:schema multi-dist-model))))

(assert-true "dynamic-addr model has no :compiled-update"
  (nil? (:compiled-update (:schema dynamic-addr-model))))

;; ---- Section 2: No constraints (trace unchanged) ----
(println "\n-- 2. No constraints --")

(let [;; Create trace via generate with known values
      trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      gf simple-model
      gf-h (force-handler simple-model)
      ;; Update same trace via both paths
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace cm/EMPTY)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace cm/EMPTY)]
  (assert-close "no-constraint: weight = 0"
    0.0 (:weight result-c) 1e-6)
  (assert-close "no-constraint: score unchanged"
    (:score trace) (:score (:trace result-c)) 1e-6)
  (assert-close "no-constraint: x unchanged"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))
    1e-10)
  (assert-close "no-constraint: y unchanged"
    (cm/get-value (cm/get-submap (:choices trace) :y))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))
    1e-10)
  (assert-close "no-constraint: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "no-constraint: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; ---- Section 3: Single site constrained ----
(println "\n-- 3. Single site constrained --")

(let [trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      gf simple-model
      gf-h (force-handler simple-model)
      constraints (cm/choicemap :x (mx/scalar 1.5))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "single-constrained: new x = 1.5"
    1.5 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-10)
  (assert-close "single-constrained: y kept from old trace"
    (cm/get-value (cm/get-submap (:choices trace) :y))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))
    1e-10)
  (assert-close "single-constrained: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "single-constrained: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6)
  (assert-close "single-constrained: discard has old x"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:discard result-c) :x))
    1e-10)
  (assert-true "single-constrained: discard does not have y"
    (not (cm/has-value? (cm/get-submap (:discard result-c) :y)))))

;; ---- Section 4: All sites constrained ----
(println "\n-- 4. All sites constrained --")

(let [trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      gf simple-model
      gf-h (force-handler simple-model)
      constraints (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "all-constrained: x = 2.0"
    2.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-10)
  (assert-close "all-constrained: y = 3.0"
    3.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)) 1e-10)
  (assert-close "all-constrained: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "all-constrained: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6)
  (assert-close "all-constrained: discard has old x"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:discard result-c) :x))
    1e-10))

;; ---- Section 5: Dependency chain (upstream change affects downstream score) ----
(println "\n-- 5. Dependency chain --")

(let [trace (make-trace-via-generate chain-model [(mx/scalar 0.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.8) :z (mx/scalar 2.0)))
      gf chain-model
      gf-h (force-handler chain-model)
      ;; Constrain :x → changes dist params for :y (depends on x) and :z (depends on x+y)
      constraints (cm/choicemap :x (mx/scalar 5.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "chain: constrained x = 5.0"
    5.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-10)
  (assert-close "chain: y kept from old trace"
    (cm/get-value (cm/get-submap (:choices trace) :y))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))
    1e-10)
  (assert-close "chain: z kept from old trace"
    (cm/get-value (cm/get-submap (:choices trace) :z))
    (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z))
    1e-10)
  (assert-close "chain: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "chain: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6)
  (assert-close "chain: discard only has x"
    (cm/get-value (cm/get-submap (:choices trace) :x))
    (cm/get-value (cm/get-submap (:discard result-c) :x))
    1e-10))

;; ---- Section 6: Multi-distribution model ----
(println "\n-- 6. Multi-distribution --")

(let [trace (make-trace-via-generate multi-dist-model []
              (cm/choicemap :g (mx/scalar 0.3) :u (mx/scalar 0.5)
                            :b (mx/scalar 1.0) :e (mx/scalar 0.2)))
      gf multi-dist-model
      gf-h (force-handler multi-dist-model)
      constraints (cm/choicemap :g (mx/scalar 0.5))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "multi-dist: constrained g = 0.5"
    0.5 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :g)) 1e-10)
  (assert-close "multi-dist: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "multi-dist: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6)
  (assert-close "multi-dist: discard correct"
    (cm/get-value (cm/get-submap (:choices trace) :g))
    (cm/get-value (cm/get-submap (:discard result-c) :g))
    1e-10))

;; ---- Section 7: Idempotent update (constrain to same values) ----
(println "\n-- 7. Idempotent update --")

(let [trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      old-x (cm/get-value (cm/get-submap (:choices trace) :x))
      old-y (cm/get-value (cm/get-submap (:choices trace) :y))
      gf simple-model
      ;; Constrain to same values
      constraints (cm/choicemap :x old-x :y old-y)
      result (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)]
  (assert-close "idempotent: weight = 0"
    0.0 (:weight result) 1e-6)
  (assert-close "idempotent: score unchanged"
    (:score trace) (:score (:trace result)) 1e-6)
  (assert-close "idempotent: discard x = old x"
    old-x (cm/get-value (cm/get-submap (:discard result) :x)) 1e-10)
  (assert-close "idempotent: discard y = old y"
    old-y (cm/get-value (cm/get-submap (:discard result) :y)) 1e-10))

;; ---- Section 8: Delta distribution ----
(println "\n-- 8. Delta distribution --")

(let [trace (make-trace-via-generate delta-model [(mx/scalar 3.0)]
              (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 2.5)))
      gf delta-model
      gf-h (force-handler delta-model)
      ;; Update with no constraints — delta site keeps deterministic value
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace cm/EMPTY)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace cm/EMPTY)]
  (assert-close "delta: x = 3.0 (deterministic)"
    3.0 (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)) 1e-10)
  (assert-close "delta: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "delta: weight = 0"
    0.0 (:weight result-c) 1e-6))

;; ---- Section 9: Round-trip: generate → update → update ----
(println "\n-- 9. Round-trip chained updates --")

(let [;; Step 1: create trace with known values
      trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
              (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
      gf simple-model
      gf-h (force-handler simple-model)
      ;; Step 2: first update — constrain :x
      c1 (cm/choicemap :x (mx/scalar 1.0))
      r1-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace c1)
      r1-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace c1)
      ;; Step 3: second update on compiled result — constrain :y
      c2 (cm/choicemap :y (mx/scalar -1.0))
      r2-c (p/update (dyn/with-key gf (rng/fresh-key 88)) (:trace r1-c) c2)
      r2-h (p/update (dyn/with-key gf-h (rng/fresh-key 88)) (:trace r1-h) c2)]
  (assert-close "chain-update-1: score matches handler"
    (:score (:trace r1-h)) (:score (:trace r1-c)) 1e-6)
  (assert-close "chain-update-1: weight matches handler"
    (:weight r1-h) (:weight r1-c) 1e-6)
  (assert-close "chain-update-2: score matches handler"
    (:score (:trace r2-h)) (:score (:trace r2-c)) 1e-6)
  (assert-close "chain-update-2: weight matches handler"
    (:weight r2-h) (:weight r2-c) 1e-6)
  (assert-close "chain-update-2: x = 1.0 (from step 2)"
    1.0 (cm/get-value (cm/get-submap (:choices (:trace r2-c)) :x)) 1e-10)
  (assert-close "chain-update-2: y = -1.0 (from step 3)"
    -1.0 (cm/get-value (cm/get-submap (:choices (:trace r2-c)) :y)) 1e-10))

;; ---- Section 10: Score/weight mathematical correctness ----
(println "\n-- 10. Mathematical correctness --")

;; 1-site gaussian: weight = log-prob(new) - log-prob(old)
(let [mu (mx/scalar 0.0) sigma (mx/scalar 1.0)
      trace (make-trace-via-generate single-site-model [mu sigma]
              (cm/choicemap :x (mx/scalar 0.3)))
      old-x (cm/get-value (cm/get-submap (:choices trace) :x))
      new-x (mx/scalar 2.0)
      constraints (cm/choicemap :x new-x)
      result (p/update (dyn/with-key single-site-model (rng/fresh-key 99)) trace constraints)
      ;; Compute expected weight analytically: gaussian log-prob
      ;; log-prob(x|mu,sigma) = -0.5*log(2*pi) - log(sigma) - 0.5*((x-mu)/sigma)^2
      log2pi (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
      lp-new (- (- log2pi) (* 0.5 (* 2.0 2.0)))  ;; x=2, mu=0, sigma=1
      lp-old (- (- log2pi) (* 0.5 (* 0.3 0.3)))]  ;; x=0.3, mu=0, sigma=1
  (assert-close "analytical: weight = new_lp - old_lp"
    (- lp-new lp-old) (:weight result) 1e-5))

;; No-op update: weight exactly 0
(let [trace (make-trace-via-generate single-site-model [(mx/scalar 0.0) (mx/scalar 1.0)]
              (cm/choicemap :x (mx/scalar 0.5)))
      result (p/update (dyn/with-key single-site-model (rng/fresh-key 99)) trace cm/EMPTY)]
  (assert-close "no-op: weight = 0"
    0.0 (:weight result) 1e-10))

;; Constrain to extreme: weight is large negative
(let [trace (make-trace-via-generate single-site-model [(mx/scalar 0.0) (mx/scalar 1.0)]
              (cm/choicemap :x (mx/scalar 0.0)))
      constraints (cm/choicemap :x (mx/scalar 100.0))
      result (p/update (dyn/with-key single-site-model (rng/fresh-key 99)) trace constraints)]
  (assert-true "extreme: weight is very negative"
    (< (mx/item (:weight result)) -100.0)))

;; Constrain to mode: weight is non-negative (moving towards higher density)
(let [trace (make-trace-via-generate single-site-model [(mx/scalar 0.0) (mx/scalar 1.0)]
              (cm/choicemap :x (mx/scalar 3.0)))
      ;; Constrain to mode (mu=0)
      constraints (cm/choicemap :x (mx/scalar 0.0))
      result (p/update (dyn/with-key single-site-model (rng/fresh-key 99)) trace constraints)]
  (assert-true "mode: weight >= 0 when constraining to mode"
    (>= (mx/item (:weight result)) (- 1e-10))))

;; ---- Section 11: Return value ----
(println "\n-- 11. Return value --")

(let [trace (make-trace-via-generate retval-model [(mx/scalar 1.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0)))
      gf retval-model
      gf-h (force-handler retval-model)
      constraints (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "retval: x+y = 5.0"
    5.0 (:retval (:trace result-c)) 1e-10)
  (assert-close "retval: matches handler"
    (:retval (:trace result-h)) (:retval (:trace result-c)) 1e-10))

;; ---- Section 12: Edge cases ----
(println "\n-- 12. Edge cases --")

;; Single-site model
(let [trace (make-trace-via-generate single-site-model [(mx/scalar 0.0) (mx/scalar 1.0)]
              (cm/choicemap :x (mx/scalar 0.5)))
      gf single-site-model
      gf-h (force-handler single-site-model)
      constraints (cm/choicemap :x (mx/scalar 1.0))
      result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
      result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
  (assert-close "single-site: score matches handler"
    (:score (:trace result-h)) (:score (:trace result-c)) 1e-6)
  (assert-close "single-site: weight matches handler"
    (:weight result-h) (:weight result-c) 1e-6))

;; Empty constraints ChoiceMap
(let [trace (make-trace-via-generate chain-model [(mx/scalar 0.0)]
              (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.8) :z (mx/scalar 2.0)))
      result (p/update (dyn/with-key chain-model (rng/fresh-key 99)) trace cm/EMPTY)]
  (assert-close "empty-cm: weight = 0"
    0.0 (:weight result) 1e-6)
  (assert-true "empty-cm: discard is empty"
    (= cm/EMPTY (:discard result))))

;; ---- Section 13: gen-fn accessor ----
(println "\n-- 13. Accessor --")

(assert-true "get-compiled-update returns fn for static model"
  (fn? (compiled/get-compiled-update simple-model)))
(assert-true "get-compiled-update returns nil for dynamic model"
  (nil? (compiled/get-compiled-update dynamic-addr-model)))

;; ---- Summary ----
(println (str "\n== WP-3 Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed =="))
(when (pos? @fail-count)
  (println (str "FAILURES: " @fail-count)))
