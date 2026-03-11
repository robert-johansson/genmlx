(ns genmlx.wp9b-fused-loop-test
  "WP-9B tests: fused unfold/scan simulate.
   Validates that fused loop execution (single mx/compile-fn dispatch for T steps)
   produces correct trace structure, scores, and state threading.

   PRNG divergence: fused and per-step paths consume randomness differently.
   Tests use:
   - Deterministic kernels (delta) for exact equivalence
   - Structural properties for stochastic kernels (score finite, correct addrs)
   - Statistical properties are NOT tested (too flaky for CI)"
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.combinators :as comb]))

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

(println "\n=== WP-9B: Fused Unfold/Scan Simulate Tests ===\n")

;; ---------------------------------------------------------------------------
;; Test kernels
;; ---------------------------------------------------------------------------

;; Simple 1-site scalar kernel (unfold) — fusable
(def k-simple
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/gaussian state 0.1))]
      x))))

;; 2-site kernel (unfold) — fusable
(def k-2site
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/gaussian state 0.1))
          y (trace :y (dist/gaussian x 0.5))]
      x))))

;; Deterministic kernel (unfold) — for exact equivalence tests
(def k-delta
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/delta (mx/add state (mx/scalar 1.0))))]
      x))))

;; Scan kernel: 1-site
(def k-scan
  (dyn/auto-key (gen [carry input]
    (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
      [x x]))))

;; Deterministic scan kernel
(def k-scan-delta
  (dyn/auto-key (gen [carry input]
    (let [x (trace :x (dist/delta (mx/add carry input)))]
      [x (mx/multiply x (mx/scalar 2.0))]))))

;; Non-fusable kernel (beta-dist has no noise transform)
(def k-beta
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/beta-dist 2 5))]
      x))))

;; ---------------------------------------------------------------------------
;; Section 0: Prerequisites — fusability detection
;; ---------------------------------------------------------------------------

(println "-- Section 0: Fusability prerequisites --")

(assert-true "k-simple has compiled-simulate"
  (some? (compiled/get-compiled-simulate k-simple)))

(assert-true "k-beta lacks compiled-simulate (non-fusable)"
  (nil? (compiled/get-compiled-simulate k-beta)))

(assert-true "k-scan has compiled-simulate"
  (some? (compiled/get-compiled-simulate k-scan)))

;; ---------------------------------------------------------------------------
;; Section 1: Fused unfold T=5 simple kernel
;; ---------------------------------------------------------------------------

(println "\n-- Section 1: Fused unfold T=5 simple kernel --")

(let [unfold (comb/unfold-combinator k-simple)
      trace (p/simulate unfold [5 (mx/scalar 0.0)])]
  ;; Score is finite
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace))))
  ;; Correct number of steps
  (assert-equal "retval count" 5 (count (:retval trace)))
  ;; Choices structure: 5 steps, each with :x
  (doseq [t (range 5)]
    (let [sub (cm/get-submap (:choices trace) t)]
      (assert-true (str "step " t " has :x")
        (cm/has-value? (cm/get-submap sub :x))))))

;; ---------------------------------------------------------------------------
;; Section 2: Fused unfold T=10 — larger loop
;; ---------------------------------------------------------------------------

(println "\n-- Section 2: Fused unfold T=10 --")

(let [unfold (comb/unfold-combinator k-simple)
      trace (p/simulate unfold [10 (mx/scalar 1.0)])]
  (mx/eval! (:score trace))
  (assert-true "T=10 score finite" (js/isFinite (mx/item (:score trace))))
  (assert-equal "T=10 retval count" 10 (count (:retval trace)))
  ;; Final state is an MLX array
  (let [final (last (:retval trace))]
    (mx/eval! final)
    (assert-true "final state is finite" (js/isFinite (mx/item final)))))

;; ---------------------------------------------------------------------------
;; Section 3: Fused unfold 2-site kernel
;; ---------------------------------------------------------------------------

(println "\n-- Section 3: Fused unfold 2-site kernel --")

(let [unfold (comb/unfold-combinator k-2site)
      trace (p/simulate unfold [5 (mx/scalar 0.0)])]
  (mx/eval! (:score trace))
  (assert-true "2-site score finite" (js/isFinite (mx/item (:score trace))))
  ;; Both addresses present at each step
  (doseq [t (range 5)]
    (let [sub (cm/get-submap (:choices trace) t)]
      (assert-true (str "step " t " has :x")
        (cm/has-value? (cm/get-submap sub :x)))
      (assert-true (str "step " t " has :y")
        (cm/has-value? (cm/get-submap sub :y))))))

;; ---------------------------------------------------------------------------
;; Section 4: Deterministic kernel — fused vs per-step exact equivalence
;; ---------------------------------------------------------------------------

(println "\n-- Section 4: Deterministic kernel equivalence --")

;; Delta kernel: state(t) = init + t + 1 (adds 1 at each step)
;; score = 0 (delta log-prob in simulate is 0)
;; Both fused and per-step should produce IDENTICAL results
(let [unfold-c (comb/unfold-combinator k-delta)
      unfold-h (comb/unfold-combinator (force-handler k-delta))
      trace-c (p/simulate unfold-c [5 (mx/scalar 0.0)])
      trace-h (p/simulate unfold-h [5 (mx/scalar 0.0)])]
  (mx/eval! (:score trace-c))
  (mx/eval! (:score trace-h))
  (assert-close "delta score compiled" 0.0 (mx/item (:score trace-c)) 1e-6)
  (assert-close "delta score handler" 0.0 (mx/item (:score trace-h)) 1e-6)
  ;; State threading: each step adds 1
  ;; state0 = 0+1=1, state1 = 1+1=2, ..., state4 = 4+1=5
  (doseq [t (range 5)]
    (let [sub-c (cm/get-submap (cm/get-submap (:choices trace-c) t) :x)
          sub-h (cm/get-submap (cm/get-submap (:choices trace-h) t) :x)
          val-c (mx/item (cm/get-value sub-c))
          val-h (mx/item (cm/get-value sub-h))]
      (assert-close (str "step " t " value match") val-h val-c 1e-6))))

;; ---------------------------------------------------------------------------
;; Section 5: State threading — values depend on previous state
;; ---------------------------------------------------------------------------

(println "\n-- Section 5: State threading --")

(let [unfold (comb/unfold-combinator k-simple)
      trace (p/simulate unfold [3 (mx/scalar 5.0)])]
  ;; Step 0 should sample near 5.0 (gaussian(5.0, 0.1))
  ;; Step 1 should sample near step 0's value
  ;; We can verify state threading by checking that values are not independent
  ;; (i.e., each step's mean is near the previous step's value)
  ;; For a single sample, we just verify the structure is correct
  (let [v0 (mx/item (cm/get-value (cm/get-submap (cm/get-submap (:choices trace) 0) :x)))
        v1 (mx/item (cm/get-value (cm/get-submap (cm/get-submap (:choices trace) 1) :x)))]
    ;; Both should be near 5.0 (since sigma=0.1, very tight)
    (assert-true "step 0 near init (sigma=0.1)"
      (< (js/Math.abs (- v0 5.0)) 1.0))
    (assert-true "step 1 near step 0 (sigma=0.1)"
      (< (js/Math.abs (- v1 v0)) 1.0))))

;; ---------------------------------------------------------------------------
;; Section 6: Fused scan T=5
;; ---------------------------------------------------------------------------

(println "\n-- Section 6: Fused scan T=5 --")

(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
              (mx/scalar 4.0) (mx/scalar 5.0)]
      trace (p/simulate scan [(mx/scalar 0.0) inputs])]
  (mx/eval! (:score trace))
  (assert-true "scan score finite" (js/isFinite (mx/item (:score trace))))
  ;; Retval structure
  (assert-true "has carry" (some? (:carry (:retval trace))))
  (assert-equal "outputs count" 5 (count (:outputs (:retval trace))))
  ;; Choices at each step
  (doseq [t (range 5)]
    (assert-true (str "scan step " t " has :x")
      (cm/has-value? (cm/get-submap (cm/get-submap (:choices trace) t) :x)))))

;; Deterministic scan: carry threading
(println "\n-- Section 6b: Deterministic scan carry threading --")

(let [scan-c (comb/scan-combinator k-scan-delta)
      scan-h (comb/scan-combinator (force-handler k-scan-delta))
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      trace-c (p/simulate scan-c [(mx/scalar 0.0) inputs])
      trace-h (p/simulate scan-h [(mx/scalar 0.0) inputs])]
  (mx/eval! (:score trace-c))
  (mx/eval! (:score trace-h))
  ;; carry(0) = 0+1=1, carry(1) = 1+2=3, carry(2) = 3+3=6
  (let [carry-c (:carry (:retval trace-c))
        carry-h (:carry (:retval trace-h))]
    (mx/eval! carry-c) (mx/eval! carry-h)
    (assert-close "scan final carry compiled" 6.0 (mx/item carry-c) 1e-5)
    (assert-close "scan final carry handler" 6.0 (mx/item carry-h) 1e-5))
  ;; Outputs: output = x * 2, so [2, 6, 12]
  (let [outputs-c (:outputs (:retval trace-c))]
    (doseq [[i expected] [[0 2.0] [1 6.0] [2 12.0]]]
      (mx/eval! (nth outputs-c i))
      (assert-close (str "scan output " i) expected (mx/item (nth outputs-c i)) 1e-5))))

;; ---------------------------------------------------------------------------
;; Section 7: Non-fusable fallback
;; ---------------------------------------------------------------------------

(println "\n-- Section 7: Non-fusable fallback --")

(let [unfold (comb/unfold-combinator k-beta)
      trace (p/simulate unfold [3 (mx/scalar 0.5)])]
  (assert-true "beta fallback valid" (instance? tr/Trace trace))
  (mx/eval! (:score trace))
  (assert-true "beta fallback score finite" (js/isFinite (mx/item (:score trace)))))

;; ---------------------------------------------------------------------------
;; Section 8: Step-scores metadata preserved
;; ---------------------------------------------------------------------------

(println "\n-- Section 8: Step-scores metadata --")

(let [unfold (comb/unfold-combinator k-simple)
      trace (p/simulate unfold [5 (mx/scalar 0.0)])]
  (let [ss (::comb/step-scores (meta trace))]
    (assert-true "has step-scores" (some? ss))
    (assert-equal "step-scores count" 5 (count ss))
    ;; Sum of step scores should equal total score
    (doseq [s ss] (mx/eval! s))
    (mx/eval! (:score trace))
    (assert-close "step-scores sum = total"
      (mx/item (:score trace))
      (reduce + (map mx/item ss))
      1e-5)))

;; ---------------------------------------------------------------------------
;; Section 8b: Fused path detection
;; ---------------------------------------------------------------------------

(println "\n-- Section 8b: Fused path detection --")

(assert-true "k-simple is fusable" (compiled/fusable-kernel? k-simple))
(assert-true "k-beta is NOT fusable" (not (compiled/fusable-kernel? k-beta)))

(let [unfold (comb/unfold-combinator k-simple)
      trace (p/simulate unfold [5 (mx/scalar 0.0)])]
  (assert-true "unfold fused path used" (::comb/fused (meta trace))))

(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      trace (p/simulate scan [(mx/scalar 0.0) inputs])]
  (assert-true "scan fused path used" (::comb/fused (meta trace))))

;; ---------------------------------------------------------------------------
;; Section 9: Variable T — different T values work
;; ---------------------------------------------------------------------------

(println "\n-- Section 9: Variable T --")

(let [unfold (comb/unfold-combinator k-simple)
      trace5 (p/simulate unfold [5 (mx/scalar 0.0)])
      trace10 (p/simulate unfold [10 (mx/scalar 0.0)])]
  (mx/eval! (:score trace5))
  (mx/eval! (:score trace10))
  (assert-true "T=5 valid" (js/isFinite (mx/item (:score trace5))))
  (assert-true "T=10 valid" (js/isFinite (mx/item (:score trace10))))
  (assert-equal "T=5 retval count" 5 (count (:retval trace5)))
  (assert-equal "T=10 retval count" 10 (count (:retval trace10))))

;; ===========================================================================
;; Phase C: Fused Map Simulate
;; ===========================================================================

;; Map kernel: 1-site
(def k-map
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/gaussian x 1.0))]
      y))))

;; Deterministic map kernel
(def k-map-delta
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/delta (mx/multiply x (mx/scalar 2.0))))]
      y))))

;; Non-fusable map kernel
(def k-map-beta
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/beta-dist 2 5))]
      y))))

;; ---------------------------------------------------------------------------
;; Section 10: Fused map N=5
;; ---------------------------------------------------------------------------

(println "\n-- Section 10: Fused map N=5 --")

(let [mapped (comb/map-combinator k-map)
      args [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
             (mx/scalar 4.0) (mx/scalar 5.0)]]
      trace (p/simulate mapped args)]
  (mx/eval! (:score trace))
  (assert-true "map score finite" (js/isFinite (mx/item (:score trace))))
  (assert-equal "map retval count" 5 (count (:retval trace)))
  ;; Each element has :y
  (doseq [i (range 5)]
    (assert-true (str "elem " i " has :y")
      (cm/has-value? (cm/get-submap (cm/get-submap (:choices trace) i) :y)))))

;; ---------------------------------------------------------------------------
;; Section 11: Fused map deterministic equivalence
;; ---------------------------------------------------------------------------

(println "\n-- Section 11: Fused map deterministic equivalence --")

(let [mapped-c (comb/map-combinator k-map-delta)
      mapped-h (comb/map-combinator (force-handler k-map-delta))
      args [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
      trace-c (p/simulate mapped-c args)
      trace-h (p/simulate mapped-h args)]
  (mx/eval! (:score trace-c))
  (mx/eval! (:score trace-h))
  (assert-close "delta map score compiled" 0.0 (mx/item (:score trace-c)) 1e-6)
  (assert-close "delta map score handler" 0.0 (mx/item (:score trace-h)) 1e-6)
  ;; Values: y = x * 2
  (doseq [i (range 3)]
    (let [val-c (mx/item (cm/get-value (cm/get-submap (cm/get-submap (:choices trace-c) i) :y)))
          val-h (mx/item (cm/get-value (cm/get-submap (cm/get-submap (:choices trace-h) i) :y)))]
      (assert-close (str "elem " i " value match") val-h val-c 1e-6))))

;; ---------------------------------------------------------------------------
;; Section 12: Fused map path detection
;; ---------------------------------------------------------------------------

(println "\n-- Section 12: Fused map path detection --")

(assert-true "k-map is fusable" (compiled/fusable-kernel? k-map))
(assert-true "k-map-beta is NOT fusable" (not (compiled/fusable-kernel? k-map-beta)))

(let [mapped (comb/map-combinator k-map)
      trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]])]
  (assert-true "map fused path used" (::comb/fused (meta trace))))

;; ---------------------------------------------------------------------------
;; Section 13: Non-fusable map fallback
;; ---------------------------------------------------------------------------

(println "\n-- Section 13: Non-fusable map fallback --")

(let [mapped (comb/map-combinator k-map-beta)
      trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]])]
  (assert-true "beta map fallback valid" (instance? tr/Trace trace))
  (mx/eval! (:score trace))
  (assert-true "beta map fallback score finite" (js/isFinite (mx/item (:score trace)))))

;; ---------------------------------------------------------------------------
;; Section 14: Fused map element-scores metadata
;; ---------------------------------------------------------------------------

(println "\n-- Section 14: Fused map element-scores metadata --")

(let [mapped (comb/map-combinator k-map)
      trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]])]
  (let [es (::comb/element-scores (meta trace))]
    (assert-true "has element-scores" (some? es))
    (assert-equal "element-scores count" 3 (count es))
    (doseq [s es] (mx/eval! s))
    (mx/eval! (:score trace))
    (assert-close "element-scores sum = total"
      (mx/item (:score trace))
      (reduce + (map mx/item es))
      1e-5)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n=== WP-9B Results ===")
(println (str "  Passed: " @pass-count "/" (+ @pass-count @fail-count)))
(println (str "  Failed: " @fail-count))
(when (pos? @fail-count)
  (println "  *** FAILURES DETECTED ***"))
