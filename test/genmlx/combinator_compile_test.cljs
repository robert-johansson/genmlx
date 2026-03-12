(ns genmlx.combinator-compile-test
  "L1-M5: Combinator-aware compilation tests.
   Tests that combinators with compilable kernels bypass per-step
   Trace/ChoiceMap construction by calling compiled-simulate directly."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn assert-true [msg actual]
  (if actual
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg "- expected truthy"))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc) (println "  PASS:" msg))
      (do (swap! fail-count inc)
          (println "  FAIL:" msg
                   "- expected" expected "got" actual "diff" diff "tol" tol)))))

(defn assert= [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc)
        (println "  FAIL:" msg)
        (println "    expected:" (pr-str expected))
        (println "    actual:  " (pr-str actual)))))

(defn compilable?
  "Check if a gen-fn kernel has a compiled-simulate function."
  [gf]
  (some? (:compiled-simulate (:schema gf))))

(println "\n=== L1-M5: Combinator Compilation Tests ===\n")

;; ---------------------------------------------------------------------------
;; Section 0: Kernel compilability verification
;; ---------------------------------------------------------------------------
;; Verify that test kernels are (or aren't) compilable at L1-M2.
;; This section is diagnostic — it confirms test prerequisites.

(println "-- Kernel compilability --")

(let [k-unfold (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   next)))
      k-map    (dyn/auto-key (gen [x]
                 (let [y (trace :y (dist/gaussian x 1.0))]
                   y)))
      k-scan   (dyn/auto-key (gen [carry input]
                 (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                   [x x])))
      k-switch (dyn/auto-key (gen []
                 (let [x (trace :x (dist/gaussian 0 1))]
                   x)))
      k-beta   (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/beta-dist 2 5))]
                   next)))]
  (assert-true "unfold kernel compilable" (compilable? k-unfold))
  (assert-true "map kernel compilable" (compilable? k-map))
  (assert-true "scan kernel compilable" (compilable? k-scan))
  (assert-true "switch kernel compilable" (compilable? k-switch))
  (assert-true "beta kernel NOT compilable" (not (compilable? k-beta))))

;; ---------------------------------------------------------------------------
;; Section 1: Unfold with compilable kernel
;; ---------------------------------------------------------------------------

(println "\n-- Unfold: compilable kernel correctness --")
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/gaussian state 0.1))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [5 (mx/scalar 0.0)])]
  ;; Trace structure
  (assert-true "trace is Trace" (instance? tr/Trace trace))
  (assert= "retval count" 5 (count (:retval trace)))
  ;; Choices: keys 0-4, each with :x
  (doseq [t (range 5)]
    (let [sub (cm/get-submap (:choices trace) t)]
      (assert-true (str "step " t " has :x")
        (cm/has-value? (cm/get-submap sub :x)))))
  ;; Score is finite
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace))))
  ;; Step-scores metadata
  (let [ss (::comb/step-scores (meta trace))]
    (assert-true "has step-scores" (some? ss))
    (assert= "step-scores count" 5 (count ss))
    (doseq [s ss] (mx/eval! s))
    (assert-close "step-scores sum = total score"
      (mx/item (:score trace))
      (reduce + (map mx/item ss))
      1e-5)))

(println "\n-- Unfold: compiled path detection --")
;; After M5, combinator sets ::comb/compiled-path on trace metadata.
;; FAILS before M5 implementation.
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/gaussian state 0.1))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [5 (mx/scalar 0.0)])]
  (assert-true "compiled path used"
    (::comb/compiled-path (meta trace))))

(println "\n-- Unfold: non-compilable fallback --")
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/beta-dist 2 5))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [3 (mx/scalar 0.5)])]
  (assert-true "fallback trace valid" (instance? tr/Trace trace))
  (assert= "fallback retval count" 3 (count (:retval trace)))
  (mx/eval! (:score trace))
  (assert-true "fallback score finite" (js/isFinite (mx/item (:score trace))))
  (assert-true "fallback NOT compiled"
    (not (::comb/compiled-path (meta trace)))))

;; ---------------------------------------------------------------------------
;; Section 2: Scan with compilable kernel
;; ---------------------------------------------------------------------------

(println "\n-- Scan: compilable kernel correctness --")
(let [kernel (dyn/auto-key (gen [carry input]
               (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                 [x x])))
      scan (comb/scan-combinator kernel)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
              (mx/scalar 4.0) (mx/scalar 5.0)]
      trace (p/simulate scan [(mx/scalar 0.0) inputs])]
  (assert-true "trace is Trace" (instance? tr/Trace trace))
  (assert-true "retval has :carry" (some? (:carry (:retval trace))))
  (assert= "outputs count" 5 (count (:outputs (:retval trace))))
  ;; Choices at each step
  (doseq [t (range 5)]
    (assert-true (str "step " t " has :x")
      (cm/has-value? (cm/get-submap (cm/get-submap (:choices trace) t) :x))))
  ;; Score
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace))))
  ;; Metadata
  (assert-true "has step-scores" (some? (::comb/step-scores (meta trace))))
  (assert-true "has step-carries" (some? (::comb/step-carries (meta trace)))))

(println "\n-- Scan: compiled path detection --")
(let [kernel (dyn/auto-key (gen [carry input]
               (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                 [x x])))
      scan (comb/scan-combinator kernel)
      trace (p/simulate scan [(mx/scalar 0.0) [(mx/scalar 1.0)]])]
  (assert-true "compiled path used"
    (::comb/compiled-path (meta trace))))

(println "\n-- Scan: carry threading (delta) --")
;; Deterministic carry threading with delta distribution.
;; carry_0=0, input=[1,2,3] → carries=[1,3,6]
(let [kernel (dyn/auto-key (gen [carry input]
               (let [x (trace :x (dist/delta carry))]
                 [(mx/add carry input) (mx/add carry input)])))
      scan (comb/scan-combinator kernel)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      trace (p/simulate scan [(mx/scalar 0.0) inputs])]
  (let [carry (:carry (:retval trace))]
    (mx/eval! carry)
    (assert-close "final carry = 6" 6.0 (mx/item carry) 1e-5))
  (let [outputs (:outputs (:retval trace))]
    (doseq [[i expected] [[0 1.0] [1 3.0] [2 6.0]]]
      (mx/eval! (nth outputs i))
      (assert-close (str "output " i) expected (mx/item (nth outputs i)) 1e-5)))
  ;; Delta log-prob=0, total score=0
  (mx/eval! (:score trace))
  (assert-close "delta score = 0" 0.0 (mx/item (:score trace)) 1e-6))

;; ---------------------------------------------------------------------------
;; Section 3: Map with compilable kernel
;; ---------------------------------------------------------------------------

(println "\n-- Map: compilable kernel correctness --")
(let [kernel (dyn/auto-key (gen [x]
               (let [y (trace :y (dist/gaussian x 1.0))]
                 y)))
      mapped (comb/map-combinator kernel)
      trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]])]
  (assert-true "trace is Trace" (instance? tr/Trace trace))
  (assert= "retval count" 3 (count (:retval trace)))
  ;; Choices
  (doseq [i (range 3)]
    (assert-true (str "element " i " has :y")
      (cm/has-value? (cm/get-submap (cm/get-submap (:choices trace) i) :y))))
  ;; Score
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace))))
  ;; Element-scores metadata
  (assert-true "has element-scores"
    (some? (::comb/element-scores (meta trace)))))

(println "\n-- Map: compiled path detection --")
(let [kernel (dyn/auto-key (gen [x]
               (let [y (trace :y (dist/gaussian x 1.0))]
                 y)))
      mapped (comb/map-combinator kernel)
      trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]])]
  (assert-true "compiled path used"
    (::comb/compiled-path (meta trace))))

(println "\n-- Map: non-compilable fallback --")
(let [kernel (dyn/auto-key (gen [x]
               (let [y (trace :y (dist/beta-dist 2 5))]
                 y)))
      mapped (comb/map-combinator kernel)
      trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]])]
  (assert-true "fallback trace valid" (instance? tr/Trace trace))
  (assert= "fallback retval count" 2 (count (:retval trace)))
  (assert-true "fallback NOT compiled"
    (not (::comb/compiled-path (meta trace)))))

;; ---------------------------------------------------------------------------
;; Section 4: Switch with compilable branch
;; ---------------------------------------------------------------------------

(println "\n-- Switch: compilable branch correctness --")
(let [branch0 (dyn/auto-key (gen []
                (let [x (trace :x (dist/gaussian 0 1))]
                  x)))
      branch1 (dyn/auto-key (gen []
                (let [x (trace :x (dist/gaussian 10 1))]
                  x)))
      sw (comb/switch-combinator branch0 branch1)
      t0 (p/simulate sw [0])
      t1 (p/simulate sw [1])]
  ;; Branch 0
  (assert-true "branch 0 trace" (instance? tr/Trace t0))
  (let [x0 (cm/get-value (cm/get-submap (:choices t0) :x))]
    (mx/eval! x0)
    (assert-true "branch 0 near 0" (< (js/Math.abs (mx/item x0)) 5)))
  ;; Branch 1
  (let [x1 (cm/get-value (cm/get-submap (:choices t1) :x))]
    (mx/eval! x1)
    (assert-true "branch 1 near 10" (< (js/Math.abs (- (mx/item x1) 10)) 5)))
  ;; Scores finite
  (mx/eval! (:score t0))
  (mx/eval! (:score t1))
  (assert-true "branch 0 score finite" (js/isFinite (mx/item (:score t0))))
  (assert-true "branch 1 score finite" (js/isFinite (mx/item (:score t1))))
  ;; Switch-idx metadata preserved
  (assert= "branch 0 idx" 0 (::comb/switch-idx (meta t0)))
  (assert= "branch 1 idx" 1 (::comb/switch-idx (meta t1))))

(println "\n-- Switch: compiled path detection --")
(let [branch0 (dyn/auto-key (gen []
                (let [x (trace :x (dist/gaussian 0 1))]
                  x)))
      sw (comb/switch-combinator branch0)
      trace (p/simulate sw [0])]
  (assert-true "compiled path used"
    (::comb/compiled-path (meta trace))))

(println "\n-- Switch: non-compilable fallback --")
(let [branch0 (dyn/auto-key (gen []
                (let [x (trace :x (dist/beta-dist 2 5))]
                  x)))
      sw (comb/switch-combinator branch0)
      trace (p/simulate sw [0])]
  (assert-true "fallback trace valid" (instance? tr/Trace trace))
  (assert-true "fallback NOT compiled"
    (not (::comb/compiled-path (meta trace)))))

(println "\n-- Switch: mixed branches --")
;; Branch 0 compilable, branch 1 non-compilable
(let [branch0 (dyn/auto-key (gen []
                (let [x (trace :x (dist/gaussian 0 1))]
                  x)))
      branch1 (dyn/auto-key (gen []
                (let [x (trace :x (dist/beta-dist 2 5))]
                  x)))
      sw (comb/switch-combinator branch0 branch1)]
  (let [t0 (p/simulate sw [0])]
    (assert-true "mixed: compilable branch compiled"
      (::comb/compiled-path (meta t0))))
  (let [t1 (p/simulate sw [1])]
    (assert-true "mixed: non-compilable branch NOT compiled"
      (not (::comb/compiled-path (meta t1))))))

;; ---------------------------------------------------------------------------
;; Section 5: Mix with compilable component
;; ---------------------------------------------------------------------------

(println "\n-- Mix: compilable component correctness --")
(let [comp0 (dyn/auto-key (gen []
              (let [x (trace :x (dist/gaussian 0 1))]
                x)))
      comp1 (dyn/auto-key (gen []
              (let [x (trace :x (dist/gaussian 10 1))]
                x)))
      log-w (mx/log (mx/array #js [0.5 0.5]))
      mix (comb/mix-combinator [comp0 comp1] log-w)
      trace (p/simulate mix [])]
  (assert-true "trace is Trace" (instance? tr/Trace trace))
  ;; Has :component-idx
  (let [idx (cm/get-choice (:choices trace) [:component-idx])]
    (mx/eval! idx)
    (assert-true "has component-idx" (contains? #{0 1} (int (mx/item idx)))))
  ;; Has component's :x
  (assert-true "has :x" (cm/has-value? (cm/get-submap (:choices trace) :x)))
  ;; Score finite
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- Mix: compiled path detection --")
(let [comp0 (dyn/auto-key (gen []
              (let [x (trace :x (dist/gaussian 0 1))]
                x)))
      log-w (mx/array #js [0.0])
      mix (comb/mix-combinator [comp0] log-w)
      trace (p/simulate mix [])]
  (assert-true "compiled path used"
    (::comb/compiled-path (meta trace))))

;; ---------------------------------------------------------------------------
;; Section 6: Edge cases
;; ---------------------------------------------------------------------------

(println "\n-- Edge: single step unfold --")
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/gaussian state 0.1))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [1 (mx/scalar 0.0)])]
  (assert= "retval count" 1 (count (:retval trace)))
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace))))
  (assert= "step-scores count" 1
    (count (::comb/step-scores (meta trace)))))

(println "\n-- Edge: zero steps unfold --")
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/gaussian state 0.1))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [0 (mx/scalar 0.0)])]
  (assert= "retval count" 0 (count (:retval trace)))
  (mx/eval! (:score trace))
  (assert-close "score = 0" 0.0 (mx/item (:score trace)) 1e-6)
  (assert= "empty choices" cm/EMPTY (:choices trace)))

(println "\n-- Edge: delta-only unfold --")
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/delta state))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [3 (mx/scalar 5.0)])]
  (mx/eval! (:score trace))
  (assert-close "delta score = 0" 0.0 (mx/item (:score trace)) 1e-6)
  (doseq [rv (:retval trace)]
    (mx/eval! rv)
    (assert-close "delta retval = 5.0" 5.0 (mx/item rv) 1e-6)))

(println "\n-- Edge: multi-site kernel --")
(let [kernel (dyn/auto-key (gen [t state]
               (let [a (trace :a (dist/gaussian state 1))
                     b (trace :b (dist/gaussian a 0.5))]
                 b)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [3 (mx/scalar 0.0)])]
  (doseq [t (range 3)]
    (let [sub (cm/get-submap (:choices trace) t)]
      (assert-true (str "step " t " has :a")
        (cm/has-value? (cm/get-submap sub :a)))
      (assert-true (str "step " t " has :b")
        (cm/has-value? (cm/get-submap sub :b)))))
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- Edge: extra args unfold --")
(let [kernel (dyn/auto-key (gen [t state scale]
               (let [next (trace :x (dist/gaussian state scale))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [3 (mx/scalar 0.0) (mx/scalar 2.0)])]
  (assert= "retval count" 3 (count (:retval trace)))
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- Edge: single element map --")
(let [kernel (dyn/auto-key (gen [x]
               (let [y (trace :y (dist/gaussian x 1.0))]
                 y)))
      mapped (comb/map-combinator kernel)
      trace (p/simulate mapped [[(mx/scalar 5.0)]])]
  (assert= "retval count" 1 (count (:retval trace)))
  (mx/eval! (:score trace))
  (assert-true "score finite" (js/isFinite (mx/item (:score trace)))))

;; ---------------------------------------------------------------------------
;; Section 7: Score self-consistency
;; ---------------------------------------------------------------------------

(println "\n-- Score self-consistency --")
;; Recompute unfold score from per-step assess. Must match total.
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/gaussian state 0.1))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      trace (p/simulate unfold [5 (mx/scalar 0.0)])]
  (let [total-recomputed
        (reduce
          (fn [acc t]
            (let [sub-choices (cm/get-submap (:choices trace) t)
                  state-val (if (zero? t)
                              (mx/scalar 0.0)
                              (nth (:retval trace) (dec t)))
                  _ (mx/eval! state-val)
                  {:keys [weight]} (p/assess kernel [t state-val] sub-choices)]
              (mx/eval! weight)
              (+ acc (mx/item weight))))
          0.0
          (range 5))]
    (mx/eval! (:score trace))
    (assert-close "score matches sum of assess"
      (mx/item (:score trace)) total-recomputed 1e-4)))

;; ---------------------------------------------------------------------------
;; Section 8: Statistical sanity
;; ---------------------------------------------------------------------------

(println "\n-- Statistical sanity --")
;; Use non-compilable kernel (beta-dist) to force handler path,
;; which has correct PRNG variation across calls.
;; NOTE: mx/compile-fn caches random ops, so compiled kernels are
;; deterministic regardless of key — a pre-existing L1-M2 limitation.
;; M5's compiled path (which bypasses DynamicGF.simulate) would need
;; its own key threading to restore variation.
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/beta-dist 2 5))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      scores (doall
               (for [_ (range 10)]
                 (let [trace (p/simulate unfold [3 (mx/scalar 0.5)])]
                   (mx/eval! (:score trace))
                   (mx/item (:score trace)))))]
  (assert-true "all scores finite" (every? js/isFinite scores))
  (assert-true "scores vary"
    (> (- (apply max scores) (apply min scores)) 0.01)))

;; ---------------------------------------------------------------------------
;; Section 9: Performance
;; ---------------------------------------------------------------------------

(println "\n-- Performance: Unfold T=50 --")
(let [kernel (dyn/auto-key (gen [t state]
               (let [next (trace :x (dist/gaussian state 0.1))]
                 next)))
      unfold (comb/unfold-combinator kernel)
      ;; Warmup
      _ (dotimes [_ 3]
          (let [t (p/simulate unfold [50 (mx/scalar 0.0)])]
            (mx/eval! (:score t))))
      ;; Timed runs
      n-runs 20
      start (js/Date.now)
      _ (dotimes [_ n-runs]
          (let [t (p/simulate unfold [50 (mx/scalar 0.0)])]
            (mx/eval! (:score t))))
      elapsed (- (js/Date.now) start)
      per-run (/ elapsed n-runs)]
  (println "  INFO: Unfold T=50," n-runs "runs, avg" (.toFixed per-run 2) "ms/run"))

(println "\n-- Performance: Map N=20 --")
(let [kernel (dyn/auto-key (gen [x]
               (let [y (trace :y (dist/gaussian x 1.0))]
                 y)))
      mapped (comb/map-combinator kernel)
      inputs [(mapv #(mx/scalar %) (range 20))]
      _ (dotimes [_ 3]
          (let [t (p/simulate mapped inputs)]
            (mx/eval! (:score t))))
      n-runs 20
      start (js/Date.now)
      _ (dotimes [_ n-runs]
          (let [t (p/simulate mapped inputs)]
            (mx/eval! (:score t))))
      elapsed (- (js/Date.now) start)
      per-run (/ elapsed n-runs)]
  (println "  INFO: Map N=20," n-runs "runs, avg" (.toFixed per-run 2) "ms/run"))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\nL1-M5 tests complete: " @pass-count " passed, " @fail-count " failed"
              " (" (+ @pass-count @fail-count) " total)"))
