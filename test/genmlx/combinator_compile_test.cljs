(ns genmlx.combinator-compile-test
  "L1-M5: Combinator-aware compilation tests.
   Tests that combinators with compilable kernels bypass per-step
   Trace/ChoiceMap construction by calling compiled-simulate directly."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
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
;; Test utilities
;; ---------------------------------------------------------------------------

(defn compilable?
  "Check if a gen-fn kernel has a compiled-simulate function."
  [gf]
  (some? (:compiled-simulate (:schema gf))))

;; ---------------------------------------------------------------------------
;; Section 0: Kernel compilability verification
;; ---------------------------------------------------------------------------

(deftest kernel-compilability-test
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
    (is (compilable? k-unfold) "unfold kernel compilable")
    (is (compilable? k-map) "map kernel compilable")
    (is (compilable? k-scan) "scan kernel compilable")
    (is (compilable? k-switch) "switch kernel compilable")
    (is (not (compilable? k-beta)) "beta kernel NOT compilable")))

;; ---------------------------------------------------------------------------
;; Section 1: Unfold with compilable kernel
;; ---------------------------------------------------------------------------

(deftest unfold-compilable-kernel-test
  (let [kernel (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   next)))
        unfold (comb/unfold-combinator kernel)
        trace (p/simulate unfold [5 (mx/scalar 0.0)])]
    (testing "trace structure"
      (is (instance? tr/Trace trace) "trace is Trace")
      (is (= 5 (count (:retval trace))) "retval count"))
    (testing "choices: keys 0-4, each with :x"
      (doseq [t (range 5)]
        (let [sub (cm/get-submap (:choices trace) t)]
          (is (cm/has-value? (cm/get-submap sub :x))
              (str "step " t " has :x")))))
    (testing "score"
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "score finite"))
    (testing "step-scores metadata"
      (let [ss (::comb/step-scores (meta trace))]
        (is (some? ss) "has step-scores")
        (is (= 5 (count ss)) "step-scores count")
        (doseq [s ss] (mx/eval! s))
        (is (h/close? (mx/item (:score trace))
                      (reduce + (map mx/item ss))
                      1e-5)
            "step-scores sum = total score")))))

(deftest unfold-compiled-path-detection-test
  (let [kernel (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   next)))
        unfold (comb/unfold-combinator kernel)
        trace (p/simulate unfold [5 (mx/scalar 0.0)])]
    (is (::comb/compiled-path (meta trace)) "compiled path used")))

(deftest unfold-non-compilable-fallback-test
  (let [kernel (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/beta-dist 2 5))]
                   next)))
        unfold (comb/unfold-combinator kernel)
        trace (p/simulate unfold [3 (mx/scalar 0.5)])]
    (is (instance? tr/Trace trace) "fallback trace valid")
    (is (= 3 (count (:retval trace))) "fallback retval count")
    (mx/eval! (:score trace))
    (is (js/isFinite (mx/item (:score trace))) "fallback score finite")
    (is (not (::comb/compiled-path (meta trace))) "fallback NOT compiled")))

;; ---------------------------------------------------------------------------
;; Section 2: Scan with compilable kernel
;; ---------------------------------------------------------------------------

(deftest scan-compilable-kernel-test
  (let [kernel (dyn/auto-key (gen [carry input]
                 (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                   [x x])))
        scan (comb/scan-combinator kernel)
        inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                (mx/scalar 4.0) (mx/scalar 5.0)]
        trace (p/simulate scan [(mx/scalar 0.0) inputs])]
    (testing "trace structure"
      (is (instance? tr/Trace trace) "trace is Trace")
      (is (some? (:carry (:retval trace))) "retval has :carry")
      (is (= 5 (count (:outputs (:retval trace)))) "outputs count"))
    (testing "choices at each step"
      (doseq [t (range 5)]
        (is (cm/has-value? (cm/get-submap (cm/get-submap (:choices trace) t) :x))
            (str "step " t " has :x"))))
    (testing "score"
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "score finite"))
    (testing "metadata"
      (is (some? (::comb/step-scores (meta trace))) "has step-scores")
      (is (some? (::comb/step-carries (meta trace))) "has step-carries"))))

(deftest scan-compiled-path-detection-test
  (let [kernel (dyn/auto-key (gen [carry input]
                 (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                   [x x])))
        scan (comb/scan-combinator kernel)
        trace (p/simulate scan [(mx/scalar 0.0) [(mx/scalar 1.0)]])]
    (is (::comb/compiled-path (meta trace)) "compiled path used")))

(deftest scan-carry-threading-test
  (testing "deterministic carry threading with delta distribution"
    (let [kernel (dyn/auto-key (gen [carry input]
                   (let [x (trace :x (dist/delta carry))]
                     [(mx/add carry input) (mx/add carry input)])))
          scan (comb/scan-combinator kernel)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          trace (p/simulate scan [(mx/scalar 0.0) inputs])]
      (let [carry (:carry (:retval trace))]
        (mx/eval! carry)
        (is (h/close? 6.0 (mx/item carry) 1e-5) "final carry = 6"))
      (let [outputs (:outputs (:retval trace))]
        (doseq [[i expected] [[0 1.0] [1 3.0] [2 6.0]]]
          (mx/eval! (nth outputs i))
          (is (h/close? expected (mx/item (nth outputs i)) 1e-5)
              (str "output " i))))
      (mx/eval! (:score trace))
      (is (h/close? 0.0 (mx/item (:score trace)) 1e-6) "delta score = 0"))))

;; ---------------------------------------------------------------------------
;; Section 3: Map with compilable kernel
;; ---------------------------------------------------------------------------

(deftest map-compilable-kernel-test
  (let [kernel (dyn/auto-key (gen [x]
                 (let [y (trace :y (dist/gaussian x 1.0))]
                   y)))
        mapped (comb/map-combinator kernel)
        trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]])]
    (testing "trace structure"
      (is (instance? tr/Trace trace) "trace is Trace")
      (is (= 3 (count (:retval trace))) "retval count"))
    (testing "choices"
      (doseq [i (range 3)]
        (is (cm/has-value? (cm/get-submap (cm/get-submap (:choices trace) i) :y))
            (str "element " i " has :y"))))
    (testing "score"
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "score finite"))
    (testing "element-scores metadata"
      (is (some? (::comb/element-scores (meta trace))) "has element-scores"))))

(deftest map-compiled-path-detection-test
  (let [kernel (dyn/auto-key (gen [x]
                 (let [y (trace :y (dist/gaussian x 1.0))]
                   y)))
        mapped (comb/map-combinator kernel)
        trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]])]
    (is (::comb/compiled-path (meta trace)) "compiled path used")))

(deftest map-non-compilable-fallback-test
  (let [kernel (dyn/auto-key (gen [x]
                 (let [y (trace :y (dist/beta-dist 2 5))]
                   y)))
        mapped (comb/map-combinator kernel)
        trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]])]
    (is (instance? tr/Trace trace) "fallback trace valid")
    (is (= 2 (count (:retval trace))) "fallback retval count")
    (is (not (::comb/compiled-path (meta trace))) "fallback NOT compiled")))

;; ---------------------------------------------------------------------------
;; Section 4: Switch with compilable branch
;; ---------------------------------------------------------------------------

(deftest switch-compilable-branch-test
  (let [branch0 (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    x)))
        branch1 (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 10 1))]
                    x)))
        sw (comb/switch-combinator branch0 branch1)
        t0 (p/simulate sw [0])
        t1 (p/simulate sw [1])]
    (testing "branch 0"
      (is (instance? tr/Trace t0) "branch 0 trace")
      (let [x0 (cm/get-value (cm/get-submap (:choices t0) :x))]
        (mx/eval! x0)
        (is (< (js/Math.abs (mx/item x0)) 5) "branch 0 near 0")))
    (testing "branch 1"
      (let [x1 (cm/get-value (cm/get-submap (:choices t1) :x))]
        (mx/eval! x1)
        (is (< (js/Math.abs (- (mx/item x1) 10)) 5) "branch 1 near 10")))
    (testing "scores finite"
      (mx/eval! (:score t0))
      (mx/eval! (:score t1))
      (is (js/isFinite (mx/item (:score t0))) "branch 0 score finite")
      (is (js/isFinite (mx/item (:score t1))) "branch 1 score finite"))
    (testing "switch-idx metadata"
      (is (= 0 (::comb/switch-idx (meta t0))) "branch 0 idx")
      (is (= 1 (::comb/switch-idx (meta t1))) "branch 1 idx"))))

(deftest switch-compiled-path-detection-test
  (let [branch0 (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    x)))
        sw (comb/switch-combinator branch0)
        trace (p/simulate sw [0])]
    (is (::comb/compiled-path (meta trace)) "compiled path used")))

(deftest switch-non-compilable-fallback-test
  (let [branch0 (dyn/auto-key (gen []
                  (let [x (trace :x (dist/beta-dist 2 5))]
                    x)))
        sw (comb/switch-combinator branch0)
        trace (p/simulate sw [0])]
    (is (instance? tr/Trace trace) "fallback trace valid")
    (is (not (::comb/compiled-path (meta trace))) "fallback NOT compiled")))

(deftest switch-mixed-branches-test
  (let [branch0 (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    x)))
        branch1 (dyn/auto-key (gen []
                  (let [x (trace :x (dist/beta-dist 2 5))]
                    x)))
        sw (comb/switch-combinator branch0 branch1)]
    (let [t0 (p/simulate sw [0])]
      (is (::comb/compiled-path (meta t0)) "mixed: compilable branch compiled"))
    (let [t1 (p/simulate sw [1])]
      (is (not (::comb/compiled-path (meta t1))) "mixed: non-compilable branch NOT compiled"))))

;; ---------------------------------------------------------------------------
;; Section 5: Mix with compilable component
;; ---------------------------------------------------------------------------

(deftest mix-compilable-component-test
  (let [comp0 (dyn/auto-key (gen []
                (let [x (trace :x (dist/gaussian 0 1))]
                  x)))
        comp1 (dyn/auto-key (gen []
                (let [x (trace :x (dist/gaussian 10 1))]
                  x)))
        log-w (mx/log (mx/array #js [0.5 0.5]))
        mix (comb/mix-combinator [comp0 comp1] log-w)
        trace (p/simulate mix [])]
    (is (instance? tr/Trace trace) "trace is Trace")
    (testing "has :component-idx"
      (let [idx (cm/get-choice (:choices trace) [:component-idx])]
        (mx/eval! idx)
        (is (contains? #{0 1} (int (mx/item idx))) "has component-idx")))
    (is (cm/has-value? (cm/get-submap (:choices trace) :x)) "has :x")
    (testing "score finite"
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "score finite"))))

(deftest mix-compiled-path-detection-test
  (let [comp0 (dyn/auto-key (gen []
                (let [x (trace :x (dist/gaussian 0 1))]
                  x)))
        log-w (mx/array #js [0.0])
        mix (comb/mix-combinator [comp0] log-w)
        trace (p/simulate mix [])]
    (is (::comb/compiled-path (meta trace)) "compiled path used")))

;; ---------------------------------------------------------------------------
;; Section 6: Edge cases
;; ---------------------------------------------------------------------------

(deftest edge-single-step-unfold-test
  (let [kernel (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   next)))
        unfold (comb/unfold-combinator kernel)
        trace (p/simulate unfold [1 (mx/scalar 0.0)])]
    (is (= 1 (count (:retval trace))) "retval count")
    (mx/eval! (:score trace))
    (is (js/isFinite (mx/item (:score trace))) "score finite")
    (is (= 1 (count (::comb/step-scores (meta trace)))) "step-scores count")))

(deftest edge-zero-steps-unfold-test
  (let [kernel (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   next)))
        unfold (comb/unfold-combinator kernel)
        trace (p/simulate unfold [0 (mx/scalar 0.0)])]
    (is (= 0 (count (:retval trace))) "retval count")
    (mx/eval! (:score trace))
    (is (h/close? 0.0 (mx/item (:score trace)) 1e-6) "score = 0")
    (is (= cm/EMPTY (:choices trace)) "empty choices")))

(deftest edge-delta-only-unfold-test
  (let [kernel (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/delta state))]
                   next)))
        unfold (comb/unfold-combinator kernel)
        trace (p/simulate unfold [3 (mx/scalar 5.0)])]
    (mx/eval! (:score trace))
    (is (h/close? 0.0 (mx/item (:score trace)) 1e-6) "delta score = 0")
    (doseq [rv (:retval trace)]
      (mx/eval! rv)
      (is (h/close? 5.0 (mx/item rv) 1e-6) "delta retval = 5.0"))))

(deftest edge-multi-site-kernel-test
  (let [kernel (dyn/auto-key (gen [t state]
                 (let [a (trace :a (dist/gaussian state 1))
                       b (trace :b (dist/gaussian a 0.5))]
                   b)))
        unfold (comb/unfold-combinator kernel)
        trace (p/simulate unfold [3 (mx/scalar 0.0)])]
    (doseq [t (range 3)]
      (let [sub (cm/get-submap (:choices trace) t)]
        (is (cm/has-value? (cm/get-submap sub :a))
            (str "step " t " has :a"))
        (is (cm/has-value? (cm/get-submap sub :b))
            (str "step " t " has :b"))))
    (mx/eval! (:score trace))
    (is (js/isFinite (mx/item (:score trace))) "score finite")))

(deftest edge-extra-args-unfold-test
  (let [kernel (dyn/auto-key (gen [t state scale]
                 (let [next (trace :x (dist/gaussian state scale))]
                   next)))
        unfold (comb/unfold-combinator kernel)
        trace (p/simulate unfold [3 (mx/scalar 0.0) (mx/scalar 2.0)])]
    (is (= 3 (count (:retval trace))) "retval count")
    (mx/eval! (:score trace))
    (is (js/isFinite (mx/item (:score trace))) "score finite")))

(deftest edge-single-element-map-test
  (let [kernel (dyn/auto-key (gen [x]
                 (let [y (trace :y (dist/gaussian x 1.0))]
                   y)))
        mapped (comb/map-combinator kernel)
        trace (p/simulate mapped [[(mx/scalar 5.0)]])]
    (is (= 1 (count (:retval trace))) "retval count")
    (mx/eval! (:score trace))
    (is (js/isFinite (mx/item (:score trace))) "score finite")))

;; ---------------------------------------------------------------------------
;; Section 7: Score self-consistency
;; ---------------------------------------------------------------------------

(deftest score-self-consistency-test
  (testing "recompute unfold score from per-step assess"
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
        (is (h/close? (mx/item (:score trace)) total-recomputed 1e-4)
            "score matches sum of assess")))))

;; ---------------------------------------------------------------------------
;; Section 8: Statistical sanity
;; ---------------------------------------------------------------------------

(deftest statistical-sanity-test
  (let [kernel (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/beta-dist 2 5))]
                   next)))
        unfold (comb/unfold-combinator kernel)
        scores (doall
                 (for [_ (range 10)]
                   (let [trace (p/simulate unfold [3 (mx/scalar 0.5)])]
                     (mx/eval! (:score trace))
                     (mx/item (:score trace)))))]
    (is (every? js/isFinite scores) "all scores finite")
    (is (> (- (apply max scores) (apply min scores)) 0.01) "scores vary")))

;; ---------------------------------------------------------------------------
;; Section 9: Performance
;; ---------------------------------------------------------------------------

(deftest performance-unfold-test
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
    (println "  INFO: Unfold T=50," n-runs "runs, avg" (.toFixed per-run 2) "ms/run")
    (is (some? per-run) "performance unfold completed")))

(deftest performance-map-test
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
    (println "  INFO: Map N=20," n-runs "runs, avg" (.toFixed per-run 2) "ms/run")
    (is (some? per-run) "performance map completed")))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
