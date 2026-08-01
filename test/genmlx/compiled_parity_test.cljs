;; @tier medium
(ns genmlx.compiled-parity-test
  "Compiled-path parity (genmlx-b210).

   Compiled execution must produce the same traces, scores, and weights as
   the handler path — or decline compilation. Covers:
   1. fused M5 kernels: iid-gaussian must NOT silently score as Delta
   2. compiled SMC extend step declines iid-gaussian kernels
   3. bernoulli/exponential log-prob boundary guards (NaN / support)
   4. M4 branch conditions: ClojureScript truthiness (numeric 0 is truthy)
   5. binding-env shadowing/destructuring declines instead of mis-resolving
   6. M4 requires a compilable return expression (no retval-nil traces)
   7. compiled regenerate resamples iid-gaussian (was: Delta, zero weight)
   8. closed-over vars deref per call (no stale baked values)
   9. noise-transform registry seam: EVERY family's out-of-support log-prob
      matches the handler, and in-support values are bit-identical

   Handler path (strip-compiled / stripped schema) is ground truth; numeric
   oracles are computed host-side.

   Run: bun run --bun nbb test/genmlx/compiled_parity_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx.random :as rng]
            [genmlx.combinators :as comb]
            [genmlx.compiled :as compiled]
            [genmlx.compiled-ops :as cops]
            [genmlx.gfi :as gfi])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Assertion helpers
;; ---------------------------------------------------------------------------

(def ^:dynamic *pass* (volatile! 0))
(def ^:dynamic *fail* (volatile! 0))

(defn assert-true [desc pred]
  (if pred
    (do (vswap! *pass* inc) (println (str "  PASS: " desc)))
    (do (vswap! *fail* inc) (println (str "  FAIL: " desc)))))

(defn assert-close [desc expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol)
      (do (vswap! *pass* inc)
          (println (str "  PASS: " desc " (" (.toFixed actual 6) " ~ " (.toFixed expected 6) ", |Δ|=" (.toExponential d 2) ")")))
      (do (vswap! *fail* inc)
          (println (str "  FAIL: " desc " (" (.toFixed actual 6) " vs " (.toFixed expected 6) ", |Δ|=" (.toExponential d 2) " > " tol ")"))))))

(def TOL 2e-4)
(def LOG2PI (js/Math.log (* 2 js/Math.PI)))

(defn norm-lpdf [x mu variance]
  (* -0.5 (+ LOG2PI (js/Math.log variance)
             (/ (* (- x mu) (- x mu)) variance))))

(defn cmv [m] (reduce-kv (fn [c k v] (cm/set-value c k (mx/scalar v))) cm/EMPTY m))

;; ===========================================================================
;; SECTION 1 — fused M5: iid-gaussian must not score as Delta
;; ===========================================================================

(println "\n== Section 1: fused M5 iid-gaussian gates ==")

(def iid-map-kernel
  (dyn/auto-key (gen [mu]
    (let [x (trace :x (dist/iid-gaussian mu 1.0 3))]
      x))))

(def iid-unfold-kernel
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/iid-gaussian state 1.0 2))]
      (mx/sum x)))))

(def mixed-kernel
  (dyn/auto-key (gen [t state]
    (let [z (trace :z (dist/gaussian state 1.0))
          x (trace :x (dist/iid-gaussian z 1.0 2))]
      z))))

(def gauss-kernel
  (dyn/auto-key (gen [t state]
    (let [next (trace :x (dist/gaussian state 0.1))]
      next))))

(assert-true "fusable-kernel?: iid-gaussian kernel NOT fusable"
             (not (cops/fusable-kernel? iid-map-kernel)))
(assert-true "fusable-kernel?: mixed gaussian+iid kernel NOT fusable"
             (not (cops/fusable-kernel? mixed-kernel)))
(assert-true "fusable-kernel?: pure gaussian kernel fusable (no regression)"
             (cops/fusable-kernel? gauss-kernel))

;; Map with iid-gaussian kernel: falls back, score = sum of element log-probs
(let [mapped (comb/map-combinator iid-map-kernel)
      trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar -2.0)]])
      mus [1.0 -2.0]
      oracle (reduce +
                     (for [i (range 2)
                           :let [xs (mx/->clj (cm/get-value
                                               (cm/get-submap
                                                (cm/get-submap (:choices trace) i) :x)))]
                           xj xs]
                       (norm-lpdf xj (nth mus i) 1.0)))
      score (mx/item (:score trace))]
  (assert-true "Map iid: fused path NOT used"
               (not (:genmlx.combinators/compiled-path (meta trace))))
  (assert-true "Map iid: score is non-zero (was silently 0)"
               (> (js/Math.abs score) 1e-6))
  (assert-close "Map iid: score = sum of per-element iid log-probs"
                oracle score TOL)
  (assert-true "Map iid: x values are sampled, not value=mu"
               (let [xs (mx/->clj (cm/get-value
                                   (cm/get-submap (cm/get-submap (:choices trace) 0) :x)))]
                 (some #(> (js/Math.abs (- % 1.0)) 1e-6) xs))))

;; Unfold with iid-gaussian kernel: falls back, score reconstructable
(let [unfold (comb/unfold-combinator iid-unfold-kernel)
      trace (p/simulate unfold [3 (mx/scalar 0.0)])
      retvals (mapv mx/item (:retval trace))
      states (into [0.0] (butlast retvals))
      oracle (reduce +
                     (for [t (range 3)
                           :let [xs (mx/->clj (cm/get-value
                                               (cm/get-submap
                                                (cm/get-submap (:choices trace) t) :x)))]
                           xj xs]
                       (norm-lpdf xj (nth states t) 1.0)))
      score (mx/item (:score trace))]
  (assert-true "Unfold iid: fused path NOT used"
               (not (:genmlx.combinators/compiled-path (meta trace))))
  (assert-true "Unfold iid: score is non-zero" (> (js/Math.abs score) 1e-6))
  (assert-close "Unfold iid: score = sum of step iid log-probs" oracle score TOL))

;; ===========================================================================
;; SECTION 2 — compiled SMC extend step gate
;; ===========================================================================

(println "\n== Section 2: compiled SMC extend gate ==")

(assert-true "smc-extend: iid-gaussian kernel declines (nil)"
             (nil? (cops/make-smc-extend-step (:schema iid-unfold-kernel)
                                              (:source iid-unfold-kernel))))
(assert-true "smc-extend: mixed kernel declines (nil)"
             (nil? (cops/make-smc-extend-step (:schema mixed-kernel)
                                              (:source mixed-kernel))))
(assert-true "smc-extend: gaussian kernel still compiles"
             (some? (cops/make-smc-extend-step (:schema gauss-kernel)
                                               (:source gauss-kernel))))

;; ===========================================================================
;; SECTION 3 — bernoulli / exponential log-prob boundaries
;; ===========================================================================

(println "\n== Section 3: log-prob boundary guards ==")

(let [lp (:log-prob (get compiled/noise-transforms-full :bernoulli))
      handler-lp (fn [pv v] (mx/item (dc/dist-log-prob (dist/bernoulli (mx/scalar pv))
                                                       (mx/scalar v))))
      compiled-lp (fn [pv v] (mx/item (lp (mx/scalar v) (mx/scalar pv))))]
  (assert-close "bernoulli lp: p=0, v=0 → 0 (was NaN)"
                0.0 (compiled-lp 0.0 0.0) 1e-9)
  (assert-close "bernoulli lp: p=1, v=1 → 0 (was NaN)"
                0.0 (compiled-lp 1.0 1.0) 1e-9)
  (assert-true "bernoulli lp: p=0, v=1 → -Inf, matches handler"
               (and (= (- js/Infinity) (compiled-lp 0.0 1.0))
                    (= (handler-lp 0.0 1.0) (compiled-lp 0.0 1.0))))
  (assert-close "bernoulli lp: interior point matches handler"
                (handler-lp 0.3 1.0) (compiled-lp 0.3 1.0) 1e-6)
  (assert-close "bernoulli lp: p=0, v=0 matches handler"
                (handler-lp 0.0 0.0) (compiled-lp 0.0 0.0) 1e-9))

(let [lp (:log-prob (get compiled/noise-transforms-full :exponential))
      handler-lp (fn [rv v] (mx/item (dc/dist-log-prob (dist/exponential (mx/scalar rv))
                                                       (mx/scalar v))))
      compiled-lp (fn [rv v] (mx/item (lp (mx/scalar v) (mx/scalar rv))))]
  (assert-true "exponential lp: v<0 → -Inf (was finite)"
               (= (- js/Infinity) (compiled-lp 2.0 -1.0)))
  (assert-true "exponential lp: v<0 matches handler"
               (= (handler-lp 2.0 -1.0) (compiled-lp 2.0 -1.0)))
  (assert-close "exponential lp: interior point matches handler"
                (handler-lp 2.0 1.5) (compiled-lp 2.0 1.5) 1e-6))

;; Model-level assess parity at the boundary
(def bern-model (dyn/auto-key (gen [p] (trace :b (dist/bernoulli p)))))

(let [choices (cmv {:b 0.0})
      w-compiled (mx/item (:weight (p/assess bern-model [0.0] choices)))
      w-handler (mx/item (:weight (p/assess (dyn/auto-key (gfi/strip-compiled bern-model))
                                            [0.0] choices)))]
  (assert-true "bern model: compiled-assess attached"
               (some? (:compiled-assess (:schema bern-model))))
  (assert-close "bern model assess: p=0, b=0 → 0 on compiled path"
                0.0 w-compiled 1e-9)
  (assert-close "bern model assess: compiled = handler" w-handler w-compiled 1e-9))

;; ===========================================================================
;; SECTION 4 — M4 branch conditions: CLJS truthiness
;; ===========================================================================

(println "\n== Section 4: M4 zero-truthiness ==")

(def m4-model
  (dyn/auto-key (gen [flag]
    (if flag
      (trace :x (dist/gaussian 5.0 1.0))
      (trace :x (dist/gaussian -5.0 1.0))))))

(assert-true "M4 model: compiled-simulate attached"
             (some? (:compiled-simulate (:schema m4-model))))

(let [k (rng/fresh-key 100)
      x-c (mx/item (cm/get-value (cm/get-submap
                                  (:choices (p/simulate (dyn/with-key m4-model k) [0]))
                                  :x)))
      x-h (mx/item (cm/get-value (cm/get-submap
                                  (:choices (p/simulate
                                             (dyn/with-key (gfi/strip-compiled m4-model) k)
                                             [0]))
                                  :x)))]
  (assert-true "M4 flag=0: numeric 0 is TRUTHY → true branch (x near +5)"
               (< (js/Math.abs (- x-c 5.0)) 5.0))
  (assert-close "M4 flag=0: compiled = handler (same key)" x-h x-c 1e-6))

(let [k (rng/fresh-key 101)
      x-c (mx/item (cm/get-value (cm/get-submap
                                  (:choices (p/simulate (dyn/with-key m4-model k) [false]))
                                  :x)))
      x-h (mx/item (cm/get-value (cm/get-submap
                                  (:choices (p/simulate
                                             (dyn/with-key (gfi/strip-compiled m4-model) k)
                                             [false]))
                                  :x)))]
  (assert-true "M4 flag=false: false branch (x near -5)"
               (< (js/Math.abs (- x-c -5.0)) 5.0))
  (assert-close "M4 flag=false: compiled = handler (same key)" x-h x-c 1e-6))

(let [k (rng/fresh-key 102)
      x-c (mx/item (cm/get-value (cm/get-submap
                                  (:choices (p/simulate (dyn/with-key m4-model k) [true]))
                                  :x)))]
  (assert-true "M4 flag=true: true branch (x near +5)"
               (< (js/Math.abs (- x-c 5.0)) 5.0)))

;; Assess parity with numeric-0 condition
(let [choices (cmv {:x 0.0})
      w-c (mx/item (:weight (p/assess m4-model [0] choices)))
      w-h (mx/item (:weight (p/assess (dyn/auto-key (gfi/strip-compiled m4-model))
                                      [0] choices)))]
  (assert-close "M4 assess flag=0: score = true-branch lpdf"
                (norm-lpdf 0.0 5.0 1.0) w-c TOL)
  (assert-close "M4 assess flag=0: compiled = handler" w-h w-c 1e-6))

;; ===========================================================================
;; SECTION 5 — binding-env shadowing / destructuring
;; ===========================================================================

(println "\n== Section 5: shadowing declines compilation ==")

(def shadow-model
  (dyn/auto-key (gen []
    (let [m 1.0]
      (trace :a (dist/gaussian m 1.0)))
    (let [m 5.0]
      (trace :b (dist/gaussian m 1.0))))))

(assert-true "shadow: rebound let name declines compilation"
             (nil? (:compiled-simulate (:schema shadow-model))))

(let [n 20
      sums (reduce (fn [[sa sb] _]
                     (let [tr (p/simulate shadow-model [])]
                       [(+ sa (mx/item (cm/get-value (cm/get-submap (:choices tr) :a))))
                        (+ sb (mx/item (cm/get-value (cm/get-submap (:choices tr) :b))))]))
                   [0.0 0.0] (range n))
      mean-a (/ (first sums) n)
      mean-b (/ (second sums) n)]
  (assert-true "shadow: :a samples from FIRST binding (mean ~1, not ~5)"
               (< (js/Math.abs (- mean-a 1.0)) 1.0))
  (assert-true "shadow: :b samples from SECOND binding (mean ~5)"
               (< (js/Math.abs (- mean-b 5.0)) 1.0)))

;; Destructured local must not fall through to a same-named namespace var
(def bait-mu 9.0)

(def destructure-model
  (dyn/auto-key (gen []
    (let [[bait-mu] [2.0]]
      (trace :a (dist/gaussian bait-mu 1.0))))))

(assert-true "destructure: declines compilation (poisoned target)"
             (nil? (:compiled-simulate (:schema destructure-model))))

(let [n 20
      mean-a (/ (reduce (fn [s _]
                          (+ s (mx/item (cm/get-value
                                         (cm/get-submap
                                          (:choices (p/simulate destructure-model [])) :a)))))
                        0.0 (range n))
                n)]
  (assert-true "destructure: :a uses the LOCAL (mean ~2), not ns var 9.0"
               (< (js/Math.abs (- mean-a 2.0)) 1.0)))

;; ===========================================================================
;; SECTION 6 — M4 requires compilable retval
;; ===========================================================================

(println "\n== Section 6: M4 retval required ==")

(def m4-noretval
  (dyn/auto-key (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0.0 1.0))
      (trace :x (dist/gaussian 1.0 1.0)))
    (str "done"))))

(assert-true "M4 no-retval: declines compilation"
             (nil? (:compiled-simulate (:schema m4-noretval))))
(assert-true "M4 no-retval: handler path returns the actual retval"
             (= "done" (:retval (p/simulate m4-noretval [true]))))

;; ===========================================================================
;; SECTION 7 — compiled regenerate resamples iid-gaussian
;; ===========================================================================

(println "\n== Section 7: compiled regenerate iid-gaussian ==")

(def iid-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0.0 1.0))
          x (trace :x (dist/iid-gaussian mu 1.0 3))]
      x)))

(assert-true "iid model: compiled-regenerate attached"
             (some? (:compiled-regenerate (:schema iid-model))))

(let [k1 (rng/fresh-key 200)
      k2 (rng/fresh-key 201)
      tr (p/simulate (dyn/with-key iid-model k1) [])
      old-x (mx/->clj (cm/get-value (cm/get-submap (:choices tr) :x)))
      r-c (p/regenerate (dyn/with-key iid-model k2) tr (sel/select :x))
      r-h (p/regenerate (dyn/with-key (gfi/strip-compiled iid-model) k2)
                        tr (sel/select :x))
      new-x-c (mx/->clj (cm/get-value (cm/get-submap (:choices (:trace r-c)) :x)))
      new-x-h (mx/->clj (cm/get-value (cm/get-submap (:choices (:trace r-h)) :x)))]
  (assert-true "iid regen: :x has shape [3] (was scalar mu under Delta bug)"
               (= 3 (count new-x-c)))
  (assert-true "iid regen: :x actually resampled"
               (some true? (map #(> (js/Math.abs (- %1 %2)) 1e-6) new-x-c old-x)))
  (assert-true "iid regen: compiled values = handler values (same key)"
               (every? true? (map #(< (js/Math.abs (- %1 %2)) 1e-5) new-x-c new-x-h)))
  (assert-close "iid regen: compiled weight = handler weight"
                (mx/item (:weight r-h)) (mx/item (:weight r-c)) 1e-5))

;; Unselected iid site: kept and re-scored identically on both paths
(let [k1 (rng/fresh-key 202)
      k2 (rng/fresh-key 203)
      tr (p/simulate (dyn/with-key iid-model k1) [])
      r-c (p/regenerate (dyn/with-key iid-model k2) tr (sel/select :mu))
      r-h (p/regenerate (dyn/with-key (gfi/strip-compiled iid-model) k2)
                        tr (sel/select :mu))]
  (assert-close "iid regen (mu selected): compiled weight = handler weight"
                (mx/item (:weight r-h)) (mx/item (:weight r-c)) 1e-5))

;; ===========================================================================
;; SECTION 8 — closed-over vars deref per call
;; ===========================================================================

(println "\n== Section 8: closed-over var freshness ==")

(def drift-mu 0.0)
(def drift-model (dyn/auto-key (gen [] (trace :x (dist/gaussian drift-mu 0.1)))))

(assert-true "drift: model compiled (var resolved through fallback)"
             (some? (:compiled-simulate (:schema drift-model))))

(let [x0 (mx/item (cm/get-value (cm/get-submap
                                 (:choices (p/simulate drift-model [])) :x)))]
  (assert-true "drift: initial value of var honored (x ~ 0)"
               (< (js/Math.abs x0) 1.0))
  (def drift-mu 5.0)
  (let [x1 (mx/item (cm/get-value (cm/get-submap
                                   (:choices (p/simulate drift-model [])) :x)))]
    (assert-true "drift: re-def'd var honored on next call (x ~ 5, was stale 0)"
                 (< (js/Math.abs (- x1 5.0)) 1.0))))

;; ===========================================================================
;; SECTION 9 — noise-transform registry is a CHECKED seam (genmlx-4x5w)
;; ===========================================================================
;;
;; compiled.cljs's noise-transform registry is a second copy of dist.cljs's
;; log-prob math. Two of its nine entries had silently drifted out of
;; support-guard agreement (measured 2026-08-01, x86_64/sm_120):
;;
;;   bernoulli p=0.3 v=0.5  handler -Inf   compiled -0.780  (FINITE)
;;   bernoulli p=0.3 v=-1   handler -Inf   compiled +0.491  (POSITIVE!)
;;   log-normal      v<=0   handler -Inf   compiled NaN
;;
;; -Inf kills one particle; NaN poisons a whole SMC population's logsumexp.
;; These rows cover EVERY registry family — including the seven that were
;; correct — so the next drift is caught automatically instead of by audit.
;;
;; The comparator below treats NaN as a FAILURE on either side, deliberately
;; unlike pef.cljs:523 which defines NaN == NaN as path AGREEMENT.

(println "\n== Section 9: noise-transform registry vs handler (all families) ==")

(defn assert-parity
  "compiled log-prob must equal the handler's. NaN on EITHER side is a failure,
   even if both are NaN — 'both paths blew up the same way' is not parity."
  ([desc h c] (assert-parity desc h c 1e-6))
  ([desc h c tol]
   (cond
     (js/Number.isNaN c)
     (do (vswap! *fail* inc)
         (println (str "  FAIL: " desc " — compiled is NaN (handler=" h ")")))
     (js/Number.isNaN h)
     (do (vswap! *fail* inc)
         (println (str "  FAIL: " desc " — handler is NaN (compiled=" c ")")))
     (or (= h c)
         (and (js/isFinite h) (js/isFinite c) (<= (js/Math.abs (- h c)) tol)))
     (do (vswap! *pass* inc) (println (str "  PASS: " desc " (h=" h " c=" c ")")))
     :else
     (do (vswap! *fail* inc)
         (println (str "  FAIL: " desc " — handler=" h " compiled=" c))))))

(defn assert-bit-identical [desc h c]
  (if (and (not (js/Number.isNaN h)) (= h c))
    (do (vswap! *pass* inc) (println (str "  PASS: " desc " (" h ")")))
    (do (vswap! *fail* inc)
        (println (str "  FAIL: " desc " — handler=" h " compiled=" c
                      " (not bit-identical)")))))

(defn nt-lp
  "The registry's :log-prob for a dist-type (the compiled path's math)."
  [dist-type]
  (:log-prob (get compiled/noise-transforms-full dist-type)))

(defn h-lp
  "Handler ground truth: dc/dist-log-prob on the real Distribution."
  [d v]
  (mx/item (dc/dist-log-prob d v)))

(defn c-lp
  "Compiled ground truth: the registry entry, same value + dist-args."
  [dist-type v dist-args]
  (mx/item (apply (nt-lp dist-type) v dist-args)))

(def S mx/scalar)
(defn A [xs] (mx/array (clj->js xs)))

;; --- 1/9 gaussian: unbounded support; pin agreement at overflow extremes ---
(doseq [v [1e30 -1e30 0.0]]
  (assert-parity (str "gaussian(0,1) lp(" v ")")
                 (h-lp (dist/gaussian (S 0.0) (S 1.0)) (S v))
                 (c-lp :gaussian (S v) [(S 0.0) (S 1.0)])))

;; --- 2/9 uniform: out of [lo,hi] and both inclusive boundaries ---
(doseq [v [-1.0 2.0 0.0 1.0 0.5]]
  (assert-parity (str "uniform(0,1) lp(" v ")")
                 (h-lp (dist/uniform (S 0.0) (S 1.0)) (S v))
                 (c-lp :uniform (S v) [(S 0.0) (S 1.0)])))

;; --- 3/9 bernoulli: v off {0,1} — the drift (was finite / POSITIVE) --------
(doseq [[p v] [[0.3 0.5] [0.3 2.0] [0.3 -1.0] [0.7 0.25]
               [0.0 1.0] [1.0 0.0] [0.0 0.0] [1.0 1.0] [0.3 0.0] [0.3 1.0]]]
  (assert-parity (str "bernoulli(" p ") lp(" v ")")
                 (h-lp (dist/bernoulli (S p)) (S v))
                 (c-lp :bernoulli (S v) [(S p)])))

;; the :flip alias must resolve to the SAME guarded entry
(assert-parity "flip(0.3) lp(0.5) via :flip alias"
               (h-lp (dist/flip (S 0.3)) (S 0.5))
               (c-lp :flip (S 0.5) [(S 0.3)]))

;; --- 4/9 exponential: v<0 out of support, v=0 boundary --------------------
(doseq [v [-1.0 0.0 1.5]]
  (assert-parity (str "exponential(2) lp(" v ")")
                 (h-lp (dist/exponential (S 2.0)) (S v))
                 (c-lp :exponential (S v) [(S 2.0)])))

;; --- 5/9 log-normal: v<=0 — the drift (was NaN, poisons logsumexp) --------
(doseq [v [0.0 -1.0 -1e-9]]
  (assert-parity (str "log-normal(0,1) lp(" v ") must be -Inf, not NaN")
                 (h-lp (dist/log-normal (S 0.0) (S 1.0)) (S v))
                 (c-lp :log-normal (S v) [(S 0.0) (S 1.0)])))

;; --- 6/9 delta: point-mass mismatch, scalar and joint vector --------------
(assert-parity "delta(1) lp(2)"
               (h-lp (dist/delta (S 1.0)) (S 2.0))
               (c-lp :delta (S 2.0) [(S 1.0)]))
(assert-parity "delta([1 2 3]) lp([1 2 4]) — joint -Inf on one mismatch"
               (h-lp (dist/delta (A [1.0 2.0 3.0])) (A [1.0 2.0 4.0]))
               (c-lp :delta (A [1.0 2.0 4.0]) [(A [1.0 2.0 3.0])]))
(assert-parity "delta([1 2 3]) lp([1 2 3]) — exact match is 0"
               (h-lp (dist/delta (A [1.0 2.0 3.0])) (A [1.0 2.0 3.0]))
               (c-lp :delta (A [1.0 2.0 3.0]) [(A [1.0 2.0 3.0])]))

;; --- 7/9 laplace: unbounded; extremes must not diverge --------------------
(doseq [v [1e30 -1e30 0.5]]
  (assert-parity (str "laplace(0,1) lp(" v ")")
                 (h-lp (dist/laplace (S 0.0) (S 1.0)) (S v))
                 (c-lp :laplace (S v) [(S 0.0) (S 1.0)])
                 1e24))                       ;; |lp| ~ 1e30 here: relative tol

;; --- 8/9 cauchy: unbounded; extremes must not diverge ---------------------
(doseq [v [1e30 -1e30 0.5]]
  (assert-parity (str "cauchy(0,1) lp(" v ")")
                 (h-lp (dist/cauchy (S 0.0) (S 1.0)) (S v))
                 (c-lp :cauchy (S v) [(S 0.0) (S 1.0)])))

;; --- 9/9 iid-gaussian: joint over the event axis --------------------------
(doseq [vs [[1e30 0.0 -1e30] [0.1 0.2 0.3]]]
  (assert-parity (str "iid-gaussian(0,1,3) lp(" vs ")")
                 (h-lp (dist/iid-gaussian (S 0.0) (S 1.0) 3) (A vs))
                 (c-lp :iid-gaussian (A vs) [(S 0.0) (S 1.0) 3])))

;; --- in-support results must be BIT-identical, not merely close -----------
;; The support guard must be a pure mx/where wrapper: it may not perturb the
;; in-support expression by even one ulp. log-normal at v=1e-6 is the sentinel
;; — the pre-fix compiled summand order gave -12.702530860900879 where the
;; handler gives -12.702531814575195 (one float32 ulp apart).
(doseq [[mu sg v] [[0.3 2.0 1e-6] [0.3 2.0 100.0] [0.0 1.0 1.5]
                   [-1.0 0.5 2.25] [0.0 1.0 0.5]]]
  (assert-bit-identical (str "log-normal(" mu "," sg ") lp(" v ") bit-identical")
                        (h-lp (dist/log-normal (S mu) (S sg)) (S v))
                        (c-lp :log-normal (S v) [(S mu) (S sg)])))
(doseq [[p v] [[0.3 0.0] [0.3 1.0] [0.7 0.0] [0.7 1.0] [0.0 0.0] [1.0 1.0]]]
  (assert-bit-identical (str "bernoulli(" p ") lp(" v ") bit-identical")
                        (h-lp (dist/bernoulli (S p)) (S v))
                        (c-lp :bernoulli (S v) [(S p)])))

;; --- model level: compiled assess vs strip-compiled (the real seam) -------
(println "\n-- Section 9b: model-level assess, compiled vs strip-compiled --")

(def ln-model (dyn/auto-key (gen [] (trace :x (dist/log-normal 0.0 1.0)))))

(assert-true "log-normal model: compiled-assess attached"
             (some? (:compiled-assess (:schema ln-model))))

(doseq [v [-1.0 0.0 1.5]]
  (let [choices (cmv {:x v})
        w-c (mx/item (:weight (p/assess ln-model [] choices)))
        w-h (mx/item (:weight (p/assess (dyn/auto-key (gfi/strip-compiled ln-model))
                                        [] choices)))]
    (assert-parity (str "ln-model assess x=" v ": compiled = handler") w-h w-c)))

(doseq [v [0.5 2.0 -1.0 0.0 1.0]]
  (let [choices (cmv {:b v})
        w-c (mx/item (:weight (p/assess bern-model [0.3] choices)))
        w-h (mx/item (:weight (p/assess (dyn/auto-key (gfi/strip-compiled bern-model))
                                        [0.3] choices)))]
    (assert-parity (str "bern-model assess p=0.3 b=" v ": compiled = handler")
                   w-h w-c)))

;; ===========================================================================
(println "\n==========================================")
(println (str "  compiled-parity: " @*pass* " passed, " @*fail* " failed"))
(println "==========================================")
(when (pos? @*fail*) (js/process.exit 1))
