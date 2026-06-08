(ns genmlx.linear-gaussian-elim-test
  "L3 joint linear-Gaussian elimination (genmlx-lwhw).

   Coupled multi-latent / affine linear-Gaussian regression models: multiple
   independent Gaussian-prior latents jointly determine Gaussian observations via
   affine means. L3 must compute the JOINT marginal log-ML (and joint posterior),
   NOT the old per-prior scalar composition (which dropped the affine coefficient
   and gave the shared-mean-only marginal).

   Ground truth is an INDEPENDENT oracle (numpy/closed-form, triple-verified
   batch=sequential=quadrature; see genmlx-lwhw), NEVER the function under test.

   Run: bun run --bun nbb test/genmlx/linear_gaussian_elim_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.method-selection :as ms]
            [genmlx.linear-gaussian :as lg])
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

;; Tolerances (float32 end-to-end; see oracle note)
(def MARG-TOL 2e-4)
(def POST-TOL 1e-4)

(defn choice-val [trace addr]
  (mx/item (cm/get-value (cm/get-submap (:choices trace) addr))))

;; ===========================================================================
;; SECTION 1 — Pure eliminator math vs oracle (independent of GFI wiring)
;; ===========================================================================
;; lg/lg-eliminate takes explicit m0, S0, obs specs {:y :h :c :r(variance)}.

(println "\n== Section 1: pure lg/lg-eliminate vs oracle ==")

(defn sc [x] (mx/scalar (double x)))
(defn vecarr [xs] (mx/array (mapv double xs)))

(defn run-pure [m0 s0-diag obs-specs]
  (let [m0v (vecarr m0)
        s0  (mx/diag (vecarr s0-diag))
        r   (lg/lg-eliminate m0v s0 obs-specs)]
    {:marg (mx/item (:marginal-ll r))
     :mean (mapv #(mx/item (mx/index (:post-mean r) %)) (range (count m0)))
     :std  (mapv #(js/Math.sqrt (mx/item (mx/index (mx/diag (:post-cov r)) %)))
                 (range (count m0)))}))

;; M1 linreg: slope,intercept ~ N(0,10); y_j ~ N(slope*x_j + intercept, 1)
(let [x [1 2 3 4 5] y [2.3 4.7 6.1 8.9 10.2]
      obs (mapv (fn [xj yj] {:y (sc yj) :h (vecarr [xj 1.0]) :c (sc 0.0) :r (sc 1.0)}) x y)
      {:keys [marg mean std]} (run-pure [0 0] [100 100] obs)]
  (println "-- M1 linreg (pure)")
  (assert-close "M1 marginal" -11.418803 marg MARG-TOL)
  (assert-close "M1 slope mean" 1.999324 (nth mean 0) POST-TOL)
  (assert-close "M1 intercept mean" 0.441145 (nth mean 1) POST-TOL)
  (assert-close "M1 slope std" 0.314661 (nth std 0) POST-TOL)
  (assert-close "M1 intercept std" 1.042666 (nth std 1) POST-TOL))

;; M2 affine: mu ~ N(0,5); y_j ~ N(2*mu + 1, 1)
(let [y [3.0 4.5 2.2]
      obs (mapv (fn [yj] {:y (sc yj) :h (vecarr [2.0]) :c (sc 1.0) :r (sc 1.0)}) y)
      {:keys [marg mean std]} (run-pure [0] [25] obs)]
  (println "-- M2 affine single-latent (pure)")
  (assert-close "M2 marginal" -6.998560 marg MARG-TOL)
  (assert-close "M2 mu mean" 1.112957 (nth mean 0) POST-TOL)
  (assert-close "M2 mu std" 0.288195 (nth std 0) POST-TOL))

;; M3 quadratic: a,b,c ~ N(0,10); y_j ~ N(a*x^2 + b*x + c, 0.5)
(let [x [-1 0 1 2] y [1.1 0.2 0.9 4.3]
      obs (mapv (fn [xj yj] {:y (sc yj) :h (vecarr [(* xj xj) xj 1.0]) :c (sc 0.0) :r (sc 0.25)}) x y)
      {:keys [marg mean std]} (run-pure [0 0 0] [100 100 100] obs)]
  ;; Oracle (batch matrix formula + 3D quadrature, both independent, agree to 1e-6).
  ;; NB: supersedes an erroneous earlier derivation (-12.497356); -12.209727 is
  ;; confirmed by np.linalg batch AND brute-force quadrature.
  (println "-- M3 quadratic (pure)")
  (assert-close "M3 marginal" -12.209727 marg MARG-TOL)
  (assert-close "M3 a mean" 1.074323 (nth mean 0) POST-TOL)
  (assert-close "M3 b mean" -0.044292 (nth mean 1) POST-TOL)
  (assert-close "M3 c mean" 0.035639 (nth mean 2) POST-TOL))

;; M4 heteroscedastic: slope,intercept ~ N(0,10); sigma=[0.5,1,2]
(let [x [1 2 3] y [2.0 3.5 8.0] s [0.5 1.0 2.0]
      obs (mapv (fn [xj yj sj] {:y (sc yj) :h (vecarr [xj 1.0]) :c (sc 0.0) :r (sc (* sj sj))}) x y s)
      {:keys [marg mean std]} (run-pure [0 0] [100 100] obs)]
  (println "-- M4 heteroscedastic (pure)")
  (assert-close "M4 marginal" -8.999312 marg MARG-TOL)
  (assert-close "M4 slope mean" 2.300389 (nth mean 0) POST-TOL)
  (assert-close "M4 intercept mean" -0.385480 (nth mean 1) POST-TOL))

;; M5 reduces to single-latent direct (= 5A): mu ~ N(0,10); y_j ~ N(mu,1)
(let [y [1.0 1.5 0.8 1.2 1.1]
      obs (mapv (fn [yj] {:y (sc yj) :h (vecarr [1.0]) :c (sc 0.0) :r (sc 1.0)}) y)
      {:keys [marg mean]} (run-pure [0] [100] obs)]
  (println "-- M5 direct (pure)")
  (assert-close "M5 marginal" -7.843255 marg MARG-TOL)
  (assert-close "M5 mu mean" 1.117764 (nth mean 0) POST-TOL))

;; ===========================================================================
;; SECTION 2 — End-to-end GFI: p/generate computes joint marginal + posterior
;; ===========================================================================

(println "\n== Section 2: GFI p/generate (joint elimination) ==")

;; M1 linreg as a gen model
(def linreg
  (dyn/auto-key
    (gen [x1 x2 x3 x4 x5]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x1) intercept) 1))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply slope x2) intercept) 1))
        (trace :y3 (dist/gaussian (mx/add (mx/multiply slope x3) intercept) 1))
        (trace :y4 (dist/gaussian (mx/add (mx/multiply slope x4) intercept) 1))
        (trace :y5 (dist/gaussian (mx/add (mx/multiply slope x5) intercept) 1))
        slope))))

(def linreg-xs (mapv mx/scalar [1.0 2.0 3.0 4.0 5.0]))
(def linreg-obs
  (-> cm/EMPTY
      (cm/set-choice [:y1] (mx/scalar 2.3)) (cm/set-choice [:y2] (mx/scalar 4.7))
      (cm/set-choice [:y3] (mx/scalar 6.1)) (cm/set-choice [:y4] (mx/scalar 8.9))
      (cm/set-choice [:y5] (mx/scalar 10.2))))

(println "-- M1 linreg via p/generate")
(let [{:keys [trace weight]} (p/generate linreg linreg-xs linreg-obs)]
  (assert-close "M1 generate marginal weight" -11.418803 (mx/item weight) MARG-TOL)
  (assert-close "M1 posterior slope choice" 1.999324 (choice-val trace :slope) POST-TOL)
  (assert-close "M1 posterior intercept choice" 0.441145 (choice-val trace :intercept) POST-TOL))

;; method-selection must report exact (both latents eliminated)
(let [sel (ms/select-method linreg linreg-obs)]
  (assert-true "M1 method = exact" (= :exact (:method sel)))
  (assert-true "M1 eliminated slope+intercept" (= #{:slope :intercept} (set (:eliminated sel))))
  (assert-true "M1 zero residual" (zero? (:n-residual sel))))

;; M2 affine single-latent via gen
(def affine-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 5))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply 2 mu) 1) 1))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply 2 mu) 1) 1))
        (trace :y3 (dist/gaussian (mx/add (mx/multiply 2 mu) 1) 1))
        mu))))
(def affine-obs
  (-> cm/EMPTY (cm/set-choice [:y1] (mx/scalar 3.0))
      (cm/set-choice [:y2] (mx/scalar 4.5)) (cm/set-choice [:y3] (mx/scalar 2.2))))
(println "-- M2 affine via p/generate")
(let [{:keys [trace weight]} (p/generate affine-model [] affine-obs)]
  (assert-close "M2 generate marginal weight" -6.998560 (mx/item weight) MARG-TOL)
  (assert-close "M2 posterior mu choice" 1.112957 (choice-val trace :mu) POST-TOL))

;; ===========================================================================
;; SECTION 3 — Honesty: declined cases are NOT claimed exact
;; ===========================================================================

(println "\n== Section 3: declined cases stay honest ==")

;; Latent-dependent noise (sigma is itself a latent) -> NOT a linear-Gaussian block
;; (mirrors 5D's structure). Must fall through; latents stay residual.
(def latent-noise-model
  (dyn/auto-key
    (gen []
      (let [mu    (trace :mu (dist/gaussian 0 10))
            sigma (trace :sigma (dist/gamma-dist 2 1))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply mu 2) 0) sigma))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply mu 2) 0) sigma))
        mu))))
(def latent-noise-obs
  (-> cm/EMPTY (cm/set-choice [:y1] (mx/scalar 1.5)) (cm/set-choice [:y2] (mx/scalar 2.0))))
(println "-- latent-dependent noise: sigma must remain residual")
(let [sel (ms/select-method latent-noise-model latent-noise-obs)]
  (assert-true "latent-noise: sigma NOT eliminated" (not (contains? (set (:eliminated sel)) :sigma)))
  (assert-true "latent-noise: method != exact OR sigma residual"
               (or (not= :exact (:method sel))
                   (contains? (:residual-addrs sel) :sigma))))

;; ===========================================================================
;; Summary
;; ===========================================================================

(println (str "\n=========================================="))
(println (str "  linear-gaussian-elim: " @*pass* " passed, " @*fail* " failed"))
(println (str "=========================================="))
(when (pos? @*fail*) (js/process.exit 1))
