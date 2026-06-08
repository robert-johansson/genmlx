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
            [genmlx.selection :as sel]
            [genmlx.mlx.random :as rng]
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
;; SECTION 3 — Honesty: the non-conjugate NOISE latent stays residual
;; ===========================================================================
;; sigma is the obs noise and is itself a (non-conjugate) latent. Under genmlx-4q9d
;; the affine block {mu} IS eliminated CONDITIONAL on sigma — but sigma must NEVER be
;; swept into the block: it stays residual (it is sampled, not analytically eliminated).

(println "\n== Section 3: non-conjugate noise latent stays residual ==")

(def latent-noise-model
  (dyn/auto-key
    (gen []
      (let [sigma (trace :sigma (dist/gamma-dist 2 1))
            mu    (trace :mu (dist/gaussian 0 10))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply mu 2) 0) sigma))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply mu 2) 0) sigma))
        mu))))
(def latent-noise-obs
  (-> cm/EMPTY (cm/set-choice [:y1] (mx/scalar 1.5)) (cm/set-choice [:y2] (mx/scalar 2.0))))
(println "-- non-conjugate noise: sigma residual, block mu eliminated conditional on it")
(let [sel (ms/select-method latent-noise-model latent-noise-obs)]
  (assert-true "latent-noise: sigma NOT eliminated (honesty)"
               (not (contains? (set (:eliminated sel)) :sigma)))
  (assert-true "latent-noise: sigma stays residual" (contains? (:residual-addrs sel) :sigma))
  (assert-true "latent-noise: block mu eliminated conditional on sigma (4q9d)"
               (contains? (set (:eliminated sel)) :mu))
  (assert-true "latent-noise: method != exact (sigma residual)" (not= :exact (:method sel))))

;; ===========================================================================
;; SECTION 4 — Regenerate: block stays Rao-Blackwellised under MH (genmlx-m3tn)
;; ===========================================================================
;; A model with an ELIGIBLE regression block {slope,intercept} (constant noise)
;; PLUS an unrelated residual latent `tau` (Gamma prior, Gaussian-scale obs `w`).
;; tau is non-conjugate w.r.t. its scale obs, so it stays residual; the block is
;; fully eliminated and a-posteriori independent of tau (no shared data). The
;; Rao-Blackwell win: the block latents are returned as EXACT posterior means
;; (zero Monte-Carlo variance) while an MH chain explores tau.

(println "\n== Section 4: regenerate (genmlx-m3tn) ==")

(defn- strip-analytical
  "Force the standard handler/compiled path (drop analytical auto-handlers)."
  [gf]
  (assoc gf :schema (dissoc (:schema gf)
                            :auto-handlers :auto-regenerate-handlers
                            :auto-regenerate-transition :conjugate-pairs
                            :has-conjugate? :analytical-plan :linear-gaussian-blocks)))

(def br-model
  (dyn/auto-key
    (gen [x1 x2 x3 x4 x5]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))
            tau       (trace :tau (dist/gamma-dist 2 1))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x1) intercept) 1))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply slope x2) intercept) 1))
        (trace :y3 (dist/gaussian (mx/add (mx/multiply slope x3) intercept) 1))
        (trace :y4 (dist/gaussian (mx/add (mx/multiply slope x4) intercept) 1))
        (trace :y5 (dist/gaussian (mx/add (mx/multiply slope x5) intercept) 1))
        (trace :w  (dist/gaussian 0 tau))
        slope))))

(def br-xs (mapv mx/scalar [1.0 2.0 3.0 4.0 5.0]))
(def br-w 1.5)
(def br-obs
  (-> cm/EMPTY
      (cm/set-choice [:y1] (mx/scalar 2.3)) (cm/set-choice [:y2] (mx/scalar 4.7))
      (cm/set-choice [:y3] (mx/scalar 6.1)) (cm/set-choice [:y4] (mx/scalar 8.9))
      (cm/set-choice [:y5] (mx/scalar 10.2))
      (cm/set-choice [:w]  (mx/scalar br-w))))

;; Independent oracle for tau: p(tau|w) ∝ Gamma(tau;2,1)·N(w;0,tau)
;;   ∝ tau·e^{-tau} · tau^{-1}·e^{-w²/(2τ²)} = e^{-tau - w²/(2τ²)}, tau>0.
;; 1D quadrature — uses only the model densities, NOT the eliminator.
(defn tau-posterior-mean [w]
  (let [grid (mapv #(* 0.001 %) (range 1 30000))      ; tau in (0,30]
        dens (mapv (fn [t] (js/Math.exp (- (- t) (/ (* w w) (* 2 t t))))) grid)
        z    (reduce + dens)
        num  (reduce + (mapv * grid dens))]
    (/ num z)))

;; -- method-selection honesty
(println "-- method-selection: block eliminated, tau residual")
(let [sel (ms/select-method br-model br-obs)]
  (assert-true "eliminated = {slope intercept}" (= #{:slope :intercept} (set (:eliminated sel))))
  (assert-true "tau is residual" (contains? (:residual-addrs sel) :tau))
  (assert-true "slope NOT residual" (not (contains? (:residual-addrs sel) :slope)))
  (assert-true "intercept NOT residual" (not (contains? (:residual-addrs sel) :intercept)))
  (assert-true "method != exact (tau residual)" (not= :exact (:method sel))))

;; -- generate: block latents = analytic conditional; tau sampled
(println "-- generate: block posterior means = analytic conditional")
(let [{:keys [trace]} (p/generate br-model br-xs br-obs)]
  (assert-close "gen slope = posterior mean" 1.999324 (choice-val trace :slope) POST-TOL)
  (assert-close "gen intercept = posterior mean" 0.441145 (choice-val trace :intercept) POST-TOL)
  (assert-true  "tau present & positive" (pos? (choice-val trace :tau))))

;; -- strip-analytical: block latents are genuinely sampled (not eliminated)
(println "-- strip-analytical: block latents are sampled, not pinned to posterior mean")
(let [{:keys [trace]} (p/generate (dyn/with-key (strip-analytical br-model) (rng/fresh-key 5))
                                  br-xs br-obs)]
  (assert-true "stripped slope != analytic posterior mean"
               (> (js/Math.abs (- (choice-val trace :slope) 1.999324)) 1e-3)))

;; -- Case B (empty selection): nothing resampled -> score unchanged, weight 0,
;;    block stays at the exact posterior mean (mirrors l3_5_regenerate Case B).
(println "-- Case B (empty selection): score consistency + weight 0")
(let [{:keys [trace]} (p/generate br-model br-xs br-obs)
      old-score (mx/item (:score trace))
      {rtrace :trace rweight :weight} (p/regenerate br-model trace (sel/select))
      new-score (mx/item (:score rtrace))]
  (assert-close "regen score = generate score" old-score new-score MARG-TOL)
  (assert-close "regen weight = 0" 0.0 (mx/item rweight) MARG-TOL)
  (assert-close "slope still posterior mean" 1.999324 (choice-val rtrace :slope) POST-TOL)
  (assert-close "intercept still posterior mean" 0.441145 (choice-val rtrace :intercept) POST-TOL))

;; -- Case A (select a block latent): the WHOLE block re-opens (option a).
;;    Selected latent resamples from prior; the unselected block latent is kept
;;    at its old value (base regenerate); weight is finite & MH-valid.
(println "-- Case A (select :slope): whole block re-opens, weight finite")
(let [{:keys [trace]} (p/generate (dyn/with-key br-model (rng/fresh-key 7)) br-xs br-obs)
      {rtrace :trace rweight :weight}
      (p/regenerate (dyn/with-key br-model (rng/fresh-key 13)) trace (sel/select :slope))]
  (assert-true  "Case A weight finite" (js/isFinite (mx/item rweight)))
  (assert-close "intercept (unselected) kept at posterior mean"
                0.441145 (choice-val rtrace :intercept) POST-TOL))

;; -- MH over tau (analytical): converges to the oracle, AND the eliminated block
;;    stays at the EXACT posterior mean every step (zero MC variance = Rao-Blackwell).
(println "-- MH over tau (analytical): oracle convergence + zero-variance block")
(let [n-steps 200 burn 50
      oracle (tau-posterior-mean br-w)
      init (:trace (p/generate (dyn/with-key br-model (rng/fresh-key 1)) br-xs br-obs))
      chain (reduce
              (fn [{:keys [trace taus slopes accepts]} i]
                (let [{rtrace :trace w :weight}
                      (p/regenerate (dyn/with-key br-model (rng/fresh-key (+ 2000 i)))
                                    trace (sel/select :tau))
                      la (mx/item w)
                      accept? (< (js/Math.log (js/Math.random)) la)
                      nt (if accept? rtrace trace)]
                  {:trace nt
                   :taus (conj taus (choice-val nt :tau))
                   :slopes (conj slopes (choice-val nt :slope))
                   :accepts (if accept? (inc accepts) accepts)}))
              {:trace init :taus [] :slopes [] :accepts 0}
              (range n-steps))
      post-taus  (subvec (:taus chain) burn)
      tau-mean   (/ (reduce + post-taus) (count post-taus))
      slopes     (:slopes chain)
      slope-mean (/ (reduce + slopes) (count slopes))
      slope-var  (/ (reduce + (mapv #(let [d (- % slope-mean)] (* d d)) slopes)) (count slopes))]
  (assert-true  "MH accept rate > 0" (pos? (:accepts chain)))
  (assert-true  (str "tau chain mean ~ oracle (" (.toFixed oracle 4) ")")
                (< (js/Math.abs (- tau-mean oracle)) 0.35))
  (assert-close "block slope stays exact posterior mean" 1.999324 slope-mean POST-TOL)
  (assert-true  "block slope has ~zero MC variance (Rao-Blackwell)" (< slope-var 1e-6)))

;; ===========================================================================
;; SECTION 5 — Partial conjugacy: conditional elimination over a residual (4q9d)
;; ===========================================================================
;; Block {slope,intercept} with NON-CONJUGATE noise: y_j ~ N(slope*x_j+intercept, sigma),
;; sigma ~ Gamma. v1 declined this. 4q9d samples sigma from prior and eliminates the block
;; CONDITIONAL on the sampled sigma -> per-sample exact block marginal; the population gives
;; E_sigma[block marginal] = the true marginal (Rao-Blackwellised IS / MH).
;;
;; Oracle: pure float64 JS quadrature of the JOINT density — independent of the Kalman
;; eliminator. cond-quad integrates out (slope,intercept) analytically by brute force.

(println "\n== Section 5: partial conjugacy / conditional elimination (genmlx-4q9d) ==")

(def br4-xs (mapv mx/scalar [1.0 2.0 3.0]))
(def br4-obs
  (-> cm/EMPTY (cm/set-choice [:y1] (mx/scalar 1.6))
      (cm/set-choice [:y2] (mx/scalar 2.4)) (cm/set-choice [:y3] (mx/scalar 3.7))))

;; sigma declared FIRST (precedes the block obs in dep-order — required so its sampled
;; value is in :choices when the block obs handlers fire and read the noise).
(def br4-model
  (dyn/auto-key
    (gen [x1 x2 x3]
      (let [sigma     (trace :sigma (dist/gamma-dist 2 1))
            slope     (trace :slope (dist/gaussian 0 3))
            intercept (trace :intercept (dist/gaussian 0 3))]
        (trace :y1 (dist/gaussian (mx/add (mx/multiply slope x1) intercept) sigma))
        (trace :y2 (dist/gaussian (mx/add (mx/multiply slope x2) intercept) sigma))
        (trace :y3 (dist/gaussian (mx/add (mx/multiply slope x3) intercept) sigma))
        slope))))

;; -- Independent quadrature oracle (float64 JS; vs the float32 eliminator) --
;; Grids kept modest + inner loop free of log/alloc (nbb is interpreted): per-sigma
;; constants are hoisted, residuals computed inline for the 3 obs.
(def LOG2PI (js/Math.log (* 2 js/Math.PI)))
(def y0 1.6) (def y1 2.4) (def y2 3.7)            ; br4-y, x = [1 2 3]
(defn log-gamma21 [sig] (- (js/Math.log sig) sig)) ; Gamma(2,1): logpdf = log(sig) - sig

(def DA 0.07) (def DB 0.07) (def DSIG 0.08)
(def AS (mapv #(+ -1.5 (* DA %)) (range 80)))      ; slope grid [-1.5, 4.1)
(def BS (mapv #(+ -3.0 (* DB %)) (range 100)))     ; intercept grid [-3.0, 4.0)
(def QS (mapv #(+ 0.02 (* DSIG %)) (range 100)))   ; sigma grid [0.02, 8.02)
;; prior log-density N(0,3), per grid value, paired with the value to avoid nth in loop
(defn- lg03 [x] (let [d (/ x 3.0)] (- (* -0.5 d d) (js/Math.log 3.0) (* 0.5 LOG2PI))))
(def A-PAIRS (mapv (fn [a] [a (lg03 a)]) AS))
(def B-PAIRS (mapv (fn [b] [b (lg03 b)]) BS))
(def LOG-DA-DB (+ (js/Math.log DA) (js/Math.log DB)))

;; log p(y|sigma) via 2D quadrature over (slope,intercept) — online logsumexp, 1 pass.
(defn cond-logm [sig]
  (let [s2 (* sig sig)
        cst (* -3.0 (+ (js/Math.log sig) (* 0.5 LOG2PI)))   ; 3-obs noise normaliser (per sig)
        m (volatile! -1e30) s (volatile! 0.0)]
    (doseq [[a la] A-PAIRS [b lb] B-PAIRS]
      (let [r0 (- y0 (+ a b)) r1 (- y1 (+ (* 2.0 a) b)) r2 (- y2 (+ (* 3.0 a) b))
            lp (+ la lb cst (* -0.5 (/ (+ (* r0 r0) (* r1 r1) (* r2 r2)) s2)))]
        (if (> lp @m)
          (do (vreset! s (+ (* @s (js/Math.exp (- @m lp))) 1.0)) (vreset! m lp))
          (vswap! s + (js/Math.exp (- lp @m))))))
    (+ @m (js/Math.log @s) LOG-DA-DB)))

;; conditional posterior means E[slope|y,sigma], E[intercept|y,sigma] (called ≤2×).
(defn cond-means [sig]
  (let [s2 (* sig sig)
        triples (vec (for [[a la] A-PAIRS [b lb] B-PAIRS]
                       (let [r0 (- y0 (+ a b)) r1 (- y1 (+ (* 2.0 a) b)) r2 (- y2 (+ (* 3.0 a) b))]
                         [a b (+ la lb (* -0.5 (/ (+ (* r0 r0) (* r1 r1) (* r2 r2)) s2)))])))
        mx-lp (reduce max (mapv #(nth % 2) triples))
        ws (mapv (fn [t] (js/Math.exp (- (nth t 2) mx-lp))) triples)
        z  (reduce + ws)]
    [(/ (reduce + (mapv (fn [t w] (* (nth t 0) w)) triples ws)) z)
     (/ (reduce + (mapv (fn [t w] (* (nth t 1) w)) triples ws)) z)]))

(defn logsumexp [xs]
  (let [m (reduce max xs)] (+ m (js/Math.log (reduce + (mapv #(js/Math.exp (- % m)) xs))))))

;; single pass over sigma; derive both the marginal and E[sigma|y]
(def sigma-logw (mapv (fn [sig] [sig (+ (log-gamma21 sig) (cond-logm sig))]) QS))
(def oracle-log-py
  (logsumexp (mapv (fn [[_ lw]] (+ lw (js/Math.log DSIG))) sigma-logw)))
(def oracle-sigma-mean
  (let [m (reduce max (mapv second sigma-logw))
        w (mapv (fn [[_ lw]] (js/Math.exp (- lw m))) sigma-logw)
        z (reduce + w)]
    (/ (reduce + (mapv (fn [[sig _] wi] (* sig wi)) sigma-logw w)) z)))

;; -- method-selection honesty
(println "-- method-selection: block eliminated, sigma residual")
(let [sel (ms/select-method br4-model br4-obs)]
  (assert-true "4q9d eliminated = {slope intercept}" (= #{:slope :intercept} (set (:eliminated sel))))
  (assert-true "4q9d sigma residual" (contains? (:residual-addrs sel) :sigma))
  (assert-true "4q9d slope NOT residual" (not (contains? (:residual-addrs sel) :slope)))
  (assert-true "4q9d method != exact (sigma residual)" (not= :exact (:method sel))))

;; -- per residual sample: block marginal is EXACT (matches the independent oracle)
(println "-- per-sample: conditional posterior mean + weight = analytic block marginal")
(let [{:keys [trace weight]} (p/generate (dyn/with-key br4-model (rng/fresh-key 3)) br4-xs br4-obs)
      sig (choice-val trace :sigma)
      [em-slope em-int] (cond-means sig)]
  (assert-close "block slope = conditional posterior mean" em-slope (choice-val trace :slope) 2e-2)
  (assert-close "block intercept = conditional posterior mean" em-int (choice-val trace :intercept) 2e-2)
  (assert-close "generate weight = conditional log-marginal" (cond-logm sig) (mx/item weight) 5e-2))

;; -- marginal (IS over sigma~prior, block eliminated) matches oracle + strictly lower
;;    variance / higher ESS than full sampling (strip-analytical).
(println "-- IS marginal ~ oracle + ESS(conditional) > ESS(full sampling)")
(let [n 400
      gen-w (fn [gf base] (mapv (fn [i]
                                  (mx/item (:weight (p/generate (dyn/with-key gf (rng/fresh-key (+ base i)))
                                                                br4-xs br4-obs))))
                                (range n)))
      cond-w  (gen-w br4-model 5000)
      strip-w (gen-w (strip-analytical br4-model) 9000)
      lme  (fn [ws] (- (logsumexp ws) (js/Math.log (count ws))))
      ess  (fn [ws] (let [m (reduce max ws)
                          e (mapv #(js/Math.exp (- % m)) ws)
                          s (reduce + e) s2 (reduce + (mapv #(* % %) e))]
                      (/ (* s s) s2)))
      cond-est (lme cond-w)
      ess-cond (ess cond-w)
      ess-full (ess strip-w)]
  (println (str "     oracle log p(y)=" (.toFixed oracle-log-py 4)
                "  cond-IS=" (.toFixed cond-est 4)
                "  ESS cond=" (.toFixed ess-cond 1) " full=" (.toFixed ess-full 1)))
  (assert-close "conditional-elim IS marginal ~ oracle" oracle-log-py cond-est 0.2)
  (assert-true  "ESS(conditional) > ESS(full sampling) — Rao-Blackwell" (> ess-cond ess-full)))

;; -- regenerate: MH over sigma re-eliminates the block conditional on the current sigma
(println "-- regenerate: MH over sigma; block re-eliminated conditional on current sigma")
(let [n-steps 150 burn 40
      init (:trace (p/generate (dyn/with-key br4-model (rng/fresh-key 11)) br4-xs br4-obs))
      chain (reduce
              (fn [{:keys [trace sigs accepts]} i]
                (let [{rt :trace w :weight}
                      (p/regenerate (dyn/with-key br4-model (rng/fresh-key (+ 7000 i)))
                                    trace (sel/select :sigma))
                      accept? (< (js/Math.log (js/Math.random)) (mx/item w))
                      nt (if accept? rt trace)]
                  {:trace nt :sigs (conj sigs (choice-val nt :sigma))
                   :accepts (if accept? (inc accepts) accepts)}))
              {:trace init :sigs [] :accepts 0}
              (range n-steps))
      post     (subvec (:sigs chain) burn)
      sig-mean (/ (reduce + post) (count post))
      final    (:trace chain)
      fsig     (choice-val final :sigma)
      [fem-slope _] (cond-means fsig)]
  (assert-true  "MH accept rate > 0" (pos? (:accepts chain)))
  (assert-true  (str "sigma chain mean ~ oracle (" (.toFixed oracle-sigma-mean 4) ")")
                (< (js/Math.abs (- sig-mean oracle-sigma-mean)) 0.35))
  (assert-close "block re-eliminated conditional on current sigma"
                fem-slope (choice-val final :slope) 2e-2))

;; ===========================================================================
;; Summary
;; ===========================================================================

(println (str "\n=========================================="))
(println (str "  linear-gaussian-elim: " @*pass* " passed, " @*fail* " failed"))
(println (str "=========================================="))
(when (pos? @*fail*) (js/process.exit 1))
