;; @tier medium
;; SMC evidence bookkeeping tests (bean genmlx-vdpx).
;;
;; Verifies, against INDEPENDENT host-math oracles (closed-form normal-normal
;; marginal, host Gaussian log-pdf, telescoping identities — never the code
;; under test):
;;   1. smc/csmc/smcp3/vsmc log-ML increments do NOT double-count when
;;      resampling is skipped: with ess-threshold 0 the total telescopes to
;;      logsumexp(final weights) - log N exactly.
;;   2. smc log-ML converges to the analytic linear-Gaussian evidence when the
;;      per-step weights are proper (all observations at init, empty later
;;      steps), with and without resampling enabled.
;;   3. csmc's reference particle carries an obs-only weight (project of the
;;      observation addresses) on the same scale as the other particles, its
;;      trajectory is retained, and unified stripping keeps particle diversity.
;;   4. smcp3 strips analytical handlers (particle diversity on L3 models) and
;;      applies the step observations in the proposal-edit branch.
;;
;; Run: bun run --bun nbb test/genmlx/smc_evidence_test.cljs

(ns genmlx.smc-evidence-test
  (:require [genmlx.protocols :as p]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.smcp3 :as smcp3])
  (:require-macros [genmlx.gen :refer [gen]]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-close [msg expected actual tol]
  (if (<= (Math/abs (- expected actual)) tol)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc) (println " FAIL" msg "  expected:" expected "  got:" actual))))

(defn realize [x] (mx/eval! x) (mx/item x))

;; ---------------------------------------------------------------------------
;; Independent host-math oracles
;; ---------------------------------------------------------------------------

(def LOG-2PI (js/Math.log (* 2 js/Math.PI)))

(defn gaussian-logpdf [x mu sigma]
  (let [z (/ (- x mu) sigma)]
    (- (* -0.5 (+ LOG-2PI (* z z))) (js/Math.log sigma))))

(defn nn-shared-marginal
  "Closed-form log p(y) for mu ~ N(m0, prior-var); y_i ~ N(mu, obs-var):
   y ~ N(m0*1, obs-var*I + prior-var*1·1ᵀ). Derived independently of any
   GenMLX path (matrix-determinant lemma on the rank-1 covariance)."
  [ys m0 prior-var obs-var]
  (let [n (count ys)
        ds (map #(- % m0) ys)
        sum-d (reduce + ds)
        sum-d2 (reduce + (map #(* % %) ds))
        denom (+ obs-var (* n prior-var))
        logdet (+ (* (dec n) (js/Math.log obs-var)) (js/Math.log denom))
        quad (/ (- sum-d2 (* (/ prior-var denom) sum-d sum-d)) obs-var)]
    (* -0.5 (+ (* n LOG-2PI) logdet quad))))

(defn host-lse
  "logsumexp of realized log-weights, host-side."
  [ws]
  (let [m (apply max ws)]
    (+ m (js/Math.log (reduce + (map #(js/Math.exp (- % m)) ws))))))

;; ---------------------------------------------------------------------------
;; Model: normal-normal, conjugate (exercises stripping) — 3 observations
;; ---------------------------------------------------------------------------

(def model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (trace :y0 (dist/gaussian mu 1))
        (trace :y1 (dist/gaussian mu 1))
        (trace :y2 (dist/gaussian mu 1))
        mu))))

(def ys [2.8 3.1 2.9])
(def closed-form (nn-shared-marginal ys 0.0 100.0 1.0))

(def full-obs
  (cm/choicemap :y0 (mx/scalar 2.8) :y1 (mx/scalar 3.1) :y2 (mx/scalar 2.9)))

(def incremental-obs
  (mapv (fn [[i y]] (cm/choicemap (keyword (str "y" i)) (mx/scalar y)))
        (map-indexed vector ys)))

(defn mu-of [trace] (realize (cm/get-choice (:choices trace) [:mu])))

(println (str "closed-form log p(y) = " closed-form))

;; ===========================================================================
(println "\n== Section 1: smc increment bookkeeping — telescoping identity ==")
;; With ess-threshold 0 resampling never fires, so the increments must
;; telescope: total log-ML == logsumexp(final weights) - log N, EXACTLY,
;; regardless of the per-step weight semantics. The pre-fix code summed
;; lse(W_t) - log N every step, re-counting the carried mass.
(let [n 100
      {:keys [log-weights log-ml-estimate]}
      (smc/smc {:particles n :ess-threshold 0 :key (rng/fresh-key 11)}
               model [] incremental-obs)
      ws (mapv realize log-weights)
      expected (- (host-lse ws) (js/Math.log n))]
  (assert-close "smc no-resample: log-ML == lse(final W) - log N"
                expected (realize log-ml-estimate) 1e-3))

;; ===========================================================================
(println "\n== Section 2: smc log-ML vs analytic linear-Gaussian oracle ==")
;; All observations constrained at init (proper IS weights), then two EMPTY
;; steps. The empty steps must contribute exactly 0 to the evidence — under
;; the pre-fix bookkeeping each empty step re-added the full IS estimate.
(let [n 2000
      empty-steps [full-obs cm/EMPTY cm/EMPTY]
      r0 (smc/smc {:particles n :ess-threshold 0 :key (rng/fresh-key 7)}
                  model [] empty-steps)
      ws (mapv realize (:log-weights r0))
      ident (- (host-lse ws) (js/Math.log n))
      r5 (smc/smc {:particles n :ess-threshold 0.5 :key (rng/fresh-key 7)}
                  model [] empty-steps)]
  (assert-close "smc (never resample): log-ML ≈ closed form"
                closed-form (realize (:log-ml-estimate r0)) 0.30)
  (assert-close "smc (never resample): identity lse(final)-logN"
                ident (realize (:log-ml-estimate r0)) 1e-3)
  (assert-close "smc (adaptive resample): log-ML ≈ closed form"
                closed-form (realize (:log-ml-estimate r5)) 0.30))

;; ===========================================================================
(println "\n== Section 3: csmc reference particle scale + retention + diversity ==")
(let [stripped (smc/strip-analytical model)
      ref-mu 2.5
      ref-cm (cm/set-choice full-obs [:mu] (mx/scalar ref-mu))
      ref-trace (:trace (p/generate stripped [] ref-cm))
      n 30
      {:keys [traces log-weights]}
      (smc/csmc {:particles n :key (rng/fresh-key 21)}
                model [] [full-obs] ref-trace)
      ws (mapv realize log-weights)
      ;; obs-only oracle: sum of host Gaussian log-pdfs of the observations
      obs-ll (fn [mu] (reduce + (map #(gaussian-logpdf % mu 1.0) ys)))
      mus (mapv mu-of traces)]
  (assert-close "csmc ref trajectory retained (mu at index 0)"
                ref-mu (nth mus 0) 1e-6)
  (assert-close "csmc ref weight = obs-only scale: sum_i log p(y_i | mu_ref)"
                (obs-ll ref-mu) (nth ws 0) 1e-3)
  (assert-close "csmc particle 1 weight on the SAME obs-only scale"
                (obs-ll (nth mus 1)) (nth ws 1) 1e-3)
  (assert-close "csmc particle 2 weight on the SAME obs-only scale"
                (obs-ll (nth mus 2)) (nth ws 2) 1e-3)
  (assert-true "csmc non-ref particles diverse (stripping fires on conjugate model)"
               (> (count (distinct (rest mus))) 1)))

;; csmc telescoping identity across incremental steps (no resampling)
(let [stripped (smc/strip-analytical model)
      ref-cm (cm/set-choice full-obs [:mu] (mx/scalar 2.5))
      ref-trace (:trace (p/generate stripped [] ref-cm))
      n 30
      {:keys [log-weights log-ml-estimate]}
      (smc/csmc {:particles n :ess-threshold 0 :key (rng/fresh-key 22)}
                model [] incremental-obs ref-trace)
      ws (mapv realize log-weights)
      expected (- (host-lse ws) (js/Math.log n))]
  (assert-close "csmc no-resample: log-ML == lse(final W) - log N"
                expected (realize log-ml-estimate) 1e-3))

;; ===========================================================================
(println "\n== Section 4: smcp3 — stripping + observations in the proposal branch ==")
;; 4a. Standard path (no kernels) on the conjugate model: stripping must keep
;; particle diversity (the analytical path would emit N identical posterior
;; means — and an exactly-correct log-ML that silently tests L3, not SMC).
(let [n 1000
      {:keys [traces log-ml-estimate]}
      (smcp3/smcp3 {:particles n :key (rng/fresh-key 31)}
                   model [] [full-obs])
      mus (mapv mu-of traces)]
  (assert-true "smcp3 particles diverse on conjugate model (analytical stripped)"
               (> (count (distinct mus)) 1))
  (assert-close "smcp3 (single step = IS): log-ML ≈ closed form"
                closed-form (realize log-ml-estimate) 0.40))

;; 4b. Proposal-edit branch must apply the step's observations. The forward
;; and backward kernels propose a fresh :mu; before the fix obs-t never
;; reached the trace, so :y1 kept its prior-sampled value.
(let [fwd (dyn/auto-key
            (gen [prev-choices]
              (trace :mu (dist/gaussian 0 10))))
      bwd (dyn/auto-key
            (gen [prev-choices]
              (trace :mu (dist/gaussian 0 10))))
      two-step-model (dyn/auto-key
                       (gen []
                         (let [mu (trace :mu (dist/gaussian 0 10))]
                           (trace :y0 (dist/gaussian mu 1))
                           (trace :y1 (dist/gaussian mu 1))
                           mu)))
      obs-seq [(cm/choicemap :y0 (mx/scalar 2.8))
               (cm/choicemap :y1 (mx/scalar 3.1))]
      n 30
      {:keys [traces log-weights log-ml-estimate]}
      (smcp3/smcp3 {:particles n :forward-kernel fwd :backward-kernel bwd
                    :key (rng/fresh-key 41)}
                   two-step-model [] obs-seq)
      y1-vals (mapv (fn [tr] (realize (cm/get-choice (:choices tr) [:y1]))) traces)
      ws (mapv realize log-weights)]
  (assert-true "smcp3 proposal branch: observed :y1 applied to EVERY particle"
               (every? #(< (Math/abs (- % 3.1)) 1e-6) y1-vals))
  (assert-true "smcp3 proposal branch: all weights finite"
               (every? js/isFinite ws))
  (assert-true "smcp3 proposal branch: log-ML finite"
               (js/isFinite (realize log-ml-estimate))))

;; 4c. smcp3 telescoping identity (proposal branch, never resample)
(let [fwd (dyn/auto-key (gen [prev-choices] (trace :mu (dist/gaussian 0 10))))
      bwd (dyn/auto-key (gen [prev-choices] (trace :mu (dist/gaussian 0 10))))
      two-step-model (dyn/auto-key
                       (gen []
                         (let [mu (trace :mu (dist/gaussian 0 10))]
                           (trace :y0 (dist/gaussian mu 1))
                           (trace :y1 (dist/gaussian mu 1))
                           mu)))
      obs-seq [(cm/choicemap :y0 (mx/scalar 2.8))
               (cm/choicemap :y1 (mx/scalar 3.1))]
      n 30
      {:keys [log-weights log-ml-estimate]}
      (smcp3/smcp3 {:particles n :forward-kernel fwd :backward-kernel bwd
                    :ess-threshold 0 :key (rng/fresh-key 43)}
                   two-step-model [] obs-seq)
      ws (mapv realize log-weights)
      expected (- (host-lse ws) (js/Math.log n))]
  (assert-close "smcp3 no-resample: log-ML == lse(final W) - log N"
                expected (realize log-ml-estimate) 1e-3))

;; ===========================================================================
(println "\n== Section 5: vsmc telescoping identity ==")
(let [n 100
      {:keys [vtrace log-ml-estimate]}
      (smc/vsmc {:particles n :ess-threshold 0 :key (rng/fresh-key 51)}
                model [] incremental-obs)
      expected (- (realize (mx/logsumexp (:weight vtrace))) (js/Math.log n))]
  (assert-close "vsmc no-resample: log-ML == lse(final W) - log N"
                expected (realize log-ml-estimate) 1e-3))

;; ===========================================================================
(println (str "\n" @passed " passed, " @failed " failed"))
(when (pos? @failed) (js/process.exit 1))
