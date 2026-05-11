;; Habituation-as-Optimal-Filtering in GenMLX
;; ===========================================
;;
;; Reproduces Sam Gershman's "Habituation as optimal filtering"
;; (Gershman 2024, iScience 27:110523) — the simplest, most universal form
;; of learning, formalized as Bayes-optimal filtering of a noisy sensory
;; signal against a smooth latent state.
;;
;; Architectural argument (same lever as rate_estimation.cljs and
;; crp_operant.cljs):
;;
;;   The habituation filter IS a generative function. One `gen` kernel —
;;   composed with the `Scan` combinator — represents the full habituation
;;   trajectory of an experimental protocol:
;;
;;     - The supplied stimulus at each cycle is the INPUT (z, x).
;;     - The agent's response is a TRACE SITE :y (Bernoulli).
;;     - The stimulus history (Z, X) + active mask is the SCAN CARRY:
;;       threaded as a value from step to step.
;;     - Per-step (x̂_t, σ_t, p-resp_t) are the per-step OUTPUTS.
;;
;;   This gives Gershman's habituation model full GFI semantics:
;;     - p/simulate runs the model forward, sampling binary responses.
;;     - p/generate conditions on observed responses (real animal data).
;;     - p/regenerate / p/assess fall out automatically.
;;     - dyn/vgenerate enables parallel-particle inference.
;;     - comb/unfold-extend gives streaming live-agent operation.
;;
;; Math (per cycle): closed-form GP regression at the just-presented z_t,
;; using all (z_τ, x_τ) for τ ≤ t. Posterior mean x̂_t and std σ_t (Eqs 6-7).
;; Response y_t = Φ((x̂_t - ψ)/σ_t) (Eq 5).
;;
;; Speed strategy (see SPEC_HABITUATION.md §6):
;;   1. Fixed-shape pre-allocated history (no dynamic Cholesky).
;;   2. Identity-substitution masking trick: inactive rows/cols form an
;;      independent identity block, so a single Cholesky-on-[T_max,T_max]
;;      gives the correct active-block posterior plus a decoupled
;;      no-op inactive block.
;;   3. All linalg (`cholesky`, `solve-triangular`) dispatches to Metal
;;      via mlx-node — already GPU-batched.
;;
;; Compare to the original notebook (`dev/habituation/habituation.ipynb`):
;;   - sklearn `GaussianProcessRegressor` instantiated per call inside
;;     nested Python `for` loops
;;     → one Scan call per protocol; carry threaded as a value.
;;   - `gp.fit(t, x); gp.predict(t[-1])` rebuilt-from-scratch per cycle
;;     → identity-substitution mask lets the kernel reuse a single fused
;;       MLX graph across all steps.
;;   - Imperative response normalization in numpy after the loop
;;     → :p-resp is in the per-step output; normalization is one line
;;       on the materialized trace.
;;
;; Seven demos — six reproductions of the paper figures plus one
;; GenMLX-original streaming-agent demo:
;;
;;   Reproductions:
;;     1. Fig 3: frequency × intensity (low/high × low/high) — habituation
;;        vs sensitization regimes.
;;     2. Fig 6: potentiation — 2nd-series learning is faster.
;;     3. Fig 5: spontaneous recovery — response amplitude vs delay.
;;     4. Fig 7: stimulus specificity — generalization gradient.
;;     5. Fig 8: dishabituation — weak/strong/repeat conditions.
;;     6. Fig 4: common test procedure — frequency dependence under
;;        equal test intervals.
;;
;;   GenMLX-original:
;;     7. Streaming live agent via comb/unfold-extend — the trace IS the
;;        agent's complete inspectable memory.
;;
;; Run: bun run --bun nbb examples/habituation.cljs

(ns habituation
  (:require [clojure.string :as str]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================================
;; Helpers
;; ============================================================================

(defn fmt
  ([v]        (fmt v 3))
  ([v digits] (.toFixed (double v) digits)))

(defn pad [s width] (.padStart (str s) width " "))

(def ONE       (mx/scalar 1.0))
(def ZERO      (mx/scalar 0.0))
(def HALF      (mx/scalar 0.5))
(def SQRT-2    (mx/scalar (Math/sqrt 2.0)))
(def EPS       (mx/scalar 1e-10))
(def P-CLAMP-LO (mx/scalar 1e-6))
(def P-CLAMP-HI (mx/scalar (- 1.0 1e-6)))

(defn normal-cdf
  "Φ(z) = 0.5 · (1 + erf(z / √2))."
  [z]
  (mx/multiply HALF (mx/add ONE (mx/erf (mx/divide z SQRT-2)))))

(defn clamp-prob
  "Clamp p to (1e-6, 1 - 1e-6) — avoids Bernoulli check-probability errors
   when σ → 0 (response saturates to 0 or 1)."
  [p]
  (mx/minimum P-CLAMP-HI (mx/maximum P-CLAMP-LO p)))

;; ============================================================================
;; GP kernel + regression
;; ============================================================================

(defn rbf-kernel-matrix
  "Squared exponential kernel matrix K[i,j] = exp(-‖Z[i] - Z[j]‖²/(2λ²)).
   Z shape [T_max, D]; returns [T_max, T_max]."
  [Z lam]
  (let [Zi   (mx/expand-dims Z 0)                  ; [1, T_max, D]
        Zj   (mx/expand-dims Z 1)                  ; [T_max, 1, D]
        diff (mx/subtract Zi Zj)                   ; [T_max, T_max, D]
        sq   (mx/sum (mx/multiply diff diff) [2])  ; [T_max, T_max]
        lam2 (mx/multiply lam lam)]
    (mx/exp (mx/divide (mx/negative sq)
                       (mx/multiply (mx/scalar 2.0) lam2)))))

(defn cho-solve
  "Solve A x = b given Cholesky L (lower-tri) of A.
   L: [T,T];  b: [T,1] or [T].  Returns same shape as b."
  [L b]
  (let [b1?  (= 1 (mx/ndim b))
        b2   (if b1? (mx/expand-dims b 1) b)
        y    (mx/solve-triangular L b2 false)              ; L y = b
        LT   (mx/transpose L)
        x    (mx/solve-triangular LT y true)               ; L^T x = y
        x'   (if b1? (mx/squeeze x [1]) x)]
    x'))

(defn gp-posterior
  "Compute (x̂, σ) at the t-th row of Z, given active mask.

   Identity-substitution trick (see SPEC §4.2): masks inactive rows/cols
   to the identity block, keeping the matrix shape static [T_max, T_max]
   while correctly conditioning only on active entries.

     Z           [T_max, D]    pre-allocated history
     X           [T_max]       pre-allocated observations
     active-mask [T_max]       0/1 float
     t           scalar int    index of the query row
     T-max       int           constant
     lam, alpha  scalars       kernel hyperparams

   Returns [xhat sigma] (both scalars)."
  [Z X active-mask t T-max lam alpha]
  (let [K        (rbf-kernel-matrix Z lam)
        eye-T    (mx/eye T-max)
        m-col    (mx/expand-dims active-mask 1)   ; [T_max, 1]
        m-row    (mx/expand-dims active-mask 0)   ; [1, T_max]
        m-outer  (mx/multiply m-col m-row)        ; [T_max, T_max]
        K-masked (mx/add (mx/multiply K m-outer)
                         (mx/multiply (mx/subtract ONE m-outer) eye-T))
        A        (mx/add K-masked (mx/multiply alpha eye-T))
        L        (mx/cholesky A)
        ;; k_vec = K[t, :] — the new point's kernel row
        k-vec    (mx/index K t)                   ; [T_max]
        k-m      (mx/multiply k-vec active-mask)
        X-m      (mx/multiply X active-mask)
        beta     (cho-solve L X-m)
        gamma    (cho-solve L k-m)
        xhat     (mx/sum (mx/multiply k-m beta))
        ;; k(z, z) = 1 for the SE kernel with unit signal variance
        sigma2   (mx/maximum EPS
                             (mx/subtract ONE
                                          (mx/sum (mx/multiply k-m gamma))))
        sigma    (mx/sqrt sigma2)]
    [xhat sigma]))

;; ============================================================================
;; The kernel — one habituation cycle as a `gen` function
;; ============================================================================
;;
;; Math (per cycle):
;;   1. Insert (z_new, x_new) at row t of (Z, X); set active-mask[t] = 1.
;;   2. K  = rbf-kernel-matrix(Z, λ).
;;   3. A  = K · m⊗m + (1 - m⊗m) · I + αI       — identity-substituted.
;;   4. L  = cholesky(A).
;;   5. β  = A⁻¹ X_masked;  γ = A⁻¹ k_vec_masked.
;;   6. x̂  = k_vec_masked · β;  σ² = 1 - k_vec_masked · γ.
;;   7. p  = Φ((x̂ - ψ) / σ).
;;   8. :y ~ Bernoulli(p).
;;
;; Carry holds all history needed for the next cycle; per-step output
;; carries (x̂, σ, p, y) for plotting.

(def hab-kernel
  "One habituation cycle.

   carry: {:Z [T_max, D]      pre-allocated history (D = 1+stim-dim)
           :X [T_max]          pre-allocated observations
           :active-mask [T_max]
           :t scalar int       current step index (zero-based)
           :T-max int          constant
           :lam scalar         length scale λ
           :alpha scalar       noise variance α
           :psi scalar}        response threshold ψ

   input: {:z [D] new (time, stim feature) row
           :x scalar observed intensity}

   Trace site: :y — Bernoulli(Φ((x̂_t - ψ)/σ_t))

   Returns [new-carry per-step-output]
   per-step-output: {:xhat :sigma :p-resp :y}"
  (dyn/auto-key
    (gen [carry input]
      (let [Z           (:Z           carry)
            X           (:X           carry)
            active-mask (:active-mask carry)
            t           (:t           carry)
            T-max       (:T-max       carry)
            lam         (:lam         carry)
            alpha       (:alpha       carry)
            psi         (:psi         carry)
            z-new       (:z input)
            x-new       (mx/scalar (double (:x input)))
            ;; (1) Insert at row t via one-hot mask + where
            D           (long (second (mx/shape Z)))
            indices     (mx/arange T-max)
            oh-t-bool   (mx/equal indices t)
            oh-t        (mx/astype oh-t-bool mx/float32)             ; [T_max]
            oh-t-col    (mx/expand-dims oh-t 1)                      ; [T_max, 1]
            z-row       (mx/expand-dims (mx/array z-new) 0)          ; [1, D]
            z-broadcast (mx/broadcast-to z-row [T-max D])
            Z'          (mx/add (mx/multiply oh-t-col z-broadcast)
                                (mx/multiply (mx/subtract ONE oh-t-col) Z))
            X'          (mx/add (mx/multiply oh-t x-new)
                                (mx/multiply (mx/subtract ONE oh-t) X))
            mask'       (mx/maximum active-mask oh-t)
            ;; (2-6) Closed-form GP posterior at the just-inserted row
            [xhat sigma] (gp-posterior Z' X' mask' t T-max lam alpha)
            ;; (7) Response probability
            p-resp      (clamp-prob (normal-cdf (mx/divide (mx/subtract xhat psi) sigma)))
            ;; (8) Trace site
            y           (trace :y (dist/bernoulli p-resp))
            new-carry   {:Z           Z'
                         :X           X'
                         :active-mask mask'
                         :t           (mx/add t (mx/scalar 1 mx/int32))
                         :T-max       T-max
                         :lam         lam
                         :alpha       alpha
                         :psi         psi}
            step-out    {:xhat   xhat
                         :sigma  sigma
                         :p-resp p-resp
                         :y      y}]
        [new-carry step-out]))))

(def hab-scan
  "Scan combinator over the per-cycle kernel — full habituation trajectory
   under the GFI."
  (comb/scan-combinator hab-kernel))

;; ============================================================================
;; Init carry and Unfold variant
;; ============================================================================

(def DEFAULT-T-MAX  50)
(def DEFAULT-LAM    1.0)
(def DEFAULT-ALPHA  0.3)
(def DEFAULT-PSI    0.5)

(defn init-carry
  "Initial carry: empty history with the given (or default) hyperparams.
   D is the dimensionality of z (1 for time-only, 2 for [time stim])."
  ([D] (init-carry D DEFAULT-T-MAX DEFAULT-LAM DEFAULT-ALPHA DEFAULT-PSI))
  ([D T-max lam alpha psi]
   {:Z           (mx/zeros [T-max D])
    :X           (mx/zeros [T-max])
    :active-mask (mx/zeros [T-max])
    :t           (mx/scalar 0 mx/int32)
    :T-max       T-max
    :lam         (mx/scalar lam)
    :alpha       (mx/scalar alpha)
    :psi         (mx/scalar psi)}))

(def hab-unfold-kernel
  "Unfold-signature kernel: (gen [step state inputs-fn]). Same body as
   hab-kernel. The `inputs-fn` extra-arg is a CLJS function mapping step
   index → {:z [..] :x ..} — same pattern as rate_estimation's
   bayesian-unfold-kernel takes a time→stim function."
  (dyn/auto-key
    (gen [step carry inputs-fn]
      (let [Z           (:Z           carry)
            X           (:X           carry)
            active-mask (:active-mask carry)
            t           (:t           carry)
            T-max       (:T-max       carry)
            lam         (:lam         carry)
            alpha       (:alpha       carry)
            psi         (:psi         carry)
            {z-new :z x-num :x} (inputs-fn step)
            x-new       (mx/scalar (double x-num))
            D           (long (second (mx/shape Z)))
            indices     (mx/arange T-max)
            oh-t-bool   (mx/equal indices t)
            oh-t        (mx/astype oh-t-bool mx/float32)
            oh-t-col    (mx/expand-dims oh-t 1)
            z-row       (mx/expand-dims (mx/array z-new) 0)
            z-broadcast (mx/broadcast-to z-row [T-max D])
            Z'          (mx/add (mx/multiply oh-t-col z-broadcast)
                                (mx/multiply (mx/subtract ONE oh-t-col) Z))
            X'          (mx/add (mx/multiply oh-t x-new)
                                (mx/multiply (mx/subtract ONE oh-t) X))
            mask'       (mx/maximum active-mask oh-t)
            [xhat sigma] (gp-posterior Z' X' mask' t T-max lam alpha)
            p-resp      (clamp-prob (normal-cdf (mx/divide (mx/subtract xhat psi) sigma)))
            y           (trace :y (dist/bernoulli p-resp))
            new-carry   {:Z           Z'
                         :X           X'
                         :active-mask mask'
                         :t           (mx/add t (mx/scalar 1 mx/int32))
                         :T-max       T-max
                         :lam         lam
                         :alpha       alpha
                         :psi         psi
                         ;; Embed per-step metrics in the carry so Demo 7 can
                         ;; read them off each step's retval (Unfold kernels
                         ;; return only the new state; no per-step output tuple
                         ;; like Scan provides).
                         :xhat        xhat
                         :sigma       sigma
                         :p-resp      p-resp
                         :y           y}]
        new-carry))))

(def hab-unfold (comb/unfold-combinator hab-unfold-kernel))

;; ============================================================================
;; Protocol helpers — build Scan inputs from experimental schedules
;; ============================================================================

(defn protocol-1d
  "1-D protocol (time-only). Returns a vector of {:z [t] :x intensity} maps.
   ts: vector of time stamps (numbers).
   xs: vector of intensities (numbers, same length as ts)."
  [ts xs]
  (mapv (fn [t x] {:z [t] :x x}) ts xs))

(defn protocol-2d
  "2-D protocol (time + stimulus). Returns a vector of {:z [t s] :x x} maps."
  [ts ss xs]
  (mapv (fn [t s x] {:z [t s] :x x}) ts ss xs))

;; ============================================================================
;; Trace inspection helpers
;; ============================================================================

(defn run-scan
  "Run hab-scan over a protocol, returns vector of per-step :p-resp values
   (materialized to JS numbers)."
  [D inputs]
  (let [carry0 (init-carry D)
        trace  (mx/tidy-run
                (fn []
                  (let [k     (rng/fresh-key)
                        tr    (p/simulate (dyn/with-key hab-scan k)
                                          [carry0 inputs])
                        outs  (:outputs (:retval tr))]
                    (mapv (fn [o]
                            (mx/materialize! (:p-resp o))
                            (mx/item (:p-resp o)))
                          outs)))
                (fn [vs] []))]
    trace))

(defn run-scan-full
  "Run hab-scan, returns vector of per-step maps with :xhat :sigma :p-resp
   (all as JS numbers)."
  [D inputs]
  (mx/tidy-run
   (fn []
     (let [carry0 (init-carry D)
           k      (rng/fresh-key)
           tr     (p/simulate (dyn/with-key hab-scan k)
                              [carry0 inputs])
           outs   (:outputs (:retval tr))]
       (mapv (fn [o]
               (mx/materialize! (:xhat o))
               (mx/materialize! (:sigma o))
               (mx/materialize! (:p-resp o))
               {:xhat   (mx/item (:xhat o))
                :sigma  (mx/item (:sigma o))
                :p-resp (mx/item (:p-resp o))})
             outs)))
   (fn [vs] [])))

(defn normalize
  "Multiply each value by 100 / first-value (paper response normalization)."
  [vs]
  (let [v0 (first vs)]
    (mapv #(* 100.0 (/ % v0)) vs)))

(defn print-bar
  "Compact horizontal bar for a normalized response value (0-200 range)."
  [v]
  (let [n   (min 60 (int (* 0.3 v)))
        bar (apply str (repeat n "█"))]
    bar))

;; ============================================================================
;; Demo 1 — Frequency × intensity (paper Fig 3)
;; ============================================================================

(println "\n============================================================")
(println "Demo 1 — Frequency × intensity (paper Fig 3)")
(println "============================================================")
(println "Two-by-two: (low/high intensity) × (low/high frequency).")
(println "Low intensity below ψ → habituation; high intensity above → sensitization.")
(println "High frequency → tighter posterior → stronger effect either way.")

(def N-REPS 10)

(defn fig3-condition
  "Return normalized-response vector across N-REPS repetitions for given
   intensity and frequency. The notebook re-runs the GP fit each repetition;
   we use a single Scan over all repetitions and read the per-step :p-resp."
  [intensity frequency]
  (let [ts     (mapv #(/ (double %) frequency) (range 1 (inc N-REPS)))
        xs     (vec (repeat N-REPS intensity))
        inputs (protocol-1d ts xs)
        ps     (run-scan 1 inputs)]
    (normalize ps)))

(let [conditions [["Intensity: Low,  Frequency: Low"  0.3 2]
                  ["Intensity: High, Frequency: Low"  0.7 2]
                  ["Intensity: Low,  Frequency: High" 0.3 10]
                  ["Intensity: High, Frequency: High" 0.7 10]]]
  (doseq [[label intensity frequency] conditions]
    (println (str "\n  " label))
    (println "  Rep   Norm-Resp   Bar")
    (let [r (fig3-condition intensity frequency)]
      (doseq [[i v] (map vector (range) r)]
        (println (str "  " (pad (inc i) 3) "   "
                      (pad (fmt v 1) 7) "   "
                      (print-bar v)))))))

;; ============================================================================
;; Demo 2 — Potentiation (paper Fig 6)
;; ============================================================================

(println "\n============================================================")
(println "Demo 2 — Potentiation (paper Fig 6)")
(println "============================================================")
(println "Two consecutive habituation series, separated by a delay.")
(println "2nd series starts roughly at the same level but falls faster:")
(println "an \"inactive memory\" of the 1st series tightens the posterior.")

(let [freq      10
      intensity 0.3
      delay     1.0
      N         10
      ;; 1st series: 9 incremental GP fits, sizes 1..9 — read at each step
      ts1       (mapv #(/ (double %) freq) (range 1 N))   ; 9 times
      xs1       (vec (repeat (count ts1) intensity))
      r1        (normalize (run-scan 1 (protocol-1d ts1 xs1)))
      ;; 2nd series: the first 10 trials are at times 1..10/freq, then
      ;; 9 more at N/freq+delay+1..9/freq. The kernel auto-handles long
      ;; histories — we run one Scan over the full 19-step sequence and
      ;; read p-resp at steps 10..18.
      ts2-pre   (mapv #(/ (double %) freq) (range 1 (inc N)))     ; 10 times
      ts2-post  (mapv #(+ (/ (double N) freq) delay (/ (double %) freq))
                      (range 1 N))                                 ; 9 times
      ts2       (vec (concat ts2-pre ts2-post))
      xs2       (vec (repeat (count ts2) intensity))
      r2-raw    (run-scan 1 (protocol-1d ts2 xs2))
      r2-2nd    (subvec (vec r2-raw) N (count r2-raw))             ; the 2nd-series part
      r2        (normalize r2-2nd)]
  (println "\n  Rep   1st-series   2nd-series")
  (doseq [[i [a b]] (map vector (range) (map vector r1 r2))]
    (println (str "  " (pad (inc i) 3) "   "
                  (pad (fmt a 1) 7) "      "
                  (pad (fmt b 1) 7)))))

;; ============================================================================
;; Demo 3 — Spontaneous recovery (paper Fig 5)
;; ============================================================================

(println "\n============================================================")
(println "Demo 3 — Spontaneous recovery (paper Fig 5)")
(println "============================================================")
(println "Habituate for N reps, then test at variable delay.")
(println "Recovery grows with delay; slower recovery after longer habituation.")

(let [freq      10
      intensity 0.3
      Ns        [15 20]
      delays    (mapv double (range 10 21))                  ; 10..20
      ;; Baseline: response to a single isolated stimulus
      r0        (let [ps (run-scan 1 (protocol-1d [(/ 1.0 freq)] [intensity]))]
                  (first ps))]
  (println (str "  Baseline (1-stim isolated): " (fmt r0 4)))
  (println "  Delay     N=15       N=20")
  (doseq [d delays]
    (let [row (mapv (fn [N]
                      (let [ts     (vec (concat (mapv #(/ (double %) freq) (range 1 (inc N)))
                                                [(/ d freq)]))
                            xs     (vec (repeat (inc N) intensity))
                            ps     (run-scan 1 (protocol-1d ts xs))
                            test-p (last ps)]
                        (* 100.0 (/ test-p r0))))
                    Ns)]
      (println (str "  " (pad (fmt d 1) 5) "    "
                    (pad (fmt (first row)  1) 6) "    "
                    (pad (fmt (second row) 1) 6))))))

;; ============================================================================
;; Demo 4 — Stimulus specificity (paper Fig 7)
;; ============================================================================

(println "\n============================================================")
(println "Demo 4 — Stimulus specificity (paper Fig 7)")
(println "============================================================")
(println "Habituate to s=1 for N reps; test at stimulus distance 0..1.")
(println "Generalization gradient: novel stimuli (large distance) restore response.")

(let [freq        10
      intensity   0.3
      N           10
      stim-tests  (mapv #(+ 1.0 (/ (double %) 9.0)) (range 10))  ; 1.0..2.0
      ;; Baseline: single-stim isolated response (Z 2-D: [t, s])
      r0         (first (run-scan 2 (protocol-2d [(/ 1.0 freq)] [1.0] [intensity])))]
  (println (str "  Baseline: " (fmt r0 4)))
  (println "  Dist     Norm-Resp")
  (doseq [s-test stim-tests]
    (let [ts     (mapv #(/ (double %) freq) (range 1 (+ N 2)))   ; N+1 times
          ss     (vec (concat (repeat N 1.0) [s-test]))
          xs     (vec (repeat (count ts) intensity))
          ps     (run-scan 2 (protocol-2d ts ss xs))
          test-p (last ps)
          norm   (* 100.0 (/ test-p r0))]
      (println (str "  " (pad (fmt (- s-test 1.0) 2) 4) "    "
                    (pad (fmt norm 1) 7))))))

;; ============================================================================
;; Demo 5 — Dishabituation (paper Fig 8)
;; ============================================================================

(println "\n============================================================")
(println "Demo 5 — Dishabituation (paper Fig 8)")
(println "============================================================")
(println "Habituate to s=1 (N reps), then test on s=1 after inserting a")
(println "novel s=2 stimulus. Stronger dishabituation for high-intensity novel.")

(let [freq      10
      intensity 0.3
      N         10
      delay     1.0
      ;; Baseline
      r0        (first (run-scan 2 (protocol-2d [(/ 1.0 freq)] [1.0] [intensity])))
      ;; (a) No dishabituation: 10 reps of s=1, last is the test
      no-dis    (let [ts (mapv #(/ (double %) freq) (range 1 (inc N)))
                      ss (vec (repeat N 1.0))
                      xs (vec (repeat N intensity))
                      ps (run-scan 2 (protocol-2d ts ss xs))]
                  (last ps))
      ;; (b) Weak: insert intensity-strength s=2 stimulus between rep 9 and 10
      weak      (let [ts (vec (concat (mapv #(/ (double %) freq) (range 1 N))     ; 9
                                       [(/ (+ 0.5 (* 0.5 N)) freq)]                ; midpoint
                                       [(/ (double N) freq)]))                     ; 10th
                      ss (vec (concat (repeat (dec N) 1.0) [2.0] [1.0]))
                      xs (vec (repeat (inc N) intensity))
                      ps (run-scan 2 (protocol-2d ts ss xs))]
                  (last ps))
      ;; (c) Strong: same as (b) but novel stimulus has 2× intensity
      strong    (let [ts (vec (concat (mapv #(/ (double %) freq) (range 1 N))
                                       [(/ (+ 0.5 (* 0.5 N)) freq)]
                                       [(/ (double N) freq)]))
                      ss (vec (concat (repeat (dec N) 1.0) [2.0] [1.0]))
                      xs (vec (concat (repeat (dec N) intensity)
                                       [(* 2.0 intensity)]
                                       [intensity]))
                      ps (run-scan 2 (protocol-2d ts ss xs))]
                  (last ps))
      ;; (d) Repeated: dishabituation habituates with repeated novel presentations
      repeat-d  (let [ts1 (vec (concat (mapv #(/ (double %) freq) (range 1 N))
                                        [(/ (+ 0.5 (* 0.5 N)) freq)]
                                        [(/ (double N) freq)]))
                      ts2 (mapv #(+ % (/ delay 1.0)) ts1)
                      ts  (vec (concat ts1 ts2))
                      ss  (vec (concat (repeat (dec N) 1.0) [2.0] [1.0]
                                        (repeat (dec N) 1.0) [2.0] [1.0]))
                      xs  (vec (concat (repeat (dec N) intensity)
                                        [(* 2.0 intensity)]
                                        [intensity]
                                        (repeat (dec N) intensity)
                                        [(* 2.0 intensity)]
                                        [intensity]))
                      ps  (run-scan 2 (protocol-2d ts ss xs))]
                  (last ps))]
  (println (str "  Baseline:  " (fmt r0 4)))
  (println "  Condition    Norm-Resp    Bar")
  (doseq [[label v] [["None"   no-dis]
                     ["Weak"   weak]
                     ["Strong" strong]
                     ["Repeat" repeat-d]]]
    (let [norm (* 100.0 (/ v r0))]
      (println (str "  " (pad label 9) "    "
                    (pad (fmt norm 1) 7) "    "
                    (print-bar norm))))))

;; ============================================================================
;; Demo 6 — Common test procedure (paper Fig 4)
;; ============================================================================

(println "\n============================================================")
(println "Demo 6 — Common test procedure (paper Fig 4)")
(println "============================================================")
(println "Habituate at low (2 Hz) or high (10 Hz) frequency, then test at")
(println "varying inter-stimulus intervals. Low-frequency training shows")
(println "weaker test response than high-frequency at each test interval.")

;; Reproducing this figure requires the notebook's full CUMULATIVE TEST
;; SEQUENCE protocol: for each test trial n at test_freq=ν, the GP sees the
;; training stims PLUS (n+1) test stims at intervals 1/ν, and we measure
;; the response to the LAST test stim. Then average over n for each test_freq.
;;
;; Verified against sklearn: per-cycle numbers match exactly. The paper's
;; effect (HIGH-train > LOW-train at every test interval) emerges from the
;; SHUFFLED, CUMULATIVE-TEST protocol — not from a single test stim. With
;; a single test stim the direction reverses because the dominant effect
;; is then how clustered training is near the test point.

(let [intensity 0.3
      delay     0.1
      ;; Use the notebook's exact protocol scaled to N=10 (training stims
      ;; = test trials = 10). Each test_freq value appears 2 times in the
      ;; test sequence (2 reps × 5 test_freqs = 10). Shuffled.
      N         10
      train-fr  [2 10]
      test-fr-base [2 4 6 8 10]
      ;; Deterministic "shuffle" — fixed permutation for reproducibility.
      ;; The notebook uses np.random.shuffle; we use a hand-chosen order.
      test-frequencies [4 10 2 6 8 10 2 8 6 4]
      run-protocol
                (fn [ts xs]
                  (mx/tidy-run
                    (fn []
                      (let [carry0 (init-carry 1)
                            inputs (protocol-1d ts xs)
                            k      (rng/fresh-key)
                            tr     (p/simulate (dyn/with-key hab-scan k)
                                               [carry0 inputs])
                            outs   (:outputs (:retval tr))
                            last-p (:p-resp (last outs))]
                        (mx/materialize! last-p)
                        (mx/item last-p)))
                    (fn [v] [])))
      ;; response[n, i] = response at the (n+1)-th test stim under train_freq[i]
      ;; and test_freq = test-frequencies[n].
      response (mapv (fn [train-f]
                       (mapv (fn [n]
                               (let [test-f   (nth test-frequencies n)
                                     train-ts (mapv #(/ (double %) train-f)
                                                    (range 1 (inc N)))
                                     t-end    (last train-ts)
                                     test-ts  (mapv #(+ t-end delay
                                                        (/ (double %) test-f))
                                                    (range 1 (+ n 2)))
                                     ts       (vec (concat train-ts test-ts))
                                     xs       (vec (repeat (count ts) intensity))]
                                 (run-protocol ts xs)))
                             (range N)))
                     train-fr)
      ;; Notebook normalization: response[0, :] is the FIRST trial response
      ;; under each train condition (with test_frequencies[0]=4 in our case).
      response-norm (mapv (fn [vs] (mapv #(* 100.0 (/ % (first vs))) vs))
                          response)
      ;; Average by unique test_freq
      avg-by-tf (mapv (fn [tf]
                        (let [indices (vec (keep-indexed
                                             #(when (= %2 tf) %1) test-frequencies))]
                          (mapv (fn [norms]
                                  (let [vs (mapv #(nth norms %) indices)]
                                    (/ (reduce + vs) (count vs))))
                                response-norm)))
                      test-fr-base)]
  (println (str "  N=" N " (training stims = test trials, "
                "matching notebook protocol)"))
  (println (str "  Shuffle order (test_frequencies): " test-frequencies))
  (println (str "  Normalization base: response at trial 0, test_freq="
                (first test-frequencies)))
  (println "  Test-Int   Low-freq train   High-freq train   (high - low)")
  (doseq [[k tf] (map vector (range) test-fr-base)]
    (let [test-int (/ 1.0 (double tf))
          low  (get-in avg-by-tf [k 0])
          high (get-in avg-by-tf [k 1])
          delta (- high low)]
      (println (str "  " (pad (fmt test-int 2) 6) "       "
                    (pad (fmt low  1) 6) "         "
                    (pad (fmt high 1) 6) "         "
                    (pad (fmt delta 1) 5))))))

;; ============================================================================
;; Demo 7 — Streaming live agent via comb/unfold-extend (GenMLX-original)
;; ============================================================================

(println "\n============================================================")
(println "Demo 7 — Streaming live agent via comb/unfold-extend")
(println "============================================================")
(println "The trace IS the agent's complete inspectable memory. Each call")
(println "to unfold-extend appends one cycle and accumulates trace :score.")

(let [freq       10
      intensity  0.3
      N          15
      milestones #{1 2 3 5 10 15}
      ;; Pre-build the stream of stimuli — in a real-time agent loop this
      ;; would come from sensors (or be appended-to by a closure with an
      ;; atom); for the demo we know the schedule in advance.
      inputs     (mapv (fn [step]
                         {:z [(/ (double step) freq)] :x intensity})
                       (range 1 (inc N)))
      inputs-fn  (fn [step-idx] (nth inputs step-idx))
      empty-tr   (comb/unfold-empty-trace hab-unfold (init-carry 1) inputs-fn)
      snaps-with-metrics
        (mx/tidy-run
          (fn []
            (loop [step 0
                   tr   empty-tr
                   snaps []]
              (if (>= step N)
                snaps
                (let [k           (rng/fresh-key)
                      {tr' :trace} (comb/unfold-extend tr cm/EMPTY k)
                      last-state  (last (:retval tr'))
                      _           (mx/materialize! (:xhat last-state)
                                                   (:sigma last-state)
                                                   (:p-resp last-state)
                                                   (:score tr'))
                      step-no     (inc step)
                      snap        (if (milestones step-no)
                                    (conj snaps
                                          {:step   step-no
                                           :xhat   (mx/item (:xhat last-state))
                                           :sigma  (mx/item (:sigma last-state))
                                           :p-resp (mx/item (:p-resp last-state))
                                           :score  (mx/item (:score tr'))})
                                    snaps)]
                  (recur step-no tr' snap)))))
          (fn [snaps] []))]
  (println "\n  Step   x̂        σ        p-resp    cum :score")
  (println "  ────   ─────    ─────    ──────    ──────────")
  (doseq [s snaps-with-metrics]
    (println (str "  " (pad (:step s) 4) "   "
                  (pad (fmt (:xhat s)) 5) "    "
                  (pad (fmt (:sigma s)) 5) "    "
                  (pad (fmt (:p-resp s)) 5) "    "
                  (pad (fmt (:score s) 2) 8)))))

;; ============================================================================
;; Summary
;; ============================================================================

(println "\n============================================================")
(println "Summary")
(println "============================================================")
(println (str "  Kernel: " "hab-kernel (1 gen function, ~30 lines body)"))
(println "  Trace site: :y (Bernoulli with closed-form Φ((x̂-ψ)/σ))")
(println "  Inference: closed-form GP regression per cycle (Cholesky on GPU)")
(println "  Architecture: kernel + Scan / Unfold combinator, GFI-complete")
(println "")
(println "  Companion notes: dev/docs/EXAMPLE_HABITUATION_NOTES.md")
(println "  Spec:            dev/docs/SPEC_HABITUATION.md")
(println "  Paper:           dev/habituation/Gershman 2024 - Habituation.pdf")
