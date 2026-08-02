(ns genmlx.dist
  "MLX-native probability distributions for GenMLX.
   All distributions use a single Distribution record with open multimethods.
   Extensible from any namespace via defdist or manual defmethod.

   All log-probs return MLX scalars (stay on GPU for autograd).
   Reparameterized sampling enables gradient flow."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.mlx.constants :refer [LOG-2PI ZERO ONE TWO HALF
                                          NEG-INF LOG-2PI-HALF MLX-PI LOG-PI
                                          SQRT-TWO TWO-PI LOG-2]])
  (:require-macros [genmlx.dist.macros :refer [defdist]]))

;; ---------------------------------------------------------------------------
;; Constants
;; ---------------------------------------------------------------------------

;; Scalar/log constants live in genmlx.mlx.constants (centralized, cached).
(def ^:private BERNOULLI-SUPPORT [ZERO ONE])

;; ---------------------------------------------------------------------------
;; Log-gamma: native MLX kernel (single Metal dispatch per call)
;; ---------------------------------------------------------------------------

;; JS-side log-gamma for scalar computations (e.g. Wishart log-prob).
(defn- log-gamma [x]
  (if (<= x 0)
    js/Infinity
    (let [x (dec x)
          t (+ x 5.5)
          s (reduce +
                    1.000000000190015
                    (map-indexed (fn [i ci] (/ ci (+ x i 1)))
                                 [76.18009172947146 -86.50532032941677
                                  24.01409824083091 -1.231739572450155
                                  1.208650973866179e-3 -5.395239384953e-6]))]
      (+ (* 0.5 (js/Math.log (* 2 js/Math.PI)))
         (* (+ x 0.5) (js/Math.log t))
         (- t)
         (js/Math.log s)))))

;; ---------------------------------------------------------------------------
;; Laplace inverse CDF helper
;; ---------------------------------------------------------------------------

(defn- laplace-icdf
  "Inverse CDF for Laplace: loc - scale * sign(u) * log(1 - 2|u|)."
  [loc scale u]
  (->> (mx/abs u)
       (mx/multiply TWO)
       (mx/subtract ONE)
       mx/log
       (mx/multiply (mx/sign u))
       (mx/multiply scale)
       (mx/subtract loc)))

;; ---------------------------------------------------------------------------
;; Log binomial-coefficient helper
;; ---------------------------------------------------------------------------

(defn- log-choose
  "Log binomial coefficient log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1).
   n and k are MLX arrays (stays on GPU for autograd)."
  [n k]
  (mx/subtract (mx/lgamma (mx/add n ONE))
               (mx/add (mx/lgamma (mx/add k ONE))
                       (mx/lgamma (mx/add (mx/subtract n k) ONE)))))

;; ---------------------------------------------------------------------------
;; Public API wrappers (backward compatible)
;; ---------------------------------------------------------------------------

(defn sample
  "Sample from a distribution."
  ([d] (dc/dist-sample d nil))
  ([d key] (dc/dist-sample d key)))

(defn log-prob
  "Compute differentiable log-probability."
  [d value]
  (dc/dist-log-prob d value))

(defn sample-reparam
  "Reparameterized sample for gradient flow."
  [d key]
  (dc/dist-reparam d key))

(defn support
  "Return the support as a sequence of values."
  [d]
  (dc/dist-support d))

;; ---------------------------------------------------------------------------
;; Parameter validation helpers
;;
;; These validators only check JS number parameters. MLX array params
;; intentionally bypass all validation — the (number? v) guard is deliberate.
;;
;; Why: During vectorized/batched inference, distribution parameters are
;; [N]-shaped MLX arrays. Validating them would require mx/eval! to extract
;; scalar values, which would:
;;   (a) force GPU evaluation, breaking the lazy computation graph,
;;   (b) fail outright for [N]-shaped arrays (no single scalar to check),
;;   (c) add overhead in the inference hot path.
;;
;; The Rust NAPI boundary (Either<&MxArray, f64>) handles type coercion
;; transparently — MLX ops accept both array and number arguments without
;; conversion on the CLJS side.
;;
;; Trade-off: users passing scalar MLX arrays with invalid values (e.g.,
;; mx/scalar -1 for a scale parameter) will get NaN or incorrect results
;; silently. This is acceptable because the primary use of MLX array params
;; is in compiled/vectorized paths where parameters are computed values,
;; not user literals.
;; ---------------------------------------------------------------------------

(defn- check-positive [dist-name param-name v]
  (when (and (number? v) (<= v 0))
    (throw (ex-info (str dist-name ": " param-name " must be positive, got " v)
                    {:distribution dist-name :parameter param-name :value v}))))

(defn- check-all-positive
  "Positivity guard for a VECTOR-valued parameter (Dirichlet's alpha).
   Only inspects a raw CLJS sequential of JS numbers — exactly the same
   restriction as check-positive: an MLX-array parameter is left alone, since
   reading it would force a GPU eval in the constructor (see the note above)."
  [dist-name param-name v]
  (when (and (sequential? v) (seq v) (every? number? v))
    (when-some [bad (first (remove pos? v))]
      (throw (ex-info (str dist-name ": every " param-name
                           " component must be positive, got " bad
                           " in " (pr-str (vec v)))
                      {:distribution dist-name :parameter param-name :value bad})))))

(defn- check-less-than [dist-name lo-name lo hi-name hi]
  (when (and (number? lo) (number? hi) (>= lo hi))
    (throw (ex-info (str dist-name ": " lo-name " must be less than " hi-name
                         ", got " lo-name "=" lo " " hi-name "=" hi)
                    {:distribution dist-name :lo lo :hi hi}))))

(defn- check-probability [dist-name param-name v]
  (when (and (number? v) (or (< v 0) (> v 1)))
    (throw (ex-info (str dist-name ": " param-name " must be in [0,1], got " v)
                    {:distribution dist-name :parameter param-name :value v}))))

(defn- check-open-probability [dist-name param-name v]
  (when (and (number? v) (or (<= v 0) (>= v 1)))
    (throw (ex-info (str dist-name ": " param-name " must be in (0,1), got " v)
                    {:distribution dist-name :parameter param-name :value v}))))

;; ---------------------------------------------------------------------------
;; Shared numeric helpers (genmlx-4pnb)
;; ---------------------------------------------------------------------------

(defn- normal-log-density
  "Element-wise Gaussian log-density core:
   -(0.5*log(2π) + log(sigma) + 0.5*((v-mu)/sigma)²).
   The exact op sequence shared by gaussian, broadcasted-normal, gaussian-vec,
   wrapped-normal and iid-gaussian — callers keep their own sum-axis logic.
   Op order is load-bearing: float32 reassociation is not bit-stable."
  [v mu sigma]
  (let [z (mx/divide (mx/subtract v mu) sigma)]
    (mx/negative
     (mx/add LOG-2PI-HALF
             (mx/log sigma)
             (mx/multiply HALF (mx/square z))))))

(defn- int-support
  "Enumerable integer support: one int32 MLX scalar per integer in [lo, hi)."
  [lo hi]
  (mapv #(mx/scalar % mx/int32) (range lo hi)))

;; ---------------------------------------------------------------------------
;; Gaussian
;; ---------------------------------------------------------------------------

(defdist gaussian
  "Gaussian (normal) distribution with mean mu and std sigma."
  [mu sigma]
  (validate (check-positive "gaussian" "sigma" sigma))
  (sample [key]
          (mx/add mu (mx/multiply sigma (rng/normal key []))))
  (log-prob [v]
            (normal-log-density v mu sigma))
  (reparam [key]
           (mx/add mu (mx/multiply sigma (rng/normal key [])))))

(defmethod dc/dist-sample-n* :gaussian [d key n]
  (let [{:keys [mu sigma]} (:params d)
        key (rng/ensure-key key)]
    (mx/add mu (mx/multiply sigma (rng/normal key [n])))))

(def normal "Alias for `gaussian`." gaussian)

;; ---------------------------------------------------------------------------
;; Uniform
;; ---------------------------------------------------------------------------

(defdist uniform
  "Continuous uniform distribution on [lo, hi]."
  [lo hi]
  (validate (check-less-than "uniform" "lo" lo "hi" hi))
  (sample [key]
          (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key []))))
  (log-prob [v]
            ;; Boundary: log(hi - lo) → -Inf when hi ≈ lo (degenerate interval).
            ;; Guarded by check-less-than for JS number params; MLX array params
            ;; skip validation (see validation helpers comment above).
            (let [in-bounds (mx/multiply (mx/less-equal lo v) (mx/less-equal v hi))
                  log-density (mx/negative (mx/log (mx/subtract hi lo)))]
              (mx/where in-bounds log-density NEG-INF)))
  (reparam [key]
           (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key [])))))

(defmethod dc/dist-sample-n* :uniform [d key n]
  (let [{:keys [lo hi]} (:params d)
        key (rng/ensure-key key)]
    (mx/add lo (mx/multiply (mx/subtract hi lo) (rng/uniform key [n])))))

;; ---------------------------------------------------------------------------
;; Bernoulli
;; ---------------------------------------------------------------------------

(defdist bernoulli
  "Bernoulli distribution with probability p."
  [p]
  (validate (check-probability "bernoulli" "p" p))
  (sample [key]
          (let [u (rng/uniform key [])]
            (mx/where (mx/less u p) ONE ZERO)))
  (log-prob [v]
            ;; xlogy pattern: 0 * log(0) = 0, not NaN (IEEE 754: 0 * -Inf = NaN).
            ;; Boundary: log(p) → -Inf when p=0, log(1-p) → -Inf when p=1.
            ;; This is mathematically correct: P(x=1|p=0) = 0, log(0) = -Inf.
            ;; The mx/where guards ensure 0 * -Inf = 0, not NaN.
            (let [log-p (mx/log p)
                  log-1-p (mx/log (mx/subtract ONE p))
                  ;; Support guard (genmlx-7oen): v must be exactly 0 or 1 —
                  ;; unguarded, lp(0.5) was finite.
                  in-support (mx/maximum (mx/equal v ZERO) (mx/equal v ONE))
                  lp (mx/add (mx/where (mx/equal v ZERO) ZERO (mx/multiply v log-p))
                             (mx/where (mx/equal v ONE) ZERO (mx/multiply (mx/subtract ONE v) log-1-p)))]
              (mx/where in-support lp NEG-INF)))
  (support [] BERNOULLI-SUPPORT))

(defmethod dc/dist-log-prob-support :bernoulli [d]
  ;; [log(1-p), log(p)] — two values, one op each (no xlogy issue)
  (let [{:keys [p]} (:params d)]
    (mx/stack [(mx/log (mx/subtract ONE p)) (mx/log p)])))

(defmethod dc/dist-sample-n* :bernoulli [d key n]
  (let [{:keys [p]} (:params d)
        key (rng/ensure-key key)
        u (rng/uniform key [n])]
    (mx/where (mx/less u p) ONE ZERO)))

(defn flip
  "Alias for bernoulli."
  [prob]
  (bernoulli prob))

;; ---------------------------------------------------------------------------
;; Shared scalar Gamma sampler (used by both Beta and Gamma below)
;; ---------------------------------------------------------------------------

(defn- gamma-sample-scalar
  "One Gamma(shape, rate) draw (JS number) via Marsaglia and Tsang's method, with
   the Ahrens-Dieter boost for shape < 1. `shape-param`/`rate` are MLX scalars.
   Stable at all shapes (~95% first-try acceptance) — the basis for the
   gamma-ratio Beta sampler that replaces Johnk's algorithm (bean genmlx-gcw4)."
  [shape-param rate key]
  ;; Read shape & rate with ONE combined GPU eval instead of two separate
  ;; mx/realize syncs. defdist auto-wraps every param with ensure-array, so these
  ;; constants are MLX arrays that each forced a Metal dispatch per draw; sharing
  ;; the dispatch halves that overhead at tens of thousands of scalar draws.
  ;; Values are unchanged — only the dispatch is shared (genmlx-3rkq).
  (mx/eval! shape-param rate)
  (let [a (mx/item shape-param)
        r (mx/item rate)
        alpha<1? (< a 1.0)
        a' (if alpha<1? (inc a) a)
        d (- a' (/ 1.0 3.0))
        c (/ 1.0 (js/Math.sqrt (* 9.0 d)))
        [key-sample key-boost] (rng/split key)
        raw (loop [k key-sample]
              ;; Three-way split per attempt: k2 is consumed by the uniform
              ;; draw, so splitting it for the next attempt correlated
              ;; successive rejection rounds (genmlx-njaq).
              (let [[k1 k2 k-next] (rng/split-n k 3)
                    x-arr (rng/normal k1 [])
                    u-arr (rng/uniform k2 [])
                    ;; One combined eval for the (x,u) candidate pair rather than
                    ;; two separate realizes — halves the per-attempt GPU syncs.
                    ;; item on an already-eval'd array is a host read, not a
                    ;; second dispatch (genmlx-3rkq).
                    _ (mx/eval! x-arr u-arr)
                    x (mx/item x-arr)
                    u (mx/item u-arr)
                    v (js/Math.pow (+ 1.0 (* c x)) 3)]
                (if (and (> v 0)
                         (< (js/Math.log u) (+ (* 0.5 x x) (* d (+ 1 (- v) (js/Math.log v))))))
                  (/ (* d v) r)
                  (recur k-next))))]
    (if alpha<1?
      (* raw (js/Math.pow (mx/realize (rng/uniform key-boost [])) (/ 1.0 a)))
      raw)))

;; ---------------------------------------------------------------------------
;; Beta
;; ---------------------------------------------------------------------------

(defdist beta-dist
  "Beta distribution with parameters alpha and beta."
  [alpha beta-param]
  (validate (check-positive "beta-dist" "alpha" alpha)
            (check-positive "beta-dist" "beta" beta-param))
  (sample [key]
    ;; Beta(a,b) = G_a / (G_a + G_b), with G ~ Gamma(shape, 1) from the stable
    ;; Marsaglia-Tsang sampler. Replaces Johnk's algorithm, which diverges (and
    ;; SIGTRAPs the native layer) at moderate+ concentration — genmlx-gcw4. This
    ;; matches the batch path (dist-sample-n* :beta-dist), which already does it.
          (let [[k1 k2] (rng/split key)
                ga (gamma-sample-scalar alpha ONE k1)
                gb (gamma-sample-scalar beta-param ONE k2)
                v  (/ ga (+ ga gb))]
            ;; clamp into the open support (0,1) — log-prob is +inf at the edges
            (mx/scalar (min (- 1.0 1e-7) (max 1e-7 v)))))
  (log-prob [v]
            ;; Boundary: log(v) → -Inf at v=0, log(1-v) → -Inf at v=1.
            ;; Mathematically correct for the beta distribution — the density
            ;; is 0 at the boundaries when alpha > 1 or beta > 1 respectively.
            ;; Support guard (genmlx-7oen): v outside [0,1] gave NaN.
            ;; xlogy guard (genmlx-4x5w): beta's mask is INCLUSIVE of 0 and 1,
            ;; so unlike gamma/inv-gamma (strict v>0) the boundary value is not
            ;; discarded by the where — beta(1,3) lp(0) and beta(2,1) lp(1)
            ;; returned NaN from 0*(-Inf), poisoning any weight that touched
            ;; them. The gate is (coefficient=0 AND log arg=0) so that d/dalpha
            ;; = log v is preserved at every interior point.
            (let [log-beta-val (mx/subtract (mx/add (mx/lgamma alpha)
                                                    (mx/lgamma beta-param))
                                            (mx/lgamma (mx/add alpha beta-param)))
                  in-support (mx/multiply (mx/greater-equal v ZERO)
                                          (mx/less-equal v ONE))
                  am1 (mx/subtract alpha ONE)
                  bm1 (mx/subtract beta-param ONE)
                  one-v (mx/subtract ONE v)
                  term-a (mx/where (mx/multiply (mx/equal am1 ZERO)
                                                (mx/equal v ZERO))
                                   ZERO
                                   (mx/multiply am1 (mx/log v)))
                  term-b (mx/where (mx/multiply (mx/equal bm1 ZERO)
                                                (mx/equal one-v ZERO))
                                   ZERO
                                   (mx/multiply bm1 (mx/log one-v)))
                  lp (-> (mx/add term-a term-b)
                         (mx/subtract log-beta-val))]
              (mx/where in-support lp NEG-INF))))

;; ---------------------------------------------------------------------------
;; Gamma
;; ---------------------------------------------------------------------------

(defdist gamma-dist
  "Gamma distribution with shape and rate parameters."
  [shape-param rate]
  (validate (check-positive "gamma-dist" "shape" shape-param)
            (check-positive "gamma-dist" "rate" rate))
  (sample [key]
    ;; Marsaglia and Tsang's method (Ahrens-Dieter boost for shape < 1),
    ;; shared with the gamma-ratio Beta sampler above.
          (mx/scalar (gamma-sample-scalar shape-param rate key)))
  (log-prob [v]
            ;; Support guard (genmlx-7oen): v <= 0 gave NaN (log of negative).
            (let [k shape-param
                  lp (-> (mx/add (mx/multiply (mx/subtract k ONE) (mx/log v))
                                 (mx/multiply k (mx/log rate)))
                         (mx/subtract (mx/multiply rate v))
                         (mx/subtract (mx/lgamma k)))]
              (mx/where (mx/greater v ZERO) lp NEG-INF))))

(defn- gamma-sample-n
  "Vectorized Marsaglia-Tsang: sample [n] gamma values with given shape and rate.
   shape-val: JS number, rate: MLX scalar, key: PRNG key, n: int.
   Exposed for reuse by beta, inv-gamma, and dirichlet batch sampling."
  [shape-val rate key n]
  (let [key (rng/ensure-key key)
        ;; For alpha < 1: Ahrens-Dieter boost — sample Gamma(alpha+1), scale by U^(1/alpha)
        alpha<1? (< shape-val 1.0)
        a (if alpha<1? (inc shape-val) shape-val)
        d (- a (/ 1.0 3.0))
        c (/ 1.0 (js/Math.sqrt (* 9.0 d)))
        d-arr (mx/scalar d)
        c-arr (mx/scalar c)
        max-iter 20]
    (loop [iter 0
           result (mx/zeros [n])
           done (mx/zeros [n]) ;; float 0.0/1.0 mask
           k key]
      (if (>= iter max-iter)
        ;; Scale by rate (and Ahrens-Dieter if alpha < 1)
        (let [samples (mx/divide result rate)]
          (if alpha<1?
            (let [[ku _] (rng/split k)
                  u (rng/uniform ku [n])]
              (mx/multiply samples (mx/power u (mx/scalar (/ 1.0 shape-val)))))
            samples))
        (let [[k1 k2 k3] (rng/split-n k 3)
              x (rng/normal k1 [n])
              u (rng/uniform k2 [n])
              ;; v = (1 + c*x)^3
              cx1 (mx/add ONE (mx/multiply c-arr x))
              v (mx/power cx1 (mx/scalar 3.0))
              ;; Accept where: v > 0 AND log(u) < 0.5*x^2 + d*(1 - v + log(v))
              v-pos (mx/greater v ZERO)
              safe-v (mx/maximum v (mx/scalar 1e-30))
              log-accept (mx/add (mx/multiply HALF (mx/square x))
                                 (mx/multiply d-arr
                                              (mx/add (mx/subtract ONE safe-v)
                                                      (mx/log safe-v))))
              accepted (mx/multiply v-pos (mx/less (mx/log u) log-accept))
              ;; Only fill not-yet-done slots
              not-done (mx/equal done ZERO)
              newly-done (mx/multiply accepted not-done)
              new-vals (mx/multiply d-arr safe-v)
              result (mx/where newly-done new-vals result)
              done (mx/where newly-done ONE done)]
          ;; Cut lazy graph chains — prevents 20 iterations of intermediates
          ;; from accumulating as a single massive computation graph.
          (mx/eval! result done)
          ;; Periodically sweep dead arrays to release Metal buffers.
          ;; No-op inside tidy scopes (in-tidy? guard in sweep-dead-arrays!).
          (when (zero? (mod iter 5))
            (mx/sweep-dead-arrays!))
          (recur (inc iter) result done k3))))))

(defmethod dc/dist-sample-n* :gamma [d key n]
  (let [{:keys [shape-param rate]} (:params d)
        shape-val (mx/realize shape-param)]
    (gamma-sample-n shape-val rate key n)))

;; Beta batch sampling via two independent gamma samples
(defmethod dc/dist-sample-n* :beta-dist [d key n]
  (let [{:keys [alpha beta-param]} (:params d)
        key (rng/ensure-key key)
        [k1 k2] (rng/split key)
        g1 (gamma-sample-n (mx/realize alpha) ONE k1 n)
        g2 (gamma-sample-n (mx/realize beta-param) ONE k2 n)]
    ;; Clamp into the open support (0,1): for alpha/beta < 1 a float32 sample can
    ;; round to exactly 0.0/1.0, where the beta density (and log-prob) is +inf.
    (mx/clip (mx/divide g1 (mx/add g1 g2)) 1e-7 (- 1.0 1e-7))))

(defn gamma-sample-vec
  "Marsaglia-Tsang gamma sampling with a per-ELEMENT shape TENSOR (rate 1). `shape`
   is an MLX array of any shape (e.g. [K] arms or [N,K] particles×arms); returns
   gamma draws of the SAME shape. Generalizes gamma-sample-n (whose shape is a JS
   scalar): the constants a/d/c become element-wise tensors and the alpha<1
   Ahrens-Dieter boost becomes an mx/where mask, so a single tensor mixing shapes
   <1 and >=1 is sampled in one call. Used by beta-sample-vec for the tensor bandit
   (bean genmlx-4ifp / genmlx-tl6p). The scalar gamma-sample-n is left untouched."
  [shape key]
  (let [key (rng/ensure-key key)
        sh  (mx/shape shape)                                 ; output element shape
        a<1 (mx/less shape ONE)                              ; per-element mask
        a   (mx/where a<1 (mx/add shape ONE) shape)          ; boost shape where <1
        d   (mx/subtract a (mx/scalar (/ 1.0 3.0)))
        c   (mx/divide ONE (mx/sqrt (mx/multiply (mx/scalar 9.0) d)))
        max-iter 20]
    (loop [iter 0, result (mx/zeros sh), done (mx/zeros sh), k key]
      (if (>= iter max-iter)
        ;; Ahrens-Dieter boost only where shape<1: multiply by U^(1/shape)
        (let [[ku _] (rng/split k)
              u (rng/uniform ku sh)
              boosted (mx/multiply result (mx/power u (mx/divide ONE shape)))]
          (mx/where a<1 boosted result))
        (let [[k1 k2 k3] (rng/split-n k 3)
              x  (rng/normal k1 sh)
              u  (rng/uniform k2 sh)
              v  (mx/power (mx/add ONE (mx/multiply c x)) (mx/scalar 3.0))   ; (1+c*x)^3
              v-pos  (mx/greater v ZERO)
              safe-v (mx/maximum v (mx/scalar 1e-30))
              ;; log(u) < 0.5*x^2 + d*(1 - v + log v)
              log-accept (mx/add (mx/multiply HALF (mx/square x))
                                 (mx/multiply d (mx/add (mx/subtract ONE safe-v) (mx/log safe-v))))
              accepted   (mx/multiply v-pos (mx/less (mx/log u) log-accept))
              not-done   (mx/equal done ZERO)
              newly-done (mx/multiply accepted not-done)
              result (mx/where newly-done (mx/multiply d safe-v) result)
              done   (mx/where newly-done ONE done)]
          ;; Cut the lazy chain so 20 iterations don't accumulate one giant graph,
          ;; and periodically release dead Metal buffers (matches gamma-sample-n).
          (mx/eval! result done)
          (when (zero? (mod iter 5)) (mx/sweep-dead-arrays!))
          (recur (inc iter) result done k3))))))

(defn beta-sample-vec
  "Per-element Beta sampling: given `alpha` and `beta` MLX tensors of the SAME
   shape, return Beta(alpha,beta) draws of that shape. Beta(a,b)=G1/(G1+G2),
   G1~Gamma(a,1), G2~Gamma(b,1); clipped into (1e-7,1-1e-7) like dist-sample-n*
   :beta-dist. One call draws a whole [K] (or [N,K]) posterior — the tensor bandit
   Thompson draw (bean genmlx-4ifp / genmlx-tl6p)."
  [alpha beta key]
  (let [[k1 k2] (rng/split (rng/ensure-key key))
        g1 (gamma-sample-vec alpha k1)
        g2 (gamma-sample-vec beta k2)]
    (mx/clip (mx/divide g1 (mx/add g1 g2)) 1e-7 (- 1.0 1e-7))))

;; ---------------------------------------------------------------------------
;; Dirichlet support constants (genmlx-4x5w) — shared by the batch sampler here
;; and by the `dirichlet` defdist further down.
;; ---------------------------------------------------------------------------

;; Simplex-membership tolerance for the Dirichlet support mask.
;; DELIBERATELY loose: |sum(v) - 1| must clear both float32 renormalization
;; slack AND the h=1e-3 one-component probe step that gradient_fd_test uses to
;; finite-difference d/dv log Dir(v|alpha) — a central difference legitimately
;; evaluates the density 1e-3 off the simplex. 1e-2 gives 10x margin there while
;; still rejecting the audit repro v=[0.3,0.3,0.3] (sum 0.9), which used to
;; score a finite +1.18.
(def ^:private SIMPLEX-TOL (mx/scalar 1e-2))

;; Floor for a normalized Dirichlet component. Mirrors the (1e-7, 1-1e-7) clamp
;; that :beta-dist applies to its gamma-ratio draw: a float32 Gamma draw with a
;; small alpha rounds to exactly 0.0, and a 0 component is outside the open
;; simplex (log-prob NaN at alpha=1, +Inf at alpha<1). Applied AFTER
;; normalization and followed by a renormalization so the draw stays exactly on
;; the simplex — clamping the ratio the way beta does would break sum(v)=1.
(def ^:private DIRICHLET-FLOOR 1e-7)

;; Dirichlet batch sampling via k independent gamma samples, then normalize
(defmethod dc/dist-sample-n* :dirichlet [d key n]
  (let [{:keys [alpha]} (:params d)
        key (rng/ensure-key key)
        alpha-vals (mx/->clj alpha)
        k (count alpha-vals)
        ks (rng/split-n key k)
        ;; Sample k gamma arrays, each [n], then stack to [k n]
        gammas (mx/stack (mapv (fn [a ki] (gamma-sample-n a ONE ki n))
                               alpha-vals ks))
        ;; gammas is [k n], sum along axis 0 -> [n], then transpose and divide
        totals (mx/sum gammas [0])
        ;; Result shape [n k]: transpose [k n] -> [n k]
        p (mx/transpose (mx/divide gammas totals))
        ;; Clamp off the boundary like :beta-dist does (genmlx-4x5w): a float32
        ;; Gamma draw with small alpha rounds to exactly 0.0, and a 0 component
        ;; makes log-prob NaN (alpha=1) or +Inf (alpha<1). Measured before this
        ;; floor: dirichlet([0.05,0.05,0.05]), n=2000 -> min component 0.0.
        ;; The lift is applied to the NORMALIZED value and followed by a
        ;; renormalization, so every row stays exactly on the simplex.
        lifted (mx/maximum p (mx/scalar DIRICHLET-FLOOR))]
    (mx/divide lifted (mx/sum lifted [-1] true))))

;; ---------------------------------------------------------------------------
;; Exponential
;; ---------------------------------------------------------------------------

(defdist exponential
  "Exponential distribution with the given rate."
  [rate]
  (validate (check-positive "exponential" "rate" rate))
  (sample [key]
          (let [u (rng/uniform key [])]
            (mx/divide (mx/negative (mx/log (mx/subtract ONE u))) rate)))
  (log-prob [v]
            (let [log-density (mx/subtract (mx/log rate) (mx/multiply rate v))
                  non-neg (mx/greater-equal v ZERO)]
              (mx/where non-neg log-density NEG-INF)))
  (reparam [key]
           (let [u (rng/uniform key [])]
             (mx/divide (mx/negative (mx/log (mx/subtract ONE u))) rate))))

(defmethod dc/dist-sample-n* :exponential [d key n]
  (let [{:keys [rate]} (:params d)
        key (rng/ensure-key key)
        u (rng/uniform key [n])]
    (mx/divide (mx/negative (mx/log (mx/subtract ONE u))) rate)))

;; ---------------------------------------------------------------------------
;; Categorical
;; ---------------------------------------------------------------------------

(defn- logits->logprobs
  "Log-softmax: normalize logits along the last axis. Shape-preserving."
  [logits]
  (if (> (count (mx/shape logits)) 1)
    (mx/subtract logits (mx/expand-dims (mx/logsumexp logits [-1]) -1))
    (mx/subtract logits (mx/logsumexp logits))))

(defdist categorical
  "Categorical distribution from log-probabilities (logits)."
  [logits]
  (sample [key]
          (rng/categorical key logits))
  (log-prob [v]
            ;; SUPPORT GUARD (genmlx-bey6). The mask is built from the RAW v,
            ;; BEFORE the int32 cast below — the cast is precisely what makes a
            ;; post-hoc floor factor useless, since it has already truncated
            ;; 1.7 to 1 and wrapped -1 to the last index. Measured before this
            ;; guard, on (dist/categorical (mx/array [-1.0 -2.0 -0.5])):
            ;;   lp(7)   = 0        <- an OUT-OF-RANGE index scored the maximum
            ;;                         possible log-probability, outscoring
            ;;                         every legal one
            ;;   lp(-1)  = -0.604   <- silent wrap to the last index
            ;;   lp(1.7) = lp(1)    <- silent int32 truncation
            ;; categorical is on the LLM-token, HMM-transition and
            ;; vectorized-mixture hot paths, so a value that outscores the
            ;; whole support is the worst shape this class can take.
            (let [raw (mx/ensure-array v)
                  n-cat (last (mx/shape logits))
                  in-support (mx/multiply
                              (mx/multiply (mx/greater-equal raw ZERO)
                                           (mx/less raw (mx/scalar n-cat)))
                              (mx/equal raw (mx/floor raw)))
                  v (mx/ensure-array v mx/int32)
                  log-probs (logits->logprobs logits)
                  lp-shape (mx/shape log-probs)
                  nd (count lp-shape)]
        ;; 1-D logits [K]: v is scalar or [N] — gather along the only axis.
        ;; Multi-dim logits [B..., K]:
        ;;  - per-particle index v whose shape matches the batch dims [B...]:
        ;;    diagonal gather via take-along-axis. A plain (take log-probs v -1)
        ;;    returns [B..., B...] (the full cross-product), silently corrupting
        ;;    the [N] score into [N,N] -> NaN ESS for vectorized models with
        ;;    per-particle logits, e.g. HMM transitions (genmlx-ql6a).
        ;;  - scalar / broadcastable v (constrained shared obs): plain gather.
              ;; All THREE shape branches are masked (genmlx-ql6a listed them;
              ;; guarding only the 1-D one would leave the per-particle HMM
              ;; path — the busiest — unguarded).
              (mx/where in-support
                        (cond
                          (= nd 1)
                          (mx/take-idx log-probs (mx/clip v 0 (dec n-cat)))

                          (= (vec (mx/shape v)) (vec (butlast lp-shape)))
                          (mx/squeeze (mx/take-along-axis
                                       log-probs
                                       (mx/expand-dims (mx/clip v 0 (dec n-cat)) (dec nd))
                                       (dec nd))
                                      [(dec nd)])

                          :else
                          (mx/take-idx log-probs (mx/clip v 0 (dec n-cat)) -1))
                        NEG-INF)))
  (support []
           (mx/materialize! logits)
           (let [n (last (mx/shape logits))]
             (int-support 0 n))))

(defmethod dc/dist-log-prob-support :categorical [d]
  ;; log_softmax(logits) — all K log-probs in one op.
  ;; For multi-dim logits [..., K]: transpose to put K first → [K, ...]
  (let [{:keys [logits]} (:params d)
        log-probs (logits->logprobs logits)]
    (if (> (count (mx/shape logits)) 1)
      (let [nd (count (mx/shape log-probs))
            perm (into [(dec nd)] (range (dec nd)))]
        (mx/transpose log-probs perm))
      log-probs)))

(defmethod dc/dist-sample-n* :categorical [d key n]
  (let [{:keys [logits]} (:params d)
        key (rng/ensure-key key)
        k (last (mx/shape logits))
        ;; Gumbel-max trick: argmax(logits + Gumbel_noise) ~ Categorical(softmax(logits))
        ;; Gumbel noise = -log(-log(U)), U ~ Uniform(0,1)
        ;; U is sampled on [0,1): clamp away from 0 or u=0 gives -Inf noise and
        ;; makes that category unselectable for the draw (genmlx-31t9).
        u (mx/maximum (rng/uniform key [n k]) (mx/scalar 1e-12))
        gumbel (mx/negative (mx/log (mx/negative (mx/log u))))
        ;; Broadcast logits [K] + gumbel [N,K] → [N,K], then argmax over axis 1
        perturbed (mx/add logits gumbel)]
    (mx/argmax perturbed 1)))

;; Gumbel-softmax (concrete) reparameterization for categorical (genmlx-0nyj).
;; Enables low-variance reparameterized gradients through a discrete choice via
;; ADEV instead of REINFORCE. CONTRACT: returns a [K] STRAIGHT-THROUGH ONE-HOT
;; vector — forward value is the exact one-hot of the Gumbel-argmax (so the trace
;; choice stays a legitimate discrete one-hot), backward gradient is the smooth
;; Gumbel-softmax Jacobian via stop_gradient(hard − soft) + soft. This is the
;; ADEV/gradient path only; ordinary categorical sample/log-prob/assess/generate
;; are unchanged (they keep integer indices + exact log-softmax). Temperature is
;; read from (:reparam-tau params) (default 0.5); smaller = lower bias, higher
;; variance. Assumes 1-D logits [K] (the sequential ADEV path); batched/vadev
;; categorical reparam needs a dist-reparam-n and is a follow-up.
(defmethod dc/dist-reparam :categorical [d key]
  (let [{:keys [logits reparam-tau]} (:params d)
        tau (mx/scalar (double (or reparam-tau 0.5)))
        k (last (mx/shape logits))
        g (rng/gumbel (rng/ensure-key key) [k])
        perturbed (mx/add (logits->logprobs logits) g)
        soft (mx/softmax (mx/divide perturbed tau) -1)
        ;; straight-through: exact one-hot forward, soft Jacobian backward.
        ;; (Ties are measure-zero under continuous Gumbel noise.)
        hard (mx/astype (mx/equal soft (mx/amax soft)) mx/float32)]
    (mx/add (mx/stop-gradient (mx/subtract hard soft)) soft)))

;; Differentiable score of a relaxed categorical: <one-hot, log_softmax(logits)>.
;; Equals log_softmax(logits)[argmax] in the forward pass, but is differentiable
;; w.r.t. BOTH the straight-through value and the logits — unlike dist-log-prob
;; :categorical, which int-casts its argument and index-gathers (corrupting a
;; one-hot). Kept off the ordinary scoring path (genmlx-0nyj).
(defmethod dc/dist-reparam-log-prob :categorical [d value]
  (let [{:keys [logits]} (:params d)]
    (mx/sum (mx/multiply value (logits->logprobs logits)))))

(defn categorical-weights
  "Categorical distribution from unnormalized weights (not logits).
   Handles zero weights safely — no log(0), no NaN gradients.
   Use this instead of (categorical (mx/log weights)) when weights may
   contain zeros and gradients are needed (ADEV, gradient descent).

   Zero weights get a large negative logit (-100) with zero gradient.
   Positive weights get log(weights) with correct gradients."
  [weights]
  (let [weights (mx/astype weights mx/float32)
        positive (mx/greater weights (mx/scalar 0))
        ;; Safe log: only compute log for positive weights
        safe-log (mx/log (mx/maximum weights (mx/scalar 1e-10)))
        ;; Zero weights → -100 (constant, zero gradient)
        logits (mx/where positive safe-log (mx/scalar -100.0))]
    (categorical logits)))

(defn weighted
  "Categorical distribution from a Clojure vector of weights.
   Each weight may be a number or an MLX array. Handles scalar
   promotion, stacking, and log transform automatically.

   Use this to write model code at the distribution layer (Layer 4)
   without dropping to raw MLX operations (Layer 0):

     (trace :eval (dist/weighted [1.0 1.5 w]))  ; clean
     ; instead of
     (trace :eval (dist/categorical              ; noisy
       (mx/log (mx/stack #js [(mx/scalar 1.0) (mx/scalar 1.5) w]))))"
  [weights]
  (let [as-mx (mapv #(if (number? %) (mx/scalar %) %) weights)
        stacked (mx/stack as-mx -1)
        logits (mx/log (mx/maximum stacked (mx/scalar 1e-30)))]
    (categorical logits)))

;; ---------------------------------------------------------------------------
;; Poisson
;; ---------------------------------------------------------------------------

(defn- poisson-sample-small*
  "Exact Poisson sampler for SMALL rates: count unit-rate exponential
   inter-arrivals (-log(1-u) ~ Exp(1)) until the cumulative sum exceeds rate.
   The log-space form of Knuth's product loop — the classic form multiplies
   uniforms against l = exp(-rate), which underflows to 0.0 for rate >= ~708
   and silently pins every sample near ~700-745 while log-prob scores the true
   distribution (genmlx-2nec). O(rate) iterations, so large rates go through
   PTRS instead. Returns a JS number count."
  [rate-val key]
  (loop [k 0 s 0.0 rk key]
    (let [[rk1 rk2] (rng/split rk)
          u (mx/realize (rng/uniform rk1 []))
          s (- s (js/Math.log (- 1.0 u)))]
      (if (> s rate-val)
        k
        (recur (inc k) s rk2)))))

(defn- poisson-sample-ptrs*
  "Exact Poisson sampler for rate >= 10: Hörmann's PTRS transformed rejection
   with squeeze (1993) — the same algorithm NumPy uses above rate 10. O(1)
   expected attempts (~94% acceptance), no underflow at any rate. Returns a JS
   number count."
  [rate-val key]
  (let [b (+ 0.931 (* 2.53 (js/Math.sqrt rate-val)))
        a (+ -0.059 (* 0.02483 b))
        inv-alpha (+ 1.1239 (/ 1.1328 (- b 3.4)))
        vr (- 0.9277 (/ 3.6224 (- b 2.0)))
        log-rate (js/Math.log rate-val)]
    (loop [rk key]
      (let [[k1 rk'] (rng/split rk)
            [k2 k3] (rng/split k1)
            u (- (mx/realize (rng/uniform k2 [])) 0.5)
            v (mx/realize (rng/uniform k3 []))
            us (- 0.5 (js/Math.abs u))
            k (js/Math.floor (+ (* (+ (/ (* 2.0 a) us) b) u) rate-val 0.43))]
        (cond
          ;; squeeze: immediate accept for the bulk
          (and (>= us 0.07) (<= v vr))
          k

          ;; obvious rejects (k negative, or us tiny with v above the hat)
          (or (neg? k) (and (< us 0.013) (> v us)))
          (recur rk')

          ;; exact acceptance test against the Poisson pmf
          (<= (js/Math.log (* v (/ inv-alpha (+ (/ a (* us us)) b))))
              (- (* k log-rate) rate-val (log-gamma (+ k 1.0))))
          k

          :else (recur rk'))))))

(defn- poisson-sample*
  "Exact Poisson sampler, correct at every rate (genmlx-2nec): inter-arrival
   counting below 10, PTRS at 10 and above. Returns a JS number count."
  [rate-val key]
  (if (< rate-val 10.0)
    (poisson-sample-small* rate-val key)
    (poisson-sample-ptrs* rate-val key)))

(defdist poisson
  "Poisson distribution with the given rate."
  [rate]
  (validate (check-positive "poisson" "rate" rate))
  (sample [key]
          (mx/scalar (poisson-sample* (mx/realize rate) key)))
  (log-prob [v]
            ;; Support guard (genmlx-7oen): counts are nonnegative integers.
            (let [in-support (mx/multiply (mx/greater-equal v ZERO)
                                          (mx/equal v (mx/floor v)))
                  lp (-> (mx/multiply v (mx/log rate))
                         (mx/subtract rate)
                         (mx/subtract (mx/lgamma (mx/add v ONE))))]
              (mx/where in-support lp NEG-INF))))

;; ---------------------------------------------------------------------------
;; Laplace
;; ---------------------------------------------------------------------------

(defdist laplace
  "Laplace distribution with location and scale."
  [loc scale]
  (validate (check-positive "laplace" "scale" scale))
  (sample [key]
          (laplace-icdf loc scale (mx/subtract (rng/uniform key []) HALF)))
  (log-prob [v]
            ;; Boundary: division by scale produces Inf when scale=0.
            ;; Guarded by check-positive for JS number params; MLX array params
            ;; skip validation (see validation helpers comment above).
            (mx/subtract
             (mx/negative (mx/log (mx/multiply TWO scale)))
             (mx/divide (mx/abs (mx/subtract v loc)) scale)))
  (reparam [key]
           (laplace-icdf loc scale (mx/subtract (rng/uniform key []) HALF))))

(defmethod dc/dist-sample-n* :laplace [d key n]
  (let [{:keys [loc scale]} (:params d)
        key (rng/ensure-key key)]
    (laplace-icdf loc scale (mx/subtract (rng/uniform key [n]) HALF))))

;; ---------------------------------------------------------------------------
;; Student-t
;; ---------------------------------------------------------------------------

(defdist student-t
  "Student-t distribution with df degrees of freedom, location and scale."
  [df loc scale]
  (validate (check-positive "student-t" "df" df)
            (check-positive "student-t" "scale" scale))
  (sample [key]
          ;; chi2(df) = Gamma(df/2, rate 1/2) — valid for ANY df > 0. The old
          ;; sum-of-int(df)-squared-normals sampler silently truncated df
          ;; (df=2.5 sampled t(2) while log-prob scored t(2.5) — sample/score
          ;; mismatch breaking importance weights; df<1 divided by an empty
          ;; chi2 sum = 0 → Inf samples) (genmlx-yeam).
          (let [[k1 k2] (rng/split key)
                df-val (mx/realize df)
                chi2 (gamma-sample-scalar (mx/scalar (/ df-val 2.0))
                                          (mx/scalar 0.5) k1)
                z (mx/realize (rng/normal k2 []))
                t (* z (js/Math.sqrt (/ df-val chi2)))]
            (mx/add loc (mx/multiply scale (mx/scalar t)))))
  (log-prob [v]
            ;; Potential overflow in (1 + z^2/df)^(-(df+1)/2) for extreme z.
            ;; Unlikely in practice: float32 range covers z up to ~1.8e19 before
            ;; z^2 overflows, and the log transform keeps intermediate values finite.
            (let [z (mx/divide (mx/subtract v loc) scale)
                  half-df (mx/multiply HALF df)
                  half-df1 (mx/multiply HALF (mx/add df ONE))
                  log-norm (mx/subtract (mx/lgamma half-df1)
                                        (mx/add (mx/lgamma half-df)
                                                (mx/multiply HALF (mx/log (mx/multiply df MLX-PI)))))]
              (-> log-norm
                  (mx/subtract (mx/log scale))
                  (mx/subtract (mx/multiply half-df1
                                            (mx/log (mx/add ONE
                                                            (mx/divide (mx/square z) df)))))))))

(defmethod dc/dist-sample-n* :student-t [d key n]
  (let [{:keys [df loc scale]} (:params d)
        df-val (mx/realize df)
        [k1 k2] (rng/split (rng/ensure-key key))
        ;; chi2(df) = Gamma(df/2, rate 1/2) — exact for fractional df
        ;; (the old [n, int(df)] normals matrix truncated df; genmlx-yeam)
        chi2 (gamma-sample-n (/ df-val 2.0) (mx/scalar 0.5) k1 n)
        ;; Standard normal -> [n]
        z (rng/normal k2 [n])
        ;; t = z * sqrt(df / chi2)
        t (mx/multiply z (mx/sqrt (mx/divide df chi2)))]
    (mx/add loc (mx/multiply scale t))))

;; ---------------------------------------------------------------------------
;; Log-Normal
;; ---------------------------------------------------------------------------

(defdist log-normal
  "Log-Normal distribution with parameters mu and sigma."
  [mu sigma]
  (validate (check-positive "log-normal" "sigma" sigma))
  (sample [key]
          (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key [])))))
  (log-prob [v]
            ;; Support guard (genmlx-7oen): v <= 0 gave NaN (log of negative).
            (let [log-v (mx/log v)
                  z (mx/divide (mx/subtract log-v mu) sigma)
                  lp (mx/negative
                      (mx/add log-v
                              LOG-2PI-HALF
                              (mx/log sigma)
                              (mx/multiply HALF (mx/square z))))]
              (mx/where (mx/greater v ZERO) lp NEG-INF)))
  (reparam [key]
           (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key []))))))

(defmethod dc/dist-sample-n* :log-normal [d key n]
  (let [{:keys [mu sigma]} (:params d)
        key (rng/ensure-key key)]
    (mx/exp (mx/add mu (mx/multiply sigma (rng/normal key [n]))))))

;; ---------------------------------------------------------------------------
;; Dirichlet
;; ---------------------------------------------------------------------------

(defdist dirichlet
  "Dirichlet distribution with concentration parameters alpha."
  [alpha]
  (validate (check-all-positive "dirichlet" "alpha" alpha))
  (sample [key]
          (let [alpha-vals (mx/->clj alpha)
                k (count alpha-vals)
                ks (rng/split-n key k)
                ;; Call gamma-sample-scalar directly: it already returns a host
                ;; float, so the old (dist-sample → mx/scalar wrap → mx/realize)
                ;; round-tripped a value we already had on the host — one wasted
                ;; GPU eval per component, k per Dirichlet draw (genmlx-3rkq).
                gammas (mapv (fn [a ki] (gamma-sample-scalar (mx/scalar a) ONE ki))
                             alpha-vals ks)
                total (reduce + gammas)
                ;; Lift exact zeros off the boundary, then renormalize so the
                ;; draw is still exactly on the simplex (genmlx-4x5w).
                lifted (mapv #(max DIRICHLET-FLOOR (/ % total)) gammas)
                lifted-total (reduce + lifted)
                normalized (mapv #(/ % lifted-total) lifted)]
            (mx/array normalized)))
  (log-prob [v]
            ;; Reduce only the trailing event (K) axis so batched values broadcast
            ;; per-particle: [K]->[] in scalar mode, [N,K]->[N] in batched mode.
            ;; A bare (mx/sum ...) collapses EVERY axis, silently summing the
            ;; particle axis into one scalar that then broadcasts onto every
            ;; particle's score (genmlx-t5qa).
            ;;
            ;; Support guard (genmlx-4x5w): the open simplex. Unguarded, this
            ;; returned NaN at alpha=1 with a zero component (0*log 0), +Inf at
            ;; alpha<1 with a zero component, NaN on a negative component, and a
            ;; finite score for any off-simplex v. The zero component is
            ;; reachable from our OWN batch sampler, which used to normalize
            ;; float32 gammas without a floor.
            (let [v (mx/ensure-array v)
                  log-beta (mx/subtract (mx/sum (mx/lgamma alpha) [-1])
                                        (mx/lgamma (mx/sum alpha [-1])))
                  ;; xlogy: (alpha_i - 1) * log v_i, with the ONE lane forced to
                  ;; exactly 0 where v_i = 0 so 0*(-Inf) = NaN never enters the
                  ;; graph (the NaN would survive the where-mask's cotangent in
                  ;; reverse mode even though the forward value is discarded).
                  ;; The gate is (alpha=1 AND v=0), not alpha=1 alone: gating on
                  ;; alpha alone would zero d/dalpha = log v at every INTERIOR
                  ;; point of a uniform Dirichlet.
                  am1 (mx/subtract alpha ONE)
                  terms (mx/where (mx/multiply (mx/equal am1 ZERO)
                                               (mx/equal v ZERO))
                                  ZERO
                                  (mx/multiply am1 (mx/log v)))
                  log-terms (mx/sum terms [-1])
                  in-support (mx/multiply
                              (mx/all (mx/greater v ZERO) -1)
                              (mx/less (mx/abs (mx/subtract (mx/sum v [-1]) ONE))
                                       SIMPLEX-TOL))]
              (mx/where in-support (mx/subtract log-terms log-beta) NEG-INF))))

;; ---------------------------------------------------------------------------
;; Delta (point mass)
;; ---------------------------------------------------------------------------

(defdist delta
  "Delta (point mass) distribution at value v.
   log-prob compares by EXACT float equality — correct for the point-mass
   semantics (constrained values pass through bit-identically), but any
   recomputation that perturbs the value scores -Inf."
  [v]
  (sample [_key] v)
  (log-prob [value]
            ;; Reduce the trailing event axes to the JOINT point-mass log-prob:
            ;; 0 iff ALL elements match, else -Inf. A scalar point mass has no
            ;; event axis (ev=0) so the elementwise mask is already the answer
            ;; ([] scalar / [N] batched). A vector/tensor point mass must be
            ;; reduced, else log-prob returns an elementwise [T] mask instead of
            ;; the joint scalar (genmlx-exw9).
            (let [eq (mx/equal v value)
                  ev (count (mx/shape v))
                  joint (reduce (fn [m _] (mx/all m (dec (count (mx/shape m)))))
                                eq (range ev))]
              (mx/where joint ZERO NEG-INF)))
  (support [] [v]))

(defmethod dc/dist-sample-n* :delta [d _key n]
  (let [{:keys [v]} (:params d)
        ;; Batch shape prepends the particle axis: scalar v → [n],
        ;; [d]-shaped v → [n d]. broadcast-to v [n] is a shape error for
        ;; any non-scalar point (genmlx-yeam).
        sh (if (mx/array? v) (mx/shape v) [])]
    (mx/broadcast-to v (into [n] sh))))

;; ---------------------------------------------------------------------------
;; Cauchy
;; ---------------------------------------------------------------------------

(defdist cauchy
  "Cauchy distribution with location and scale."
  [loc scale]
  (validate (check-positive "cauchy" "scale" scale))
  (sample [key]
    ;; Inverse CDF: loc + scale * tan(pi * (u - 0.5))
          (let [u (rng/uniform key [])
                z (mx/subtract u HALF)]
            (mx/add loc (mx/multiply scale
                                     (mx/divide (mx/sin (mx/multiply MLX-PI z))
                                                (mx/cos (mx/multiply MLX-PI z)))))))
  (log-prob [v]
    ;; -log(pi * scale * (1 + ((v - loc) / scale)^2))
            (let [z (mx/divide (mx/subtract v loc) scale)]
              (mx/negative
               (mx/add LOG-PI
                       (mx/log scale)
                       (mx/log (mx/add ONE (mx/square z)))))))
  (reparam [key]
           (let [u (rng/uniform key [])
                 z (mx/subtract u HALF)]
             (mx/add loc (mx/multiply scale
                                      (mx/divide (mx/sin (mx/multiply MLX-PI z))
                                                 (mx/cos (mx/multiply MLX-PI z))))))))

(defmethod dc/dist-sample-n* :cauchy [d key n]
  (let [{:keys [loc scale]} (:params d)
        key (rng/ensure-key key)
        u (rng/uniform key [n])
        z (mx/subtract u HALF)]
    (mx/add loc (mx/multiply scale
                             (mx/divide (mx/sin (mx/multiply MLX-PI z))
                                        (mx/cos (mx/multiply MLX-PI z)))))))

;; ---------------------------------------------------------------------------
;; Inverse Gamma
;; ---------------------------------------------------------------------------

(defdist inv-gamma
  "Inverse-Gamma distribution with shape and scale parameters."
  [shape-param scale-param]
  (validate (check-positive "inv-gamma" "shape" shape-param)
            (check-positive "inv-gamma" "scale" scale-param))
  (sample [key]
    ;; InvGamma(shape, scale) = scale / G where G ~ Gamma(shape, rate=1)
    ;; (genmlx-21kt: the old "gamma(shape, 1/scale), then invert" comment
    ;; mis-described this — the code samples rate=1 and divides scale, not 1/g).
          (let [g (dc/dist-sample (gamma-dist shape-param ONE) key)]
            (mx/divide scale-param g)))
  (log-prob [v]
    ;; log p(v) = shape*log(scale) - log-gamma(shape) - (shape+1)*log(v) - scale/v
    ;; Support guard (genmlx-7oen): v <= 0 gave NaN (log of negative).
            (let [lp (-> (mx/multiply shape-param (mx/log scale-param))
                         (mx/subtract (mx/lgamma shape-param))
                         (mx/subtract (mx/multiply (mx/add shape-param ONE) (mx/log v)))
                         (mx/subtract (mx/divide scale-param v)))]
              (mx/where (mx/greater v ZERO) lp NEG-INF))))

(defmethod dc/dist-sample-n* :inv-gamma [d key n]
  (let [{:keys [shape-param scale-param]} (:params d)
        g (gamma-sample-n (mx/realize shape-param) ONE key n)]
    (mx/divide scale-param g)))

;; ---------------------------------------------------------------------------
;; Geometric
;; ---------------------------------------------------------------------------

(defdist geometric
  "Geometric distribution: number of failures before first success, p in (0,1]."
  [p]
  (validate
   ;; p=1 is legal (k=0 w.p. 1; the log-prob xlogy guard handles it), but
   ;; p=0 sampled floor(log u / log 1) = -Inf garbage (genmlx-yeam).
   (check-probability "geometric" "p" p)
   (when (and (number? p) (zero? p))
     (throw (ex-info "geometric: p must be in (0,1], got 0"
                     {:distribution "geometric" :parameter "p" :value p}))))
  (sample [key]
    ;; Inverse CDF: floor(log(1-u) / log(1-p)). Use (1-u), not u: rng/uniform
    ;; returns [0,1) which INCLUDES 0, and log(0) = -Inf -> +Inf, a sample
    ;; outside the support; 1-u in (0,1] keeps the log finite (genmlx-lgun).
          (let [u (rng/uniform key [])
                log-1mu (mx/log (mx/subtract ONE u))
                log-1mp (mx/log (mx/subtract ONE p))]
            (mx/floor (mx/divide log-1mu log-1mp))))
  (log-prob [v]
    ;; log p(k) = k * log(1-p) + log(p)
    ;; xlogy guard: at p=1 (legal — success on the first trial, k=0 w.p. 1)
    ;; the k*log(1-p) term is 0*-Inf = NaN without it (genmlx-yeam).
    ;; Support guard (genmlx-7oen): k must be a nonnegative integer — an
    ;; unguarded lp(-1) scored HIGHER than lp(0).
            (let [in-support (mx/multiply (mx/greater-equal v ZERO)
                                          (mx/equal v (mx/floor v)))
                  lp (mx/add (mx/where (mx/equal v ZERO)
                                       ZERO
                                       (mx/multiply v (mx/log (mx/subtract ONE p))))
                             (mx/log p))]
              (mx/where in-support lp NEG-INF)))
  (support []
    ;; Dynamic support up to 0.999 quantile (capped at 10000)
           (let [p-val (mx/realize p)
                 max-k (min 10000 (int (js/Math.ceil (/ (js/Math.log 0.001)
                                                        (js/Math.log (- 1.0 p-val))))))]
             (int-support 0 (inc max-k)))))

(defmethod dc/dist-sample-n* :geometric [d key n]
  (let [{:keys [p]} (:params d)
        key (rng/ensure-key key)
        u (rng/uniform key [n])
        ;; (1-u), not u: avoid log(0)=-Inf -> +Inf at the u=0 boundary (genmlx-lgun)
        log-1mu (mx/log (mx/subtract ONE u))
        log-1mp (mx/log (mx/subtract ONE p))]
    (mx/floor (mx/divide log-1mu log-1mp))))

;; ---------------------------------------------------------------------------
;; Negative Binomial
;; ---------------------------------------------------------------------------

(defdist neg-binomial
  "Negative binomial (Polya) distribution.
   r: number of successes, p: probability of success, p in (0,1)."
  [r p]
  (validate
   (check-positive "neg-binomial" "r" r)
   ;; Strictly open: the gamma-Poisson sampler's rate p/(1-p) divides by
   ;; zero at p=1, and p=0 never terminates — the closed-interval check
   ;; let both through to garbage (genmlx-yeam).
   (check-open-probability "neg-binomial" "p" p))
  (sample [key]
    ;; Gamma-Poisson mixture: lambda ~ Gamma(r, p/(1-p)), then x ~ Poisson(lambda).
    ;; The Poisson stage shares poisson-sample* — the inline Knuth product loop
    ;; inherited the exp(-lambda) underflow for large gamma draws (genmlx-2nec).
          (let [[k1 k2] (rng/split key)
                rate (mx/divide p (mx/subtract ONE p))
                g (dc/dist-sample (gamma-dist r rate) k1)
                g-val (mx/realize g)]
            (mx/scalar (poisson-sample* g-val k2))))
  (log-prob [v]
    ;; log C(v + r - 1, v) + r*log(p) + v*log(1-p)
    ;; Support guard (genmlx-7oen): counts are nonnegative integers.
            (let [log-coeff (log-choose (mx/subtract (mx/add v r) ONE) v)
                  in-support (mx/multiply (mx/greater-equal v ZERO)
                                          (mx/equal v (mx/floor v)))
                  lp (-> log-coeff
                         (mx/add (mx/multiply r (mx/log p)))
                         (mx/add (mx/multiply v (mx/log (mx/subtract ONE p)))))]
              (mx/where in-support lp NEG-INF))))

;; ---------------------------------------------------------------------------
;; Binomial
;; ---------------------------------------------------------------------------

(defdist binomial
  "Binomial distribution: n trials with success probability p."
  [n-trials p]
  (validate (check-probability "binomial" "p" p))
  (sample [key]
          (let [nt (int (mx/realize n-trials))
                ks (rng/split-n key nt)
                p-val (mx/realize p)
                successes (count (filter #(< (mx/realize (rng/uniform % [])) p-val) ks))]
            (mx/scalar successes)))
  (log-prob [v]
    ;; log C(n, k) + k*log(p) + (n-k)*log(1-p)
    ;; xlogy guards (cf. bernoulli): at p=0 with v=0 the first term is
    ;; 0*log(0) = 0*-Inf = NaN without the guard; at p=1 with v=n the
    ;; second term likewise (genmlx-yeam).
            ;; Support guard (genmlx-7oen): k must be an integer in [0, n].
            (let [log-coeff (log-choose n-trials v)
                  n-minus-v (mx/subtract n-trials v)
                  in-support (-> (mx/greater-equal v ZERO)
                                 (mx/multiply (mx/less-equal v n-trials))
                                 (mx/multiply (mx/equal v (mx/floor v))))
                  lp (-> log-coeff
                         (mx/add (mx/where (mx/equal v ZERO)
                                           ZERO
                                           (mx/multiply v (mx/log p))))
                         (mx/add (mx/where (mx/equal n-minus-v ZERO)
                                           ZERO
                                           (mx/multiply n-minus-v
                                                        (mx/log (mx/subtract ONE p))))))]
              (mx/where in-support lp NEG-INF)))
  (support []
           (let [nt (int (mx/realize n-trials))]
             (int-support 0 (inc nt)))))

(defmethod dc/dist-sample-n* :binomial [d key n]
  (let [{:keys [n-trials p]} (:params d)
        nt (int (mx/realize n-trials))
        key (rng/ensure-key key)
        ;; [n, nt] uniform draws, compare with p, sum successes
        u (rng/uniform key [n nt])
        successes (mx/sum (mx/where (mx/less u p) ONE ZERO) [1])]
    successes))

;; ---------------------------------------------------------------------------
;; Discrete Uniform
;; ---------------------------------------------------------------------------

(defdist discrete-uniform
  "Discrete uniform distribution on integers [lo, hi]."
  [lo hi]
  (validate (check-less-than "discrete-uniform" "lo" lo "hi" hi))
  (sample [key]
          (let [lo-val (int (mx/realize lo))
                hi-val (int (mx/realize hi))
                n (inc (- hi-val lo-val))]
            (mx/scalar (+ lo-val (int (* (mx/realize (rng/uniform key [])) n))) mx/int32)))
  (log-prob [v]
            ;; Support guard (genmlx-4x5w, completing the genmlx-7oen sweep):
            ;; the support is the INTEGERS in [lo, hi]. Without the floor
            ;; factor every other discrete family carries, lp(3.5) equalled
            ;; lp(3) — an impossible value scored exactly as well as a legal
            ;; one, on a family whose own sampler only ever emits integers.
            (let [lo-val (mx/realize lo)
                  hi-val (mx/realize hi)
                  n (inc (- hi-val lo-val))
                  in-range (-> (mx/greater-equal v lo)
                               (mx/multiply (mx/less-equal v hi))
                               (mx/multiply (mx/equal v (mx/floor v))))]
              (mx/where in-range (mx/scalar (- (js/Math.log n))) NEG-INF)))
  (support []
           (let [lo-val (int (mx/realize lo))
                 hi-val (int (mx/realize hi))]
             (int-support lo-val (inc hi-val)))))

(defmethod dc/dist-sample-n* :discrete-uniform [d key n]
  (let [{:keys [lo hi]} (:params d)
        key (rng/ensure-key key)]
    (rng/randint key (int (mx/realize lo)) (inc (int (mx/realize hi))) [n])))

;; ---------------------------------------------------------------------------
;; Standard-normal tail mass in log space (genmlx-g2iu)
;; ---------------------------------------------------------------------------

(def ^:private ERFC-COEFFS
  "Numerical Recipes' Chebyshev fit for erfc (`erfcc`), lowest-order term last
   for Horner evaluation. Documented fractional error < 1.2e-7 — float32
   epsilon — so in LOG space it is an absolute error of ~1.2e-7 nats at every
   argument, which is why it can be evaluated without ever exponentiating."
  (mapv mx/scalar
        [0.17087277 -0.82215223 1.48851587 -1.13520398 0.27886807
         -0.18628806 0.09678418 0.37409196 1.00002368 -1.26551223]))

(defn- log-q
  "log(1 - Phi(x)), the log upper-tail mass of the standard normal, for x >= 0.

   erfcc is `t * exp(P(t) - u^2)`; we return `log t + P(t) - u^2` and never
   form the exponential. That is the whole point: at x = 7 the tail mass is
   1.3e-12, which `1 - erf(x)` cannot represent in float32 (both terms round
   to 1.0) but which -27.4 nats represents exactly.

   Only defined for x >= 0 — every caller here reflects first."
  [x]
  (let [u (mx/divide x SQRT-TWO)
        t (mx/divide ONE (mx/add ONE (mx/multiply HALF u)))
        poly (reduce (fn [acc c] (mx/add c (mx/multiply t acc)))
                     (first ERFC-COEFFS)
                     (rest ERFC-COEFFS))]
    (mx/subtract (mx/add (mx/log t) (mx/subtract poly (mx/multiply u u)))
                 LOG-2)))

(defn- log-normal-mass
  "log(Phi(b) - Phi(a)) for a <= b, accurate on both tails and across zero.

   The naive `log(Phi(b) - Phi(a))` via `0.5(1 + erf(.))` collapses beyond
   ~5-6 sigma: both terms round to 1.0, the difference underflows, and since
   this value is SUBTRACTED from a log-density the collapse became a large
   positive bonus (genmlx-g2iu). Two moves remove it:

     1. Reflect an entirely-non-positive interval onto the upper tail, using
        Phi(b) - Phi(a) = Phi(-a) - Phi(-b). One code path now serves both
        tails — the old code was wrong on BOTH, symmetrically.
     2. Off the origin, work in log space: Q(a) - Q(b) = exp(log Q(a)) *
        (1 - exp(log Q(b) - log Q(a))), i.e. `log-q a + log(-expm1 d)`. No
        term ever rounds to 1.0, so nothing cancels.

   The erf difference is kept where it is strictly better — while a' <= 0 it
   is cancellation-FREE (the two erf values straddle zero, so the subtraction
   is really an addition), and while b' <= 1 both erf values are small enough
   to carry full relative precision. Measured against a float64 reference the
   crossover is continuous and the branch taken is the more accurate one at
   every interval tried.

   Residual float32 limit, stated rather than hidden: a very NARROW interval
   in the far tail differences two large logs, e.g. [5, 5.00001] carries ~7e-3
   nats of error. That is inherent to float32 and is bounded, unlike the 60-nat
   sign-flipped bonus it replaces. A zero-WIDTH interval returns -Inf here
   (hence +Inf density), which is the honest limit of a point mass; `validate`
   already rejects lo >= hi, so it can only arise if (hi-lo)/sigma underflows."
  [a b]
  (let [refl? (mx/less-equal b ZERO)
        a' (mx/where refl? (mx/negative b) a)
        b' (mx/where refl? (mx/negative a) b)
        erf? (mx/logical-or (mx/less-equal a' ZERO) (mx/less-equal b' ONE))
        ;; Both branches are fed IN-DOMAIN arguments in both cases, so the
        ;; unselected branch stays finite and mx/where's backward pass cannot
        ;; manufacture a NaN. The (0,2) and (-1,1) dummies never reach the
        ;; result and carry no gradient: the where on the INPUT already zeroes
        ;; their cotangent.
        qa (log-q (mx/where erf? ZERO a'))
        qb (log-q (mx/where erf? TWO b'))
        tail (mx/add qa (mx/log (mx/negative (mx/expm1 (mx/subtract qb qa)))))
        ea (mx/erf (mx/divide (mx/where erf? a' (mx/negative ONE)) SQRT-TWO))
        eb (mx/erf (mx/divide (mx/where erf? b' ONE) SQRT-TWO))
        near (mx/log (mx/multiply HALF (mx/subtract eb ea)))]
    (mx/where erf? near tail)))

;; ---------------------------------------------------------------------------
;; Far-tail truncated-normal sampling (genmlx-smrk)
;; ---------------------------------------------------------------------------

(def ^:private TAIL-THRESHOLD
  "Reflected lower bound above which the native inverse-CDF sampler is unusable.
   MLX's random::truncated_normal draws uniform(erf(a/sqrt2), erf(b/sqrt2)); at
   a = 3 that lower endpoint is erf(2.12) = 0.99722, still well clear of 1.0 in
   float32, so the native path is exact below here and is kept."
  (mx/scalar 3.0))

;; Kept in-domain so the UNSELECTED tail branch is always finite: any interval
;; above TAIL-THRESHOLD works, [4,5] is simply a cheap representative.
(def ^:private TAIL-DUMMY-LO (mx/scalar 4.0))
(def ^:private TAIL-DUMMY-HI (mx/scalar 5.0))

;; Mirror of the above for the DIFFERENTIABLE central branch (genmlx-84mq). Both
;; branches are always evaluated, so in the tail case the central branch would
;; otherwise compute Phi(7) = Phi(8) = 1.0, hence erfinv(1) = +inf; the forward
;; clip hides that, but mx/where's backward multiplies the unselected branch by
;; 0 and 0*inf = NaN — which is exactly how the tail gradient started returning
;; NaN. Any in-domain interval works; [-1, 1] is a cheap representative.
(def ^:private CENTRAL-DUMMY-LO (mx/scalar -1.0))
(def ^:private CENTRAL-DUMMY-HI (mx/scalar 1.0))
;; Floor for the seed sweep only — keeps log(z) in domain if the asymptotic
;; undershoots on the first pass; never reached once the sweep has converged.
(def ^:private SEED-FLOOR (mx/scalar 1e-6))

(defn- log-phi
  "log of the standard-normal PDF."
  [z]
  (mx/negative (mx/add LOG-2PI-HALF (mx/multiply HALF (mx/square z)))))

(defn- log-q-inverse
  "Solve log-q(z) = target for z >= lo, in the upper tail.

   Two stages, both branchless and differentiable:

   1. Seed from the tail asymptotic log Q(z) ~ -z^2/2 - log z - log(2pi)/2,
      i.e. the fixed point z = sqrt(2s - 2 log z) with s = -target - log(2pi)/2.
      Three sweeps put the seed within ~1e-3 of the root over the whole domain.
   2. Newton on f(z) = log-q(z) - target. Since f'(z) = -phi(z)/Q(z), the step
      is (log-q z - target) * exp(log-q z - log-phi z) — the Mills ratio, which
      is ~1/z in the tail, so the step is well scaled at every magnitude and
      needs no damping.

   Two of each is the measured float32 floor, not a guess. Sweeping V over a
   400-point grid on [3.1,4], [4,5], [7,8], [10,20], [7,1000], [50,60] and the
   narrow [5,5.001], the worst |z - z_at(8,8)| is 9.5e-7 at (2,2) — one float32
   ULP at z = 7 — and (3,3) and (4,4) do not improve it. Dropping to a single
   Newton step is 10-100x worse (1e-5 .. 1e-4), so 2 is the knee.

   Each iteration costs a full log-q, which is ~27 kernel launches; this is the
   hot part of the sampler, so the count is worth keeping honest."
  [target lo]
  (let [s (mx/maximum (mx/subtract (mx/negative target) LOG-2PI-HALF) ONE)
        two-s (mx/multiply TWO s)
        seed (reduce (fn [z _]
                       (mx/sqrt (mx/maximum (mx/subtract two-s
                                                         (mx/multiply TWO (mx/log z)))
                                            SEED-FLOOR)))
                     (mx/sqrt two-s)
                     (range 2))]
    (reduce (fn [z _]
              (let [lq (log-q z)]
                (mx/add z (mx/multiply (mx/subtract lq target)
                                       (mx/exp (mx/subtract lq (log-phi z)))))))
            (mx/maximum seed lo)
            (range 2))))

(defn- truncated-normal-standard
  "z ~ TruncatedNormal(0, 1) on [a, b], of the given shape.

   MLX's native `random::truncated_normal` (random.cpp:324, ported from JAX) is
   an inverse-CDF draw through `erf`/`erfinv`. Beyond ~5-6 sigma `erf(a/sqrt2)`
   and `erf(b/sqrt2)` both round to 1.0f, so `uniform(a,b)` is degenerate at
   1.0, `erfinv(1)` is +inf, and the trailing clip turns that into a bound —
   every draw from (0,1,7,8) came back as exactly 8.0, which LOOKS in-bounds
   and is why no bounds assertion ever caught it (genmlx-smrk).

   The tail branch below does the same inverse-CDF draw in LOG space, where
   nothing saturates. Sampling Q uniformly on [Q(b), Q(a)] is
     log Q_t = log Q(a) + log(exp d - V*expm1 d),   d = log Q(b) - log Q(a)
   which stays exact as d -> -inf (it degenerates to log Q(a) + log V), and
   `log-q-inverse` maps it back to z.

   Native is retained below TAIL-THRESHOLD, where it is exact and cheaper.
   Both branches are always evaluated (MLX has no lazy branch), so a
   truncated-normal draw costs the tail arithmetic even in the central case.

   The two branches deliberately SHARE one key rather than splitting it. That
   is safe because exactly one of them reaches the result on any given lane, so
   no draw is ever a function of both and the correlation between them is
   unobservable. It is also worth 2.8 ms per call: `rng/split` measured 2.77 ms
   on sm_120, more than the whole tail arithmetic, and splitting here would buy
   independence that nothing can observe. (Do not 'restore' the split for
   hygiene — the reason it is absent is measured, not an oversight.)"
  ([key a b shape] (truncated-normal-standard key a b shape false))
  ([key a b shape differentiable?]
  (let [key (rng/ensure-key key)
        ;; Reflect an entirely-non-positive interval onto the upper tail, the
        ;; same move log-normal-mass makes — the native sampler is broken on
        ;; BOTH far tails, symmetrically.
        refl? (mx/less-equal b ZERO)
        a' (mx/where refl? (mx/negative b) a)
        b' (mx/where refl? (mx/negative a) b)
        tail? (mx/greater a' TAIL-THRESHOLD)
        ;; In-domain dummies keep the unselected branch finite, so mx/where's
        ;; backward pass cannot manufacture a NaN.
        at (mx/where tail? a' TAIL-DUMMY-LO)
        bt (mx/where tail? b' TAIL-DUMMY-HI)
        la (log-q at)
        d  (mx/subtract (log-q bt) la)
        v  (rng/uniform key shape)
        target (mx/add la (mx/log (mx/subtract (mx/exp d)
                                               (mx/multiply v (mx/expm1 d)))))
        ;; the exact root lies in [at, bt]; the clip only absorbs float32 residue
        z-tail (mx/clip (log-q-inverse target at) at bt)
        ;; Central branch. `rng/truncated-normal` is MLX's native inverse-CDF
        ;; draw — exact and cheaper, but it propagates NO gradient through
        ;; `lower`/`upper`, so every `d/dmu`, `d/dsigma` term that reaches z
        ;; through the standardized bounds is silently dropped (genmlx-84mq).
        ;; That is invisible for `sample` (values are correct) and wrong for
        ;; `reparam`, so the differentiable form is opt-in per call site: the
        ;; same inverse CDF written in CLJS, where a and b carry their
        ;; dependence. It reuses the tail branch's uniform for the reason the
        ;; docstring gives — exactly one branch reaches the result per lane, so
        ;; sharing is unobservable and saves a 2.77 ms rng/split.
        z-central (if differentiable?
                    ;; In-domain dummies when the TAIL branch will win, for the same
                    ;; reason at/bt exist above: Phi saturates to 1.0 out there, so
                    ;; erfinv(1) = inf and where's backward turns 0*inf into NaN.
                    (let [ac (mx/where tail? CENTRAL-DUMMY-LO a)
                          bc (mx/where tail? CENTRAL-DUMMY-HI b)
                          phi (fn [x] (mx/multiply HALF
                                                   (mx/add ONE (mx/erf (mx/divide x SQRT-TWO)))))
                          pa (phi ac)
                          pb (phi bc)
                          p  (mx/add pa (mx/multiply v (mx/subtract pb pa)))
                          z  (mx/multiply SQRT-TWO
                                          (mx/erfinv (mx/subtract (mx/multiply TWO p) ONE)))]
                      ;; the exact root lies in [ac, bc]; clip absorbs float32 residue,
                      ;; mirroring the tail branch
                      (mx/clip z ac bc))
                    (rng/truncated-normal key a b shape))]
    (mx/where tail?
              (mx/where refl? (mx/negative z-tail) z-tail)
              z-central))))

;; ---------------------------------------------------------------------------
;; Truncated Normal
;; ---------------------------------------------------------------------------

(defdist truncated-normal
  "Truncated normal distribution on [lo, hi] with parameters mu and sigma."
  [mu sigma lo hi]
  (validate (check-positive "truncated-normal" "sigma" sigma)
            (check-less-than "truncated-normal" "lo" lo "hi" hi))
  (sample [key]
          (let [z (truncated-normal-standard key
                                             (mx/divide (mx/subtract lo mu) sigma)
                                             (mx/divide (mx/subtract hi mu) sigma)
                                             [])]
            (mx/add mu (mx/multiply sigma z))))
  (log-prob [v]
    ;; log p(v) = log N(v; mu, sigma) - log(Phi(b) - Phi(a))
    ;; where a = (lo - mu)/sigma, b = (hi - mu)/sigma
            (let [z (mx/divide (mx/subtract v mu) sigma)
          ;; Standard normal log-pdf, minus log(sigma)
                  log-pdf (mx/subtract (log-phi z) (mx/log sigma))
          ;; Normalization in log space — see log-normal-mass. The old
          ;; `log(max(Phi(b) - Phi(a), 1e-38))` clamp is gone: it did not
          ;; bound the far-tail error, it only relabelled it (the clamp is
          ;; what turned the collapse into a fixed +87.5 nat bonus).
                  log-norm (log-normal-mass
                            (mx/divide (mx/subtract lo mu) sigma)
                            (mx/divide (mx/subtract hi mu) sigma))
          ;; Bounds check
                  in-bounds (mx/multiply (mx/greater-equal v lo) (mx/less-equal v hi))]
              (mx/where in-bounds (mx/subtract log-pdf log-norm) NEG-INF)))
  (reparam [key]
           ;; differentiable? = true (genmlx-84mq): z depends on mu/sigma through
           ;; BOTH standardized bounds, so the pathwise derivative is
           ;;   d/dmu [mu + sigma*z(a(mu), b(mu))] = 1 + sigma*(dz/da*da/dmu + dz/db*db/dmu)
           ;; The native central sampler drops the bracketed term and returns a bare 1.0.
           ;; `sample` above deliberately stays on the native path — values are correct
           ;; there and it is cheaper; only the gradient path needs the CLJS inverse CDF.
           (let [z (truncated-normal-standard key
                                              (mx/divide (mx/subtract lo mu) sigma)
                                              (mx/divide (mx/subtract hi mu) sigma)
                                              []
                                              true)]
             (mx/add mu (mx/multiply sigma z)))))

(defmethod dc/dist-sample-n* :truncated-normal [d key n]
  (let [{:keys [mu sigma lo hi]} (:params d)
        key (rng/ensure-key key)
        a (mx/divide (mx/subtract lo mu) sigma)
        b (mx/divide (mx/subtract hi mu) sigma)
        z (truncated-normal-standard key a b [n])]
    (mx/add mu (mx/multiply sigma z))))

;; ---------------------------------------------------------------------------
;; Matrix-distribution helper
;; ---------------------------------------------------------------------------

(defn- as-square
  "Reshape a flat [k*k] array into [k k]; pass 2-d arrays through unchanged."
  [a]
  (if (= 1 (mx/ndim a))
    (let [k (int (js/Math.sqrt (mx/size a)))]
      (mx/reshape a [k k]))
    a))

;; ---------------------------------------------------------------------------
;; Multivariate Normal (via Cholesky) — manual definition
;; ---------------------------------------------------------------------------

(defn multivariate-normal
  "Create a Multivariate Normal distribution.
   mean-vec: [k] array, cov-matrix: [k k] positive definite array.
   Cholesky decomposition and L-inverse are computed once at construction."
  [mean-vec cov-matrix]
  (let [mu (mx/ensure-array mean-vec)
        cov-2d (as-square (mx/ensure-array cov-matrix))
        L (mx/cholesky cov-2d)
        _ (mx/materialize! L)
        Li (mx/tri-inv L false)
        k (first (mx/shape mu))
        log-det-sigma (mx/multiply TWO
                                   (mx/sum (mx/log (mx/diag L))))
        nc (mx/multiply (mx/scalar -0.5)
                        (mx/add (mx/scalar (* k LOG-2PI)) log-det-sigma))
        neg-half (mx/scalar -0.5)]
    (mx/materialize! Li nc)
    (dc/->Distribution :multivariate-normal
                       {:mean-vec mu :cov-matrix cov-2d :cholesky-L L
                        :L-inv Li :k k :norm-const nc :neg-half neg-half})))

(defmethod dc/dist-sample* :multivariate-normal [d key]
  (let [{:keys [mean-vec cholesky-L k]} (:params d)
        key (rng/ensure-key key)
        z (rng/normal key [k])]
    (mx/add mean-vec
            (mx/flatten (mx/matmul cholesky-L (mx/reshape z [k 1]))))))

(defmethod dc/dist-log-prob :multivariate-normal [d v]
  (let [{:keys [mean-vec L-inv k norm-const neg-half]} (:params d)
        v (mx/ensure-array v)
        diff (mx/subtract v mean-vec)
        y (mx/flatten (mx/matmul L-inv (mx/reshape diff [k 1])))
        mahal (mx/sum (mx/square y))]
    (mx/add (mx/multiply neg-half mahal) norm-const)))

(defmethod dc/dist-reparam :multivariate-normal [d key]
  (let [{:keys [mean-vec cholesky-L k]} (:params d)
        key (rng/ensure-key key)
        z (rng/normal key [k])]
    (mx/add mean-vec
            (mx/flatten (mx/matmul cholesky-L (mx/reshape z [k 1]))))))

(defmethod dc/dist-sample-n* :multivariate-normal [d key n]
  (let [{:keys [mean-vec cholesky-L k]} (:params d)
        key (rng/ensure-key key)
        z (rng/normal key [n k])
        ;; z: [n, k], L^T: [k, k] -> samples: [n, k]
        samples (mx/matmul z (mx/transpose cholesky-L))]
    (mx/add mean-vec samples)))

;; ---------------------------------------------------------------------------
;; Broadcasted Normal (independent element-wise Gaussians)
;; ---------------------------------------------------------------------------

(defdist broadcasted-normal
  "Independent element-wise normal distribution.
   mu and sigma are MLX arrays of any shape. Samples N(mu_i, sigma_i) independently."
  [mu sigma]
  (sample [key]
          (let [sh (mx/shape mu)]
            (mx/add mu (mx/multiply sigma (rng/normal key sh)))))
  (log-prob [v]
            (let [k (count (mx/shape mu))   ; event rank (mu carries the event shape)
                  lp (normal-log-density v mu sigma)]
              ;; Sum only over the trailing event axes so a leading particle axis
              ;; survives in batched mode: [...mu]->scalar, [N,...mu]->[N]. Summing
              ;; over ALL axes collapsed the particle axis to a scalar and
              ;; corrupted per-particle weights (genmlx-lgun); cf. gaussian-vec.
              (if (pos? k)
                (mx/sum lp (vec (range (- k) 0)))
                lp)))
  (reparam [key]
           (let [sh (mx/shape mu)]
             (mx/add mu (mx/multiply sigma (rng/normal key sh))))))

(defmethod dc/dist-sample-n* :broadcasted-normal [d key n]
  (let [{:keys [mu sigma]} (:params d)
        key (rng/ensure-key key)
        sh (mx/shape mu)]
    (mx/add mu (mx/multiply sigma (rng/normal key (into [n] sh))))))

;; ---------------------------------------------------------------------------
;; Gaussian Vec — independent Gaussians, log-prob summed over last axis
;; ---------------------------------------------------------------------------

(defdist gaussian-vec
  "Vector of independent Gaussians with shared or element-wise sigma.
   log-prob sums over the last axis, so:
     [D]-shaped value -> scalar log-prob (scalar mode)
     [N,D]-shaped value -> [N]-shaped log-prob (batched mode)
   Ideal for vectorized inference with one trace site per latent vector."
  [mu sigma]
  (sample [key]
          (mx/add mu (mx/multiply sigma (rng/normal key (mx/shape mu)))))
  (log-prob [v]
            (mx/sum (normal-log-density v mu sigma) [-1]))
  (reparam [key]
           (mx/add mu (mx/multiply sigma (rng/normal key (mx/shape mu))))))

(defmethod dc/dist-sample-n* :gaussian-vec [d key n]
  (let [{:keys [mu sigma]} (:params d)
        key (rng/ensure-key key)
        sh (mx/shape mu)]
    (mx/add mu (mx/multiply sigma (rng/normal key (into [n] sh))))))

;; ---------------------------------------------------------------------------
;; Beta-Uniform Mixture — convenience wrapper
;; ---------------------------------------------------------------------------

(defn beta-uniform-mixture
  "Mixture of Beta(alpha, beta-param) with probability theta and
   Uniform(0,1) with probability (1 - theta). Common prior for bounded params."
  [theta alpha beta-param]
  (dc/mixture [(beta-dist (mx/ensure-array alpha) (mx/ensure-array beta-param))
               (uniform ZERO ONE)]
              (mx/array [(js/Math.log theta)
                         (js/Math.log (- 1.0 theta))])))

;; ---------------------------------------------------------------------------
;; Piecewise Uniform
;; ---------------------------------------------------------------------------

(defdist piecewise-uniform
  "Piecewise uniform distribution over bins defined by sorted boundary points.
   bounds: MLX array of N+1 boundary points (sorted).
   probs:  MLX array of N unnormalized bin probabilities."
  [bounds probs]
  (sample [key]
          (let [[k1 k2] (rng/split key)
                ;; Choose bin via categorical — stays as MLX uint32 (no mx/item)
                log-probs (mx/log probs)
                bin-idx (rng/categorical k1 log-probs)
                ;; Index bounds with MLX integer — vectorizable + differentiable
                lo (mx/index bounds bin-idx)
                hi (mx/index bounds (mx/add bin-idx (mx/scalar 1 mx/int32)))
                u (rng/uniform k2 [])]
            (mx/add lo (mx/multiply u (mx/subtract hi lo)))))
  (log-prob [v]
    ;; Vectorized bin assignment using mx/where — works for scalar and [N]-shaped v
            (let [bounds-vals (mx/->clj bounds)
                  probs-vals (mx/->clj probs)
                  total (reduce + probs-vals)]
              (reduce
               (fn [acc [p [lo hi]]]
                 (let [width (- hi lo)
                       log-density (mx/scalar (- (js/Math.log p) (js/Math.log total) (js/Math.log width)))
                       in-bin (mx/multiply (mx/greater-equal v (mx/scalar lo))
                                           (mx/less v (mx/scalar hi)))]
                   (mx/where in-bin log-density acc)))
               NEG-INF
               (map vector probs-vals (partition 2 1 bounds-vals))))))

;; ---------------------------------------------------------------------------
;; Wishart
;; ---------------------------------------------------------------------------

(defn- log-multivariate-gamma
  "Log of the multivariate gamma function Gamma_k(a)."
  [a k]
  (+ (* k (dec k) 0.25 (js/Math.log js/Math.PI))
     (reduce + (map (fn [i] (log-gamma (- a (* 0.5 i)))) (range k)))))

(defn wishart
  "Wishart distribution with df degrees of freedom and [k x k] scale matrix V.
   Uses Bartlett decomposition for sampling."
  [df scale-matrix]
  (let [V-2d (as-square (mx/ensure-array scale-matrix))
        df-val (if (mx/array? df) (mx/realize df) df)
        L (mx/cholesky V-2d)
        _ (mx/materialize! L)
        k (first (mx/shape V-2d))]
    (dc/->Distribution :wishart
                       {:df df-val :scale-matrix V-2d :cholesky-L L :k k})))

(defmethod dc/dist-sample* :wishart [d key]
  (let [{:keys [df cholesky-L k]} (:params d)
        key (rng/ensure-key key)
        ;; Pre-split enough keys for all samples: k diagonal + k*(k-1)/2 off-diagonal
        n-keys (+ k (quot (* k (dec k)) 2))
        ks (rng/split-n key n-keys)
        ;; Build lower-triangular A (Bartlett decomposition)
        ;; Diagonal: A_ii ~ sqrt(chi²(df - i + 1)), chi²(n) = Gamma(n/2, 1/2)
        ;; Off-diagonal: A_ij ~ N(0,1)
        ;; Index keys sequentially without mutable state
        [A-data _]
        (reduce
         (fn [[rows ki] i]
           (let [[row ki']
                 (reduce
                  (fn [[cols ki] j]
                    (cond
                      (= i j) ;; diagonal: sqrt(chi²(df - i))
                      (let [chi2-df (- df i)
                            g (dc/dist-sample (gamma-dist (mx/scalar (/ chi2-df 2.0))
                                                          ONE)
                                              (nth ks ki))]
                        [(conj cols (mx/sqrt (mx/multiply TWO g))) (inc ki)])
                      (> i j) ;; below diagonal: N(0,1)
                      [(conj cols (rng/normal (nth ks ki) [])) (inc ki)]
                      :else ;; above diagonal: 0
                      [(conj cols ZERO) ki]))
                  [[] ki]
                  (range k))]
             [(conj rows row) ki']))
         [[] 0]
         (range k))
        ;; Build A matrix
        A (mx/reshape (mx/stack (mapv mx/stack A-data)) [k k])
        ;; W = L * A * A^T * L^T
        LA (mx/matmul cholesky-L A)
        W (mx/matmul LA (mx/transpose LA))]
    (mx/materialize! W)
    W))

(defmethod dc/dist-log-prob :wishart [d x]
  (let [{:keys [df scale-matrix k]} (:params d)
        x (mx/ensure-array x)
        x-2d (as-square x)
        ;; Recompute V-inv and log-det-V from scale-matrix (not precomputed)
        ;; so gradient tape is preserved for differentiating w.r.t. V.
        V-inv (mx/inv scale-matrix)
        log-det-V (mx/spd-logdet scale-matrix)
        log-det-X (mx/spd-logdet x-2d)
        ;; log p(X) = ((df-k-1)/2)*log|X| - (1/2)*tr(V^{-1}X) - (df*k/2)*log(2)
        ;;            - (df/2)*log|V| - log_multivariate_gamma(df/2, k)
        half-df (/ df 2.0)
        term1 (mx/multiply (mx/scalar (/ (- df k 1) 2.0)) log-det-X)
        tr-VinvX (mx/sum (mx/multiply V-inv (mx/transpose x-2d))) ;; tr(A*B) = sum(A .* B^T)
        term2 (mx/multiply (mx/scalar -0.5) tr-VinvX)
        term3 (mx/scalar (- (* half-df k (js/Math.log 2.0))))
        term4 (mx/multiply (mx/scalar (- half-df)) log-det-V)
        term5 (mx/scalar (- (log-multivariate-gamma half-df k)))]
    (mx/add term1 term2 term3 term4 term5)))

;; ---------------------------------------------------------------------------
;; Inverse Wishart
;; ---------------------------------------------------------------------------

(defn inv-wishart
  "Inverse Wishart distribution with df degrees of freedom and [k x k] scale matrix Psi.
   Sample: W ~ Wishart(df, Psi^{-1}), return W^{-1}."
  [df scale-matrix]
  (let [Psi-2d (as-square (mx/ensure-array scale-matrix))
        df-val (if (mx/array? df) (mx/realize df) df)
        k (first (mx/shape Psi-2d))
        Psi-inv (mx/inv Psi-2d)
        _ (mx/materialize! Psi-inv)
        ;; Build internal Wishart(df, Psi^{-1}) for sampling
        wish (wishart df-val Psi-inv)]
    (dc/->Distribution :inv-wishart
                       {:df df-val :scale-matrix Psi-2d :k k
                        :wish wish})))

(defmethod dc/dist-sample* :inv-wishart [d key]
  (let [{:keys [wish]} (:params d)
        W (dc/dist-sample wish key)]
    (mx/inv W)))

(defmethod dc/dist-log-prob :inv-wishart [d x]
  (let [{:keys [df scale-matrix k]} (:params d)
        x (mx/ensure-array x)
        x-2d (as-square x)
        X-inv (mx/inv x-2d)
        ;; Recompute log-det from raw matrices (not precomputed) so gradient
        ;; tape is preserved for differentiating w.r.t. Psi and X.
        log-det-Psi (mx/spd-logdet scale-matrix)
        log-det-X (mx/spd-logdet x-2d)
        ;; log p(X) = (df/2)*log|Psi| - (df*k/2)*log(2) - log_multivariate_gamma(df/2, k)
        ;;            - ((df+k+1)/2)*log|X| - (1/2)*tr(Psi * X^{-1})
        half-df (/ df 2.0)
        term1 (mx/multiply (mx/scalar half-df) log-det-Psi)
        term2 (mx/scalar (- (* half-df k (js/Math.log 2.0))))
        term3 (mx/scalar (- (log-multivariate-gamma half-df k)))
        term4 (mx/multiply (mx/scalar (- (/ (+ df k 1) 2.0))) log-det-X)
        tr-PsiXinv (mx/sum (mx/multiply scale-matrix (mx/transpose X-inv)))
        term5 (mx/multiply (mx/scalar -0.5) tr-PsiXinv)]
    (mx/add term1 term2 term3 term4 term5)))

;; ---------------------------------------------------------------------------
;; Product distribution (re-export from dist.core)
;; ---------------------------------------------------------------------------

(def product "Alias for `genmlx.dist.core/product` — product of independent component distributions." dc/product)

;; ---------------------------------------------------------------------------
;; Von Mises — circular distribution
;; ---------------------------------------------------------------------------

(defn- wrap-angle
  "Wrap value to [-π, π)."
  [x]
  (mx/subtract x
               (mx/multiply TWO-PI
                            (mx/floor (mx/divide (mx/add x MLX-PI)
                                                 TWO-PI)))))

(defdist von-mises
  "Von Mises distribution on [-π, π) with mean direction mu and concentration kappa."
  [mu kappa]
  (validate (check-positive "von-mises" "kappa" kappa))
  (sample [key]
          ;; Best's rejection algorithm
          (let [k-val (mx/realize kappa)
                ;; Standard algorithm parameters:
                ;; τ = 1 + sqrt(1 + 4κ²), ρ = (τ - sqrt(2τ))/(2κ), r = (1 + ρ²)/(2ρ)
                tau2 (+ 1.0 (js/Math.sqrt (+ 1.0 (* 4.0 k-val k-val))))
                rho (/ (- tau2 (js/Math.sqrt (* 2.0 tau2))) (* 2.0 k-val))
                r (/ (+ 1.0 (* rho rho)) (* 2.0 rho))]
            (loop [k key]
              (let [[k1 k2 k3] (rng/split-n k 3)
                    u1 (mx/realize (rng/uniform k1 []))
                    z (js/Math.cos (* js/Math.PI u1))
                    f (/ (+ 1.0 (* r z)) (+ r z))
                    c (* k-val (- r f))
                    u2 (mx/realize (rng/uniform k2 []))]
                ;; Best (1979) acceptance conditions for ratio c*exp(1-c):
                ;; Squeeze (no log): c*(2-c) >= u2  [since exp(x) >= 1+x]
                ;; Log test (exact): log(c/u2) >= c - 1
                (if (or (>= (* c (- 2.0 c)) u2)
                        (>= (js/Math.log (/ c u2)) (- c 1.0)))
                  ;; Accept: angle = sign(u3 - 0.5) * acos(f) + mu
                  (let [u3 (mx/realize (rng/uniform k3 []))
                        theta (* (js/Math.sign (- u3 0.5)) (js/Math.acos f))]
                    (wrap-angle (mx/add (mx/scalar theta) mu)))
                  (recur k3))))))
  (log-prob [v]
    ;; log p(x) = κ cos(x - μ) - log(2π) - log(I₀(κ))
    ;; log(I₀(κ)) = log(i0e(κ)) + |κ|  since i0e(κ) = exp(-|κ|) I₀(κ)
            (let [log-I0 (mx/add (mx/log (mx/bessel-i0e kappa)) (mx/abs kappa))
                  log-norm (mx/add (mx/scalar LOG-2PI) log-I0)]
              (mx/subtract (mx/multiply kappa (mx/cos (mx/subtract v mu)))
                           log-norm))))

(defmethod dc/dist-sample-n* :von-mises [d key n]
  (let [{:keys [mu kappa]} (:params d)
        key (rng/ensure-key key)
        ;; Pre-compute scalar rejection params (same for all n samples)
        k-val (mx/realize kappa)
        tau2 (+ 1.0 (js/Math.sqrt (+ 1.0 (* 4.0 k-val k-val))))
        rho (/ (- tau2 (js/Math.sqrt (* 2.0 tau2))) (* 2.0 k-val))
        r (/ (+ 1.0 (* rho rho)) (* 2.0 rho))
        r-arr (mx/scalar r)
        k-arr (mx/scalar k-val)
        max-iter 20]
    (loop [iter 0
           result (mx/zeros [n])
           done (mx/zeros [n]) ;; float 0.0/1.0 mask (same as gamma)
           k key]
      (if (>= iter max-iter)
        (wrap-angle (mx/add result mu))
        (let [[k1 k2 k3 k4] (rng/split-n k 4)
              u1 (rng/uniform k1 [n])
              u2 (rng/uniform k2 [n])
              u3 (rng/uniform k3 [n])
              ;; z = cos(π u1), f = (1 + r*z)/(r + z), c = κ(r - f)
              z (mx/cos (mx/multiply MLX-PI u1))
              f (mx/divide (mx/add ONE (mx/multiply r-arr z))
                           (mx/add r-arr z))
              c (mx/multiply k-arr (mx/subtract r-arr f))
              ;; Best (1979) acceptance conditions for ratio c*exp(1-c):
              ;; Squeeze (no log): c*(2-c) >= u2  [since exp(x) >= 1+x]
              ;; Log test (exact): log(c/u2) >= c - 1
              safe-c (mx/maximum c (mx/scalar 1e-30))
              cond1 (mx/greater-equal (mx/multiply c (mx/subtract (mx/scalar 2.0) c))
                                      u2)
              cond2 (mx/greater-equal (mx/log (mx/divide safe-c u2))
                                      (mx/subtract c ONE))
              accepted (mx/maximum cond1 cond2)
              ;; theta = sign(u3 - 0.5) * arccos(clamp(f))
              sign-u3 (mx/sign (mx/subtract u3 HALF))
              safe-f (mx/minimum (mx/maximum f (mx/scalar -1.0)) ONE)
              theta (mx/multiply sign-u3 (mx/arccos safe-f))
              ;; Only fill not-yet-done slots
              not-done (mx/equal done ZERO)
              newly-done (mx/multiply accepted not-done)
              result (mx/where newly-done theta result)
              done (mx/where newly-done ONE done)]
          (recur (inc iter) result done k4))))))

;; ---------------------------------------------------------------------------
;; Wrapped Cauchy — closed-form circular distribution
;; ---------------------------------------------------------------------------

(defdist wrapped-cauchy
  "Wrapped Cauchy distribution on [-π, π) with mean mu and concentration rho (0 < ρ < 1)."
  [mu rho]
  (validate (check-open-probability "wrapped-cauchy" "rho" rho))
  (sample [key]
    ;; Inverse CDF: mu + 2*atan2((1-rho)*tan(π*(u-0.5)), (1+rho))
          (let [u (mx/realize (rng/uniform key []))
                r (mx/realize rho)
                theta (+ (mx/realize mu)
                         (* 2.0 (js/Math.atan2
                                 (* (- 1.0 r) (js/Math.tan (* js/Math.PI (- u 0.5))))
                                 (+ 1.0 r))))]
            (wrap-angle (mx/scalar theta))))
  (log-prob [v]
    ;; log(1 - ρ²) - log(2π) - log(1 - 2ρ cos(x - μ) + ρ²)
            (let [rho-sq (mx/square rho)]
              (mx/subtract
               (mx/subtract (mx/log (mx/subtract ONE rho-sq))
                            (mx/scalar LOG-2PI))
               (mx/log (mx/add (mx/subtract ONE
                                            (mx/multiply (mx/multiply TWO rho)
                                                         (mx/cos (mx/subtract v mu))))
                               rho-sq))))))

(defmethod dc/dist-sample-n* :wrapped-cauchy [d key n]
  (let [key (rng/ensure-key key)
        ks (rng/split-n key n)]
    (mx/stack (mapv #(dc/dist-sample d %) ks))))

;; ---------------------------------------------------------------------------
;; Wrapped Normal — Gaussian wrapped onto circle
;; ---------------------------------------------------------------------------

(defdist wrapped-normal
  "Wrapped normal distribution on [-π, π) with mean mu and std sigma."
  [mu sigma]
  (validate (check-positive "wrapped-normal" "sigma" sigma))
  (sample [key]
    ;; Sample from N(μ, σ), wrap to [-π, π)
          (let [x (mx/add mu (mx/multiply sigma (rng/normal key [])))]
            (wrap-angle x)))
  (log-prob [v]
    ;; Series sum: log Σ_{k=-K}^{K} exp(normal-log-prob(x + 2πk; μ, σ))
    ;; Truncated at K=3 terms
            (let [terms (mapv (fn [k]
                                (let [shifted (mx/add v (mx/multiply TWO-PI (mx/scalar k)))]
                                  (normal-log-density shifted mu sigma)))
                              (range -3 4))]
      ;; logsumexp over the 7 terms
              (reduce mx/logaddexp terms))))

(defmethod dc/dist-sample-n* :wrapped-normal [d key n]
  (let [{:keys [mu sigma]} (:params d)
        key (rng/ensure-key key)]
    (wrap-angle (mx/add mu (mx/multiply sigma (rng/normal key [n]))))))

;; ---------------------------------------------------------------------------
;; IID distribution — wraps any base distribution for stacked trace sites
;; ---------------------------------------------------------------------------
;;
;; (iid base-dist t) creates a distribution that samples T independent values.
;; In scalar mode: sample → [T], log-prob([T]) → scalar.
;; In batched mode: sample-n(N) → [N, T], log-prob([N,T]) → [N].
;;
;; When base-dist params are already batched (e.g., means [N,T] from batched
;; latent variables), sample/sample-n detect this and handle correctly.

(defn iid
  "IID distribution: sample t independent values from base-dist.
   base-dist: any Distribution record (gaussian, uniform, etc.)
   t: number of independent samples (positive integer)

   sample  → [T]-shaped tensor
   log-prob([T] values) → scalar (sum of element log-probs)
   sample-n(N) → [N, T]-shaped tensor"
  [base-dist t]
  (dc/->Distribution :iid {:base-dist base-dist :t t}))

(defmethod dc/dist-sample* :iid [d key]
  (let [{:keys [base-dist t]} (:params d)
        key (rng/ensure-key key)
        ;; Sample T values independently via split keys
        ks (rng/split-n key t)
        samples (mx/stack (mapv #(dc/dist-sample base-dist %) ks))
        ;; samples shape: [T] for scalar params, [T, N] for [N]-shaped params
        ;; We need [T] or [N, T] respectively
        ndim (count (mx/shape samples))]
    (if (> ndim 1)
      ;; [T, N] → [N, T] via transpose
      (mx/transpose samples)
      ;; [T] — scalar mode, already correct
      samples)))

(defmethod dc/dist-log-prob :iid [d vs]
  (let [{:keys [base-dist]} (:params d)            ; t unused here (genmlx-21kt)
        vs (mx/ensure-array vs)
        val-shape (mx/shape vs)
        ndim (count val-shape)
        ;; Compute element-wise log-probs via base dist.
        ;; Broadcasting handles all shape combinations:
        ;;   vals [T], params scalar → lps [T]
        ;;   vals [T], params [T]   → lps [T]
        ;;   vals [T], params [N,T] → lps [N,T] (broadcast [T] with [N,T])
        ;;   vals [N,T], params scalar → lps [N,T]
        ;;   vals [N,T], params [N,T] → lps [N,T]
        element-lps (dc/dist-log-prob base-dist vs)]
    ;; Sum over the T dimension (last axis) to get per-particle log-prob.
    (mx/sum element-lps -1)))

(defmethod dc/dist-sample-n* :iid [d key n]
  (let [{:keys [base-dist t]} (:params d)
        key (rng/ensure-key key)
        ;; Probe a single base sample to detect param shapes
        [k1 k2] (rng/split key)
        probe (dc/dist-sample base-dist k1)
        probe-shape (mx/shape probe)
        probe-ndim (count probe-shape)]
    (cond
      ;; Base sample returns [N, T] — params already fully batched.
      ;; A single sample IS the result. Don't stack.
      (>= probe-ndim 2)
      probe

      ;; Base sample returns [N] — params have batch dim but not T.
      ;; Use sample method which stacks T values and transposes to [N, T].
      (= probe-ndim 1)
      (dc/dist-sample* d k2)

      ;; Base sample returns scalar — sample N*T flat, reshape to [N, T].
      :else
      (let [flat (dc/dist-sample-n base-dist k2 (* n t))]
        (mx/reshape flat [n t])))))

(defmethod dc/dist-reparam :iid [d key]
  (let [{:keys [base-dist t]} (:params d)
        key (rng/ensure-key key)
        ks (rng/split-n key t)
        ;; Mirror dist-sample* :iid exactly: [T,N] -> [N,T] for [N]-batched base
        ;; params so reparam values are laid out [N,T] like the sampling path,
        ;; not transposed (which would mis-pair particles with scores). No-op for
        ;; the scalar [T] case (genmlx-exw9).
        stacked (mx/stack (mapv #(dc/dist-reparam base-dist %) ks))]
    (if (> (count (mx/shape stacked)) 1)
      (mx/transpose stacked)
      stacked)))

;; ---------------------------------------------------------------------------
;; IID Gaussian — specialized for maximum performance
;; ---------------------------------------------------------------------------
;;
;; Optimized iid gaussian that generates all T samples in one MLX op.
;; mu can be scalar (shared) or [T]-shaped (per-element means).
;; sigma is scalar or [T]-shaped.

(defn iid-gaussian
  "Optimized IID Gaussian. mu/sigma can be scalar or [T]-shaped.
   Generates T samples in a single noise draw."
  [mu sigma t]
  ;; Validate BEFORE ensure-array: check-positive only inspects JS numbers,
  ;; so converting first made the check dead code (genmlx-yeam).
  (check-positive "iid-gaussian" "sigma" sigma)
  (let [mu (mx/ensure-array mu)
        sigma (mx/ensure-array sigma)]
    (dc/->Distribution :iid-gaussian {:mu mu :sigma sigma :t t})))

(defmethod dc/dist-sample* :iid-gaussian [d key]
  (let [{:keys [mu sigma t]} (:params d)
        key (rng/ensure-key key)
        noise (rng/normal key [t])]
    (mx/add mu (mx/multiply sigma noise))))

(defmethod dc/dist-log-prob :iid-gaussian [d vs]
  (let [{:keys [mu sigma]} (:params d)
        vs (mx/ensure-array vs)
        ;; Handle broadcasting for batched case
        mu-shape (mx/shape mu)
        val-shape (mx/shape vs)
        [vals-bc mu-bc]
        (cond
          ;; Both 1D, different lengths: outer-product case (vgenerate constraint)
          (and (= (count val-shape) 1)
               (= (count mu-shape) 1)
               (not= (first val-shape) (first mu-shape)))
          [(mx/reshape vs [1 (first val-shape)])
           (mx/expand-dims mu -1)]

          ;; vals 2D, mu 1D matching first axis: batched with scalar mu
          (and (= (count mu-shape) 1)
               (> (count val-shape) 1)
               (= (first mu-shape) (first val-shape))
               (not= (first mu-shape) (last val-shape)))
          [vs (mx/expand-dims mu -1)]

          ;; All other cases: natural broadcasting works
          :else
          [vs mu])
        element-lps (normal-log-density vals-bc mu-bc sigma)]
    (mx/sum element-lps -1)))

(defmethod dc/dist-sample-n* :iid-gaussian [d key n]
  (let [{:keys [mu sigma t]} (:params d)
        key (rng/ensure-key key)
        noise (rng/normal key [n t])
        sh (mx/shape mu)
        ;; Expand mu for broadcasting: scalar→scalar, [T]→[1,T], [N]→[N,1]
        mu-nd (cond
                ;; scalar or already-2D mu broadcasts naturally
                (not= (count sh) 1) mu
                ;; 1D: could be [T] or [N]. If length=t → [T] params (row),
                ;; otherwise [N] batched (column). KNOWN AMBIGUITY: when
                ;; N == t a batched [N] mu is misread as per-element [T]
                ;; params — shape-pun class tracked in genmlx-ql6a; fixing
                ;; it needs an explicit batch-axis marker, not a heuristic.
                (= (first sh) t) (mx/reshape mu [1 t])
                :else (mx/expand-dims mu -1))]
    (mx/add mu-nd (mx/multiply sigma noise))))

(defmethod dc/dist-reparam :iid-gaussian [d key]
  (let [{:keys [mu sigma t]} (:params d)
        key (rng/ensure-key key)
        noise (rng/normal key [t])]
    (mx/add mu (mx/multiply sigma noise))))
