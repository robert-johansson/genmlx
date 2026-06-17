(ns genmlx.encapsulated
  "Encapsulated randomness — Cusumano-Towner 2020 PhD thesis, Chapter 4.5.

   ## What this is

   A generative function normally exposes an EXACT density: the trace's score
   is log p(tau; x). §4.5 relaxes this. A generative function with encapsulated
   randomness owns internal randomness omega (NOT part of the choice dictionary
   tau), and its realized score is a value of an unbiased density ESTIMATOR

       xi(x, tau, omega),   with   E_omega[ xi(x, tau, omega) ] = p(tau; x)
                                                                       (Eq 4.3)

   The trace carries omega alongside tau (the optional `omega` field on
   genmlx.trace/Trace), so the realized score is reproducible: re-evaluating the
   estimator at the recorded omega yields the same xi. Reproducibility is what
   makes identity update/project return weight 0 exactly, while a genuine move
   (changed value or args) resamples omega and pays the pseudo-marginal ratio
   log xi' − log xi_old.

   ## Why it matters (the three §4.5 use cases)

   - Models that call black-box stochastic code whose density is intractable but
     unbiasedly estimable (here: `marginalized-gaussian`, a latent-variable
     likelihood estimated by importance sampling over the nuisance).
   - Finite mixtures whose normalizer is integrated by Monte Carlo over an
     encapsulated component index (`mixture-density`).
   - Pseudo-marginal MCMC: the score becomes an unbiased estimator, and the
     chain still targets the exact posterior (`pseudo-marginal-mh`,
     Andrieu-Roberts 2009; the auxiliary-variable argument is Eq 4.3-4.4).

   ## Contract on the caller-supplied estimator (the §4.5 obligation)

   `encapsulated` takes a single observed address and three closures:
     :sample-value          (fn [key args] -> v)        draws v ~ p(.; args)
     :sample-omega          (fn [key args v] -> omega)  draws encapsulated randomness
     :log-density-estimate  (fn [args v omega] -> logxi-scalar)
   Obligations:
     (1) E_omega[ exp(log-density-estimate args v omega) ] = p(v; args)  (Eq 4.3)
     (2) log-density-estimate is DETERMINISTIC given omega (reproducibility)
     (3) xi >= 0 (the estimate is a non-negative density; logxi finite)

   ## NOTE on score-type

   An encapsulated trace is tagged :joint (the default). Its score is an
   estimate, but it is a REPRODUCIBLE function of the recorded omega and every
   GFI op re-derives it consistently — so, unlike an L3 :marginal trace (which
   pins latents at posterior means and switches decomposition between ops), it
   composes like an ordinary joint density and is safe for trace-MH ratios.
   See genmlx.trace score-type notes and ARCHITECTURE §3.3.

   ## Limitation (scope)

   An EncapsulatedGF is a STANDALONE / top-level generative function (and the
   likelihood in pseudo-marginal MCMC). It is NOT a combinator/vmap kernel:
   shape-batched combinators (genmlx.vmap, Map/Unfold/Scan) reconstruct
   per-element traces via make-trace WITHOUT the omega field, which would break
   the estimator's reproducibility (identity ops would no longer cost weight 0).
   Use it directly, not wrapped in a combinator."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.mlx.constants :refer [ZERO LOG-2PI]]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.diff :as diff]))

;; ---------------------------------------------------------------------------
;; Key handling
;;
;; EncapsulatedGF honours the SAME metadata key dynamic.cljs uses
;; (:genmlx.dynamic/key), so dyn/with-key and dyn/auto-key work transparently
;; on it. The keyword literal is referenced as data — no namespace dependency,
;; no circular-require risk. A real PRNG key is an MLX array; the auto-key
;; sentinel is a keyword; absent => fresh per call (REPL/test convenience).
;; ---------------------------------------------------------------------------

(defn- enc-key [gf]
  (let [k (:genmlx.dynamic/key (meta gf))]
    (if (or (nil? k) (keyword? k)) (rng/fresh-key) k)))

;; ---------------------------------------------------------------------------
;; Internal helpers
;; ---------------------------------------------------------------------------

(defn- addr-value
  "Leaf value stored at the GF's single observed address in a choicemap, or nil."
  [choices addr]
  (cm/get-value (cm/get-submap choices addr)))

(defn- same-value?
  "True if MLX values a and b are equal (identity fast-path, else elementwise).
   Used to decide whether an update at the observed address is a genuine move
   (resample omega) or a no-op (reuse omega → weight 0)."
  [a b]
  (or (identical? a b)
      (and (some? a) (some? b)
           (= (mx/realize-clj a) (mx/realize-clj b)))))

(defn- same-args?
  "True if argument vectors a and b are element-wise equal (MLX-aware)."
  [a b]
  (and (= (count a) (count b))
       (every? true?
               (map (fn [x y]
                      (if (or (mx/array? x) (mx/array? y))
                        (same-value? x y)
                        (= x y)))
                    a b))))

(defn- make-enc-trace
  "Build an encapsulated Trace: choices = {addr v}, score = logxi, omega stored,
   retval = v. Tagged :joint (see ns docstring)."
  [gf args addr v logxi omega]
  (tr/with-score-type
    (tr/make-trace {:gen-fn gf :args args
                    :choices (cm/set-value cm/EMPTY addr v)
                    :retval v :score logxi :omega omega})
    :joint))

(defn- resample-result
  "Build the {:trace :weight :discard} result for a genuine move at new value
   `v` under `args`: draw fresh omega with `key`, recompute logxi', weight is
   logxi' minus the OLD trace's stored score (the pseudo-marginal ratio). The
   old score is the auxiliary variable being reused as the move's denominator."
  [gf args addr v old-score discard key]
  (let [{:keys [sample-omega log-density-estimate]} gf
        omega' (sample-omega key args v)
        logxi' (log-density-estimate args v omega')]
    {:trace (make-enc-trace gf args addr v logxi' omega')
     :weight (mx/subtract logxi' old-score)
     :discard discard}))

;; ---------------------------------------------------------------------------
;; EncapsulatedGF — full GFI over a single observed address
;; ---------------------------------------------------------------------------

(defrecord EncapsulatedGF [addr sample-value sample-omega log-density-estimate]
  p/IGenerativeFunction
  (simulate [gf args]
    (let [[k1 k2] (rng/split (enc-key gf))
          v (sample-value k1 args)
          omega (sample-omega k2 args v)
          logxi (log-density-estimate args v omega)]
      (make-enc-trace gf args addr v logxi omega)))

  p/IGenerate
  (generate [gf args constraints]
    (let [constraint (cm/get-submap constraints addr)]
      (if (cm/has-value? constraint)
        ;; Observed value constrained: draw fresh omega, weight = score = logxi.
        (let [v (cm/get-value constraint)
              [_ k2] (rng/split (enc-key gf))
              omega (sample-omega k2 args v)
              logxi (log-density-estimate args v omega)]
          {:trace (make-enc-trace gf args addr v logxi omega) :weight logxi})
        ;; Unconstrained: equivalent to simulate, weight 0.
        {:trace (p/simulate gf args) :weight ZERO})))

  p/IAssess
  (assess [gf args choices]
    (let [v (addr-value choices addr)]
      (when (nil? v)
        (throw (ex-info (str "assess: encapsulated address " addr
                             " not found in provided choices.")
                        {:addr addr})))
      (let [[_ k2] (rng/split (enc-key gf))
            omega (sample-omega k2 args v)
            logxi (log-density-estimate args v omega)]
        {:retval v :weight logxi})))

  p/IPropose
  (propose [gf args]
    (let [trace (p/simulate gf args)]
      {:choices (:choices trace) :weight (:score trace) :retval (:retval trace)}))

  p/IProject
  (project [gf trace selection]
    ;; The whole estimated density lives at the single address; reuse the
    ;; stored score (= logxi at the recorded omega). project-all = score,
    ;; project-none = 0.
    (if (and selection (sel/selected? selection addr))
      (:score trace)
      ZERO))

  p/IUpdate
  (update [gf trace constraints]
    (let [args (:args trace)
          old-v (addr-value (:choices trace) addr)
          constraint (cm/get-submap constraints addr)]
      (if (and (cm/has-value? constraint)
               (not (same-value? (cm/get-value constraint) old-v)))
        ;; Genuine move: new observed value → resample omega, pay logxi'−old.
        (let [new-v (cm/get-value constraint)
              [_ k2] (rng/split (enc-key gf))]
          (resample-result gf args addr new-v (:score trace)
                           (cm/set-value cm/EMPTY addr old-v) k2))
        ;; No change at the observed address: reuse omega → weight 0.
        {:trace trace :weight ZERO :discard cm/EMPTY})))

  p/IRegenerate
  (regenerate [gf trace selection]
    (let [args (:args trace)]
      (if (and selection (sel/selected? selection addr))
        ;; Selected: propose a fresh observed value from the marginal, with a
        ;; fresh estimate. The §4.5 estimator regenerate ratio logxi'−logxi_old
        ;; replaces the exact-density "resample-from-prior ⇒ weight 0" because
        ;; the marginal proposal density is itself only estimated.
        (let [[k1 k2] (rng/split (enc-key gf))
              new-v (sample-value k1 args)]
          (dissoc (resample-result gf args addr new-v (:score trace) cm/EMPTY k2)
                  :discard))
        ;; Unselected: reuse omega → weight 0.
        {:trace trace :weight ZERO})))

  p/IUpdateWithArgs
  (update-with-args [gf trace new-args argdiffs constraints]
    (let [old-args (:args trace)
          old-v (addr-value (:choices trace) addr)
          constraint (cm/get-submap constraints addr)
          value-changed (and (cm/has-value? constraint)
                             (not (same-value? (cm/get-value constraint) old-v)))
          new-v (if (cm/has-value? constraint) (cm/get-value constraint) old-v)
          ;; The argdiff is a trusted caller hint (Gen.jl model): no-change ⇒
          ;; trust unchanged. Otherwise compare the ACTUAL args, so a
          ;; conservative :unknown at genuinely-unchanged args stays a no-op
          ;; (weight 0) rather than paying spurious estimator noise — i.e.
          ;; update-with-args at x'=x reduces to update.
          args-changed (and (not (diff/no-change? argdiffs))
                            (not (same-args? old-args new-args)))]
      (if (or args-changed value-changed)
        ;; Pseudo-marginal move generator: resample omega under new args at the
        ;; (possibly new) value, weight = logxi'(new-args) − old-score.
        (let [[_ k2] (rng/split (enc-key gf))
              discard (if value-changed (cm/set-value cm/EMPTY addr old-v) cm/EMPTY)]
          (resample-result gf new-args addr new-v (:score trace) discard k2))
        ;; Caller asserts no arg change and the value is unchanged: reuse omega.
        {:trace trace :weight ZERO :discard cm/EMPTY})))

  p/IUpdateWithDiffs
  (update-with-diffs [gf trace constraints argdiffs]
    (p/update-with-args gf trace (:args trace) argdiffs constraints))

  p/IHasArgumentGrads
  (has-argument-grads [_] nil))

(defn encapsulated
  "Construct a generative function with encapsulated randomness (§4.5).

   opts:
     :addr                  keyword — the single observed address.
     :sample-value          (fn [key args] -> v)        draws v ~ p(.; args).
     :sample-omega          (fn [key args v] -> omega)  draws encapsulated randomness.
     :log-density-estimate  (fn [args v omega] -> logxi) DETERMINISTIC given omega;
                            unbiased: E_omega[exp logxi] = p(v; args).

   Returns an EncapsulatedGF implementing the full GFI with score = log xi and
   trace.omega = omega. Use dyn/with-key / dyn/auto-key for PRNG control."
  [{:keys [addr sample-value sample-omega log-density-estimate]}]
  (assert (keyword? addr) "encapsulated: :addr must be a keyword")
  (assert (fn? sample-value) "encapsulated: :sample-value must be a fn")
  (assert (fn? sample-omega) "encapsulated: :sample-omega must be a fn")
  (assert (fn? log-density-estimate) "encapsulated: :log-density-estimate must be a fn")
  (->EncapsulatedGF addr sample-value sample-omega log-density-estimate))

;; ---------------------------------------------------------------------------
;; Gaussian log-density helper (estimator-internal; the TEST oracle is an
;; independent JS implementation, never this).
;; ---------------------------------------------------------------------------

(defn- gauss-logpdf
  "logN(x; mu, sigma), elementwise over MLX arrays. sigma is an MLX scalar."
  [x mu sigma]
  (mx/subtract
   (mx/subtract (mx/scalar (* -0.5 LOG-2PI)) (mx/log sigma))
   (mx/multiply (mx/scalar 0.5)
                (mx/square (mx/divide (mx/subtract x mu) sigma)))))

;; ---------------------------------------------------------------------------
;; Estimator library 1 — marginalized Gaussian (black-box stochastic code)
;;
;; The observed datum vector y of length n is generated as y_i = z_i + noise_i
;; with z_i ~ N(theta, tau^2), noise_i ~ N(0, sigma^2). The marginal density
;;   p(y_i; theta) = N(y_i; theta, sqrt(tau^2 + sigma^2))
;; is treated as a black box: it is estimated, per factor independently, by
;; K-sample importance sampling over the nuisance z with the prior N(theta,tau)
;; as the proposal (so weights reduce to the likelihood N(y_i; z, sigma)):
;;   xi_i = (1/K) sum_k N(y_i; z_{i,k}, sigma),   z_{i,k} ~iid N(theta, tau)
;;   log xi = sum_i logmeanexp_k logN(y_i; z_{i,k}, sigma)
;; INDEPENDENT z per factor i (shape [n,K]) is essential: a shared nuisance
;; correlates the factors and biases the product (Andrieu-Roberts pitfall).
;; ---------------------------------------------------------------------------

(defn marginalized-gaussian
  "Encapsulated marginalized-Gaussian likelihood (§4.5 black-box example).

   opts: {:addr kw, :n int, :tau num, :sigma num, :k int (IS samples)}
   args to the returned GF: [theta] (an MLX scalar — the inferred location).

   Returns {:gf EncapsulatedGF
            :exact-log-density (fn [args y] -> logp scalar)  ;; the true marginal
            :marginal-sigma S}                               ;; sqrt(tau^2+sigma^2)"
  [{:keys [addr n tau sigma k] :or {addr :y k 16}}]
  (let [tau-mx (mx/scalar (double tau))
        sigma-mx (mx/scalar (double sigma))
        S (js/Math.sqrt (+ (* tau tau) (* sigma sigma)))
        S-mx (mx/scalar S)
        log-k (mx/scalar (js/Math.log k))
        theta-of (fn [args] (let [t (first args)]
                              (if (mx/array? t) t (mx/scalar (double t)))))
        sample-value
        (fn [key args]
          (mx/add (theta-of args) (mx/multiply S-mx (rng/normal key [n]))))
        sample-omega
        (fn [key args _v]
          ;; z[n,K] ~ N(theta, tau): independent nuisance per (factor, sample).
          (mx/add (theta-of args) (mx/multiply tau-mx (rng/normal key [n k]))))
        log-density-estimate
        (fn [_args y z]
          (let [y2 (mx/reshape y [n 1])
                lp (gauss-logpdf y2 z sigma-mx)            ;; [n,K]
                per-i (mx/subtract (mx/logsumexp lp [1]) log-k)] ;; [n]
            (mx/sum per-i)))
        exact-log-density
        (fn [args y] (mx/sum (gauss-logpdf y (theta-of args) S-mx)))]
    {:gf (encapsulated {:addr addr
                        :sample-value sample-value
                        :sample-omega sample-omega
                        :log-density-estimate log-density-estimate})
     :exact-log-density exact-log-density
     :marginal-sigma S}))

;; ---------------------------------------------------------------------------
;; Estimator library 2 — finite mixture with Monte-Carlo-integrated index
;; ("finite mixture with unknown Z": the normalizing sum over the component
;; index is estimated by sampling the index rather than summing it).
;;
;;   p(y) = sum_k pi_k N(y; mu_k, sigma_k)
;;   xi   = (1/K) sum_i N(y; mu_{k_i}, sigma_{k_i}),  k_i ~iid Categorical(pi)
;; ---------------------------------------------------------------------------

(defn mixture-density
  "Encapsulated finite-mixture density (§4.5 unknown-Z example).

   opts: {:addr kw, :weights [num...], :means [num...], :sigmas [num...], :k int}
   args to the returned GF: [] (a fixed mixture; the value y is a scalar).

   Returns {:gf EncapsulatedGF
            :exact-log-density (fn [args y] -> logp scalar)}"
  [{:keys [addr weights means sigmas k] :or {addr :y k 32}}]
  (let [m (count weights)
        wsum (reduce + weights)
        log-pi (mx/array (mapv #(js/Math.log (/ % wsum)) weights))  ;; normalized [m]
        logits (mx/array (mapv #(js/Math.log %) weights))           ;; categorical logits [m]
        mu (mx/array (mapv double means))                           ;; [m]
        sig (mx/array (mapv double sigmas))                         ;; [m]
        log-k (mx/scalar (js/Math.log k))
        cat-rows (fn [key rows]
                   ;; sample `rows` iid category indices: broadcast logits to
                   ;; [rows,m], categorical samples along last axis → [rows].
                   (rng/categorical key (mx/broadcast-to (mx/reshape logits [1 m]) [rows m])))
        ensure-y (fn [v] (if (mx/array? v) v (mx/scalar (double v))))
        sample-value
        (fn [key _args]
          (let [[k1 k2] (rng/split key)
                idx (cat-rows k1 1)                       ;; [1]
                mu-i (mx/take-idx mu idx)                 ;; [1]
                sig-i (mx/take-idx sig idx)]              ;; [1]
            (mx/reshape (mx/add mu-i (mx/multiply sig-i (rng/normal k2 [1]))) [])))
        sample-omega
        (fn [key _args _v] (cat-rows key k))             ;; K indices [K]
        log-density-estimate
        (fn [_args y idx]
          (let [yv (ensure-y y)
                mu-k (mx/take-idx mu idx)                 ;; [K]
                sig-k (mx/take-idx sig idx)               ;; [K]
                lp (gauss-logpdf yv mu-k sig-k)]          ;; [K]
            (mx/subtract (mx/logsumexp lp [0]) log-k)))
        exact-log-density
        (fn [_args y]
          (let [yv (ensure-y y)
                lp (mx/add log-pi (gauss-logpdf yv mu sig))]  ;; [m]
            (mx/logsumexp lp [0])))]
    {:gf (encapsulated {:addr addr
                        :sample-value sample-value
                        :sample-omega sample-omega
                        :log-density-estimate log-density-estimate})
     :exact-log-density exact-log-density}))

;; ---------------------------------------------------------------------------
;; Pseudo-marginal Metropolis-Hastings (§4.5 / Andrieu-Roberts 2009)
;;
;; Infers a parameter theta whose likelihood p(y | theta) is available only as
;; an unbiased estimate xi from an EncapsulatedGF (y constrained, theta = arg).
;; Each proposed theta' draws a fresh xi'; the CURRENT state reuses its stored
;; xi (the trace score + omega) as the acceptance denominator. The extended
;; (theta, omega) chain is reversible and the theta-marginal is the exact
;; posterior because E_omega[xi] = p(y|theta) (Eq 4.3).
;; ---------------------------------------------------------------------------

(defn pseudo-marginal-mh
  "Pseudo-marginal random-walk MH on a scalar parameter theta.

   opts:
     :enc-gf       EncapsulatedGF likelihood (args = [theta], observed value y).
     :y            observed value (MLX array) constrained at the GF's address.
     :theta0       initial theta (number).
     :log-prior    (fn [theta-num] -> log p(theta)) host-side prior log-density.
     :step         random-walk std (number).
     :samples      number of post-burn samples to collect.
     :burn         burn-in steps (default 0).
     :key          PRNG key (default fresh).

   Returns {:samples [theta-num ...] :accept-rate r}. The likelihood is never
   computed exactly — only the encapsulated estimator is used."
  [{:keys [enc-gf y theta0 log-prior step samples burn key]
    :or {burn 0}}]
  (assert (instance? EncapsulatedGF enc-gf)
          "pseudo-marginal-mh: :enc-gf must be an EncapsulatedGF (see `encapsulated`)")
  (let [addr (:addr enc-gf)
        rk (rng/ensure-key (or key (rng/fresh-key)))
        obs (cm/set-value cm/EMPTY addr y)
        ;; Initialise the trace at theta0 (draws the initial stored xi/omega).
        keyed (fn [k] (vary-meta enc-gf assoc :genmlx.dynamic/key k))
        [k0 rk0] (rng/split rk)
        init-trace (:trace (p/generate (keyed k0) [(mx/scalar (double theta0))] obs))
        n-total (+ burn samples)]
    (loop [cur-trace init-trace
           theta theta0
           i 0
           rk rk0
           acc (transient [])
           n-acc 0]
      (if (>= i n-total)
        {:samples (persistent! acc)
         :accept-rate (/ n-acc (double n-total))}
        (let [[k-prop k-move k-acc rk'] (rng/split-n rk 4)
              ;; Random-walk proposal theta' = theta + step * N(0,1).
              theta' (+ theta (* step (mx/item (rng/normal k-prop []))))
              ;; Pseudo-marginal move: change the arg theta→theta', redraw omega.
              ;; weight = logxi'(theta') − logxi_old (the stored auxiliary).
              {prop-trace :trace weight :weight}
              (p/update-with-args (keyed k-move) cur-trace
                                  [(mx/scalar theta')] :unknown cm/EMPTY)
              log-alpha (+ (mx/item weight)
                           (- (log-prior theta') (log-prior theta)))
              u (mx/item (rng/uniform k-acc []))
              accept? (< (js/Math.log u) log-alpha)
              ;; On reject KEEP the old trace (with its stored old xi); on accept
              ;; carry the proposed trace forward (its xi' becomes the next aux).
              next-trace (if accept? prop-trace cur-trace)
              next-theta (if accept? theta' theta)]
          (recur next-trace next-theta (inc i) rk'
                 (if (>= i burn) (conj! acc next-theta) acc)
                 (+ n-acc (if accept? 1 0))))))))
