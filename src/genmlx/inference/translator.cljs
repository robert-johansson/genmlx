(ns genmlx.inference.translator
  "General trace translators — Cusumano-Towner 2020 PhD thesis, §3.6-3.7.

   A trace translator maps a trace of model P1 to a trace of model P2 whose
   choice dictionaries live in different spaces. It is the machinery behind
   reversible-jump MCMC, coarse-to-fine SMC, and bridging between discrete and
   continuous representations. The involutive-MCMC kernel in genmlx.inference.mcmc
   is the SYMMETRIC special case (P1=P2, Q1=Q2, h an involution, §3.7); this
   namespace is the general case (Def 3.6.1, Eq 3.12).

   ## Structure (Def 3.6.1)

   A deterministic trace translator is built from
     - p2  : the target generative function,
     - q1  : a forward auxiliary proposal (a GF taking [in-choices] -> rho1),
     - q2  : a backward auxiliary proposal (a GF taking [out-choices] -> rho2),
     - h   : a bijection (tau1, rho1) <-> (tau2, rho2).
   The bijection uses the same convention as the involutive kernel:
     h : (fn [in-choices aux-choices]
            -> {:trace out-choices :aux out-aux :log-det-jacobian (optional)})

   ## Importance weight (Eq 3.12)

     w = [ p2(tau2) q2(rho2; tau2) ] / [ p1(tau1) q1(rho1; tau1) ] * |det J_h|

   In log space (what apply-translator computes):

     log w = (score2 - score1) + (q2-score - q1-score) + log|det J_h|

   where score1 is the input trace's score, score2 is generate(p2, args2,
   tau2).score, q1-score is the forward proposal log-density (DENOMINATOR), and
   q2-score is the backward proposal log-density (NUMERATOR). log|det J_h| is the
   Jacobian of the bijection over the CONTINUOUS coordinates; discrete
   coordinates contribute no Jacobian column. It comes from h (:log-det-jacobian),
   else from the translator's :jacobian-fn (typically built with jacobian-logdet,
   AD via mx/grad), else 0 (a volume-preserving move — discrete/copy moves)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.mlx.constants :refer [ZERO]]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Jacobian via automatic differentiation (mx/grad)
;;
;; The AD Jacobian is built row-by-row with mx/grad; its log|det| comes from
;; mx/logabsdet (QR-based, general — correct for the non-symmetric Jacobians
;; bijections produce, where the SPD-only mx/spd-logdet would be silently wrong;
;; genmlx-rqp9).
;; ---------------------------------------------------------------------------

(defn jacobian-matrix
  "The n x n Jacobian of g : R^n -> R^n at point x ([n]-shaped MLX vector),
   row i = d g_i / d x (computed by mx/grad on the scalar component g_i)."
  [g x n]
  (mx/stack
   (mapv (fn [i] ((mx/grad (fn [xv] (mx/index (g xv) i))) x))
         (range n))))

(defn jacobian-logdet
  "log|det J_g(x)| (MLX scalar) for a bijective numeric transform
   g : R^n -> R^n at x. The AD Jacobian (mx/grad on g) is fed to mx/logabsdet
   (QR, general / orientation-safe).

   SPARSITY (thesis §3.6.2): pass g and x restricted to the TRANSFORMED
   continuous coordinates only. Coordinates that h copies unchanged contribute
   identity columns (determinant 1), so excluding them leaves log|det| unchanged
   while reducing the matrix from m x m to (m-c) x (m-c) — O((m-c)^3) instead of
   O(m^3). sparse-jacobian-logdet packages this from the full transform."
  [g x n]
  (mx/logabsdet (jacobian-matrix g x n)))

(defn sparse-jacobian-logdet
  "Sparsity-aware (thesis §3.6.2) log|det J| (MLX scalar). `g-full` : R^m -> R^m
   is the full transform and `x` the full input point [m]; `transformed-idxs`
   lists the c' indices of the coordinates h actually transforms. The
   determinant is taken over the (c' x c') submatrix of the Jacobian at those
   indices (the copied coordinates are identity rows/cols, determinant 1).
   Equals jacobian-logdet on the full matrix but over the smaller block."
  [g-full x transformed-idxs]
  (let [m (first (mx/shape x))                  ;; vector length (not rank)
        full-j (jacobian-matrix g-full x m)
        idx (mx/array (vec transformed-idxs) mx/int32)
        sub (mx/take-idx (mx/take-idx full-j idx 0) idx 1)]  ;; rows then cols
    (mx/logabsdet sub)))

;; ---------------------------------------------------------------------------
;; read / write / copy — introspection API for writing bijections h
;; ---------------------------------------------------------------------------

(defn read-choice
  "Value at addr in a choicemap (the bijection's input side)."
  [choices addr]
  (cm/get-value (cm/get-submap choices addr)))

(def read-aux
  "Value at addr in an auxiliary choicemap. Alias of read-choice (same shape)."
  read-choice)

(defn write-choice
  "Set a TRANSFORMED output choice at addr. (Conceptually @write — its value
   is a function of the inputs, so it contributes a Jacobian column.)"
  [choices addr value]
  (cm/set-value choices addr value))

(defn copy-choice
  "Copy the value at `from-addr` of `src` into `to-addr` of `dst` UNCHANGED.
   (Conceptually @copy — an identity column, excluded from the Jacobian by
   sparse-jacobian-logdet.) from-addr defaults to to-addr."
  ([dst src addr] (copy-choice dst src addr addr))
  ([dst src to-addr from-addr]
   (cm/set-value dst to-addr (read-choice src from-addr))))

(def write-aux
  "Set a backward-auxiliary choice. Alias of write-choice."
  write-choice)

;; ---------------------------------------------------------------------------
;; The translator
;; ---------------------------------------------------------------------------

(defrecord DeterministicTraceTranslator [p2 q1 q2 h jacobian-fn volume-preserving?])

(defn trace-translator
  "Construct a general (deterministic) trace translator (Def 3.6.1).

   opts:
     :p2                target generative function (required).
     :q1                forward auxiliary proposal GF [in-choices]->rho1 (nil ok).
     :q2                backward auxiliary proposal GF [out-choices]->rho2 (nil ok).
     :h                 bijection (fn [in-choices aux-choices]
                          -> {:trace out-choices :aux out-aux
                              :log-det-jacobian (optional)}) (required).
     :jacobian-fn       optional (fn [in-choices aux-choices] -> log|det J|)
                        used when h omits :log-det-jacobian (e.g. via
                        jacobian-logdet).
     :volume-preserving? assert that this move has unit Jacobian (log|det J| = 0)
                        — discrete or copy-only moves. Required when neither h
                        nor :jacobian-fn supplies a Jacobian: a continuous
                        transform with NO declared Jacobian and this flag unset
                        is an error (a missing Jacobian must be explicit, never
                        a silent 0)."
  [{:keys [p2 q1 q2 h jacobian-fn volume-preserving?]}]
  (assert (some? p2) "trace-translator: :p2 (target model) is required")
  (assert (fn? h) "trace-translator: :h (bijection) must be a fn")
  (->DeterministicTraceTranslator p2 q1 q2 h jacobian-fn (boolean volume-preserving?)))

(defn- log-det-of
  "Resolve log|det J| for one application: h's value, else the translator's
   :jacobian-fn, else 0 IFF the translator is declared :volume-preserving?.
   A non-volume-preserving move with no declared Jacobian throws — a missing
   change-of-variables term must be explicit, never a silent 0 (CLAUDE.md:
   no no-op that lies about what it does)."
  [tt h-result in-choices aux-choices]
  (cond
    (contains? h-result :log-det-jacobian)
    (let [j (:log-det-jacobian h-result)] (if (mx/array? j) j (mx/scalar j)))
    (:jacobian-fn tt)
    (let [j ((:jacobian-fn tt) in-choices aux-choices)] (if (mx/array? j) j (mx/scalar j)))
    (:volume-preserving? tt) ZERO
    :else (throw (ex-info (str "trace-translator: no Jacobian for a non-"
                               "volume-preserving move. Return :log-det-jacobian "
                               "from h, set :jacobian-fn (e.g. via jacobian-logdet), "
                               "or pass :volume-preserving? true for discrete/copy-only moves.")
                          {:genmlx/error :translator-missing-jacobian}))))

(defn apply-translator
  "Apply trace translator `tt` to input trace `in-trace`, producing a trace of
   the target model p2 with new args `args2`. Returns
     {:trace out-trace :weight log-w :aux rho2 :log-det-jacobian ldj}
   where log-w is the Eq 3.12 importance weight (log domain). `key` supplies
   entropy for the forward auxiliary proposal.

   PRECONDITION: `in-trace` must be JOINT-scored — the Eq 3.12 weight
   differences score1 against score2, so a :marginal trace (latents pinned at
   the posterior mean) would silently corrupt it. Asserted (genmlx-540f class);
   all internal call sites already strip the analytical path."
  [tt in-trace args2 key]
  (tr/assert-joint! in-trace :apply-translator)
  (let [{:keys [p2 q1 q2 h]} tt
        in-choices (:choices in-trace)
        [k1 k2] (rng/split (rng/ensure-key key))
        ;; 1. forward auxiliary rho1 ~ q1(in-choices)   (denominator)
        fwd (if q1 (p/propose (dyn/with-key q1 k1) [in-choices])
                   {:choices cm/EMPTY :weight ZERO})
        rho1 (:choices fwd)
        q1-score (:weight fwd)
        ;; 2. apply the bijection h
        hres (h in-choices rho1)
        out-choices (:trace hres)
        rho2 (or (:aux hres) cm/EMPTY)
        ;; 3. target trace, fully constrained -> score2 (= generate weight).
        ;; Strip the L3 analytical path: a conjugate p2 would otherwise return a
        ;; :marginal score with latents pinned at the posterior mean, corrupting
        ;; the importance weight (genmlx-540f). Translators need joint scores.
        gen (p/generate (dyn/with-key (dyn/strip-analytical-path p2) k2) args2 out-choices)
        out-trace (:trace gen)
        score2 (:score out-trace)
        ;; 4. backward auxiliary density q2(rho2; out-choices)   (numerator)
        q2-score (if q2 (:weight (p/assess (dyn/auto-key q2) [out-choices] rho2)) ZERO)
        ;; 5. Jacobian of the continuous part of h
        ldj (log-det-of tt hres in-choices rho1)
        ;; 6. Eq 3.12 weight in log domain
        score1 (:score in-trace)
        log-w (mx/add (mx/subtract score2 score1)
                      (mx/subtract q2-score q1-score)
                      ldj)]
    {:trace out-trace :weight log-w :aux rho2 :log-det-jacobian ldj}))

(defn translator-weight
  "The Eq 3.12 importance weight (an MLX scalar) of applying `tt` to `in-trace`
   at target args `args2`. Convenience wrapper over apply-translator."
  [tt in-trace args2 key]
  (:weight (apply-translator tt in-trace args2 key)))

;; ---------------------------------------------------------------------------
;; Reversible-jump MCMC (§3.7.4) — structure-changing moves via translators
;; ---------------------------------------------------------------------------

(defn- applicable-moves
  "Indices of moves whose :applicable? predicate accepts `trace` (a move with no
   predicate is always applicable)."
  [moves trace]
  (vec (keep-indexed (fn [i {pred :applicable?}]
                       (when (or (nil? pred) (pred trace)) i))
                     moves)))

(defn reversible-jump-mh-step
  "One reversible-jump MH step. `moves` is a vector of maps
     {:translator tt :args2 args :applicable? (fn [trace] -> bool, optional)}
   each a translator from the current model P to P (possibly changing
   structure). An APPLICABLE move is chosen uniformly, the translator applied,
   and the proposed trace accepted with probability min(1, exp(log-w + corr))
   where log-w is the Eq 3.12 weight (auxiliary proposals + Jacobian) and corr =
   log(#applicable-at-current) - log(#applicable-at-proposed) corrects for the
   discrete move-selection probability (0 for split/merge, where exactly one
   move is applicable in each state). Returns {:trace t' :accepted? bool :move i}."
  [current-trace moves key]
  (let [[k-sel k-app k-acc] (rng/split-n (rng/ensure-key key) 3)
        appl (applicable-moves moves current-trace)
        n-cur (count appl)]
    (if (zero? n-cur)
      {:trace current-trace :accepted? false :move nil}
      (let [i (nth appl (int (mx/realize (rng/randint k-sel 0 n-cur []))))
            {:keys [translator args2]} (nth moves i)
            {:keys [trace weight]} (apply-translator translator current-trace
                                                     (or args2 (:args current-trace)) k-app)
            n-prop (count (applicable-moves moves trace))
            ;; If the proposed state has no applicable moves the reverse move is
            ;; unselectable (reverse prob 0), so the Hastings ratio is 0 and the
            ;; move MUST be rejected; corr = log(n-cur) - log(0) = +Inf in the
            ;; denominator, i.e. -Inf correction (genmlx-jtou). corr=0 here could
            ;; wrongly accept and break detailed balance.
            corr (if (pos? n-prop) (- (js/Math.log n-cur) (js/Math.log n-prop)) ##-Inf)
            log-w (+ (mx/realize weight) corr)
            accept? (u/accept-mh? log-w k-acc)]
        {:trace (if accept? trace current-trace) :accepted? accept? :move i}))))

(defn reversible-jump-mh
  "Reversible-jump MCMC. Returns {:samples [trace ...] :accept-rate r}.

   opts: {:moves [{:translator tt :args2 args} ...] :samples N :burn B
          :thin T :key prng-key}
   init-trace: the starting model trace (e.g. from p/generate with observations)."
  [{:keys [moves samples burn thin key] :or {burn 0 thin 1}} init-trace]
  (let [rk (rng/ensure-key (or key (rng/fresh-key)))]
    (loop [t init-trace i 0 rk rk acc (transient []) n-acc 0 n-steps 0]
      (if (>= i (+ burn (* samples thin)))
        {:samples (persistent! acc)
         :accept-rate (if (pos? n-steps) (/ n-acc (double n-steps)) 0.0)}
        (let [[k rk'] (rng/split rk)
              {:keys [trace accepted?]} (reversible-jump-mh-step t moves k)
              keep? (and (>= i burn) (zero? (mod (- i burn) thin)))]
          (recur trace (inc i) rk'
                 (if keep? (conj! acc trace) acc)
                 (+ n-acc (if accepted? 1 0))
                 (inc n-steps)))))))

;; ---------------------------------------------------------------------------
;; Coarse-to-fine SMC (§3.6.4) — bridge a sequence of models with translators
;; ---------------------------------------------------------------------------

(defn- logmeanexp-js
  "log( mean( exp xs ) ) over a JS vector of log-values (numerically stable)."
  [xs]
  (let [m (reduce max xs)]
    (+ m (js/Math.log (/ (reduce + (map #(js/Math.exp (- % m)) xs))
                         (double (count xs)))))))

(defn coarse-to-fine-smc
  "SMC over a sequence of models P_0 (coarse) -> ... -> P_T (fine) bridged by
   trace translators (thesis §3.6.4).

   opts:
     :models       [P_0 ... P_T]                  (T+1 generative functions)
     :stage-args   [args_0 ... args_T]            (args per model; default [] each)
     :stage-obs    [obs_0 ... obs_T]              (constraints per stage; default EMPTY)
     :translators  [T_{0->1} ... T_{(T-1)->T}]    (T translators)
     :n-particles  N
     :key          prng-key
     :resample?    resample between stages (default true)

   Returns {:particles [trace ...] :log-weights [number ...] :log-ml number}.
   (:log-weights and :log-ml are realized JS numbers, not MLX scalars — the SMC
   loop materializes per-particle weights for host-side resampling/logmeanexp.)
   log-ml is the accumulated unbiased marginal-likelihood estimate for the fine
   model: at stage 0 the importance weights are P_0.generate weights; each
   translator application multiplies in its Eq 3.12 weight; resampling adds the
   usual logmeanexp normaliser increments."
  [{:keys [models stage-args stage-obs translators n-particles key resample?]
    :or {resample? true}}]
  (let [n n-particles
        T (dec (count models))
        args-of (fn [s] (nth stage-args s []))
        obs-of (fn [s] (or (nth stage-obs s nil) cm/EMPTY))
        rk (rng/ensure-key (or key (rng/fresh-key)))
        ;; Stage 0: import N particles from the coarse model.
        [k0 rk] (rng/split rk)
        keys0 (rng/split-n k0 n)
        ;; strip the L3 analytical path so a conjugate coarse model imports true
        ;; joint-scored particles (not posterior-mean-pinned :marginal traces).
        p0 (dyn/strip-analytical-path (nth models 0))
        init (mapv (fn [kk]
                     (let [{:keys [trace weight]} (p/generate (dyn/with-key p0 kk)
                                                              (args-of 0) (obs-of 0))]
                       {:trace trace :lw (mx/realize weight)}))
                   keys0)]
    (loop [stage 0
           parts init
           log-ml 0.0
           rk rk]
      (if (>= stage T)
        {:particles (mapv :trace parts)
         :log-weights (mapv :lw parts)
         :log-ml (+ log-ml (logmeanexp-js (mapv :lw parts)))}
        (let [;; normalise + (optionally) resample before bridging to the next model
              lws (mapv :lw parts)
              [k-res rk1] (rng/split rk)
              ;; The SMC log-Z increment (logmeanexp over the current weights) is
              ;; only valid when paired with a weight RESET — i.e. when a resample
              ;; actually happens. Without resampling, weights accumulate and a
              ;; SINGLE terminal logmeanexp over the fully-accumulated weights is
              ;; the unbiased single-path IS estimate; adding a per-stage
              ;; increment too would double-count (genmlx-jtou).
              [parts' log-ml']
              (if resample?
                ;; systematic-resample takes MLX log-weight scalars; our lws are
                ;; realized JS numbers, so box them. Reset weights to 0 and bank
                ;; the normaliser increment.
                (let [idxs (u/systematic-resample (mapv mx/scalar lws) n k-res)]
                  [(mapv (fn [j] {:trace (:trace (nth parts j)) :lw 0.0}) idxs)
                   (+ log-ml (logmeanexp-js lws))])
                [parts log-ml])
              ;; bridge every particle to the next model via the translator
              tt (nth translators stage)
              keysT (rng/split-n rk1 (inc n))
              bridged (mapv (fn [j]
                              (let [{:keys [trace lw]} (nth parts' j)
                                    {t2 :trace w :weight}
                                    (apply-translator tt trace (args-of (inc stage))
                                                      (nth keysT j))]
                                {:trace t2 :lw (+ lw (mx/realize w))}))
                            (range n))]
          (recur (inc stage) bridged log-ml' (nth keysT n)))))))
