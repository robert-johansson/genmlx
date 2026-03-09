(ns genmlx.inference.conjugate
  "Conjugate prior middleware for analytical parameter marginalization.

   Marginalizes model parameters (not temporal states) via conjugate
   prior-likelihood pairs. Each observation analytically updates the
   posterior over the parameter and contributes a closed-form marginal LL.

   Three conjugate families:

   1. Normal-Normal (NN): μ ~ N(m, τ²), x ~ N(μ, σ²)
      Known observation variance, unknown mean.

   2. Beta-Binomial (BB): p ~ Beta(α, β), x ~ Bernoulli(p)
      Binary observations with unknown probability.

   3. Gamma-Poisson (GP): λ ~ Gamma(α, β), x ~ Poisson(λ)
      Count observations with unknown rate.

   Each pair provides:
   - defdist distributions (work under standard or conjugate handler)
   - Pure update function (posterior + marginal LL)
   - make-*-dispatch for use with wrap-analytical

   Observation distributions carry a :prior-addr linking them to
   their conjugate prior site. Multiple priors compose naturally
   via wrap-analytical — nil-fallthrough ensures each handler only
   processes its own observations.

   State keys (shared across families):
   - :conjugate-posteriors  {addr -> posterior-map}
   - :conjugate-ll          accumulated marginal LL"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.dist.macros :refer [defdist]]))

;; ---------------------------------------------------------------------------
;; Pure conjugate update functions (Level 1)
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI 1.8378770664093453)

(defn nn-update
  "Normal-Normal conjugate update.
   prior:   {:mean array, :var array} — prior on μ
   obs:     observed value (array)
   obs-var: known observation variance σ² (array)
   mask:    observation mask (1=observed, 0=missing)

   Returns {:posterior {:mean :var}, :ll marginal LL}.

   Posterior: N(m', τ'²) where
     τ'² = 1 / (1/τ² + 1/σ²)
     m'  = τ'² · (m/τ² + x/σ²)
   Marginal:  N(x | m, τ² + σ²)"
  [prior obs obs-var mask]
  (let [{:keys [mean var]} prior
        ;; Marginal LL: N(obs | prior-mean, prior-var + obs-var)
        S (mx/add var obs-var)
        innov (mx/subtract obs mean)
        ll (mx/multiply (mx/scalar -0.5)
             (mx/add (mx/scalar LOG-2PI)
               (mx/add (mx/log S)
                 (mx/divide (mx/multiply innov innov) S))))
        ;; Posterior
        inv-prior (mx/divide (mx/scalar 1.0) var)
        inv-obs (mx/divide (mx/scalar 1.0) obs-var)
        new-var (mx/divide (mx/scalar 1.0) (mx/add inv-prior inv-obs))
        new-mean (mx/multiply new-var
                   (mx/add (mx/multiply inv-prior mean)
                           (mx/multiply inv-obs obs)))
        ;; Masked: if mask=0, posterior unchanged, ll=0
        final-mean (mx/add (mx/multiply mask new-mean)
                           (mx/multiply (mx/subtract (mx/scalar 1.0) mask) mean))
        final-var (mx/add (mx/multiply mask new-var)
                          (mx/multiply (mx/subtract (mx/scalar 1.0) mask) var))]
    {:posterior {:mean final-mean :var final-var}
     :ll (mx/multiply mask ll)}))

(defn bb-update
  "Beta-Binomial conjugate update.
   prior: {:alpha array, :beta array} — Beta prior on p
   obs:   observed value (0 or 1, as float array)
   mask:  observation mask (1=observed, 0=missing)

   Returns {:posterior {:alpha :beta}, :ll marginal LL}.

   Posterior: Beta(α + x, β + 1 - x)
   Marginal:  p(x) = x·α/(α+β) + (1-x)·β/(α+β)"
  [prior obs mask]
  (let [{:keys [alpha beta]} prior
        sum-ab (mx/add alpha beta)
        ;; Marginal LL: log(x·α + (1-x)·β) - log(α + β)
        ll (mx/subtract
             (mx/log (mx/add (mx/multiply obs alpha)
                             (mx/multiply (mx/subtract (mx/scalar 1.0) obs) beta)))
             (mx/log sum-ab))
        ;; Posterior
        one-minus-obs (mx/subtract (mx/scalar 1.0) obs)
        new-alpha (mx/add alpha (mx/multiply mask obs))
        new-beta (mx/add beta (mx/multiply mask one-minus-obs))]
    {:posterior {:alpha new-alpha :beta new-beta}
     :ll (mx/multiply mask ll)}))

(defn gp-update
  "Gamma-Poisson conjugate update.
   prior: {:shape array, :rate array} — Gamma prior on λ (shape α, rate β)
   obs:   observed count (non-negative integer, as float array)
   mask:  observation mask (1=observed, 0=missing)

   Returns {:posterior {:shape :rate}, :ll marginal LL}.

   Posterior: Gamma(α + x, β + 1)
   Marginal:  NegBin(x | α, β/(β+1))"
  [prior obs mask]
  (let [{:keys [shape rate]} prior
        ;; Marginal LL: NegBin(x | r=α, p=β/(β+1))
        ;; log P(x) = lgamma(α+x) - lgamma(α) - lgamma(x+1) + α·log(β/(β+1)) + x·log(1/(β+1))
        bp1 (mx/add rate (mx/scalar 1.0))
        ll (-> (mx/lgamma (mx/add shape obs))
               (mx/subtract (mx/lgamma shape))
               (mx/subtract (mx/lgamma (mx/add obs (mx/scalar 1.0))))
               (mx/add (mx/multiply shape (mx/subtract (mx/log rate) (mx/log bp1))))
               (mx/add (mx/multiply obs (mx/negative (mx/log bp1)))))
        ;; Posterior
        new-shape (mx/add shape (mx/multiply mask obs))
        new-rate (mx/add rate mask)]
    {:posterior {:shape new-shape :rate new-rate}
     :ll (mx/multiply mask ll)}))

;; ---------------------------------------------------------------------------
;; Distributions (carry conjugate structure)
;; ---------------------------------------------------------------------------

;; -- Normal-Normal --

(defdist nn-prior
  "Normal prior on mean parameter: μ ~ N(prior-mean, prior-std²).
   Under standard handler: samples from prior.
   Under conjugate handler: returns posterior mean."
  [prior-mean prior-std]
  (sample [key]
    (dc/dist-sample (dist/gaussian prior-mean prior-std) key))
  (log-prob [v]
    (dc/dist-log-prob (dist/gaussian prior-mean prior-std) v)))

(defdist nn-obs
  "Normal observation linked to a Normal prior on the mean.
   prior-addr:  keyword address of the nn-prior site this updates.
   mu-value:    sampled/inferred mean (used under standard handler).
   obs-std:     known observation noise std dev.
   mask:        1=observed, 0=missing.

   Under standard handler: N(mu-value, obs-std).
   Under conjugate handler: analytical posterior update on prior-addr."
  [prior-addr mu-value obs-std mask]
  (sample [key]
    (dc/dist-sample (dist/gaussian mu-value obs-std) key))
  (log-prob [v]
    (mx/multiply mask
      (dc/dist-log-prob (dist/gaussian mu-value obs-std) v))))

;; -- Beta-Binomial --

(defdist bb-prior
  "Beta prior on probability parameter: p ~ Beta(alpha, beta-param).
   Under standard handler: samples from prior.
   Under conjugate handler: returns posterior mean α/(α+β)."
  [alpha beta-param]
  (sample [key]
    (dc/dist-sample (dist/beta-dist alpha beta-param) key))
  (log-prob [v]
    (dc/dist-log-prob (dist/beta-dist alpha beta-param) v)))

(defdist bb-obs
  "Bernoulli observation linked to a Beta prior.
   prior-addr: keyword address of the bb-prior site.
   p-value:    sampled/inferred probability (used under standard handler).
   mask:       1=observed, 0=missing.

   Under standard handler: Bernoulli(p-value).
   Under conjugate handler: analytical posterior update."
  [prior-addr p-value mask]
  (sample [key]
    (dc/dist-sample (dist/bernoulli p-value) key))
  (log-prob [v]
    (mx/multiply mask
      (dc/dist-log-prob (dist/bernoulli p-value) v))))

;; -- Gamma-Poisson --

(defdist gp-prior
  "Gamma prior on rate parameter: λ ~ Gamma(shape-param, rate-param).
   Under standard handler: samples from prior.
   Under conjugate handler: returns posterior mean α/β."
  [shape-param rate-param]
  (sample [key]
    (dc/dist-sample (dist/gamma-dist shape-param rate-param) key))
  (log-prob [v]
    (dc/dist-log-prob (dist/gamma-dist shape-param rate-param) v)))

(defdist gp-obs
  "Poisson observation linked to a Gamma prior.
   prior-addr:   keyword address of the gp-prior site.
   lambda-value: sampled/inferred rate (used under standard handler).
   mask:         1=observed, 0=missing.

   Under standard handler: Poisson(lambda-value).
   Under conjugate handler: analytical posterior update."
  [prior-addr lambda-value mask]
  (sample [key]
    (dc/dist-sample (dist/poisson lambda-value) key))
  (log-prob [v]
    (mx/multiply mask
      (dc/dist-log-prob (dist/poisson lambda-value) v))))

;; ---------------------------------------------------------------------------
;; Handler middleware (Level 2)
;; ---------------------------------------------------------------------------
;;
;; Each make-*-dispatch creates a dispatch map for wrap-analytical.
;; Posterior state lives in :conjugate-posteriors {addr -> posterior}.
;; Marginal LL accumulates in :conjugate-ll.
;;
;; The obs handler checks prior-addr from the distribution's params
;; against the target-addr captured in the dispatch closure.
;; Returns nil for non-matching addrs (falls through via wrap-analytical).

(defn- ll-shape
  "Infer the shape for LL initialization from a posterior value."
  [v]
  (mx/shape v))

(defn make-nn-dispatch
  "Create Normal-Normal conjugate dispatch map.

   target-addr: keyword address of the nn-prior site

   Returns dispatch map: {:nn-prior handler, :nn-obs handler}.
   Self-initializing: first encounter uses the prior's hyperparams."
  [target-addr]
  {:nn-prior
   (fn [state addr dist]
     (if (= addr target-addr)
       (let [{:keys [prior-mean prior-std]} (:params dist)
             posterior (or (get-in state [:conjugate-posteriors addr])
                          {:mean prior-mean
                           :var (mx/multiply prior-std prior-std)})]
         [(:mean posterior)
          (-> state
              (assoc-in [:conjugate-posteriors addr] posterior)
              (update :choices cm/set-value addr (:mean posterior)))])
       nil))

   :nn-obs
   (fn [state addr dist]
     (let [{:keys [prior-addr obs-std mask]} (:params dist)]
       (if (= prior-addr target-addr)
         (let [posterior (get-in state [:conjugate-posteriors target-addr])
               constraint (cm/get-submap (:constraints state) addr)
               obs (cm/get-value constraint)
               obs-var (mx/multiply obs-std obs-std)
               {:keys [posterior ll]} (nn-update posterior obs obs-var mask)]
           [obs (-> state
                    (assoc-in [:conjugate-posteriors target-addr] posterior)
                    (update :choices cm/set-value addr obs)
                    (update :conjugate-ll
                      #(mx/add (or % (mx/zeros (ll-shape (:mean posterior)))) ll)))])
         nil)))})

(defn make-bb-dispatch
  "Create Beta-Binomial conjugate dispatch map.

   target-addr: keyword address of the bb-prior site

   Returns dispatch map: {:bb-prior handler, :bb-obs handler}."
  [target-addr]
  {:bb-prior
   (fn [state addr dist]
     (if (= addr target-addr)
       (let [{:keys [alpha beta-param]} (:params dist)
             posterior (or (get-in state [:conjugate-posteriors addr])
                          {:alpha alpha :beta beta-param})
             ;; Posterior mean: α/(α+β)
             post-mean (mx/divide (:alpha posterior)
                                  (mx/add (:alpha posterior) (:beta posterior)))]
         [post-mean
          (-> state
              (assoc-in [:conjugate-posteriors addr] posterior)
              (update :choices cm/set-value addr post-mean))])
       nil))

   :bb-obs
   (fn [state addr dist]
     (let [{:keys [prior-addr mask]} (:params dist)]
       (if (= prior-addr target-addr)
         (let [posterior (get-in state [:conjugate-posteriors target-addr])
               constraint (cm/get-submap (:constraints state) addr)
               obs (cm/get-value constraint)
               {:keys [posterior ll]} (bb-update posterior obs mask)]
           [obs (-> state
                    (assoc-in [:conjugate-posteriors target-addr] posterior)
                    (update :choices cm/set-value addr obs)
                    (update :conjugate-ll
                      #(mx/add (or % (mx/zeros (ll-shape (:alpha posterior)))) ll)))])
         nil)))})

(defn make-gp-dispatch
  "Create Gamma-Poisson conjugate dispatch map.

   target-addr: keyword address of the gp-prior site

   Returns dispatch map: {:gp-prior handler, :gp-obs handler}."
  [target-addr]
  {:gp-prior
   (fn [state addr dist]
     (if (= addr target-addr)
       (let [{:keys [shape-param rate-param]} (:params dist)
             posterior (or (get-in state [:conjugate-posteriors addr])
                          {:shape shape-param :rate rate-param})
             ;; Posterior mean: α/β
             post-mean (mx/divide (:shape posterior) (:rate posterior))]
         [post-mean
          (-> state
              (assoc-in [:conjugate-posteriors addr] posterior)
              (update :choices cm/set-value addr post-mean))])
       nil))

   :gp-obs
   (fn [state addr dist]
     (let [{:keys [prior-addr mask]} (:params dist)]
       (if (= prior-addr target-addr)
         (let [posterior (get-in state [:conjugate-posteriors target-addr])
               constraint (cm/get-submap (:constraints state) addr)
               obs (cm/get-value constraint)
               {:keys [posterior ll]} (gp-update posterior obs mask)]
           [obs (-> state
                    (assoc-in [:conjugate-posteriors target-addr] posterior)
                    (update :choices cm/set-value addr obs)
                    (update :conjugate-ll
                      #(mx/add (or % (mx/zeros (ll-shape (:shape posterior)))) ll)))])
         nil)))})

;; ---------------------------------------------------------------------------
;; High-level API
;; ---------------------------------------------------------------------------

(defn conjugate-generate
  "Run a gen function under conjugate handler(s).

   gf:          DynamicGF with conjugate prior/obs trace sites
   args:        gen function arguments
   constraints: choicemap with observation constraints
   dispatches:  vector of dispatch maps (from make-*-dispatch)
   key:         PRNG key

   opts (map):
   - :param-store        parameter store for param sites
   - :init-posteriors    initial {addr -> posterior} (default: self-init from priors)

   Returns handler result with :conjugate-posteriors and :conjugate-ll."
  [gf args constraints dispatches key & [opts]]
  (let [{:keys [param-store init-posteriors]} opts
        transition (reduce ana/wrap-analytical h/generate-transition dispatches)
        init-state (cond-> {:choices cm/EMPTY
                            :score (mx/scalar 0.0)
                            :weight (mx/scalar 0.0)
                            :key key
                            :constraints constraints
                            :conjugate-posteriors (or init-posteriors {})}
                     param-store (assoc :param-store param-store))]
    (rt/run-handler transition init-state
      (fn [rt] (apply (:body-fn gf) rt args)))))

(defn conjugate-fold
  "Fold a per-step gen function over T timesteps under conjugate handler(s).

   step-fn:    gen function with conjugate prior/obs trace sites
   dispatches: vector of dispatch maps
   T:          number of timesteps
   context-fn: (fn [t] -> {:args [...], :constraints choicemap})

   Posteriors persist across timesteps — online Bayesian parameter learning.

   Returns {:ll total marginal LL, :posteriors final posteriors}."
  [step-fn dispatches T context-fn]
  (loop [t 0
         posteriors {}
         acc-ll nil]
    (if (>= t T)
      {:ll (or acc-ll (mx/scalar 0.0)) :posteriors posteriors}
      (let [{:keys [args constraints]} (context-fn t)
            result (conjugate-generate
                     step-fn args constraints dispatches
                     (rng/fresh-key t)
                     {:init-posteriors posteriors})
            step-ll (:conjugate-ll result)]
        (recur (inc t)
               (:conjugate-posteriors result)
               (if (and acc-ll step-ll)
                 (mx/add acc-ll step-ll)
                 (or step-ll acc-ll)))))))
