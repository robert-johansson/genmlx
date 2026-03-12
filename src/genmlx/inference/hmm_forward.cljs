(ns genmlx.inference.hmm-forward
  "HMM forward algorithm middleware for discrete latent state-space models.

   The discrete analog of the Kalman middleware. Instead of Gaussian beliefs,
   the handler maintains a K-dimensional log-probability vector (forward
   messages). Analytically marginalizes discrete latent states, computing
   exact marginal likelihoods.

   Two levels of API (matching Kalman):

   1. Pure building blocks — hmm-predict, hmm-update.
      Use directly in gen function bodies for explicit control.

   2. Handler middleware — make-hmm-transition + hmm-generate + hmm-fold.
      The cognitive architecture uses hmm-latent and hmm-obs trace sites.
      hmm-fold runs it over T timesteps under the HMM handler.
      Same gen function works under standard handlers (sampling instead
      of marginalizing).

   Belief representation: {:log-alpha [P,K]-shaped log-probability vector}
   where P = batch dimension (e.g. units), K = number of discrete states."
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
;; Distributions that carry discrete structure
;; ---------------------------------------------------------------------------

(defdist hmm-latent
  "Discrete latent state transition.
   log-trans-row: log transition probabilities FROM prev state [K]-shaped.
   prev-state:    previous state index (int, used under standard handler).

   Under standard handler: samples from categorical.
   Under HMM handler: provides transition matrix row for forward step."
  [log-trans-row prev-state]
  (sample [key]
    (dc/dist-sample
      (dc/->Distribution :categorical {:logits log-trans-row})
      key))
  (log-prob [v]
    (dc/dist-log-prob
      (dc/->Distribution :categorical {:logits log-trans-row})
      v)))

(defdist hmm-obs
  "Observation under discrete latent state.
   log-emission-probs: [K]-shaped log p(obs | state=k) for each state k.
   mask: 1=observed, 0=missing (masked obs contribute 0 to LL).

   Under standard handler: returns the log-prob of the latent state's emission.
   Under HMM handler: provides emission log-probs for update step."
  [log-emission-probs mask]
  (sample [key]
    ;; Under standard handler, this is just a scoring distribution
    ;; (the observation is always constrained). Return dummy value.
    (mx/scalar 0.0))
  (log-prob [v]
    ;; v is ignored — log-emission-probs already contains the per-state LL
    ;; Under standard handler this isn't directly useful; use kalman-obs
    ;; pattern or constrain this site.
    (mx/scalar 0.0)))

;; ---------------------------------------------------------------------------
;; Pure HMM operations (Level 1)
;; ---------------------------------------------------------------------------

(defn hmm-predict
  "HMM predict step: propagate belief through transition matrix.
   log-alpha:    [P,K] or [K] log-probability belief vector
   log-trans:    [K,K] log transition matrix (rows = from, cols = to)
   Returns updated [P,K] or [K] log-alpha."
  [log-alpha log-trans]
  ;; alpha'[j] = sum_i alpha[i] * T[i,j]
  ;; In log-space: log-alpha'[j] = logsumexp_i(log-alpha[i] + log-T[i,j])
  ;; log-alpha: [P,K] -> expand to [P,K,1]
  ;; log-trans: [K,K] -> broadcast [1,K,K]
  ;; sum:       [P,K,K] -> logsumexp over axis -2 -> [P,K]
  (let [ndim (count (mx/shape log-alpha))
        expanded (if (= ndim 1)
                   ;; [K] -> [K,1]
                   (mx/expand-dims log-alpha -1)
                   ;; [P,K] -> [P,K,1]
                   (mx/expand-dims log-alpha -1))
        ;; expanded: [...,K,1] + log-trans: [K,K] -> [...,K,K]
        joint (mx/add expanded log-trans)]
    ;; logsumexp over the "from" axis (second-to-last)
    (mx/logsumexp joint [-2])))

(defn hmm-update
  "HMM update step: incorporate one observation.
   log-alpha:          [P,K] or [K] current log belief
   log-emission-probs: [P,K] or [K] log p(obs | state=k)
   mask:               [P] or scalar (1=observed, 0=missing)
   Returns {:log-alpha updated, :ll per-element marginal LL}."
  [log-alpha log-emission-probs mask]
  ;; unnormalized: log-alpha + log-emission
  (let [unnorm (mx/add log-alpha log-emission-probs)
        ;; Marginal LL: logsumexp over states
        ll-raw (mx/logsumexp unnorm [-1])
        ;; Normalize: log-alpha' = unnorm - logsumexp(unnorm)
        norm-alpha (mx/subtract unnorm (mx/expand-dims ll-raw -1))
        ;; Masked: if mask=0, keep old belief, ll=0
        ndim-mask (count (mx/shape mask))
        mask-expanded (if (zero? ndim-mask)
                        mask
                        (mx/expand-dims mask -1))
        new-alpha (mx/add (mx/multiply mask-expanded norm-alpha)
                          (mx/multiply (mx/subtract (mx/scalar 1.0) mask-expanded)
                                       log-alpha))
        masked-ll (mx/multiply mask ll-raw)]
    {:log-alpha new-alpha :ll masked-ll}))

(defn hmm-step
  "One complete HMM step: predict + update.
   log-alpha:          [P,K] current belief
   log-trans:          [K,K] transition matrix (log-space)
   log-emission-probs: [P,K] emission log-probs
   mask:               [P] observation mask
   Returns {:log-alpha updated, :ll per-element marginal LL}."
  [log-alpha log-trans log-emission-probs mask]
  (let [predicted (hmm-predict log-alpha log-trans)]
    (hmm-update predicted log-emission-probs mask)))

;; ---------------------------------------------------------------------------
;; Handler middleware (Level 2)
;; ---------------------------------------------------------------------------
;;
;; The handler intercepts hmm-latent and hmm-obs trace sites.
;; Observation LL accumulates in :hmm-ll ([P]-shaped, per-element),
;; NOT in :score/:weight. Same design as Kalman middleware.
;;
;; Key design: initial belief = uniform over K states.
;; log-alpha = -log(K) for all states.

(defn make-hmm-dispatch
  "Create HMM dispatch map for use with wrap-analytical.

   latent-addr: keyword address of the discrete latent state site
   log-trans:   [K,K] log transition matrix

   Returns dispatch map: {:hmm-latent handler, :hmm-obs handler}."
  [latent-addr log-trans]
  {:hmm-latent
   (fn [state addr dist]
     (if (= addr latent-addr)
       ;; HMM predict: propagate belief through transition matrix
       (let [K (last (mx/shape log-trans))
             n (:hmm-n state)
             log-alpha (or (:hmm-belief state)
                           ;; Uniform prior: log(1/K) = -log(K)
                           (mx/multiply (mx/scalar (- (js/Math.log K)))
                                        (mx/ones (if n [n K] [K]))))
             new-alpha (hmm-predict log-alpha log-trans)
             ;; Return MAP state as the "sampled" value (for choices)
             map-state (mx/argmax new-alpha -1)]
         [map-state
          (-> state
              (assoc :hmm-belief new-alpha)
              (update :choices cm/set-value addr map-state))])
       ;; Not our latent addr — delegate (will fall through to base)
       nil))

   :hmm-obs
   (fn [state addr dist]
     (let [{:keys [log-emission-probs mask]} (:params dist)
           log-alpha (:hmm-belief state)
           n (:hmm-n state)
           {:keys [log-alpha ll]} (hmm-update log-alpha log-emission-probs mask)]
       ;; Store constrained observation in choices
       (let [constraint (cm/get-submap (:constraints state) addr)
             obs (when (cm/has-value? constraint) (cm/get-value constraint))]
         [(or obs (mx/scalar 0.0))
          (-> state
              (assoc :hmm-belief log-alpha)
              (cond-> obs (update :choices cm/set-value addr obs))
              (update :hmm-ll
                #(mx/add (or % (mx/zeros (if n [n] []))) ll)))])))})

(defn make-hmm-transition
  "Handler middleware: wraps generate-transition for HMM forward algorithm.

   latent-addr: keyword address of the discrete latent state
   log-trans:   [K,K] log transition matrix

   Returns a transition function that:
   - hmm-latent sites: forward predict, return MAP state
   - hmm-obs sites: forward update, accumulate LL in :hmm-ll
   - Other sites: delegate to generate-transition"
  [latent-addr log-trans]
  (let [dispatch (make-hmm-dispatch latent-addr log-trans)]
    (ana/wrap-analytical h/generate-transition dispatch)))

(defn hmm-generate
  "Run a gen function body under the HMM forward handler.

   gf:          DynamicGF with hmm-latent/hmm-obs trace sites
   args:        gen function arguments
   constraints: choicemap with observation constraints
   latent-addr: keyword address of the discrete latent state
   log-trans:   [K,K] log transition matrix
   n:           number of elements in batch dimension (0 for unbatched)
   K:           number of discrete states
   key:         PRNG key

   opts (map):
   - :param-store  parameter store for param sites
   - :init-belief  initial [P,K] or [K] log-alpha (default: uniform)

   Returns handler result map with :retval, :choices, :score, :weight,
   plus :hmm-belief and :hmm-ll."
  [gf args constraints latent-addr log-trans n K key & [opts]]
  (let [{:keys [param-store init-belief]} opts
        transition (make-hmm-transition latent-addr log-trans)
        uniform-prior (mx/multiply (mx/scalar (- (js/Math.log K)))
                                   (mx/ones (if (pos? n) [n K] [K])))
        init-state (cond-> {:choices cm/EMPTY
                            :score (mx/scalar 0.0)
                            :weight (mx/scalar 0.0)
                            :key key
                            :constraints constraints
                            :hmm-n (if (pos? n) n nil)
                            :hmm-belief (or init-belief uniform-prior)}
                     param-store (assoc :param-store param-store))]
    (rt/run-handler transition init-state
      (fn [rt] (apply (:body-fn gf) rt args)))))

(defn hmm-fold
  "Fold a per-step gen function over T timesteps under the HMM handler.

   step-fn:     gen function with hmm-latent and hmm-obs trace sites
   latent-addr: keyword address of the discrete latent state
   log-trans:   [K,K] log transition matrix
   n:           number of elements (units), 0 for unbatched
   K:           number of discrete states
   T:           number of timesteps
   context-fn:  (fn [t] -> {:args [step-fn-args], :constraints choicemap})

   Returns {:ll [P]-shaped total marginal LL, :belief final log-alpha}."
  [step-fn latent-addr log-trans n K T context-fn]
  (let [zero-ll (if (pos? n) (mx/zeros [n]) (mx/scalar 0.0))]
    (loop [t 0
           belief nil  ;; nil = use default uniform prior
           acc-ll zero-ll]
      (if (>= t T)
        {:ll acc-ll :belief belief}
        (let [{:keys [args constraints]} (context-fn t)
              result (hmm-generate
                       step-fn args constraints latent-addr log-trans n K
                       (rng/fresh-key t)
                       {:init-belief belief})
              step-ll (or (:hmm-ll result) zero-ll)]
          (recur (inc t)
                 (:hmm-belief result)
                 (mx/add acc-ll step-ll)))))))
