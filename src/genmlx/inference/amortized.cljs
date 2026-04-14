(ns genmlx.inference.amortized
  "Amortized inference via trained neural proposals.
   Train an encoder q(z|x; θ) to approximate the posterior p(z|x),
   then use it as a proposal in importance sampling.

   Training uses reparameterized ELBO (VAE-style) via nn/value-and-grad:
   encoder(x) → [μ, log σ] → z = μ + σε → ELBO = log p(z,x) - log q(z|x;θ).

   Inference uses the standard GFI: simulate the guide, score under the model.

   Encoders are nn layer maps (or atoms wrapping them). See genmlx.nn for
   the layer representation."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.nn :as nn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Posterior families (20.2)
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI (js/Math.log (* 2.0 js/Math.PI)))

(def gaussian-posterior
  "Gaussian posterior family: encoder outputs [mu, log-sigma] per latent.
   Reparameterized sample: z = mu + sigma * eps."
  {:n-params 2
   :sample (fn [raw-params key d]
             (let [mus      (mx/slice raw-params 0 d)
                   log-sigs (mx/slice raw-params d (* 2 d))
                   sigs     (mx/exp log-sigs)
                   eps      (rng/normal key [d])]
               {:values (mx/add mus (mx/multiply sigs eps))
                :log-prob (mx/sum
                            (mx/negative
                              (mx/add (mx/scalar (* 0.5 LOG-2PI))
                                      log-sigs
                                      (mx/multiply (mx/scalar 0.5) (mx/square eps)))))}))})

(def log-normal-posterior
  "Log-normal posterior family: encoder outputs [mu, log-sigma] per latent.
   Reparameterized sample: z = exp(mu + sigma * eps), for positive latents.
   log q(z) = log q_normal(log z) - log z = N(log z; mu, sigma) - sum(log z)."
  {:n-params 2
   :sample (fn [raw-params key d]
             (let [mus      (mx/slice raw-params 0 d)
                   log-sigs (mx/slice raw-params d (* 2 d))
                   sigs     (mx/exp log-sigs)
                   eps      (rng/normal key [d])
                   log-z    (mx/add mus (mx/multiply sigs eps))
                   z        (mx/exp log-z)
                   log-prob (mx/subtract
                              (mx/sum
                                (mx/negative
                                  (mx/add (mx/scalar (* 0.5 LOG-2PI))
                                          log-sigs
                                          (mx/multiply (mx/scalar 0.5) (mx/square eps)))))
                              (mx/sum log-z))]
               {:values z
                :log-prob log-prob}))})

;; ---------------------------------------------------------------------------
;; Internal: ensure we have an atom wrapping a layer map
;; ---------------------------------------------------------------------------

(defn- ensure-ref
  "Coerce encoder to an atom if it isn't one already."
  [encoder]
  (if (instance? Atom encoder) encoder (atom encoder)))

(defn- fwd
  "Get the current forward function from an encoder (atom or map)."
  [encoder]
  (if (instance? Atom encoder)
    (:forward @encoder)
    (:forward encoder)))

;; ---------------------------------------------------------------------------
;; ELBO loss for training
;; ---------------------------------------------------------------------------

(defn make-elbo-loss
  "Create a reparameterized ELBO loss function for training a neural proposal.

   encoder: nn layer map or atom — the encoder network
   model: target generative function
   latent-addrs: vector of latent address keywords (length d)

   Options:
     :model-args-fn    (fn [data] -> model-args-vec), default (fn [x] [x])
     :observations-fn  (fn [data] -> choicemap), default returns EMPTY
     :posterior-family  posterior family spec (default gaussian-posterior)
     :log-joint-fn     (fn [latent-values data] -> scalar log p(z, x))
                       Pure MxArray function for computing the model log-joint.
                       REQUIRED for gradient-based training via nn/value-and-grad.

                       Why: MLX autograd (MxArray.valueAndGrad) traces MxArray
                       operations to build a computation graph. The GFI handler
                       system uses a volatile! cell for state threading — this is
                       opaque to MLX's tracer, so gradients through p/generate
                       are always zero. The log-joint-fn must express the same
                       math as the model using only mx/* operations.

                       Example for z ~ N(0,1), x|z ~ N(z, 0.5):
                         (fn [latent-values data]
                           (let [z (mx/index latent-values 0)
                                 x (mx/squeeze data)]
                             (mx/add (gaussian-log-prob z 0 1)
                                     (gaussian-log-prob x z 0.5))))

                       If omitted, falls back to p/generate (works for evaluation
                       and scoring, but NOT for gradient computation).

   Returns (fn [forward-fn data] -> scalar loss) for use with nn/value-and-grad."
  [encoder model latent-addrs & {:keys [model-args-fn observations-fn posterior-family
                                         log-joint-fn]
                                  :or {model-args-fn (fn [x] [x])
                                       observations-fn (fn [_] cm/EMPTY)
                                       posterior-family gaussian-posterior}}]
  (let [model (dyn/auto-key model)
        d (count latent-addrs)
        sample-fn (:sample posterior-family)]
    (fn [forward-fn data]
      (let [out (forward-fn data)
            {:keys [values log-prob]} (sample-fn out (rng/fresh-key) d)
            log-joint
            (if log-joint-fn
              ;; Pure MxArray path — gradients flow through
              (log-joint-fn values data)
              ;; Handler path — no gradient flow, for evaluation only
              (let [obs (observations-fn data)
                    constraints (reduce (fn [cm [i addr]]
                                          (cm/set-choice cm [addr] (mx/index values i)))
                                        obs
                                        (map-indexed vector latent-addrs))
                    model-args (model-args-fn data)
                    {:keys [weight]} (p/generate model model-args constraints)]
                weight))
            elbo (mx/subtract log-joint log-prob)]
        (mx/negative elbo)))))

;; ---------------------------------------------------------------------------
;; Training loop (20.3)
;; ---------------------------------------------------------------------------

(defn train-proposal
  "Train a neural proposal encoder via reparameterized ELBO.

   encoder: nn layer map or atom
   loss-fn: (fn [forward-fn data] -> scalar loss), e.g. from make-elbo-loss
   dataset: vector of training data points (MLX arrays)

   Options:
     :iterations  number of training steps (default 300)
     :optimizer   optimizer type :adam, :sgd, :adamw (default :adam)
     :lr          learning rate (default 0.01)
     :batch-size  number of data points per training step (default 1)
     :shuffle     shuffle dataset at epoch boundaries (default true when batch-size > 1)

   Returns vector of loss values (JS numbers)."
  [encoder loss-fn dataset & {:keys [iterations optimizer lr batch-size shuffle]
                               :or {iterations 300 optimizer :adam lr 0.01
                                    batch-size 1}}]
  (let [enc-ref  (ensure-ref encoder)
        shuffle? (if (some? shuffle) shuffle (> batch-size 1))
        n        (count dataset)
        opt      (nn/optimizer optimizer lr)
        shuffled-order (fn [] (vec (clojure.core/shuffle (range n))))]
    (if (<= batch-size 1)
      ;; Single-sample mode
      (let [vg (nn/value-and-grad enc-ref loss-fn)]
        (mapv (fn [i]
                (let [data (nth dataset (mod i n))]
                  (nn/training-step! enc-ref opt vg data)))
              (range iterations)))
      ;; Minibatch mode
      (let [batch-loss-fn (fn [fwd batch]
                            (mx/divide (reduce mx/add (mapv #(loss-fn fwd %) batch))
                                       (mx/scalar (count batch))))
            vg (nn/value-and-grad enc-ref batch-loss-fn)]
        (loop [i 0
               pos 0
               order (if shuffle? (shuffled-order) (vec (range n)))
               losses (transient [])]
          (if (>= i iterations)
            (persistent! losses)
            (let [[pos order] (if (>= pos n)
                                [0 (if shuffle? (shuffled-order) order)]
                                [pos order])
                  end (min (+ pos batch-size) n)
                  batch-indices (subvec order pos end)
                  batch (mapv #(nth dataset %) batch-indices)
                  loss (nn/training-step! enc-ref opt vg batch)]
              (recur (inc i) (long end) order (conj! losses loss)))))))))

;; ---------------------------------------------------------------------------
;; Neural importance sampling
;; ---------------------------------------------------------------------------

(defn neural-importance-sampling
  "Importance sampling using a trained neural guide as proposal.

   guide: generative function (e.g. gen fn with spliced encoder)
   model: target generative function
   guide-args: arguments to the guide
   model-args: arguments to the model
   observations: choicemap of observed data

   Options:
     :samples  number of IS samples (default 100)

   Returns {:traces [Trace ...] :log-weights [MLX-scalar ...]
            :log-ml-estimate MLX-scalar}"
  [{:keys [samples] :or {samples 100}} guide model guide-args model-args observations]
  (let [model (dyn/auto-key model)
        guide (dyn/auto-key guide)
        results
        (mapv
          (fn [_]
            (let [{:keys [choices weight]} (p/propose guide guide-args)
                  guide-score weight
                  all-constraints (cm/merge-cm choices observations)
                  {:keys [trace weight]} (p/generate model model-args all-constraints)
                  log-w (mx/subtract weight guide-score)]
              {:trace trace :log-weight log-w}))
          (range samples))
        traces     (mapv :trace results)
        log-weights (mapv :log-weight results)
        weights-arr (u/materialize-weights log-weights)
        log-ml (mx/subtract (mx/logsumexp weights-arr)
                             (mx/scalar (js/Math.log samples)))]
    {:traces traces
     :log-weights log-weights
     :log-ml-estimate log-ml}))

;; ---------------------------------------------------------------------------
;; Vectorized neural importance sampling (20.1)
;; ---------------------------------------------------------------------------

(defn vectorized-neural-importance-sampling
  "Vectorized importance sampling using a trained neural guide as proposal.
   Runs the guide N times (sequential, cheap), then scores all proposals
   under the model in a single batched vgenerate call (GPU-parallel).

   guide: generative function (e.g. gen fn with spliced encoder)
   model: target generative function (must be a DynamicGF)
   guide-args: arguments to the guide
   model-args: arguments to the model
   observations: choicemap of observed data (scalar values)

   Options:
     :samples  number of IS samples (default 100)

   Returns {:vtrace VectorizedTrace :log-weights [N]-shaped MLX array
            :log-ml-estimate MLX scalar}"
  [{:keys [samples] :or {samples 100}} guide model guide-args model-args observations]
  (let [guide (dyn/auto-key guide)
        proposals (mapv (fn [_]
                          (let [{:keys [choices weight]} (p/propose guide guide-args)]
                            {:choices choices :guide-score weight}))
                        (range samples))
        guide-cms (mapv :choices proposals)
        stacked-cm (cm/stack-choicemaps guide-cms mx/stack)
        all-constraints (cm/merge-cm stacked-cm observations)
        key (rng/fresh-key)
        vtrace (dyn/vgenerate model model-args all-constraints samples key)
        guide-scores-arr (mx/stack (mapv :guide-score proposals))
        log-weights (mx/subtract (:weight vtrace) guide-scores-arr)]
    (mx/eval! log-weights)
    (let [log-ml (mx/subtract (mx/logsumexp log-weights)
                               (mx/scalar (js/Math.log samples)))]
      {:vtrace vtrace
       :log-weights log-weights
       :log-ml-estimate log-ml})))
