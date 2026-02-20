(ns genmlx.inference.amortized
  "Amortized inference via trained neural proposals.
   Train an encoder q(z|x; θ) to approximate the posterior p(z|x),
   then use it as a proposal in importance sampling.

   Training uses reparameterized ELBO (VAE-style) via nn.valueAndGrad:
   encoder(x) → [μ, log σ] → z = μ + σε → ELBO = log p(z,x) - log q(z|x;θ).

   Inference uses the standard GFI: simulate the guide, score under the model."
  (:require [genmlx.mlx :as mx]
            [genmlx.nn :as nn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; ELBO loss for training
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI (js/Math.log (* 2.0 js/Math.PI)))

(defn make-elbo-loss
  "Create a reparameterized ELBO loss function for training a Gaussian neural proposal.

   encoder: nn.Module mapping data → [2*d] (d means ++ d log-sigmas)
   model: target generative function
   latent-addrs: vector of latent address keywords (length d)

   Options:
     :model-args-fn   (fn [data] -> model-args-vec), default (fn [x] [x])
     :observations-fn  (fn [data] -> choicemap), default returns EMPTY

   Returns (fn [data] -> scalar loss) suitable for nn/value-and-grad."
  [encoder model latent-addrs & {:keys [model-args-fn observations-fn]
                                  :or {model-args-fn (fn [x] [x])
                                       observations-fn (fn [_] cm/EMPTY)}}]
  (let [d (count latent-addrs)]
    (fn [data]
      (let [;; Forward through encoder
            out (.forward encoder data)
            mus      (mx/slice out 0 d)
            log-sigs (mx/slice out d (* 2 d))
            sigs     (mx/exp log-sigs)
            ;; Reparameterized sample: z = μ + σε
            eps (mx/random-normal [d])
            zs  (mx/add mus (mx/multiply sigs eps))
            ;; Build constraint choicemap: latent values + observations
            obs (observations-fn data)
            constraints (reduce (fn [cm [i addr]]
                                  (cm/set-choice cm [addr] (mx/index zs i)))
                                obs
                                (map-indexed vector latent-addrs))
            ;; Score under model: log p(z, x)
            model-args (model-args-fn data)
            {:keys [weight]} (p/generate model model-args constraints)
            log-joint weight
            ;; Analytic log q(z|x;θ) = Σ_i [-0.5 log(2π) - log σ_i - 0.5 ((z_i-μ_i)/σ_i)²]
            log-q (mx/sum
                    (mx/negative
                      (mx/add (mx/scalar (* 0.5 LOG-2PI))
                              log-sigs
                              (mx/multiply (mx/scalar 0.5) (mx/square eps)))))
            ;; ELBO = log p(z,x) - log q(z|x;θ); loss = -ELBO
            elbo (mx/subtract log-joint log-q)]
        (mx/negative elbo)))))

;; ---------------------------------------------------------------------------
;; Training loop
;; ---------------------------------------------------------------------------

(defn train-proposal!
  "Train a neural proposal encoder via reparameterized ELBO.

   encoder: nn.Module
   loss-fn: (fn [data] -> scalar loss), e.g. from make-elbo-loss
   dataset: vector of training data points (MLX arrays)

   Options:
     :iterations  number of training steps (default 300)
     :optimizer   optimizer type :adam, :sgd, :adamw (default :adam)
     :lr          learning rate (default 0.01)

   Returns vector of loss values (JS numbers)."
  [encoder loss-fn dataset & {:keys [iterations optimizer lr]
                               :or {iterations 300 optimizer :adam lr 0.01}}]
  (let [opt (nn/optimizer optimizer lr)
        vg  (nn/value-and-grad encoder loss-fn)
        n   (count dataset)]
    (mapv (fn [i]
            (let [data (nth dataset (mod i n))]
              (nn/step! encoder opt vg data)))
          (range iterations))))

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
  (let [results
        (mapv
          (fn [_]
            (let [;; Propose from guide
                  {:keys [choices weight]} (p/propose guide guide-args)
                  guide-score weight
                  ;; Merge proposed latents with observations
                  all-constraints (cm/merge-cm choices observations)
                  ;; Score under model
                  {:keys [trace weight]} (p/generate model model-args all-constraints)
                  model-weight weight
                  ;; IS weight = log p(z,x) - log q(z|x)
                  log-w (mx/subtract model-weight guide-score)]
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
