(ns genmlx.inference
  "Public inference API for GenMLX.
   Re-exports all inference algorithms."
  (:require [genmlx.inference.importance :as is-ns]
            [genmlx.inference.mcmc :as mcmc-ns]
            [genmlx.inference.smc :as smc-ns]
            [genmlx.inference.smcp3 :as smcp3-ns]
            [genmlx.inference.vi :as vi-ns]
            [genmlx.inference.kernel :as kern-ns]
            [genmlx.inference.diagnostics :as diag-ns]
            [genmlx.inference.adev :as adev-ns]
            [genmlx.inference.amortized :as amortized-ns]
            [genmlx.inference.enumerate :as enum-ns]))

;; ---------------------------------------------------------------------------
;; Importance sampling
;; ---------------------------------------------------------------------------

(def importance-sampling is-ns/importance-sampling)
(def importance-resampling is-ns/importance-resampling)

;; ---------------------------------------------------------------------------
;; MCMC
;; ---------------------------------------------------------------------------

(def mh mcmc-ns/mh)
(def mh-step mcmc-ns/mh-step)
(def mh-custom mcmc-ns/mh-custom)
(def mh-custom-step mcmc-ns/mh-custom-step)
(def mala mcmc-ns/mala)
(def hmc mcmc-ns/hmc)
(def nuts mcmc-ns/nuts)

;; Gibbs
(def gibbs mcmc-ns/gibbs)
(def gibbs-step-with-support mcmc-ns/gibbs-step-with-support)

;; Involutive MCMC
(def involutive-mh mcmc-ns/involutive-mh)
(def involutive-mh-step mcmc-ns/involutive-mh-step)

;; ---------------------------------------------------------------------------
;; SMC
;; ---------------------------------------------------------------------------

(def smc smc-ns/smc)
(def csmc smc-ns/csmc)

;; ---------------------------------------------------------------------------
;; SMCP3
;; ---------------------------------------------------------------------------

(def smcp3 smcp3-ns/smcp3)

;; ---------------------------------------------------------------------------
;; Variational inference
;; ---------------------------------------------------------------------------

(def vi vi-ns/vi)
(def vi-from-model vi-ns/vi-from-model)
(def programmable-vi vi-ns/programmable-vi)

;; VI objectives
(def elbo-objective vi-ns/elbo-objective)
(def iwelbo-objective vi-ns/iwelbo-objective)
(def pwake-objective vi-ns/pwake-objective)
(def qwake-objective vi-ns/qwake-objective)
(def reinforce-estimator vi-ns/reinforce-estimator)

;; ---------------------------------------------------------------------------
;; Inference composition
;; ---------------------------------------------------------------------------

(def mh-kernel kern-ns/mh-kernel)
(def update-kernel kern-ns/update-kernel)
(def chain kern-ns/chain)
(def repeat-kernel kern-ns/repeat-kernel)
(def seed kern-ns/seed)
(def cycle-kernels kern-ns/cycle-kernels)
(def mix-kernels kern-ns/mix-kernels)
(def run-kernel kern-ns/run-kernel)

;; Kernel reversal
(def with-reversal kern-ns/with-reversal)
(def symmetric-kernel kern-ns/symmetric-kernel)
(def reversal kern-ns/reversal)
(def symmetric? kern-ns/symmetric?)
(def reversed kern-ns/reversed)

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(def ess diag-ns/ess)
(def r-hat diag-ns/r-hat)
(def sample-mean diag-ns/sample-mean)
(def sample-std diag-ns/sample-std)
(def sample-quantiles diag-ns/sample-quantiles)
(def summarize diag-ns/summarize)

;; ---------------------------------------------------------------------------
;; ADEV gradient estimation
;; ---------------------------------------------------------------------------

(def has-reparam? adev-ns/has-reparam?)
(def adev-execute adev-ns/adev-execute)
(def adev-surrogate adev-ns/adev-surrogate)
(def adev-gradient adev-ns/adev-gradient)
(def adev-optimize adev-ns/adev-optimize)

;; ---------------------------------------------------------------------------
;; Amortized inference
;; ---------------------------------------------------------------------------

(def make-elbo-loss amortized-ns/make-elbo-loss)
(def train-proposal amortized-ns/train-proposal)
(def neural-importance-sampling amortized-ns/neural-importance-sampling)

;; ---------------------------------------------------------------------------
;; Enumerative inference
;; ---------------------------------------------------------------------------

(def enumerate-joint enum-ns/enumerate-joint)
(def enumerate-marginals enum-ns/enumerate-marginals)
(def enumerate-marginal-likelihood enum-ns/enumerate-marginal-likelihood)
