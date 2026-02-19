(ns genmlx.inference
  "Public inference API for GenMLX.
   Re-exports all inference algorithms."
  (:require [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc-ns]
            [genmlx.inference.smcp3 :as smcp3-ns]
            [genmlx.inference.vi :as vi-ns]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.diagnostics :as diag]))

;; ---------------------------------------------------------------------------
;; Importance sampling
;; ---------------------------------------------------------------------------

(def importance-sampling is/importance-sampling)
(def importance-resampling is/importance-resampling)

;; ---------------------------------------------------------------------------
;; MCMC
;; ---------------------------------------------------------------------------

(def mh mcmc/mh)
(def mh-step mcmc/mh-step)
(def mh-custom mcmc/mh-custom)
(def mh-custom-step mcmc/mh-custom-step)
(def mala mcmc/mala)
(def hmc mcmc/hmc)
(def nuts mcmc/nuts)

;; Gibbs
(def gibbs mcmc/gibbs)
(def gibbs-step-with-support mcmc/gibbs-step-with-support)

;; Involutive MCMC
(def involutive-mh mcmc/involutive-mh)
(def involutive-mh-step mcmc/involutive-mh-step)

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

(def mh-kernel kern/mh-kernel)
(def update-kernel kern/update-kernel)
(def chain kern/chain)
(def repeat-kernel kern/repeat-kernel)
(def seed kern/seed)
(def cycle-kernels kern/cycle-kernels)
(def mix-kernels kern/mix-kernels)
(def run-kernel kern/run-kernel)

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(def ess diag/ess)
(def r-hat diag/r-hat)
(def sample-mean diag/sample-mean)
(def sample-std diag/sample-std)
(def sample-quantiles diag/sample-quantiles)
(def summarize diag/summarize)
