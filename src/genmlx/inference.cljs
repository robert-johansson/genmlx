(ns genmlx.inference
  "Public inference API for GenMLX.
   Re-exports all inference algorithms."
  (:require [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.vi :as vi]
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
(def mala mcmc/mala)
(def hmc mcmc/hmc)
(def nuts mcmc/nuts)

;; ---------------------------------------------------------------------------
;; SMC
;; ---------------------------------------------------------------------------

(def smc smc/smc)

;; ---------------------------------------------------------------------------
;; Variational inference
;; ---------------------------------------------------------------------------

(def vi vi/vi)
(def vi-from-model vi/vi-from-model)

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(def ess diag/ess)
(def r-hat diag/r-hat)
(def sample-mean diag/sample-mean)
(def sample-std diag/sample-std)
(def sample-quantiles diag/sample-quantiles)
(def summarize diag/summarize)
