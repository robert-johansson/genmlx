(ns genmlx.control.decision-value
  "Decision-value for the metareasoner (genmlx-nrkq): the REAL downstream value
   of acting on the current posterior — max expected utility, or negative Bayes
   risk of a point estimate. This is the controller's reward signal at :stop.

   It is NEVER an inference diagnostic (ESS / log-ML). Those measure how well the
   sampler is doing, not how good a DECISION the posterior supports — optimizing
   them is the classic metareasoning trap (a controller that chases ESS spends
   compute making the sampler 'look healthy' instead of making the decision
   better). `assert-downstream!` enforces this at the boundary.

   This namespace is pure (Layer-A flow + the eval! boundary to read particle
   latents). It may be used standalone or via the steppable SMCState helpers."
  (:require [genmlx.inference.util :as u]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]))

(defn assert-downstream!
  "Guard: a decision-value spec must be a downstream quantity, never a sampler
   diagnostic. Throws if handed something that looks like a steppable `peek` map
   (carries :ess / :log-ml-estimate / :log-ml)."
  [x]
  (when (and (map? x)
             (some #(contains? x %) [:ess :log-ml-estimate :log-ml]))
    (throw (ex-info "decision-value must be a downstream quantity (max-EU / neg Bayes risk), never ESS/log-ML"
                    {:offending-keys (vec (keys x))})))
  x)

(defn weighted-latent
  "Extract {:probs [N] :values [N]} (clj doubles) for a scalar latent `addr`
   from an SMCState (or any {:traces :log-weights} map). Realizes at the
   boundary — call between steps, not in a hot inner loop."
  [{:keys [traces log-weights]} addr]
  (let [{:keys [probs]} (u/normalize-log-weights log-weights)
        ;; Stack the N per-particle latent scalars and read them in ONE GPU
        ;; transfer (not N item calls) — the VOC loop calls this twice per step.
        values (mx/->clj (mx/stack (mapv (fn [t] (cm/get-value (cm/get-submap (:choices t) addr)))
                                         traces)))]
    {:probs probs :values values}))

(defn weighted-mean
  "Posterior mean E_posterior[theta] = sum_i p_i * value_i over a weighted-latent
   map {:probs [N] :values [N]}."
  [{:keys [probs values]}]
  (reduce + 0.0 (map * probs values)))

(defn weighted-variance
  "Posterior variance Var_posterior(theta) = sum_i p_i * (value_i - mean)^2 over a
   weighted-latent map. Used by `neg-bayes-risk` as the squared-error decision-value."
  [{:keys [probs values] :as wl}]
  (let [m (weighted-mean wl)]
    (reduce + 0.0 (map (fn [p v] (let [d (- v m)] (* p d d))) probs values))))

(defn neg-bayes-risk
  "Decision-value under squared-error loss for the posterior-mean point estimate
   = -Var_posterior(theta). Higher (closer to 0) = a more decisive posterior, so
   folding more data / spending more compute that sharpens the posterior raises
   it. A genuine downstream value, not a sampler diagnostic."
  [weighted-latent-map]
  (- (weighted-variance weighted-latent-map)))

(defn max-eu
  "Decision-value = max_a sum_i p_i * utility(value_i, a) over a discrete action
   set — the Bayes-optimal expected utility of the best terminal decision."
  [{:keys [probs values]} utility actions]
  (apply max
         (map (fn [a] (reduce + 0.0 (map (fn [p v] (* p (utility v a))) probs values)))
              actions)))
