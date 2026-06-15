(ns genmlx.control.meta-mdp
  "The metareasoner (genmlx-nrkq): an agent pointed at COMPUTATION. control =
   genmlx.agents pointed at the inference process itself.

   Shape-B realization (the biased_planners precedent: recursion + soft policy,
   NO dense [S,A,S'] tensor). The meta-state is CONTINUOUS (a posterior summary +
   a value-of-computation), so make-mdp-agent's tabular VI does not apply; instead
   the policy is a soft choice over computational actions weighted by a Monte-Carlo
   one-step value-of-computation. For v1.0 the controller is MYOPIC (blinkered,
   one-step VOC) over the action set [:continue :stop]. :add-particle / :refine
   (particle-growth substrate) and :switch-method (live SMC<->MCMC translation)
   are deferred — the steppable substrate advances per observation, so :continue
   = spend the next step, :stop = commit the current decision.

   The controller's reward at :stop is a DOWNSTREAM decision-value (neg Bayes risk
   / max-EU), NEVER ESS or log-ML (see decision-value/assert-downstream!). The
   per-step compute cost comes from the i0s4 cost meter. The scheduler (the SOLE
   side effect) is genmlx.world.proc — the controller is NEVER the scheduler. We
   realize the control loop by GATING a base steppable's `done?` with the VOC
   stop, so proc drives it directly (its wall-clock budget = the hard cap; the VOC
   = the resource-rational stop). One-way dep: this ns requires genmlx.agents but
   nothing in genmlx.agents requires genmlx.control."
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.agents.helpers :as h]
            [genmlx.inference.cost :as cost]
            [genmlx.control.decision-value :as dv]
            [genmlx.gen :refer [gen]]))

(def actions
  "The v1.0 meta-action set. :add-particle / :refine / :switch-method are deferred."
  [:continue :stop])

(defn- controlled-steppable
  "Wrap a base steppable {:init :step :done? :best} into a controller-gated one.
   `dv-fn` : base-state -> downstream decision-value.
   `act`   : {:eu [eu-continue eu-stop]} -> :continue | :stop (the policy).
   `lambda`: compute-cost weight. `cost-key`: which cost-meter field is the cost.

   The first advance initializes the particle cloud (no posterior exists yet, so
   no VOC check). Thereafter each step is a myopic one-step VOC gate: trial-advance
   (the MC lookahead = one real step, also where the compute cost is metered), and
   if the value gained does not justify its cost the policy chooses :stop, which
   flips the wrapped done? so proc halts."
  [base dv-fn act lambda cost-key hysteresis]
  (let [bstep (:step base)
        bdone? (:done? base)
        bbest (:best base)]
    {:init (fn [] {:base ((:init base)) :stopped? false :stop-streak 0
                   :total-cost cost/zero :control-steps 0 :last-voc nil})
     :step (fn [{:keys [base control-steps] :as cs}]
             (if (zero? control-steps)
               ;; First advance: initialize the particle cloud (decision-value
               ;; is undefined before any observation is folded).
               (let [{trial :result c :cost} (cost/measure-step #(bstep base))]
                 (-> cs (assoc :base trial)
                     (update :total-cost cost/cost+ c)
                     (update :control-steps inc)))
               ;; Myopic blinkered VOC + HYSTERESIS: a single noisy down-estimate
               ;; must NOT stop a run that is still improving. Require `hysteresis`
               ;; consecutive stop-leans before committing the stop (Russell &
               ;; Wefald meta-greedy stop, hardened against MC noise).
               (let [{trial :result c :cost} (cost/measure-step #(bstep base))
                     step-cost (get c cost-key 0)
                     voc (- (dv-fn trial) (dv-fn base) (* lambda step-cost))
                     action (act {:eu [voc 0.0]})]
                 (if (= action :continue)
                   (-> cs (assoc :base trial :last-voc voc :stop-streak 0)
                       (update :total-cost cost/cost+ c)
                       (update :control-steps inc))
                   (let [streak (inc (:stop-streak cs))]
                     (if (>= streak hysteresis)
                       ;; committed stop — flip done? so proc halts
                       (-> cs (assoc :stopped? true :last-voc voc)
                           (update :total-cost cost/cost+ c))
                       ;; tentative stop — keep going (commit the trial), bump streak
                       (-> cs (assoc :base trial :last-voc voc :stop-streak streak)
                           (update :total-cost cost/cost+ c)
                           (update :control-steps inc))))))))
     :done? (fn [{:keys [base stopped?]}] (or stopped? (bdone? base)))
     :best (fn [{:keys [base]}] (bbest base))}))

(defn make-metareasoner
  "Build a metareasoner. opts:
     :alpha     rationality (##Inf = deterministic meta-greedy argmax; default ##Inf)
     :lambda    compute-cost weight in the VOC (default 0.0 = compute is free)
     :latent-addr  the scalar latent whose posterior decision-value is optimized
     :decision-value-fn  optional override: base-state -> downstream value
                         (default: neg Bayes risk of `latent-addr`)
     :cost-key  cost-meter field used as the per-step cost (default :forced-evals)

   Returns {:params :policy :act :decision-value :control}, mirroring the agents
   make-*-agent return shape. :policy is a generative function over the meta-action
   (p/simulate works on it); :control wraps a base steppable for genmlx.world.proc."
  [{:keys [alpha lambda latent-addr decision-value-fn cost-key hysteresis]
    :or {alpha ##Inf lambda 0.0 cost-key :forced-evals hysteresis 3}}]
  (let [dv-fn (or decision-value-fn
                  (fn [base-state]
                    (dv/neg-bayes-risk (dv/weighted-latent base-state latent-addr))))
        ;; The meta-policy IS a generative function: action ~ softmax(alpha * EU).
        ;; This is 'control = agents pointed at computation' — the same
        ;; planning-as-inference realization agents use, over computational EU.
        policy (dyn/auto-key
                 (gen [meta-s]
                   (trace :meta-action (h/softmax-action alpha (mx/array (:eu meta-s))))))
        act (fn act-fn
              ([meta-s] (act-fn meta-s nil))
              ([meta-s key]
               ;; assert the reward is downstream, never a sampler diagnostic
               (dv/assert-downstream! meta-s)
               (let [pol (if key (dyn/with-key policy key) policy)
                     tr (p/simulate pol [meta-s])
                     idx (mx/realize (cm/get-value (cm/get-submap (:choices tr) :meta-action)))]
                 (nth actions (int idx)))))]
    {:params {:alpha alpha :lambda lambda :cost-key cost-key :latent-addr latent-addr
              :hysteresis hysteresis}
     :policy policy
     :decision-value dv-fn
     :act act
     :control (fn [base] (controlled-steppable base dv-fn act lambda cost-key hysteresis))}))

(defn switch-method-translate
  "DEFERRED for v1.0 (genmlx-nrkq open question): live SMC<->MCMC state
   translation. The steppable peek payload is not portable across substrates
   (accept-rate vs weights/ESS/log-ML), and re-interpreting a weighted particle
   set as an MCMC chain start is unspecified. Stubbed so the seam is named, not
   silently absent."
  [_from-state]
  (throw (ex-info "switch-method is deferred for v1.0 (live SMC<->MCMC translation unspecified)"
                  {:status :not-implemented})))
