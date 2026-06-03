(ns agentmodels.joint-5d-inference
  "Procrastination JOINT inference (agentmodels Ch 5d) — joint inference of biases
   AND preferences. Extends the z4si single-latent (:bias) pattern in
   `biased_inverse.cljs` to MULTIPLE traced latents: :reward, :alpha, :discount,
   inverted by exact host-side enumeration over finite priors. (:bias is FIXED to
   :naive — agentmodels 5d's `sophisticatedOrNaive='naive'` — not jointly inferred.)

   An agent procrastinates: it WAITS day after day, then maybe works near the
   deadline. From an observed wait/work sequence we jointly infer the agent's
   reward, softmax-noise alpha, and hyperbolic discount k — comparing two models:

     - OPTIMAL            : discount fixed to {0} (no present bias). To explain the
                            observed waiting it must infer a LOW reward and HIGH noise.
     - POSSIBLY-DISCOUNTING: discount free over {0, .5, 1, 2, 4}. It can explain the
                             waiting as time-inconsistency, keeping reward higher and
                             noise lower.

   The likelihood of an observed action is exactly the softmax policy probability of
   the forward biased planner (`make-biased-mdp-agent`), scored with `p/assess` — no
   bespoke likelihood. The procrastination horizon SHRINKS along the sequence: the
   action at wait-state W_w is scored at the remaining horizon (deadline - w), the
   faithful reading of agentmodels' `observe(act(state, 0), action)`.

   Reuse (no duplicated recursion/enumeration): bp/make-biased-mdp-agent +
   bp/procrastination-mdp (forward model), bi/action-cm (z4si choicemap),
   inv/normalize-logs (stable softmax), h/uniform-draw|weighted-draw (finite priors).
   Zero engine change.

   Scope: 5d procrastination only (capability v1). Bandit reward-myopia across
   horizons is split to a follow-up bean (research-thin; needs a bandit MDP)."
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.agents.helpers :as h]
            [genmlx.agents.inverse :as inv]
            [agentmodels.biased-inverse :as bi])
  (:require-macros [genmlx.gen :refer [gen]]))

;; agentmodels Ch 5d priors
(def reward-vals  [0.5 2 3 4 5 6 7 8])
(def alpha-vals   [0.1 1 10 100 1000])
(def alpha-probs  [0.1 0.2 0.2 0.2 0.3])
(def discount-vals-discounting [0 0.5 1 2 4])
(def discount-vals-optimal     [0])

;; ===========================================================================
;; Forward agents over the joint finite prior
;; ===========================================================================

(defn build-agents
  "Map {[reward-idx alpha-idx discount-idx] -> {:agent :alpha :reward :discount}}.
   The procrastination MDP is rebuilt ONCE per reward value (reward is an MDP-build
   arg), then shared across the alpha x discount grid."
  [{:keys [reward-vals alpha-vals discount-vals work-cost wait-cost deadline n-iters bias]
    :or   {work-cost -1.0 wait-cost -0.1 deadline 10 n-iters 12 bias :naive}}]
  (let [mdps (mapv (fn [r] (bp/procrastination-mdp {:reward r :work-cost work-cost
                                                    :wait-cost wait-cost :deadline deadline}))
                   reward-vals)]
    (into {}
          (for [ri (range (count reward-vals))
                ai (range (count alpha-vals))
                di (range (count discount-vals))]
            (let [a (nth alpha-vals ai) k (nth discount-vals di)]
              [[ri ai di]
               {:agent   (bp/make-biased-mdp-agent
                           {:mdp (nth mdps ri) :alpha a :gamma 1.0 :n-iters n-iters}
                           {:discount k :bias bias})
                :alpha a :reward (nth reward-vals ri) :discount k}])))))

(defn- eu-at
  "EU vector over actions at wait-state s, horizon t (delay 0)."
  [agent s t]
  (mapv #((:eu agent) s % t 0) (range 2)))

;; ===========================================================================
;; The multi-latent joint generative function (extends z4si biased-agent-model)
;; ===========================================================================

(defn joint-procrastination-model
  "Joint GF tracing :reward, :alpha, :discount (indices into finite prior boxes);
   the sampled tuple selects a precomputed forward agent; one softmax-action site
   :a0 :a1 ... per observed wait-state, scored at the remaining horizon
   (horizon-fn s). All agents + EU-row arrays precomputed before `gen` (the body is
   re-run per enumerated tuple, so it only indexes)."
  [{:keys [states horizon-fn reward-vals alpha-vals discount-vals alpha-probs] :as cfg} agents]
  (let [reward-box   (h/uniform-draw reward-vals)
        alpha-box    (if alpha-probs (h/weighted-draw alpha-vals alpha-probs) (h/uniform-draw alpha-vals))
        discount-box (h/uniform-draw discount-vals)
        rows (into {} (for [[k {:keys [agent]}] agents]
                        [k (mapv (fn [s] (mx/array (clj->js (eu-at agent s (horizon-fn s))) mx/float32))
                                 states)]))]
    (gen []
      (let [ri (trace :reward   (:dist reward-box))
            ai (trace :alpha    (:dist alpha-box))
            di (trace :discount (:dist discount-box))
            er (rows [ri ai di])
            al (:alpha (agents [ri ai di]))]
        (doseq [i (range (count states))]
          (trace (keyword (str "a" i)) (h/softmax-action al (nth er i))))
        [ri ai di]))))

(defn- multi-full-cm
  "Choicemap {:reward ri, :alpha ai, :discount di, :a0 a0, ...}."
  [ri ai di actions]
  (-> (bi/action-cm actions)
      (cm/set-choice [:reward] ri)
      (cm/set-choice [:alpha] ai)
      (cm/set-choice [:discount] di)))

;; ===========================================================================
;; Exact enumeration over the joint finite prior
;; ===========================================================================

(defn- marginal [post vals axis]
  (reduce (fn [m [tup pr]] (update m (nth vals (nth tup axis)) (fnil + 0.0) pr)) {} post))

(defn expect
  "Posterior expectation of a latent given its {value -> prob} marginal."
  [marg]
  (reduce (fn [s [v pr]] (+ s (* v pr))) 0.0 marg))

(defn joint-posterior
  "Exact P(reward,alpha,discount | actions). Enumerate the joint finite prior; for
   each tuple assess the joint GF on the full choicemap; normalize. Returns
   {:joint {[ri ai di] prob} :marginals {:reward {...} :alpha {...} :discount {...}}}."
  [{:keys [states actions reward-vals alpha-vals discount-vals] :as cfg} agents]
  (assert (= (count states) (count actions))
          (str "5d joint: states/actions length mismatch " (count states) " vs " (count actions)))
  (let [model (joint-procrastination-model cfg agents)
        tuples (for [ri (range (count reward-vals))
                     ai (range (count alpha-vals))
                     di (range (count discount-vals))] [ri ai di])
        logw (into {} (for [[ri ai di :as tup] tuples]
                        [tup (mx/item (:weight (p/assess (dyn/auto-key model) []
                                                         (multi-full-cm ri ai di actions))))]))
        post (inv/normalize-logs logw)]
    {:joint post
     :marginals {:reward   (marginal post reward-vals 0)
                 :alpha    (marginal post alpha-vals 1)
                 :discount (marginal post discount-vals 2)}}))

(defn predict-work
  "Posterior-predictive P(work) at a wait-state (predict-state, predict-horizon):
   sum over the joint posterior of each tuple's softmax P(work=1)."
  [post agents predict-state predict-horizon]
  (reduce (fn [acc [tup pr]]
            (let [{:keys [agent alpha]} (agents tup)
                  eus (eu-at agent predict-state predict-horizon)
                  m   (apply max (mapv #(* alpha %) eus))
                  es  (mapv #(Math/exp (- (* alpha %) m)) eus)
                  pw  (/ (nth es 1) (reduce + es))]   ; action 1 = work
              (+ acc (* pr pw))))
          0.0 post))

;; ===========================================================================
;; Online (incremental) posterior time-series
;; ===========================================================================

(defn online-posteriors
  "Posterior expectations + predict-work after each prefix of the observed
   sequence (index 0 = prior). Reuses joint-posterior on truncated observations."
  [{:keys [states actions] :as cfg} agents predict-state predict-horizon]
  (mapv (fn [L]
          (let [c (assoc cfg :states (vec (take L states)) :actions (vec (take L actions)))
                {:keys [joint marginals]} (joint-posterior c agents)]
            {:n L
             :E-reward   (expect (:reward marginals))
             :E-alpha    (expect (:alpha marginals))
             :E-discount (expect (:discount marginals))
             :predict-work (predict-work joint agents predict-state predict-horizon)}))
        (range (inc (count states)))))

;; ===========================================================================
;; Demo configs (the canonical 8-waits procrastination observation)
;; ===========================================================================

(def DEADLINE 10)

(defn demo-cfg
  "cfg for a model. :optimal? true => discount fixed {0}; else Possibly-Discounting.
   :extra-work? true => append a work action at W_8 (task completes)."
  [{:keys [optimal? extra-work?]}]
  (let [n-wait    8                              ; 8 observed waits at W_0..W_7
        states    (vec (range n-wait))           ; W_0..W_7
        actions   (vec (repeat n-wait 0))        ; all wait
        states'   (if extra-work? (conj states n-wait) states)        ; + W_8
        actions'  (if extra-work? (conj actions 1) actions)]           ; + work
    {:states states' :actions actions'
     :horizon-fn #(- DEADLINE %)                 ; remaining horizon at W_w
     :reward-vals reward-vals :alpha-vals alpha-vals :alpha-probs alpha-probs
     :discount-vals (if optimal? discount-vals-optimal discount-vals-discounting)
     :deadline DEADLINE}))

(defn analyze
  "Run a model end-to-end: build agents, joint posterior, predict-work, marginal
   expectations. predict at W_8 (horizon 2)."
  [opts]
  (let [cfg    (demo-cfg opts)
        agents (build-agents cfg)
        {:keys [joint marginals] :as pp} (joint-posterior cfg agents)]
    {:cfg cfg :agents agents :posterior pp
     :predict-work (predict-work joint agents 8 2)
     :E-reward   (expect (:reward marginals))
     :E-alpha    (expect (:alpha marginals))
     :E-discount (expect (:discount marginals))}))
