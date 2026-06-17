(ns genmlx.agents.remote
  "PROVISIONAL — NOT part of the frozen v1.0 agents surface (CONTRACTS.md fences it
   out explicitly; its signatures may change and are NOT pinned by a contracts test).

   The agents-side glue for the FIRST EXTERNAL ENVIRONMENT (ROADMAP Phase 3 item 5):
   pure flow ABOVE the genmlx.world.net membrane that reifies the agent/environment
   boundary as real network I/O. (It lives in genmlx.agents — not genmlx.world —
   because it depends on the agents layer, e.g. agent/sample-next; the membrane must
   not depend on agents, so the layering points the right way: agents -> world.net.)

   Three parts, all pure above the membrane:

   1. THE RL SEAM — `gym-transport` composes the neutral net/request crossing into a
      Gym-style episodic transport {:step :reset :close}. The RL contract (action /
      observation / reward / done / step / reset) belongs HERE in the agents layer,
      NOT in the neutral membrane (genmlx.world.net stays domain-free).

   2. SERVE an agents environment behind the membrane — `mdp-env-handler` /
      `pomdp-env-handler` turn an MDP/POMDP bundle into the single
      (fn [route payload] -> response-map) genmlx.world.net/serve! hosts. The
      world's TRUE state lives behind the membrane (a server-side atom — the
      external world's own state, deliberately outside pure flow) and the world OWNS
      its transition randomness (a private key the agent never sees); the agent only
      learns the next state/observation by receiving it across the wire.

   3. DRIVE an agent against a REMOTE environment — `remote-mdp-rollout` /
      `remote-pomdp-rollout` run the SAME act / transition / observe / filter loop as
      agent/simulate-mdp and pomdp/simulate-pomdp, but the transition + observation
      cross the wire. act stays pure on the client; the POSTed action is the
      ACTUATOR, the returned observation is the SENSOR. The loops are ASYNC ONLY at
      the wire crossing (pr/let over a transport call); the per-step decision math
      (policy, belief filter) is pure and synchronous between awaits — no GPU eval
      inside the await loop. They return promesa promises of the SAME trajectory
      shapes as their in-process twins.

   FAITHFULNESS: at the deterministic regime (alpha = ##Inf, noise = 0) the remote
   trajectory is BIT-IDENTICAL to the in-process simulate-mdp / simulate-pomdp run —
   the same parity regime as the fused rollouts (genmlx.agents.rollout,
   pomdp/fused-simulate-pomdp). That equality is the proof the membrane is faithful
   and thin: the agent code is unchanged whether the env is in-process or external.
   Off that regime (finite alpha / noise > 0) only the DISTRIBUTION matches, never
   the per-step sequence — RNG keys do not cross the wire.

   Zero engine change; reuses agent/sample-next, the existing belief filter,
   build-mdp tensors, and the GFI policy."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.agents.agent :as agent]
            [genmlx.world.net :as net]
            [cljs.reader :as reader]
            [promesa.core :as pr]))

;; ===========================================================================
;; 1. The RL seam — Gym episodic transport over the neutral crossing
;; ===========================================================================

(defn gym-transport
  "Compose net/request into the Gym-style episodic transport the rollouts use:
   {:step (fn [action] -> Promise<obs-map>) :reset (fn [] -> Promise<obs-map>)
    :close (fn [] -> nil)} where obs-map = {:obs :reward :done :info}. :step POSTs
   {:action action} to /step, :reset POSTs {} to /reset. The RL contract lives here,
   in the agents layer — the membrane (genmlx.world.net) stays neutral.

   `:close` is intentionally INERT: this transport holds only the url, not the live
   listener, so it cannot stop the server. The server lifecycle is owned by
   genmlx.world.net/with-server (its p/finally tears the listener down) — never by
   the client transport."
  [url]
  {:step  (fn [action] (net/request url "/step" {:action action}))
   :reset (fn [] (net/request url "/reset"))
   :close (fn [] nil)})

;; ===========================================================================
;; 2. Serve an agents environment behind the membrane (server side)
;; ===========================================================================
;;
;; The world's transition randomness (when noise > 0) lives HERE, server-side, in a
;; world-owned key the agent never sees — genuine externality. At noise = 0 the
;; T-row is one-hot so the transition is deterministic and key-independent, which is
;; what makes across-wire == in-process parity exact.

(defn- world-step
  "Apply the world transition behind the membrane. Deterministic at noise 0; if a
   world-owned `kref` holds a key, advance it and sample (stochastic world)."
  [T kref s a]
  (if-let [k @kref]
    (let [[k1 k2] (rng/split k)]
      (reset! kref k2)
      (agent/sample-next T s a k1))
    (agent/sample-next T s a)))

(defn mdp-env-handler
  "Build the single genmlx.world.net/serve! handler (fn [route payload] -> map) that
   hosts an MDP behind the membrane. `mdp` is a build-mdp bundle
   ({:T :R :terminals :start-idx ...}). Options:
     :start — start state index (default the mdp's :start-idx)
     :key   — a world-owned RNG key for a STOCHASTIC world (omit ⇒ deterministic).
   /reset and /step yield {:obs s' :reward r :done term? :info {}} — the next state
   is the (fully observed) sensor; the reward is R[s,a]."
  [{:keys [T R terminals start-idx]} & [{:keys [start key]}]]
  (let [s0   (or start start-idx)
        st   (atom s0)
        kref (atom key)]
    (fn [route payload]
      (case route
        "/reset" (do (reset! st s0)
                     {:obs s0 :reward 0.0 :done (contains? terminals s0) :info {}})
        "/step"  (let [a (:action payload)]
                   (if-not (number? a)
                     {:error "step requires an integer :action"}
                     (let [s  @st
                           s' (world-step T kref s a)
                           r  (double (mx/item (mx/idx (mx/idx R s) a)))]
                       (reset! st s')
                       {:obs s' :reward r :done (contains? terminals s') :info {}})))
        {:error (str "unknown route " route)}))))

(defn pomdp-env-handler
  "Build the serve! handler hosting a POMDP (restaurant-gridworld bundle) behind the
   membrane. The world holds the TRUE physical location AND the hidden latent world;
   /step transitions the location via the true world's T and returns
   {:obs {:loc s' :sense o} ...} — the location is fully observed; `:sense` is the
   partial observation o = (observe true-world loc') that reveals the latent (a
   keyword is sent as its name; remote-pomdp-rollout decodes it). `pomdp-agent` is a
   make-pomdp-agent result; `env` carries :true-world and :start-idx. Options :start,
   :key as in mdp-env-handler."
  [pomdp-agent env & [{:keys [start key]}]]
  (let [true-world (:true-world env)
        true-mdp   (:mdp ((:world-agents pomdp-agent) true-world))
        T          (:T true-mdp)
        terminals  (:terminals true-mdp)
        observe    (:observe pomdp-agent)
        ;; EDN-encode the observation for the wire (JSON has no keywords/vectors):
        ;; pr-str round-trips ANY observe range (a keyword, nil, or a [restaurant
        ;; open?] vector) — remote-pomdp-rollout decodes with reader/read-string.
        enc        pr-str
        s0         (or start (:start-idx env))
        st         (atom s0)
        kref       (atom key)]
    (fn [route payload]
      (case route
        "/reset" (do (reset! st s0)
                     {:obs    {:loc s0 :sense (enc (observe true-world s0))}
                      :reward 0.0 :done (contains? terminals s0) :info {}})
        "/step"  (let [a (:action payload)]
                   (if-not (number? a)
                     {:error "step requires an integer :action"}
                     (let [s  @st
                           s' (world-step T kref s a)
                           o  (observe true-world s')]
                       (reset! st s')
                       {:obs    {:loc s' :sense (enc o)}
                        :reward 0.0 :done (contains? terminals s') :info {}})))
        {:error (str "unknown route " route)}))))

;; ===========================================================================
;; 3. Drive an agent against a REMOTE environment (client side)
;; ===========================================================================

(defn remote-mdp-rollout
  "Drive `agent` (a make-mdp-agent result) against a REMOTE MDP over `transport`
   (a gym-transport) for at most `horizon` steps, stopping at a terminal. Mirrors
   agent/simulate-mdp, but the transition crosses the wire: act (pure, client) →
   POST action (actuator) → receive {:obs s' :reward :done} (sensor) → repeat.
   Returns a PROMISE of {:states [s0 s1 ...] :actions [a0 ...] :rewards [r0 ...]} —
   the simulate-mdp shape plus per-step rewards. `:key` threads per-step policy
   sub-keys exactly as simulate-mdp does (split 3-ways, the transition half
   discarded since the world owns it); at alpha=##Inf it is irrelevant and the
   trajectory matches simulate-mdp bit-for-bit at noise 0."
  [{:keys [act]} transport horizon & [{:keys [key]}]]
  (pr/let [o0 ((:reset transport))]
    (let [start (:obs o0)]
      (pr/loop [s start, step 0, k key, done? (:done o0)
                states [start], actions [], rewards []]
        (if (or (>= step horizon) done?)
          {:states states :actions actions :rewards rewards}
          (let [[k-act _k-world k'] (if k (rng/split-n k 3) [nil nil nil])
                a (if k-act (act s k-act) (act s))]
            (pr/let [o ((:step transport) a)]
              (pr/recur (:obs o) (inc step) k' (:done o)
                        (conj states (:obs o)) (conj actions a) (conj rewards (:reward o))))))))))

(defn- decode-obs
  "Reverse `enc`: read the EDN wire form of an observation back to its value
   (keyword, nil, vector, …). The peer always sends a pr-str string; a non-string
   (defensive) passes through unchanged."
  [s]
  (if (string? s) (reader/read-string s) s))

(defn remote-pomdp-rollout
  "Drive a POMDP `agent` (make-pomdp-agent result) against a REMOTE POMDP over
   `transport` (a gym-transport). Mirrors pomdp/simulate-pomdp, but the transition +
   observation cross the wire: act from belief (pure) → POST action (actuator) →
   receive {:loc s' :sense o} (sensor) → FILTER the belief on o (the inbound
   constraint) → repeat. Returns a PROMISE of
   {:states :actions :observations :beliefs} (the simulate-pomdp shape). At
   alpha=##Inf / noise=0 the states, actions, observations and beliefs match
   simulate-pomdp exactly. (At finite alpha act uses auto-key, same as
   simulate-pomdp, so neither is per-step reproducible — hence no :key option.)"
  [{:keys [act update-belief prior]} transport horizon]
  (pr/let [o0 ((:reset transport))]
    (let [start (:loc (:obs o0))]
      (pr/loop [s start, b prior, step 0, done? (:done o0)
                states [start], actions [], obss [], beliefs [prior]]
        (if (or (>= step horizon) done?)
          {:states states :actions actions :observations obss :beliefs beliefs}
          (let [a (act b s)]
            (pr/let [o ((:step transport) a)]
              (let [obsmap (:obs o)
                    s'     (:loc obsmap)
                    sense  (decode-obs (:sense obsmap))
                    b'     (update-belief b s' sense)]
                (pr/recur s' b' (inc step) (:done o)
                          (conj states s') (conj actions a)
                          (conj obss sense) (conj beliefs b'))))))))))
