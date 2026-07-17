(ns genmlx.control.cybernetic
  "Cybernetic (comparator/TOTE) controlled-steppable — the control axis's second
   controller family (genmlx-cyst; the long-promised mct-1jj0 contribution).

   Wraps a base steppable {:init :step :done? :best} with a MONITOR -> COMPARE
   -> CODE -> GATE stop controller (the Metacognitive Control System shape of
   Wells 2019, Frontiers in Psychology 10:2621, generalized away from the
   psychology): per cycle it reads a scalar signal m_t off the base state (a
   pure read), runs a TOTE test m_t > tau with hysteresis / min-gap /
   min-engage arming, and on a pass draws stop ~ Bernoulli(gate-hazard). An
   optional CODE register decays on a fixed clock (r <- lambda*r every :every
   base advances, refreshed to 1 by every veto) and opens a second exit route
   stop ~ Bernoulli(eps*(1-r)). Scheduled sites (instructed trials) bypass the
   comparator entirely — the do()-operator that separates a product-form gate
   hazard (the ridge law in control_cybernetic_test.cljs).

   OBJECTIVE-FREE BY CONSTRUCTION: this controller is a discrepancy reducer (a
   thermostat), not a bargain-hunter. It never requires
   genmlx.control.decision-value or genmlx.inference.cost — the sibling
   constraint to decision-value/assert-downstream!, enforced by a grep-guard in
   the test file. The control axis is about WHEN TO STOP; value-of-computation
   (meta-mdp) is one answer, not the definition.

   THE STOP DECISIONS ARE FIRST-CLASS, INVERTIBLE OBJECTS (mct ADR-0002 /
   ADR-0015): generation draws each decision from the controller's PRNG key
   and records a site {:kind :gate|:decay|:scheduled :t :hazard :bit ...};
   `replay-gf` is the generative function over that site sequence, so
   p/generate constrained to the recorded bits (`episode-log-lik`) yields the
   trajectory log-likelihood — the same structure that generated it,
   distributionally identical. Controller latents are therefore recoverable
   from stopping behaviour alone (metacognitive profiling); the VOC controller
   has no analog of this (its stop is not a traced choice).

   Pure state transitions throughout: controller state (:t, arming flag, r,
   schedule queue, PRNG key, sites) threads through :step exactly as the
   base's state does; the wrapped steppable stays a value. Scalar GFI only —
   the replay gf branches on mx/item, incompatible with batched vsimulate
   (same limitation as the reference controlled-loop)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.gen :refer [gen]]))

(def ^:private EPS 1e-6)

(defn- clip01 [p] (-> p (max EPS) (min (- 1.0 EPS))))

(defn- ->num [x] (if (number? x) x (mx/item x)))

(defn- draw-bit
  "One Bernoulli(p) stop decision off the controller's PRNG key.
   Returns [bit next-key]."
  [key p]
  (let [[k1 k2] (rng/split (rng/ensure-key key))
        u (mx/item (rng/uniform k1 []))]
    [(if (< u p) 1 0) k2]))

(defn tote-controlled-steppable
  "Wrap `base` {:init :step :done? :best} with a cybernetic (comparator/TOTE)
   stop controller. Returns the same steppable shape. opts:
     :monitor      (fn [base-state] -> scalar m) — pure read        (REQUIRED)
     :tau          reference tolerance; the TOTE test is m > tau    (REQUIRED)
     :gate-hazard  (fn [ctrl-state] -> p) — stop prob on TOTE-pass  (REQUIRED)
     :hysteresis?  re-arm only after m falls back below tau (default true)
     :min-gap      base advances required after any gate/scheduled decision
                   before the next TOTE test (default 0)
     :min-engage   no TOTE tests before this many base advances (default 0)
     :register     {:lambda l :every n :eps e :refresh-on-veto? true} — the
                   decaying code register + decay exit route
                   (stop ~ Bernoulli(eps*(1-r)) at each decay boundary);
                   nil/absent = off, so the minimal TOTE gate is usable alone
     :schedule     [{:t k :hazard (fn [ctrl-state] -> p)} ...] — instructed
                   trials (the do()-operator): at base-advance count >= k the
                   next cycle draws stop ~ Bernoulli(hazard), bypassing the
                   comparator; a failed trial refreshes the register and
                   disarms, exactly like a veto
     :key          PRNG key for the controller's stop decisions (nil = fresh
                   entropy per :init; a fixed key = a reproducible episode
                   given a deterministic base)
     :trace?       record stop sites for replay inversion (default true)

   Decision cycles (gate / scheduled draws) do NOT advance the base; `t`
   counts base advances only (the reference-implementation convention). The
   gate/scheduled hazard fns receive the controller state with :m assoc'd.
   Exits: :goal (gate accept), :scheduled (instructed accept), :decay
   (register faded), :censored (the base's own done? fired first).
   :best returns {:result (base :best) :sites :exit :t :r}."
  [base {:keys [monitor tau gate-hazard hysteresis? min-gap min-engage
                register schedule key trace?]
         :or {hysteresis? true min-gap 0 min-engage 0 trace? true}}]
  (assert (fn? monitor) "cybernetic: :monitor fn is required")
  (assert (number? tau) "cybernetic: :tau is required")
  (assert (fn? gate-hazard) "cybernetic: :gate-hazard fn is required")
  (assert (or hysteresis? (pos? min-gap))
          (str "cybernetic: :hysteresis? false with :min-gap 0 livelocks after "
               "a veto (the comparator re-arms instantly and the base never "
               "advances) — enable one of them"))
  (let [binit (:init base) bstep (:step base)
        bdone? (:done? base) bbest (:best base)
        {:keys [lambda every eps refresh-on-veto?]
         :or {refresh-on-veto? true}} register
        _ (when register
            (assert (and (number? lambda) (integer? every) (pos? every)
                         (number? eps))
                    "cybernetic: :register needs {:lambda number :every pos-int :eps number}"))
        record (fn [cs site] (if trace? (update cs :sites conj site) cs))
        ;; a veto (gate or failed scheduled trial) refreshes the code register
        ;; and disarms the comparator: r <- 1, one full re-crossing of tau
        ;; (hysteresis) and min-gap base advances before the next test
        veto (fn [cs]
               (assoc cs
                      :r (if (and register refresh-on-veto?) 1.0 (:r cs))
                      :below? false :since-gate 0))]
    {:init
     (fn []
       {:base (binit)
        :key (rng/ensure-key key)
        :t 0 :r 1.0 :below? true :since-gate nil
        :sites [] :exit nil
        :schedule-q (vec (sort-by :t schedule))})

     :step
     (fn [{:keys [base key t r below? since-gate schedule-q] :as cs}]
       (cond
         (some? (:exit cs)) cs
         (bdone? base) (assoc cs :exit :censored)
         :else
         (let [m (->num (monitor base))
               view (assoc cs :m m)
               due (when (and (seq schedule-q) (>= t (:t (first schedule-q))))
                     (first schedule-q))
               armed? (and (> m tau)
                           (or (not hysteresis?) below?)
                           (or (nil? since-gate) (>= since-gate min-gap))
                           (>= t min-engage))
               decay-due? (and register (pos? t) (zero? (mod t every)))]
           (cond
             ;; --- instructed trial (do()-operator): bypasses the comparator
             due
             (let [pz (clip01 (->num ((:hazard due) view)))
                   [bit key'] (draw-bit key pz)
                   cs' (-> cs (assoc :key key' :schedule-q (subvec schedule-q 1))
                           (record {:kind :scheduled :t t :m m :hazard pz :bit bit}))]
               (if (= bit 1) (assoc cs' :exit :scheduled) (veto cs')))

             ;; --- comparator TOTE-pass -> gated exit attempt
             armed?
             (let [pz (clip01 (->num (gate-hazard view)))
                   [bit key'] (draw-bit key pz)
                   cs' (-> cs (assoc :key key')
                           (record {:kind :gate :t t :m m :hazard pz :bit bit}))]
               (if (= bit 1) (assoc cs' :exit :goal) (veto cs')))

             ;; --- ordinary cycle: decay check, then advance the base
             :else
             (let [[r' cs' exit?]
                   (if decay-due?
                     (let [r2 (* lambda r)
                           hz (clip01 (* eps (- 1.0 r2)))
                           [bit key'] (draw-bit key hz)]
                       [r2 (-> cs (assoc :key key')
                               (record {:kind :decay :t t :r r2 :hazard hz :bit bit}))
                        (= bit 1)])
                     [r cs false])]
               (if exit?
                 (assoc cs' :r r' :exit :decay)
                 (assoc cs' :base (bstep base) :r r' :t (inc t)
                        :below? (if (<= m tau) true below?)
                        :since-gate (when since-gate (inc since-gate)))))))))

     :done? (fn [cs] (or (some? (:exit cs)) (bdone? (:base cs))))

     :best (fn [cs] {:result (bbest (:base cs))
                     :sites (:sites cs)
                     :exit (or (:exit cs)
                               (when (bdone? (:base cs)) :censored))
                     :t (:t cs) :r (:r cs)})}))

;; ---------------------------------------------------------------------------
;; GFI inversion: the replay generative function over a recorded site sequence
;; ---------------------------------------------------------------------------

(defn recorded-hazard
  "The theta-free replay hazard: score each site at exactly the hazard it was
   generated with (the identity-law default). For inversion, pass a hazard-fn
   re-deriving each site kind's hazard from the unknown params, e.g.
     (fn [site [a b]] (case (:kind site)
                        :gate      (* (- 1.0 a) b)
                        :scheduled b
                        :decay     (:hazard site)))"
  [site _params]
  (:hazard site))

(defn replay-gf
  "The companion inversion object: a generative function over a recorded
   stop-site sequence. Traces :stop0..:stopN-1 ~ Bernoulli(hazard-fn site
   params) in order, terminating at the first fired bit (the controlled-loop
   structure, mct ADR-0002 'traced stop'). p/generate constrained to the
   recorded bits yields the trajectory log-likelihood — the same structure
   that generated the episode; the same program p/simulates fresh stop
   trajectories over the sites."
  [sites hazard-fn]
  (let [sites (vec sites)
        n (count sites)]
    (dyn/auto-key
     (gen [params]
       (if (zero? n)
         {:stop-index nil :stopped? false}
         (loop [i 0]
           (let [pz (clip01 (->num (hazard-fn (nth sites i) params)))
                 s (trace (keyword (str "stop" i)) (dist/bernoulli pz))
                 fired? (>= (mx/item s) 1.0)]
             (if (or fired? (>= (inc i) n))
               {:stop-index (when fired? i) :stopped? fired?}
               (recur (inc i))))))))))

(defn sites->constraints
  "Choicemap constraining :stopI to each recorded site's bit."
  [sites]
  (reduce (fn [c [i site]]
            (cm/set-value c (keyword (str "stop" i)) (mx/scalar (:bit site))))
          cm/EMPTY (map-indexed vector sites)))

(defn episode-log-lik
  "log p(recorded stop trajectory | params): constrained p/generate replay of
   `sites` through replay-gf — the GFI inversion consumer (the reference
   implementation's episode-loglik2, generalized over the site kinds)."
  [sites hazard-fn params]
  (if (empty? sites)
    0.0
    (mx/item (:weight (p/generate (replay-gf sites hazard-fn) [params]
                                  (sites->constraints sites))))))
