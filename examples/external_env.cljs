(ns external-env
  "FLAGSHIP — the FIRST EXTERNAL ENVIRONMENT (ROADMAP Phase 3, item 5). The agent/
   environment boundary reified as REAL network I/O over the Bun world membrane
   (genmlx.world.net), with the agent staying a pure generative function above it.

   In GFI terms the boundary is two directed crossings:
     • ACTUATOR — an outbound sampled action committed to the world (an HTTP POST).
     • SENSOR   — an inbound observation read back (the response), the datum a model
       conditions on.

   WHAT THIS CLAIMS: the agent/env boundary is a real serialize → IO → deserialize
   round-trip over Bun network; the SERVER owns the world's true state and the
   randomness the agent must model; the agent reaches the world ONLY through the
   transport seam (it never imports or calls the server handler directly, never
   shares the world's entropy, never assesses the service's density).
   WHAT THIS DOES NOT CLAIM: network robustness, remote hosts, auth, fault
   tolerance, or that the peer is untrusted. This is the membrane PATTERN at the
   seam (the AUTONOMOUS face of genmlx-gsoi), not a hardened network stack.

   Three self-checking sections, each spinning up a localhost environment SERVER and
   reaching it ONLY through the membrane (the agent never touches Bun):

   1. MEMBRANE FAITHFULNESS (MDP). The SAME make-mdp-agent runs an episode across
      the wire; at the deterministic regime (alpha = ##Inf, noise = 0) the
      across-wire trajectory is BIT-IDENTICAL to the in-process simulate-mdp run —
      proof the membrane is faithful and thin (the agent code is unchanged whether
      the environment is in-process or external). The corridor world is tie-free, so
      the argmax policy has a unique action at every state; the server only performs
      the deterministic transition (the client computes all actions), so there is no
      Q to drift across the wire.

   2. SENSOR / ACTUATOR + INFERENCE (POMDP). A restaurant-gridworld whose latent
      world is hidden behind the membrane; the agent POSTs actions (actuators),
      receives observations (sensors), and FILTERS its belief on them (the inbound
      constraint). The whole trajectory + belief sequence match in-process
      simulate-pomdp exactly, and the belief snaps to the true world after the
      signpost observation arrives across the wire.

   3. AN API IS AN ENVIRONMENT — MODEL IT, DON'T GF IT. A localhost 'noisy oracle'
      service owns a hidden latent AND a noise source whose scale the agent does NOT
      know exactly. The agent holds a PURE, APPROXIMATE observation model
      p(reading | theta) as a generative function, queries the service (actuators),
      and conditions that model on the returns (sensors) — importance sampling
      recovers the latent UNDER THE CLIENT'S MODEL. The service is never a generative
      function (the client never uses the server's true noise params, never assesses
      its density); we model what it RETURNS. The estimate is cross-checked against
      the exact conjugate posterior of the client's model (an independent oracle) and
      the true latent.

   Localhost only — no external network egress, no credentials. Clean teardown via
   genmlx.world.net/with-server (no leaked listeners). Zero engine change.

   Run:  bun run --bun nbb examples/external_env.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.worlds :as worlds]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as penv]
            [genmlx.agents.remote :as remote]
            [genmlx.world.net :as net]
            [genmlx.inference.importance :as is]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; tiny self-check harness (same shape as the other flagship examples)
;; ---------------------------------------------------------------------------

(def ^:private fails (atom 0))

(defn- check-true [label ok]
  (println (str (if ok "  ✓ " "  ✗ FAIL ") label))
  (when-not ok (swap! fails inc)) ok)

(defn- check-close [label expected actual tol]
  (let [ok (<= (abs (- expected actual)) tol)]
    (println (str (if ok "  ✓ " "  ✗ FAIL ") label
                  " — expected " (.toFixed expected 4) ", got " (.toFixed actual 4)))
    (when-not ok (swap! fails inc)) ok))

(defn- belief-max-err [bs1 bs2 worlds]
  (apply max 0.0 (for [[a b] (map vector bs1 bs2), w worlds]
                   (Math/abs (- (double (get a w 0.0)) (double (get b w 0.0)))))))

;; an uncaught promesa rejection EXITS 0 (verified) — which would let the self-check
;; pass vacuously. Make any unhandled rejection a loud, nonzero failure.
(js/process.on "unhandledRejection"
               (fn [e] (println (str "\n✗ FAIL: unhandled rejection: "
                                     (or (and e (.-message e)) e)))
                 (js/process.exit 1)))

;; ===========================================================================
;; Section 1 — MDP across the wire: membrane faithfulness
;; ===========================================================================

(defn- section-1-mdp []
  (println "\n========== 1. MDP across the wire — membrane faithfulness ==========")
  (let [n       7
        mdp     (worlds/line-mdp {:n n})            ; 1×7 corridor, goal at idx 6, tie-free
        ag      (agent/make-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 24})
        start   (:start-idx mdp)
        H       12
        in-proc (agent/simulate-mdp ag start H)]
    (check-true "agent is in the deterministic regime (alpha = ##Inf)"
                (= ##Inf (:alpha (:params ag))))
    (net/with-server (remote/mdp-env-handler mdp {:start start})
      (fn [url]
        (pr/let [rem (remote/remote-mdp-rollout ag (remote/gym-transport url) H)]
          (println (str "  in-process states: " (:states in-proc)))
          (println (str "  across-wire states: " (:states rem)))
          (check-true "across-wire states == in-process states (exact parity)"
                      (= (:states in-proc) (:states rem)))
          (check-true "across-wire actions == in-process actions (exact parity)"
                      (= (:actions in-proc) (:actions rem)))
          (check-true (str "agent reached the goal (idx " (dec n) ") across the wire")
                      (= (dec n) (last (:states rem))))
          (check-true "each step was a real network round-trip (>=1 transition)"
                      (pos? (count (:actions rem))))
          :ok)))))

;; ===========================================================================
;; Section 2 — POMDP across the wire: sensor / actuator + belief filtering
;; ===========================================================================

(def ^:private pgrid
  [[:A    :empty :B]
   [:wall :empty :wall]
   [:wall :empty :wall]
   [:wall :empty :wall]
   [:wall :empty :wall]])
(def ^:private signpost 7)

(defn- section-2-pomdp []
  (println "\n========== 2. POMDP across the wire — sensor/actuator + belief ==========")
  (let [tw      :A
        env     (penv/restaurant-gridworld {:grid pgrid :goals [:A :B] :signpost signpost
                                            :true-world tw :start [1 4]})
        pa      (pomdp/make-pomdp-agent (assoc env :alpha ##Inf :gamma 1.0 :n-iters 40))
        start   (:start-idx env)
        worlds  (:worlds pa)
        H       12
        in-proc (pomdp/simulate-pomdp pa env start H)]
    (net/with-server (remote/pomdp-env-handler pa env {:start start})
      (fn [url]
        (pr/let [rem (remote/remote-pomdp-rollout pa (remote/gym-transport url) H)]
          (println (str "  in-process states: " (:states in-proc)))
          (println (str "  across-wire states: " (:states rem)))
          (println (str "  across-wire observations (sensor): " (:observations rem)))
          (println (str "  final belief: " (last (:beliefs rem))))
          (check-true "across-wire states == in-process states (exact parity)"
                      (= (:states in-proc) (:states rem)))
          (check-true "across-wire actions == in-process actions (exact parity)"
                      (= (:actions in-proc) (:actions rem)))
          (check-true "across-wire observations == in-process observations (sensor parity)"
                      (= (:observations in-proc) (:observations rem)))
          (check-true "belief sequence == in-process (max err < 1e-5)"
                      (< (belief-max-err (:beliefs in-proc) (:beliefs rem) worlds) 1e-5))
          (check-true (str "sensor at the signpost was received across the wire (true world " (name tw) ")")
                      (boolean (some #{tw} (:observations rem))))
          (check-true "the across-wire belief actually updated (final != prior)"
                      (not= (last (:beliefs rem)) (:prior pa)))
          (check-close (str "belief snapped to the true world P(" (name tw) ")=1")
                       1.0 (double (get (last (:beliefs rem)) tw 0.0)) 1e-6)
          :ok)))))

;; ===========================================================================
;; Section 3 — an API IS an environment: model it, don't GF it
;; ===========================================================================
;;
;; The agent's PURE, APPROXIMATE model of the external oracle: theta ~ prior, and
;; each reading the service returns is assumed N(theta, obs-sd). The server's TRUE
;; noise scale (server-obs-sd) DIFFERS — the agent does not know the channel exactly.
;; We never assess the SERVICE's density; we only condition THIS model on the values
;; it returns (identical discipline to a real sensor: the sensor is not a GF, but
;; p(reading | state) is).

(def ^:private prior-mu      0.0)
(def ^:private prior-sd      3.0)
(def ^:private obs-sd        1.0)    ; the CLIENT's assumed noise scale (a modelling assumption)
(def ^:private server-obs-sd 0.7)    ; the oracle's TRUE noise scale (agent never uses this)
(def ^:private K             6)      ; number of queries (actuators)
(def ^:private theta*        1.5)    ; the latent the oracle holds (agent never sees it)

(def ^:private oracle-model
  (gen []
    (let [theta (trace :theta (dist/gaussian prior-mu prior-sd))]
      (dotimes [i K]
        (trace (keyword (str "r" i)) (dist/gaussian theta obs-sd)))
      theta)))

(defn- collect-readings
  "Issue K sequential queries (actuators) and gather the K readings (sensors), each
   a real POST across the neutral net/request crossing (no fake reward/done)."
  [url k]
  (pr/loop [i 0, acc []]
    (if (>= i k)
      acc
      (pr/let [resp (net/request url "/query" {:i i})]
        (pr/recur (inc i) (conj acc (:reading resp)))))))

(defn- section-3-api []
  (println "\n========== 3. an API is an environment — model it, don't GF it ==========")
  (let [oracle-key (atom (rng/fresh-key 7))      ; world-owned noise (agent does NOT own it)
        handler    (fn [route _payload]
                     (case route
                       "/query" (let [[k1 k2] (rng/split @oracle-key)]
                                  (reset! oracle-key k2)
                                  {:reading (mx/item (dist/sample (dist/gaussian theta* server-obs-sd) k1))})
                       {:error (str "unknown route " route)}))]
    (net/with-server handler
      (fn [url]
        (pr/let [readings (collect-readings url K)]
          (let [;; --- exact conjugate posterior of the CLIENT'S model (independent oracle) ---
                sum-r    (reduce + readings)
                prec0    (/ 1.0 (* prior-sd prior-sd))
                precN    (+ prec0 (/ K (* obs-sd obs-sd)))
                post-var (/ 1.0 precN)
                post-sd  (Math/sqrt post-var)
                post-mu  (* post-var (+ (* prec0 prior-mu) (/ sum-r (* obs-sd obs-sd))))
                ;; --- importance sampling over the PURE model, conditioned on the
                ;;     sensor readings (the service is never a GF) ---
                constraints (cm/from-map (into {} (map-indexed
                                                    (fn [i r] [(keyword (str "r" i)) (mx/scalar r)])
                                                    readings)))
                {:keys [traces log-weights]} (is/importance-sampling
                                               {:samples 4000 :key (rng/fresh-key 42)}
                                               oracle-model [] constraints)
                lw     (mapv mx/item log-weights)
                m      (apply max lw)
                ws     (mapv #(Math/exp (- % m)) lw)
                z      (reduce + ws)
                thetas (mapv #(mx/item (:retval %)) traces)
                is-mu  (/ (reduce + (map * ws thetas)) z)]
            (println (str "  readings (sensors across the wire): " (mapv #(.toFixed % 3) readings)))
            (println (str "  exact conjugate posterior (client model): mean " (.toFixed post-mu 4)
                          "  sd " (.toFixed post-sd 4)))
            (println (str "  importance-sampling posterior mean: " (.toFixed is-mu 4)
                          "  (true theta* = " theta* ")"))
            (check-true (str "all " K " readings crossed the network as real queries")
                        (= K (count readings)))
            (check-true "every reading is a finite number (a real sensor value)"
                        (every? #(and (number? %) (js/isFinite %)) readings))
            (check-true "the server OWNS the noise: readings vary (the client holds no entropy over them)"
                        (> (count (distinct readings)) 1))
            (check-true "the client never used the server's true noise scale (model is only an approximation)"
                        (not= obs-sd server-obs-sd))
            (check-true "the conjugate posterior identifies the latent (|mean - theta*| < 3 sd)"
                        (< (abs (- post-mu theta*)) (* 3.0 post-sd)))
            (check-close "importance-sampling posterior matches the exact conjugate posterior"
                         post-mu is-mu 0.2)
            :ok))))))

;; ===========================================================================
;; driver
;; ===========================================================================

(defn run
  "Run all three sections in sequence (each spins up + tears down its own server).
   Returns a promise resolving to :ok / exiting nonzero on any failed check. A
   watchdog converts a wedged network round-trip into a loud exit(124) rather than
   letting it hang to the slow-tier cap."
  []
  (let [wd (js/setTimeout
             (fn [] (println "\n✗ FAIL: watchdog timeout — external-env wedged") (js/process.exit 124))
             90000)]
    (-> (pr/let [_ (section-1-mdp)
                 _ (section-2-pomdp)
                 _ (section-3-api)]
          (js/clearTimeout wd)
          (println (str "\n" (if (zero? @fails) "ALL CHECKS PASSED ✓"
                                 (str @fails " CHECK(S) FAILED ✗"))))
          (when (pos? @fails) (js/process.exit 1))
          :ok)
        (pr/catch (fn [e]
                    (js/clearTimeout wd)
                    (println (str "\n✗ FAIL: external-env flagship errored: " (.-message e)
                                  "\n" (.-stack e)))
                    (js/process.exit 1))))))

(defn run-or-skip
  "Run the self-checking flagship, or skip cleanly if the Bun network membrane is
   unavailable (e.g. not running under Bun). Returns a promise (the test wrapper
   awaits it as its top-level promise)."
  []
  (if-not (net/available?)
    (do (println "\nSKIP: Bun network membrane unavailable (run under `bun run --bun nbb`). (clean skip, exit 0)")
        (pr/resolved :skipped))
    (run)))

(def ^:private main?
  (boolean (some #(re-find #"external_env\.cljs" (str %)) (array-seq js/process.argv))))

(when main? (run-or-skip))
