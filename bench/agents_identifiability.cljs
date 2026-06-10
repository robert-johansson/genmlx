(ns bench.agents-identifiability
  "Identifiability-ridge contrast: diagnostic vs non-diagnostic observations.

   The gating-novelty claim for the agents axis: inverting an agent only
   concentrates the posterior where the observed behavior is DIAGNOSTIC —
   where the competing hypotheses assign different likelihoods to the observed
   actions. On non-diagnostic data the posterior stays at the prior, and flat
   'ridge' directions of the hypothesis space stay flat no matter how the
   identified directions contract. (Cf. Evans, Stuhlmueller & Goodman 2016,
   'Learning the Preferences of Ignorant, Inconsistent Agents', AAAI.)

   Two panels, both exact inference on existing agents-as-GFs code:

   Panel A — TRAJECTORY contrast (temptation MDP, biased-agent GF).
     The same exact bias posterior (assess-enumeration over {:naive,
     :sophisticated}) on two observation sites:
       diagnostic     — n safe actions at the START state, where the naive
                        policy is an exact [0.5 0.5] tie but the sophisticated
                        agent prefers the safe route. At alpha=Inf the per-obs
                        likelihood ratio is exactly 1:2, giving the closed form
                        P(:soph | n) = 1/(1 + 2^-n); at alpha=2 the same shape,
                        softer.
       non-diagnostic — n actions at the SAFE1 state, where both actions lead
                        to veg: the EU rows are identical across actions AND
                        biases, so the policy is [0.5 0.5] under both
                        hypotheses at ANY alpha, the likelihood cancels, and
                        the posterior equals the prior EXACTLY for every n.

   Panel B — DIMENSION ridge (restaurant-choice IRL, agentmodels Ch 4).
     Exact enumeration over the 3x3x3 utility-table grid: the observed step(s)
     toward Donut-South contract the DONUT marginal, while the VEG and NOODLE
     marginals stay ~uniform — the flat ridge direction the observation says
     nothing about. Three regimes show the diagnosticity gate: alpha=2 with one
     step (the chapter's near-uninformative baseline), alpha=100 with one step
     (the same step IS diagnostic for a near-rational agent), alpha=100 with
     the 4-step Donut-South trajectory (evidence accumulates). Side-by-side
     marginals + the veg x noodle joint table are emitted as figure data.

   Output: results/agents-identifiability/data.json
   Usage:  bun run --bun nbb bench/agents_identifiability.cljs"
  (:require [clojure.string :as str]
            [agentmodels.biased-inverse :as bi]
            [agentmodels.irl :as irl]
            [genmlx.agents.inverse :as inv]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/agents-identifiability")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(defn entropy
  "Shannon entropy in nats of a probability collection."
  [ps]
  (- (reduce + (map #(if (pos? %) (* % (js/Math.log %)) 0.0) ps))))

(defn alpha-label
  "JSON-safe alpha: ##Inf stringifies to null through JSON.stringify, so encode
   it as the string \"Inf\"."
  [alpha]
  (if (= alpha ##Inf) "Inf" alpha))

;; ===========================================================================
;; Panel A — trajectory contrast on the biased-agent GF
;; ===========================================================================

(def N-OBS-GRID [0 1 2 3 4 5 6])
(def PANEL-A-ALPHAS [##Inf 2.0])

(def mdp (bi/temptation-mdp))

(defn- bias-cfg [alpha states actions]
  {:mdp mdp :alpha alpha :discount 1.0 :n-iters 10
   :states states :actions actions})

(def prior-entropy-bias (entropy [0.5 0.5]))

(defn- posterior-row [alpha n state action]
  (let [post (bi/bias-posterior (bias-cfg alpha
                                          (vec (repeat n state))
                                          (vec (repeat n action))))
        h    (entropy (vals post))]
    {:n_obs n
     :posterior post
     :entropy_nats h
     :contraction_nats (- prior-entropy-bias h)}))

(defn- run-panel-a-alpha
  "Diagnostic (safe at START) and non-diagnostic (any action at SAFE1) curves
   at one alpha, with two independent cross-checks: the diagnostic curve against
   the policy-idiom closed form 1/(1 + r^n), and the non-diagnostic posterior
   against the prior (must match exactly)."
  [alpha]
  (let [agents  (bi/bias-agents {:mdp mdp :alpha alpha :discount 1.0 :n-iters 10})
        pi-n    (js/Math.exp (inv/action-loglik (:naive agents) 0 1))
        pi-s    (js/Math.exp (inv/action-loglik (:sophisticated agents) 0 1))
        r       (/ pi-n pi-s)
        diag    (mapv #(posterior-row alpha % 0 1) N-OBS-GRID)
        nondiag (mapv #(posterior-row alpha % 2 0) N-OBS-GRID)
        ref-err (apply max (map (fn [{:keys [n_obs posterior]}]
                                  (js/Math.abs (- (:sophisticated posterior)
                                                  (/ 1.0 (+ 1.0 (js/Math.pow r n_obs))))))
                                diag))
        nd-err  (apply max (map (fn [{:keys [posterior]}]
                                  (js/Math.abs (- (:sophisticated posterior) 0.5)))
                                nondiag))]
    (println (str "\n  alpha = " (alpha-label alpha)
                  "   per-obs likelihood ratio r = pi_n(a1|s0)/pi_s(a1|s0) = " (.toFixed r 4)))
    (println "  n   P(:soph|diag)  H(diag)   P(:soph|nondiag)  H(nondiag)")
    (doseq [[d nd] (map vector diag nondiag)]
      (println (str "  " (:n_obs d)
                    "   " (.toFixed (:sophisticated (:posterior d)) 4)
                    "         " (.toFixed (:entropy_nats d) 4)
                    "    " (.toFixed (:sophisticated (:posterior nd)) 4)
                    "            " (.toFixed (:entropy_nats nd) 4))))
    (println (str "  diagnostic curve vs closed form 1/(1+r^n): max |diff| = "
                  (.toExponential ref-err 2)
                  "   non-diagnostic vs prior: max |diff| = "
                  (.toExponential nd-err 2)))
    {:alpha (alpha-label alpha)
     :per_obs_likelihood_ratio r
     :diagnostic diag
     :non_diagnostic nondiag
     :cross_checks
     {:diagnostic_closed_form {:form "P(:soph | n safe) = 1/(1 + r^n), r from the independent policy idiom"
                               :max_abs_diff ref-err}
      :non_diagnostic_equals_prior {:max_abs_diff nd-err}}}))

(defn run-panel-a []
  (println (str "\n" (apply str (repeat 60 "-"))))
  (println "  Panel A — trajectory contrast (temptation MDP)")
  (println (apply str (repeat 60 "-")))
  {:description "Exact bias posterior vs number of observations, diagnostic (start state) vs non-diagnostic (safe1 state) trajectories, at alpha=Inf (closed form 1/(1+2^-n)) and alpha=2 (same shape, softer)"
   :model {:source "examples/agentmodels/biased_inverse.cljs"
           :mdp "temptation (5 states, 2 actions)"
           :diagnostic_site "state 0 (start): naive policy ties [0.5 0.5], sophisticated prefers safe — finite likelihood ratio per observation"
           :non_diagnostic_site "state 2 (safe1): both actions lead to veg, EU rows identical across actions and biases — policy [0.5 0.5] under BOTH hypotheses at any alpha"}
   :prior {:naive 0.5 :sophisticated 0.5 :entropy_nats prior-entropy-bias}
   :n_obs_grid N-OBS-GRID
   :series (mapv run-panel-a-alpha PANEL-A-ALPHAS)})

;; ===========================================================================
;; Panel B — dimension ridge in restaurant-choice IRL
;; ===========================================================================

(defn- marginal
  "Marginalize a {[di vi ni ti ai] -> prob} posterior onto tuple position `idx`,
   labelling outcomes with `vals-at-idx`."
  [posterior idx vals-at-idx]
  (let [m (reduce (fn [acc [tup pr]] (update acc (nth tup idx) (fnil + 0.0) pr))
                  {} posterior)]
    (mapv #(get m % 0.0) (range (count vals-at-idx)))))

(def h-prior-3 (js/Math.log 3))

(defn- marginal-stats [probs]
  {:values (vec irl/food-vals)
   :probs probs
   :entropy_nats (entropy probs)
   :contraction_nats (- h-prior-3 (entropy probs))})

(defn- run-panel-b-regime [label alpha observations obs-label]
  (let [spec      {:donut-vals irl/food-vals :veg-vals irl/food-vals
                   :noodle-vals irl/food-vals
                   :time-cost-vals [-0.04] :alpha-vals [alpha]}
        agents    (irl/build-agents spec)
        posterior (irl/joint-posterior agents observations)
        donut-m   (marginal posterior 0 irl/food-vals)
        veg-m     (marginal posterior 1 irl/food-vals)
        noodle-m  (marginal posterior 2 irl/food-vals)
        vn-joint  (vec (for [vi (range 3)]
                         (vec (for [ni (range 3)]
                                (reduce (fn [acc [tup pr]]
                                          (if (and (= (nth tup 1) vi) (= (nth tup 2) ni))
                                            (+ acc pr) acc))
                                        0.0 posterior)))))
        summ      (irl/summarize posterior agents)
        pri       (irl/prior-summary agents)]
    (println (str "\n  [" label "] alpha=" alpha ", obs=" obs-label))
    (println (str "    P(favourite = donut): prior " (.toFixed (:p-donut-favorite pri) 4)
                  " -> posterior " (.toFixed (:p-donut-favorite summ) 4)
                  "   (veg " (.toFixed (:p-veg-favorite summ) 4)
                  ", noodle " (.toFixed (:p-noodle-favorite summ) 4) ")"))
    (doseq [[nm m] [["donut" donut-m] ["veg" veg-m] ["noodle" noodle-m]]]
      (println (str "    " (.padEnd nm 7) " marginal ["
                    (str/join " " (map #(.toFixed % 4) m))
                    "]  H = " (.toFixed (entropy m) 4)
                    "  (contraction " (.toFixed (- h-prior-3 (entropy m)) 4) ")")))
    {:label label
     :alpha alpha
     :observations obs-label
     :marginals {:donut  (marginal-stats donut-m)
                 :veg    (marginal-stats veg-m)
                 :noodle (marginal-stats noodle-m)}
     :veg_noodle_joint {:rows "veg value 0..2" :cols "noodle value 0..2"
                        :probs vn-joint}
     :p_favorite {:prior {:donut (:p-donut-favorite pri)
                          :veg (:p-veg-favorite pri)
                          :noodle (:p-noodle-favorite pri)}
                  :posterior {:donut (:p-donut-favorite summ)
                              :veg (:p-veg-favorite summ)
                              :noodle (:p-noodle-favorite summ)}}}))

(defn run-panel-b []
  (println (str "\n" (apply str (repeat 60 "-"))))
  (println "  Panel B — dimension ridge (restaurant IRL)")
  (println (apply str (repeat 60 "-")))
  {:description "Exact utility-table posterior over the 3x3x3 grid: the donut dimension contracts with diagnostic evidence, the veg/noodle ridge stays ~flat in every regime"
   :model {:source "examples/agentmodels/irl.cljs"
           :grid "restaurant-temptation (agentmodels Ch 4 / 5e geometry)"
           :spec "utility-only: donut/veg/noodle in {0,1,2}, timeCost=-0.04 fixed"}
   :prior {:marginal_each_value [(/ 1.0 3) (/ 1.0 3) (/ 1.0 3)]
           :entropy_nats h-prior-3}
   :regimes
   [(run-panel-b-regime "baseline-alpha2-1step" 2.0 irl/single-left-obs
                        "single leftward step (chapter baseline: low rationality washes out the evidence)")
    (run-panel-b-regime "alpha100-1step" 100.0 irl/single-left-obs
                        "single leftward step (near-rational agent: the same step IS diagnostic)")
    (run-panel-b-regime "alpha100-4step" 100.0 irl/donut-south-obs
                        "4-step Donut-South trajectory (evidence accumulates)")]})

;; ===========================================================================
;; Main
;; ===========================================================================

(println (apply str (repeat 70 "=")))
(println "  Agents axis: identifiability-ridge contrast (diagnostic vs non-diagnostic)")
(println (apply str (repeat 70 "=")))

(def t0 (js/performance.now))
(def panel-a (run-panel-a))
(def panel-b (run-panel-b))
(def elapsed-ms (- (js/performance.now) t0))

(println (str "\n" (apply str (repeat 70 "="))))
(println "  SUMMARY")
(println (apply str (repeat 70 "=")))
(let [a-inf     (first (:series panel-a))
      a-diag    (last (:diagnostic a-inf))
      a-nondiag (last (:non_diagnostic a-inf))
      b-hi      (second (:regimes panel-b))]
  (println (str "  A (alpha=Inf): after " (:n_obs a-diag) " diagnostic obs, entropy "
                (.toFixed prior-entropy-bias 4) " -> " (.toFixed (:entropy_nats a-diag) 4)
                " nats; after " (:n_obs a-nondiag) " non-diagnostic obs, "
                (.toFixed (:entropy_nats a-nondiag) 4) " (= prior, exactly)"))
  (println (str "  B (alpha=100, 1 step): donut marginal contracts by "
                (.toFixed (get-in b-hi [:marginals :donut :contraction_nats]) 4)
                " nats; veg/noodle ridge: "
                (.toFixed (get-in b-hi [:marginals :veg :contraction_nats]) 4) " / "
                (.toFixed (get-in b-hi [:marginals :noodle :contraction_nats]) 4)
                " (stays ~flat)")))
(println (str "  Total: " (.toFixed (/ elapsed-ms 1000) 1) "s"))

(write-json "data.json"
  {:experiment "agents-identifiability"
   :description "Identifiability-ridge contrast: posterior concentrates on diagnostic observations, stays at the prior on non-diagnostic ones; flat ridge directions stay flat"
   :citation "Evans, Stuhlmueller & Goodman (2016), 'Learning the Preferences of Ignorant, Inconsistent Agents', AAAI"
   :timestamp (.toISOString (js/Date.))
   :panel_a_trajectory_contrast panel-a
   :panel_b_dimension_ridge panel-b
   :duration_ms elapsed-ms})

(println "\nAgents identifiability experiment complete.")
