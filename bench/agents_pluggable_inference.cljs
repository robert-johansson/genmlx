(ns bench.agents-pluggable-inference
  "Pluggable inference on ONE agent generative function (agentmodels Ch 6).

   The agents-axis claim: an agent is a generative function, and the inference
   backend used to invert it is pluggable and orthogonal to the agent definition.
   This artifact takes the SAME gridworld goal-inference GF
   (examples/agentmodels/ch06_pluggable_inference.cljs) and reads its posterior
   P(goal | observed actions) off THREE ways:

     EXACT — enumerate the finite goal prior, p/assess the full trajectory,
             normalize. Closed form (no sampling) — the ground truth.
     IS    — sample :goal from the prior, constrain the actions, group normalized
             particle weights by sampled goal.
     MH    — Metropolis-Hastings over the :goal latent (prior-proposal regenerate).

   and measures TV(exact, sampler) as a function of the sample budget N, with
   replicate seeds per N. The claim is convergence: IS -> exact and MH -> exact,
   TV < eps, at the canonical N^(-1/2) Monte Carlo rate — NEVER 'exact = sampled'
   (a single sampling run is an estimate, not the posterior). The posterior is
   deliberately NON-degenerate ({:a ~0.74, :b ~0.21, :c ~0.04}) so agreement is a
   real three-way match, not a 0/1 collapse.

   A fourth backend — gradient/amortized recovery of a CONTINUOUS utility vector
   through the differentiable planner (genmlx.agents.differentiable) — closes the
   chapter: when the latent is continuous, the inference backend swaps to Adam.
   We record its loss-history (the optimization 'figure').

   Finite-alpha regime: alpha = 2.0. At alpha = ##Inf the softmax policies become
   argmax 0/1 indicators and the sampling likelihoods degenerate; alpha = 2 keeps
   every goal hypothesis at finite likelihood so the weights genuinely vary.

   Output: results/agents-pluggable-inference/data.json
   Usage:  bun run --bun nbb bench/agents_pluggable_inference.cljs"
  (:require [agentmodels.ch06-pluggable-inference :as ch06]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

;; ---------------------------------------------------------------------------
;; Infrastructure (matches sibling bench/agents_is_exact.cljs)
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/agents-pluggable-inference")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(defn mean [xs] (/ (reduce + xs) (count xs)))

(defn std [xs]
  (let [m (mean xs)]
    (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) xs))
                     (max 1 (dec (count xs)))))))

(defn log-log-slope
  "Least-squares slope of log10(TV) vs log10(N) — expect ~ -1/2 (Monte Carlo rate)."
  [ns tvs]
  (let [xs (map #(js/Math.log10 %) ns)
        ys (map #(js/Math.log10 (max % 1e-12)) tvs)
        mx- (mean xs) my- (mean ys)
        sxy (reduce + (map (fn [x y] (* (- x mx-) (- y my-))) xs ys))
        sxx (reduce + (map (fn [x] (let [d (- x mx-)] (* d d))) xs))]
    {:slope (/ sxy sxx) :intercept (- my- (* (/ sxy sxx) mx-))}))

;; ---------------------------------------------------------------------------
;; Experiment configuration
;; ---------------------------------------------------------------------------

;; Default sweep tops out at N=1000 — MH is sequential, so the large-N tail
;; dominates wall-clock, and the N^(-1/2) rate is already unmistakable by 1000
;; (going to 3000 would ~triple runtime for a cosmetic extra point).
;; GENMLX_BENCH_QUICK=1 runs a reduced grid (smoke / CI).
(def quick? (some? (aget (.-env js/process) "GENMLX_BENCH_QUICK")))
(def N-GRID (if quick? [10 30 100] [10 30 100 300 1000]))
;; Fewer replicates at the expensive large-N tail (its TV is small and already
;; well-estimated); more where it is cheap and the variance matters.
(defn replicates [n] (if quick? 4 (if (>= n 1000) 8 (if (>= n 300) 12 16))))

(def actions ch06/obs-actions)
(def exact (ch06/exact-goal-posterior actions))   ; ground truth, computed once

;; ---------------------------------------------------------------------------
;; Convergence sweep: TV(exact, sampler) vs N for IS and MH
;; ---------------------------------------------------------------------------

(defn is-curve []
  (vec
    (for [[ni n] (map-indexed vector N-GRID)]
      (let [reps (replicates n)
            runs (mapv (fn [rep]
                         (let [seed (+ 10000 (* 1000 ni) rep)
                               {:keys [posterior ess]} (ch06/is-goal-posterior actions n (rng/fresh-key seed))]
                           (mx/clear-cache!)
                           {:tv (ch06/tv exact posterior) :ess ess}))
                       (range reps))
            tvs (mapv :tv runs) esss (mapv :ess runs)]
        {:n n :replicates reps
         :tv_mean (mean tvs) :tv_std (std tvs) :tv_max (apply max tvs)
         :ess_mean (mean esss) :ess_frac (/ (mean esss) n)}))))

(defn mh-curve []
  (vec
    (for [[ni n] (map-indexed vector N-GRID)]
      (let [reps (replicates n)
            runs (mapv (fn [rep]
                         (let [seed (+ 20000 (* 1000 ni) rep)
                               {:keys [posterior]} (ch06/mh-goal-posterior
                                                     actions {:samples n :burn (max 1 (quot n 10))
                                                              :key (rng/fresh-key seed)})]
                           (mx/clear-cache!)
                           (ch06/tv exact posterior)))
                       (range reps))]
        {:n n :replicates reps
         :tv_mean (mean runs) :tv_std (std runs) :tv_max (apply max runs)}))))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(println (apply str (repeat 72 "=")))
(println "  Agents axis: pluggable inference on the identical goal-inference GF")
(println (apply str (repeat 72 "=")))
(println (str "\n  exact posterior (ground truth): " (pr-str (ch06/normalize-map exact))))
(println (str "  N grid: " N-GRID "   alpha = " ch06/ALPHA " (finite)"))

(def t0 (js/performance.now))

;; Reference single-run posteriors (a large budget) for the agreement table.
(def REF-N (if quick? 1000 2000))
(def is-ref (ch06/is-goal-posterior actions REF-N (rng/fresh-key 42)))
(def mh-ref (ch06/mh-goal-posterior actions {:samples REF-N :burn (quot REF-N 10) :key (rng/fresh-key 43)}))

(println (str "\n  -- reference posteriors @ N=" REF-N " --"))
(println (str "     EXACT      : " (pr-str (ch06/normalize-map exact))))
(println (str "     IMPORTANCE : " (pr-str (:posterior is-ref)) "  (ESS " (.toFixed (:ess is-ref) 1) ", TV " (.toExponential (ch06/tv exact (:posterior is-ref)) 2) ")"))
(println (str "     MCMC (MH)  : " (pr-str (:posterior mh-ref)) "  (TV " (.toExponential (ch06/tv exact (:posterior mh-ref)) 2) ")"))

(println "\n  -- convergence sweep (TV vs N) --")
(def is-c (is-curve))
(def mh-c (mh-curve))
(doseq [[is-row mh-row] (map vector is-c mh-c)]
  (println (str "     N=" (.padStart (str (:n is-row)) 5)
                "   IS TV=" (.toExponential (:tv_mean is-row) 3)
                " (ESS " (.toFixed (* 100 (:ess_frac is-row)) 0) "%)"
                "   MH TV=" (.toExponential (:tv_mean mh-row) 3))))

(def is-fit (log-log-slope N-GRID (map :tv_mean is-c)))
(def mh-fit (log-log-slope N-GRID (map :tv_mean mh-c)))
(println (str "\n  IS log-log slope: " (.toFixed (:slope is-fit) 3)
              "   MH log-log slope: " (.toFixed (:slope mh-fit) 3)
              "   (MC rate expectation: -0.5)"))

;; Fourth backend: gradient/amortized recovery (the optimization figure).
(println "\n  -- gradient/amortized recovery (continuous utility latent) --")
(def grad (ch06/gradient-recovery))
(println (str "     planted=" (:plant-utils grad)
              "  recovered=" (mapv #(js/Number (.toFixed % 3)) (:rec-utils grad))
              "  loss " (.toFixed (:plant-loss grad) 3) " -> " (.toFixed (:rec-loss grad) 3)))

(def elapsed-ms (- (js/performance.now) t0))
(println (str "\n  Total: " (.toFixed (/ elapsed-ms 1000) 1) "s"))

(write-json "data.json"
  {:experiment "agents-pluggable-inference"
   :description "Pluggable inference on one agent GF: exact / IS / MH agree on P(goal|actions); TV->0 at the MC rate; plus gradient recovery of a continuous utility latent."
   :claim "The same agent generative function admits exact enumeration (p/assess), importance sampling, Metropolis-Hastings, and gradient/amortized inference; the sampling backends converge to the exact posterior in total variation at the canonical N^(-1/2) rate. Worded as 'sampler -> exact, TV < eps' — a single run is an estimate, never the posterior. The posterior is non-degenerate so agreement is a real three-way match."
   :model {:source "examples/agentmodels/ch06_pluggable_inference.cljs"
           :world "5x5 gridworld, 3 goals (:a up-left, :b down-left, :c right)"
           :latent ":goal in {:a :b :c}, uniform prior"
           :observation {:states (vec ch06/obs-states) :actions (vec ch06/obs-actions)
                         :note "left,left,up — two ambiguous lefts then a disambiguating up"}
           :alpha ch06/ALPHA :n_iters ch06/N-ITERS}
   :finite_alpha_regime {:alpha ch06/ALPHA
                         :rationale "At alpha=Inf argmax policies make the sampling likelihoods degenerate; alpha=2 keeps all goal hypotheses at finite likelihood."}
   :exact_posterior (ch06/normalize-map exact)
   :reference_posteriors {:n REF-N
                          :exact (ch06/normalize-map exact)
                          :importance {:posterior (:posterior is-ref) :ess (:ess is-ref)
                                       :tv_to_exact (ch06/tv exact (:posterior is-ref))}
                          :mh {:posterior (:posterior mh-ref)
                               :tv_to_exact (ch06/tv exact (:posterior mh-ref))}}
   :convergence {:n_grid N-GRID
                 :replicates_per_n (mapv replicates N-GRID)
                 :importance {:curve is-c :tv_slope_fit (assoc is-fit :expected -0.5)}
                 :mh {:curve mh-c :tv_slope_fit (assoc mh-fit :expected -0.5)}}
   :gradient_recovery {:plant_utils (:plant-utils grad)
                       :recovered_utils (:rec-utils grad)
                       :plant_loss (:plant-loss grad)
                       :recovered_loss (:rec-loss grad)
                       :loss_history (:loss-history grad)}
   :config {:seed_scheme "IS seed = 10000 + 1000*n-index + rep; MH seed = 20000 + ...; rng/fresh-key(seed); MLX RNG is bit-reproducible"}
   :duration_ms elapsed-ms})
