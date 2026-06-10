(ns bench.agents-is-exact
  "IS -> exact on the IDENTICAL agent generative function (TV vs N).

   The agents-axis claim: an agent is a generative function, and the inference
   method used to invert it is pluggable and orthogonal. This artifact runs the
   SAME biased-agent GF (examples/agentmodels/biased_inverse.cljs) through two
   inference methods:

     EXACT — enumerate the finite bias prior, score the full trajectory with
             p/assess, normalize. Deterministic closed form (no sampling).
     IS    — sample :bias from the prior, constrain the observed actions,
             group normalized particle weights by sampled bias.

   and measures TV(exact, IS) as a function of particle count N, with replicate
   seeds per N. The claim is convergence: IS -> exact, TV < eps, at the
   canonical N^(-1/2) Monte Carlo rate — NEVER 'exact = sampled' (a single IS
   run is an estimate, not the posterior).

   Finite-alpha regime: everything runs at alpha = 2.0. At alpha = ##Inf the
   softmax policies become argmax 0/1 indicators, the IS likelihood weights
   degenerate to {0, constant}, and TV-vs-N is no longer informative (IS either
   nails the posterior or produces zero-weight particles). alpha = 2 keeps both
   bias hypotheses at finite likelihood so the weights genuinely vary.

   Exactness is itself cross-checked against an independent oracle:
   bias-posterior-via-policy (the inverse.cljs goal-inference idiom — one
   forward agent per bias, action log-liks summed host-side) and the
   math-verifier reference constants from agentmodels_biased_inverse_test.cljs
   Section 8.

   Output: results/agents-is-exact/data.json
   Usage:  bun run --bun nbb bench/agents_is_exact.cljs"
  (:require [agentmodels.biased-inverse :as bi]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/agents-is-exact")))

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

;; ---------------------------------------------------------------------------
;; Experiment configuration
;; ---------------------------------------------------------------------------

(def ALPHA 2.0)
(def N-GRID [10 30 100 300 1000 3000])

(defn replicates
  "Replicate seeds per N: 20 up to N=1000, 10 at N=3000 (the large-N tail is
   cheap to estimate — TV is already small — and dominates wall-clock)."
  [n]
  (if (> n 1000) 10 20))

(def mdp (bi/temptation-mdp))

(defn cfg [states actions]
  {:mdp mdp :alpha ALPHA :discount 1.0 :n-iters 10
   :states states :actions actions})

(def trajectories
  ;; action 0 = head toward the temptation gate ; action 1 = take the safe route
  [{:name "safe-x3"  :states [0 0 0] :actions [1 1 1] :seed-base 10000
    ;; math-verifier reference (test Section 8): exact P(:soph | safe x3) @ alpha=2
    :reference {:key :sophisticated :value 0.5515 :tol 1e-3}}
   {:name "tempt-x3" :states [0 0 0] :actions [0 0 0] :seed-base 20000
    :reference {:key :naive :value 0.5638 :tol 1e-3}}])

;; ---------------------------------------------------------------------------
;; Total variation distance over the finite bias support
;; ---------------------------------------------------------------------------

(defn tv-distance
  "TV(p, q) = (1/2) sum_b |p(b) - q(b)| over the bias support."
  [p q]
  (* 0.5 (reduce + (map #(js/Math.abs (- (get p % 0.0) (get q % 0.0)))
                        bi/bias-values))))

;; ---------------------------------------------------------------------------
;; Least-squares slope of log10(TV) vs log10(N) — expect ~ -1/2 (MC rate)
;; ---------------------------------------------------------------------------

(defn log-log-slope [ns tvs]
  (let [xs (map #(js/Math.log10 %) ns)
        ys (map #(js/Math.log10 (max % 1e-12)) tvs)
        n  (count xs)
        mx- (mean xs)
        my- (mean ys)
        sxy (reduce + (map (fn [x y] (* (- x mx-) (- y my-))) xs ys))
        sxx (reduce + (map (fn [x] (let [d (- x mx-)] (* d d))) xs))]
    {:slope (/ sxy sxx) :intercept (- my- (* (/ sxy sxx) mx-))}))

;; ---------------------------------------------------------------------------
;; Per-trajectory runner
;; ---------------------------------------------------------------------------

(defn run-trajectory [{:keys [name states actions seed-base reference]}]
  (println (str "\n" (apply str (repeat 60 "-"))))
  (println (str "  Trajectory " name ": states=" states " actions=" actions))
  (println (apply str (repeat 60 "-")))
  (let [c        (cfg states actions)
        exact    (bi/bias-posterior c)
        ;; Independent oracle 1: the inverse.cljs policy idiom (no joint GF).
        oracle   (bi/bias-posterior-via-policy c)
        x-diff   (apply max (map #(js/Math.abs (- (get exact %) (get oracle %)))
                                 bi/bias-values))
        ;; Independent oracle 2: math-verifier reference constant.
        ref-val  (get exact (:key reference))
        ref-diff (js/Math.abs (- ref-val (:value reference)))
        _ (println (str "  exact (assess-enum):    " exact))
        _ (println (str "  policy-idiom oracle:    " oracle
                        "  (max |diff| = " (.toExponential x-diff 2) ")"))
        _ (println (str "  math-verifier ref " (:key reference) " = " (:value reference)
                        "  (|diff| = " (.toExponential ref-diff 2)
                        (when (> ref-diff (:tol reference)) "  ** OUT OF TOL **") ")"))
        curve
        (vec
          (for [[ni n] (map-indexed vector N-GRID)]
            (let [reps (replicates n)
                  runs (mapv (fn [rep]
                               (let [seed (+ seed-base (* 1000 ni) rep)
                                     {:keys [posterior ess]}
                                     (bi/is-bias-posterior c n (rng/fresh-key seed))]
                                 (mx/clear-cache!)
                                 {:tv (tv-distance exact posterior) :ess ess}))
                             (range reps))
                  tvs  (mapv :tv runs)
                  esss (mapv :ess runs)
                  row  {:n n :replicates reps
                        :tv_mean (mean tvs) :tv_std (std tvs)
                        :tv_max (apply max tvs) :tvs tvs
                        :ess_mean (mean esss)
                        :ess_frac (/ (mean esss) n)}]
              (println (str "  N=" (.padStart (str n) 5)
                            "  TV mean=" (.toExponential (:tv_mean row) 3)
                            "  std=" (.toExponential (:tv_std row) 2)
                            "  max=" (.toExponential (:tv_max row) 2)
                            "  ESS=" (.toFixed (:ess_mean row) 1)
                            " (" (.toFixed (* 100 (:ess_frac row)) 1) "% of N)"
                            "  [" reps " reps]"))
              row)))
        fit (log-log-slope N-GRID (map :tv_mean curve))]
    (println (str "  log-log slope of mean TV vs N: " (.toFixed (:slope fit) 3)
                  "  (MC rate expectation: -0.5)"))
    {:name name
     :states (vec states) :actions (vec actions)
     :exact exact
     :exact_cross_checks
     {:policy_idiom {:posterior oracle :max_abs_diff x-diff}
      :math_verifier_reference {:quantity (clojure.core/name (:key reference))
                                :value (:value reference)
                                :abs_diff ref-diff
                                :tol (:tol reference)
                                :within_tol (<= ref-diff (:tol reference))}}
     :curve curve
     :tv_slope_fit (assoc fit :expected -0.5)
     :seed_base seed-base}))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(println (apply str (repeat 70 "=")))
(println "  Agents axis: IS -> exact on the identical biased-agent GF")
(println (apply str (repeat 70 "=")))
(println (str "\n  alpha = " ALPHA " (finite — at alpha=Inf argmax policies make IS degenerate)"))
(println (str "  N grid: " N-GRID ", replicates per N: 20 (10 at N=3000)"))
(println "  TV(exact, IS) per replicate; deterministic seeds (MLX RNG is bit-reproducible)")

(def t0 (js/performance.now))
(def results (mapv run-trajectory trajectories))
(def elapsed-ms (- (js/performance.now) t0))

(println (str "\n" (apply str (repeat 70 "="))))
(println "  SUMMARY — IS -> exact (TV < eps), never 'exact = sampled'")
(println (apply str (repeat 70 "=")))
(println "\n| Trajectory | TV @ N=10 | TV @ N=3000 | slope | ESS% @ N=3000 |")
(println "|------------|-----------|-------------|-------|---------------|")
(doseq [r results]
  (let [first-row (first (:curve r))
        last-row  (last (:curve r))]
    (println (str "| " (.padEnd (:name r) 10)
                  " | " (.padStart (.toExponential (:tv_mean first-row) 2) 9)
                  " | " (.padStart (.toExponential (:tv_mean last-row) 2) 11)
                  " | " (.padStart (.toFixed (get-in r [:tv_slope_fit :slope]) 2) 5)
                  " | " (.padStart (.toFixed (* 100 (:ess_frac last-row)) 1) 13)
                  " |"))))
(println (str "\n  Total: " (.toFixed (/ elapsed-ms 1000) 1) "s"))

(write-json "data.json"
  {:experiment "agents-is-exact"
   :description "IS -> exact on the identical biased-agent GF: TV(exact, IS) vs N at finite alpha"
   :claim "The same agent generative function admits exact enumeration (p/assess) and importance sampling; IS converges to the exact posterior in total variation at the canonical N^(-1/2) rate. Worded as 'IS -> exact, TV < eps' — a single IS run is an estimate, never the posterior itself."
   :finite_alpha_regime
   {:alpha ALPHA
    :rationale "At alpha=Inf the softmax policies are argmax 0/1 indicators: IS particle weights degenerate to {0, constant} and TV-vs-N is uninformative. alpha=2 keeps both bias hypotheses at finite likelihood."}
   :model {:source "examples/agentmodels/biased_inverse.cljs"
           :mdp "temptation (5 states, 2 actions, donut Rd=5 / veg Rv=8, k=1)"
           :latent ":bias in {:naive, :sophisticated}, uniform prior"
           :discount 1.0 :n_iters 10}
   :timestamp (.toISOString (js/Date.))
   :config {:n_grid N-GRID
            :replicates_per_n (mapv replicates N-GRID)
            :seed_scheme "seed = seed-base + 1000*n-index + replicate; rng/fresh-key(seed); MLX RNG is bit-reproducible"}
   :trajectories results
   :duration_ms elapsed-ms})

(println "\nAgents IS->exact experiment complete.")
