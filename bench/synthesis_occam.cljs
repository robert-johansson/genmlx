(ns bench.synthesis-occam
  "Measured synthesis Occam experiment (genmlx-heaw).

   Deterministic scorer over the FROZEN LLM-authored programs in
   bench/fixtures/synthesis_occam_programs.edn (produced one-shot by
   bench/synthesis_occam_generate.cljs against qwen3-0.6b base, knowledge mode).

   Three measured results -> results/synthesis/data.json:

     1. FIRE-RATE  — over the synthesized models, the fraction where analytical
        elimination fires (score-model* routes to :exact / :kalman). Knowledge
        mode reaches only the normal-normal family, so the firing family is
        normal-normal; the denominator includes non-conjugate models (bernoulli/
        exponential likelihoods) so the rate is a genuine measurement.

     2. |IS - analytic|  — on the firing subset, the gap between the EXACT
        analytical marginal evidence (score-model) and a VECTORIZED importance-
        sampling estimate of the same quantity (one batched model run over N
        particles, via vectorized-importance-sampling). Explicit seeded MLX keys
        make it reproducible; the batched path keeps the Metal buffer count O(1).
        Shows IS converges to the value GenMLX computes in closed form.

     3. OCCAM CURVE  — a deterministic complexity ladder over one dataset:
        Bayesian log model-evidence (normal-normal marginal, exact) vs a
        non-Bayesian max-likelihood baseline. Bayesian evidence peaks at the
        true (simplest adequate) complexity and declines; max-likelihood keeps
        rising -> overfits. This is the measured Occam's-razor contrast.

   Run: bun run --bun nbb bench/synthesis_occam.cljs
   (or via the harness:  run_experiments.cljs --only synthesis)"
  (:require [genmlx.llm.msa :as msa]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [cljs.reader :as reader])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/synthesis")))

(def fixture-path
  (.resolve path-mod (js/process.cwd) "bench/fixtures/synthesis_occam_programs.edn"))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  Wrote: " filepath))))

(defn mean [xs] (if (seq xs) (/ (reduce + xs) (count xs)) 0))

(defn median [xs]
  (if (empty? xs)
    0
    (let [s (vec (sort xs)) n (count s) h (quot n 2)]
      (if (odd? n) (nth s h) (/ (+ (nth s (dec h)) (nth s h)) 2)))))

(def is-particles 4000)

;; Reproducibility: every stochastic step below uses an explicit seeded MLX key
;; (MLX RNG + compute are bit-reproducible), so the IS estimate and the
;; prior-predictive datum are identical run-to-run. The IS estimate uses the
;; VECTORIZED path (one batched model run over N particles) rather than a
;; sequential N-generate loop: shape-based batching keeps the Metal buffer COUNT
;; O(1) instead of O(N), avoiding the 499000-buffer limit (bug genmlx-5ucd). The
;; batched handler bypasses the analytical dispatcher by design, so it is genuine
;; importance sampling from the prior — no analytical shortcut to strip.

;; ===========================================================================
;; Load the frozen LLM-authored programs
;; ===========================================================================

(when-not (.existsSync fs fixture-path)
  (println "FATAL: fixture not found at" fixture-path)
  (println "Generate it first:  bun run --bun nbb bench/synthesis_occam_generate.cljs")
  (js/process.exit 1))

(def records (reader/read-string (.readFileSync fs fixture-path "utf8")))

(println "============================================================")
(println " synthesis-Occam: deterministic scoring")
(println "============================================================")
(println "Fixture:" fixture-path)
(println "Frozen programs:" (count records) "\n")

;; ===========================================================================
;; Deliverable 1 — fire-rate over the synthesized models
;; ===========================================================================

(println "-- [1] fire-rate --")

(def scored-models
  (vec
   (for [rec records]
     (let [gf     (msa/eval-model (:code rec))
           method (when gf (:method (msa/score-model* gf (:observations rec))))
           fired? (boolean (#{:exact :kalman} method))]
       (assoc rec
              :gf gf
              :eval-ok? (boolean gf)
              :method (when method (name method))
              :fired? fired?
              :family (when fired? "normal-normal"))))))

(def n-candidates (count scored-models))
(def n-eval-ok   (count (filter :eval-ok? scored-models)))
(def n-parsed    (count (filter :parsed? scored-models)))
(def firing      (filterv :fired? scored-models))
(def n-fired     (count firing))
(def fire-rate   (if (pos? n-eval-ok) (/ n-fired n-eval-ok) 0))

(defn group-rate [key-fn]
  (->> scored-models
       (filter :eval-ok?)
       (group-by key-fn)
       (map (fn [[k models]]
              [k {:n (count models)
                  :fired (count (filter :fired? models))
                  :rate (/ (count (filter :fired? models)) (count models))}]))
       (into {})))

(def by-task (group-rate :task-id))

(println (str "  candidates: " n-candidates "   eval-ok: " n-eval-ok
              "   parsed: " n-parsed))
(println (str "  fired (exact/kalman): " n-fired
              "   fire-rate: " (.toFixed (* 100.0 fire-rate) 1) "% of eval-ok"))
(doseq [[task {:keys [n fired rate]}] (sort-by key by-task)]
  (println (str "    " (name task) ": " fired "/" n
                " (" (.toFixed (* 100.0 rate) 0) "%)")))

;; ===========================================================================
;; Deliverable 2 — |IS - analytic| on the firing subset
;; ===========================================================================

(println "\n-- [2] |IS - analytic| on firing subset (" is-particles "particles) --")

;; No observed dataset -> compare at a MODEL-CONSISTENT datum: the prior-
;; predictive mean of the observation site (mean over K batched prior draws).
;; This is the fair point to compare an IS estimator to the exact marginal
;; evidence; an arbitrary out-of-scale datum inflates IS variance for reasons
;; unrelated to the estimator. Vectorized + explicit keys -> deterministic.
(def prior-pred-K 256)

(defn central-obs [gf obs-addr seed]
  (let [vt (dyn/vsimulate gf [] prior-pred-K (rng/fresh-key seed))]
    (mx/item (mx/mean (cm/get-choice (:choices vt) [obs-addr])))))

(defn obs-choicemap [obs-addr v]
  (cm/set-value cm/EMPTY obs-addr (mx/scalar v)))

(def is-gaps
  (vec
   (for [[i rec] (map-indexed vector firing)]
     (let [gf      (:gf rec)
           addr    (:obs-addr rec)
           obs-val (central-obs gf addr (+ 31000 i))
           exact   (msa/score-model gf {addr obs-val})
           is      (mx/item (:log-ml-estimate
                             (is/vectorized-importance-sampling
                              {:samples is-particles :key (rng/fresh-key (+ 9000 i))}
                              gf [] (obs-choicemap addr obs-val))))
           gap     (js/Math.abs (- exact is))]
       (mx/clear-cache!)
       (println (str "    " (name (:task-id rec))
                     "  y*=" (.toFixed obs-val 3)
                     "  exact=" (.toFixed exact 4)
                     "  IS=" (.toFixed is 4)
                     "  |gap|=" (.toFixed gap 4)))
       {:task-id (name (:task-id rec))
        :obs obs-val :exact exact :is is :gap gap}))))

(def gap-values (mapv :gap is-gaps))

;; Histogram bins for |gap|
(def gap-bin-edges [0.0 0.02 0.05 0.1 0.2 0.5 1.0])
(defn bin-label [lo hi] (str (.toFixed lo 2) "-" (if hi (.toFixed hi 2) "inf")))
(def gap-histogram
  (let [edges gap-bin-edges]
    (vec
     (for [i (range (count edges))]
       (let [lo (nth edges i)
             hi (when (< (inc i) (count edges)) (nth edges (inc i)))
             in-bin (filter (fn [g] (and (>= g lo) (or (nil? hi) (< g hi)))) gap-values)]
         {:bin (bin-label lo hi) :lo lo :hi hi :count (count in-bin)})))))

(def gap-summary
  {:n (count gap-values)
   :mean (mean gap-values)
   :median (median gap-values)
   :max (if (seq gap-values) (apply max gap-values) 0)})

(println (str "  n=" (:n gap-summary)
              "  mean=" (.toFixed (:mean gap-summary) 4)
              "  median=" (.toFixed (:median gap-summary) 4)
              "  max=" (.toFixed (:max gap-summary) 4)))
(doseq [{:keys [bin count]} gap-histogram]
  (println (str "    [" bin "] " (apply str (repeat count "#")) " " count)))

;; ===========================================================================
;; Deliverable 3 — Occam curve (Bayesian log-ML vs max-likelihood)
;; ===========================================================================

(println "\n-- [3] Occam curve: Bayesian log-ML vs max-likelihood --")

;; Fixed dataset: 8 observations clustered around a single true mean (~2.0).
(def occ-data [2.31 1.62 2.84 1.19 2.47 1.98 2.05 1.55])
(def occ-tau 5.0)    ;; prior sd on each latent group-mean:  mu_g ~ N(0, tau)
(def occ-sigma 1.0)  ;; observation noise sd:                y_i  ~ N(mu_g, sigma)

;; Normal-normal group models with STATIC literal addresses (so conjugacy
;; fires, exactly like bench/conjugacy mean-model). One model per group size.
(def nn-group-1
  (dyn/auto-key (gen [] (let [mu (trace :mu (dist/gaussian 0 5))]
                          (trace :y0 (dist/gaussian mu 1)) mu))))
(def nn-group-2
  (dyn/auto-key (gen [] (let [mu (trace :mu (dist/gaussian 0 5))]
                          (trace :y0 (dist/gaussian mu 1))
                          (trace :y1 (dist/gaussian mu 1)) mu))))
(def nn-group-4
  (dyn/auto-key (gen [] (let [mu (trace :mu (dist/gaussian 0 5))]
                          (trace :y0 (dist/gaussian mu 1))
                          (trace :y1 (dist/gaussian mu 1))
                          (trace :y2 (dist/gaussian mu 1))
                          (trace :y3 (dist/gaussian mu 1)) mu))))
(def nn-group-8
  (dyn/auto-key (gen [] (let [mu (trace :mu (dist/gaussian 0 5))]
                          (trace :y0 (dist/gaussian mu 1))
                          (trace :y1 (dist/gaussian mu 1))
                          (trace :y2 (dist/gaussian mu 1))
                          (trace :y3 (dist/gaussian mu 1))
                          (trace :y4 (dist/gaussian mu 1))
                          (trace :y5 (dist/gaussian mu 1))
                          (trace :y6 (dist/gaussian mu 1))
                          (trace :y7 (dist/gaussian mu 1)) mu))))

(defn group-model-for [k]
  (case k 1 nn-group-1, 2 nn-group-2, 4 nn-group-4, 8 nn-group-8))

(defn group-obs [ys]
  (into {} (map-indexed (fn [j y] [(keyword (str "y" j)) y]) ys)))

;; Closed-form normal-normal marginal log-evidence for a group (prior mu~N(0,tau),
;; y_i ~ N(mu,sigma)). Matrix-determinant-lemma form of log N(y; 0, sigma^2 I + tau^2 11^T).
(defn nn-marginal-closed [ys tau sigma]
  (let [k  (count ys)
        s2 (* sigma sigma)
        t2 (* tau tau)
        S  (reduce + (map #(* % %) ys))
        T  (reduce + ys)
        logdet (+ (* (dec k) (js/Math.log s2)) (js/Math.log (+ s2 (* k t2))))
        quad   (/ (- S (* (/ t2 (+ s2 (* k t2))) (* T T))) s2)]
    (- (* -0.5 k (js/Math.log (* 2 js/Math.PI)))
       (* 0.5 logdet)
       (* 0.5 quad))))

;; Max-likelihood: plug in mu_hat = group mean (the MLE), no integration.
(defn group-mle [ys sigma]
  (let [k  (count ys)
        mu (/ (reduce + ys) k)
        s2 (* sigma sigma)]
    (reduce + (map (fn [y] (- (* -0.5 (js/Math.log (* 2 js/Math.PI s2)))
                              (/ (* (- y mu) (- y mu)) (* 2 s2))))
                   ys))))

(defn partition-groups [data m]
  (partition (/ (count data) m) data))

(def occam-ladder
  (vec
   (for [m [1 2 4 8]]
     (let [groups (map vec (partition-groups occ-data m))
           gsize  (/ (count occ-data) m)
           bayes-closed (reduce + (map #(nn-marginal-closed % occ-tau occ-sigma) groups))
           bayes-genmlx (reduce + (map #(msa/score-model (group-model-for (count %))
                                                         (group-obs %)) groups))
           mle    (reduce + (map #(group-mle % occ-sigma) groups))]
       {:m m :group-size gsize
        :bayes-logML bayes-closed
        :bayes-genmlx bayes-genmlx
        :mle mle}))))

(defn argmax-by [k rows] (:m (apply max-key k rows)))
(def selected-by-bayes (argmax-by :bayes-logML occam-ladder))
(def selected-by-mle   (argmax-by :mle occam-ladder))
(def genmlx-vs-closed-max
  (apply max (map #(js/Math.abs (- (:bayes-logML %) (:bayes-genmlx %))) occam-ladder)))

(println (str "  m | group-size | bayes-logML | bayes(genmlx) |   MLE"))
(doseq [{:keys [m group-size bayes-logML bayes-genmlx mle]} occam-ladder]
  (println (str "  " m " |     " group-size "      | "
                (.toFixed bayes-logML 3) "    | "
                (.toFixed bayes-genmlx 3) "      | "
                (.toFixed mle 3))))
(println (str "  Bayesian evidence selects m=" selected-by-bayes
              " (simplest adequate); max-likelihood selects m=" selected-by-mle
              " (most complex)."))
(println (str "  GenMLX-exact vs closed-form max |diff| = "
              (.toFixed genmlx-vs-closed-max 6)))

;; ===========================================================================
;; Emit results/synthesis/data.json
;; ===========================================================================

(write-json
 "data.json"
 {:experiment "synthesis"
  :description (str "Measured synthesis Occam: fire-rate of analytical elimination over "
                    "LLM-synthesized models, |IS - analytic| on the firing subset, and a "
                    "Bayesian-evidence vs max-likelihood Occam curve.")
  :timestamp (.toISOString (js/Date.))
  :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}
  :config {:synthesizer "qwen3-0.6b base, knowledge mode (frozen programs)"
           :conjugate_family "normal-normal"
           :is_particles is-particles}
  :fire_rate {:n_candidates n-candidates
              :n_eval_ok n-eval-ok
              :n_parsed n-parsed
              :n_fired n-fired
              :fire_rate fire-rate
              :by_family {:normal-normal n-fired :none (- n-eval-ok n-fired)}
              :by_task (into {} (map (fn [[k v]] [k v]) by-task))}
  :is_vs_analytic {:n_firing (:n gap-summary)
                   :particles is-particles
                   :summary gap-summary
                   :histogram gap-histogram
                   :per_model is-gaps}
  :occam {:data occ-data
          :tau occ-tau
          :sigma occ-sigma
          :true_complexity 1
          :ladder occam-ladder
          :selected_by_bayes selected-by-bayes
          :selected_by_mle selected-by-mle
          :genmlx_vs_closed_max_abs_diff genmlx-vs-closed-max}
  :per_model (mapv (fn [r] {:task-id (name (:task-id r))
                            :eval-ok (:eval-ok? r)
                            :parsed (:parsed? r)
                            :obs-depends (:obs-depends? r)
                            :method (:method r)
                            :fired (:fired? r)})
                   scored-models)})

(mx/force-gc!)

(println "\n============================================================")
(println (str " fire-rate=" (.toFixed (* 100.0 fire-rate) 1) "%  ("
              n-fired "/" n-eval-ok ")    "
              "|IS-analytic| median=" (.toFixed (:median gap-summary) 4) "    "
              "Occam: bayes->m" selected-by-bayes " vs mle->m" selected-by-mle))
(println "============================================================")
