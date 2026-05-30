(ns examples.qwen36-hyperprior
  "Hyperprior injection: take LLM-generated probabilistic code, walk the
   parsed form, and rewrite numeric Beta hyperparameters into TRACED
   random variables drawn from a hyperprior. The LLM proposes structure;
   inference picks the hyperparameters.

   Pipeline:
     LLM → parse  →  postwalk-rewrite  →  re-emit
                  ↑ this is the new step
                                       → SCI eval → wrap → IS log p(obs)
                                       → compare to numerical ground truth
                                       → compare to prescribed-prior baselines

   Run: bun run --bun nbb examples/qwen36_hyperprior.cljs"
  (:require [genmlx.llm.backend :as llm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [edamame.core :as eda]
            [sci.core :as sci]
            [clojure.walk :as walk]
            [clojure.string :as str]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/Qwen3.6-35B-A3B-4bit"))

(def n-importance-samples 1000)
(def observations [1 1 1 1 1 1 1 1 0 0])
(def n-flips (count observations))
(def k-heads (apply + observations))

;; Reference (from prescribed-prior run + previous Exp(1) hyperprior run):
(def reference-results
  [{:name "Beta(2,2) (prescribed)"      :log-ml -6.166 :note "best prescribed"}
   {:name "Beta(1,1) (prescribed)"      :log-ml -6.205}
   {:name "Beta(0.5,0.5) (prescribed)"  :log-ml -6.418}
   {:name "Hierarchical Exp(1) (GT)"    :log-ml -6.665 :note "previous run, Riemann sum"}
   {:name "fixed θ=0.5 (prescribed)"    :log-ml -6.932 :note "worst prescribed"}])

;; ---------------------------------------------------------------------------
;; SCI environment — adds dist/exponential because the rewrite introduces it
;; ---------------------------------------------------------------------------

(def sci-env
  {:namespaces
   {'dist {'gaussian    dist/gaussian
           'uniform     dist/uniform
           'bernoulli   dist/bernoulli
           'beta        dist/beta-dist
           'gamma       dist/gamma-dist
           'exponential dist/exponential}
    'mx   {'add mx/add 'multiply mx/multiply 'scalar mx/scalar 'item mx/item}}})

;; ===========================================================================
;; THE REWRITE — the only new piece. ~15 lines.
;; ===========================================================================

(defn beta-with-numeric-args?
  "True if `form` is `(dist/beta NUM NUM)`."
  [form]
  (and (or (list? form) (seq? form))
       (= 3 (count form))
       (= 'dist/beta (first form))
       (number? (nth form 1))
       (number? (nth form 2))))

(defn hyperprior-form
  "Replacement form: (dist/beta (trace :alpha-h-N (dist/gamma 2 1))
                                (trace :beta-h-N  (dist/gamma 2 1)))
   Gamma(2, 1) shape-rate has mean 2, mode 1 — better matched to typical
   Beta hyperparameter scales than Exp(1) (which puts most mass below 1).
   `counter` ensures fresh addresses if there are multiple Beta sites."
  [counter]
  (let [n (swap! counter inc)]
    (list 'dist/beta
          (list 'trace (keyword (str "alpha-h-" n)) '(dist/gamma 2 1))
          (list 'trace (keyword (str "beta-h-"  n)) '(dist/gamma 2 1)))))

(defn inject-hyperprior
  "Walk `form`, rewrite every (dist/beta NUM NUM) into the hierarchical version.
   Returns [rewritten-form n-rewrites]."
  [form]
  (let [counter (atom 0)
        rewritten (walk/postwalk
                   (fn [f] (if (beta-with-numeric-args? f)
                             (hyperprior-form counter)
                             f))
                   form)]
    [rewritten @counter]))

;; ===========================================================================
;; Numerical ground truth for the hierarchical model.
;; log p(obs) = ∫∫ p(α) p(β) p(obs | α, β) dα dβ
;; with Exp(1) priors on (α, β) and analytical Beta-Binomial inner term.
;; ===========================================================================

(def ^:private lanczos-c
  [0.99999999999980993 676.5203681218851 -1259.1392167224028
   771.32342877765313 -176.61502916214059 12.507343278686905
   -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7])

(defn lgamma
  "Lanczos approximation. ~14 digits for x > 0.5; reflection for x < 0.5."
  [x]
  (if (< x 0.5)
    (- (Math/log (/ Math/PI (Math/sin (* Math/PI x)))) (lgamma (- 1 x)))
    (let [x' (- x 1)
          g 7
          a (loop [i 1, s (first lanczos-c)]
              (if (>= i (count lanczos-c))
                s
                (recur (inc i) (+ s (/ (nth lanczos-c i) (+ x' i))))))
          t (+ x' g 0.5)]
      (+ (* 0.5 (Math/log (* 2 Math/PI)))
         (* (+ x' 0.5) (Math/log t))
         (- t)
         (Math/log a)))))

(defn log-beta [a b] (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))

(defn log-beta-binomial-ml
  "Closed-form log p(obs | Beta(α,β), n trials, k heads)."
  [a b k n]
  (- (log-beta (+ a k) (+ b (- n k))) (log-beta a b)))

(defn log-sum-exp [xs]
  (let [m (apply max xs)]
    (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))))

(defn numerical-hierarchical-log-ml
  "Riemann sum over (α, β) ∈ (0, max] × (0, max] with Gamma(2,1) prior weighting.
   log p(α) = log α − α  (shape=2, rate=1), same for β."
  [k n max-val delta]
  (let [vals (vec (range delta (+ max-val delta) delta))
        log-d2 (* 2 (Math/log delta))
        terms (for [a vals, b vals]
                (+ (Math/log a) (- a)
                   (Math/log b) (- b)
                   (log-beta-binomial-ml a b k n)
                   log-d2))]
    (log-sum-exp terms)))

;; ---------------------------------------------------------------------------
;; Standard helpers (same as previous probes)
;; ---------------------------------------------------------------------------

(defn extract-code [text]
  (let [t (str/trim (or text ""))]
    (cond
      (str/blank? t) ""
      (re-find #"```" t)
      (let [m (re-find #"```(?:clojure|cljs|clj|clojurescript)?\s*\n?([\s\S]*?)```" t)]
        (if m (str/trim (nth m 1)) t))
      (str/starts-with? t "(") t
      :else (let [i (str/index-of t "(")] (if i (subs t i) "")))))

(defn parse-cljs [code]
  (try {:ok? true :form (eda/parse-string code {:all true})}
       (catch :default e {:ok? false :err (.-message e)})))

(defn sci-eval-fn [code]
  (try
    (let [v (sci/eval-string code sci-env)]
      (cond
        (fn? v) {:fn v}
        (and (var? v) (fn? (deref v))) {:fn (deref v)}
        :else {:err (str "value is not a function: " (pr-str v))}))
    (catch :default e {:err (.-message e)})))

(defn wrap-as-gf [model-fn]
  (dyn/auto-key (gen [n] (model-fn trace n))))

(defn observations->constraints [obs]
  (reduce (fn [cm [i v]]
            (cm/set-choice cm
                           [(keyword (str "flip" i))]
                           (mx/scalar v mx/int32)))
          cm/EMPTY
          (map-indexed vector obs)))

(defn estimate-log-ml [gf args constraints n]
  (let [weights (mapv (fn [_]
                        (mx/item (:weight (p/generate gf args constraints))))
                      (range n))
        max-w  (apply max weights)
        log-ml (- (log-sum-exp weights) (Math/log n))
        ess    (/ (Math/pow (reduce + (map #(Math/exp (- % max-w)) weights)) 2)
                  (reduce + (map #(Math/exp (* 2 (- % max-w))) weights)))]
    {:log-ml log-ml :ess ess}))

(defn now-ms [] (.now js/performance))
(defn fmt-ms [t0] (str (.toFixed (- (now-ms) t0) 0) "ms"))
(defn pad [s n] (str s (apply str (repeat (max 0 (- n (count s))) " "))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(println "============================================================")
(println " Qwen3.6-35B-A3B-4bit  —  Hyperprior injection")
(println "============================================================")
(println " observations:" observations "  k=" k-heads "heads, n=" n-flips)
(println " IS samples  :" n-importance-samples)
(println " hyperprior  : α, β ~ Gamma(2, 1)  (mean 2, mode 1 — tighter than Exp(1))")
(println "============================================================")

(pr/let
  [_   (println "\n[1] Loading model")
   t0  (now-ms)
   m   (llm/load-model model-path)
   _   (println "    type:" (:type m) "  load:" (fmt-ms t0))

   ;; -------- Step 2: LLM-generate the baseline Beta-Bernoulli model --------
   _   (println "\n[2] Generating a baseline Beta-Bernoulli model from the LLM…")
   t1  (now-ms)
   text (llm/generate-text-raw
         m (str
            "Probabilistic programs are written as `(fn [trace ...args] body)` "
            "where `(trace <addr> <dist>)` samples and records under <addr>.\n"
            "Distributions: dist/gaussian, dist/uniform, dist/bernoulli, "
            "dist/beta, dist/exponential.\n\n"
            "Example — N IID Gaussians sharing a mean:\n"
            "(fn [trace n]\n"
            "  (let [mu (trace :mu (dist/gaussian 0 10))]\n"
            "    (mapv (fn [i]\n"
            "            (trace (keyword (str \"obs\" i)) (dist/gaussian mu 1)))\n"
            "          (range n))))\n\n"
            "Now write a probabilistic program of two arguments [trace n] for "
            "n IID coin flips. Use Beta(1, 1) as the prior on theta at address "
            ":theta. Sample n Bernoulli draws at addresses :flip0, …, :flip<n-1>. "
            "Output ONLY the (fn [trace n] ...) form.")
         {:max-tokens 250
          :temperature 0
          :system-prompt
          "You are a probabilistic programming assistant. Output only valid ClojureScript code."})
   _   (println "    gen time:" (fmt-ms t1))
   code (extract-code text)
   _   (println "    code (LLM-as-emitted):")
   _   (doseq [line (str/split code #"\n")] (println "    │" line))

   parse (parse-cljs code)
   _   (println "    parse:" (if (:ok? parse) "✓" (str "✗ " (:err parse))))

   ;; -------- Step 3: APPLY THE REWRITE — the only new step --------
   _   (println "\n[3] Hyperprior injection (postwalk over the parsed form)")
   rewrite-result (when (:ok? parse) (inject-hyperprior (:form parse)))
   rewritten      (first  rewrite-result)
   n-rewrites     (second rewrite-result)
   _   (println "    rewrites :" n-rewrites
                "  (each (dist/beta NUM NUM) → traced α, β from Gamma(2,1) hyperprior)")
   new-code (when rewritten (pr-str rewritten))
   _   (when new-code
         (println "    rewritten cljs:")
         (doseq [line (str/split new-code #"\n")] (println "    │" line)))

   ;; -------- Step 4: SCI eval the rewritten code --------
   _   (println "\n[4] SCI eval the rewritten code")
   evald (when new-code (sci-eval-fn new-code))
   _   (cond
         (nil? evald) (println "    skipped")
         (:err evald) (println "    err:" (:err evald))
         :else (println "    ✓ got hierarchical model fn"))

   ;; -------- Step 5: Wrap + sanity simulate --------
   _   (println "\n[5] Wrap + sanity simulate")
   gf  (when (:fn evald) (wrap-as-gf (:fn evald)))
   sim (when gf (try (p/simulate gf [n-flips])
                     (catch :default e {:err (.-message e)})))
   _   (when (and sim (not (:err sim)))
         (let [paths (cm/addresses (:choices sim))
               score (mx/item (:score sim))]
           (println "    trace sites:" (count paths))
           (println "    paths      :" paths)
           (println "    log-prior  :" (.toFixed score 4)
                    "  (richer trace → typically a bit lower than non-hier)")))
   _   (when (:err sim) (println "    err:" (:err sim)))

   ;; -------- Step 6: IS log marginal likelihood --------
   _   (println "\n[6] Importance sampling log p(obs) on the hierarchical model")
   t2  (now-ms)
   constraints (observations->constraints observations)
   is-result (when gf (try (estimate-log-ml gf [n-flips] constraints n-importance-samples)
                           (catch :default e {:err (.-message e)})))
   _   (when (and is-result (not (:err is-result)))
         (println "    is time   :" (fmt-ms t2))
         (println "    log p(obs):" (.toFixed (:log-ml is-result) 4))
         (println "    ESS / N   :" (.toFixed (:ess is-result) 1) "/" n-importance-samples
                  (str "(" (.toFixed (* 100.0 (/ (:ess is-result) n-importance-samples)) 1)
                       "% — likely lower than fixed-prior runs because the joint"
                       " sampler now also has to wander in (α,β))")))

   ;; -------- Step 7: Numerical ground truth --------
   _   (println "\n[7] Numerical ground truth (2-D Riemann over (α, β))")
   gt  (numerical-hierarchical-log-ml k-heads n-flips 8.0 0.05)
   _   (println "    grid       : (0.05 .. 8.0) × (0.05 .. 8.0)  step 0.05")
   _   (println "    log p(obs) ≈" (.toFixed gt 4)
                "  (Gamma(2,1)²·analytical Beta-Binomial)")
   _   (when (and is-result (not (:err is-result)))
         (println "    IS error   :" (.toFixed (Math/abs (- (:log-ml is-result) gt)) 4)))

   ;; -------- Step 8: Comparison panel --------
   _   (println "\n[8] All models, ranked by log p(obs)")
   all-rows (concat reference-results
                    (when (and is-result (not (:err is-result)))
                      [{:name "Hierarchical Gamma(2,1) (IS)"
                        :log-ml (:log-ml is-result)
                        :note "this run"}])
                    [{:name "Hierarchical Gamma(2,1) (GT)"
                      :log-ml gt
                      :note "Riemann sum, this run"}])
   ranked   (vec (sort-by (fn [r] (- (:log-ml r))) all-rows))
   _   (println (str "    " (pad "rank" 6) (pad "model" 32)
                     (pad "log p(obs)" 14) (pad "note" 30)))
   _   (println "    ─────────────────────────────────────────────────────────────────────────")
   _   (doseq [[i r] (map-indexed vector ranked)]
         (println (str "    "
                       (pad (str (inc i)) 6)
                       (pad (:name r) 32)
                       (pad (.toFixed (:log-ml r) 4) 14)
                       (pad (or (:note r) "") 30))))

   _   (println "\n============================================================\n")]
  nil)
