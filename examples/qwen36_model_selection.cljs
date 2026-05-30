(ns examples.qwen36-model-selection
  "Bayesian model selection over LLM-proposed probabilistic programs.

   Earlier finding (free generation, temp=0.7): all 4 candidates collapse to
   Beta(1,1). The LLM has a sharp prior on this question. To get genuine
   model-selection signal, we now prescribe K *specific* priors, one per
   candidate, and let the data rank them.

   Pipeline per candidate:
     prompt with prescribed prior  →  cljs source (greedy)
       → parse → SCI eval → wrap → simulate (sanity)
       → IS log p(obs)
     compare to analytical Beta-Binomial ground truth.

   Run: bun run --bun nbb examples/qwen36_model_selection.cljs"
  (:require [genmlx.llm.backend :as llm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [edamame.core :as eda]
            [sci.core :as sci]
            [clojure.string :as str]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/Qwen3.6-35B-A3B-4bit"))

;; ---------------------------------------------------------------------------
;; Configuration
;; ---------------------------------------------------------------------------

(def n-importance-samples 500)

(def observations [1 1 1 1 1 1 1 1 0 0])  ; 8 heads, 2 tails
(def n-flips (count observations))
(def k-heads (apply + observations))

;; ---------------------------------------------------------------------------
;; Candidate specs — each prescribes a different prior on theta.
;; Analytical log-ML computed offline using log B(α+k, β+n−k) − log B(α, β)
;; for Beta priors, and n·log(0.5) for the fixed-bias case.
;; ---------------------------------------------------------------------------

(def candidate-specs
  [{:name        "Beta(1,1)"
    :prior-text  "Use Beta(1, 1) as the prior on theta at address :theta."
    :analytical  -6.2046}    ; log B(9,3) − log B(1,1) = log(1/495)
   {:name        "Beta(2,2)"
    :prior-text  "Use Beta(2, 2) as the prior on theta at address :theta."
    :analytical  -6.1664}    ; log B(10,4) − log B(2,2)
   {:name        "Beta(0.5,0.5)"
    :prior-text  (str "Use Beta(0.5, 0.5) as the prior on theta at address "
                      ":theta (this is the Jeffreys prior, U-shaped).")
    :analytical  -6.4175}    ; log B(8.5, 2.5) − log B(0.5, 0.5)
   {:name        "fixed θ=0.5"
    :prior-text  (str "DO NOT use a prior. theta is fixed at 0.5. "
                      "Do not trace a :theta site. Sample n Bernoulli(0.5) "
                      "draws directly.")
    :analytical  (* n-flips (Math/log 0.5))}])  ; -6.9315

;; ---------------------------------------------------------------------------
;; SCI environment
;; ---------------------------------------------------------------------------

(def sci-env
  {:namespaces
   {'dist {'gaussian    dist/gaussian
           'uniform     dist/uniform
           'bernoulli   dist/bernoulli
           'beta        dist/beta-dist
           'exponential dist/exponential}
    'mx   {'add      mx/add
           'multiply mx/multiply
           'scalar   mx/scalar
           'item     mx/item}}})

;; ---------------------------------------------------------------------------
;; Prompt assembly
;; ---------------------------------------------------------------------------

(def system-prompt
  "You are a probabilistic programming assistant. Output only valid ClojureScript code.")

(def few-shot
  (str
   "Probabilistic programs are written as `(fn [trace ...args] body)` where\n"
   "`(trace <addr> <dist>)` samples and records under <addr>.\n"
   "Distributions: dist/gaussian (mu sigma), dist/uniform (lo hi),\n"
   "dist/bernoulli (p), dist/beta (alpha beta), dist/exponential (rate).\n\n"
   "Example — N IID Gaussian observations sharing one mean:\n"
   "(fn [trace n]\n"
   "  (let [mu (trace :mu (dist/gaussian 0 10))]\n"
   "    (mapv (fn [i]\n"
   "            (trace (keyword (str \"obs\" i)) (dist/gaussian mu 1)))\n"
   "          (range n))))\n"))

(defn build-prompt [spec]
  (str few-shot
       "\nNow:\n"
       "Write a probabilistic program of two arguments [trace n] for n IID\n"
       "coin flips. " (:prior-text spec) "\n"
       "Sample n Bernoulli draws at addresses :flip0, :flip1, …, :flip<n-1>.\n"
       "Return the vector of flips.\n\n"
       "Output ONLY the (fn [trace n] ...) form."))

;; ---------------------------------------------------------------------------
;; Inline helpers
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

(defn log-sum-exp [xs]
  (let [m (apply max xs)]
    (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))))

(defn estimate-log-ml [gf args constraints n]
  (try
    (let [weights (mapv (fn [_]
                          (mx/item (:weight (p/generate gf args constraints))))
                        (range n))
          max-w  (apply max weights)
          log-ml (- (log-sum-exp weights) (Math/log n))
          ess    (/ (Math/pow (reduce + (map #(Math/exp (- % max-w)) weights)) 2)
                    (reduce + (map #(Math/exp (* 2 (- % max-w))) weights)))]
      {:log-ml log-ml :ess ess})
    (catch :default e {:err (.-message e)})))

(defn now-ms [] (.now js/performance))
(defn fmt-ms [t0] (str (.toFixed (- (now-ms) t0) 0) "ms"))

(defn one-line [s] (str/replace (str/trim s) #"\s+" " "))

(defn truncate [s n]
  (if (<= (count s) n) s (str (subs s 0 n) "…")))

(defn pad [s n]
  (str s (apply str (repeat (max 0 (- n (count s))) " "))))

;; ---------------------------------------------------------------------------
;; Build & score one candidate
;; ---------------------------------------------------------------------------

(defn build-candidate
  "Generate, parse, eval, wrap, simulate, score against observations."
  [m spec]
  (pr/let [text  (llm/generate-text-raw
                  m (build-prompt spec)
                  {:max-tokens 250
                   :temperature 0
                   :system-prompt system-prompt})
           code  (extract-code text)
           parse (parse-cljs code)]
    (if-not (:ok? parse)
      (assoc spec :raw text :code code :status :parse-failed :err (:err parse))
      (let [evald (sci-eval-fn code)]
        (if (:err evald)
          (assoc spec :raw text :code code :status :eval-failed :err (:err evald))
          (let [gf (wrap-as-gf (:fn evald))
                sanity (try {:trace (p/simulate gf [n-flips])}
                            (catch :default e {:err (.-message e)}))]
            (if (:err sanity)
              (assoc spec :raw text :code code :status :sim-failed :err (:err sanity))
              (let [constraints (observations->constraints observations)
                    is (estimate-log-ml gf [n-flips] constraints n-importance-samples)]
                (if (:err is)
                  (assoc spec :raw text :code code :status :score-failed :err (:err is))
                  (assoc spec
                         :raw text :code code
                         :status :ok
                         :sanity-score (mx/item (:score (:trace sanity)))
                         :log-ml (:log-ml is)
                         :ess    (:ess is)))))))))))

(defn build-all
  "Sequential — all candidates share one model, concurrent runs would race the cache."
  [m specs i acc]
  (if (>= i (count specs))
    (pr/resolved acc)
    (pr/let [c (build-candidate m (nth specs i))]
      (build-all m specs (inc i) (conj acc c)))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(println "============================================================")
(println " Qwen3.6-35B-A3B-4bit  —  Prescribed-priors model selection")
(println "============================================================")
(println " observations:" observations "  (k=" k-heads "heads, n=" n-flips ")")
(println " candidates  :" (count candidate-specs)
         "  IS samples per candidate:" n-importance-samples)
(println "============================================================")

(pr/let
  [_   (println "\n[1] Loading model")
   t0  (now-ms)
   m   (llm/load-model model-path)
   _   (println "    type:" (:type m) "  load:" (fmt-ms t0))

   _   (println "\n[2] Synthesizing & scoring" (count candidate-specs) "candidates…")
   t1  (now-ms)
   results (build-all m candidate-specs 0 [])
   _   (println "    total time:" (fmt-ms t1))

   _   (doseq [c results]
         (println "\n  ─── " (:name c) " ───")
         (println "    prompt-extra:" (:prior-text c))
         (println "    code        :" (truncate (:code c) 240))
         (case (:status c)
           :ok          (do
                          (println "    sanity log-prior:" (.toFixed (:sanity-score c) 4))
                          (println "    IS log p(obs)   :" (.toFixed (:log-ml c) 4))
                          (println "    analytical      :" (.toFixed (:analytical c) 4))
                          (println "    error           :"
                                   (.toFixed (Math/abs (- (:log-ml c) (:analytical c))) 4))
                          (println "    ESS / N         :" (.toFixed (:ess c) 1)
                                   "/" n-importance-samples))
           :parse-failed (println "    PARSE FAILED   :" (:err c))
           :eval-failed  (println "    EVAL FAILED    :" (:err c))
           :sim-failed   (println "    SIMULATE FAILED:" (:err c))
           :score-failed (println "    SCORING FAILED :" (:err c))
           (println "    UNKNOWN STATUS :" (:status c))))

   ;; -------- Empirical ranking from IS estimates --------
   ok       (filter #(= :ok (:status %)) results)
   ranked   (vec (sort-by (fn [c] (- (:log-ml c))) ok))
   _   (println "\n[3] Empirical ranking by IS log p(obs)")
   _   (if (empty? ranked)
         (println "    (no successful candidates)")
         (do
           (println (str "    " (pad "rank" 5) (pad "name" 16)
                         (pad "IS log p(obs)" 16)
                         (pad "analytical" 14)
                         (pad "error" 10)
                         (pad "ESS" 8)))
           (println "    ──────────────────────────────────────────────────────────────────────")
           (doseq [[i c] (map-indexed vector ranked)]
             (println (str "    "
                           (pad (str (inc i)) 5)
                           (pad (:name c) 16)
                           (pad (.toFixed (:log-ml c) 4) 16)
                           (pad (.toFixed (:analytical c) 4) 14)
                           (pad (.toFixed (Math/abs (- (:log-ml c) (:analytical c))) 4) 10)
                           (pad (.toFixed (:ess c) 0) 8))))))

   ;; -------- Analytical ranking, for comparison --------
   _   (println "\n[4] Analytical ranking (ground truth)")
   _   (let [analytical-ranked (vec (sort-by (fn [c] (- (:analytical c))) candidate-specs))]
         (doseq [[i c] (map-indexed vector analytical-ranked)]
           (println (str "    "
                         (pad (str (inc i)) 5)
                         (pad (:name c) 16)
                         (pad (.toFixed (:analytical c) 4) 14)))))

   ;; -------- Did the IS ranking match the truth? --------
   _   (let [ok-names      (mapv :name ranked)
             analytical-by (mapv :name (vec (sort-by (fn [c] (- (:analytical c))) ok)))]
         (println "\n[5] Did IS recover the analytical ordering?")
         (println "    IS order        :" ok-names)
         (println "    analytical order:" analytical-by)
         (println "    match?          :" (= ok-names analytical-by)))

   _   (println "\n============================================================\n")]
  nil)
