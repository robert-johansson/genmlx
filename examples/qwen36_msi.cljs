(ns examples.qwen36-msi
  "Model Synthesis as Inference: synthesize a probabilistic program with the
   LLM, then use it as a regular GenMLX gen function — condition on
   observations and estimate log marginal likelihood by importance sampling.

   Pipeline:
     prompt → LLM → cljs source → SCI eval → wrap to DynamicGF
       → p/simulate (sanity)
       → p/generate × N with observations  (importance sampling)
       → log marginal likelihood estimate
       → compare to analytical Beta-Binomial ground truth.

   Run: bun run --bun nbb examples/qwen36_msi.cljs"
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
;; SCI environment.  Adds `keyword` & `str` (already in clojure.core but
;; spelled out here so the API exposed to the LLM is fully visible).
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
;; Task
;; ---------------------------------------------------------------------------

(def observations
  "Eight heads, two tails."
  [1 1 1 1 1 1 1 1 0 0])

(def n-flips (count observations))

(def n-importance-samples 500)

;; Beta-Binomial marginal likelihood with Beta(α=1, β=1) prior, k=8 heads, n=10:
;;   p(obs) = B(α+k, β+n−k) / B(α, β) = B(9, 3) / B(1, 1) = (8! · 2!)/11! = 1/495
(def analytical-log-ml (- (Math/log 495)))

(def task
  {:prompt
   (str "Write a probabilistic program of two arguments [trace n]:\n"
        " - sample a coin bias `theta` from Beta(1, 1) at address :theta\n"
        " - sample `n` IID Bernoulli(theta) draws at addresses :flip0, :flip1, …, :flip<n-1>\n"
        " - return the vector of flips\n\n"
        "Output ONLY the (fn [trace n] ...) form.")

   :few-shot
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
    "          (range n))))\n")})

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

(defn wrap-as-gf
  "Wrap `(fn [trace n] body)` into a DynamicGF whose simulate/generate take [n]."
  [model-fn]
  (dyn/auto-key (gen [n] (model-fn trace n))))

(defn observations->constraints
  "Build a choicemap with :flip0…:flipK constrained to the given int values."
  [obs]
  (reduce (fn [cm [i v]]
            (cm/set-choice cm
                           [(keyword (str "flip" i))]
                           (mx/scalar v mx/int32)))
          cm/EMPTY
          (map-indexed vector obs)))

(defn log-sum-exp
  "Numerically stable log-sum-exp."
  [xs]
  (let [m (apply max xs)]
    (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))))

(defn now-ms [] (.now js/performance))
(defn fmt-ms [t0] (str (.toFixed (- (now-ms) t0) 0) "ms"))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(println "============================================================")
(println " Qwen3.6-35B-A3B-4bit  —  Model Synthesis as Inference")
(println "============================================================")

(pr/let
  [;; -------- Step 1: Load model --------
   _   (println "\n[1] Loading model from" model-path)
   t0  (now-ms)
   m   (llm/load-model model-path)
   _   (println "    type     :" (:type m) "   load:" (fmt-ms t0))

   ;; -------- Step 2: Prompt --------
   _   (println "\n[2] Prompt to LLM (few-shot teaches iteration over addresses):")
   full-prompt (str (:few-shot task) "\nNow:\n" (:prompt task))
   _   (doseq [line (str/split full-prompt #"\n")]
         (println "    │" line))

   ;; -------- Step 3: Generate --------
   _   (println "\n[3] Generating (greedy, max 250 tokens)…")
   t1  (now-ms)
   text (llm/generate-text-raw
         m full-prompt
         {:max-tokens 250
          :temperature 0
          :system-prompt
          "You are a probabilistic programming assistant. Output only valid ClojureScript code."})
   _   (println "    gen time :" (fmt-ms t1))
   _   (println "    raw output:")
   _   (doseq [line (str/split text #"\n")]
         (println "    │" line))

   ;; -------- Step 4: Extract & parse --------
   _   (println "\n[4] Extract & parse")
   code (extract-code text)
   _   (println "    extracted:" (pr-str code))
   parse (parse-cljs code)
   _   (if (:ok? parse)
         (println "    parse    : ✓")
         (println "    parse    : ✗" (:err parse)))

   ;; -------- Step 5: SCI eval --------
   _   (println "\n[5] SCI eval")
   evald (when (:ok? parse) (sci-eval-fn code))
   _   (cond
         (nil? evald) (println "    skipped")
         (:err evald) (println "    err      :" (:err evald))
         :else (println "    ok       : got model fn"))

   ;; -------- Step 6: Wrap + simulate (sanity) --------
   _   (println "\n[6] Wrap + simulate (sanity check)")
   gf  (when (:fn evald) (wrap-as-gf (:fn evald)))
   sim-trace
   (when gf
     (try (p/simulate gf [n-flips])
          (catch :default e (println "    err      :" (.-message e)) nil)))
   _   (when sim-trace
         (let [paths (cm/addresses (:choices sim-trace))]
           (println "    addresses:" (count paths) "trace sites,"
                    "expecting" (inc n-flips) "(:theta + :flip0..flipN)")
           (println "    score    :" (.toFixed (mx/item (:score sim-trace)) 4))))

   ;; -------- Step 7: Build observation constraints --------
   _   (println "\n[7] Observation constraints")
   _   (println "    observations:" observations
                "  (k=" (apply + observations) "heads, n=" n-flips ")")
   constraints (observations->constraints observations)
   _   (println "    constraint paths:" (cm/addresses constraints))

   ;; -------- Step 8: Single p/generate --------
   _   (println "\n[8] Single p/generate")
   single (when gf
            (try (p/generate gf [n-flips] constraints)
                 (catch :default e (println "    err      :" (.-message e)) nil)))
   _   (when single
         (let [w (mx/item (:weight single))
               theta-val (cm/get-choice (:choices (:trace single)) [:theta])
               theta* (cond-> theta-val (mx/array? theta-val) mx/item)]
           (println "    weight   :" (.toFixed w 4) " (= log p(obs | one trace))")
           (println "    theta    :" (.toFixed theta* 4)
                    " (sampled from prior, not posterior)")))

   ;; -------- Step 9: Importance sampling for log marginal likelihood --------
   _   (println "\n[9] Importance sampling — log p(obs)")
   _   (println "    drawing" n-importance-samples "weighted samples…")
   t2  (now-ms)
   weights
   (when gf
     (try
       (mapv (fn [_]
               (let [r (p/generate gf [n-flips] constraints)]
                 (mx/item (:weight r))))
             (range n-importance-samples))
       (catch :default e
         (println "    err      :" (.-message e))
         nil)))
   _   (when weights
         (let [log-ml (- (log-sum-exp weights) (Math/log n-importance-samples))
               ess    (/ (Math/pow (reduce + (map Math/exp
                                                  (map #(- % (apply max weights)) weights)))
                                   2)
                         (reduce + (map #(Math/exp (* 2 (- % (apply max weights)))) weights)))]
           (println "    is time  :" (fmt-ms t2))
           (println "    estimate : log p(obs) ≈" (.toFixed log-ml 4))
           (println "    analytic : log p(obs)  =" (.toFixed analytical-log-ml 4)
                    "  (Beta-Binomial closed form)")
           (println "    error    :" (.toFixed (Math/abs (- log-ml analytical-log-ml)) 4))
           (println "    ESS      :" (.toFixed ess 1) "/" n-importance-samples)))

   _   (println "\n============================================================\n")]
  nil)
