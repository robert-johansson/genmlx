(ns examples.qwen36-gen-synthesis
  "Step-by-step probe: can the Qwen3.6-35B-A3B-4bit base model synthesize
   a small GenMLX gen function from a natural-language description?

   Trick: the model writes `(fn [trace] body)` — a plain ClojureScript fn
   that takes the trace operation as a parameter. We wrap it into a real
   DynamicGF afterwards. This sidesteps the `gen` macro entirely, so the
   model only needs to know:
     - `(trace addr dist)` records a sample at addr from dist
     - dist/gaussian, dist/uniform, dist/bernoulli  -- the basic distributions
     - dist/beta is wired to dist/beta-dist for ergonomics

   Run: bun run --bun nbb examples/qwen36_gen_synthesis.cljs"
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
;; SCI environment. Inlined so the available API is visible at a glance.
;; The model can reference any of these in the (fn [trace] ...) it writes.
;; ---------------------------------------------------------------------------

(def sci-env
  {:namespaces
   {'dist {'gaussian   dist/gaussian
           'uniform    dist/uniform
           'bernoulli  dist/bernoulli
           'beta       dist/beta-dist
           'exponential dist/exponential}
    'mx   {'add        mx/add
           'multiply   mx/multiply
           'scalar     mx/scalar
           'item       mx/item}}})

;; ---------------------------------------------------------------------------
;; Task: a coin-flip model with a Beta(1,1) prior on the bias.
;; ---------------------------------------------------------------------------

(def task
  {:prompt
   (str
    "Write a probabilistic program for a single coin flip with an unknown bias.\n"
    "Sample the bias `theta` from a Beta(1, 1) prior, then sample one\n"
    "Bernoulli(theta) draw at address :flip. Return the flip outcome.\n\n"
    "Output ONLY the (fn [trace] ...) form. No prose, no markdown.")

   :few-shot
   (str
    "Probabilistic programs are written as `(fn [trace] body)` where\n"
    "`(trace <addr> <dist>)` samples and records under <addr>.\n"
    "Distributions available: dist/gaussian (mu sigma), dist/uniform (lo hi),\n"
    "dist/bernoulli (p), dist/beta (alpha beta), dist/exponential (rate).\n\n"
    "Example -- a noisy single-measurement model:\n"
    "(fn [trace]\n"
    "  (let [mu (trace :mu (dist/gaussian 0 10))\n"
    "        x  (trace :obs (dist/gaussian mu 1))]\n"
    "    x))\n")

   :expected-addresses #{[:theta] [:flip]}})

;; ---------------------------------------------------------------------------
;; Inline helpers (visible, small).
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

(defn sci-eval-fn
  "Eval `code` with `sci-env` exposed. Expect a fn value.
   Returns {:fn f} or {:err msg}."
  [code]
  (try
    (let [v (sci/eval-string code sci-env)]
      (cond
        (fn? v) {:fn v}
        (and (var? v) (fn? (deref v))) {:fn (deref v)}
        :else {:err (str "value is not a function: " (pr-str v))}))
    (catch :default e {:err (.-message e)})))

(defn wrap-as-gf
  "Promote `(fn [trace] body)` into a real DynamicGF. The outer `gen` body
   just forwards the closure-bound `trace` to the model fn -- so the model
   never had to know about the `gen` macro at all."
  [model-fn]
  (dyn/auto-key (gen [] (model-fn trace))))

(defn now-ms [] (.now js/performance))
(defn fmt-ms [t0] (str (.toFixed (- (now-ms) t0) 0) "ms"))

;; ---------------------------------------------------------------------------
;; Run.
;; ---------------------------------------------------------------------------

(println "============================================================")
(println " Qwen3.6-35B-A3B-4bit GenMLX gen-function synthesis probe")
(println "============================================================")

(pr/let
  [;; -------- Step 1: Load model. --------
   _   (println "\n[1] Loading model from" model-path)
   t0  (now-ms)
   m   (llm/load-model model-path)
   _   (println "    type     :" (:type m))
   _   (println "    load time:" (fmt-ms t0))

   ;; -------- Step 2: Compose prompt (few-shot + task). --------
   _   (println "\n[2] Prompt to LLM (few-shot teaching + task):")
   full-prompt (str (:few-shot task) "\nNow:\n" (:prompt task))
   _   (doseq [line (str/split full-prompt #"\n")]
         (println "    │" line))

   ;; -------- Step 3: Generate. --------
   _   (println "\n[3] Generating (greedy, max 200 tokens)…")
   t1  (now-ms)
   text (llm/generate-text-raw
         m full-prompt
         {:max-tokens 200
          :temperature 0
          :system-prompt
          "You are a probabilistic programming assistant. Output only valid ClojureScript code."})
   _   (println "    gen time :" (fmt-ms t1))
   _   (println "    raw output:")
   _   (doseq [line (str/split text #"\n")]
         (println "    │" line))

   ;; -------- Step 4: Extract & parse. --------
   _   (println "\n[4] Extract & parse")
   code (extract-code text)
   _   (println "    extracted:" (pr-str code))
   parse (parse-cljs code)
   _   (if (:ok? parse)
         (println "    parse    : ✓")
         (println "    parse    : ✗" (:err parse)))

   ;; -------- Step 5: Eval in SCI to get the model fn. --------
   _   (println "\n[5] SCI eval (with dist/* and mx/* in scope)")
   evald (when (:ok? parse) (sci-eval-fn code))
   _   (cond
         (nil? evald) (println "    skipped  : (parse failed)")
         (:err evald) (println "    err      :" (:err evald))
         :else (println "    ok       : got model fn"))

   ;; -------- Step 6: Wrap into a real DynamicGF and simulate. --------
   _   (println "\n[6] Wrap into DynamicGF and simulate")
   gf  (when (:fn evald) (wrap-as-gf (:fn evald)))
   sim-result
   (when gf
     (try {:trace (p/simulate gf [])}
          (catch :default e {:err (.-message e)})))
   _   (cond
         (nil? sim-result) (println "    skipped  : (eval failed)")
         (:err sim-result) (println "    err      :" (:err sim-result))
         :else (println "    ok       : simulate returned a trace"))

   ;; -------- Step 7: Inspect the trace. --------
   _   (println "\n[7] Trace inspection")
   trace (:trace sim-result)
   _   (when trace
         (let [choices (:choices trace)
               score   (mx/item (:score trace))
               retval  (:retval trace)]
           (println "    score    :" (.toFixed score 4))
           (println "    retval   :" (pr-str (cond-> retval (mx/array? retval) mx/item)))
           (println "    addresses:" (pr-str (cm/addresses choices)))
           (doseq [path (cm/addresses choices)]
             (let [v  (cm/get-choice choices path)
                   v* (cond-> v (mx/array? v) mx/item)]
               (println "    " path "=" (pr-str v*))))))

   ;; -------- Step 8: Verify against task expectations. --------
   _   (println "\n[8] Property checks")
   checks
   (when trace
     (let [choices (:choices trace)
           paths   (set (cm/addresses choices))
           score   (mx/item (:score trace))
           unwrap  (fn [path]
                     (when (contains? paths path)
                       (let [v (cm/get-choice choices path)]
                         (cond-> v (mx/array? v) mx/item))))
           theta   (unwrap [:theta])
           flip    (unwrap [:flip])]
       [{:label "expected addresses present"
         :pass? (= paths (:expected-addresses task))
         :info  (pr-str paths)}
        {:label "score is finite"
         :pass? (and (number? score) (js/isFinite score))
         :info  (str score)}
        {:label "theta in [0,1]"
         :pass? (and (number? theta) (<= 0 theta) (<= theta 1))
         :info  (pr-str theta)}
        {:label "flip is 0 or 1"
         :pass? (and (number? flip) (or (= flip 0) (= flip 1)))
         :info  (pr-str flip)}]))
   _   (when checks
         (doseq [{:keys [label pass? info]} checks]
           (println "    " (if pass? "✓" "✗") label "  →" info)))

   ;; -------- Summary. --------
   _   (let [total (count (or checks []))
             passed (count (filter :pass? (or checks [])))]
         (println "\n============================================================")
         (println " Result:" passed "/" total "checks passed")
         (println "============================================================\n"))]
  nil)
