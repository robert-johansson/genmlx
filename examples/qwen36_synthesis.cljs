(ns examples.qwen36-synthesis
  "Step-by-step probe: can the Qwen3.6-35B-A3B-4bit base model synthesize
   correct ClojureScript without fine-tuning?

   No reader-constrained byte loop, no msa, no codegen helpers.
   Just: load → prompt → generate → extract → parse → eval → test.
   Every step is visible top-to-bottom and prints what it sees.

   Run: bun run --bun nbb examples/qwen36_synthesis.cljs"
  (:require [genmlx.llm.backend :as llm]
            [edamame.core :as eda]
            [sci.core :as sci]
            [clojure.string :as str]
            [promesa.core :as pr]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/Qwen3.6-35B-A3B-4bit"))

;; ---------------------------------------------------------------------------
;; The task. One concrete function, one small set of cases.
;; ---------------------------------------------------------------------------

(def task
  {:prompt
   (str "Write a ClojureScript function named `factorial` that computes "
        "n! for non-negative integers n (with 0! = 1).\n"
        "Output ONLY the (defn ...) form. No prose, no markdown fences.")
   :fn-name 'factorial
   :tests [{:in 0 :out 1}
           {:in 1 :out 1}
           {:in 5 :out 120}
           {:in 7 :out 5040}]})

;; ---------------------------------------------------------------------------
;; Tiny helpers, kept inline so they stay visible.
;; ---------------------------------------------------------------------------

(defn extract-code
  "Pull a (...) form out of free-form LLM text.
   Strategy: prefer fenced block, else first paren onward."
  [text]
  (let [t (str/trim (or text ""))]
    (cond
      (str/blank? t) ""
      (re-find #"```" t)
      (let [m (re-find #"```(?:clojure|cljs|clj|clojurescript)?\s*\n?([\s\S]*?)```" t)]
        (if m (str/trim (nth m 1)) t))
      (str/starts-with? t "(") t
      :else (let [i (str/index-of t "(")] (if i (subs t i) "")))))

(defn parse-cljs
  "Edamame parse with all reader macros enabled.
   Returns {:ok? bool :form _ :err _}."
  [code]
  (try
    {:ok? true :form (eda/parse-string code {:all true})}
    (catch :default e
      {:ok? false :err (.-message e)})))

(defn sci-eval-fn
  "Evaluate code in SCI, return {:fn f} or {:err msg}.
   Handles both `(fn ...)` (returns fn) and `(defn name ...)` (returns var)."
  [code]
  (try
    (let [v (sci/eval-string code)]
      (cond
        (fn? v) {:fn v}
        (and (var? v) (fn? (deref v))) {:fn (deref v)}
        :else {:err (str "value is not a function: " (pr-str v))}))
    (catch :default e {:err (.-message e)})))

(defn now-ms [] (.now js/performance))

(defn fmt-ms [start] (str (.toFixed (- (now-ms) start) 0) "ms"))

;; ---------------------------------------------------------------------------
;; Run it.
;; ---------------------------------------------------------------------------

(println "============================================================")
(println " Qwen3.6-35B-A3B-4bit ClojureScript synthesis probe")
(println "============================================================")

(pr/let
  [;; -------- Step 1: Load the model. --------
   _   (println "\n[1] Loading model from" model-path)
   t0  (now-ms)
   m   (llm/load-model model-path)
   _   (println "    type        :" (:type m))
   _   (println "    vocab size  :" (llm/vocab-size (:tokenizer m)))
   _   (println "    EOS token id:" (llm/eos-token-id (:tokenizer m)))
   _   (println "    load time   :" (fmt-ms t0))

   ;; -------- Step 2: Show the prompt. --------
   _   (println "\n[2] Prompt:")
   _   (doseq [line (str/split (:prompt task) #"\n")]
         (println "    " line))

   ;; -------- Step 3: Generate (greedy, raw chat template). --------
   ;;   Using generate-text-raw so the ChatML scaffold is explicit
   ;;   (it injects the qwen3_5_moe think-skip tokens automatically,
   ;;   see backend.cljs:194).
   _   (println "\n[3] Generating (greedy, max 256 tokens)…")
   t1  (now-ms)
   text (llm/generate-text-raw
         m (:prompt task)
         {:max-tokens 256
          :temperature 0
          :system-prompt
          "You are a ClojureScript code assistant. Output only valid ClojureScript code."})
   _   (println "    gen time    :" (fmt-ms t1))
   _   (println "    raw output  :")
   _   (doseq [line (str/split text #"\n")]
         (println "    │" line))

   ;; -------- Step 4: Extract code. --------
   _   (println "\n[4] Extracting code")
   code (extract-code text)
   _   (println "    extracted   :" (pr-str code))

   ;; -------- Step 5: Parse with edamame. --------
   _   (println "\n[5] Parsing")
   parse (parse-cljs code)
   _   (if (:ok? parse)
         (println "    ok          : ✓  form =" (pr-str (:form parse)))
         (println "    ok          : ✗  err  =" (:err parse)))

   ;; -------- Step 6: SCI eval. --------
   _   (println "\n[6] SCI eval")
   evald (when (:ok? parse) (sci-eval-fn code))
   _   (cond
         (nil? evald) (println "    skipped     : (parse failed)")
         (:err evald) (println "    err         :" (:err evald))
         :else (println "    ok          : function ready"))

   ;; -------- Step 7: Run test cases. --------
   _   (println "\n[7] Test cases")
   results
   (when (:fn evald)
     (mapv (fn [{:keys [in out]}]
             (try (let [actual ((:fn evald) in)]
                    {:in in :out out :actual actual :pass? (= out actual)})
                  (catch :default e
                    {:in in :out out :err (.-message e) :pass? false})))
           (:tests task)))
   _   (when results
         (doseq [{:keys [in out actual err pass?]} results]
           (println "    " (if pass? "✓" "✗")
                    (str "(" (:fn-name task) " " in ")")
                    "=>" (or err (pr-str actual))
                    (when-not pass? (str "  expected " out)))))

   ;; -------- Summary. --------
   _   (println "\n============================================================")
   _   (let [total (count (:tests task))
             passed (count (filter :pass? (or results [])))]
         (println " Result:" passed "/" total "tests passed"))
   _   (println "============================================================\n")]
  nil)
