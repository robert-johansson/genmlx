(ns genmlx.codegen.eval
  "Native-free ClojureScript reader/eval spine for code synthesis (genmlx-t246).

   These helpers depend ONLY on edamame (the reader) + sci.core (the evaluator) —
   no @mlx-node native code. They were extracted from genmlx.llm.codegen so a
   downstream consumer (e.g. arc3-solver) can use the synthesis-eval spine
   without loading the native LM stack (@mlx-node/lm) that genmlx.llm.codegen's
   other requires pull in. genmlx.llm.codegen re-exports these for back-compat."
  (:require [clojure.string :as str]
            [edamame.core :as eda]
            [sci.core :as sci]))

(def ^:private eda-opts
  "Edamame parse options: enable all reader macros (#(), @, #\"\", ', etc.)"
  {:all true})

(defn prefix-status
  "Classify a string prefix using edamame (SCI's reader).
   Returns :complete, :incomplete, or :invalid.
   Handles all ClojureScript syntax including #(), @deref, #\"regex\", etc."
  [s]
  (try (eda/parse-string s eda-opts)
       :complete
       (catch :default e
         (if (re-find #"EOF" (.-message e))
           :incomplete
           :invalid))))

(defn parse-form
  "Parse a string into a ClojureScript form via edamame, returning nil if it
   does not parse. Used to recover a form from generated code best-effort."
  [s]
  (try (eda/parse-string s eda-opts) (catch :default _ nil)))

(defn valid-cljs?
  "Is code-str a complete, syntactically valid ClojureScript form?"
  [code-str]
  (and (seq code-str) (= :complete (prefix-status code-str))))

(defn eval-cljs
  "Evaluate a ClojureScript string in SCI. Returns {:result val} or {:error string}."
  [code-str]
  (try {:result (sci/eval-string code-str)}
       (catch :default e
         {:error (.-message e)})))

(defn eval-fn
  "Evaluate code-str, expecting a function result.
   Handles both (fn ...) which returns a function directly,
   and (defn name ...) which returns a var containing a function.
   Returns {:fn function} or {:error string}."
  [code-str]
  (let [r (eval-cljs code-str)
        v (:result r)]
    (cond
      (:error r) r
      (fn? v) {:fn v}
      (and (var? v) (fn? (deref v))) {:fn (deref v)}
      :else {:error "Result is not a function"})))

;; ===========================================================================
;; Code extraction, behavioral verification, and structural scoring.
;; Native-free (string + reader/eval only); genmlx.llm.codegen re-exports these
;; so the synthesis-eval spine (and the Phase-1 reward path, genmlx-ugkv) can use
;; them WITHOUT loading the native LM stack that genmlx.llm.codegen pulls in.
;; ===========================================================================

(defn extract-code
  "Extract ClojureScript code from LLM output text (peels markdown fences / prose)."
  [text]
  (cond
    (not (seq text))
    ""

    ;; Fenced code block
    (re-find #"```(?:clojure|cljs|clojurescript|clj)?\s*\n" text)
    (if-let [m (re-find #"```(?:clojure|cljs|clojurescript|clj)?\s*\n([\s\S]*?)```" text)]
      (str/trim (nth m 1))
      "")

    ;; Starts with paren -- raw code
    (str/starts-with? (str/trim text) "(")
    (str/trim text)

    ;; Strip prefix to first paren
    :else
    (if-let [idx (str/index-of text "(")]
      (subs text idx)
      "")))

(defn verify-transition-fn
  "Verify a transition function against observed transitions.

   code-str:    ClojureScript evaluating to (fn [state action] -> state)
   transitions: [{:state map :action keyword :expected map}]

   Returns {:accuracy :total :correct :failures :error?}"
  [code-str transitions]
  (let [total (count transitions)
        r (eval-fn code-str)
        f (:fn r)]
    (if (:error r)
      {:accuracy 0.0 :total total :correct 0 :failures [] :error (:error r)}
      (let [results (map-indexed
                      (fn [i {:keys [state action expected]}]
                        (try
                          (let [actual (f state action)]
                            (if (= expected actual)
                              {:correct true}
                              {:correct false
                               :index i :state state :action action
                               :expected expected :actual actual}))
                          (catch :default e
                            {:correct false
                             :index i :state state :action action
                             :expected expected :actual (str "ERROR: " (.-message e))})))
                      transitions)
            correct (count (filter :correct results))
            failures (vec (remove :correct results))]
        {:accuracy (if (zero? total) 1.0 (/ correct total))
         :total total
         :correct correct
         :failures failures}))))

(defn- form-contains?
  "Does the form tree contain this symbol anywhere?"
  [form sym]
  (cond
    (= form sym) true
    (sequential? form) (some #(form-contains? % sym) form)
    :else false))

(defn- count-occurrences
  "Count occurrences of sym in form tree."
  [form sym]
  (cond
    (= form sym) 1
    (sequential? form) (transduce (map #(count-occurrences % sym)) + 0 form)
    :else 0))

(defn- returns-map-literals?
  "Check if case/cond branches return map literals."
  [form]
  (cond
    (map? form) true
    (and (sequential? form) (= 'case (first form)))
    (let [branches (drop 2 form)
          values (take-nth 2 (rest branches))]
      (some map? values))
    (sequential? form) (some returns-map-literals? form)
    :else false))

(def ^:private occurrence-weights
  "Per-occurrence penalty weights for commonly misused patterns."
  {'assoc -2 'assoc-in -3 'update-in -3 'cond-> -4})

(defn score-structure
  "Score the structural quality of a ClojureScript form.
   Higher = more idiomatic for transition functions.

   Rewards: case dispatch, map literal returns, dec/inc, :keys destructuring.
   Penalizes: assoc, assoc-in, update-in, cond-> (commonly misused patterns)."
  [form]
  (+ (if (form-contains? form 'case) 5 0)
     (if (returns-map-literals? form) 4 0)
     (if (form-contains? form 'dec) 3 0)
     (if (form-contains? form 'inc) 3 0)
     (if (form-contains? form :keys) 2 0)
     (reduce-kv (fn [acc sym w] (+ acc (* w (count-occurrences form sym))))
                0 occurrence-weights)))
