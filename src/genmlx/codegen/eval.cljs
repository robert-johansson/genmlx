(ns genmlx.codegen.eval
  "Native-free ClojureScript reader/eval spine for code synthesis (genmlx-t246).

   These helpers depend ONLY on edamame (the reader) + sci.core (the evaluator) —
   no @mlx-node native code. They were extracted from genmlx.llm.codegen so a
   downstream consumer (e.g. arc3-solver) can use the synthesis-eval spine
   without loading the native LM stack (@mlx-node/lm) that genmlx.llm.codegen's
   other requires pull in. genmlx.llm.codegen re-exports these for back-compat."
  (:require [edamame.core :as eda]
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
