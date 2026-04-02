(ns genmlx.llm.codegen
  "Grammar-constrained ClojureScript code generation from an LLM.

   The key insight: cljs.reader/read-string distinguishes three outcomes:
     1. Succeeds        -> complete valid form
     2. Throws 'EOF...' -> incomplete but valid prefix
     3. Throws other    -> invalid

   This three-way classification IS an incremental grammar for ClojureScript.
   At each byte step, we test candidate bytes against the reader and mask
   invalid ones to -inf. The LLM can only produce syntactically valid code.

   Layers:
     7.1  Reader constraint (prefix-status, valid-next-bytes, reader-constraint)
     7.2  Post-hoc validation (valid-cljs?, fn-form?, transition-fn-form?)
     7.3  Chat template (format-chat, code-system-prompt)
     7.4  Code extraction (extract-code)
     7.5  Generation (generate-cljs, generate-cljs-n)
     7.6  Reader-constrained byte GF (make-reader-constrained-gf)
     7.7  Execution (eval-cljs, eval-fn)
     7.8  Verification (verify-transition-fn)
     7.9  Revision (revise)
     7.10 Synthesis loop (synthesize-loop)
     7.11 Structural scoring (score-structure)
     7.12 Scored generation (generate-and-score, generate-and-rank)"
  (:require [edamame.core :as eda]
            [clojure.string :as str]
            [sci.core :as sci]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.llm.bytes :as bytes]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================
;; 7.1 Reader-as-grammar constraint
;; ============================================================

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

(def ^:private candidate-bytes
  "Printable ASCII + whitespace bytes as single-char strings."
  (into (mapv #(js/String.fromCharCode %) (range 32 127))
        ["\n" "\t"]))

(defn valid-next-bytes
  "Return the set of bytes that maintain a valid prefix."
  [prefix]
  (into #{}
        (filter (fn [b]
                  (#{:incomplete :complete} (prefix-status (str prefix b)))))
        candidate-bytes))

(defn reader-constraint
  "Return a map of {char -> :incomplete|:complete} for valid next bytes.
   Invalid bytes are excluded."
  [prefix]
  (into {}
        (keep (fn [b]
                (let [s (prefix-status (str prefix b))]
                  (when (not= :invalid s) [b s]))))
        candidate-bytes))

(defn- suppress-complete
  "Before min-bytes, treat :complete as :incomplete to prevent
   premature stopping on single-char atoms like '<'."
  [constraint]
  (into {} (map (fn [[ch s]] [ch (if (= :complete s) :incomplete s)])) constraint))

;; ============================================================
;; 7.2 Post-hoc validation
;; ============================================================

(defn valid-cljs?
  "Is code-str a complete, syntactically valid ClojureScript form?"
  [code-str]
  (and (seq code-str) (= :complete (prefix-status code-str))))

(defn fn-form?
  "Is form a (fn ...) expression?"
  [form]
  (and (list? form) (= 'fn (first form))))

(defn defn-form?
  "Is form a (defn ...) expression?"
  [form]
  (and (list? form) (= 'defn (first form))))

(defn transition-fn-form?
  "Is form a (fn [state action] ...) -- a 2-arg fn?"
  [form]
  (and (fn-form? form)
       (let [args (second form)]
         (and (vector? args) (= 2 (count args))))))

;; ============================================================
;; 7.3 Chat template
;; ============================================================

(def ^:private code-system-prompt
  "You are a ClojureScript code assistant. Output ONLY valid ClojureScript code.
No explanations, no markdown fences, no comments unless part of the code.
Syntax: (fn [args] body), (let [bindings] body), (case val clauses default),
:keywords, {:maps}, [vectors], #{sets}. Example: (fn [x] (+ x 1))")

(defn format-chat
  "Format a Qwen3 chat prompt from system and user messages.
   Includes think-skip tokens to bypass Qwen3's thinking mode
   and go straight to code output."
  [system-msg user-msg]
  (str "<|im_start|>system\n" system-msg "<|im_end|>\n"
       "<|im_start|>user\n" user-msg "<|im_end|>\n"
       "<|im_start|>assistant\n<think>\n\n</think>\n\n"))

;; ============================================================
;; 7.4 Code extraction
;; ============================================================

(defn extract-code
  "Extract ClojureScript code from LLM output text."
  [text]
  (if-not (seq text)
    ""
    (cond
      ;; Fenced code block
      (re-find #"```(?:clojure|cljs|clojurescript|clj)?\s*\n" text)
      (let [m (re-find #"```(?:clojure|cljs|clojurescript|clj)?\s*\n([\s\S]*?)```" text)]
        (if m (str/trim (nth m 1)) ""))

      ;; Starts with paren -- raw code
      (str/starts-with? (str/trim text) "(")
      (str/trim text)

      ;; Strip prefix to first paren
      :else
      (let [idx (str/index-of text "(")]
        (if idx (subs text idx) "")))))

;; ============================================================
;; 7.5 Reader-constrained byte GF
;; ============================================================

(defn make-reader-constrained-gf
  "Create a byte-level GF that only produces valid ClojureScript.

   Uses cljs.reader/read-string as an incremental grammar constraint.
   At each byte, filters byte marginals to those maintaining a valid prefix.
   Stops when the prefix forms a complete ClojureScript expression.

   model-map: {:model :tokenizer :type} from llm/load-model.
   opts:
     :trie       pre-built byte trie (from bytes/build-byte-trie)
     :min-bytes  minimum bytes before allowing :complete to stop (default 10)
     -- pass (bytes/prepare tokenizer) to share across GFs."
  ([model-map] (make-reader-constrained-gf model-map {}))
  ([model-map opts]
   (let [{:keys [model]} model-map
         {:keys [min-bytes] :or {min-bytes 10}} opts
         {:keys [trie]} (if (:trie opts)
                          opts
                          (bytes/prepare (:tokenizer model-map)))]
     (dyn/auto-key
      (gen [prompt-ids max-bytes]
           (if (zero? max-bytes)
             []
             (do
               (llm/init-cache! model)
               (try
                 (loop [i 0
                        trie-pos trie
                        prefix ""
                        logprobs (bytes/logits->logprobs
                                  (llm/forward-prefill model prompt-ids))
                        bytes-acc []]
                   (if (>= i max-bytes)
                     bytes-acc
                     (let [raw-lps (bytes/byte-logprobs trie-pos logprobs)
                           constraint (reader-constraint prefix)
                           effective (if (< i min-bytes)
                                       (suppress-complete constraint)
                                       constraint)
                           valid-lps (into {}
                                           (filter (fn [[ch _]]
                                                     (contains? effective ch)))
                                           raw-lps)]
                       (if (empty? valid-lps)
                         bytes-acc
                         (let [{:keys [dist chars]}
                               (bytes/byte-marginals->categorical valid-lps)
                               idx (trace (keyword (str "b" i)) dist)
                               chosen-byte (nth chars (mx/item idx))
                               status (get effective chosen-byte)
                               new-prefix (str prefix chosen-byte)
                               next-node (get-in trie-pos [:children chosen-byte])
                               new-acc (conj bytes-acc chosen-byte)]
                           (if (= :complete status)
                             new-acc
                             (if (bytes/trie-leaf? next-node)
                               (recur (inc i) trie new-prefix
                                      (bytes/logits->logprobs
                                       (llm/forward-step model
                                                         (bytes/commit-token-id next-node)))
                                      new-acc)
                               (recur (inc i) next-node new-prefix logprobs
                                      new-acc))))))))
                 (finally
                   (llm/reset-cache! model))))))))))

;; ============================================================
;; 7.6 Code generation
;; ============================================================

(defn generate-cljs
  "Generate ClojureScript code from a natural language prompt.

   Two modes:
     :token-level? true  — use model's native chat API (fast, best with
                           fine-tuned models), validate post-hoc with reader
     :token-level? false — byte-level with reader constraint (guaranteed
                           valid syntax, but slower and can distort fine-tuned
                           model quality)

   model-map: {:model :tokenizer :type} from llm/load-model.
   prompt:    natural language description of what to generate.
   opts:
     :token-level?   use token-level generation (default true)
     :max-tokens     max tokens for token-level (default 200)
     :temperature    sampling temperature for token-level (default 0.3)
     :max-bytes      max bytes for byte-level (default 500)
     :min-bytes      min bytes before :complete can stop (default 10)
     :system-prompt  override code-system-prompt
     :prepared       {:token-index :trie} from bytes/prepare"
  ([model-map prompt] (generate-cljs model-map prompt {}))
  ([model-map prompt opts]
   (let [{:keys [token-level? system-prompt]
          :or {token-level? true system-prompt code-system-prompt}} opts]
     (if token-level?
       ;; Token-level: native chat API + post-hoc validation
       (let [{:keys [max-tokens temperature]
              :or {max-tokens 200 temperature 0.3}} opts]
         (pr/let [text (llm/generate-text model-map prompt
                                          {:max-tokens max-tokens
                                           :temperature temperature
                                           :system-prompt system-prompt})
                  code (extract-code text)
                  valid (valid-cljs? code)
                  form (when valid
                         (try (eda/parse-string code eda-opts) (catch :default _ nil)))]
           {:code code
            :valid? valid
            :form form
            :text text}))
       ;; Byte-level: reader-constrained generation
       (let [{:keys [tokenizer]} model-map
             {:keys [max-bytes] :or {max-bytes 500}} opts
             chat-str (format-chat system-prompt prompt)
             gf (make-reader-constrained-gf model-map opts)]
         (pr/let [ids (llm/encode tokenizer chat-str true)
                  prompt-ids (vec ids)
                  trace (p/simulate gf [prompt-ids max-bytes])
                  text (apply str (:retval trace))
                  code (extract-code text)
                  valid (valid-cljs? code)
                  form (when valid
                         (try (eda/parse-string code eda-opts) (catch :default _ nil)))]
           {:code code
            :valid? valid
            :form form
            :trace trace
            :text text}))))))

(defn generate-cljs-n
  "Generate n independent ClojureScript candidates.
   Returns a vector sorted by :valid? descending."
  ([model-map prompt n] (generate-cljs-n model-map prompt n {}))
  ([model-map prompt n opts]
   (pr/let [results (pr/all (repeatedly n #(generate-cljs model-map prompt opts)))]
     (vec (sort-by (fn [r] (if (:valid? r) 0 1)) results)))))

;; ============================================================
;; 7.7 Code execution
;; ============================================================

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

;; ============================================================
;; 7.8 Transition function verification
;; ============================================================

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

;; ============================================================
;; 7.9 Revision
;; ============================================================

(defn- format-failures [failures]
  (str/join "\n\n"
    (map (fn [{:keys [state action expected actual]}]
           (str "State: " (pr-str state) "\n"
                "Action: " (pr-str action) "\n"
                "Expected: " (pr-str expected) "\n"
                "Got: " (pr-str actual)))
         failures)))

(defn revise
  "Revise a failing transition function using the LLM.

   model-map: the LLM
   code-str:  current failing code
   failures:  vector from verify-transition-fn
   opts:      same as generate-cljs"
  ([model-map code-str failures] (revise model-map code-str failures {}))
  ([model-map code-str failures opts]
   (let [prompt (str "This ClojureScript transition function has bugs:\n\n"
                     code-str "\n\n"
                     "It fails on these transitions:\n\n"
                     (format-failures failures) "\n\n"
                     "Write a corrected version of the function.")]
     (generate-cljs model-map prompt opts))))

;; ============================================================
;; 7.10 Synthesis loop
;; ============================================================

(defn synthesize-loop
  "Generate, verify, and revise a transition function until it works.

   model-map:   the LLM
   prompt:      natural language description for initial generation
   transitions: test cases [{:state :action :expected}]
   opts:
     :max-revisions    max revision rounds (default 3)
     :target-accuracy  stop threshold (default 1.0)
     + all generate-cljs opts"
  ([model-map prompt transitions]
   (synthesize-loop model-map prompt transitions {}))
  ([model-map prompt transitions opts]
   (let [{:keys [max-revisions target-accuracy]
          :or {max-revisions 3 target-accuracy 1.0}} opts]
     (pr/let [initial (generate-cljs model-map prompt opts)
              code (:code initial)
              result (verify-transition-fn code transitions)]
       (pr/loop [rev 0
                 code code
                 accuracy (:accuracy result)
                 failures (:failures result)
                 history [{:code code
                           :accuracy (:accuracy result)
                           :failures (:failures result)}]
                 best-code code
                 best-accuracy (:accuracy result)]
         (if (or (>= accuracy target-accuracy) (>= rev max-revisions))
           {:code (if (>= accuracy best-accuracy) code best-code)
            :accuracy (max accuracy best-accuracy)
            :revisions rev
            :converged? (>= (max accuracy best-accuracy) target-accuracy)
            :history history}
           (pr/let [revised (revise model-map code failures opts)
                    new-code (:code revised)
                    result (verify-transition-fn new-code transitions)
                    new-acc (:accuracy result)
                    new-failures (:failures result)]
             (pr/recur (inc rev)
                       new-code
                       new-acc
                       new-failures
                       (conj history {:code new-code
                                      :accuracy new-acc
                                      :failures new-failures})
                       (if (> new-acc best-accuracy) new-code best-code)
                       (max new-acc best-accuracy)))))))))

;; ============================================================
;; 7.11 Structural scoring
;; ============================================================

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
    (sequential? form) (reduce + 0 (map #(count-occurrences % sym) form))
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
     (* -2 (count-occurrences form 'assoc))
     (* -3 (count-occurrences form 'assoc-in))
     (* -3 (count-occurrences form 'update-in))
     (* -4 (count-occurrences form 'cond->))))

;; ============================================================
;; 7.12 Scored generation
;; ============================================================

(defn generate-and-score
  "Generate code via chat API, then score it through the GFI.
   Returns the model's log-probability (weight) alongside the code.

   model-map: {:model :tokenizer :type} from llm/load-model.
   gf:        a token-level GF from llm-core/make-llm-gf on the same model.
   prompt:    natural language description.
   opts:      same as generate-cljs, plus generate-cljs opts."
  ([model-map gf prompt] (generate-and-score model-map gf prompt {}))
  ([model-map gf prompt opts]
   (let [{:keys [tokenizer]} model-map
         {:keys [temperature max-tokens system-prompt]
          :or {temperature 0.7 max-tokens 200 system-prompt code-system-prompt}} opts]
     (pr/let [text (llm/generate-text model-map prompt
                     {:max-tokens max-tokens :temperature temperature
                      :system-prompt system-prompt})
              code (extract-code text)
              valid (valid-cljs? code)
              form (when valid
                     (try (eda/parse-string code eda-opts) (catch :default _ nil)))
              ;; Score via GFI: encode generated tokens, constrain all
              gen-ids (llm/encode tokenizer text false)
              gen-vec (vec gen-ids)
              constraints (apply cm/choicemap
                            (mapcat (fn [i id]
                                      [(keyword (str "t" i)) (mx/scalar id mx/int32)])
                                    (range) gen-vec))
              prompt-str (format-chat system-prompt prompt)
              raw-ids (llm/encode tokenizer prompt-str true)
              prompt-ids (vec raw-ids)
              {:keys [weight]} (p/generate gf [prompt-ids (count gen-vec)] constraints)]
       {:code code
        :valid? valid
        :form form
        :text text
        :weight (mx/item weight)
        :struct-score (if form (score-structure form) -10)}))))

(defn generate-and-rank
  "Generate N candidates, score each with model weight + structural quality,
   optionally verify against transitions. Returns candidates sorted by
   combined score (best first).

   model-map:   {:model :tokenizer :type} from llm/load-model.
   gf:          token-level GF from llm-core/make-llm-gf.
   prompt:      natural language description.
   n:           number of candidates.
   opts:
     :transitions    [{:state :action :expected}] for behavioral testing
     :struct-weight  multiplier for structural score (default 2)
     + all generate-and-score opts"
  ([model-map gf prompt n] (generate-and-rank model-map gf prompt n {}))
  ([model-map gf prompt n opts]
   (let [{:keys [transitions struct-weight] :or {struct-weight 2}} opts]
     (pr/loop [i 0, results []]
       (if (>= i n)
         (let [scored (mapv (fn [r]
                              (let [behavioral (when (and (:valid? r) transitions)
                                                (verify-transition-fn (:code r) transitions))
                                    accuracy (when behavioral (:accuracy behavioral))
                                    combined (+ (:weight r)
                                                (* struct-weight (:struct-score r))
                                                (if (= 1 accuracy) 100 0))]
                                (assoc r
                                       :accuracy accuracy
                                       :failures (:failures behavioral)
                                       :combined combined)))
                            results)]
           (vec (sort-by :combined > scored)))
         (pr/let [r (generate-and-score model-map gf prompt opts)]
           (pr/recur (inc i) (conj results r))))))))
