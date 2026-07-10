(ns genmlx.llm.toolcall
  "Grammar-constrained qwen3_xml tool calls (genmlx-4tcd).

   Ornith/Qwen3.6's agentic dialect (vLLM --tool-call-parser qwen3_xml):

     <tool_call>
     <function=get_weather>
     <parameter=location>
     Paris
     </parameter>
     </function>
     </tool_call>

   vLLM PARSES this after the fact and retries on malformed output. We do
   strictly better: the dialect becomes a regex over the declared tool list,
   compiled by the existing grammar stack (grammar.cljs regex→DFA→token
   masks) so malformed tool calls are UNREPRESENTABLE at sampling time.

   Contextual activation lives IN the regex, not in bespoke middleware:
   the top-level language is  prose (block prose)*  — free text runs in a
   self-loop state where (almost) every token is valid, and the single
   opening literal '<' commits the DFA to a well-formed block. This means
   the constraint composes unchanged with BOTH constraint layers:
     - scalar GFI ops via grammar/constrain (wrap-grammar middleware)
     - [K]-lane batched sampling via grammar/build-vtables +
       grammar/vectorized-hook (make-llm-gf-batched :hook) — K constrained
       tool-call candidates in ONE forward (genmlx-9uyg)
   The <tool_call>/</tool_call> SPECIAL tokens (single ids in the Qwen
   vocab) advance the DFA through their full literal text exactly like a
   spelled-out sequence, so both tokenizations stay on-grammar.

   Tool declarations:
     [{:name \"get_weather\"
       :params [{:name \"location\"}                        ; free-text value
                {:name \"unit\" :pattern \"(celsius|fahrenheit)\"}]}]
   A param may carry :pattern — a grammar.cljs regex for its VALUE (the
   schema-aware argument constraint; genmlx-4tcd stretch). Params without
   one accept any <-free text. Function names and parameter names are
   grammar-bound to the DECLARED list — an undeclared name cannot be
   sampled. Values containing '<' are unrepresentable by design (the
   dialect's own closing-tag ambiguity).

   Restrictions inherited from grammar.cljs: printable-ASCII alphabet
   (genmlx-6dax); param repetition is allowed by the grammar (the star over
   the param alternation) — callers who need each-param-once semantics
   check the PARSE (parse-tool-calls), which reports duplicates."
  (:require [clojure.string :as str]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.schema-grammar :as sg]))

;; ---------------------------------------------------------------------------
;; Regex construction
;; ---------------------------------------------------------------------------

(def ^:private prose
  "Free-text run: any printable-ASCII char except '<', or newline. The '<'
   exclusion is what arms the contextual switch — the only way to emit '<'
   is to enter (and therefore complete) a well-formed tool-call block."
  "([^<]|\\n)*")

(defn- param-regex
  "One  <parameter=NAME>\\nVALUE\\n</parameter>\\n  unit. The VALUE slot is,
   in priority order: :pattern (a grammar.cljs regex), :schema (a malli
   schema compiled by schema-grammar — the value is its canonical EDN
   serialization, e.g. :int → -?digits, [:enum ...] → the alternation),
   else any <-free text."
  [{:keys [name pattern schema]}]
  (str "<parameter=" (sg/re-quote name) ">\\n"
       (or pattern
           (when schema (sg/schema->regex schema))
           "([^<]|\\n)*")
       "\\n</parameter>\\n"))

(defn- function-body-regex
  "NAME>\\n(param|param|…)*  — parameter names bound to THIS tool.
   `max-params` (opt) bounds the repetition: a hard cap on block length,
   useful when sampling under a token budget (an unbounded star lets a
   hot-temperature model add parameters until max-tokens truncates the
   block mid-tag — a sampling artifact, not a grammar failure)."
  [{:keys [name params]} max-params]
  (str (sg/re-quote name) ">\\n"
       (when (seq params)
         (str "(" (str/join "|" (map param-regex params)) ")"
              (if max-params (str "{0," max-params "}") "*")))))

(defn block-regex
  "One complete tool-call block over the declared `tools`.
   opts {:max-params n} bounds per-call parameter count."
  ([tools] (block-regex tools {}))
  ([tools {:keys [max-params]}]
   (str "<tool_call>\\n<function="
        "(" (str/join "|" (map #(function-body-regex % max-params) tools)) ")"
        "</function>\\n</tool_call>")))

(defn tool-call-regex
  "The full constrained language:  prose (block prose)*  — contextual
   activation in the regex itself. opts:
     :require-call? — at least one block (eos masked until a call completes)
     :block-first?  — the block must open IMMEDIATELY (no leading prose);
                      implies require-call. The right shape under tight
                      token budgets / adversarial temperatures.
     :max-calls     — cap the number of blocks (1 = exactly-one-call with
                      require/block-first; bounds worst-case sample length)
     :max-params    — per-call parameter-count bound (see block-regex)."
  ([tools] (tool-call-regex tools {}))
  ([tools {:keys [require-call? block-first? max-calls] :as opts}]
   (let [b (block-regex tools opts)
         more (fn [n-used]
                (cond
                  (nil? max-calls) (str "(" b prose ")*")
                  (> max-calls n-used) (str "(" b prose "){0," (- max-calls n-used) "}")
                  :else ""))]
     (cond
       block-first?  (str b prose (more 1))
       require-call? (str prose b prose (more 1))
       :else         (str prose (more 0))))))

(defn compile-toolcall
  "Compile the tool-call constraint for `tools` over `tokenizer` — a
   grammar.cljs constraint map usable by BOTH wrap-grammar/constrain
   (scalar) and build-vtables/vectorized-hook (batched). opts passes
   through to compile-constraint (:token-index reuse, :max-precompute)
   plus :require-call? for the regex."
  ([tokenizer tools] (compile-toolcall tokenizer tools {}))
  ([tokenizer tools opts]
   (gram/compile-constraint tokenizer (tool-call-regex tools opts) opts)))

;; ---------------------------------------------------------------------------
;; Independent parse oracle (string scanning — deliberately NOT the DFA, so
;; tests verify the grammar against a second implementation of the dialect)
;; ---------------------------------------------------------------------------

(defn- parse-block
  "Parse the inside of one block (text between <tool_call>\\n and
   </tool_call>). Returns {:name .. :args {pname value}} or {:error ..}."
  [inside]
  (let [m (re-matches #"<function=([A-Za-z0-9_.-]+)>\n([\s\S]*)</function>\n" inside)]
    (if-not m
      {:error (str "malformed function envelope: " (pr-str inside))}
      (let [[_ fname body] m]
        (loop [rest' body args {} order []]
          (if (str/blank? rest')
            {:name fname :args args :param-order order}
            (let [pm (re-find #"^<parameter=([A-Za-z0-9_.-]+)>\n([\s\S]*?)\n</parameter>\n" rest')]
              (if-not pm
                {:error (str "malformed parameter at: " (pr-str rest'))}
                (let [[whole pname pval] pm]
                  (recur (subs rest' (count whole))
                         (assoc args pname pval)
                         (conj order pname)))))))))))

(defn parse-tool-calls
  "Scan `text` for <tool_call>…</tool_call> blocks and parse each against
   the qwen3_xml dialect. Returns {:calls [{:name :args :param-order}]
   :errors [..]}; a call whose function or parameter names are not in
   `tools` (when given) contributes an error. This is the verification
   oracle for the grammar (an independent scanner, not the DFA)."
  ([text] (parse-tool-calls text nil))
  ([text tools]
   (let [declared (when tools
                    (into {} (map (fn [t] [(:name t) (set (map :name (:params t)))])
                                  tools)))]
     (loop [s text calls [] errors []]
       (let [open (.indexOf s "<tool_call>\n")]
         (if (neg? open)
           {:calls calls
            :errors (cond-> errors
                      (>= (.indexOf s "<tool_call>") 0)
                      (conj "unterminated/misformatted <tool_call> opener"))}
           (let [after (subs s (+ open 12))
                 close (.indexOf after "</tool_call>")]
             (if (neg? close)
               {:calls calls :errors (conj errors "unclosed <tool_call>")}
               (let [inside (subs after 0 close)
                     parsed (parse-block inside)
                     errs (cond-> []
                            (:error parsed) (conj (:error parsed))
                            (and declared (:name parsed)
                                 (not (contains? declared (:name parsed))))
                            (conj (str "undeclared function " (:name parsed)))
                            (and declared (:name parsed)
                                 (contains? declared (:name parsed))
                                 (not (every? (get declared (:name parsed))
                                              (keys (:args parsed)))))
                            (conj (str "undeclared parameter(s) for " (:name parsed))))]
                 (recur (subs after (+ close 12))
                        (if (:error parsed) calls (conj calls (dissoc parsed :error)))
                        (into errors errs)))))))))))
