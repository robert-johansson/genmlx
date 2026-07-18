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
                {:name \"unit\" :pattern \"(celsius|fahrenheit)\"}
                {:name \"pred\" :cljs true}]}]              ; reader-level value
   A param may carry :pattern — a grammar.cljs regex for its VALUE (the
   schema-aware argument constraint; genmlx-4tcd stretch) — or :cljs true —
   the READER-LEVEL constraint (genmlx-3g0t): the value must be exactly one
   complete, delimiter-opened ClojureScript form, enforced byte-granularly
   by edamame through the hybrid masker (see hybrid-masker below). Params
   without either accept any <-free text. Function names and parameter
   names are grammar-bound to the DECLARED list — an undeclared name cannot
   be sampled. Values containing '<' are unrepresentable by design (the
   dialect's own closing-tag ambiguity) — EXCEPT inside a :cljs value,
   where the reader governs and '<' (comparisons!) is representable; the
   closing tag stays unambiguous because the masker forbids the literal
   closing sequence inside the value and force-emits it on completion.

   Restrictions inherited from grammar.cljs: printable-ASCII alphabet
   (genmlx-6dax); param repetition is allowed by the grammar (the star over
   the param alternation) — callers who need each-param-once semantics
   check the PARSE (parse-tool-calls), which reports duplicates."
  (:require [clojure.string :as str]
            [genmlx.codegen.eval :as ceval]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.schema-grammar :as sg]
            [genmlx.mlx :as mx]))

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
   else any <-free text. A :cljs param deliberately takes the permissive
   any-text slot: the DFA models only the ENVELOPE there (a single
   self-loop state the hybrid masker parks in while the reader governs the
   value bytes — see hybrid-masker)."
  [{:keys [name pattern schema cljs]}]
  (str "<parameter=" (sg/re-quote name) ">\\n"
       (or (when-not cljs pattern)
           (when (and schema (not cljs)) (sg/schema->regex schema))
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

;; ---------------------------------------------------------------------------
;; Reader-level (:cljs) argument support — genmlx-3g0t
;; ---------------------------------------------------------------------------
;;
;; A :cljs param's VALUE is exactly one complete, delimiter-opened ('(' '['
;; '{') ClojureScript form, classified by ceval/cljs-arg-status (edamame
;; parse-string-all — a second form or trailing junk is :invalid, so
;; one-argument-one-form is enforced at sampling time, not post-hoc).
;;
;; The hybrid masker owns three regions:
;;   :envelope — the qwen3_xml dialect, masked by the DFA as before.
;;   :value    — BYTE-GRANULAR reader masking: only single-printable-byte
;;               tokens whose appended char keeps the value a valid
;;               exactly-one-form prefix. Byte granularity is what makes the
;;               reader (unbounded state, no DFA-style prefix merging)
;;               tractable per step: ~98 short edamame parses, vs testing
;;               the whole vocab. The cost is one forward per byte inside
;;               the value (~form-length extra forwards per call).
;;   :closing  — the form completed: the ONLY continuation is the literal
;;               "\n</parameter>\n", offered as any token that is a prefix
;;               of the remainder (multi-byte closing tokens allowed).
;; The DFA is parked in the value's self-loop state during :value (value
;; chars are never fed to it — they may contain '<') and advances over the
;; envelope + closing text, so envelope tracking stays exact.

(def ^:private cljs-closing "\n</parameter>\n")

(defn- cljs-support
  "Build the reader-leg lookup tables when some param declares :cljs true;
   nil otherwise. Throws when a param name is :cljs in one tool and plain
   in another — the masker keys the reader region on the name alone."
  [tools token-index]
  (let [cljs-names (set (for [t tools, p (:params t) :when (:cljs p)] (:name p)))]
    (when (seq cljs-names)
      (let [plain-names (set (for [t tools, p (:params t) :when (not (:cljs p))] (:name p)))
            clash (seq (filter plain-names cljs-names))]
        (when clash
          (throw (ex-info (str "toolcall: parameter name(s) " (pr-str (vec clash))
                               " declared :cljs in one tool and plain in another"
                               " — the hybrid masker keys the reader region on"
                               " the parameter name alone; rename one.")
                          {:genmlx/error :cljs-param-name-clash :names (vec clash)}))))
      (let [n-close     (count cljs-closing)
            byte-ids    (js/Map.)
            closing-ids (vec (repeatedly n-close #(array)))]
        (dotimes [i (count token-index)]
          (let [t (nth token-index i)]
            (when (seq t)
              (when (and (= 1 (count t))
                         (let [cc (.charCodeAt t 0)]
                           (or (<= 32 cc 126) (= cc 10) (= cc 9))))
                (when-not (.has byte-ids t) (.set byte-ids t (array)))
                (.push (.get byte-ids t) i))
              (dotimes [k n-close]
                (when (and (<= (count t) (- n-close k))
                           (= t (subs cljs-closing k (+ k (count t)))))
                  (.push (nth closing-ids k) i))))))
        {:names       cljs-names
         :openers     (mapv #(str "<parameter=" % ">\n") cljs-names)
         :byte-ids    byte-ids
         :closing     cljs-closing
         :closing-ids closing-ids}))))

(def ^:private value-open-chars ["(" "[" "{"])

(defn cljs-value-chars
  "The allowed next BYTES for a :cljs value prefix `v` (exposed for tests).
   Empty prefix admits only the delimiter openers; afterwards any printable
   byte (incl. \\n \\t) that keeps cljs-arg-status non-:invalid, minus any
   byte completing the literal closing sequence inside the value."
  [v]
  (if (str/blank? v)
    value-open-chars
    (into []
          (keep (fn [cc]
                  (let [c (js/String.fromCharCode cc)
                        v' (str v c)]
                    (when (and (not= :invalid (ceval/cljs-arg-status v'))
                               (not (str/ends-with? v' "\n</parameter>")))
                      c))))
          (concat (range 32 127) [10 9]))))

(def ^:private printable-value-char?
  (let [ok (set (into (mapv #(js/String.fromCharCode %) (range 32 127)) ["\n" "\t"]))]
    (fn [c] (contains? ok c))))

(def ^:private value-top-k 64)

(defn- admissible-token-text?
  "May token text `t` extend the :cljs value `v`? Char-wise walk (matching
   hybrid-advance's consumption): every intermediate prefix must stay
   non-:invalid, completion may occur only at the FINAL char (no crossing),
   and the closing sequence must never appear."
  [v t blank?]
  (let [n (count t)]
    (and (pos? n)
         (every? printable-value-char? t)
         (or (not blank?) (contains? #{"(" "[" "{"} (subs t 0 1)))
         (loop [j 0 v' v]
           (if (>= j n)
             true
             (let [v'' (str v' (subs t j (inc j)))
                   s (ceval/cljs-arg-status v'')]
               (cond
                 (= :invalid s) false
                 (str/ends-with? v'' "\n</parameter>") false
                 (and (= :complete s) (< (inc j) n)) false
                 :else (recur (inc j) v''))))))))

(defn cljs-value-token-ids
  "Token ids admissible as the next emission in a :cljs value `v`: the
   SOUND FLOOR of every reader-valid single printable byte (progress can
   never dead-end and the tail of the distribution stays representable at
   byte granularity) plus the logit-guided top-K multi-byte tokens whose
   full text is admissible (admissible-token-text?). The multi-byte offers
   are what keep emission on natural BPE tokenization — a pure single-byte
   mask samples near-noise; testing the whole vocab against the reader
   (~84us/parse) is intractable, so candidates come from the model's own
   top-K (`cand-ids`).

   ENGINE-SPECIAL EXCLUSION (the eos-as-symbol bug): special tokens decode
   to printable text (`<|im_end|>`, `<think>`, …) that the READER happily
   accepts as symbol characters — without this guard the top-K path admits
   the EOS ID ITSELF mid-form (the engine then ends the turn with the block
   unclosed, observed live) and the think markers (which would flip think
   mode and bypass the mask entirely). Multi-byte candidates equal to
   eos-id or containing '<' AT ALL are rejected — every angle-special is
   covered, and '<' itself (comparisons!) stays expressible through the
   single-byte floor, whose plain char token carries no engine semantics."
  [{:keys [byte-ids]} token-index v cand-ids eos-id]
  (let [out (array)
        seen (js/Set.)
        blank? (str/blank? v)]
    (doseq [c (cljs-value-chars v)]
      (when-let [ids (.get byte-ids c)]
        (dotimes [j (.-length ids)]
          (let [i (aget ids j)]
            (when-not (.has seen i) (.add seen i) (.push out i))))))
    (doseq [i cand-ids]
      (when-not (.has seen i)
        (let [t (nth token-index i nil)]
          (when (and t (> (count t) 1)
                     (not= i eos-id)
                     (not (str/includes? t "<"))
                     (admissible-token-text? v t blank?))
            (.add seen i)
            (.push out i)))))
    out))

(defn- hybrid-advance
  "Consume newly visible token texts: value chars accumulate on :value,
   envelope + closing chars accumulate for one DFA advance. Returns the new
   tracker state. Throws typed errors on masker-invariant violations (a
   char the mask should have made unsampleable)."
  [{:keys [mode tail value remaining state] :as st} dfa openers texts]
  (let [max-open (reduce max 0 (map count openers))
        sb (js/Array.)]                        ; env/closing chars, in order
    (loop [ts texts mode mode tail tail value value remaining remaining]
      (if (empty? ts)
        (let [env (.join sb "")
              state' (if (seq env) (gram/dfa-advance-string dfa state env) state)]
          (assoc st :mode mode :tail tail :value value :remaining remaining
                 :state state'))
        (let [txt (first ts)
              n (count txt)
              [mode tail value remaining]
              (loop [j 0 mode mode tail tail value value remaining remaining]
                (if (>= j n)
                  [mode tail value remaining]
                  (let [c (subs txt j (inc j))]
                    (case mode
                      :envelope
                      (let [tail' (let [t (str tail c)]
                                    (if (> (count t) max-open)
                                      (subs t (- (count t) max-open)) t))]
                        (.push sb c)
                        (if (some #(str/ends-with? tail' %) openers)
                          (recur (inc j) :value "" "" remaining)
                          (recur (inc j) :envelope tail' value remaining)))
                      :value
                      (let [v' (str value c)
                            s (ceval/cljs-arg-status v')]
                        (when (= :invalid s)
                          (throw (ex-info "toolcall hybrid: sampled byte made the :cljs value invalid — masker invariant broken"
                                          {:genmlx/error :cljs-masker-invariant
                                           :value v'})))
                        (if (= :complete s)
                          (recur (inc j) :closing tail v' cljs-closing)
                          (recur (inc j) :value tail v' remaining)))
                      :closing
                      (do (when-not (= c (subs remaining 0 1))
                            (throw (ex-info "toolcall hybrid: sampled char diverged from the forced closing sequence"
                                            {:genmlx/error :cljs-masker-invariant
                                             :char c :remaining remaining})))
                          (.push sb c)
                          (let [r' (subs remaining 1)]
                            (if (seq r')
                              (recur (inc j) :closing tail value r')
                              (recur (inc j) :envelope "" nil nil))))))))]
          (recur (rest ts) mode tail value remaining))))))

(def ^:private neg-inf (- js/Infinity))

(defn- ids-mask-logits
  "Mask logits to exactly `id-arrays` (JS arrays of token ids): -inf
   everywhere else, EOS included (a :cljs value/closing region can never
   end the turn)."
  [logits id-arrays]
  (let [logits-n (last (mx/shape logits))
        buf (js/Float32Array. logits-n)]
    (.fill buf neg-inf)
    (doseq [ids id-arrays]
      (dotimes [j (.-length ids)]
        (let [i (aget ids j)]
          (when (< i logits-n) (aset buf i 0.0)))))
    (mx/add logits (.fromFloat32 mx/core buf #js [logits-n]))))

(defn- opener-progress?
  "True when `tail` ends inside a (possibly-complete) prefix of some opener
   — the only region where a DFA-legal token could straddle into the value."
  [tail openers]
  (boolean
   (some (fn [o]
           (some #(str/ends-with? tail (subs o 0 %))
                 (range 1 (inc (min (count tail) (count o))))))
         openers)))

(defn- straddle-filtered-mask
  "Copy of the DFA mask `src` with tokens masked whose text would cross a
   :cljs opener boundary INTO value content — value bytes must come from
   reader-masked steps, never ride an envelope token."
  [constraint src tail openers]
  (let [buf (js/Float32Array. src)
        token-index (:token-index constraint)]
    (dotimes [i (.-length buf)]
      (when (zero? (aget buf i))
        (let [comb (str tail (nth token-index i ""))]
          (when (some (fn [o]
                        (let [idx (.indexOf comb o)]
                          (and (>= idx 0) (< (+ idx (count o)) (count comb)))))
                      openers)
            (aset buf i neg-inf)))))
    buf))

(defn hybrid-masker
  "Per-turn logits masker for a constraint carrying :cljs-support — the
   genmlx-3g0t reader leg. Same (fn [logits vis] -> masked-logits) shape as
   the plain DFA masker; :envelope behaves identically to it (including the
   small state->mask memo), :value/:closing come from the reader tables."
  [constraint]
  (let [{:keys [dfa token-index cljs-support]} constraint
        {:keys [openers closing closing-ids]} cljs-support
        st* (volatile! {:mode :envelope :tail "" :value nil :remaining nil
                        :state (:start dfa) :seen 0 :memo {}})]
    (fn [logits vis]
      (let [{:keys [seen memo] :as st} @st*
            texts (into [] (keep #(let [t (nth token-index % nil)]
                                    (when (seq t) t)))
                        (subvec vis seen))
            st' (assoc (hybrid-advance st dfa openers texts) :seen (count vis))]
        (case (:mode st')
          :envelope
          (let [state (:state st')
                memo' (if (or (contains? memo state) (>= (count memo) 64))
                        memo
                        (assoc memo state (gram/get-mask constraint state)))
                base  (or (get memo' state) (gram/get-mask constraint state))]
            (vreset! st* (assoc st' :memo memo'))
            (if (opener-progress? (:tail st') openers)
              (let [buf (straddle-filtered-mask constraint base (:tail st') openers)]
                (gram/apply-mask (assoc constraint :masks {state buf}) state logits))
              (gram/apply-mask (assoc constraint :masks {state base}) state logits)))
          :value
          (let [flat (let [sh (mx/shape logits)]
                       (if (> (count sh) 1)
                         (mx/reshape logits [(last sh)])
                         logits))
                cand (mx/->clj (mx/slice (mx/argsort (mx/negative flat))
                                         0 value-top-k))
                ids (cljs-value-token-ids cljs-support token-index
                                          (:value st') cand
                                          (:eos-id constraint))]
            (vreset! st* (assoc st' :memo memo))
            (when (zero? (.-length ids))
              (throw (ex-info "toolcall hybrid: no valid continuation for the :cljs value — reader dead-end"
                              {:genmlx/error :cljs-masker-dead-end
                               :value (:value st')})))
            (ids-mask-logits logits [ids]))
          :closing
          (let [k (- (count closing) (count (:remaining st')))]
            (vreset! st* (assoc st' :memo memo))
            (ids-mask-logits logits [(nth closing-ids k)])))))))

(defn compile-toolcall
  "Compile the tool-call constraint for `tools` over `tokenizer` — a
   grammar.cljs constraint map usable by BOTH wrap-grammar/constrain
   (scalar) and build-vtables/vectorized-hook (batched). opts passes
   through to compile-constraint (:token-index reuse, :max-precompute)
   plus :require-call? for the regex. When some param declares :cljs true,
   the result additionally carries :cljs-support — consume it through
   hybrid-masker (the plain DFA mask alone would allow any <-free text in
   that value slot)."
  ([tokenizer tools] (compile-toolcall tokenizer tools {}))
  ([tokenizer tools opts]
   (let [c (gram/compile-constraint tokenizer (tool-call-regex tools opts) opts)]
     (if-let [cs (cljs-support tools (:token-index c))]
       (assoc c :cljs-support cs)
       c))))

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
