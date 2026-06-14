(ns genmlx.llm.structured
  "Schema-typed structured generation: an LLM-as-GF whose output conforms to a
   malli schema (genmlx-xi71, the Phase-2 flagship).

   A malli schema is the structured-output contract. `genmlx.llm.schema-grammar`
   compiles it to a regex over canonical EDN; this namespace wires that regex
   into the existing byte-level constrained decoder (`genmlx.llm.bytes` +
   `genmlx.llm.grammar`) so the model emits ONLY conforming data. Because the
   result is an ordinary generative function over byte trace sites, the full GFI
   applies and it composes (via splice) inside larger gen fns.

   Public surface (all take pre-encoded prompt-ids; sync — see `encode-prompt`):
     (gen-structured model-map schema opts) -> DynamicGF over [prompt-ids max-bytes]
     (sample   model-map schema prompt-ids opts)        -> {:value :text :trace}
     (score    model-map schema prompt-ids value opts)  -> {:logp :text :conforms?}
     (generate model-map schema prompt-ids partial opts)-> {:value :weight :text ...}

   Scoring contract (documents genmlx-h2ki): `:logp` is the log-density of the
   value under the SCHEMA-CONSTRAINED byte GF — i.e. the sum over bytes of the
   per-step categorical re-normalized over the grammar-alive bytes (a greedy
   byte-tokenization conditional), NOT the raw model evidence p_LM(text). It is
   the correct importance weight for THIS GF and is comparable across values of
   the same schema; it is not a model-marginal likelihood."
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.bytes :as bytes]
            [genmlx.llm.grammar :as grammar]
            [genmlx.llm.schema-grammar :as sg]
            [promesa.core :as pr]))

;; ============================================================
;; Per-step constrained byte log-probabilities (pure, given logprobs)
;; ============================================================

(defn- logsumexp-vals
  "logsumexp over a seq of JS numbers. -Inf for empty."
  [xs]
  (let [m (reduce (fn [a b] (if (> b a) b a)) js/Number.NEGATIVE_INFINITY xs)]
    (if (= m js/Number.NEGATIVE_INFINITY)
      js/Number.NEGATIVE_INFINITY
      (+ m (js/Math.log (reduce (fn [s x] (+ s (js/Math.exp (- x m)))) 0.0 xs))))))

(defn- constrained-lp
  "Constrained categorical log-prob of choosing `ch` from the alive-set, i.e.
   raw[ch] - logsumexp(raw over alive). Matches what dist/categorical assigns
   the masked byte distribution in the handler path."
  [valid-lps ch]
  (- (get valid-lps ch) (logsumexp-vals (vals valid-lps))))

(defn- prepared-trie [model-map opts]
  (or (:trie opts) (:trie (bytes/prepare (:tokenizer model-map)))))

;; ============================================================
;; The GF
;; ============================================================

(defn gen-structured
  "A generative function constraining an LLM to emit a value conforming to
   `schema`. Returns a DynamicGF over [prompt-ids max-bytes] with byte trace
   sites :b0,:b1,... (the same shape as bytes/constrain-bytes). Reuse a shared
   trie via (assoc opts :trie (:trie (bytes/prepare tokenizer)))."
  ([model-map schema] (gen-structured model-map schema {}))
  ([model-map schema opts]
   (bytes/constrain-bytes model-map
                          (grammar/compile-regex (sg/schema->regex schema))
                          (merge {:commit-eager? true} opts))))

(defn decode-value
  "Parse the byte retval of a structured trace into a typed value, validated
   against `schema`. Returns {:value :text :ok? :error}."
  [schema trace]
  (let [text (apply str (:retval trace))]
    (assoc (sg/parse-and-validate schema text) :text text)))

;; ============================================================
;; sample (simulate + parse)
;; ============================================================

(defn sample
  "Generate one schema-conforming value. Returns {:value :text :trace :ok?}.
   prompt-ids: pre-encoded vector of int token ids. opts: :max-bytes (256),
   :trie (shared)."
  ([model-map schema prompt-ids] (sample model-map schema prompt-ids {}))
  ([model-map schema prompt-ids opts]
   (let [max-bytes (:max-bytes opts 256)
         gf (gen-structured model-map schema opts)
         tr (p/simulate gf [prompt-ids max-bytes])]
     (merge {:trace tr} (decode-value schema tr)))))

;; ============================================================
;; score (assess: constrained log-density of a given value)
;; ============================================================

(defn score
  "Teacher-force the canonical serialization of `value` through the
   schema-constrained byte distribution and return its log-density.
   Returns {:logp :text :conforms?}. :logp is ##-Inf if `value` is off-grammar.
   See the namespace docstring for the scoring contract (genmlx-h2ki)."
  ([model-map schema prompt-ids value] (score model-map schema prompt-ids value {}))
  ([model-map schema prompt-ids value opts]
   (let [{:keys [model]} model-map
         trie   (prepared-trie model-map opts)
         target (sg/schema->canonical-str schema value)
         dfa    (grammar/compile-regex (sg/schema->regex schema))
         tlen   (count target)]
     (bytes/with-byte-cache model
       (fn []
         (loop [i 0
                trie-pos trie
                dfa-state (:start dfa)
                logprobs (bytes/logits->logprobs (llm/forward-prefill model prompt-ids))
                logp 0.0]
           (if (>= i tlen)
             {:logp logp :text target :conforms? (contains? (:accept dfa) dfa-state)}
             (let [c (subs target i (inc i))
                   [tp* lp*] (bytes/resolve-replay-step model trie trie-pos logprobs c)
                   valid (bytes/alive-byte-lps tp* lp* dfa dfa-state)]
               (if-not (contains? valid c)
                 {:logp js/Number.NEGATIVE_INFINITY :text target
                  :conforms? false :error :off-grammar :stuck-at i}
                 (let [lp (constrained-lp valid c)
                       next-dfa (grammar/dfa-advance dfa dfa-state c)
                       [next-pos next-lps] (bytes/trie-advance model trie tp* lp* c true)]
                   (recur (inc i) next-pos next-dfa next-lps (+ logp lp))))))))))))

;; ============================================================
;; generate (condition on partial structure; importance weight)
;; ============================================================

(defn- dual-density
  "Replay `text` once through both `base-dfa` and `cond-dfa`, returning
   {:base lp :cond lp} — the log-densities of text under each constrained GF.
   One forward pass; both DFAs advance in CLJS over the shared byte stream."
  [model trie prompt-ids text base-dfa cond-dfa]
  (let [tlen (count text)]
    (bytes/with-byte-cache model
      (fn []
        (loop [i 0
               trie-pos trie
               b-state (:start base-dfa)
               c-state (:start cond-dfa)
               logprobs (bytes/logits->logprobs (llm/forward-prefill model prompt-ids))
               base 0.0 cond* 0.0]
          (if (>= i tlen)
            {:base base :cond cond*}
            (let [c       (subs text i (inc i))
                  [tp* lp*] (bytes/resolve-replay-step model trie trie-pos logprobs c)
                  b-valid (bytes/alive-byte-lps tp* lp* base-dfa b-state)
                  c-valid (bytes/alive-byte-lps tp* lp* cond-dfa c-state)
                  base'   (if (contains? b-valid c) (+ base (constrained-lp b-valid c)) js/Number.NEGATIVE_INFINITY)
                  cond*'  (if (contains? c-valid c) (+ cond* (constrained-lp c-valid c)) js/Number.NEGATIVE_INFINITY)
                  [next-pos next-lps] (bytes/trie-advance model trie tp* lp* c true)]
              (recur (inc i) next-pos
                     (grammar/dfa-advance base-dfa b-state c)
                     (grammar/dfa-advance cond-dfa c-state c)
                     next-lps base' cond*'))))))))

(defn generate
  "Condition the structured GF on a partial value: fix the fields given in
   `partial` (a map keyed like the schema), sample the rest from the model
   constrained to the schema, and return the importance weight.

   Because the model is autoregressive and the fixed-field bytes are forced in
   sequence, the sampled value is an EXACT sample from p(rest | fixed, prompt,
   schema), and the weight is the log-evidence of the fixed fields:
     weight = base-density(text) - cond-density(text) = Σ_fixed-bytes base-clp.

   Returns {:value :weight :text :trace :base-logp :cond-logp :ok?}.
   prompt-ids: pre-encoded. opts: :max-bytes (256), :trie (shared)."
  ([model-map schema prompt-ids partial] (generate model-map schema prompt-ids partial {}))
  ([model-map schema prompt-ids partial opts]
   (let [{:keys [model]} model-map
         trie        (prepared-trie model-map opts)
         opts*       (assoc opts :trie trie)
         cond-schema (sg/constrain-schema schema partial)
         base-dfa    (grammar/compile-regex (sg/schema->regex schema))
         cond-dfa    (grammar/compile-regex (sg/schema->regex cond-schema))
         max-bytes   (:max-bytes opts 256)
         tr          (p/simulate (gen-structured model-map cond-schema opts*) [prompt-ids max-bytes])
         text        (apply str (:retval tr))
         {:keys [base cond]} (dual-density model trie prompt-ids text base-dfa cond-dfa)
         parsed      (sg/parse-and-validate schema text)]
     (merge {:weight (- base cond) :base-logp base :cond-logp cond :text text :trace tr}
            parsed))))

;; ============================================================
;; Async convenience
;; ============================================================

(defn encode-prompt
  "Encode a prompt string to int token ids (promise). Wraps the model's chat
   template when :chat? true (default false: raw encoding)."
  ([model-map prompt] (encode-prompt model-map prompt {}))
  ([{:keys [tokenizer]} prompt {:keys [chat? add-special?] :or {add-special? false}}]
   (pr/let [ids (llm/encode tokenizer prompt add-special?)]
     (vec ids))))
