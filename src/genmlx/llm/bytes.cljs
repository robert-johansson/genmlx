(ns genmlx.llm.bytes
  "Byte-level LLM generation via token marginalization.

   Standard BPE tokenizers operate on multi-byte tokens, but many applications
   (grammar constraints, character-level control, byte-level scoring) need
   per-byte distributions. This namespace bridges the gap: given token-level
   log-probabilities from an LLM, it marginalizes over the token vocabulary
   to produce per-byte categorical distributions.

   Architecture:
     token-index -> byte trie (precomputed, once per tokenizer)
     trie-node + token-logprobs -> byte marginals (logsumexp gather)
     byte marginals -> indexed categorical -> standard handler transition

   Each trace site :b0, :b1, ... samples one byte. The gen body tracks trie
   position and token boundaries in its own loop (same pattern as core.cljs).
   When a trie leaf is reached, the accumulated bytes form a complete token,
   the KV cache advances by one token step, and fresh token logprobs are
   fetched for the next byte marginal.

   Greedy tokenization: when a trie node has both token-ids AND children,
   always continue (extend to a longer token). Only commit a token when
   forced (leaf node with no children). This matches BPE longest-match.

   LIMITATION — grammar-constrained paths are printable-ASCII only
   (genmlx-6dax): constrain-bytes filters byte marginals through DFAs from
   genmlx.llm.grammar, whose wildcard alphabet is ASCII 32–126. Regex
   wildcards/negated classes never admit non-ASCII bytes or newlines, so
   DFA-constrained generation/scoring of non-ASCII content is off-grammar
   (unconstrained byte generation via make-byte-llm-gf is not affected)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.grammar :as grammar])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================
;; TokenByteTrie
;; ============================================================

(defn- trie-insert
  "Insert one token into the trie by walking its byte characters.
   Purely functional — returns updated node."
  [node token-id token-str]
  (if (empty? token-str)
    (update node :token-ids (fnil conj #{}) token-id)
    (update-in node [:children (subs token-str 0 1)]
               (fnil trie-insert {:children {} :token-ids #{}})
               token-id (subs token-str 1))))

(defn- collect-all-ids
  "Bottom-up pass: precompute :all-token-ids on every node — the union of
   all token IDs reachable from that subtree, stored both as a set (for
   membership tests) and a sorted Int32Array (for efficient gather)."
  [node]
  (let [enriched-children (into {}
                                (map (fn [[ch child]]
                                       [ch (collect-all-ids child)]))
                                (:children node))
        all-ids (->> (vals enriched-children)
                     (map :all-token-ids)
                     (reduce into (:token-ids node #{})))
        sorted-arr (js/Int32Array. (clj->js (vec (sort all-ids))))]
    (assoc node
           :children enriched-children
           :all-token-ids all-ids
           :all-token-ids-arr sorted-arr)))

(defn build-byte-trie
  "Build a trie mapping byte sequences to token IDs.

   token-index: vector of decoded token strings (from grammar/build-token-index).
   Each token is inserted by walking its characters. Empty/nil tokens are skipped.

   Returns a trie node:
     :children          {single-char-string -> child-node}
     :token-ids         #{int} tokens that terminate exactly here
     :all-token-ids     #{int} all tokens reachable from this subtree
     :all-token-ids-arr Int32Array of sorted reachable IDs (for gather)"
  [token-index]
  (->> token-index
       (map-indexed vector)
       (reduce (fn [trie [id tok]]
                 (if (seq tok)
                   (trie-insert trie id tok)
                   trie))
               {:children {} :token-ids #{}})
       collect-all-ids))

(defn trie-lookup
  "Navigate the trie by a byte string, returning the subtrie at the end
   of the path, or nil if no such path exists."
  [trie byte-str]
  (reduce (fn [node i]
            (if-let [child (get-in node [:children (subs byte-str i (inc i))])]
              child
              (reduced nil)))
          trie
          (range (count byte-str))))

;; ============================================================
;; Byte marginals via logsumexp gather
;; ============================================================

(defn- logsumexp-gather
  "Compute logsumexp over entries of logprobs-f32 at positions given by
   indices-i32. Numerically stable via max-subtraction. Returns a JS number.
   Returns -Infinity for empty index arrays."
  [logprobs-f32 indices-i32]
  (let [n (.-length indices-i32)]
    (if (zero? n)
      js/Number.NEGATIVE_INFINITY
      (let [max-val (loop [i 0, m js/Number.NEGATIVE_INFINITY]
                      (if (< i n)
                        (let [v (aget logprobs-f32 (aget indices-i32 i))]
                          (recur (inc i) (if (> v m) v m)))
                        m))
            sum-exp (loop [i 0, s 0.0]
                      (if (< i n)
                        (recur (inc i)
                               (+ s (js/Math.exp
                                     (- (aget logprobs-f32 (aget indices-i32 i))
                                        max-val))))
                        s))]
        (+ max-val (js/Math.log sum-exp))))))

(defn byte-logprobs
  "Compute byte-level marginal log-probabilities from token logprobs.

   trie-node:      a node in the byte trie (root or deeper)
   token-logprobs: MxArray [vocab-size] of log P(token | context)

   Returns {single-char-string -> log-prob} where each log-prob is the
   logsumexp over all tokens reachable through that byte's child subtree.

   Materializes token-logprobs to Float32Array once (~600KB for 151K vocab),
   then gathers on CPU per child byte. At most 256 children."
  [trie-node token-logprobs]
  (mx/eval! token-logprobs)
  (let [f32 (.toFloat32 token-logprobs)]
    (into {}
          (map (fn [[ch child]]
                 [ch (logsumexp-gather f32 (:all-token-ids-arr child))]))
          (:children trie-node))))

;; ============================================================
;; Byte marginals -> indexed categorical
;; ============================================================

(defn byte-marginals->categorical
  "Convert a byte-marginal map {char -> log-prob} to an indexed categorical.

   Returns {:dist  categorical distribution over [0..n-1]
            :chars vector of byte characters (index -> char mapping)}

   After sampling index i from the categorical, the chosen byte character
   is (nth chars i)."
  [byte-lps]
  (let [{:keys [chars values]}
        (reduce-kv (fn [acc ch lp]
                     (-> acc
                         (update :chars conj ch)
                         (update :values conj lp)))
                   {:chars [] :values []}
                   byte-lps)]
    {:dist (dist/categorical (mx/array values))
     :chars chars}))

;; ============================================================
;; Shared preparation (build once, reuse across GFs)
;; ============================================================

(defn prepare
  "Precompute byte-level structures for a tokenizer.

   Returns {:token-index token-index :trie trie}. Pass to make-byte-llm-gf
   and constrain-bytes via the opts map to avoid rebuilding the trie
   (151K tokens, ~400K nodes) for each GF.

   If not provided, each GF builds its own — correct but redundant."
  [tokenizer]
  (let [token-index (grammar/build-token-index tokenizer)
        trie (build-byte-trie token-index)]
    {:token-index token-index :trie trie}))

;; ============================================================
;; Internal helpers
;; ============================================================

(defn- f64-logsumexp
  "logsumexp over a full logit vector computed in JS float64. MLX's logsumexp is
   float32; over a 248K-element vocab the reduction drifts ~0.04, leaving the
   normalization under-subtracted so the byte marginals summed to ~1.04
   (genmlx-h2ki). Materializing once and reducing in float64 (max-subtracted)
   recovers a sum of 1.0 to float32 precision. Returns a JS number."
  [logits]
  (mx/eval! logits)
  (let [f32 (.toFloat32 logits)
        n (.-length f32)
        m (loop [i 0 mx js/Number.NEGATIVE_INFINITY]
            (if (< i n) (recur (inc i) (let [v (aget f32 i)] (if (> v mx) v mx))) mx))]
    (if (= m js/Number.NEGATIVE_INFINITY)
      js/Number.NEGATIVE_INFINITY
      (let [s (loop [i 0 s 0.0]
                (if (< i n) (recur (inc i) (+ s (js/Math.exp (- (aget f32 i) m)))) s))]
        (+ m (js/Math.log s))))))

(defn logits->logprobs
  "Numerically stable log-softmax: logits - logsumexp(logits), with the
   normalization computed in float64 (f64-logsumexp) so the full-vocab byte
   marginals sum to 1.0 rather than drifting to ~1.04 in float32 (genmlx-h2ki).
   Returns MxArray [vocab-size]."
  [logits]
  (mx/subtract logits (mx/scalar (f64-logsumexp logits))))

(defn trie-leaf?
  "A trie node is a leaf when it has no children — the accumulated bytes
   form a complete token that must be committed."
  [node]
  (empty? (:children node)))

(defn commit-token-id
  "Select the token ID to commit at a leaf node. A leaf's token-ids set
   contains exactly the tokens whose byte sequence ends here. Duplicate-text
   tokens (same bytes, different ids) are resolved to the SMALLEST id: the
   committed id conditions the KV cache, so an arbitrary (first set) pick made
   logits nondeterministic for identical byte choices, breaking
   choices-as-sufficient-statistic (genmlx-xwxh)."
  [node]
  (apply min (:token-ids node)))

(defn with-byte-cache
  "Run thunk under the model's KV-cache lifecycle: initialize the cache,
   run the body, and reset the cache in a finally block. Returns the body's
   value. The thunk closes over the gen body's trace/state."
  [model thunk]
  (llm/init-cache! model)
  (try
    (thunk)
    (finally
      (llm/reset-cache! model))))

(defn trie-advance
  "Advance trie position after choosing chosen-byte at trie-pos. Returns
   [next-trie-pos next-logprobs]. At a leaf the accumulated bytes form a
   complete token: reset to the trie root and commit the token via a cached
   forward step, yielding fresh logprobs. Otherwise descend and keep the
   current logprobs.

   eager? (default false): commit at the FIRST token boundary (any node with
   token-ids), i.e. shortest-match tokenization, instead of greedy longest-match.
   This keeps the trie near the root so any grammar-required byte stays
   reachable — it eliminates the mid-token stranding that longest-match suffers
   when a grammar boundary falls inside a multi-byte token (e.g. an enum literal
   whose bytes span an awkward tokenization). Used by structured generation,
   where validity matters more than matching the model's natural tokenization."
  ([model trie trie-pos logprobs chosen-byte]
   (trie-advance model trie trie-pos logprobs chosen-byte false))
  ([model trie trie-pos logprobs chosen-byte eager?]
   (let [next-node (get-in trie-pos [:children chosen-byte])]
     (if (or (trie-leaf? next-node)
             (and eager? (seq (:token-ids next-node))))
       [trie (logits->logprobs (llm/forward-step model (commit-token-id next-node)))]
       [next-node logprobs]))))

(defn alive-byte-lps
  "Byte marginals at trie-pos filtered to bytes that keep the DFA alive from
   dfa-state. Map {char -> raw-marginal-logprob} (raw, not yet renormalized)."
  [trie-pos logprobs dfa dfa-state]
  (let [alive (:alive dfa)]
    (into {} (filter (fn [[ch _]]
                       (contains? alive (grammar/dfa-advance dfa dfa-state ch))))
          (byte-logprobs trie-pos logprobs))))

(defn resolve-grammar-step
  "Compute the DFA-alive byte marginals at the current position. Greedy BPE
   traversal can strand us mid-token when the grammar requires a byte that is
   not a continuation of the current trie node (e.g. a separator after a
   keyword that is itself a token-with-children). In that case, if the current
   node is a committable token, force-commit it — advance the KV cache, reset to
   the trie root, refetch logprobs — and retry once, so the boundary byte
   becomes reachable. Returns [trie-pos' logprobs' valid-lps]; valid-lps may be
   empty only if generation is genuinely stuck (DFA dead or no committable
   continuation)."
  [model trie trie-pos logprobs dfa dfa-state]
  (let [valid (alive-byte-lps trie-pos logprobs dfa dfa-state)]
    (if (or (seq valid) (empty? (:token-ids trie-pos)))
      [trie-pos logprobs valid]
      (let [lp' (logits->logprobs (llm/forward-step model (commit-token-id trie-pos)))
            valid' (alive-byte-lps trie lp' dfa dfa-state)]
        [trie lp' valid']))))

(defn resolve-replay-step
  "Replay analogue of resolve-grammar-step: ensure the known `target-byte` is
   reachable from trie-pos. If it is not a child but trie-pos is a committable
   token, force-commit (advance the cache, reset to root, refetch logprobs) so
   the byte becomes reachable. Returns [trie-pos' logprobs']. Drives scoring on
   the same cache trajectory as constrained generation."
  [model trie trie-pos logprobs target-byte]
  (if (or (contains? (:children trie-pos) target-byte)
          (empty? (:token-ids trie-pos)))
    [trie-pos logprobs]
    [trie (logits->logprobs (llm/forward-step model (commit-token-id trie-pos)))]))

;; ============================================================
;; Public API: unconstrained byte-level generation
;; ============================================================

(defn make-byte-llm-gf
  "Create a generative function that samples bytes from an LLM.

   model-map: {:model :tokenizer :type} from llm/load-model.
   opts (optional):
     :token-index  pre-built token index (from grammar/build-token-index)
     :trie         pre-built byte trie (from build-byte-trie)
     — or pass the result of (prepare tokenizer) to avoid rebuilding.

   Returns a DynamicGF that takes [prompt-ids max-bytes]:
     prompt-ids  vector of int token IDs (from llm/encode)
     max-bytes   maximum number of bytes to generate

   Each generated byte is a trace site :b0, :b1, ... with a categorical
   distribution over valid next bytes (marginalized from token logprobs).
   The return value is a vector of single-character strings. Use
   (apply str retval) or decode-byte-trace to get the generated text.

   Token boundaries are managed internally via greedy trie traversal.
   The KV cache is initialized at the start, advanced when tokens are
   committed, and reset in a finally block.

   Not safe for concurrent execution on the same model instance."
  ([model-map] (make-byte-llm-gf model-map {}))
  ([model-map opts]
   (let [{:keys [model tokenizer]} model-map
         {:keys [token-index trie]}
         (if (and (:token-index opts) (:trie opts))
           opts
           (prepare tokenizer))]
     (dyn/auto-key
      (gen [prompt-ids max-bytes]
           (if (zero? max-bytes)
             []
             (with-byte-cache model
               (fn []
                 (loop [i 0
                        trie-pos trie
                        logprobs (logits->logprobs
                                  (llm/forward-prefill model prompt-ids))
                        bytes-acc []]
                   (if (>= i max-bytes)
                     bytes-acc
                     (let [byte-lps (byte-logprobs trie-pos logprobs)
                           {:keys [dist chars]}
                           (byte-marginals->categorical byte-lps)

                           idx (trace (keyword (str "b" i)) dist)
                           chosen-byte (nth chars (mx/item idx))
                           [next-pos next-logprobs]
                           (trie-advance model trie trie-pos logprobs chosen-byte)]

                       (recur (inc i) next-pos next-logprobs
                              (conj bytes-acc chosen-byte)))))))))))))

;; ============================================================
;; Byte trace decoding
;; ============================================================

(defn decode-byte-trace
  "Extract generated text from a byte-level trace.

   The gen body returns a vector of single-character strings. This function
   joins them into a single string.

   trace: a Trace from simulate/generate on a byte-level GF."
  [trace]
  (apply str (:retval trace)))

;; ============================================================
;; Grammar-constrained byte generation
;; ============================================================

(defn constrain-bytes
  "Create a grammar-constrained byte-level generative function.

   model-map:  {:model :tokenizer :type} from llm/load-model.
   constraint: either a compiled DFA (from grammar/compile-regex, with :alive)
               or a regex string (compiled internally).
   opts (optional):
     :token-index  pre-built token index
     :trie         pre-built byte trie
     — pass (prepare tokenizer) to share with other GFs.

   Returns a DynamicGF with the same trace structure as make-byte-llm-gf
   (:b0, :b1, ...) but each byte categorical only contains bytes that
   keep the DFA in a live state (can still reach acceptance).

   Much simpler than token-level grammar masking (grammar/wrap-grammar):
   the DFA advances exactly one character per trace site, and we filter
   the byte-marginal map (at most 256 entries) rather than masking a
   151K-entry logit vector.

   opts :commit-eager? (default false): use shortest-match tokenization (commit
   at the first token boundary) so grammar boundaries inside multi-byte tokens
   never strand generation. See trie-advance.

   Generation stops when the DFA has no valid continuations.

   LIMITATION: the DFA alphabet is printable ASCII (32–126) — regex wildcards
   and negated classes never admit non-ASCII bytes or newlines, so constrained
   output is ASCII-only (see genmlx.llm.grammar, genmlx-6dax)."
  ([model-map constraint] (constrain-bytes model-map constraint {}))
  ([model-map constraint opts]
   (let [{:keys [model tokenizer]} model-map
         eager? (boolean (:commit-eager? opts))
         dfa (if (string? constraint)
               (grammar/compile-regex constraint)
               constraint)
         {:keys [trie]}
         (if (:trie opts)
           opts
           (prepare tokenizer))]
     (dyn/auto-key
      (gen [prompt-ids max-bytes]
           (if (zero? max-bytes)
             []
             (with-byte-cache model
               (fn []
                 (loop [i 0
                        trie-pos trie
                        dfa-state (:start dfa)
                        logprobs (logits->logprobs
                                  (llm/forward-prefill model prompt-ids))
                        bytes-acc []]
                   (if (>= i max-bytes)
                     bytes-acc
                     (let [[trie-pos* logprobs* valid-lps]
                           (resolve-grammar-step model trie trie-pos logprobs dfa dfa-state)]
                       (if (empty? valid-lps)
                         bytes-acc
                         (let [{:keys [dist chars]}
                               (byte-marginals->categorical valid-lps)

                               idx (trace (keyword (str "b" i)) dist)
                               chosen-byte (nth chars (mx/item idx))
                               next-dfa (grammar/dfa-advance dfa dfa-state chosen-byte)
                               [next-pos next-logprobs]
                               (trie-advance model trie trie-pos* logprobs* chosen-byte eager?)]

                           (recur (inc i) next-pos next-dfa next-logprobs
                                  (conj bytes-acc chosen-byte)))))))))))))))



