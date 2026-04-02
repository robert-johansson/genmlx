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
   forced (leaf node with no children). This matches BPE longest-match."
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
                 (if (and tok (pos? (count tok)))
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
  (let [entries (vec byte-lps)
        chars (mapv first entries)
        logits (mx/array (mapv second entries))]
    {:dist (dist/categorical logits)
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

(defn logits->logprobs
  "Numerically stable log-softmax: logits - logsumexp(logits).
   Returns MxArray [vocab-size]."
  [logits]
  (mx/subtract logits (mx/logsumexp logits)))

(defn trie-leaf?
  "A trie node is a leaf when it has no children — the accumulated bytes
   form a complete token that must be committed."
  [node]
  (empty? (:children node)))

(defn commit-token-id
  "Select the token ID to commit at a leaf node. A leaf's token-ids set
   contains exactly the tokens whose byte sequence ends here."
  [node]
  (first (:token-ids node)))

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
   (let [{:keys [model]} model-map
         {:keys [token-index trie]}
         (if (and (:token-index opts) (:trie opts))
           opts
           (prepare (:tokenizer model-map)))]
     (dyn/auto-key
      (gen [prompt-ids max-bytes]
           (if (zero? max-bytes)
             []
             (do
               (llm/init-cache! model)
               (try
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
                           next-node (get-in trie-pos [:children chosen-byte])]

                       (if (trie-leaf? next-node)
                         (recur (inc i) trie
                                (logits->logprobs
                                 (llm/forward-step model
                                                   (commit-token-id next-node)))
                                (conj bytes-acc chosen-byte))
                         (recur (inc i) next-node logprobs
                                (conj bytes-acc chosen-byte))))))
                 (finally
                   (llm/reset-cache! model))))))))))

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

   Generation stops when the DFA has no valid continuations."
  ([model-map constraint] (constrain-bytes model-map constraint {}))
  ([model-map constraint opts]
   (let [{:keys [model]} model-map
         dfa (if (string? constraint)
               (grammar/compile-regex constraint)
               constraint)
         {:keys [trie]}
         (if (:trie opts)
           opts
           (prepare (:tokenizer model-map)))
         alive (:alive dfa)]
     (dyn/auto-key
      (gen [prompt-ids max-bytes]
           (if (zero? max-bytes)
             []
             (do
               (llm/init-cache! model)
               (try
                 (loop [i 0
                        trie-pos trie
                        dfa-state (:start dfa)
                        logprobs (logits->logprobs
                                  (llm/forward-prefill model prompt-ids))
                        bytes-acc []]
                   (if (>= i max-bytes)
                     bytes-acc
                     (let [raw-lps (byte-logprobs trie-pos logprobs)
                           valid-lps (into {}
                                           (filter (fn [[ch _]]
                                                     (let [s (grammar/dfa-advance
                                                              dfa dfa-state ch)]
                                                       (and (not= s :dead)
                                                            (contains? alive s)))))
                                           raw-lps)]
                       (if (empty? valid-lps)
                         bytes-acc
                         (let [{:keys [dist chars]}
                               (byte-marginals->categorical valid-lps)

                               idx (trace (keyword (str "b" i)) dist)
                               chosen-byte (nth chars (mx/item idx))
                               next-dfa (grammar/dfa-advance dfa dfa-state chosen-byte)
                               next-node (get-in trie-pos [:children chosen-byte])]

                           (if (trie-leaf? next-node)
                             (recur (inc i) trie next-dfa
                                    (logits->logprobs
                                     (llm/forward-step model
                                                       (commit-token-id next-node)))
                                    (conj bytes-acc chosen-byte))
                             (recur (inc i) next-node next-dfa logprobs
                                    (conj bytes-acc chosen-byte))))))))
                 (finally
                   (llm/reset-cache! model))))))))))



