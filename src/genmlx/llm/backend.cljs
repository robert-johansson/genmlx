(ns genmlx.llm.backend
  "Thin wrapper over mlx-node: model loading, tokenizer, forward pass,
   and log-probability extraction for LLM integration.

   This is Layer 0 of the LLM integration — everything above (token-transition
   handler, beam search, grammar constraints) builds on these functions."
  (:require ["@mlx-node/lm" :as mlx-lm]
            [genmlx.mlx :as mx]
            [promesa.core :as p]))

;; ---------------------------------------------------------------------------
;; Model loading
;; ---------------------------------------------------------------------------

(defn load-model
  "Load an LLM from a directory. Returns a promise of
   {:model <Qwen3Model|Qwen35Model> :tokenizer <Qwen3Tokenizer> :type keyword}.

   The model directory must contain config.json and safetensors weights.
   Model type is auto-detected from config.json."
  [model-path]
  (p/let [model (.loadModel mlx-lm model-path)
          model-type (.detectModelType mlx-lm model-path)
          tokenizer (.fromPretrained (.-Qwen3Tokenizer mlx-lm)
                                     (str model-path "/tokenizer.json"))]
    {:model model
     :tokenizer tokenizer
     :type (keyword model-type)}))

;; ---------------------------------------------------------------------------
;; Tokenizer
;; ---------------------------------------------------------------------------

(defn encode
  "Encode text to token IDs. Returns a promise of a Uint32Array.
   add-special-tokens defaults to false (raw encoding)."
  ([tokenizer text] (encode tokenizer text false))
  ([tokenizer text add-special-tokens]
   (.encode tokenizer text add-special-tokens)))

(defn decode
  "Decode token IDs (Uint32Array) to text. Returns a promise of a string."
  [tokenizer token-ids]
  (.decode tokenizer token-ids))

(defn vocab-size
  "Return the vocabulary size (synchronous)."
  [tokenizer]
  (.vocabSize tokenizer))

(defn eos-token-id
  "Return the EOS token ID (synchronous)."
  [tokenizer]
  (.getEosTokenId tokenizer))

(defn pad-token-id
  "Return the PAD token ID (synchronous)."
  [tokenizer]
  (.getPadTokenId tokenizer))

(defn id->token
  "Convert a token ID to its string representation. Returns nil if unknown."
  [tokenizer id]
  (.idToToken tokenizer id))

(defn token->id
  "Convert a token string to its ID. Returns nil if unknown."
  [tokenizer token]
  (.tokenToId tokenizer token))

;; ---------------------------------------------------------------------------
;; Forward pass
;; ---------------------------------------------------------------------------

(defn- ->id-vec
  "Coerce token IDs to a ClojureScript vector of ints."
  [token-ids]
  (if (vector? token-ids) token-ids (vec token-ids)))

(defn forward-pass
  "Run a forward pass through the model.

   Takes a model and token IDs (Uint32Array or cljs vector of ints).
   Returns logits for the last position as an MxArray of shape [vocab_size].

   All operations stay on the MLX graph — no materialization to typed arrays."
  [model token-ids]
  (let [ids (->id-vec token-ids)
        n (count ids)
        input (mx/reshape (mx/array ids mx/int32) [1 n])
        logits (.forward model input)]
    (-> logits (mx/index 0) (mx/index (dec n)))))

(defn next-token-logprobs
  "Get log-probabilities for the next token given context.

   Takes a model and token IDs (Uint32Array or cljs vector).
   Returns an MxArray of shape [vocab_size] containing log p(token | context)
   for every token in the vocabulary.

   Uses numerically stable log-softmax: log_softmax(x) = x - log(sum(exp(x)))
   with max-subtraction for stability."
  [model token-ids]
  (let [logits (forward-pass model token-ids)
        max-val (mx/amax logits)
        shifted (mx/subtract logits max-val)]
    (mx/subtract shifted (mx/log (mx/sum (mx/exp shifted))))))

;; ---------------------------------------------------------------------------
;; KV cache management
;; ---------------------------------------------------------------------------

(defn init-cache!
  "Initialize KV caches for incremental generation.
   Must be called before forward-step. Mutates model state."
  [model]
  (.initKvCaches model))

(defn reset-cache!
  "Clear KV caches after generation. Mutates model state."
  [model]
  (.resetKvCaches model))

(defn forward-prefill
  "Run a cached forward pass over the full prompt.

   Processes all tokens at once, populates the KV cache, and returns
   logits for the last position as an MxArray of shape [vocab_size].

   Must call init-cache! before this."
  [model token-ids]
  (let [ids (->id-vec token-ids)
        n (count ids)
        input (mx/reshape (mx/array ids mx/int32) [1 n])
        logits (.forwardWithCache model input true)]
    (-> logits (mx/index 0) (mx/index (dec n)))))

(defn forward-step
  "Run a single-token cached forward pass.

   Takes one token ID (integer), uses the KV cache from previous calls,
   and returns logits for the next position as an MxArray of shape [vocab_size].

   Constant time in sequence length — does not recompute the full context."
  [model token-id]
  (let [input (mx/reshape (mx/scalar token-id mx/int32) [1 1])
        logits (.forwardWithCache model input true)]
    (-> logits (mx/index 0) (mx/index 0))))

;; ---------------------------------------------------------------------------
;; Text generation (smoke test / convenience)
;; ---------------------------------------------------------------------------

(defn generate-text
  "Generate text from a prompt using the chat API.
   Returns a promise of the generated text string.

   opts map:
     :max-tokens     — maximum new tokens (default 100)
     :temperature    — sampling temperature (default 0.7)
     :system-prompt  — optional system message prepended to chat"
  ([model-map prompt] (generate-text model-map prompt {}))
  ([{:keys [model]} prompt {:keys [max-tokens temperature system-prompt]
                            :or {max-tokens 100 temperature 0.7}}]
   (let [messages (cond-> []
                    system-prompt (conj {:role "system" :content system-prompt})
                    true (conj {:role "user" :content prompt}))]
     (p/let [result (.chat model
                           (clj->js messages)
                           (clj->js {:maxNewTokens max-tokens
                                     :temperature temperature}))]
       (.-text result)))))
