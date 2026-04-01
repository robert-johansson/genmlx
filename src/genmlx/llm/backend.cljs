(ns genmlx.llm.backend
  "Thin wrapper over mlx-node: model loading, tokenizer, forward pass,
   and log-probability extraction for LLM integration.

   This is Layer 0 of the LLM integration — everything above (token-transition
   handler, beam search, grammar constraints) builds on these functions."
  (:require ["@mlx-node/core" :as mlx-node]
            ["@mlx-node/lm" :as mlx-lm]
            ["./bridge.js" :as bridge]
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
;; Forward pass and array boundary
;; ---------------------------------------------------------------------------

(defn forward-pass
  "Run a forward pass through the model.

   Takes a model and token IDs (Uint32Array or cljs vector of ints).
   Returns logits for the last position as a MLX array of shape [vocab_size].

   The logits are sliced on the mlx-node side before crossing the boundary,
   so only [vocab_size] floats are materialized (not [1, T, vocab_size])."
  [model token-ids]
  (let [ids (if (instance? js/Uint32Array token-ids)
              token-ids
              (js/Uint32Array.from (clj->js token-ids)))
        ;; forwardLastLogits handles MxArray creation, forward pass,
        ;; slicing, and Float32Array extraction entirely in JS context
        f32 (.forwardLastLogits bridge model ids)]
    ;; Cross the boundary: Float32Array → MLX array
    (mx/array f32)))

(defn next-token-logprobs
  "Get log-probabilities for the next token given context.

   Takes a model and token IDs (Uint32Array or cljs vector).
   Returns a MLX array of shape [vocab_size] containing log p(token | context)
   for every token in the vocabulary.

   Uses numerically stable log-softmax: log_softmax(x) = x - log(sum(exp(x)))
   with max-subtraction for stability."
  [model token-ids]
  (let [logits (forward-pass model token-ids)
        max-val (mx/amax logits)
        shifted (mx/subtract logits max-val)
        log-sum-exp (mx/log (mx/sum (mx/exp shifted)))]
    (mx/subtract shifted log-sum-exp)))

;; ---------------------------------------------------------------------------
;; Text generation (smoke test / convenience)
;; ---------------------------------------------------------------------------

(defn generate-text
  "Generate text from a prompt using the chat API.
   Returns a promise of the generated text string.

   opts map:
     :max-tokens  — maximum new tokens (default 100)
     :temperature — sampling temperature (default 0.7)"
  ([model-map prompt] (generate-text model-map prompt {}))
  ([{:keys [model]} prompt {:keys [max-tokens temperature]
                            :or {max-tokens 100 temperature 0.7}}]
   (p/let [result (.chat model
                         (clj->js [{:role "user" :content prompt}])
                         (clj->js {:maxNewTokens max-tokens
                                   :temperature temperature}))]
     (.-text result))))
