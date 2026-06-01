(ns genmlx.llm.backend
  "Thin wrapper over mlx-node: model loading, tokenizer, forward pass,
   and log-probability extraction for LLM integration.

   This is Layer 0 of the LLM integration — everything above (token-transition
   handler, beam search, grammar constraints) builds on these functions."
  (:require ["@mlx-node/lm" :as mlx-lm :refer [ChatSession]]
            ;; Qwen3Tokenizer is sourced from @mlx-node/core directly (it is a
            ;; first-class core export). @mlx-node/lm's re-export of it was
            ;; removed upstream (mlx-node #57); relying on it broke on a clean
            ;; `tsc -b`. See bean genmlx-mwm4.
            ["@mlx-node/core" :as mlx-core]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [promesa.core :as p]))

;; ---------------------------------------------------------------------------
;; Model loading
;; ---------------------------------------------------------------------------

(defn load-model
  "Load an LLM from a directory. Returns a promise of
   {:model <Model> :tokenizer <Tokenizer> :type keyword}.

   The model directory must contain config.json, safetensors weights,
   and tokenizer.json. Model type is auto-detected from config.json.
   Supports Qwen3, Qwen3.5, Gemma4, and other HuggingFace models."
  [model-path]
  (p/let [model (.loadModel mlx-lm model-path)
          model-type (.detectModelType mlx-lm model-path)
          tokenizer (.fromPretrained (.-Qwen3Tokenizer mlx-core)
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
  (.initCaches model))

(defn reset-cache!
  "Clear KV caches after generation. Mutates model state."
  [model]
  (.resetCaches model))

(defn- forward-with-cache
  "Dispatch forwardWithCache — always pass use_cache=true."
  [model input]
  (.forwardWithCache model input true))

(defn forward-prefill
  "Run a cached forward pass over the full prompt.

   Processes all tokens at once, populates the KV cache, and returns
   logits for the last position as an MxArray of shape [vocab_size].

   Must call init-cache! before this."
  [model token-ids]
  (let [ids (->id-vec token-ids)
        n (count ids)
        input (mx/reshape (mx/array ids mx/int32) [1 n])
        logits (forward-with-cache model input)]
    (-> logits (mx/index 0) (mx/index 0))))

(defn forward-step
  "Run a single-token cached forward pass.

   Takes one token ID (integer), uses the KV cache from previous calls,
   and returns logits for the next position as an MxArray of shape [vocab_size].

   Constant time in sequence length — does not recompute the full context."
  [model token-id]
  (let [input (mx/reshape (mx/scalar token-id mx/int32) [1 1])
        logits (forward-with-cache model input)]
    (-> logits (mx/index 0) (mx/index 0))))

;; ---------------------------------------------------------------------------
;; Text generation (smoke test / convenience)
;; ---------------------------------------------------------------------------

(defn generate-text
  "Generate text from a prompt using the ChatSession API.
   Returns a promise of the generated text string.

   opts map:
     :max-tokens     — maximum new tokens (default 100)
     :temperature    — sampling temperature (default 0.7)
     :system-prompt  — optional system message prepended to chat"
  ([model-map prompt] (generate-text model-map prompt {}))
  ([{:keys [model]} prompt {:keys [max-tokens temperature system-prompt]
                            :or {max-tokens 100 temperature 0.7}}]
   (let [session (ChatSession. model
                               (clj->js (cond-> {:maxNewTokens max-tokens
                                                 :temperature temperature}
                                          system-prompt (assoc :system system-prompt))))]
     (p/let [result (.send session prompt)]
       (.-text result)))))

(defn generate-text-raw
  "Generate text by building a ChatML prompt manually and decoding token-by-token.
   Bypasses ChatSession — works for models where ChatSession has issues (e.g. Qwen2).

   opts map:
     :max-tokens     — maximum new tokens (default 100)
     :temperature    — sampling temperature (default 0, greedy argmax)
     :seed           — PRNG seed for reproducible sampling (optional)
     :system-prompt  — optional system message (default 'You are a helpful assistant.')"
  ([model-map prompt] (generate-text-raw model-map prompt {}))
  ([{:keys [model tokenizer type]} prompt {:keys [max-tokens temperature seed system-prompt]
                                          :or {max-tokens 100
                                               temperature 0
                                               system-prompt "You are a helpful assistant."}}]
   (let [think-skip (if (#{:qwen3 :qwen3_5 :qwen3_5_moe} type)
                      "<think>\n\n</think>\n\n"
                      "")
         chat-str (str "<|im_start|>system\n" system-prompt "<|im_end|>\n"
                       "<|im_start|>user\n" prompt "<|im_end|>\n"
                       "<|im_start|>assistant\n" think-skip)
         eos-id (eos-token-id tokenizer)
         greedy? (or (nil? temperature) (<= temperature 0))
         inv-temp (when-not greedy? (mx/scalar (/ 1.0 temperature)))]
     (p/let [ids-raw (encode tokenizer chat-str true)
             prompt-ids (vec ids-raw)]
       (init-cache! model)
       (try
         (let [logits (forward-prefill model prompt-ids)]
           (if greedy?
             (loop [i 0, acc [], logits logits]
               (if (>= i max-tokens)
                 (p/let [text (decode tokenizer (js/Uint32Array.from (clj->js acc)))]
                   text)
                 (let [tok-id (mx/item (mx/argmax logits))]
                   (if (= tok-id eos-id)
                     (p/let [text (decode tokenizer (js/Uint32Array.from (clj->js acc)))]
                       text)
                     (let [next-logits (forward-step model tok-id)]
                       (recur (inc i) (conj acc tok-id) next-logits))))))
             (let [rk (rng/ensure-key (when seed (rng/fresh-key seed)))]
               (loop [i 0, acc [], logits logits, rk rk]
                 (if (>= i max-tokens)
                   (p/let [text (decode tokenizer (js/Uint32Array.from (clj->js acc)))]
                     text)
                   (let [[sample-key next-key] (rng/split rk)
                         tok-id (mx/item (rng/categorical sample-key (mx/multiply logits inv-temp)))]
                     (if (= tok-id eos-id)
                       (p/let [text (decode tokenizer (js/Uint32Array.from (clj->js acc)))]
                         text)
                       (let [next-logits (forward-step model tok-id)]
                         (recur (inc i) (conj acc tok-id) next-logits next-key)))))))))
         (finally
           (reset-cache! model)))))))
