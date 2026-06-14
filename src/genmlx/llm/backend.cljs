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
            [genmlx.llm.qwen3-forward :as fwd]
            [promesa.core :as p]))

;; ---------------------------------------------------------------------------
;; GenMLX-owned forward (f6ov): a value-level CLJS forward over genmlx.rs
;; primitives instead of upstream's per-model forward structs.
;;
;; CljsForwardModel wraps the functional qwen3-forward (whose prefill/step RETURN
;; the next cache) in a single mutable cache cell, so it presents the SAME
;; stateful API as the upstream model (init-cache! / forward-prefill /
;; forward-step / reset-cache!) and every caller (LLM-as-GF, byte/structured GFs)
;; works unchanged. The atom is the one audited KV-cache mutation boundary.
;; ---------------------------------------------------------------------------

(defrecord CljsForwardModel [fwd cache])  ; fwd = {:config :weights}; cache = atom

(defn cljs-forward-model? [model] (instance? CljsForwardModel model))

;; ---------------------------------------------------------------------------
;; Model loading
;; ---------------------------------------------------------------------------

(defn load-model
  "Load an LLM from a directory. Returns a promise of
   {:model <Model> :tokenizer <Tokenizer> :type keyword}.

   The model directory must contain config.json, safetensors weights,
   and tokenizer.json. Model type is auto-detected from config.json.
   Supports Qwen3, Qwen3.5, Gemma4, and other HuggingFace models.

   opts :cljs-forward? (default false) — load a GenMLX-OWNED forward (f6ov):
   :model is a CljsForwardModel driving a pure-CLJS Qwen3 forward over the
   genmlx.rs primitives, decoupled from upstream's model structs. Supports the
   forward/cache API used by the LLM-as-GF (standard Qwen3 only; ChatSession-based
   convenience paths still need the upstream model). The upstream model is not
   loaded in this mode."
  ([model-path] (load-model model-path {}))
  ([model-path {:keys [cljs-forward?]}]
   (p/let [model-type (.detectModelType mlx-lm model-path)
           tokenizer (.fromPretrained (.-Qwen3Tokenizer mlx-core)
                                      (str model-path "/tokenizer.json"))]
     (if cljs-forward?
       {:model (->CljsForwardModel (fwd/load-model model-path) (atom nil))
        :tokenizer tokenizer
        :type (keyword model-type)}
       (p/let [model (.loadModel mlx-lm model-path)]
         {:model model
          :tokenizer tokenizer
          :type (keyword model-type)})))))

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

(defn- ids->input
  "Build a [1 N] int32 model-input MxArray from token ids.

   Constructs the [1 N] array directly (fromInt32 with the shape baked in)
   rather than building [N] and reshaping to [1 N]. A reshaped array can be
   bound to a non-default GPU stream in some mlx-node/MLX builds; feeding it to
   the model then crashes the attention kernel with
   'no Stream(gpu, 1) in current thread' on certain GPUs (reproduced on M2 Max,
   not on M4). Baking the shape into construction keeps the input on the default
   GPU stream. See bean genmlx-7siy."
  [ids]
  (let [v (->id-vec ids)]
    (mx/array v [1 (count v)] mx/int32)))

(defn forward-pass
  "Run a forward pass through the model.

   Takes a model and token IDs (Uint32Array or cljs vector of ints).
   Returns logits for the last position as an MxArray of shape [vocab_size].

   All operations stay on the MLX graph — no materialization to typed arrays."
  [model token-ids]
  (if (cljs-forward-model? model)
    (fwd/next-token-logits (:fwd model) (->id-vec token-ids))
    (let [ids (->id-vec token-ids)
          input (ids->input ids)
          logits (.forward model input)
          ;; Index the LAST position from the logits' ACTUAL time dimension, not
          ;; the input token count: this mlx-node build's .forward returns
          ;; [1 1 vocab] (last position only). The old (dec n) indexed row n-1 of a
          ;; 1-row matrix → out-of-range garbage (decoded to "导图" instead of the
          ;; real next token). Using (dec t) is correct whether .forward returns
          ;; [1 1 vocab] or a full [1 T vocab].
          t (nth (mx/shape logits) 1)]
      (-> logits (mx/index 0) (mx/index (dec t))))))

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
  (if (cljs-forward-model? model)
    (reset! (:cache model) nil)
    (.initCaches model)))

(defn reset-cache!
  "Clear KV caches after generation. Mutates model state."
  [model]
  (if (cljs-forward-model? model)
    (reset! (:cache model) nil)
    (.resetCaches model)))

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
  (if (cljs-forward-model? model)
    (let [ids (->id-vec token-ids)
          [logits cache] (fwd/prefill (:fwd model) ids)]
      (reset! (:cache model) {:cache cache :offset (count ids)})
      logits)
    (let [input (ids->input token-ids)
          logits (forward-with-cache model input)]
      (-> logits (mx/index 0) (mx/index 0)))))

(defn forward-step
  "Run a single-token cached forward pass.

   Takes one token ID (integer), uses the KV cache from previous calls,
   and returns logits for the next position as an MxArray of shape [vocab_size].

   Constant time in sequence length — does not recompute the full context."
  [model token-id]
  (if (cljs-forward-model? model)
    (let [{:keys [cache offset]} @(:cache model)
          [logits cache'] (fwd/step (:fwd model) cache offset token-id)]
      (reset! (:cache model) {:cache cache' :offset (inc offset)})
      logits)
    (let [input (ids->input [token-id])
          logits (forward-with-cache model input)]
      (-> logits (mx/index 0) (mx/index 0)))))

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
         inv-temp (when-not greedy? (mx/scalar (/ 1.0 temperature)))
         decode-acc (fn [acc]
                      (decode tokenizer (js/Uint32Array.from (clj->js acc))))
         ;; Pick the next token id and advance the PRNG key. Greedy ignores
         ;; the key and takes the argmax; sampled splits the key and draws.
         pick (if greedy?
                (fn [logits rk] [(mx/item (mx/argmax logits)) rk])
                (fn [logits rk]
                  (let [[sample-key next-key] (rng/split rk)]
                    [(mx/item (rng/categorical sample-key (mx/multiply logits inv-temp)))
                     next-key])))]
     (p/let [ids-raw (encode tokenizer chat-str true)
             prompt-ids (vec ids-raw)]
       (init-cache! model)
       (try
         (loop [i 0
                acc []
                logits (forward-prefill model prompt-ids)
                rk (rng/ensure-key (when seed (rng/fresh-key seed)))]
           (if (>= i max-tokens)
             (decode-acc acc)
             (let [[tok-id next-rk] (pick logits rk)]
               (if (= tok-id eos-id)
                 (decode-acc acc)
                 (recur (inc i) (conj acc tok-id)
                        (forward-step model tok-id) next-rk)))))
         (finally
           (reset-cache! model)))))))
