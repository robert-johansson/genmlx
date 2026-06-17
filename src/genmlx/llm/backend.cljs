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
            ;; f6ov: model-family-dispatching façade over the GenMLX-owned
            ;; forwards (vanilla Qwen3 + Qwen3.5 hybrid GatedDeltaNet). Same
            ;; 6-fn interface either way; routes on config.json model_type.
            [genmlx.llm.forward :as fwd]
            [promesa.core :as p]))

;; ---------------------------------------------------------------------------
;; GenMLX-owned forward (f6ov): a value-level CLJS forward over genmlx.rs
;; primitives instead of upstream's per-model forward structs.
;;
;; CljsForwardModel wraps the functional family forward (genmlx.llm.forward,
;; whose prefill/step RETURN the next cache) in a single mutable cache cell, so
;; it presents the SAME stateful API as the upstream model (init-cache! /
;; forward-prefill / forward-step / reset-cache!) and every caller (LLM-as-GF,
;; byte/structured GFs) works unchanged. The forward dispatches on model_type
;; (vanilla Qwen3 or Qwen3.5 hybrid). The atom is the one audited KV-cache
;; mutation boundary.
;; ---------------------------------------------------------------------------

(defrecord CljsForwardModel [fwd cache])  ; fwd = {:config :weights :impl}; cache = atom

(defn cljs-forward-model? [model] (instance? CljsForwardModel model))

;; ---------------------------------------------------------------------------
;; Install guard — Tier-B LLM-forward capability probe (genmlx-91b3)
;;
;; Tier-A (genmlx.mlx) asserts the native CORE is alive at require time. Tier-B
;; asserts, at load-model, that the forward surface the LLM-as-GF actually drives
;; is present — and it differs by path because f6ov flipped the default:
;;   - OWNED path (default for supported families): the GenMLX-owned forward
;;     (genmlx.llm.forward) supplies the fns this backend actually drives —
;;     load-model, next-token-logits (the uncached scoring path), prefill, step.
;;   - UPSTREAM path ({:cljs-forward? false} / unsupported family): the loaded
;;     mlx-node model INSTANCE supplies native .forward + .forwardWithCache.
;; The upstream check is the one that catches the genmlx-7siy stale-prebuilt
;; fallback (@mlx-node/core@0.0.6 lacks Qwen3.5 .forward), turning a cryptic
;; "Could not find instance method: forward" into a clear rebuild instruction.
;; Both assert CAPABILITY, not version.
;; ---------------------------------------------------------------------------

(defn assert-owned-forward!
  "Tier-B (owned path): the GenMLX-owned forward must expose the fns this backend
   actually drives on the CljsForwardModel path — load-model (constructs the
   model), next-token-logits (the uncached scoring path forward-pass drives),
   prefill, and step (the cached path). Throws a clear ex-info naming the missing
   fn if genmlx.llm.forward is broken; returns nil when complete. (Deliberately
   does NOT check fwd/forward — it is not invoked on the owned LLM-as-GF path.)"
  []
  (let [present {:load-model        (fn? fwd/load-model)
                 :next-token-logits (fn? fwd/next-token-logits)
                 :prefill           (fn? fwd/prefill)
                 :step              (fn? fwd/step)}
        missing (->> present (remove (comp true? val)) (mapv key))]
    (when (seq missing)
      (throw (ex-info
               (str "genmlx.llm.backend: the GenMLX-owned forward is missing "
                    (pr-str missing) " — genmlx.llm.forward is broken or partially built. "
                    "Rebuild the native layer and reinstall:\n"
                    "  (cd mlx-node && yarn build:native) && bun install")
               {:genmlx/error :owned-forward-incomplete :missing missing})))
    nil))

(defn assert-upstream-forward!
  "Tier-B (upstream path): the loaded mlx-node model instance must expose
   .forward + .forwardWithCache. Throws a clear ex-info (naming the missing
   capability + rebuild steps) when the binary is stale/incompatible — the guard
   that converts the genmlx-7siy cryptic NAPI failure into an actionable error.
   Returns nil when capable."
  [model]
  (let [present {:forward          (fn? (.-forward model))
                 :forwardWithCache (fn? (.-forwardWithCache model))}
        missing (->> present (remove (comp true? val)) (mapv key))]
    (when (seq missing)
      (throw (ex-info
               (str "genmlx.llm.backend: the loaded mlx-node model is missing "
                    (pr-str missing) " — the native binary is stale/incompatible "
                    "(the napi loader likely fell back to an older prebuilt). Rebuild it:\n"
                    "  git submodule update --init --recursive\n"
                    "  (cd mlx-node && yarn build:native) && bun install\n"
                    "Or, for a supported family, force the GenMLX-owned forward with "
                    "{:cljs-forward? true}.")
               {:genmlx/error :upstream-forward-incompatible :missing missing})))
    nil))

;; ---------------------------------------------------------------------------
;; Model loading
;; ---------------------------------------------------------------------------

(def ^:private crashing-native-moe-types
  "model_type values whose native (upstream mlx-node) forward hard-crashes the
   process with an uncatchable SIGTRAP on real checkpoints. Currently the
   256-expert qwen3_5_moe MoE/VLM forward (genmlx-5luk): the panic is a C++
   exception in the native gather_mm / mlx_qwen35_moe_forward over the expert
   weight tensors during prefill, surfaced as SIGTRAP and uncatchable from CLJS.
   No qwen3_5_moe checkpoint is verified-working and the owned CLJS forward does
   not implement the MoE family, so loading one is refused by default."
  #{"qwen3_5_moe"})

(defn unsupported-native-moe?
  "True when `model-type` would route to a known-crashing native MoE forward and
   the caller has NOT opted in via {:allow-native-moe? true}. Pure and
   side-effect-free so the load-model guard is unit-testable without loading a
   model, and usable as a public 'would this model be refused?' query (genmlx-5luk)."
  [model-type opts]
  (and (contains? crashing-native-moe-types model-type)
       (not (:allow-native-moe? opts))))

(defn load-model
  "Load an LLM from a directory. Returns a promise of
   {:model <Model> :tokenizer <Tokenizer> :type keyword}.

   The model directory must contain config.json, safetensors weights,
   and tokenizer.json. Model type is auto-detected from config.json.
   Supports Qwen3, Qwen3.5, Gemma4, and other HuggingFace models.

   Forward selection (f6ov), :cljs-forward? in opts:
   - omitted (DEFAULT): SMART — use the GenMLX-OWNED pure-CLJS forward for the
     model families it implements (genmlx.llm.forward/supported?: qwen3, qwen3_5)
     and the upstream model otherwise. So trusted Qwen3/Qwen3.5 checkpoints run on
     the owned forward by default; MoE/Gemma/other types fall back to upstream
     automatically (and auto-upgrade once the owned forward learns the family).
   - true:  force the GenMLX-owned forward (throws if the family is unsupported).
   - false: force the upstream model (the borrowed-forward fallback).
   The owned forward (CljsForwardModel) drives the forward/cache API used by the
   LLM-as-GF; it does NOT load the upstream model, so the ChatSession-based
   generate-text path requires {:cljs-forward? false} (generate-text-raw works on
   either). The VLM path (genmlx.llm.vision/load-vlm) is separate and unaffected."
  ([model-path] (load-model model-path {}))
  ([model-path opts]
   (p/let [model-type (.detectModelType mlx-lm model-path)
           tokenizer (.fromPretrained (.-Qwen3Tokenizer mlx-core)
                                      (str model-path "/tokenizer.json"))]
     ;; genmlx-5luk: refuse a known-crashing native MoE forward BEFORE the native
     ;; .loadModel below, so callers get a CATCHABLE rejection instead of the
     ;; uncatchable SIGTRAP the native qwen3_5_moe prefill raises once the model
     ;; is loaded and simulate runs a forward. Opt in with {:allow-native-moe?
     ;; true} to attempt it anyway at your own risk.
     (if (unsupported-native-moe? model-type opts)
       ;; Return a REJECTED promise, not (throw …): a throw inside this p/let body
       ;; is wrapped by promesa/nbb and loses the ex-info data, so a caller's
       ;; p/catch would see the message but not :genmlx/error. p/rejected carries
       ;; the ex-info object through intact.
       (p/rejected
        (ex-info
         (str "genmlx.llm/load-model: model_type \"" model-type "\" is not "
              "supported — its native MoE forward crashes the process with "
              "an uncatchable SIGTRAP on real checkpoints (e.g. "
              "Qwen3.6-35B-A3B-4bit; bean genmlx-5luk). Load a dense qwen3 / "
              "qwen3_5 checkpoint, or pass {:allow-native-moe? true} to "
              "bypass this guard at your own risk.")
         {:genmlx/error :unsupported-model-type
          :model-type (keyword model-type)
          :model-path model-path}))
       ;; Explicit :cljs-forward? always wins (keeps the borrowed forward reachable
       ;; as a one-release fallback); otherwise default to owned iff the owned
       ;; forward implements this checkpoint's family.
       (let [use-cljs? (if (contains? opts :cljs-forward?)
                         (boolean (:cljs-forward? opts))
                         (fwd/supported? model-path))]
         (if use-cljs?
           (do
             ;; Tier-B (owned path): assert the owned forward surface before use.
             (assert-owned-forward!)
             {:model (->CljsForwardModel (fwd/load-model model-path) (atom nil))
              :tokenizer tokenizer
              :type (keyword model-type)})
           (p/let [model (.loadModel mlx-lm model-path)]
             ;; Tier-B (upstream path): assert the loaded instance exposes the
             ;; native forward methods — catches the genmlx-7siy stale prebuilt.
             (assert-upstream-forward! model)
             {:model model
              :tokenizer tokenizer
              :type (keyword model-type)})))))))

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
          t (second (mx/shape logits))]
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
   (when (cljs-forward-model? model)
     (throw (ex-info (str "generate-text uses ChatSession, which requires the "
                          "upstream model. Load with {:cljs-forward? false}, or "
                          "use generate-text-raw (works on the owned forward).")
                     {:model-type (type model)})))
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
