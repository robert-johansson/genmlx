(ns genmlx.llm.backend
  "Thin wrapper over mlx-node: model loading, tokenizer, forward pass,
   and log-probability extraction for LLM integration.

   This is Layer 0 of the LLM integration — everything above (token-transition
   handler, beam search, grammar constraints) builds on these functions."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            ;; f6ov: model-family-dispatching façade over the GenMLX-owned
            ;; forwards (vanilla Qwen3 + Qwen3.5 hybrid GatedDeltaNet). Same
            ;; 6-fn interface either way; routes on config.json model_type.
            [genmlx.llm.forward :as fwd]
            [promesa.core :as p]
            ["fs" :as fs]))

;; ---------------------------------------------------------------------------
;; Native LLM surface — the single @genmlx/core addon (bean genmlx-qt34,
;; finishing the PR #143 genmlx-core migration; backend was the last holdout)
;;
;; GenMLX rides on ONE MLX-linking .node addon: @genmlx/core, the GenMLX-owned
;; superset over stock mlx-core. Loading a SECOND MLX-linking addon in the same
;; process is the proven-broken case — incompatible MxArray NAPI types + separate
;; Metal pools → SIGTRAP (bean genmlx-nldo). The former @mlx-node/lm dependency
;; pulled in exactly such a second core (upstream @mlx-node/core), which is why
;; every llm/* test crashed once @genmlx/core also loaded. So this backend now
;; loads, detects, and generates entirely through @genmlx/core's native classes
;; (Qwen3Model / Qwen35Model / … with .load / .forward / .forwardWithCache /
;; .initCaches / .resetCaches / .chatSessionStart, plus Qwen3Tokenizer) — exactly
;; as genmlx.world.train already does.
;;
;; js/require (CommonJS), NOT an ESM :require: @genmlx/core's package `main` is a
;; bare Node-API addon (./index.node) with no "import" export condition, so an ESM
;; :require compiles to an `import` of a .node file, which bun 1.3.x rejects ("To
;; load Node-API modules, use require()…"). js/require is the correct .node load.
;; ---------------------------------------------------------------------------

(defonce ^:private mlx-core (js/require "@genmlx/core"))

(defn- detect-model-type
  "Read config.json's model_type, mirroring the former @mlx-node/lm detectModelType
   so the owned/upstream dispatch and the trace-site :type are unchanged: default
   \"qwen3\"; normalize gemma4_text → gemma4; a Qwen3 backbone exported as a base
   model (architectures has Qwen3Model but not Qwen3ForCausalLM) is an embedding
   model → \"harrier\". Returns the model_type string."
  [model-path]
  (let [config (-> (.readFileSync fs (str model-path "/config.json") "utf8")
                   (js/JSON.parse))
        raw    (or (.-model_type config) "qwen3")
        mtype  (if (= raw "gemma4_text") "gemma4" raw)
        archs  (or (.-architectures config) #js [])]
    (if (and (= mtype "qwen3")
             (.includes archs "Qwen3Model")
             (not (.includes archs "Qwen3ForCausalLM")))
      "harrier"
      mtype)))

(defn- load-upstream-model
  "Load a native @genmlx/core model instance for the upstream forward path,
   dispatching on model_type. Each native class exposes .forward /
   .forwardWithCache / .initCaches / .resetCaches (driven by forward-pass /
   forward-prefill / forward-step) and .chatSessionStart (generate-text), so the
   downstream backend fns are unchanged. Returns a promise of the instance, or
   throws if @genmlx/core has no model class for this family."
  [model-type model-path]
  (if-let [cls (case model-type
                 "qwen3"       (.-Qwen3Model mlx-core)
                 "qwen3_5"     (.-Qwen35Model mlx-core)
                 ;; Both MoE families share the native Qwen35MoeModel engine:
                 ;; qwen3_5_moe (256-expert) and qwen3_next (the 80B
                 ;; Qwen3-Coder-Next, config.json model_type "qwen3_next").
                 "qwen3_5_moe" (.-Qwen35MoeModel mlx-core)
                 "qwen3_next"  (.-Qwen35MoeModel mlx-core)
                 "gemma4"      (.-Gemma4Model mlx-core)
                 "harrier"     (.-HarrierModel mlx-core)
                 "lfm2"        (.-Lfm2Model mlx-core)
                 nil)]
    (.load cls model-path)
    (throw (ex-info (str "genmlx.llm.backend: @genmlx/core has no native model "
                         "class for model_type " (pr-str model-type)
                         " — load a supported family (qwen3 / qwen3_5 / gemma4 / "
                         "harrier / lfm2 / qwen3_5_moe / qwen3_next) or extend "
                         "load-upstream-model.")
                    {:genmlx/error :unsupported-upstream-type
                     :model-type model-type :model-path model-path}))))

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

(def ^:private native-moe-types
  "model_type values whose forward exists ONLY on the native (upstream mlx-node)
   path — the GenMLX-owned CLJS forward (genmlx.llm.forward) implements just the
   dense qwen3 / qwen3_5 families, never the MoE ones. Both members route to the
   native Qwen35MoeModel engine: qwen3_5_moe (256-expert) and qwen3_next (the 80B
   Qwen3-Coder-Next). Whether that native forward is SAFE to run depends on the
   GPU backend — see unsupported-native-moe? (genmlx-5luk, re-gated by mlx-2h4l)."
  #{"qwen3_5_moe" "qwen3_next"})

(defn unsupported-native-moe?
  "True when loading `model-type` would route to the native MoE forward AND that
   forward crashes on the current GPU backend — i.e. on Metal, where the
   256-expert gather_mm / mlx_qwen35_moe_forward over the expert tensors raises an
   uncatchable C++ SIGTRAP during prefill (genmlx-5luk). On CUDA the same native
   forward is verified safe (no SIGTRAP, correct output at ~native tok/s; mlx-2h4l),
   so it is NOT refused there. {:allow-native-moe? true} overrides the Metal
   refusal at the caller's risk.

   Platform-gated via mx/metal-is-available? (false on CUDA, true on Metal); the
   load-model guard stays unit-testable without loading a model, and this remains
   usable as a public 'would this model be refused here?' query."
  [model-type opts]
  (and (contains? native-moe-types model-type)
       (mx/metal-is-available?)
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
     The MoE families (qwen3_next — the 80B Qwen3-Coder-Next — and qwen3_5_moe)
     have no owned forward, so they route to the native Qwen35MoeModel; on CUDA
     this is the default path, on Metal it is refused (unsupported-native-moe?).
   - true:  force the GenMLX-owned forward (throws if the family is unsupported).
   - false: force the upstream model (the borrowed-forward fallback).
   The owned forward (CljsForwardModel) drives the forward/cache API used by the
   LLM-as-GF; it does NOT load the upstream model, so the ChatSession-based
   generate-text path requires {:cljs-forward? false} (generate-text-raw works on
   either). The VLM path (genmlx.llm.vision/load-vlm) is separate and unaffected."
  ([model-path] (load-model model-path {}))
  ([model-path opts]
   (p/let [model-type (detect-model-type model-path)
           tokenizer (.fromPretrained (.-Qwen3Tokenizer mlx-core)
                                      (str model-path "/tokenizer.json"))]
     ;; genmlx-5luk (re-gated, mlx-2h4l): on Metal, refuse a crashing native MoE
     ;; forward BEFORE the native .load below, so callers get a CATCHABLE
     ;; rejection instead of the uncatchable SIGTRAP the native MoE prefill raises
     ;; once the model is loaded and simulate runs a forward. On CUDA the native
     ;; MoE forward is verified safe, so unsupported-native-moe? is false there and
     ;; load proceeds. Opt in with {:allow-native-moe? true} to bypass on Metal.
     (if (unsupported-native-moe? model-type opts)
       ;; Return a REJECTED promise, not (throw …): a throw inside this p/let body
       ;; is wrapped by promesa/nbb and loses the ex-info data, so a caller's
       ;; p/catch would see the message but not :genmlx/error. p/rejected carries
       ;; the ex-info object through intact.
       (p/rejected
        (ex-info
         (str "genmlx.llm/load-model: model_type \"" model-type "\" is not "
              "supported on this Metal backend — its native MoE forward crashes "
              "the process with an uncatchable SIGTRAP on real checkpoints (e.g. "
              "Qwen3.6-35B-A3B-4bit; bean genmlx-5luk). The same forward runs on "
              "CUDA (mlx-2h4l). Load a dense qwen3 / qwen3_5 checkpoint, or pass "
              "{:allow-native-moe? true} to bypass this guard at your own risk.")
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
           (p/let [model (load-upstream-model model-type model-path)]
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
;; Tier-2 branchable cache (native MoE only) — bean mlx-19wy / P2
;;
;; The opaque numeric branch id IS the handle. branch-cache! forks the
;; model-internal cache at its CURRENT position (after init-cache! +
;; forward-prefill, plus any forward-steps) into an INDEPENDENT branch —
;; O(prefix) once; forward-branch then advances THAT branch in place,
;; O(1)/step, exactly like forward-step but against the isolated branch.
;; Only the native qwen3_next / qwen3_5_moe path exposes this surface; the
;; dense CljsForwardModel does not (it must keep the replay path).
;; ---------------------------------------------------------------------------

(defn supports-branching?
  "True iff `model` is the native MoE class exposing the branchable-cache
   surface. The dense CljsForwardModel has no native branch surface."
  [model]
  (and (not (cljs-forward-model? model))
       (fn? (.-branchCache model))))

(defn branch-cache!
  "Fork the model-internal cache at its CURRENT position into an independent
   branch; returns the branch's opaque numeric id. Call AFTER init-cache! +
   forward-prefill (and any forward-steps). O(prefix) once. Native MoE only."
  [model]
  (.branchCache model))

(defn branch-from
  "Fork a new sub-branch from existing branch `id`; returns the new id."
  [model id]
  (.branchFrom model id))

(defn forward-branch
  "Advance branch `id` by one token, returning next-position logits of shape
   [vocab]. O(1) in sequence length — same contract as forward-step, against
   the isolated branch instead of the model-internal cache."
  [model id token-id]
  (-> (.forwardBranch model id (ids->input [token-id]))
      (mx/index 0) (mx/index 0)))

(defn dispose-branch!
  "Free branch `id` and its cache tensors. Idempotent. Native MoE only."
  [model id]
  (.disposeBranch model id))

;; ---------------------------------------------------------------------------
;; Flat VLM prefill (native MoE VLM only) — flat-VLM-prefill
;;
;; Image-conditioned prefill that writes the FLAT model-internal caches (the same
;; caches branch-cache! forks), so an image-conditioned prefix becomes branchable.
;; The vision merge otherwise lives only on the paged path, where branchCache is
;; refused. After vlm-prefill-flat!, branch-cache! / forward-branch work unchanged
;; — the substrate for resource-rational VISION (expensive look once, cheap
;; branched re-looks / particles).
;; ---------------------------------------------------------------------------

(defn supports-vlm-prefill?
  "True iff `model` exposes the native flat VLM prefill (image-conditioned
   branchable prefix). Native MoE VLM only."
  [model]
  (and (not (cljs-forward-model? model))
       (fn? (.-vlmPrefillFlat model))))

(defn vlm-prefill-flat!
  "Image-conditioned FLAT prefill. Native-preprocesses the raw image bytes, merges
   vision features into inputs_embeds, and runs the decoder over them writing the
   model-internal FLAT caches. Returns last-position logits of shape [vocab].
   After this, branch-cache! / forward-branch work unchanged on the image-
   conditioned prefix.

   `tokens` — a Uint32Array of chat-rendered prompt token ids containing one
              IMAGE_TOKEN_ID (248056) per image (native expands each to its grid).
   `images` — a seq of Uint8Array image bytes (PNG/JPEG).

   Mutates model state: rebuilds the flat caches internally (do NOT call
   init-cache! first). Native MoE VLM only (see supports-vlm-prefill?)."
  [model tokens images]
  (-> (.vlmPrefillFlat model tokens (clj->js (vec images)))
      (mx/index 0) (mx/index 0)))

;; ---------------------------------------------------------------------------
;; Text generation (smoke test / convenience)
;; ---------------------------------------------------------------------------

(defn generate-text
  "Generate text from a prompt using the native chatSessionStart API.
   Returns a promise of the generated text string.

   opts map:
     :max-tokens       — maximum new tokens (default 100)
     :temperature      — sampling temperature (default 0.7)
     :system-prompt    — optional system message prepended to chat
     :reasoning-effort — native reasoningEffort control (default \"none\").
                         ChatResult.text contains only post-</think> content,
                         and the TemplateHonoring families (qwen3 / qwen3_5 /
                         qwen3_5_moe / qwen3_next) default to thinking ON when
                         reasoningEffort is unset — so without this default a
                         thinking model spends the whole token budget inside
                         <think> and a non-thinking coder model (qwen3_next,
                         which never emits </think> after the injected <think>
                         opener) returns \"\" ALWAYS (bean genmlx-87ga).
                         Defaulting to \"none\" mirrors generate-text-raw's
                         explicit think-skip. Pass \"medium\"/\"high\" to
                         re-enable thinking (.text is then the post-think
                         answer only)."
  ([model-map prompt] (generate-text model-map prompt {}))
  ([{:keys [model]} prompt {:keys [max-tokens temperature system-prompt reasoning-effort]
                            :or {max-tokens 100 temperature 0.7
                                 reasoning-effort "none"}}]
   (when (cljs-forward-model? model)
     (throw (ex-info (str "generate-text drives the native model's chatSessionStart, "
                          "which requires the upstream model. Load with "
                          "{:cljs-forward? false}, or use generate-text-raw (works "
                          "on the owned forward).")
                     {:model-type (type model)})))
   ;; chatSessionStart applies the chat template internally and starts a fresh
   ;; turn-1 conversation per call (system prompt goes in the message list, not a
   ;; :system config field), so it matches the old new-ChatSession-per-call
   ;; isolation. ChatResult exposes the decoded text via .text.
   (let [messages (clj->js (cond-> []
                             system-prompt (conj {:role "system" :content system-prompt})
                             :always       (conj {:role "user" :content prompt})))
         config   (clj->js (cond-> {:maxNewTokens max-tokens :temperature temperature}
                             reasoning-effort (assoc :reasoningEffort reasoning-effort)))]
     (p/let [result (.chatSessionStart model messages config)]
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
