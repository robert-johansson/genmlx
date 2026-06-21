(ns genmlx.world.train
  "The TRAINING face of the world membrane (Phase 0, bean genmlx-zftr) — a thin,
   honest boundary over the native GRPO training engine, a sibling to `genmlx.mlx`
   (the COMPUTE membrane), `genmlx.world.net` (the NETWORK membrane) and
   `genmlx.world.proc` (the SCHEDULER).

   THE ONE EFFECT — training is RL's `eval!`-equivalent. A GRPO step generates a
   group of completions, scores them with a PURE reward, computes group-relative
   advantages + a clipped surrogate, and applies an AdamW update that MUTATES the
   model's weights and the optimizer's moment state IN PLACE on the model thread.
   That in-place mutation is the sole side effect of this face. Everything above it
   stays pure: the reward is a pure `(prompt, completion) -> number`, the rollout is
   the engine's own generation, and the policy is a GF.

   Same architectural MOVE as mlx.cljs, different substrate GRADE:

     mlx-node : compute effect  : eval! -> Metal      SYNC dispatch to a TRANSPARENT
                                                       substrate (your own values,
                                                       realized exactly)
     training : learning effect : train-step! -> AdamW ASYNC crossing that MUTATES a
                                                       borrowed model IN PLACE; the
                                                       updated weights are observable
                                                       state OWNED BY THE CALLER (like
                                                       the KV cache), not hidden global
                                                       state

   THE MUTABLE BOUNDARY (the quarantine). The native `GrpoTrainingEngine` handle is
   the ONE mutable resource of this face — created inside a blessed scope, consumed
   locally, torn down on exit. It never escapes into pure code, and its in-place
   weight updates are a PARALLEL path that never composes back into the pure
   GFI-score gradient flow. Fenced exactly like the two existing precedents:
   `world.net`'s `with-server` (a live Bun.serve listener) and `llm/backend.cljs`'s
   KV-cache atom (mutation inside try/finally). `with-trainer` is the blessed scope
   (p/finally teardown); `make-trainer!` is the low-level escape hatch.

   GOTCHA (load-bearing): the native `trainStepAuto` reward callback awaits a NATIVE
   `js/Promise<number[]>`. A promesa promise is NOT `instanceof js/Promise`, so the
   reward bridge resolves with a native `js/Promise.resolve` (NOT promesa) — the same
   native-callback rule that bites `world.net`'s Bun fetch handler. Client-side
   promesa (the `train-step!` return, `with-trainer`) is fine.

   THE REWARD BRIDGE (the seam to Phase 1). `train-step!` marshals a PURE CLJS
   reward-fn into the native callback. Phase 0 ships a trivial reward (completion
   length) to prove the round-trip mutates weights. Phase 1 (genmlx-ugkv) swaps in a
   GFI quantity — `msa/score-model` marginal log-ML or `codegen/verify` accuracy —
   through the SAME seam, no new native code.

   sync/async split: a training step crosses to a native promise, so `train-step!` /
   `generate-batch` / `with-trainer` return promesa promises (CLAUDE.md's 'sync math,
   async events', like model load / `.chat`). The reward math itself is pure & sync.

   Native surface VERIFIED 2026-06-20 against @genmlx/core/index.d.ts (the addon
   GenMLX loads — NOT @mlx-node/core, which the original spec named): the runtime
   class is `GrpoTrainingEngine` (factory `fromQwen35`/`fromQwen35Moe`, ctor takes a
   Qwen3 model); checkpointing of WEIGHTS lives on the model handle (`saveModel`),
   the engine only persists OPTIMIZER moments (`saveOptimizerState`); there is no
   native `dispose`/`saveCheckpoint`, and `GrpoEngineConfig` has no `seed` field."
  (:require [promesa.core :as p]))

;; ===========================================================================
;; Native binding (lazy, off the @genmlx/core object `c` — mirrors mlx.cljs:27).
;; The two names bound here off `c` (GrpoTrainingEngine, createRandomQwen35Checkpoint)
;; are the Phase-0 training surface; membrane_coverage_test scans this file and
;; classifies them WRAPPED. The remaining training exports (Sft engine, the reward
;; registry, the result/persistence types, the other random-checkpoint helpers)
;; stay on the omission allowlist until Phase 1-4 tap them.
;; ===========================================================================

(defonce ^:private c (js/require "@genmlx/core"))

(def ^:private GrpoTrainingEngine (.-GrpoTrainingEngine c))
(def ^:private create-random-qwen35-checkpoint (.-createRandomQwen35Checkpoint c))

(defn available?
  "True when the native GRPO training engine backing this face is present. Pure flow
   above can branch on this to skip cleanly when @genmlx/core lacks the training
   surface (mirrors world.net/proc `available?`)."
  []
  (boolean (and c (fn? GrpoTrainingEngine))))

;; ===========================================================================
;; Config marshalling — CLJS map -> native camelCase config objects.
;; ===========================================================================

(def ^:private loss-type->native
  {:grpo "grpo" :dapo "dapo" :dr-grpo "dr_grpo" :bnpo "bnpo"})

(defn- ->engine-config
  "Map a CLJS config to a native GrpoEngineConfig (#js, camelCase). The curated kebab
   keys below are mapped explicitly; ANY OTHER native GrpoEngineConfig field (e.g.
   :gradientClipValue, :weightDecay, :optimizerType, :presencePenalty, :adamwBeta1)
   can be passed camelCase under `:raw` and is merged verbatim. Unrecognized TOP-LEVEL
   keys are IGNORED. Only set keys are sent, so native defaults apply for the rest.

   Supported kebab keys: :learning-rate :group-size :kl-coef (alias :beta) :loss-type
   :max-completion-length :temperature :top-p :top-k :clip-epsilon :gradient-clip-norm
   :gradient-accumulation-steps :repetition-penalty :enable-thinking :lm-head-chunk-size
   :forward-chunk-size.

   :enable-thinking false adds empty <think></think> tags to the prompt so a Qwen3
   model emits a DIRECT answer instead of a (possibly truncated) reasoning block —
   essential when the completion must be parseable output (e.g. code). The two
   chunk-size knobs cap peak memory for large-vocab models (the native docs
   recommend :lm-head-chunk-size 2 and :forward-chunk-size 4 for group-size >= 4).

   `:beta`/`:kl-coef` -> klCoef. Setting it > 0 applies a true KL-to-base penalty
   through the autograd path (genmlx-65d5): on the first KL-enabled step the native
   engine snapshots the frozen base policy, then each step adds the k3 KL term
   `KL(ref‖policy)` (β-scaled) regularizing toward that base. KL(ref‖policy) and its
   gradient are 0 at step 1 (policy == ref by construction); the effect grows as the
   policy diverges. Leave it 0 (default) for KL-free training. There is no `:seed` —
   the engine owns its MLX sampler RNG (training RNG ≠ GenMLX inference RNG; no native
   config field exists to seed it)."
  [config]
  (clj->js
    (merge
      (cond-> {}
        (contains? config :learning-rate)               (assoc :learningRate (:learning-rate config))
        (contains? config :group-size)                  (assoc :groupSize (:group-size config))
        (or (contains? config :kl-coef)
            (contains? config :beta))                   (assoc :klCoef (or (:kl-coef config) (:beta config)))
        (contains? config :loss-type)                   (assoc :lossType (get loss-type->native (:loss-type config)
                                                                             (name (:loss-type config))))
        (contains? config :max-completion-length)       (assoc :maxCompletionLength (:max-completion-length config))
        (contains? config :temperature)                 (assoc :temperature (:temperature config))
        (contains? config :top-p)                       (assoc :topP (:top-p config))
        (contains? config :top-k)                       (assoc :topK (:top-k config))
        (contains? config :clip-epsilon)                (assoc :clipEpsilon (:clip-epsilon config))
        (contains? config :gradient-clip-norm)          (assoc :gradientClipNorm (:gradient-clip-norm config))
        (contains? config :gradient-accumulation-steps) (assoc :gradientAccumulationSteps (:gradient-accumulation-steps config))
        (contains? config :repetition-penalty)          (assoc :repetitionPenalty (:repetition-penalty config))
        (contains? config :enable-thinking)             (assoc :enableThinking (:enable-thinking config))
        (contains? config :lm-head-chunk-size)          (assoc :lmHeadChunkSize (:lm-head-chunk-size config))
        (contains? config :forward-chunk-size)          (assoc :forwardChunkSize (:forward-chunk-size config)))
      (:raw config))))

(def ^:private reward-type->native
  {:length "Length" :tool-use "ToolUse" :xml-format "XmlFormat" :json-schema "JsonSchema"})

(defn- ->builtin-reward
  "Map a CLJS reward config to a native BuiltinRewardConfig (#js)."
  [{:keys [reward-type weight min-length max-length use-chars? required-tags
           allowed-tools required-fields required?]}]
  (clj->js
    (cond-> {:rewardType (get reward-type->native reward-type (name reward-type))}
      (some? weight)         (assoc :weight weight)
      (some? min-length)     (assoc :minLength min-length)
      (some? max-length)     (assoc :maxLength max-length)
      (some? use-chars?)     (assoc :useChars use-chars?)
      (some? required-tags)  (assoc :requiredTags required-tags)
      (some? allowed-tools)  (assoc :allowedTools allowed-tools)
      (some? required-fields)(assoc :requiredFields required-fields)
      (some? required?)      (assoc :required required?))))

;; Tiny default Qwen3.5 architecture for the random-checkpoint test/example helper.
(def ^:private tiny-qwen35-defaults
  {:vocabSize 256 :hiddenSize 64 :numLayers 2 :numHeads 4 :numKvHeads 2
   :intermediateSize 128 :rmsNormEps 1e-6 :headDim 16
   :tieWordEmbeddings true :attentionBias false :maxPositionEmbeddings 512
   :padTokenId 0 :eosTokenId 1 :bosTokenId 2
   :linearNumValueHeads 4 :linearNumKeyHeads 2 :linearKeyHeadDim 16 :linearValueHeadDim 16
   :linearConvKernelDim 4 :fullAttentionInterval 2 :partialRotaryFactor 0.5 :ropeTheta 10000.0})

;; ===========================================================================
;; Prompt + result marshalling.
;; ===========================================================================

(defn- ->chat-msg [m]
  (cond
    (string? m) #js {:role "user" :content m}
    (map? m)    #js {:role (name (:role m :user)) :content (:content m)}
    :else       m))

(defn- ->prompt
  "A prompt is one chat conversation. A bare string becomes a single user turn."
  [p]
  (if (string? p)
    #js [(->chat-msg p)]
    (into-array (map ->chat-msg p))))

(defn- ->prompts
  "vector of prompts -> native ChatMessage[][]."
  [prompts]
  (into-array (map ->prompt prompts)))

(defn- ->metrics
  "Native EngineStepMetrics -> CLJS metrics map."
  [m]
  {:step               (.-step m)
   :loss               (.-loss m)
   :reward-mean        (.-meanReward m)
   :reward-std         (.-stdReward m)
   :advantage-mean     (.-meanAdvantage m)
   :advantage-std      (.-stdAdvantage m)
   :total-tokens       (.-totalTokens m)
   :gradients-applied? (.-gradientsApplied m)
   :generation-ms      (.-generationTimeMs m)
   :training-ms        (.-trainingTimeMs m)
   :peak-memory-mb     (.-peakMemoryMb m)
   :active-memory-mb   (.-activeMemoryMb m)})

(defn- ->epoch-metrics [m]
  {:epoch          (.-epoch m)
   :avg-loss       (.-avgLoss m)
   :avg-reward     (.-avgReward m)
   :total-steps    (.-totalSteps m)
   :total-tokens   (.-totalTokens m)
   :epoch-time-secs (.-epochTimeSecs m)})

;; ===========================================================================
;; The Trainer — a quarantine record mirroring CljsForwardModel (backend.cljs:34).
;;   engine : the IMMUTABLE native GrpoTrainingEngine handle (the borrowed resource)
;;   model  : the IMMUTABLE borrowed native model handle (must outlive the trainer;
;;            the engine mutates ITS weights in place — see dispose!)
;;   state  : the ONE mutable cell (an atom), holding {:disposed? bool}
;; ===========================================================================

(defrecord Trainer [engine model state])

(defn trainer?
  "True iff `x` is a Trainer handle from this face."
  [x]
  (instance? Trainer x))

(defn disposed?
  "True once `trainer` has been disposed (or reset, which is terminal natively)."
  [trainer]
  (boolean (:disposed? @(:state trainer))))

(defn- ensure-live! [trainer]
  (when (disposed? trainer)
    (throw (ex-info "trainer is disposed (or reset); construct a new one"
                    {:genmlx/error :trainer-disposed}))))

;; ===========================================================================
;; Construction — blessed scope + low-level escape hatch (mirrors with-server).
;; ===========================================================================

(defn make-trainer!
  "[low-level] Build a GRPO training engine over an ALREADY-LOADED native model
   handle (`model`) and a CLJS `config` (see `->engine-config` for keys). The model
   handle must OUTLIVE the trainer — the engine borrows it and mutates its weights in
   place. `:family` selects the native factory: `:qwen35` (default) / `:qwen35-moe` /
   `:qwen3`. The caller OWNS teardown via `(dispose! trainer)`. Prefer `with-trainer`,
   which guarantees teardown."
  ([model config] (make-trainer! model config {}))
  ([model config {:keys [family] :or {family :qwen35}}]
   (when-not (available?)
     (throw (ex-info "world.train unavailable: @genmlx/core training engine absent"
                     {:genmlx/error :train-unavailable})))
   (let [cfg    (->engine-config config)
         engine (case family
                  :qwen35     (.fromQwen35 GrpoTrainingEngine model cfg)
                  :qwen35-moe (.fromQwen35Moe GrpoTrainingEngine model cfg)
                  :qwen3      (js/Reflect.construct GrpoTrainingEngine #js [model cfg])
                  (throw (ex-info (str "unknown :family " family)
                                  {:genmlx/error :bad-family :family family})))]
     (->Trainer engine model (atom {:disposed? false})))))

(defn dispose!
  "Tear down `trainer`: drop the model-thread training state (the engine's terminal
   `reset` — optimizer moments + step counter; the model thread and its already-trained
   weights stay live) so the borrowed model can host a NEW trainer, and mark this
   handle disposed so further effectful ops throw. Idempotent. Does NOT free the
   borrowed `model` (the caller owns it) or revert its weights.

   The native model thread hosts only ONE active training run, so freeing it on
   teardown is required for `with-trainer` to be re-enterable over the same model.

   A `reset` failure (e.g. the model thread already exited) is NOT raised — raising
   would mask a body error when dispose! runs inside `with-trainer`'s teardown — but it
   IS recorded as `:teardown-error` on the trainer state (read it with `teardown-error`).
   In that case the model-thread run was NOT released and a new trainer over the same
   model may be rejected."
  [trainer]
  (when-not (disposed? trainer)
    (try (.reset (:engine trainer))
         (catch :default e (swap! (:state trainer) assoc :teardown-error e))))
  (swap! (:state trainer) assoc :disposed? true)
  nil)

(defn teardown-error
  "The error captured if `dispose!`'s native `reset` failed (the model-thread training
   run was not released), else nil."
  [trainer]
  (:teardown-error @(:state trainer)))

(defn with-trainer
  "[blessed scope] Build a trainer over `model`+`config`, call `(f trainer)` — which
   MUST return a promise — and GUARANTEE the trainer is disposed afterwards, on
   success OR failure (the p/finally runs even when f's promise rejects). Returns the
   promise of `(f trainer)`'s result. This is the only place a trainer lifecycle
   should live in tests/examples."
  ([model config f] (with-trainer model config {} f))
  ([model config opts f]
   (let [trainer (make-trainer! model config opts)]
     ;; p/handle (not p/finally): under nbb a `p/finally` teardown followed by a
     ;; downstream `p/catch` double-settles — the catch handler runs yet the promise
     ;; stays rejected. p/handle disposes on BOTH arms and re-raises exactly once.
     (-> (p/let [r (f trainer)] r)
         (p/handle (fn [r e] (dispose! trainer) (if e (throw e) r)))))))

;; ===========================================================================
;; THE ONE EFFECT — a GRPO step (the reward bridge to Phase 1 lives here).
;; ===========================================================================

(defn train-step!
  "Run one GRPO training step (the training `eval!`-equivalent — the sole
   weight-mutating effect of this face) as an EXPLICIT generate -> score -> train
   round-trip:
     1. the engine generates `group-size` completions per prompt,
     2. the PURE `reward-fn` scores each completion in CLJS (the reward bridge),
     3. the engine computes group-relative advantages + a clipped surrogate and
        applies an AdamW update IN PLACE on the model thread.

     trainer   : a Trainer (from make-trainer!/with-trainer)
     prompts   : vector of prompt strings, or vectors of {:role :content} chat maps
     reward-fn : a PURE (fn [prompt completion-text] -> number), called once per
                 completion across all prompts*group-size rollouts. THE Phase-1 seam:
                 swap in a GFI-quantity scorer (msa/score-model, codegen/verify) with
                 no change to this bridge.

   Returns a promesa promise of a metrics map: {:step :loss :reward-mean :reward-std
   :advantage-mean :advantage-std :total-tokens :gradients-applied? :completions
   :rewards :completion-lengths :generation-ms :training-ms :peak-memory-mb
   :active-memory-mb}.

   This composes generateBatchForTraining + trainStepWithGenerations rather than the
   all-in-one native trainStepAuto: it keeps the GFI reward scorer in CLJS between the
   two awaits (no ThreadsafeFunction marshalling) and trains UNCONDITIONALLY — the
   trainStepAuto path additionally filters out `finish_reason='length'` completions as
   an OOM guard, which would skip the step for a model whose rollouts never emit EOS."
  [trainer prompts reward-fn]
  (ensure-live! trainer)
  (let [engine     (:engine trainer)
        js-prompts (->prompts prompts)
        n-prompts  (count prompts)
        gen-start  (.now js/Date)]
    (-> (.generateBatchForTraining engine js-prompts)
        (p/then
          (fn [gen]
            (let [texts   (vec (.-completionTexts gen))
                  g       (quot (count texts) (max 1 n-prompts))
                  ;; prompt-major: completion i belongs to prompt (i // group-size)
                  rewards (mapv (fn [i] (double (reward-fn (nth prompts (quot i g))
                                                           (nth texts i))))
                                (range (count texts)))
                  gen-ms  (- (.now js/Date) gen-start)]
              (-> (.trainStepWithGenerations engine js-prompts (clj->js rewards) gen)
                  (p/then (fn [m]
                            (assoc (->metrics m)
                                   :generation-ms gen-ms
                                   :completions texts
                                   :rewards rewards
                                   :completion-lengths (vec (.-completionLengths gen))))))))))))

(defn register-builtin-reward!
  "Register a NATIVE built-in reward on the engine (e.g. {:reward-type :length
   :max-length 64 :use-chars? true}). `:reward-type` ∈ #{:length :tool-use
   :xml-format :json-schema}. An alternative to the CLJS reward bridge for cheap
   length/format rewards."
  [trainer reward-config]
  (ensure-live! trainer)
  (.registerBuiltinReward (:engine trainer) (->builtin-reward reward-config))
  nil)

;; ===========================================================================
;; Lifecycle passthroughs (native GrpoTrainingEngine methods/getters).
;; ===========================================================================

(defn step
  "Current global step count (native getter)."
  [trainer]
  (.-step (:engine trainer)))

(defn epoch
  "Current epoch (native getter)."
  [trainer]
  (.-epoch (:engine trainer)))

(defn start-epoch!
  "Mark the start of a training epoch."
  [trainer]
  (ensure-live! trainer)
  (.startEpoch (:engine trainer))
  nil)

(defn end-epoch!
  "Mark the end of a training epoch (`epoch-secs` = wall time in seconds). Returns a
   CLJS epoch-metrics map."
  [trainer epoch-secs]
  (ensure-live! trainer)
  (->epoch-metrics (.endEpoch (:engine trainer) epoch-secs)))

(defn generate-batch
  "Generate completions for `prompts` WITHOUT any weight update (inspection only).
   Returns a promise of a vector of completion strings."
  [trainer prompts]
  (ensure-live! trainer)
  (-> (.generateBatch (:engine trainer) (->prompts prompts))
      (p/then vec)))

;; ===========================================================================
;; Checkpointing — OPTIMIZER state only (the engine's resumable AdamW moments +
;; step). Model WEIGHTS are checkpointed on the model handle (its own `saveModel`),
;; NOT here — the native engine exposes no weight save.
;; ===========================================================================

(defn save-optimizer-state!
  "Persist the AdamW optimizer moments + step to `path` (NOT model weights — save
   those via the model handle's own `saveModel`). Returns a promise. No-op natively
   if the engine uses SGD."
  [trainer path]
  (ensure-live! trainer)
  (.saveOptimizerState (:engine trainer) path))

(defn load-optimizer-state!
  "Restore AdamW optimizer moments + step from `path`. Returns a promise."
  [trainer path]
  (ensure-live! trainer)
  (.loadOptimizerState (:engine trainer) path))

;; ===========================================================================
;; Test/example helper — spin up a tiny RANDOM Qwen3.5 checkpoint so the membrane
;; round-trip is exercisable without a real >3GB checkpoint (real-checkpoint
;; training is gated by genmlx-o94r). Load the result with Qwen35Model.load.
;; ===========================================================================

(defn random-qwen35-checkpoint!
  "[test/example helper] Write a RANDOM tiny Qwen3.5 dense checkpoint to directory
   `dir`. `config` is a CLJS map of Qwen35Config fields (camelCase keyword keys, e.g.
   {:numLayers 2 :hiddenSize 64}); it is merged over a tiny default architecture.
   Returns a promise of `dir`. Load it with `(.load Qwen35Model dir)` to get a
   trainable model handle."
  ([dir] (random-qwen35-checkpoint! dir {}))
  ([dir config]
   (-> (create-random-qwen35-checkpoint (clj->js (merge tiny-qwen35-defaults config)) dir)
       (p/then (fn [_] dir)))))
