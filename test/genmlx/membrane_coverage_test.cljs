;; @tier fast
(ns genmlx.membrane-coverage-test
  "genmlx-0vwn / genmlx-e3jg: compute-membrane coverage drift-guard.

   The @genmlx/core surface is the substrate (Layer C) the GenMLX membrane
   (Layer 0 = mlx.cljs + mlx/random.cljs) binds to. This test PARTITIONS the
   full runtime function-export surface into:

     WRAPPED              — bound somewhere in the membrane source, AND
     INTENTIONAL-OMISSIONS — every other export, each with an honest reason

   and asserts the partition TILES the surface in BOTH directions:

     A. nothing unaccounted  — a new/unwrapped export not on the allowlist FAILS
                               (re-audit: wire it, or add an omission entry)
     B. no stale omissions   — an allowlisted name deleted upstream FAILS
                               (the f6ov/dbce rebase tax surfacing as a diff)
     C. clean partition      — an omission that became wrapped FAILS (reclassify)

   This is the real drift guard the bare export-count pin could not be: a count
   only flags THAT the surface moved; the partition pinpoints WHAT moved and what
   to do about it. The count is kept as a coarse canary (catches an add+omit done
   in one commit, which A/B/C alone would pass).

   SoT for the surface = the LIVE @genmlx/core runtime module (what `core` below reads
   via js/require, and what this test partitions). The package ships no `types` field;
   its `index.d.ts` is a regenerated snapshot that can lag the runtime (e.g. `Gradients`
   is a live export absent from the d.ts), so the runtime — not any .d.ts — is authority.
   NO no-op stubs: an export is either genuinely wrapped or honestly omitted with a
   category + reason."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]))

(def ^:private core (js/require "@genmlx/core"))
(def ^:private fs (js/require "fs"))

;; The membrane = the Layer-0 files that bind @genmlx/core: the pure COMPUTE
;; membrane (mlx.cljs + mlx/random.cljs) plus the TRAINING membrane face
;; (world/train.cljs, genmlx-zftr — the GRPO engine behind its mutable-state
;; quarantine). An export is WRAPPED iff its native name is referenced in one of
;; these as a property (.-name c) or a method call (.name c …).
(def ^:private membrane-src
  (str (.readFileSync fs "src/genmlx/mlx.cljs" "utf8")
       "\n"
       (.readFileSync fs "src/genmlx/mlx/random.cljs" "utf8")
       "\n"
       (.readFileSync fs "src/genmlx/world/train.cljs" "utf8")))

(defn- referenced?
  "True if the membrane BINDS native export `nm` off the @genmlx/core object
   `c` — as a property ref (.-nm c) or a method head (.nm c …), the uniform wrap
   convention in both Layer-0 files (each `(defonce ^:private c (js/require ...))`).
   Scoping to the `c` receiver (rather than any object) is deliberate: it stops a
   phantom match on a same-named method read off a DIFFERENT object — e.g.
   .valueAndGrad / .vmap / .computeGradients on the MxArray module `M`, or
   .gcAndSweep on bun:jsc — none of which bind a core export. (Verified: the
   c-scoped and unscoped predicates select the identical wrapped set today; the
   scoping only forecloses a future upstream rename TO one of those phantom names
   silently classifying as wrapped.) The trailing non-identifier guard stops a
   short name matching as a substring of a longer export (e.g. `sum` in
   `.logsumexp`).

   Known limitation (genmlx-0vwn, accepted): a 'dangling wrap' — a (.-x c) left in
   the source after upstream DELETES export x — is not flagged here directly; the
   223 count canary + tiling invariant in coverage-accounting-test catch it
   indirectly (a deletion drops the surface count below the pin)."
  [nm]
  (some? (re-find (re-pattern (str "\\.-?" nm "\\s+c[^A-Za-z0-9_]")) membrane-src)))

;; ---------------------------------------------------------------------------
;; INTENTIONAL OMISSIONS — the coverage matrix, machine-checked.
;; Each entry is an @genmlx/core export deliberately NOT wrapped in the pure
;; compute membrane, with the reason it belongs elsewhere (or nowhere). Classes
;; appear here too: a JS `class` is a function export, so it counts toward the
;; runtime surface and must be accounted for. Categories double as the "where
;; does a new export of this kind belong" map for the next re-audit.
;; ---------------------------------------------------------------------------
(def ^:private intentional-omissions
  {;; native op is INCORRECT — a custom reconstruction is used instead
   :broken
   #{"broadcastTo"}            ; mis-fills size-1 dims; mx/broadcast-to reconstructs (guarded below)

   ;; profiling instrumentation — wire-on-demand behind the i0s4 cost meter
   :profiling-gated-i0s4
   #{"getProfilingData" "isProfilingEnabled" "setProfilingEnabled" "resetProfilingData"}

   ;; Native training surface. The GRPO engine + the random-Qwen3.5 checkpoint helper
   ;; are now WRAPPED in genmlx.world.train (genmlx-zftr Phase 0) behind the
   ;; mutable-state quarantine — never the pure compute membrane. The remainder bind
   ;; incrementally as Phase 1-4 tap them (SFT engine, reward registry, result/
   ;; persistence types, the Qwen3/MoE random-checkpoint helpers). (genmlx-706r)
   :training-orchestration
   #{"buildRewardOutputs"
     "createRandomQwen3Checkpoint" "createRandomQwen35MoeCheckpoint"
     "SftTrainingEngine" "NativeRewardRegistry"
     "Gradients" "OutputStore" "ResponseStore"}

   ;; offline weight/format conversion tooling — not graph ops
   :model-conversion
   #{"convertForeignWeights" "convertGgufToSafetensors" "convertModel" "convertParquetToJsonl"}

   ;; OCR / document-pipeline classes — GenMLX does not use these; its VLM path
   ;; (llm/vision.cljs) routes through the Qwen VL model classes below, not the
   ;; PaddleOCR / document-understanding pipeline.
   :ocr-vlm-document
   #{"createPaddleocrVlConfig" "createQianfanOcrConfig"
     "documentToXlsx" "formatDocument" "saveToXlsx"
     "parsePaddleResponse" "parseToolCallsFromText" "parseVlmOutput"
     "DocLayoutModel" "DocOrientationModel" "DocUnwarpModel" "PrivacyFilterModel"
     "QianfanOCRModel" "TextDetModel" "TextRecModel"
     "VLModel" "VlmChatResult"}

   ;; loaded-model / tokenizer / generation classes — bound directly via @genmlx/core
   ;; native classes in llm/backend.cljs (load-upstream-model + Qwen3Tokenizer +
   ;; chatSessionStart) and llm/vision.cljs (bean genmlx-qt34), the LLM orchestration
   ;; boundary, not the pure compute membrane. The per-token GFI path reaches the
   ;; low-level forward, not these high-level classes.
   :llm-orchestration
   #{"Gemma4Model" "HarrierModel" "Lfm2Model" "Qwen3Model" "Qwen35Model" "Qwen35MoeModel"
     "Qwen3Tokenizer"
     "BatchGenerationResult" "GenerationResult" "ChatStreamHandle"}

   ;; a foreign tensor class distinct from MxArray (the membrane's value type)
   :foreign-tensor-type
   #{"Tensor"}

   ;; native MLX graph-caching compile — DELIBERATELY bypassed. The membrane's
   ;; compile-fn (mlx.cljs) is an identity pass-through: GenMLX compilation uses
   ;; noise transforms + the expression compiler (Level 1), not MLX's compile
   ;; (which would sever the autograd tape across model-body eval!). vmap — the
   ;; sibling transform relocated alongside it into @genmlx/core — IS wrapped.
   :compile-strategy-bypass
   #{"compileFn"}

   ;; quantized QMV microbenchmark — a perf-measurement entry point exposed by
   ;; the CUDA genmlx-core build, not a graph op the membrane wraps (genmlx-0vwn
   ;; surface re-pin for the Jetson Thor / CUDA port).
   :benchmark-microbench
   #{"quantizedQmvMicrobench"}

   ;; seeds the CALLING napi thread's MLX default RNG (thread-local in this
   ;; fork). Not a membrane concern: GenMLX's inference PRNG is keyed
   ;; (mlx/random.cljs), and reproducible TRAINING generation rides the
   ;; GrpoEngineConfig `seed` field instead (applied on the model thread at
   ;; InitTraining — a caller-thread seed can never reach the training
   ;; sampler). Kept as a primitive for main-thread unkeyed draws
   ;; (MxArray.categorical) in probes/tests (genmlx-at2q).
   :calling-thread-rng
   #{"seedGlobalRng"}})

(def ^:private omitted (reduce into #{} (vals intentional-omissions)))

(def ^:private exported-fns
  (set (filter #(fn? (aget core %)) (js-keys core))))

;; ---------------------------------------------------------------------------
;; Op-correctness oracles for the wired ops (genmlx-0vwn)
;; ---------------------------------------------------------------------------
(defn- i32 [a] (mx/->clj (mx/astype a mx/int32)))

(deftest new-ops-correctness-test
  (testing "the newly-wired pure ops produce correct values (closed-form / Math oracles)"
    ;; log-softmax([1,2,3]) = [1,2,3] - logsumexp = [-2.4076, -1.4076, -0.4076]
    (is (h/close? -2.40760 (first (mx/->clj (mx/log-softmax (mx/array [1.0 2.0 3.0])))) 1e-4))
    ;; logical truth tables
    (is (= [1 0 0 0] (i32 (mx/logical-and (mx/array [1 1 0 0]) (mx/array [1 0 1 0])))))
    (is (= [1 1 1 0] (i32 (mx/logical-or (mx/array [1 1 0 0]) (mx/array [1 0 1 0])))))
    (is (= [1 0] (i32 (mx/logical-not (mx/array [0 1])))))
    ;; isfinite: 1 for finite, 0 for inf
    (is (= [1 0] (i32 (mx/isfinite (mx/array [1.0 ##Inf])))))
    ;; cumprod / roll
    (is (= [1 2 6 24] (mx/->clj (mx/cumprod (mx/array [1.0 2.0 3.0 4.0]) 0))))
    (is (= [3 0 1 2] (mx/->clj (mx/roll (mx/array [0.0 1.0 2.0 3.0]) 1))))
    ;; pad: [1,2,3] with 1-before/1-after, const 0 → [0,1,2,3,0]
    (is (= [0 1 2 3 0] (mx/->clj (mx/pad (mx/array [1.0 2.0 3.0]) [1 1] 0.0))))
    ;; trig / hyperbolic vs Math
    (is (h/close? 0.0 (mx/item (mx/sinh (mx/scalar 0.0))) 1e-5))
    (is (h/close? 1.0 (mx/item (mx/cosh (mx/scalar 0.0))) 1e-5))
    (is (h/close? (/ js/Math.PI 6) (mx/item (mx/arcsin (mx/scalar 0.5))) 1e-5))
    (is (h/close? (/ js/Math.PI 4) (mx/item (mx/arctan (mx/scalar 1.0))) 1e-5))))

(deftest wired-introspection-test
  (testing "the newly-wired memory introspection + Metal barrier (read-only / effectful)"
    (is (number? (mx/get-memory-limit)) "get-memory-limit returns a byte count")
    (is (number? (mx/get-wired-limit)) "get-wired-limit returns a byte count")
    (let [s (mx/memory-stats)]
      (is (every? #(contains? s %) [:active :peak :cache :wired-limit])
          "memory-stats keys are kebab-cased from MemoryStats")
      (is (every? number? (vals s)) "memory-stats values are numbers")
      (is (>= (:active s) 0)))
    (let [snap (mx/get-memory-snapshot)]
      (is (every? #(contains? snap %) [:active-bytes :peak-bytes :cache-bytes])
          "get-memory-snapshot keys are kebab-cased from GpuMemorySnapshot")
      (is (>= (:active-bytes snap) 0)))
    ;; synchronize() returns void → nil; the barrier must not throw
    (is (nil? (mx/synchronize!)) "synchronize! is a void Metal barrier")
    ;; gpu-architecture-gen: the real M-generation integer (e.g. 16 = M4) on
    ;; Metal; nil off-Metal — never an Apple-parse of a CUDA arch string
    ;; (genmlx-yjyl honesty gate).
    (if (mx/metal-is-available?)
      (do (is (number? (mx/gpu-architecture-gen)) "gpu-architecture-gen returns an integer on Metal")
          (is (>= (mx/gpu-architecture-gen) 0)))
      (is (nil? (mx/gpu-architecture-gen)) "gpu-architecture-gen is nil off-Metal (no fabricated gen)"))))

(deftest broadcast-to-omission-test
  (testing "native broadcastTo is OMITTED (broken: mis-fills size-1 dims); the custom broadcast-to stays"
    (let [src (.readFileSync fs "src/genmlx/mlx.cljs" "utf8")]
      (is (nil? (re-find #"\.-broadcastTo c" src)) "mlx.cljs does NOT wrap native (.-broadcastTo c)")
      (is (nil? (re-find #"\.broadcastTo " src)) "mlx.cljs does NOT call native (.broadcastTo ...)")
      (is (some? (re-find #"defn broadcast-to" src)) "the custom broadcast-to reconstruction is present")
      (is (contains? (:broken intentional-omissions) "broadcastTo")
          "broadcastTo is on the :broken omission allowlist"))))

;; ---------------------------------------------------------------------------
;; The drift guard proper — the e3jg/0vwn floor deliverable.
;; ---------------------------------------------------------------------------
(deftest coverage-partition-test
  (testing "every @genmlx/core export is WRAPPED ∪ INTENTIONAL-OMISSIONS (two-directional)"
    ;; A — nothing unaccounted: an export neither wrapped nor on the allowlist
    (let [unaccounted (sort (remove referenced? (remove omitted exported-fns)))]
      (is (empty? unaccounted)
          (str "Unaccounted @genmlx/core exports — wire them into the membrane "
               "or add to intentional-omissions with a reason (genmlx-0vwn): "
               (vec unaccounted))))
    ;; B — no stale omissions: an allowlisted name no longer exported upstream
    (let [stale (sort (remove exported-fns omitted))]
      (is (empty? stale)
          (str "Stale omissions — these were deleted from @genmlx/core (the "
               "rebase tax surfacing); remove them from the allowlist: " (vec stale))))
    ;; C — clean partition: an omission that is now actually wrapped
    (let [misclassified (sort (filter referenced? omitted))]
      (is (empty? misclassified)
          (str "These are on the omission allowlist but ARE referenced in the "
               "membrane now — they got wired; reclassify as WRAPPED: " (vec misclassified))))))

(deftest coverage-accounting-test
  (testing "the partition tiles the full surface (wrapped ⊎ omitted = exports)"
    (let [wrapped (filter referenced? exported-fns)]
      ;; Coarse canary: catches a surface change even when add+omit happen together.
      (is (= 225 (count exported-fns))
          (str "@genmlx/core surface size changed: " (count exported-fns)
               " fns (pinned at 225) — the partition test above pinpoints what moved."))
      (is (= 48 (count omitted))
          (str "intentional-omissions size changed: " (count omitted) " (pinned at 48)."))
      (is (= (count exported-fns) (+ (count wrapped) (count omitted)))
          (str "partition must tile exactly: wrapped " (count wrapped)
               " + omitted " (count omitted) " = exports " (count exported-fns))))))

(cljs.test/run-tests)
