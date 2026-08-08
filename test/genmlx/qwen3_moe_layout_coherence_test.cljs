;; @tier heavy
(ns genmlx.qwen3-moe-layout-coherence-test
  "Dual-checkpoint CUDA coherence guard (genmlx-6ijc / mlx-cft4).

   Both the 35B Qwen3.6-A3B VLM (model_type=qwen3_5_moe) and the 80B Qwen3-Coder-Next
   (model_type=qwen3_next) load through the SAME Qwen3_5MoeModel Rust path, but ship
   OPPOSITE GatedDeltaNet in_proj layouts and therefore need OPPOSITE fused_qkvz_layout:

     - 35B: ships SEPARATE in_proj_qkv/z which merge_split_projections concatenates into
            a CONTIGUOUS in_proj_qkvz -> fused_qkvz_layout = FALSE (do NOT de-interleave).
     - 80B: ships a NATIVE per-key-head INTERLEAVED in_proj_qkvz                -> TRUE.

   A single hardcoded constant silently scrambles one model's 36 GDN layers (commit
   190feb0 hardcoded FALSE -> fixed the 35B, scrambled the 80B; repaired in 1b03b4a by
   gating on native-vs-merged). This test pins each model's GREEDY continuation, verified
   argmax-exact to the Python oracle on the SAME checkpoint (mlx_vlm for the 35B, mlx_lm
   qwen3_next for the 80B), so the gate can never re-break either family unnoticed. Unlike
   tier2_branch_roundtrip / paged_vs_flat (self-consistency checks that pass even if every
   layer is scrambled), this asserts COHERENCE against an external oracle.

   Heavy + CUDA-only (loads the 18GB + 42GB MoEs). Each case is gated on its model dir
   existing and skips cleanly otherwise (mirrors llm_branched_test). Ids are pinned
   directly (no chat template) so the assertion is template-independent.

   Run:
     bunx --bun nbb@1.4.208 -cp src:test:... test/genmlx/qwen3_moe_layout_coherence_test.cljs

   Model dirs resolve as: env override -> the CURRENT user's HF hub cache at the
   PINNED snapshot revision (see hub-snapshot). Revision-locked on purpose — the
   :oracle ids are only valid for that exact snapshot.
     export GENMLX_VLM_MODEL=/path/to/Qwen3.6-35B-A3B-4bit/snapshots/<hash>   # optional
     export GENMLX_MOE_MODEL=/path/to/Qwen3-Coder-Next-4bit/snapshots/<hash>  # optional

   By DEFAULT only case 0 (35B, ~20GB) runs; case 1 (80B, ~42GB) prints a skip.
   The backend has no model unload and GC does not promptly free native weights,
   so both-in-one-process peaks at ~62GB resident — enough to destabilize a
   122GB box under other load (2026-07-07 reboot). This file is `@tier heavy`,
   which run.sh always runs 1-way, so the tier protects against CROSS-file
   overlap; this flag protects against loading both models WITHIN one process.
   To cover the 80B:
     GENMLX_COHERENCE_CASE=1 ...   # PREFERRED: one model per process
     GENMLX_COHERENCE_BOTH=1 ...   # both in one process, at your own risk

   A case whose model dir is absent skips, and if NOTHING was checked the file
   emits the `SKIP` + `Ran 0 tests` anchor pair so run.sh scores it SKIP rather
   than PASS. That mattered: until 2026-08-08 the defaults were absolute paths
   under another machine's home (/home/robert/...), so on this host both cases
   skipped, no anchor was printed, and the battery scored a guard that verified
   nothing as green (genmlx-pc9o)."
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]))

(defn- mat [a] (mx/materialize! a) a)
(defn- greedy [l] (mx/item (mx/argmax l)))

;; ---------------------------------------------------------------------------
;; Per-host model resolution (genmlx-pc9o)
;;
;; The defaults below used to be absolute paths under `/home/robert/...` — a
;; DIFFERENT machine's home (username `robert`, not `robertj`; the same layout
;; `docs/fork/SYNC-RUNBOOK.md` uses for its GENMLX var). On every other host
;; both cases therefore skipped, and because this is a hand-rolled harness with
;; no cljs.test summary, run.sh's SKIP classifier — which anchors on
;; `Ran 0 tests` — could not see the skip and scored the file PASS. So the
;; battery counted a coherence guard that verified NOTHING. Measured
;; 2026-08-08: both cases skipped, exit 0, no `skipped` column in the tier tally.
;;
;; Resolution order is env override -> the current user's HF hub cache at the
;; PINNED revision. Revision-locked deliberately: the `:oracle` ids below are
;; only valid for that exact snapshot, so this must never follow a symlink that
;; a later `hf download` could re-point (which is why it does NOT use the
;; ~/.cache/models symlink farm, convenient though that would be).
;; ---------------------------------------------------------------------------

(defn- hub-snapshot
  "`~/.cache/huggingface/hub/models--<repo>/snapshots/<rev>` for the CURRENT user."
  [repo rev]
  (str (.homedir os) "/.cache/huggingface/hub/models--" repo "/snapshots/" rev))

(def cases
  [{:name "35B qwen3_5_moe (merged in_proj_qkv/z -> fused_qkvz_layout=false)"
    :dir (or (some-> js/process .-env .-GENMLX_VLM_MODEL)
             (hub-snapshot "mlx-community--Qwen3.6-35B-A3B-4bit"
                           "38740b847e4cb78f352aba30aa41c76e08e6eb46"))
    ;; "What is the capital of France? ..." (chat-templated, <think> mode). mlx_vlm oracle.
    :input-ids [248045 846 198 3710 369 279 6511 314 9338 30 21134 303 799 11316 13
                248046 198 248045 74455 198 248068 198]
    :oracle    [8160 579 264 7047 1817 25 271 16]}
   {:name "80B qwen3_next coder (native interleaved in_proj_qkvz -> fused_qkvz_layout=true)"
    :dir (or (some-> js/process .-env .-GENMLX_MOE_MODEL)
             (hub-snapshot "mlx-community--Qwen3-Coder-Next-4bit"
                           "7b9321eabb85ce79625cac3f61ea691e4ea984b5"))
    ;; "Write a one-line Python function to add two numbers." (chat-templated). mlx_lm oracle.
    :input-ids [151644 872 198 7985 264 825 8447 13027 729 311 912 1378 5109 13
                151645 198 151644 77091 198]
    :oracle    [73594 12669 198 718 284 12459 264 11 293 25 264 488 293 198]}])

(defn run-case [{:keys [name dir input-ids oracle]}]
  (if-not (.existsSync fs (str dir "/config.json"))
    (do (println "SKIP" name "— model dir not found:" dir)
        (pr/resolved :skip))
    (pr/let [{:keys [model]} (llm/load-model dir)
             _ (llm/init-cache! model)
             l0 (mat (llm/forward-prefill model (vec input-ids)))
             n (count oracle)
             ids (loop [i 0 l l0 acc []]
                   (if (>= i n)
                     acc
                     (let [t (greedy l)]
                       (recur (inc i) (mat (llm/forward-step model t)) (conj acc t)))))
             ok (= (vec ids) (vec oracle))]
      (println (if ok "✓ PASS" "✗ FAIL") name)
      (println "   got   " (vec ids))
      (println "   oracle" (vec oracle))
      (pr/resolved ok))))

(def ^:private case-select
  ;; One process per case bounds peak memory to one model (no unload API).
  (some-> js/process .-env .-GENMLX_COHERENCE_CASE js/parseInt))

(def ^:private allow-both?
  "Opt-in for loading BOTH models in one process (~62GB resident, 20 + 42).
   Off by default: the backend has no unload and GC does not promptly free
   native weights (see the ns docstring's 2026-07-07 reboot). `@tier heavy`
   already guarantees no OTHER test file is resident alongside this one; this
   flag is the remaining within-process guard. The 80B's omission is PRINTED as
   a skip, never hidden — cover it with GENMLX_COHERENCE_CASE=1."
  (some? (some-> js/process .-env .-GENMLX_COHERENCE_BOTH)))

(defn- emit-nothing-checked!
  "Emit the anchor pair run.sh needs to score this file SKIP instead of PASS.
   Per test/TESTING.md the classifier wants a `SKIP` line within three lines
   above a `Ran 0 tests` summary — but that path is only reached for cljs.test
   files, and this is a hand-rolled harness. Without the anchor an all-skipped
   run fell through to `else status=PASS`, so a coherence guard that verified
   NOTHING was tallied green (measured 2026-08-08). Both lines are true: zero
   cases ran."
  [reason]
  (println (str "SKIP qwen3_moe_layout_coherence — " reason))
  (println "Ran 0 tests containing 0 assertions."))

;; set! exitCode, never (js/process.exit 1): process.exit truncates buffered
;; stdout, which would cut the very summary the failure report needs.
(if (and (some? case-select) (not (js/isNaN case-select)))
  (pr/let [r (run-case (nth cases case-select))]
    (println (str "\n== qwen3_moe_layout_coherence case " case-select ": "
                  (if (= :skip r) "skipped" (if r "passed" "FAILED")) " =="))
    (cond
      (= :skip r)        (emit-nothing-checked!
                          (str "case " case-select " model dir absent on this host"))
      (not (true? r))    (set! (.-exitCode js/process) 1)))
  (pr/let [;; sequential, and case 1 gated — see allow-both?
           r0 (run-case (nth cases 0))
           r1 (if allow-both?
                (run-case (nth cases 1))
                (do (println "SKIP" (:name (nth cases 1))
                             "— not run by default (both models resident is ~62GB and the"
                             "backend has no unload). Set GENMLX_COHERENCE_BOTH=1, or better,"
                             "run it alone with GENMLX_COHERENCE_CASE=1.")
                    :skip))
           results [r0 r1]
           checked (vec (remove #(= :skip %) results))
           passed (count (filter true? checked))]
    (println (str "\n== qwen3_moe_layout_coherence: " passed "/" (count checked) " checked passed, "
                  (count (filter #(= :skip %) results)) " skipped =="))
    (cond
      ;; MUST precede the every? branch: (every? true? []) is vacuously TRUE,
      ;; which is exactly how zero-cases-checked used to exit 0.
      (empty? checked)              (emit-nothing-checked!
                                     "no case had its model dir present on this host")
      (not (every? true? checked))  (set! (.-exitCode js/process) 1))))
