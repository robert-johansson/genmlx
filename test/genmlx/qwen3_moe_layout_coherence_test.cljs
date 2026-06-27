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
     export GENMLX_VLM_MODEL=/path/to/Qwen3.6-35B-A3B-4bit/snapshots/<hash>   # optional; has default
     export GENMLX_MOE_MODEL=/path/to/Qwen3-Coder-Next-4bit/snapshots/<hash>  # optional; has default
     bunx --bun nbb@1.4.208 -cp src:test:... test/genmlx/qwen3_moe_layout_coherence_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [promesa.core :as pr]
            ["fs" :as fs]))

(defn- mat [a] (mx/materialize! a) a)
(defn- greedy [l] (mx/item (mx/argmax l)))

(def cases
  [{:name "35B qwen3_5_moe (merged in_proj_qkv/z -> fused_qkvz_layout=false)"
    :dir (or (some-> js/process .-env .-GENMLX_VLM_MODEL)
             (str "/home/robert/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
                  "/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46"))
    ;; "What is the capital of France? ..." (chat-templated, <think> mode). mlx_vlm oracle.
    :input-ids [248045 846 198 3710 369 279 6511 314 9338 30 21134 303 799 11316 13
                248046 198 248045 74455 198 248068 198]
    :oracle    [8160 579 264 7047 1817 25 271 16]}
   {:name "80B qwen3_next coder (native interleaved in_proj_qkvz -> fused_qkvz_layout=true)"
    :dir (or (some-> js/process .-env .-GENMLX_MOE_MODEL)
             (str "/home/robert/code/mlx/models/Qwen3-Coder-Next-4bit"
                  "/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5"))
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

(pr/let [;; sequential (not parallel) to keep peak memory bounded
         r0 (run-case (nth cases 0))
         r1 (run-case (nth cases 1))
         results [r0 r1]
         checked (vec (remove #(= :skip %) results))
         passed (count (filter true? checked))]
  (println (str "\n== qwen3_moe_layout_coherence: " passed "/" (count checked) " checked passed, "
                (count (filter #(= :skip %) results)) " skipped =="))
  (js/process.exit (if (every? true? checked) 0 1)))
