;; @tier exclude — loads Ornith-1.0-35B TWICE (owned + native upstream; ~40-80 GB
;; resident depending on quant). Run manually on the Thor:
;;   bunx --bun nbb@1.4.208 test/genmlx/llm_qwen35_moe_forward_parity_test.cljs
;;   ORNITH_QUANT=4bit bunx --bun nbb@1.4.208 test/genmlx/llm_qwen35_moe_forward_parity_test.cljs
(ns genmlx.llm-qwen35-moe-forward-parity-test
  "genmlx-g6vk GATE (Ornith Phase 3): parity of the GenMLX-owned qwen3_5_moe
   TEXT forward — the qwen35 hybrid stack with the per-layer sparse-MoE MLP
   branch, packed experts driven by mx/gather-qmm — against the LIVE native
   upstream forward on the real Ornith-1.0-35B checkpoint.

   Follows llm_qwen35_forward_parity_test's shape. Spec §9 adjustments:
   (a) top-5-ids EXACT is asserted even though the checkpoint is quantized —
       the owned expert path drives the SAME gather_qmm kernel as native, so
       expert numerics match; the dense (attn/GDN/router/shared) projections
       are dequantized-bf16 vs native fused-quantized kernels, which CAN split
       bf16-ULP ties — if the exact gate fails there, the result is REPORTED
       (band asserted) so the divergence is visible, not silently skipped.
   (b) routing parity: a PURE-OP reference MoE block (host-side top-k, per-
       expert dense matmul over natively-dequantized expert tensors — an
       independent code path from moe-mlp's argsort/gather-qmm) is compared
       against q35/moe-mlp on layer 0's real weights. A wrong expert set,
       missing renorm, or mis-gated shared expert diverges immediately.
   (c) NEVER gates on sampled text (native MoE forward jitters on CUDA,
       spec §9.1 / genmlx-ba06): all comparisons are logit-level.

   ORNITH_QUANT env selects the checkpoint (default 8bit — the parity target;
   4bit is the fast iteration target). Skips cleanly when absent."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.forward :as fwd]
            [genmlx.llm.qwen35-forward :as q35]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))
(defn assert= [label expected actual]
  (assert-true (str label " (=" (pr-str expected) ")") (= expected actual)))

(def quant (or (.-ORNITH_QUANT js/process.env) "8bit"))
(def model-dir
  (let [base (str (.-HOME js/process.env)
                  "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-"
                  quant "/snapshots")]
    (when (.existsSync fs base)
      (str base "/" (first (js->clj (.readdirSync fs base)))))))

(def prompt "The capital of France is")

(defn- log-softmax [logits] (mx/subtract logits (mx/logsumexp logits)))

(defn- topk-ids [lp k]
  (mx/eval! lp)
  (let [f32 (.toFloat32 lp)]
    (->> (range (.-length f32))
         (map (fn [i] [i (aget f32 i)]))
         (sort-by second >)
         (take k)
         (mapv first))))

(defn- max-abs-diff [a-lp b-lp ids]
  (mx/eval! a-lp) (mx/eval! b-lp)
  (let [a (.toFloat32 a-lp) b (.toFloat32 b-lp)]
    (reduce max (map (fn [i] (js/Math.abs (- (aget a i) (aget b i)))) ids))))

;; ---------------------------------------------------------------------------
;; Pure-op reference MoE block (independent of moe-mlp's code path)
;; ---------------------------------------------------------------------------

(defn- pure-moe-reference
  "Reference qwen3_5_moe MLP for layer prefix p (e.g. '...layers.0.mlp.'):
   host-side top-k over an f32 softmax of the router, per-expert DENSE matmul
   over natively-dequantized expert tensors, renormalized weighted sum, plus
   the sigmoid-gated shared expert. hn [1 T hidden] -> [T hidden]."
  [{:keys [hidden n-active expert-qz]} w p hn T]
  (let [x     (mx/reshape hn [T hidden])
        gate-w (get w (str p "gate.weight"))
        probs (mx/softmax (mx/astype (mx/matmul x (mx/transpose gate-w)) mx/float32) -1)
        pv    (mx/->clj probs)
        dq    (fn [proj] (mx/dequantize (get w (str p "switch_mlp." proj ".weight"))
                                        (get w (str p "switch_mlp." proj ".scales"))
                                        (get w (str p "switch_mlp." proj ".biases"))
                                        {:bits (:bits expert-qz)
                                         :group-size (:group-size expert-qz)}))
        gate-dq (dq "gate_proj") up-dq (dq "up_proj") down-dq (dq "down_proj")
        sh-w  (fn [s] (get w (str p "shared_expert." s)))
        rows
        (vec
         (for [t (range T)]
           (let [row  (nth pv t)
                 topk (take n-active (sort-by second > (map-indexed vector row)))
                 wsum (reduce + (map second topk))
                 xt   (mx/take-idx x (mx/array [t] [1] mx/int32) 0) ; [1 hidden]
                 moe  (reduce
                       (fn [acc [e pe]]
                         (let [ge (mx/index gate-dq e)
                               ue (mx/index up-dq e)
                               de (mx/index down-dq e)
                               h  (mx/multiply (mx/silu (mx/matmul xt (mx/transpose ge)))
                                               (mx/matmul xt (mx/transpose ue)))
                               y  (mx/matmul h (mx/transpose de))]
                           (mx/add acc (mx/multiply y (/ pe wsum)))))
                       (mx/zeros [1 hidden] (mx/dtype x)) topk)
                 sh   (mx/multiply
                       (mx/silu (mx/matmul xt (mx/transpose (sh-w "gate_proj.weight"))))
                       (mx/matmul xt (mx/transpose (sh-w "up_proj.weight"))))
                 sh   (mx/matmul sh (mx/transpose (sh-w "down_proj.weight")))
                 sg   (mx/sigmoid (mx/matmul xt (mx/transpose (get w (str p "shared_expert_gate.weight")))))]
             (mx/add moe (mx/multiply sh sg)))))]
    (mx/concatenate rows 0)))

;; ---------------------------------------------------------------------------

(if-not model-dir
  (println (str "SKIP llm-qwen35-moe-forward-parity: Ornith-1.0-35B-" quant " not cached"))
  (pr/let [_   (println (str "== Ornith-1.0-35B-" quant " owned qwen3_5_moe forward parity ==\n"))
           _   (assert-true "fwd/supported? (all three gates down)" (fwd/supported? model-dir))
           m   (llm/load-model model-dir)          ; SMART default => owned now
           tok (:tokenizer m)
           ids-raw (llm/encode tok prompt false)
           ids (vec ids-raw)]
    (assert-true "smart default loads the OWNED forward (CljsForwardModel)"
                 (llm/cljs-forward-model? (:model m)))
    (println (str "  prompt " (pr-str prompt) " -> " (count ids) " tokens"))

    ;; ---- (b) routing oracle: pure-op reference vs moe-mlp, layer 0 ----
    (let [fm  (:fwd (:model m))
          cfg (:config fm)
          w   (:weights fm)
          T   (count ids)
          wp  "language_model.model."
          embed (get w (str wp "embed_tokens.weight"))
          h0  (mx/reshape (mx/take-idx embed (mx/array ids [T] mx/int32) 0)
                          [1 T (:hidden cfg)])
          hn  (mx/rms-norm h0 (get w (str wp "layers.0.post_attention_layernorm.weight"))
                           (:eps cfg))
          got (mx/reshape (q35/moe-mlp cfg w (str wp "layers.0.mlp.") hn T)
                          [T (:hidden cfg)])
          ref (pure-moe-reference cfg w (str wp "layers.0.mlp.") hn T)
          scale (mx/item (mx/amax (mx/abs ref)))
          d   (/ (mx/item (mx/amax (mx/abs (mx/subtract ref got)))) (max scale 1.0))]
      (println (str "    [info] layer-0 moe-mlp vs pure-op reference: rel|diff| = "
                    (.toExponential d 2) " (scale " (.toFixed scale 3) ")"))
      (assert-true "moe-mlp matches independent pure-op reference (routing+renorm+shared)"
                   (< d 5e-2)))

    ;; ---- owned-side gates: prefill/step vs uncached ----
    (let [logits (llm/forward-pass (:model m) ids)
          lp     (log-softmax logits)
          am     (mx/item (mx/argmax logits))
          t5     (topk-ids lp 5)]
      (println (str "    [info] owned argmax=" am " top5=" (pr-str t5)))
      (llm/init-cache! (:model m))
      (let [pf-logits (llm/forward-prefill (:model m) ids)]
        (assert= "cached prefill argmax == uncached" am (mx/item (mx/argmax pf-logits)))
        ;; Same code path, but on the MoE this is a REPEATED evaluation, and
        ;; the in-situ quantized expert path (gather-qmm) is inherently
        ;; non-deterministic run-to-run (genmlx-cnhi: kernel-level gather_mm;
        ;; band 0.2, widen to 0.3 if flake). The old <1e-3 band predated the
        ;; cnhi finding and could only pass by draw; the DENSE parity test
        ;; keeps the exact same-path assertion (no MoE, bit-deterministic —
        ;; re-verified fused in the ps8a 9B probe, max-abs-diff=0).
        (let [pf-d (max-abs-diff logits pf-logits t5)]
          (println (str "    [info] cached-vs-uncached (repeat-eval) top5 max|dlogprob| = "
                        (.toFixed pf-d 5)))
          (assert-true "cached prefill logits == uncached (same path, MoE jitter band <0.2)"
                       (< pf-d 0.2)))
        (let [step-logits (llm/forward-step (:model m) am)
              ext-logits  (llm/forward-pass (:model m) (conj ids am))]
          (assert= "incremental step argmax == full-recompute argmax"
                   (mx/item (mx/argmax ext-logits)) (mx/item (mx/argmax step-logits)))
          (assert-true "step vs full-recompute top-5 band (<0.3)"
                       (< (max-abs-diff step-logits ext-logits
                                        (topk-ids (log-softmax ext-logits) 5)) 0.3))))

      ;; ---- parity vs LIVE native upstream ----
      (pr/let [mu (llm/load-model model-dir {:cljs-forward? false})]
        (let [up  (log-softmax (llm/forward-pass (:model mu) ids))
              cl  lp
              up-ids (topk-ids up 5)
              cl-ids (topk-ids cl 5)
              d   (max-abs-diff cl up up-ids)]
          (println (str "    [info] owned-vs-native top5: owned=" (pr-str cl-ids)
                        " native=" (pr-str up-ids)
                        " max|dlogprob|=" (.toFixed d 5)))
          (assert= "owned argmax == native argmax"
                   (mx/item (mx/argmax up)) (mx/item (mx/argmax cl)))
          ;; spec §9a: exact top-5 asserted, quantized-skip NOT applied.
          (assert= "owned top-5 ids == native top-5 ids (EXACT, §9a)" up-ids cl-ids)
          (assert-true "owned vs native top-5 logprob band (<0.25)" (< d 0.25))
          (println (str "\n=== qwen35-moe-forward-parity (" quant "): "
                        @pass " PASS, " @fail " FAIL ==="))
          (when (pos? @fail) (js/process.exit 1)))))))
