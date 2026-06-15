;; @tier slow
(ns genmlx.llm-qwen35-forward-parity-test
  "f6ov P6 GATE: parity of the GenMLX-owned Qwen3.5 HYBRID forward
   (genmlx.llm.qwen35-forward — GatedDeltaNet linear layers + every-4th
   full-attention layer with partial RoPE + output gate) against:
     (1) the golden-oracle pins (llm_forward_golden_test) for qwen3.5-0.8b/4b, and
     (2) the LIVE upstream `.forward` on the 0.8b checkpoint.

   The CLJS forward runs the gated-delta recurrence as pure ops (the upstream
   `use_kernel=false` reference) over the genmlx.rs primitives; upstream runs the
   fused Metal kernel. So they agree on argmax + the full top-5 ranking EXACTLY,
   and on logprob VALUES within a bf16 cross-kernel band (the SSM SIMD-reduction
   order + bf16 residual rounding differ across the layer stack). Argmax/top-5-ids
   exact is the strong invariant; the logprob band bounds gross divergence (a real
   bug would scramble the ranking or be off by whole nats).

   Also checks the GenMLX-owned KV cache: cached prefill == uncached forward
   (same code path, exact), and prefill+step tracks the uncached forward of the
   extended sequence (validates incremental conv-state + recurrent-state + the
   full-attention KV cache). 4B additionally exercises the GQA-in-GDN path
   (linear_num_value_heads 32 > linear_num_key_heads 16, repeat factor 2).

   Skips each model cleanly if its checkpoint is absent."
  (:require [genmlx.llm.backend :as llm]
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
(defn assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol) (do (swap! pass inc) (println (str "  PASS: " label " (diff=" (.toFixed d 6) ")")))
        (do (swap! fail inc) (println (str "  FAIL: " label " expected=" expected " actual=" actual " (diff=" (.toFixed d 6) ")"))))))

(def prompt "The capital of France is")
(def expected-prompt-ids [760 6511 314 9338 369])
(def model-root (str (.-HOME js/process.env) "/.cache/models"))

;; Golden pins (captured 2026-06-14 from the upstream bf16 forward; same as
;; llm_forward_golden_test). argmax 11751 decodes to " Paris" for both.
(def models
  [{:name "qwen3.5-0.8b" :dir "qwen3.5-0.8b-mlx-bf16" :upstream? true
    :argmax 11751 :decoded " Paris" :argmax-lp -2.171875
    :top5 [11751 279 7172 25 198]}
   {:name "qwen3.5-4b" :dir "qwen3.5-4b-mlx-bf16" :upstream? false
    :argmax 11751 :decoded " Paris" :argmax-lp -0.601563
    :top5 [11751 7172 264 3750 279]}])

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

(defn check-model [{:keys [name dir upstream? argmax decoded argmax-lp top5]}]
  (let [path (str model-root "/" dir)]
    (if-not (.existsSync fs path)
      (do (println (str "\n== " name " — SKIP (absent: " path ") ==")) (pr/resolved nil))
      (pr/let [m   (llm/load-model path {:cljs-forward? true})
               tok (:tokenizer m)
               ids-raw (llm/encode tok prompt false)
               ids (vec ids-raw)
               dec (llm/decode tok (js/Uint32Array.from #js [argmax]))]
        (println (str "\n== " name " CLJS hybrid forward parity ==  prompt=" (pr-str prompt)))
        (assert= "tokenizer prompt-ids stable" expected-prompt-ids ids)

        ;; ---- (1) golden-pin reproduction (uncached next-token) ----
        (let [logits (llm/forward-pass (:model m) ids)         ; [vocab]
              lp     (log-softmax logits)]
          (assert= "uncached argmax == golden (Paris id)" argmax (mx/item (mx/argmax logits)))
          (assert= "argmax decodes to expected word" decoded dec)
          (assert= "top-5 token ids == golden" top5 (topk-ids lp 5))
          (assert-close "argmax logprob within cross-kernel band" argmax-lp
                        (mx/item (mx/index lp argmax)) 0.25)

          ;; ---- (2) cached prefill == uncached (same code path) ----
          (llm/init-cache! (:model m))
          (let [pf-logits (llm/forward-prefill (:model m) ids)] ; [vocab]
            (assert= "cached prefill argmax == golden" argmax (mx/item (mx/argmax pf-logits)))
            (assert-true "cached prefill logits == uncached (exact, same path)"
                         (< (max-abs-diff logits pf-logits top5) 1e-3))

            ;; ---- (3) prefill + step tracks uncached forward of [ids + argmax] ----
            (let [step-logits (llm/forward-step (:model m) argmax)
                  ext-logits  (llm/forward-pass (:model m) (conj ids argmax))]
              (assert= "incremental step argmax == full-recompute argmax"
                       (mx/item (mx/argmax ext-logits)) (mx/item (mx/argmax step-logits)))
              (assert-true "step vs full-recompute agree on top-5 (<0.3, cached recurrence)"
                           (< (max-abs-diff step-logits ext-logits (topk-ids (log-softmax ext-logits) 5)) 0.3)))))

        ;; ---- (4) direct parity vs LIVE upstream .forward (0.8b only) ----
        (if-not upstream?
          (pr/resolved nil)
          (pr/let [mu (llm/load-model path {:cljs-forward? false})]
            (let [cl (log-softmax (llm/forward-pass (:model m) ids))
                  up (log-softmax (llm/forward-pass (:model mu) ids))
                  up-ids (topk-ids up 5)
                  d (max-abs-diff cl up up-ids)]
              (println (str "    [info] CLJS-vs-upstream top5 max logprob diff = " (.toFixed d 5)))
              (assert= "CLJS argmax == upstream argmax" (mx/item (mx/argmax up)) (mx/item (mx/argmax cl)))
              (assert= "CLJS top-5 ids == upstream top-5 ids" up-ids (topk-ids cl 5))
              (assert-true "CLJS vs upstream logprobs agree on top-5 (<0.25, bf16 cross-kernel)"
                           (< d 0.25)))))))))

(pr/let [_ (check-model (first models))
         _ (check-model (second models))]
  (println (str "\n=== qwen35-forward-parity: " @pass " PASS, " @fail " FAIL ===")))
