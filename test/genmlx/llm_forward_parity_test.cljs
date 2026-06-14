;; @tier slow
(ns genmlx.llm-forward-parity-test
  "f6ov P2/P5 parity gate: the GenMLX-owned CLJS Qwen3 forward
   (genmlx.llm.qwen3-forward, composed over the genmlx.rs fast:: primitives +
   mx/load-safetensors) must match upstream's forward on the same checkpoint.
   Compares DIRECTLY against the live upstream forward (no hardcoded golden):
   argmax + top-5 token ids must be identical; logits within bf16 cross-kernel
   tolerance. Skips cleanly if the qwen3-0.6b checkpoint is absent."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.qwen3-forward :as fwd]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

(def ^:private dir (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-mlx-bf16"))

(defn- topk [logits k]
  (mx/eval! logits)
  (let [f32 (.toFloat32 logits)]
    (->> (range (.-length f32)) (map (fn [i] [i (aget f32 i)])) (sort-by second >) (take k))))

(if-not (.existsSync fs (str dir "/model.safetensors"))
  (println "SKIP llm-forward-parity-test: qwen3-0.6b checkpoint absent")
  (pr/let [up (llm/load-model dir)
           ids-raw (llm/encode (:tokenizer up) "The capital of France is" false)
           ids (vec ids-raw)
           ;; upstream forward (cached prefill — the path the LLM-as-GF uses)
           _ (llm/init-cache! (:model up))
           up-logits (llm/forward-prefill (:model up) ids)
           _ (mx/eval! up-logits)
           _ (llm/reset-cache! (:model up))
           ;; GenMLX-owned CLJS forward
           cljs-model (fwd/load-model dir)
           cljs-logits (fwd/next-token-logits cljs-model ids)
           _ (mx/eval! cljs-logits)]
    (println "\n== f6ov forward parity (qwen3-0.6b) ==")
    (let [up-amax   (mx/item (mx/argmax up-logits))
          cljs-amax (mx/item (mx/argmax cljs-logits))
          up-top    (mapv first (topk up-logits 5))
          cljs-top  (mapv first (topk cljs-logits 5))
          uf (.toFloat32 up-logits)
          cf (.toFloat32 cljs-logits)
          max-diff (reduce max (map (fn [i] (js/Math.abs (- (aget uf i) (aget cf i)))) up-top))]
      (pr/let [dec (llm/decode (:tokenizer up) (js/Uint32Array.from #js [cljs-amax]))]
        (println "  upstream argmax:" up-amax " cljs argmax:" cljs-amax " decoded:" (pr-str dec))
        (println "  upstream top5:" (pr-str up-top))
        (println "  cljs     top5:" (pr-str cljs-top))
        (println "  max logit diff over top5:" (.toFixed max-diff 4))
        (assert-true "argmax token id matches upstream" (= up-amax cljs-amax))
        (assert-true "argmax decodes to \" Paris\"" (= " Paris" dec))
        (assert-true "top-5 token ids identical to upstream" (= up-top cljs-top))
        (assert-true "top-5 logits within bf16 cross-kernel tol (<0.5)" (< max-diff 0.5))

        ;; --- KV cache consistency (P3): prefill == uncached; step matches the
        ;; uncached forward of the extended sequence (argmax exact, logits bf16) ---
        (println "\n-- KV cache (prefill + step) --")
        (let [[pf cache] (fwd/prefill cljs-model ids)
              pf-diff (let [pf32 (do (mx/eval! pf) (.toFloat32 pf))]
                        (reduce max (map (fn [i] (js/Math.abs (- (aget pf32 i) (aget cf i)))) up-top)))
              next-id (mx/item (mx/argmax pf))
              [st _] (fwd/step cljs-model cache (count ids) next-id)
              unc-ext (fwd/next-token-logits cljs-model (conj ids next-id))]
          (mx/eval! st) (mx/eval! unc-ext)
          (assert-true "prefill last-logits == uncached forward (exact)" (< pf-diff 1e-3))
          (assert-true "step argmax == uncached(prompt+token) argmax"
                       (= (mx/item (mx/argmax st)) (mx/item (mx/argmax unc-ext)))))
        (println (str "\n=== forward-parity: " @pass " PASS, " @fail " FAIL ==="))))))
