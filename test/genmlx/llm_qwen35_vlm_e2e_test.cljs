;; @tier exclude — loads Ornith-1.0-35B (owned OR native, ~20-25 GB per phase)
;; plus the test image. MEMORY SAFETY (this test OOM-killed the Thor when both
;; models shared one process, 2026-07-10): it runs in TWO phases, one model per
;; process, comparing through a result file. Run manually, owned phase first:
;;   GENMLX_VLM_E2E_MODE=owned  bunx --bun nbb@1.4.208 test/genmlx/llm_qwen35_vlm_e2e_test.cljs
;;   GENMLX_VLM_E2E_MODE=native bunx --bun nbb@1.4.208 test/genmlx/llm_qwen35_vlm_e2e_test.cljs
;; and keep a MemAvailable watchdog beside it on this box (genmlx-h3p5).
(ns genmlx.llm-qwen35-vlm-e2e-test
  "genmlx-w3og GATE (Ornith Phase 4, decoder stage): the owned VLM prefill —
   native preprocessing -> OWNED vision tower -> image-pad expansion ->
   feature scatter -> 3-axis interleaved M-RoPE -> owned qwen3_5_moe decoder —
   against the native vlmPrefillFlat on the same image + prompt.

   Logit-level gates only (the native MoE forward jitters on CUDA — genmlx-
   ba06); the owned phase also decodes a short greedy continuation through the
   owned cache at compressed M-RoPE positions (offset + rope-delta,
   genmlx-52mh) as an unasserted human sanity check. The owned phase writes
   {argmax, top-k ids, top-k logprobs, continuation} to a result file; the
   native phase loads ONLY the native model and asserts against the file."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.qwen35-forward :as q35]
            [genmlx.llm.qwen35-vision-forward :as vis]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as node-path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))
(defn assert= [label expected actual]
  (assert-true (str label " (=" (pr-str expected) ")") (= expected actual)))

(def mode (or (.-GENMLX_VLM_E2E_MODE js/process.env) "owned"))
(def result-file (node-path/join (.tmpdir os) "genmlx-vlm-e2e-owned.json"))

(def model-dir
  (let [base (str (.-HOME js/process.env)
                  "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
    (when (.existsSync fs base)
      (str base "/" (first (js->clj (.readdirSync fs base)))))))
(def image-path (str (.-HOME js/process.env) "/code/mlx/ornith/image.png"))

(def prompt
  (str "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
       "What is shown in this image? Answer in a few words.<|im_end|>\n"
       "<|im_start|>assistant\n"))

(defn- log-softmax [logits] (mx/subtract logits (mx/logsumexp logits)))
(defn- topk [lp k]
  (mx/eval! lp)
  (let [f32 (.toFloat32 lp)]
    (->> (range (.-length f32)) (map (fn [i] [i (aget f32 i)]))
         (sort-by second >) (take k)
         (mapv (fn [[i v]] [i v])))))

(defn- finish []
  (println (str "\n=== qwen35-vlm-e2e (" mode "): " @pass " PASS, " @fail " FAIL ==="))
  (when (pos? @fail) (js/process.exit 1)))

(defn- run-owned []
  (pr/let [img (.readFileSync fs image-path)
           m   (llm/load-model model-dir)          ; owned (smart default)
           tok (:tokenizer m)
           raw (llm/encode tok prompt false)
           tokens (vec raw)]
    (println "== owned VLM prefill (phase 1/2) ==")
    (mx/force-gc!)                                  ; trim the dequant transient
    (assert-true "prompt carries exactly one <|image_pad|> marker"
                 (= 1 (count (filter #(= % vis/image-token-id) tokens))))
    (let [fm  (:fwd (:model m))
          vcfg (vis/load-vision-config model-dir)
          {:keys [logits cache seq-len rope-delta]} (vis/vlm-prefill fm vcfg [img] tokens)
          lp  (log-softmax logits)
          am  (mx/item (mx/argmax logits))
          t5  (topk lp 5)]
      (println (str "  expanded seq-len=" seq-len " rope-delta=" rope-delta
                    " | owned argmax=" am " top5=" (pr-str (mapv first t5))))
      (assert-true "owned VLM prefill produced finite logits"
                   (js/isFinite (second (first t5))))
      ;; Continuation decodes through the backend BRANCH surface (genmlx-7f93):
      ;; install-prefill! adopts the image-conditioned cache + rope-delta into
      ;; the CljsForwardModel cell, and a branch forked off it must carry the
      ;; M-RoPE shift transparently — the same offset math as the old direct
      ;; (q35/step fm c (+ seq-len rope-delta i) id) loop, now owned by the
      ;; branch ledger. A wrong/lost delta degrades this continuation visibly
      ;; (the genmlx-52mh failure mode).
      (llm/install-prefill! (:model m) {:cache cache :seq-len seq-len
                                        :rope-delta rope-delta})
      (let [bid (llm/branch-cache! (:model m))]
        (assert-true "owned model exposes the branch surface over the VLM prefix"
                     (llm/supports-branching? (:model m)))
        (pr/let [ids (loop [i 0 id am acc [am]]
                       (if (< i 11)
                         (let [lg (llm/forward-branch (:model m) bid id)
                               nid (mx/item (mx/argmax lg))]
                           (recur (inc i) nid (conj acc nid)))
                         acc))
                 text (llm/decode tok (js/Uint32Array.from (into-array ids)))]
          (llm/dispose-branch! (:model m) bid)
          (println (str "  owned greedy continuation (decoded on a BRANCH off the image prefix): "
                        (pr-str text)))
          (.writeFileSync fs result-file
                          (js/JSON.stringify
                           (clj->js {:tokens tokens :argmax am :top5 t5
                                     :seq-len seq-len :rope-delta rope-delta
                                     :continuation text})))
          (println (str "  result saved -> " result-file
                        "\n  now run: GENMLX_VLM_E2E_MODE=native …"))
          (finish))))))

(defn- run-native []
  (if-not (.existsSync fs result-file)
    (do (println (str "SKIP native phase: " result-file " missing — run the owned phase first"))
        (js/process.exit 1))
    (pr/let [own (js->clj (js/JSON.parse (.readFileSync fs result-file "utf8"))
                          :keywordize-keys true)
             img (.readFileSync fs image-path)
             mu  (llm/load-model model-dir {:cljs-forward? false})]
      (println "== native oracle vs saved owned results (phase 2/2) ==")
      (mx/force-gc!)
      (let [nl  (.vlmPrefillFlat ^js (:model mu)
                                 (js/Uint32Array.from (into-array (:tokens own)))
                                 (into-array [img]))
            nl  (mx/reshape nl [-1])
            nlp (log-softmax nl)
            n5  (topk nlp 5)
            own-t5 (mapv (fn [[i v]] [(long i) v]) (:top5 own))
            band (reduce max (map (fn [[_ a] [_ b]] (js/Math.abs (- a b))) own-t5 n5))]
        (println (str "    [info] native argmax=" (mx/item (mx/argmax nl))
                      " top5=" (pr-str (mapv first n5))
                      " | owned continuation was " (pr-str (:continuation own))))
        (assert= "owned VLM-prefill argmax == native" (mx/item (mx/argmax nl)) (:argmax own))
        ;; The native oracle JITTERS (genmlx-ba06): at T=617 its own reruns flip
        ;; the rank-5 token between near-ties (observed 1 vs 27 across runs).
        ;; Ranks 1-4 are stable — assert those exactly; the tail as set-overlap.
        (assert= "owned top-4 ids == native top-4 ids"
                 (subvec (mapv first n5) 0 4) (subvec (mapv first own-t5) 0 4))
        (assert-true "top-5 sets overlap >= 4 (rank-5 is a jittering near-tie)"
                     (>= (count (filter (set (mapv first n5)) (mapv first own-t5))) 4))
        ;; Band sized to the MEASURED platform noise at this length: the native
        ;; MoE forward's own run-to-run top-5 logprob spread at T=617 is
        ;; 1.375-2.625 nats (kernel-level jitter, genmlx-ba06/cnhi; measured
        ;; 2026-07-10). Owned-vs-native measured 0.75-2.5 — INSIDE that noise.
        ;; The strong invariant at this length is the exact top-5 ranking above.
        (assert-true (str "top-5 logprob band (<3.0 — the T=617 platform noise "
                          "floor; got " (.toFixed band 5) ")")
                     (< band 3.0))
        (finish)))))

(if-not (and model-dir (.existsSync fs image-path))
  (println "SKIP llm-qwen35-vlm-e2e: checkpoint or test image absent")
  (case mode
    "owned"  (run-owned)
    "native" (run-native)
    (println (str "unknown GENMLX_VLM_E2E_MODE " (pr-str mode)))))
