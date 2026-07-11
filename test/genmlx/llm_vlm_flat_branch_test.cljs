;; @tier exclude — loads Qwen3.6-35B-A3B-4bit (~20 GB resident). Run manually:
;;   bunx --bun nbb@1.4.208 test/genmlx/llm_vlm_flat_branch_test.cljs
;; GENMLX_VLM_MOE_MODEL overrides the checkpoint dir; GENMLX_VLM_IMAGE the
;; image. Keep a MemAvailable watchdog beside it on this box (genmlx-h3p5).
(ns genmlx.llm-vlm-flat-branch-test
  "genmlx-52mh coverage (promoted from the qwen36-compat/ work-tree probe,
   genmlx-j3yc): image conditioning is BRANCHABLE on the native 35B-A3B VLM.
   Prefill an image into the FLAT cache (vlm-prefill-flat!), fork the
   image-conditioned prefix, and greedy-decode a continuation on BOTH the
   branch and the model-internal cache. The win this pins: the expensive
   image look happens once, then branched re-looks/particles are cheap —
   the vision leg of the second-path mechanic.

   Gates (argmax/structure only — the quantized MoE expert path jitters in
   situ, genmlx-cnhi, so sampled TEXT content is never asserted):
     V1 vlmPrefillFlat surface present on the native VLM
     V2 chat template renders >= 1 image marker
     V3 flat image prefill returns [vocab]-shaped logits
     V4 branch surface live after the image prefill
     V5 branch greedy == model-internal greedy over 24 tokens
        (argmax-stable across the fork)
     V6 non-empty continuation"
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn- check [ok? msg]
  (if ok?
    (do (swap! pass inc) (println "  PASS" msg))
    (do (swap! fail inc) (println "  FAIL" msg))))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_VLM_MOE_MODEL)
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))

(def img-path
  (or (some-> js/process .-env .-GENMLX_VLM_IMAGE)
      (first (filter #(.existsSync fs %) ["mlx-node/images/demo.png" "genmlx.png"]))))

(def IMAGE-TOKEN 248056)

(defn- mat [a] (mx/materialize! a) a)
(defn- greedy [l] (mx/item (mx/argmax l)))
(defn- count-tok [arr v]
  (let [n (.-length arr)]
    (loop [i 0, c 0]
      (if (>= i n) c (recur (inc i) (if (= (aget arr i) v) (inc c) c))))))

(defn- summary []
  (println (str "\n== llm-vlm-flat-branch: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(if-not (and model-dir (.existsSync fs model-dir) img-path)
  (do (println "SKIP llm-vlm-flat-branch — Qwen3.6-35B-A3B-4bit not cached, or no image")
      (summary))
  ;; explicit NATIVE load: vlm-prefill-flat! is the native flat-vision surface
  ;; (genmlx-52mh); the smart default would pick the owned forward here, whose
  ;; image path is forward-prefill {:images} instead (llm_owned_vlm_gf_test).
  ;; applyChatTemplate resolves in the pr/let — it returns a promise, and a
  ;; plain let here hands count-tok a Promise (n = undefined -> infinite loop).
  (pr/let [{:keys [model tokenizer]} (llm/load-model model-dir {:cljs-forward? false})
           _ (println "== flat-VLM branchable image prefill on the 35B-A3B VLM ==")
           _ (check (llm/supports-vlm-prefill? model) "V1 model exposes vlmPrefillFlat")
           buf (.readFileSync fs img-path)
           img (js/Uint8Array. (.-buffer buf) (.-byteOffset buf) (.-byteLength buf))
           messages (clj->js [{:role "user"
                               :content "Describe this image in one short sentence."
                               :images [img]}])
           tokens (.applyChatTemplate tokenizer messages true)]
    (let [n-img (count-tok tokens IMAGE-TOKEN)
          eos (llm/eos-token-id tokenizer)]
      (println "  chat tokens:" (.-length tokens) " image-markers(248056):" n-img)
      (check (>= n-img 1) "V2 chat template renders an image marker")
      (let [l0 (mat (llm/vlm-prefill-flat! model tokens [img]))
            vocab (first (mx/shape l0))]
        (check (> vocab 200000) (str "V3 flat image prefill logits [vocab=" vocab "]"))
        (check (llm/supports-branching? model) "V4 branchable after the image prefill")
        (let [bid (llm/branch-cache! model)
              decode-n 24
              toksA (loop [i 0, l l0, acc []]     ; branch (fork of the image prefix)
                      (if (>= i decode-n) acc
                          (let [t (greedy l)]
                            (if (= t eos) (conj acc t)
                                (recur (inc i) (mat (llm/forward-branch model bid t))
                                       (conj acc t))))))
              toksB (loop [i 0, l l0, acc []]     ; model-internal cache (same prefix)
                      (if (>= i decode-n) acc
                          (let [t (greedy l)]
                            (if (= t eos) (conj acc t)
                                (recur (inc i) (mat (llm/forward-step model t))
                                       (conj acc t))))))]
          (llm/dispose-branch! model bid)
          (check (= toksA toksB) "V5 branch greedy == model-internal greedy (argmax-stable across the fork)")
          (check (>= (count toksA) 1) "V6 non-empty continuation")
          (pr/let [textA (llm/decode tokenizer (js/Uint32Array.from (clj->js toksA)))]
            (println "\n  vision-grounded continuation (branch):" (pr-str textA))
            (println "  (eyeball: should describe the image — proving the image conditioned the branchable prefix)")
            (summary)))))))
