;; @tier exclude — loads the native Ornith-1.0-35B-4bit (~19 GB) as the oracle;
;; needs the test image at ~/code/mlx/ornith/image.png. Run manually:
;;   bunx --bun nbb@1.4.208 test/genmlx/llm_qwen35_vision_parity_test.cljs
(ns genmlx.llm-qwen35-vision-parity-test
  "genmlx-w3og GATE (Ornith Phase 4, tower stage): parity of the GenMLX-owned
   CLJS vision tower (genmlx.llm.qwen35-vision-forward) against the native
   vlmVisionFeatures debug tap on a real image + the real Ornith vision
   weights.

   Both sides consume the SAME native preprocessing (mx/vlm-preprocess /
   Qwen35VLImageProcessor — image decode is I/O, kept native by design), so a
   feature mismatch isolates to the tower itself: patch embed, interpolated
   position embeddings, 2D rotary attention, block mask, or merger.

   The owned side loads ONLY the vision_tower.* tensors (bf16, ~1.5 GB — the
   tower is identical across the 3/4/8-bit quants since vision is never
   quantized), so the oracle model can be the light 4-bit checkpoint."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.qwen3-forward :as q3]
            [genmlx.llm.qwen35-vision-forward :as vis]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

(def model-dir
  (let [base (str (.-HOME js/process.env)
                  "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
    (when (.existsSync fs base)
      (str base "/" (first (js->clj (.readdirSync fs base)))))))
(def image-path (str (.-HOME js/process.env) "/code/mlx/ornith/image.png"))

(defn- load-vision-weights
  "Only the vision_tower.* tensors from the checkpoint's shards."
  [dir]
  (reduce (fn [m f]
            (into m (filter (fn [[k _]] (.startsWith k "vision_tower.")))
                  (mx/load-safetensors f)))
          {} (q3/weight-files dir)))

(if-not (and model-dir (.existsSync fs image-path))
  (println "SKIP llm-qwen35-vision-parity: checkpoint or test image absent")
  (let [img   (.readFileSync fs image-path)
        [pv grid-arr] (mx/vlm-preprocess [img])
        grids (mapv vec (mx/->clj grid-arr))
        vcfg  (vis/load-vision-config model-dir)
        vw    (load-vision-weights model-dir)]
    (println "== owned qwen3.5 vision tower parity (Ornith-4bit oracle) ==")
    (println "  image:" image-path)
    (println "  pixel-values" (pr-str (mx/shape pv)) "| grid" (pr-str grids))
    (assert-true "vision config parsed (27 blocks)" (= 27 (:depth vcfg)))
    (assert-true (str "vision weights loaded (" (count vw) " tensors, bf16 only)")
                 (and (pos? (count vw))
                      (contains? vw "vision_tower.patch_embed.proj.weight")
                      (not-any? #(.endsWith % ".scales") (keys vw))))
    (let [own (vis/vision-features vw vcfg pv grids)
          [t h w] (first grids)
          m   (:merge vcfg)
          exp-rows (* t (quot h m) (quot w m))]
      (println "  owned features" (pr-str (mx/shape own)))
      (assert-true (str "owned features shape [" exp-rows " " (:out-hidden vcfg) "]")
                   (= [exp-rows (:out-hidden vcfg)] (mx/shape own)))
      ;; --- native oracle ---
      (pr/let [mu (llm/load-model model-dir {:cljs-forward? false})]
        (let [nat (.vlmVisionFeatures ^js (:model mu) (into-array [img]))
              _   (assert-true "native features same shape" (= (mx/shape own) (mx/shape nat)))
              of  (mx/astype own mx/float32)
              nf  (mx/astype nat mx/float32)
              scale (mx/item (mx/amax (mx/abs nf)))
              dmax  (/ (mx/item (mx/amax (mx/abs (mx/subtract of nf)))) scale)
              dmean (/ (mx/item (mx/mean (mx/abs (mx/subtract of nf)))) scale)]
          (println (str "    [info] native scale=" (.toFixed scale 3)
                        " rel-max|diff|=" (.toExponential dmax 2)
                        " rel-mean|diff|=" (.toExponential dmean 2)))
          (assert-true "owned tower matches native features (rel-max < 2e-2, bf16 band)"
                       (< dmax 2e-2))
          (assert-true "mean deviation tight (rel-mean < 2e-3)"
                       (< dmean 2e-3))
          (println (str "\n=== qwen35-vision-parity: " @pass " PASS, " @fail " FAIL ==="))
          (when (pos? @fail) (js/process.exit 1)))))))
