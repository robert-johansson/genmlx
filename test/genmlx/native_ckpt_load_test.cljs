;; @tier fast
(ns genmlx.native-ckpt-load-test
  "genmlx-qhy4 (L4): the engine-layout -> HF remap that lets the owned
   forward serve a native training save directly (no Python converter).
   Fabricated tiny weights map + config; asserts exact names, split
   shapes, and the native-save-file detection rules.

   Run: bun run --bun nbb test/genmlx/native_ckpt_load_test.cljs"
  (:require [genmlx.llm.qwen3-forward :as q3f]
            [genmlx.mlx :as mx]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))
(defn assert-equal [label expected actual]
  (if (= expected actual)
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc)
        (println "  FAIL" label "\n    expected:" (pr-str expected)
                 "\n    actual:  " (pr-str actual)))))

;; tiny GDN dims: num_k=1 kd=2, num_v=2 vd=2 -> key_dim 2, value_dim 4,
;; qkv rows = 2*2+4 = 8, z rows = 4; ba rows = 2+2
(def dir (fs/mkdtempSync (path/join (os/tmpdir) "native-ckpt-")))
(fs/writeFileSync
 (path/join dir "config.json")
 (js/JSON.stringify
  (clj->js {:text_config {:linear_num_key_heads 1 :linear_key_head_dim 2
                          :linear_num_value_heads 2 :linear_value_head_dim 2}})))

(defn- arr2 [rows cols]
  (mx/reshape (mx/arange 0 (* rows cols) 1) [rows cols]))

(def weights
  {"embedding.weight"                         (arr2 6 3)
   "final_norm.weight"                        (mx/ones [3])
   "layers.0.linear_attn.in_proj_qkvz.weight" (arr2 12 3)
   "layers.0.linear_attn.in_proj_ba.weight"   (arr2 4 3)
   "layers.0.linear_attn.a_log"               (mx/ones [2])
   "layers.0.mlp.gate_proj.weight"            (arr2 4 3)
   "visual.blocks.0.attn.proj.weight"         (arr2 3 3)})

(println "\n-- native->hf-weights --")
(let [hf (q3f/native->hf-weights weights dir)]
  (assert-equal "key names remapped exactly"
                #{"language_model.model.embed_tokens.weight"
                  "language_model.model.norm.weight"
                  "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
                  "language_model.model.layers.0.linear_attn.in_proj_z.weight"
                  "language_model.model.layers.0.linear_attn.in_proj_b.weight"
                  "language_model.model.layers.0.linear_attn.in_proj_a.weight"
                  "language_model.model.layers.0.linear_attn.A_log"
                  "language_model.model.layers.0.mlp.gate_proj.weight"
                  "vision_tower.blocks.0.attn.proj.weight"}
                (set (keys hf)))
  (assert-equal "qkv split rows" [8 3]
                (mx/shape (get hf "language_model.model.layers.0.linear_attn.in_proj_qkv.weight")))
  (assert-equal "z split rows" [4 3]
                (mx/shape (get hf "language_model.model.layers.0.linear_attn.in_proj_z.weight")))
  (assert-equal "b/a split rows" [[2 3] [2 3]]
                [(mx/shape (get hf "language_model.model.layers.0.linear_attn.in_proj_b.weight"))
                 (mx/shape (get hf "language_model.model.layers.0.linear_attn.in_proj_a.weight"))])
  ;; content: rows of qkvz [12 3] arange — qkv = rows 0..7, z = rows 8..11
  (assert-equal "split content is the row range (first z element)"
                (* 8 3)
                (mx/item (mx/index (mx/reshape
                                    (get hf "language_model.model.layers.0.linear_attn.in_proj_z.weight")
                                    [12])
                                   0))))

(println "\n-- native-save-file detection --")
(let [nd (fs/mkdtempSync (path/join (os/tmpdir) "native-detect-"))]
  (assert-equal "no weights file -> nil" nil (q3f/native-save-file nd))
  (fs/writeFileSync (path/join nd "weights.safetensors") "x")
  (assert-equal "engine save detected"
                (path/join nd "weights.safetensors") (q3f/native-save-file nd))
  (fs/writeFileSync (path/join nd "model.safetensors") "x")
  (assert-equal "HF layout wins when both exist" nil (q3f/native-save-file nd)))

(println (str "\n== native-ckpt-load: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
