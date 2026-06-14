;; @tier slow
(ns genmlx.llm-weights-test
  "f6ov P1: GenMLX-owned safetensors weight loading (mx/load-safetensors over the
   genmlx.rs loadSafetensors export). Decouples weight access from upstream's
   per-model structs. Skips cleanly if the checkpoint is absent."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            ["fs" :as fs]))

(def ^:private dir
  (str (.-HOME js/process.env) "/.cache/models/qwen3.5-0.8b-mlx-bf16"))
(def ^:private st-path (str dir "/model.safetensors"))

(deftest load-safetensors-roundtrip
  (if-not (.existsSync fs st-path)
    (println "SKIP llm-weights-test: checkpoint absent" st-path)
    (testing "load weights as a {name -> MxArray} map"
      (let [w (mx/load-safetensors st-path)]
        (is (map? w) "returns a CLJS map")
        (is (pos? (count w)) "loads tensors")
        (is (every? string? (keys w)) "keys are tensor names")
        (testing "embedding weight is a real 2D [vocab hidden] tensor"
          (let [embed (get w "language_model.model.embed_tokens.weight")]
            (is (some? embed) "embed_tokens.weight present")
            (when embed
              (let [sh (vec (mx/shape embed))]
                (is (= 2 (count sh)) "2D")
                (is (> (first sh) 100000) "vocab dim large")
                ;; lazily materializes to a finite value (graph leaf is real)
                (is (js/isFinite (mx/item (mx/sum embed))) "materializes finite")))))
        (testing "config.json parses with the model type"
          (let [cfg (js/JSON.parse (.readFileSync fs (str dir "/config.json") "utf8"))]
            (is (= "qwen3_5" (.-model_type cfg)) "model_type qwen3_5")))))))

(cljs.test/run-tests)
