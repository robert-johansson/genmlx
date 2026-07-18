;; @tier fast
(ns genmlx.llm.overlay-test
  "Mechanics tests for the trained-partial-save overlay (genmlx-vjsp):
   q35/overlay-weights merges a partial save's tensors over the base weight
   map and REFUSES unknown keys. Exercised against hand-crafted safetensors
   dirs (the format is 8-byte LE header length + JSON header + raw data) —
   no 35B needed for the logic; the real-checkpoint E2E lives in the
   grpo_sessions serve check."
  (:require [genmlx.llm.qwen35-forward :as q35]
            [genmlx.mlx :as mx]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn- assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(defn- write-safetensors!
  "Minimal single-file F32 safetensors writer for fixtures.
   tensors = [[name [floats...]] ...]."
  [file tensors]
  (let [[header-map _]
        (reduce (fn [[m off] [nm vals]]
                  (let [len (* 4 (count vals))]
                    [(assoc m nm {:dtype "F32" :shape [(count vals)]
                                  :data_offsets [off (+ off len)]})
                     (+ off len)]))
                [{} 0] tensors)
        hbuf   (js/Buffer.from (js/JSON.stringify (clj->js header-map)) "utf8")
        lenbuf (js/Buffer.alloc 8)]
    (.writeBigUInt64LE lenbuf (js/BigInt (.-length hbuf)) 0)
    (let [bufs (map (fn [[_ vals]]
                      (let [b (js/Buffer.alloc (* 4 (count vals)))]
                        (doseq [[i v] (map-indexed vector vals)]
                          (.writeFloatLE b v (* 4 i)))
                        b))
                    tensors)]
      (.writeFileSync fs file
                      (js/Buffer.concat (into-array (concat [lenbuf hbuf] bufs)))))))

(defn- make-dir! [nm tensors]
  (let [dir (path/join (os/tmpdir) (str "genmlx-overlay-test-" (.-pid js/process) "-" nm))]
    (.mkdirSync fs dir #js {:recursive true})
    (.writeFileSync fs (path/join dir "config.json")
                    (js/JSON.stringify #js {:model_type "qwen3_5"}))
    (write-safetensors! (path/join dir "model.safetensors") tensors)
    dir))

(println "\n-- overlay-weights mechanics --")
(let [base-dir  (make-dir! "base" [["a.weight" [1.0 2.0]] ["b.weight" [3.0 4.0]]])
      good-dir  (make-dir! "good" [["a.weight" [10.0 20.0]]])
      bad-dir   (make-dir! "bad"  [["c.weight" [9.0 9.0]]])
      base      (into {} (mx/load-safetensors (path/join base-dir "model.safetensors")))
      merged    (q35/overlay-weights base good-dir)]
  (assert-true "merged keeps the base keyset"
               (= #{"a.weight" "b.weight"} (set (keys merged))))
  (assert-true "overlaid tensor takes the trained values"
               (= [10.0 20.0] (vec (mx/->clj (get merged "a.weight")))))
  (assert-true "non-overlaid tensor keeps the base values"
               (= [3.0 4.0] (vec (mx/->clj (get merged "b.weight")))))
  (assert-true "an overlay key absent from base throws :overlay-key-mismatch"
               (try (q35/overlay-weights base bad-dir) false
                    (catch :default e
                      (= :overlay-key-mismatch (:genmlx/error (ex-data e))))))
  (doseq [d [base-dir good-dir bad-dir]]
    (.rmSync fs d #js {:recursive true :force true})))

(println (str "\n== llm-overlay: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
