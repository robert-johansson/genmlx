;; @tier slow
(ns genmlx.llm-sharded-load-test
  "genmlx-sbif (Ornith Phase 1): the owned loader reads HF sharded checkpoints.

   No small sharded checkpoint is cached, so this test MAKES one: it byte-splits
   the real qwen3.5-0.8b model.safetensors into 3 shards + a
   model.safetensors.index.json in a temp dir (tensor bytes copied verbatim,
   offsets rebased per shard — the exact HF layout), then loads the checkpoint
   both ways through fwd/load-model and asserts the weight maps are identical
   tensor-for-tensor and the forward runs end-to-end on the sharded copy.

   Also pins the both-layouts precedence: the 0.8b checkpoint ships a redundant
   index.json NEXT TO its single model.safetensors, and weight-files must pick
   the single file. Skips cleanly if the checkpoint is absent."
  (:require [genmlx.llm.forward :as fwd]
            [genmlx.llm.qwen3-forward :as q3]
            [genmlx.mlx :as mx]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as node-path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

;; ---------------------------------------------------------------------------
;; Byte-level safetensors splitter (test fixture builder)
;; ---------------------------------------------------------------------------
;; safetensors layout: u64-LE header length | JSON header | data blob.
;; Header entries: {name {dtype, shape, data_offsets [start end]}} with offsets
;; relative to the blob. We partition tensors into n groups (in blob order),
;; copy each tensor's bytes verbatim, rebase offsets cumulatively per shard.

(defn- write-shard! [out-path entries blob blob0]
  (let [hdr (js/Object.)
        slices #js []]
    (loop [es entries off 0]
      (when-let [[nm meta] (first es)]
        (let [offs (.-data_offsets meta)
              s    (aget offs 0)
              e    (aget offs 1)
              size (- e s)]
          (unchecked-set hdr nm
                         #js {:dtype (.-dtype meta)
                              :shape (.-shape meta)
                              :data_offsets #js [off (+ off size)]})
          (.push slices (.subarray blob (+ blob0 s) (+ blob0 e)))
          (recur (rest es) (+ off size)))))
    (let [hjson (js/JSON.stringify hdr)
          pad   (mod (- 8 (mod (count hjson) 8)) 8)
          hbuf  (js/Buffer.from (str hjson (apply str (repeat pad " "))) "utf8")
          lbuf  (js/Buffer.alloc 8)]
      (.writeBigUInt64LE lbuf (js/BigInt (.-length hbuf)) 0)
      (.writeFileSync fs out-path (js/Buffer.concat (into-array (concat [lbuf hbuf] slices)))))))

(defn- split-safetensors!
  "Split st-path into n shards + index.json in out-dir (HF sharded layout)."
  [st-path out-dir n]
  (let [buf   (.readFileSync fs st-path)
        hlen  (js/Number (.readBigUInt64LE buf 0))
        hdr   (js/JSON.parse (.toString buf "utf8" 8 (+ 8 hlen)))
        blob0 (+ 8 hlen)
        names (->> (js-keys hdr)
                   (remove #(= % "__metadata__"))
                   (sort-by #(aget (.-data_offsets (unchecked-get hdr %)) 0))
                   vec)
        parts (vec (partition-all (js/Math.ceil (/ (count names) n)) names))
        total (count parts)
        wmap  (js/Object.)]
    (doseq [[i part] (map-indexed vector parts)]
      (let [fname (str "model-" (.padStart (str (inc i)) 5 "0")
                       "-of-" (.padStart (str total) 5 "0") ".safetensors")]
        (doseq [nm part] (unchecked-set wmap nm fname))
        (write-shard! (node-path/join out-dir fname)
                      (map (fn [nm] [nm (unchecked-get hdr nm)]) part)
                      buf blob0)))
    (.writeFileSync fs (node-path/join out-dir "model.safetensors.index.json")
                    (js/JSON.stringify #js {:metadata #js {:total_size (.-length buf)}
                                            :weight_map wmap}))
    total))

;; ---------------------------------------------------------------------------

(def ^:private src-dir (str (.-HOME js/process.env) "/.cache/models/qwen3.5-0.8b-mlx-bf16"))

(if-not (.existsSync fs (str src-dir "/model.safetensors"))
  (println "SKIP llm-sharded-load-test: qwen3.5-0.8b checkpoint absent")
  (let [tmp (.mkdtempSync fs (node-path/join (.tmpdir os) "genmlx-sharded-"))]
    (try
      (println "\n== sharded checkpoint loading (genmlx-sbif) ==")
      ;; precedence: the source dir ships BOTH layouts; single-file must win
      (assert-true "both-layouts dir resolves to the single file"
                   (= [(str src-dir "/model.safetensors")] (q3/weight-files src-dir)))
      (let [n (split-safetensors! (str src-dir "/model.safetensors") tmp 3)]
        (.copyFileSync fs (str src-dir "/config.json") (node-path/join tmp "config.json"))
        (assert-true "fixture: split into 3 shards" (= 3 n))
        (assert-true "sharded dir: loadable-weights?" (fwd/loadable-weights? tmp))
        (assert-true "sharded dir: fwd/supported?" (fwd/supported? tmp))
        (assert-true "sharded dir: weight-files resolves 3 shard paths"
                     (= 3 (count (q3/weight-files tmp))))
        (let [ms (fwd/load-model src-dir)
              mh (fwd/load-model tmp)
              ws (:weights ms)
              wh (:weights mh)]
          (assert-true "identical tensor name sets"
                       (= (set (keys ws)) (set (keys wh))))
          ;; every 40th tensor byte-identical after (dequantized) load
          (let [sample (take-nth 40 (sort (keys ws)))]
            (assert-true (str "sampled tensors identical (" (count sample) " checked)")
                         (every? (fn [k]
                                   (and (= (mx/shape (get ws k)) (mx/shape (get wh k)))
                                        (zero? (mx/item (mx/amax
                                                         (mx/abs (mx/subtract (get ws k)
                                                                              (get wh k))))))))
                                 sample)))
          ;; the forward actually runs end-to-end on the sharded load
          (let [ids [9707 11 1879]
                as  (mx/item (mx/argmax (fwd/next-token-logits ms ids)))
                ah  (mx/item (mx/argmax (fwd/next-token-logits mh ids)))]
            (assert-true (str "next-token argmax matches single-file load (" as ")")
                         (= as ah)))))
      (finally
        (.rmSync fs tmp #js {:recursive true :force true})))
    (println (str "\n=== llm-sharded-load: " @pass " PASS, " @fail " FAIL ==="))
    (when (pos? @fail) (js/process.exit 1))))
