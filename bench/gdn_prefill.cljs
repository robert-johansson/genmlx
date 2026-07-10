(ns bench.gdn-prefill
  "genmlx-ps8a I5 gate: owned 35B TEXT prefill cost with the fused GDN scan,
   swept over prefill chunk size — the re-sweep the fuse mandates (the old
   chunk=48 optimum existed only to bound the GDN host-loop graph).

   Reference numbers (genmlx-puip P0, 2026-07-10, pre-fuse, same T=624 sweep
   methodology): chunk 48 → 45.3 ms/token, 96 → 46.0, 192 → 54.6; native
   ~17.6 ms/token. Decode step at T=624: ~131 ms warm.

   Each config runs ONE timed full prefill after a small JIT-warmup prefill
   (matching the P0 methodology — per-config cold-shape instantiation is part
   of the measurement, the CUDA-JIT compile is not). Decode is timed from the
   last config's cache: first step reported separately (cold shape), then the
   mean of the warm steps.

   Output: results/gdn-prefill/data.json

   Usage (guarded, ONE GPU process):
     ~/genmlx-guarded-run.sh gdn-prefill-bench \\
       bunx --bun nbb@1.4.208 bench/gdn_prefill.cljs
   Env: GENMLX_GDN_BENCH_MODEL overrides the checkpoint dir;
        GENMLX_GDN_BENCH_T overrides the prompt length (default 624)."
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.qwen35-forward :as q35]
            [promesa.core :as pr]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/gdn-prefill")))

(defn- ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn- write-json [filename data]
  (ensure-dir out-dir)
  (let [filepath (str out-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

(def model-dir
  (or (aget (.-env js/process) "GENMLX_GDN_BENCH_MODEL")
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))

(def target-t
  (let [v (aget (.-env js/process) "GENMLX_GDN_BENCH_T")]
    (if v (js/parseInt v 10) 624)))

(defn- now [] (js/performance.now))

(defn- chunked-prefill!
  "Drive forward-cached over `ids` in `chunk`-token blocks with the per-chunk
   materialize boundary (the vlm-prefill discipline). Returns
   [last-logits cache elapsed-ms]."
  [fm ids chunk]
  (let [T (count ids)
        t0 (now)]
    (loop [start 0
           cache (q35/init-cache fm)
           logits nil]
      (if (>= start T)
        (do (mx/materialize! logits)
            [logits cache (- (now) t0)])
        (let [n (min chunk (- T start))
              block (subvec ids start (+ start n))
              [lg cache'] (q35/forward-cached fm block cache start)]
          (q35/materialize-cache! cache')
          (recur (+ start n) cache' lg))))))

(if-not (and model-dir (.existsSync fs model-dir))
  (do (println "SKIP — no 35B checkpoint at" (str model-dir)) (js/process.exit 1))
  (pr/let [mm (llm/load-model model-dir {:cljs-forward? true})
           {:keys [model tokenizer]} mm
           enc (llm/encode tokenizer
                           (apply str (repeat 80 "Probabilistic programs denote measures over traces, and inference is conditioning. ")))]
    (let [fm  (:fwd model)
          ids (vec (take target-t (vec enc)))
          T   (count ids)]
      (println "== ps8a gdn-prefill bench ==  model:" model-dir)
      (println "T =" T "tokens (requested" target-t ")")
      ;; JIT warmup: small chunked prefill, discarded.
      (println "-- warmup (chunk 48 over first 96 tokens) --")
      (chunked-prefill! fm (vec (take 96 ids)) 48)
      (mx/force-gc!)
      (let [configs (into [] (distinct [48 96 192 312 T]))
            rows
            (vec
             (for [chunk configs]
               (let [[_ _cache ms] (chunked-prefill! fm ids chunk)
                     per-tok (/ ms T)]
                 (mx/force-gc!)
                 (println (str "chunk " chunk ": " (.toFixed ms 0) " ms total, "
                               (.toFixed per-tok 1) " ms/token"))
                 {:chunk chunk :total-ms ms :ms-per-token per-tok})))
            ;; Decode regression check from a fresh chunk-48 prefill cache.
            [_ cache _] (chunked-prefill! fm ids 48)
            step-times
            (loop [i 0 cache cache offset T acc []]
              (if (= i 9)
                acc
                (let [t0 (now)
                      [lg cache'] (q35/forward-cached fm [42] cache offset)]
                  (mx/materialize! lg)
                  (q35/materialize-cache! cache')
                  (recur (inc i) cache' (inc offset) (conj acc (- (now) t0))))))
            warm (rest step-times)
            warm-mean (/ (reduce + warm) (count warm))]
        (println (str "decode @T=" T ": cold " (.toFixed (first step-times) 0)
                      " ms, warm mean " (.toFixed warm-mean 0) " ms over "
                      (count warm) " steps"))
        (write-json "data.json"
                    {:bean "genmlx-ps8a"
                     :model model-dir
                     :T T
                     :prefill rows
                     :decode {:cold-ms (first step-times)
                              :warm-mean-ms warm-mean
                              :warm-ms (vec warm)}
                     :reference {:pre-fuse-ms-per-token {:chunk48 45.3 :chunk96 46.0 :chunk192 54.6}
                                 :native-ms-per-token 17.6
                                 :pre-fuse-decode-warm-ms 131
                                 :source "genmlx-puip P0(b)(ii), 2026-07-10"}})
        (js/process.exit 0)))))
