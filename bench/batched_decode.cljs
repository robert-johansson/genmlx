(ns bench.batched-decode
  "genmlx-9uyg payoff number: lockstep [K]-lane decode cost vs K on the owned
   forward. Decode is weight-bandwidth-bound, so K lanes through one batched
   step should cost ~1 lane — the reason Route B (GFI-native particles) beats
   K sequential scalar decodes.

   Methodology: one B=1 prefill; for each K — tile via a fresh
   forward-step-batched sequence (init/prefill per K so lazy tiling is
   inside the measured shape, but a JIT warmup step is not) — time N-STEPS
   lockstep steps feeding constant tokens, report ms/step and the scaling
   ratio vs K=1. Per-step mx/eval! on the logits (the decode boundary).

   Output: results/batched-decode/data.json

   Usage (guarded, ONE GPU process):
     ~/genmlx-guarded-run.sh batched-decode-bench \\
       bunx --bun nbb@1.4.208 bench/batched_decode.cljs
   Env: GENMLX_BATCH_BENCH_MODEL overrides the checkpoint dir (default
        ~/.cache/models/qwen3.5-0.8b-mlx-bf16);
        GENMLX_BATCH_BENCH_KS comma-separated K sweep (default 1,2,4,8)."
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [promesa.core :as pr]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/batched-decode")))

(defn- ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn- write-json [filename data]
  (ensure-dir out-dir)
  (let [p (.join path-mod out-dir filename)]
    (.writeFileSync fs p (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote " p))))

(def model-dir
  (or (aget (.-env js/process) "GENMLX_BATCH_BENCH_MODEL")
      (str (.-HOME js/process.env) "/.cache/models/qwen3.5-0.8b-mlx-bf16")))

(def ks
  (if-let [s (aget (.-env js/process) "GENMLX_BATCH_BENCH_KS")]
    (mapv js/parseInt (.split s ","))
    [1 2 4 8]))

(def prompt "The capital of France is")
(def N-STEPS 32)
(def WARMUP 4)

(defn- now [] (js/performance.now))

(defn- bench-k
  "ms/step for K lockstep lanes: prefill B=1, warm up (tiles + JIT), then
   time N-STEPS batched steps, eval!-ing each step's logits."
  [model ids k tok]
  (llm/init-cache! model)
  (llm/forward-prefill model ids)
  (let [tok-k (mx/array (vec (repeat k tok)) [k] mx/int32)]
    (dotimes [_ WARMUP] (mx/eval! (llm/forward-step-batched model tok-k)))
    (let [t0 (now)]
      (dotimes [_ N-STEPS] (mx/eval! (llm/forward-step-batched model tok-k)))
      (let [ms (/ (- (now) t0) N-STEPS)]
        (llm/reset-cache! model)
        (mx/force-gc!)
        ms))))

(pr/let [m (llm/load-model model-dir {:cljs-forward? true})
         ids-raw (llm/encode (:tokenizer m) prompt false)]
  (let [ids (vec ids-raw)
        model (:model m)]
    (llm/init-cache! model)
    (let [tok (mx/item (mx/argmax (llm/forward-prefill model ids)))]
      (llm/reset-cache! model)
      (println (str "== batched-decode bench ==  model=" model-dir
                    "  Ks=" (pr-str ks) "  steps/K=" N-STEPS))
      (let [rows (mapv (fn [k]
                         (let [ms (bench-k model ids k tok)]
                           (println (str "  K=" k "  " (.toFixed ms 2) " ms/step  ("
                                         (.toFixed (/ ms k) 2) " ms/lane-step)"))
                           {:k k :ms-per-step ms :ms-per-lane-step (/ ms k)}))
                       ks)
            base (:ms-per-step (first rows))]
        (doseq [{:keys [k ms-per-step]} rows]
          (println (str "  K=" k " costs " (.toFixed (/ ms-per-step base) 2)
                        "x the K=" (:k (first rows)) " step")))
        (write-json "data.json"
                    {:model model-dir :prompt-tokens (count ids)
                     :n-steps N-STEPS :warmup WARMUP :rows rows})))))
