(ns genmlx.llm-cache-benchmark
  "Phase 2.3: Benchmark KV cache speedup for LLM generation."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [promesa.core :as pr]))

(def model-dir (str (.-HOME js/process.env) "/.cache/models"))

(defn bench [label n f]
  (dotimes [_ 3] (f))
  (let [start (.now js/performance)]
    (dotimes [_ n] (f))
    (let [elapsed (- (.now js/performance) start)
          per-op  (/ elapsed n)]
      (println (str "  " label ": " (.toFixed per-op 1) "ms/call"
                    " (" n " iters, " (.toFixed elapsed 0) "ms total)"))
      per-op)))

(println "Loading model...")
(pr/let
  [m        (llm/load-model (str model-dir "/qwen3-0.6b-mlx-bf16"))
   tok      (:tokenizer m)
   raw-ids  (llm/encode tok "The capital of France is")
   ids      (vec raw-ids)
   _        (println (str "Prompt: " (count ids) " tokens\n"))

   cached   (llm-core/make-llm-gf m)
   uncached (llm-core/make-llm-gf-uncached m)]

  ;; Verify correctness: same tokens from both paths
  (let [c-trace (p/simulate (dyn/with-key cached (rng/fresh-key 42)) [ids 5])
        u-trace (p/simulate (dyn/with-key uncached (rng/fresh-key 42)) [ids 5])]
    (assert (= (:retval c-trace) (:retval u-trace))
            "cached and uncached must produce identical tokens"))
  (println "Correctness check: cached = uncached ✓\n")

  ;; Benchmark across sequence lengths
  (let [key-fn #(dyn/with-key % (rng/fresh-key 42))]
    (doseq [{:keys [tokens iters]} [{:tokens 5  :iters 5}
                                     {:tokens 20 :iters 3}
                                     {:tokens 50 :iters 2}]]
      (println (str "== Generate " tokens " tokens =="))
      (let [t-cached   (bench "cached  " iters
                              #(p/simulate (key-fn cached) [ids tokens]))
            t-uncached (bench "uncached" iters
                              #(p/simulate (key-fn uncached) [ids tokens]))]
        (println (str "  Speedup: " (.toFixed (/ t-uncached t-cached) 1) "x\n")))))

  (println "Done."))
