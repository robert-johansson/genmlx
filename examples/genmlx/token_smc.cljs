(ns genmlx.token-smc-example
  "Token-SMC demo (genmlx-5qk7): grammar-twisted synthesis on the resident
   model — N particles over one prompt pay prefill once, decode under a
   regex-DFA twist, and resample by weight. The 'second path' in one page:
   many cheap partial hypotheses, principled reallocation, no replay.

   Requires the dense 0.8B (or any llm/load-model-able checkpoint):
     bunx --bun nbb@1.4.208 examples/genmlx/token_smc.cljs [model-dir]"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.smc :as tsmc]
            [promesa.core :as pr]
            ["os" :as os]
            ["path" :as path]))

(def model-dir
  (or (second (.slice js/process.argv 2))
      (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")))

(pr/let [mm (llm/load-model model-dir)
         {:keys [model tokenizer]} mm
         enc (llm/encode tokenizer "Call me at ")]
  (pr/let [;; the twist: completions must look like a US phone number
        constraint (gram/compile-constraint tokenizer "[0-9]{3}-[0-9]{4}")
        prompt (vec enc)
        decoder (tsmc/decoder-for mm)
        _ (println "decoder:" (if (llm/supports-branching? model) "native branches" "replay")
                   "| particles pay prefill ONCE, then fork")
        result0 (tsmc/token-smc {:particles 8
                                :max-tokens 10
                                :eos-id (llm/eos-token-id tokenizer)
                                :proposal :grammar-masked
                                :constraint constraint
                                :decoder decoder
                                :key (rng/fresh-key 2026)
                                :callback (fn [{:keys [step ess resampled?]}]
                                            (println (str "  step " step
                                                          "  ESS " (.toFixed ess 2)
                                                          (when resampled? "  [resampled]"))))}
                               mm prompt)
        result (tsmc/decode-particles! mm result0)]
    (println "\nlog-ML estimate:" (mx/realize (:log-ml-estimate result)))
    (doseq [[i pt] (map-indexed vector (:particles result))]
      (println (str "  particle " i ": " (pr-str (:text pt))
                    "  log-w " (.toFixed (mx/realize (:log-w pt)) 3))))))
