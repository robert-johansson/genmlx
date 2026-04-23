(ns fim-demo-temp
  "FIM demo with temperature control. Compares temp=0.7 vs temp=0.1."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def model-path
  (or (first *command-line-args*)
      (str (.-HOME js/process.env) "/.cache/models/qwen25-coder-3b-cljs-fused")))

(defn make-llm-gf-temp
  "Like make-llm-gf but with temperature control.
   Temperature < 1 sharpens the distribution (more deterministic).
   Temperature > 1 flattens it (more random)."
  [model-map temperature]
  (let [{:keys [model tokenizer]} model-map
        eos (llm/eos-token-id tokenizer)]
    (dyn/auto-key
     (gen [prompt-ids max-tokens]
          (if (zero? max-tokens)
            prompt-ids
            (do
              (llm/init-cache! model)
              (try
                (let [logits (mx/divide (llm/forward-prefill model prompt-ids)
                                        (mx/scalar temperature))]
                  (loop [i 0, context prompt-ids, logits logits]
                    (if (>= i max-tokens)
                      context
                      (let [tok (trace (keyword (str "t" i)) (dist/categorical logits))
                            tok-id (mx/item tok)]
                        (if (= tok-id eos)
                          (conj context tok-id)
                          (let [next-logits (mx/divide (llm/forward-step model tok-id)
                                                       (mx/scalar temperature))]
                            (recur (inc i) (conj context tok-id) next-logits)))))))
                (finally
                  (llm/reset-cache! model)))))))))

(defn fim-prompt [prefix suffix]
  (str "<|fim_prefix|>" prefix "<|fim_suffix|>" suffix "<|fim_middle|>"))

(defn generate-fim [gf tok prefix suffix max-tokens]
  (pr/let [prompt (fim-prompt prefix suffix)
           ids-raw (llm/encode tok prompt false)
           ids (vec ids-raw)
           trace (p/simulate gf [ids max-tokens])
           text (llm-core/decode-trace tok trace)]
    text))

(println "\n╔══════════════════════════════════════════════════╗")
(println "║   FIM Temperature Comparison                    ║")
(println "║   Qwen2.5-Coder-3B + ClojureScript LoRA        ║")
(println "╚══════════════════════════════════════════════════╝")

(pr/let [_ (println "\nLoading model...")
         bundle (llm/load-model model-path)
         tok (:tokenizer bundle)
         _ (println "Model loaded.\n")

         gf-hot  (make-llm-gf-temp bundle 0.7)
         gf-warm (make-llm-gf-temp bundle 0.3)
         gf-cold (make-llm-gf-temp bundle 0.1)

         run-one (fn [prefix suffix label]
                   (println (str "━━━ " label " ━━━"))
                   (println (str "  prefix: " (pr-str (subs prefix 0 (min 50 (count prefix))))))
                   (println (str "  suffix: " (pr-str suffix)))
                   (pr/let [t07 (generate-fim gf-hot tok prefix suffix 40)
                            _ (println (str "  temp=0.7: " (pr-str t07)))
                            t03 (generate-fim gf-warm tok prefix suffix 40)
                            _ (println (str "  temp=0.3: " (pr-str t03)))
                            t01 (generate-fim gf-cold tok prefix suffix 40)
                            _ (println (str "  temp=0.1: " (pr-str t01)))]
                     (println)))]

  (pr/let [_ (run-one "(defn factorial [n]\n  " ")" "factorial body")
           _ (run-one "(defn greet [name]\n  " ")" "greet body")
           _ (run-one "(let [data [1 2 3 4 5]\n        result " "]\n    (println result))" "let binding")
           _ (run-one "(defn filter-even [xs]\n  " "\n  (println \"count:\" (count results)))" "filter-even body")
           _ (run-one "(defn parse-input [s]\n  " ")" "parse-input body")]
    (println "Done.")))
