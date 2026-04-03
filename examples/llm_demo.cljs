(ns examples.llm-demo
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [promesa.core :as pr]))

(def model-dir (str (.-HOME js/process.env) "/.cache/models"))

(println "Loading model...")
(pr/let
  [m       (llm/load-model (str model-dir "/qwen3-0.6b-mlx-bf16"))
   _       (println "Model loaded. Creating LLM generative function...")
   llm-gf  (llm-core/make-llm-gf m)
   tok     (:tokenizer m)
   raw-ids (llm/encode tok "The best programming language is")
   ids     (vec raw-ids)
   _       (println (str "Prompt: 'The best programming language is' (" (count ids) " tokens)\n"))]

  ;; 1. SIMULATE — sample freely from the LLM prior
  (println "== 1. Simulate (free generation) ==")
  (let [trace (p/simulate llm-gf [ids 10])]
    (pr/let [text (llm-core/decode-trace tok trace)]
      (println "  Text:" text)
      (println "  log p(sequence):" (.toFixed (mx/item (:score trace)) 2))))

  ;; 2. GENERATE — force " Clojure" at position 0
  (println "\n== 2. Constrain first token → ' Clojure' ==")
  (pr/let [clj-ids (llm/encode tok " Clojure")]
    (let [clj-id (first (vec clj-ids))
          result (p/generate llm-gf [ids 10]
                   (cm/set-value cm/EMPTY :t0 (mx/scalar clj-id mx/int32)))]
      (pr/let [text (llm-core/decode-trace tok (:trace result))]
        (println "  Text:" text)
        (println "  log p(constraint):" (.toFixed (mx/item (:weight result)) 4)))))

  ;; 3. GENERATE — force " Rust" at position 0
  (println "\n== 3. Constrain first token → ' Rust' ==")
  (pr/let [rust-ids (llm/encode tok " Rust")]
    (let [rust-id (first (vec rust-ids))
          result (p/generate llm-gf [ids 10]
                   (cm/set-value cm/EMPTY :t0 (mx/scalar rust-id mx/int32)))]
      (pr/let [text (llm-core/decode-trace tok (:trace result))]
        (println "  Text:" text)
        (println "  log p(constraint):" (.toFixed (mx/item (:weight result)) 4)))))

  ;; 4. Compare languages — direct log-prob from the model
  ;; Note: after "...is", the next token has a leading space
  (println "\n== 4. Which language does the model prefer? ==")
  (pr/let [lp (llm/next-token-logprobs (:model m) ids)
           _  (mx/eval! lp)
           ;; Encode each to get the first token ID (handles space prefix)
           pairs (pr/all
                   (mapv (fn [lang]
                           (pr/let [enc (llm/encode tok (str " " lang))]
                             [lang (first (vec enc))]))
                         ["Python" "Rust" "Java" "Haskell"
                          "C" "Go" "Lisp" "JavaScript" "Zig"]))]
    (let [scores (mapv (fn [[lang tid]]
                         [lang (mx/item (mx/take-idx lp (mx/scalar tid mx/int32)))])
                       pairs)
          sorted (sort-by second > scores)]
      (doseq [[lang s] sorted]
        (println (str "  " lang ": log p = " (.toFixed s 4))))))

  (println "\nDone."))
