(ns fim-demo
  "Fill-in-the-Middle demo with Qwen2.5-Coder-0.5B via GenMLX.
   Shows FIM generation, scoring completions, and ranking candidates."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]))

(def model-path
  (or (first *command-line-args*)
      (str (.-HOME js/process.env) "/.cache/models/Qwen2.5-Coder-0.5B-mlx")))

(defn fim-prompt
  "Build a FIM prompt: prefix <|fim_suffix|> suffix <|fim_middle|>"
  [prefix suffix]
  (str "<|fim_prefix|>" prefix "<|fim_suffix|>" suffix "<|fim_middle|>"))

(defn score-completion
  "Score a candidate completion under the FIM model.
   Returns {:text :log-weight :perplexity-per-token}."
  [gf tokenizer fim-ids completion-text]
  (pr/let [comp-ids-raw (llm/encode tokenizer completion-text false)
           comp-ids (vec comp-ids-raw)
           n (count comp-ids)
           constraints (reduce (fn [m i]
                                 (cm/set-value m (keyword (str "t" i))
                                               (mx/scalar (nth comp-ids i) mx/int32)))
                               (cm/choicemap)
                               (range n))
           result (p/generate gf [fim-ids n] constraints)
           w (mx/item (:weight result))]
    {:text completion-text
     :log-weight w
     :per-token (/ w n)
     :n-tokens n}))

(println "\n╔══════════════════════════════════════════════════╗")
(println "║   GenMLX Fill-in-the-Middle Demo                ║")
(println "║   Qwen2.5-Coder-0.5B · Raw Token Path · GFI    ║")
(println "╚══════════════════════════════════════════════════╝")

(pr/let [_ (println "\nLoading model...")
         bundle (llm/load-model model-path)
         gf (llm-core/make-llm-gf bundle)
         tok (:tokenizer bundle)
         _ (println "Model loaded.\n")

         ;; === Demo 1: Generate FIM completions ===
         _ (println "━━━ Demo 1: FIM Generation ━━━")
         _ (println "  Prefix: (defn factorial [n]")
         _ (println "  Suffix: )")
         _ (println "  (model fills in the body)\n")

         prompt1 (fim-prompt "(defn factorial [n]\n  " ")")
         ids1-raw (llm/encode tok prompt1 false)
         ids1 (vec ids1-raw)

         trace1 (p/simulate gf [ids1 40])
         text1 (llm-core/decode-trace tok trace1)
         _ (println "  Generated:" (pr-str text1))

         ;; === Demo 2: Score competing completions ===
         _ (println "\n━━━ Demo 2: Score Competing Completions ━━━")
         _ (println "  Prefix: (defn add [a b]")
         _ (println "  Suffix: )")
         _ (println "  Which body does the model prefer?\n")

         prompt2 (fim-prompt "(defn add [a b]\n  " ")")
         ids2-raw (llm/encode tok prompt2 false)
         ids2 (vec ids2-raw)

         candidates ["(+ a b)"
                     "(- a b)"
                     "(* a b)"
                     "(str a b)"
                     "(/ a b)"]

         scores (pr/all (mapv #(score-completion gf tok ids2 %) candidates))
         ranked (sort-by :log-weight > scores)
         _ (doseq [{:keys [text log-weight per-token n-tokens]} ranked]
             (println (str "  " (.toFixed log-weight 2)
                           "  (" (.toFixed per-token 2) "/tok, " n-tokens " toks)"
                           "  " (pr-str text))))

         ;; === Demo 3: ClojureScript-specific FIM ===
         _ (println "\n━━━ Demo 3: ClojureScript Pattern Completion ━━━")
         _ (println "  Prefix: (let [results (map")
         _ (println "  Suffix:       data)]")
         _ (println "  What mapping function?\n")

         prompt3 (fim-prompt "(let [results (map " "\n                    data)]")
         ids3-raw (llm/encode tok prompt3 false)
         ids3 (vec ids3-raw)

         cljs-candidates ["#(inc %)"
                          "#(str %)"
                          "#(* % %)"
                          "identity"
                          "first"]
         scores3 (pr/all (mapv #(score-completion gf tok ids3 %) cljs-candidates))
         ranked3 (sort-by :log-weight > scores3)
         _ (doseq [{:keys [text log-weight per-token n-tokens]} ranked3]
             (println (str "  " (.toFixed log-weight 2)
                           "  (" (.toFixed per-token 2) "/tok, " n-tokens " toks)"
                           "  " (pr-str text))))

         ;; === Demo 4: Multi-line infill ===
         _ (println "\n━━━ Demo 4: Multi-line Infill ━━━")
         _ (println "  Prefix: (defn process-items [items]")
         _ (println "  Suffix: (println \"Done processing\" (count results)))")
         _ (println)

         prompt4 (fim-prompt
                   "(defn process-items [items]\n  "
                   "\n  (println \"Done processing\" (count results)))")
         ids4-raw (llm/encode tok prompt4 false)
         ids4 (vec ids4-raw)
         trace4 (p/simulate gf [ids4 50])
         text4 (llm-core/decode-trace tok trace4)
         _ (println "  Generated:" (pr-str text4))]

  (println "\n━━━ Summary ━━━")
  (println "  FIM works by encoding prefix + suffix with special tokens")
  (println "  The model generates the middle via raw token path (no chat template)")
  (println "  p/generate scores any candidate completion under the model's distribution")
  (println "  This enables: ranking, SMC, grammar-constrained FIM, and more"))
