(ns genmlx.qwen2-coder-test
  "Test Qwen2.5-Coder-0.5B loading and generation via qwen2→qwen3 path."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]))

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc) (println (str "  FAIL: " label)))))

(def model-path (str (.-HOME js/process.env) "/.cache/models/Qwen2.5-Coder-0.5B-mlx"))

(println "\n== Qwen2.5-Coder-0.5B via qwen2→qwen3 path ==\n")

(pr/let [;; Test 1: Model loads successfully
         _ (println "-- 1. Model loading --")
         bundle (llm/load-model model-path)
         _ (assert-true "model loaded" (some? (:model bundle)))
         _ (assert-true "tokenizer loaded" (some? (:tokenizer bundle)))
         _ (assert-true "type is :qwen2" (= :qwen2 (:type bundle)))
         _ (println "  Model type:" (:type bundle))

         ;; Test 2: Tokenizer works
         _ (println "\n-- 2. Tokenizer --")
         tokens (llm/encode (:tokenizer bundle) "(defn hello [name]" true)
         _ (assert-true "tokenizer encodes" (> (.-length tokens) 0))
         _ (println "  Token count:" (.-length tokens))

         decoded (llm/decode (:tokenizer bundle) tokens)
         _ (assert-true "tokenizer decodes" (string? decoded))
         _ (println "  Decoded:" (pr-str decoded))

         ;; Test 3: FIM tokens exist in tokenizer
         _ (println "\n-- 3. FIM token check --")
         fim-prefix (llm/encode (:tokenizer bundle) "<|fim_prefix|>" false)
         fim-middle (llm/encode (:tokenizer bundle) "<|fim_middle|>" false)
         fim-suffix (llm/encode (:tokenizer bundle) "<|fim_suffix|>" false)
         _ (assert-true "fim_prefix encodes to single token" (= 1 (.-length fim-prefix)))
         _ (assert-true "fim_middle encodes to single token" (= 1 (.-length fim-middle)))
         _ (assert-true "fim_suffix encodes to single token" (= 1 (.-length fim-suffix)))
         _ (println "  FIM token IDs — prefix:" (aget fim-prefix 0)
                    "middle:" (aget fim-middle 0)
                    "suffix:" (aget fim-suffix 0))

         ;; Test 4: Raw FIM generation via make-llm-gf (no chat template)
         _ (println "\n-- 4. FIM generation via raw token path --")
         gf (llm-core/make-llm-gf bundle)
         fim-prompt "<|fim_prefix|>(defn greet [name]\n  <|fim_suffix|>)\n<|fim_middle|>"
         fim-ids-raw (llm/encode (:tokenizer bundle) fim-prompt false)
         fim-ids (vec fim-ids-raw)
         _ (println "  FIM prompt tokens:" (count fim-ids))
         trace (p/simulate gf [fim-ids 20])
         fim-text (llm-core/decode-trace (:tokenizer bundle) trace)
         _ (assert-true "FIM produced text via raw path" (and (string? fim-text)
                                                              (> (count fim-text) 0)))
         _ (println "  FIM infill:" (pr-str fim-text))

         ;; Test 5: FIM via p/generate (constrained generation)
         _ (println "\n-- 5. FIM with p/generate (score a completion) --")
         target-ids-raw (llm/encode (:tokenizer bundle) "(str \"Hello, \" name)" false)
         target-ids (vec target-ids-raw)
         _ (println "  Target tokens:" (count target-ids) "ids:" target-ids)
         constraints (reduce (fn [m i]
                               (cm/set-value m (keyword (str "t" i))
                                             (mx/scalar (nth target-ids i) mx/int32)))
                             (cm/choicemap)
                             (range (count target-ids)))
         gen-result (p/generate gf [fim-ids (count target-ids)] constraints)
         _ (assert-true "p/generate returned trace" (some? (:trace gen-result)))
         _ (assert-true "p/generate returned weight" (some? (:weight gen-result)))
         weight-val (mx/item (:weight gen-result))
         _ (assert-true "weight is finite" (js/isFinite weight-val))
         _ (println "  Log-weight:" weight-val)
         gen-text (llm-core/decode-trace (:tokenizer bundle) (:trace gen-result))
         _ (assert-true "constrained text matches target" (= gen-text "(str \"Hello, \" name)"))
         _ (println "  Constrained output:" (pr-str gen-text))]

  (println (str "\n== Results: " @pass-count " passed, " @fail-count " failed =="))
  (when (pos? @fail-count)
    (js/process.exit 1)))
