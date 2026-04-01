(ns genmlx.llm-backend-test
  "Phase 1: Test genmlx.llm.backend — model loading, tokenizer, forward pass,
   log-probs, and text generation."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.mlx :as mx]
            [promesa.core :as p]))

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc) (println (str "  FAIL: " label)))))

(defn assert-close [label expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc) (println (str "  PASS: " label " (diff=" (.toFixed diff 6) ")")))
      (do (swap! fail-count inc) (println (str "  FAIL: " label " expected=" expected " actual=" actual))))))

(def model-dir (str (.-HOME js/process.env) "/.cache/models"))

;; ---------------------------------------------------------------
;; 1.1 load-model
;; ---------------------------------------------------------------
(println "\n== 1.1 load-model ==")

(p/let
 [m (llm/load-model (str model-dir "/qwen3-0.6b-mlx-bf16"))
  _ (assert-true "returns map" (map? m))
  _ (assert-true "has :model" (some? (:model m)))
  _ (assert-true "has :tokenizer" (some? (:tokenizer m)))
  _ (assert-true "has :type" (some? (:type m)))
  _ (assert-true ":type is :qwen3" (= (:type m) :qwen3))
  _ (println (str "  Type: " (:type m)))

   ;; ---------------------------------------------------------------
   ;; 1.2 Tokenizer wrappers
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.2 Tokenizer wrappers ==")
  tok (:tokenizer m)

   ;; vocab-size
  vs (llm/vocab-size tok)
  _ (println (str "  Vocab size: " vs))
  _ (assert-true "vocab-size > 100k" (> vs 100000))

   ;; eos-token-id
  eos (llm/eos-token-id tok)
  _ (println (str "  EOS token ID: " eos))
  _ (assert-true "eos is number" (number? eos))

   ;; pad-token-id
  pad (llm/pad-token-id tok)
  _ (println (str "  PAD token ID: " pad))
  _ (assert-true "pad is number" (number? pad))

   ;; encode / decode round-trip
  ids (llm/encode tok "Hello, world!")
  _ (println (str "  Encoded: " (vec ids)))
  _ (assert-true "encode returns Uint32Array" (instance? js/Uint32Array ids))
  _ (assert-true "encode non-empty" (> (.-length ids) 0))
  text (llm/decode tok ids)
  _ (assert-true "round-trip" (= text "Hello, world!"))

   ;; id->token / token->id
  _ (let [tok-str (llm/id->token tok (first (vec ids)))]
      (println (str "  Token 0: id=" (first (vec ids)) " → '" tok-str "'"))
      (assert-true "id->token non-nil" (some? tok-str)))

   ;; encode with special tokens
  ids-special (llm/encode tok "Hi" true)
  ids-plain (llm/encode tok "Hi" false)
  _ (println (str "  With special: " (.-length ids-special) " tokens, without: " (.-length ids-plain)))
  _ (assert-true "special tokens adds tokens" (>= (.-length ids-special) (.-length ids-plain)))

   ;; ---------------------------------------------------------------
   ;; 1.3 forward-pass
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.3 forward-pass ==")
  prompt-ids (llm/encode tok "The capital of France is")
  _ (println (str "  Prompt: " (.-length prompt-ids) " tokens"))

  logits (llm/forward-pass (:model m) prompt-ids)
  _ (println (str "  Logits shape: " (mx/shape logits)))
  _ (assert-true "logits is 1D" (= (count (mx/shape logits)) 1))
  _ (assert-true "logits size >= vocab" (>= (first (mx/shape logits)) vs))

   ;; forward-pass also works with a cljs vector
  logits2 (llm/forward-pass (:model m) (vec prompt-ids))
  _ (assert-true "cljs vector input works" (= (mx/shape logits2) (mx/shape logits)))

   ;; ---------------------------------------------------------------
   ;; 1.4 next-token-logprobs
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.4 next-token-logprobs ==")
  lp (llm/next-token-logprobs (:model m) prompt-ids)
  _ (println (str "  Log-probs shape: " (mx/shape lp)))
  _ (assert-true "logprobs shape >= vocab" (>= (first (mx/shape lp)) vs))

   ;; All log-probs should be <= 0
  max-lp (mx/item (mx/amax lp))
  _ (println (str "  Max log-prob: " max-lp))
  _ (assert-true "max log-prob <= 0" (<= max-lp 0.0001))

   ;; exp(logprobs) should sum to ~1.0
  prob-sum (mx/item (mx/sum (mx/exp lp)))
  _ (println (str "  Prob sum: " prob-sum))
  _ (assert-close "probs sum to 1.0" 1.0 prob-sum 0.01)

   ;; Argmax should predict "Paris"
  argmax-id (mx/item (mx/argmax lp))
  predicted (llm/decode tok (js/Uint32Array.from #js [argmax-id]))
  _ (println (str "  Argmax: " argmax-id " → '" predicted "'"))
  _ (assert-true "predicts Paris" (re-find #"(?i)paris" predicted))

   ;; ---------------------------------------------------------------
   ;; 1.5 generate-text
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.5 generate-text ==")
  response (llm/generate-text m "What is 2+2? Reply with just the number."
                              {:max-tokens 20 :temperature 0})
  _ (println (str "  Response: '" response "'"))
  _ (assert-true "returns string" (string? response))
  _ (assert-true "non-empty" (> (count response) 0))

   ;; ---------------------------------------------------------------
   ;; 1.6 Load VLM (Qwen3.5)
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.6 Load VLM ==")
  vlm (llm/load-model (str model-dir "/qwen3.5-0.8b-mlx-bf16"))
  _ (assert-true "VLM loaded" (some? (:model vlm)))
  _ (assert-true "VLM type is :qwen3_5" (= (:type vlm) :qwen3_5))
  _ (println (str "  VLM type: " (:type vlm)))

   ;; VLM forward pass
  vlm-ids (llm/encode (:tokenizer vlm) "Hello")
  vlm-logits (llm/forward-pass (:model vlm) vlm-ids)
  _ (println (str "  VLM logits shape: " (mx/shape vlm-logits)))
  _ (assert-true "VLM logits 1D" (= (count (mx/shape vlm-logits)) 1))]

  (println (str "\n== Phase 1: " @pass-count " passed, " @fail-count " failed ==")))
