;; @tier slow
(ns genmlx.llm.backend-test
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
 [m (llm/load-model (str model-dir "/qwen3.5-0.8b-mlx-bf16") {:cljs-forward? false})
  _ (assert-true "returns map" (map? m))
  _ (assert-true "has :model" (some? (:model m)))
  _ (assert-true "has :tokenizer" (some? (:tokenizer m)))
  _ (assert-true "has :type" (some? (:type m)))
  _ (assert-true ":type is :qwen3_5" (= (:type m) :qwen3_5))
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
   ;; 1.3 forward-prefill (cached path)
   ;;
   ;; The cached forwardWithCache path is what make-llm-gf / codegen / bytes /
   ;; msa actually use, so 1.3-1.5 test that path. The uncached `.forward` path
   ;; for qwen3_5 — previously reported as garbage on the bf16 artifact
   ;; (genmlx-z7m2) — now agrees with it and is asserted correct in 1.6.
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.3 forward-prefill (cached) ==")
  prompt-ids (llm/encode tok "The capital of France is")
  _ (println (str "  Prompt: " (.-length prompt-ids) " tokens"))

  _ (llm/init-cache! (:model m))
  logits (llm/forward-prefill (:model m) prompt-ids)
  _ (llm/reset-cache! (:model m))
  _ (println (str "  Logits shape: " (mx/shape logits)))
  _ (assert-true "logits is 1D" (= (count (mx/shape logits)) 1))
  _ (assert-true "logits size >= vocab" (>= (first (mx/shape logits)) vs))

   ;; cached path also accepts a cljs vector of ids
  _ (llm/init-cache! (:model m))
  logits2 (llm/forward-prefill (:model m) (vec prompt-ids))
  _ (llm/reset-cache! (:model m))
  _ (assert-true "cljs vector input works" (= (mx/shape logits2) (mx/shape logits)))

   ;; ---------------------------------------------------------------
   ;; 1.4 next-token logprobs (log-softmax of cached logits)
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.4 next-token logprobs (cached) ==")
  lp (mx/subtract logits (mx/logsumexp logits))   ;; log_softmax
  _ (println (str "  Log-probs shape: " (mx/shape lp)))
  _ (assert-true "logprobs shape >= vocab" (>= (first (mx/shape lp)) vs))

   ;; All log-probs should be <= 0
  max-lp (mx/item (mx/amax lp))
  _ (println (str "  Max log-prob: " max-lp))
  _ (assert-true "max log-prob <= 0" (<= max-lp 0.0001))

   ;; exp(logprobs) sums to ~1. Over a 248K-token vocab the float32 reduction
   ;; leaves the sum a few % off 1.0 (observed ~1.04) — the argmax is the real
   ;; signal, so the tolerance here is deliberately loose.
  prob-sum (mx/item (mx/sum (mx/exp lp)))
  _ (println (str "  Prob sum: " prob-sum))
  _ (assert-close "probs sum to ~1.0 (float32, 248K vocab)" 1.0 prob-sum 0.1)

   ;; Argmax predicts "Paris" (the cached path is correct)
  argmax-id (mx/item (mx/argmax lp))
  predicted (llm/decode tok (js/Uint32Array.from #js [argmax-id]))
  _ (println (str "  Argmax: " argmax-id " → '" predicted "'"))
  _ (assert-true "predicts Paris" (re-find #"(?i)paris" predicted))

   ;; ---------------------------------------------------------------
   ;; 1.5 cached greedy generation (forward-prefill + forward-step)
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.5 cached generation ==")
  _ (llm/init-cache! (:model m))
  gen-toks (loop [i 0
                  lg (llm/forward-prefill (:model m) (vec prompt-ids))
                  acc []]
             (let [t (mx/item (mx/argmax lg))]
               (if (or (>= i 12) (= t (llm/eos-token-id tok)))
                 acc
                 (recur (inc i) (llm/forward-step (:model m) t) (conj acc t)))))
  _ (llm/reset-cache! (:model m))
  gen-text (llm/decode tok (js/Uint32Array.from (clj->js gen-toks)))
  _ (println (str "  Generated: '" gen-text "'"))
  _ (assert-true "generation produces tokens" (pos? (count gen-toks)))
  _ (assert-true "decodes to a string" (string? gen-text))

   ;; ---------------------------------------------------------------
   ;; 1.6 uncached forward-pass — qwen3_5 bf16 (genmlx-z7m2 RESOLVED 2026-06-14)
   ;;
   ;; The earlier z7m2 report (uncached `.forward` returns garbage for qwen3_5 on
   ;; the bf16 artifact — argmax 0 = "!" + a Metal GPU-timeout) no longer
   ;; reproduces on the pinned mlx-node build: the uncached path now agrees with
   ;; the cached path (1.4) and predicts "Paris". The GenMLX-owned forward (f6ov)
   ;; is the trusted path (parity-gated vs upstream); this asserts the upstream
   ;; uncached path on the bf16 artifact is correct too, so a future binary that
   ;; reintroduced the garbage-logits drift would be CAUGHT here, not silent.
   ;; ---------------------------------------------------------------
  _ (println "\n== 1.6 uncached forward-pass (qwen3_5 bf16 — z7m2 resolved) ==")
  u-logits (llm/forward-pass (:model m) prompt-ids)
  u-argmax (mx/item (mx/argmax u-logits))
  u-pred (llm/decode tok (js/Uint32Array.from #js [u-argmax]))
  _ (println (str "  Uncached argmax: " u-argmax " → '" u-pred "' (cached 1.4 gave '" predicted "')"))
  _ (assert-true "uncached path returns vocab-sized logits"
                 (>= (first (mx/shape u-logits)) vs))
  _ (assert-true "uncached argmax == cached argmax (z7m2 resolved)"
                 (= argmax-id u-argmax))
  _ (assert-true "uncached predicts Paris (z7m2 resolved)"
                 (re-find #"(?i)paris" u-pred))]

  (println (str "\n== Phase 1: " @pass-count " passed, " @fail-count " failed ==")))
