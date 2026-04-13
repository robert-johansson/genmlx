(ns genmlx.llm-gemma4-test
  "P1-1: Test Gemma4 E2B as generative function.
   Covers model loading, tokenizer, forward pass, KV cache, GFI ops,
   grammar constraints, and byte-level generation."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.llm.grammar :as grammar]
            [genmlx.llm.bytes :as bytes]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dist :as dist]
            [promesa.core :as pm]))

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [label v]
  (if v
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc) (println (str "  FAIL: " label)))))

(defn- assert-equal [label expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc) (println (str "  FAIL: " label " expected=" expected " actual=" actual)))))

(def model-dir (str (.-HOME js/process.env) "/.cache/models/gemma4-e2b-it-4bit-safe"))

;; ---------------------------------------------------------------
;; 1. Model loading
;; ---------------------------------------------------------------
(println "\n== 1. Model loading ==")

(pm/let
 [m (llm/load-model model-dir)]

 (assert-true "load-model returns map" (map? m))
 (assert-true "has :model key" (some? (:model m)))
 (assert-true "has :tokenizer key" (some? (:tokenizer m)))
 (assert-equal "type is :gemma4" :gemma4 (:type m))

 ;; ---------------------------------------------------------------
 ;; 2. Tokenizer
 ;; ---------------------------------------------------------------
 (println "\n== 2. Tokenizer ==")

 (let [tok (:tokenizer m)]
   (assert-equal "vocab size is 262144" 262144 (llm/vocab-size tok))
   (assert-true "eos token ID is a number" (number? (llm/eos-token-id tok)))

   (pm/let [ids (.encode tok "Hello world" false)
            decoded (.decode tok ids)]
     (assert-true "encode returns Uint32Array" (instance? js/Uint32Array ids))
     (assert-true "encode produces tokens" (pos? (.-length ids)))
     (assert-true "decode round-trips" (= "Hello world" decoded))

     ;; ---------------------------------------------------------------
     ;; 3. Forward pass (uncached)
     ;; ---------------------------------------------------------------
     (println "\n== 3. Forward pass ==")

     (let [logits (llm/next-token-logprobs (:model m) ids)]
       (assert-equal "logprobs shape is [vocab_size]" [262144] (vec (mx/shape logits)))
       (let [max-lp (mx/item (mx/amax logits))]
         (assert-true "max log-prob <= 0" (<= max-lp 0))))

     ;; ---------------------------------------------------------------
     ;; 4. KV cache cycle
     ;; ---------------------------------------------------------------
     (println "\n== 4. KV cache ==")

     (.initKvCaches (:model m))
     (assert-true "initKvCaches succeeds" true)

     (let [prefill-logits (llm/forward-prefill (:model m) ids)]
       (assert-equal "prefill logits shape" [262144] (vec (mx/shape prefill-logits)))

       (let [step-logits (llm/forward-step (:model m) 4)]
         (assert-equal "step logits shape" [262144] (vec (mx/shape step-logits)))

         (.resetKvCaches (:model m))
         (assert-true "resetKvCaches succeeds" true)

         ;; Verify fresh cycle works after reset
         (.initKvCaches (:model m))
         (let [fresh-logits (llm/forward-prefill (:model m) ids)]
           (assert-equal "fresh prefill after reset" [262144] (vec (mx/shape fresh-logits))))
         (.resetKvCaches (:model m))))

     ;; ---------------------------------------------------------------
     ;; 5. make-llm-gf + GFI ops
     ;; ---------------------------------------------------------------
     (println "\n== 5. GFI ops ==")

     (pm/let [gf (llm-core/make-llm-gf m)
              prompt-ids [2]  ;; BOS token
              ;; simulate — generate text
              trace (p/simulate gf [prompt-ids 8])]
       (assert-true "simulate returns trace" (some? trace))
       (println "    generated:" (pr-str (:retval trace)))

       ;; assess — score text
       (pm/let [{:keys [weight]} (p/assess gf [prompt-ids 8] (:choices trace))]
         (assert-true "assess returns finite weight" (js/isFinite (mx/item weight))))

       ;; generate with constraints
       (pm/let [obs (:choices trace)
                {:keys [trace weight]} (p/generate gf [prompt-ids 8] obs)]
         (assert-true "generate returns trace" (some? trace))
         (assert-true "generate weight is finite" (js/isFinite (mx/item weight)))))

     ;; ---------------------------------------------------------------
     ;; 6. Grammar constraints
     ;; ---------------------------------------------------------------
     (println "\n== 6. Grammar constraints ==")

     (println "    compiling constraint (262K vocab, may take a moment)...")
     (let [constraint (grammar/compile-constraint (:tokenizer m) "[0-9]+")]
       (assert-true "constraint compiled" (some? (:dfa constraint)))
       (assert-true "token-index built" (= 262144 (count (:token-index constraint))))
       (pm/let [gf (llm-core/make-llm-gf m)
                constrained-gf (grammar/constrain gf constraint)
                trace (p/simulate constrained-gf [[2] 4])]
         (let [tok-ids (:retval trace)
               ;; Skip BOS token, decode generated tokens only
               gen-ids (rest tok-ids)
               tok (:tokenizer m)
               id-arr (js/Uint32Array. (clj->js gen-ids))]
           (pm/let [text (.decode tok id-arr)]
             (assert-true "grammar output is digits" (some? (re-matches #"[0-9]+" text)))
             (println "    constrained text:" (pr-str text))))))

     ;; ---------------------------------------------------------------
     ;; 7. Byte-level generation
     ;; ---------------------------------------------------------------
     (println "\n== 7. Byte-level generation ==")

     (pm/let [byte-gf (bytes/make-byte-llm-gf m)
              trace (p/simulate byte-gf [[2] 8])]
       (assert-true "byte-level simulate returns trace" (some? trace))
       (println "    byte-level:" (pr-str (:retval trace))))

     ;; ---------------------------------------------------------------
     ;; Summary
     ;; ---------------------------------------------------------------
     (println (str "\n== Results: " @pass-count " passed, " @fail-count " failed =="))
     (when (pos? @fail-count)
       (js/process.exit 1)))))
