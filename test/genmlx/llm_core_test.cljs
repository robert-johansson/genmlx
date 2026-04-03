(ns genmlx.llm-core-test
  "Phase 2: Test genmlx.llm.core — LLM as generative function.
   Verifies simulate, generate, assess, constrained generation,
   importance sampling convergence, and EOS handling."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [promesa.core :as pr]))

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

(pr/let
  [m (llm/load-model (str model-dir "/qwen3-0.6b-mlx-bf16"))
   tok (:tokenizer m)

   ;; Build the LLM generative function
   llm-gf (llm-core/make-llm-gf m)

   ;; Encode a short prompt for all tests
   prompt-ids-raw (llm/encode tok "The capital of France is")
   prompt-ids (vec prompt-ids-raw)

   ;; -----------------------------------------------------------------
   ;; 2.1 Simulate
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.1 simulate ==")
   trace (p/simulate llm-gf [prompt-ids 5])

   _ (assert-true "trace exists" (some? trace))
   _ (assert-true "trace has choices" (some? (:choices trace)))
   _ (assert-true "trace has score" (some? (:score trace)))
   _ (assert-true "trace has retval" (some? (:retval trace)))

   ;; Check trace sites :t0 through :t4 exist
   choices (:choices trace)
   _ (assert-true ":t0 exists" (cm/has-value? (cm/get-submap choices :t0)))
   _ (assert-true ":t4 exists" (cm/has-value? (cm/get-submap choices :t4)))
   _ (assert-true ":t5 absent" (not (cm/has-value? (cm/get-submap choices :t5))))

   ;; Score should be negative (log probability)
   score-val (mx/item (:score trace))
   _ (println (str "  Score: " score-val))
   _ (assert-true "score < 0" (< score-val 0))

   ;; Return value should be prompt + 5 new tokens
   _ (assert-true "retval is vector" (vector? (:retval trace)))
   _ (assert-true "retval length = prompt + 5"
                  (= (count (:retval trace)) (+ (count prompt-ids) 5)))

   ;; Decode the generated text
   gen-text (llm-core/decode-trace tok trace)
   _ (println (str "  Generated: '" gen-text "'"))
   _ (assert-true "decode returns string" (string? gen-text))

   ;; -----------------------------------------------------------------
   ;; 2.2 Simulate — score = sum of per-token log-probs
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.2 score verification ==")

   ;; Manually compute log-probs for each generated token
   _ (let [retval (:retval trace)
           manual-score
           (loop [i 0, acc 0.0]
             (if (>= i 5)
               acc
               (let [context (subvec retval 0 (+ (count prompt-ids) i))
                     lp (llm/next-token-logprobs (:model m) context)
                     tok-val (mx/item (cm/get-value
                                        (cm/get-submap choices
                                                       (keyword (str "t" i)))))
                     tok-lp (mx/item (mx/take-idx lp tok-val))]
                 (recur (inc i) (+ acc tok-lp)))))]
       (println (str "  Manual score: " manual-score))
       ;; bf16 model weights → ~0.2 accumulated precision over 5 forward passes
       (assert-close "score = sum of token log-probs"
                     manual-score score-val 0.2))

   ;; -----------------------------------------------------------------
   ;; 2.3 Generate — partial constraints
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.3 generate (partial constraints) ==")

   ;; Constrain :t0 to the argmax token (most likely next token)
   lp0 (llm/next-token-logprobs (:model m) prompt-ids)
   best-id (mx/item (mx/argmax lp0))
   best-lp (mx/item (mx/take-idx lp0 best-id))
   _ (println (str "  Constraining :t0 to token " best-id
                   " (log-prob " (.toFixed best-lp 4) ")"))

   constraints (cm/set-value cm/EMPTY :t0 (mx/scalar best-id mx/int32))
   gen-result (p/generate llm-gf [prompt-ids 5] constraints)

   gen-trace (:trace gen-result)
   gen-weight (mx/item (:weight gen-result))
   gen-score (mx/item (:score gen-trace))

   ;; Constrained token should appear in trace
   t0-val (mx/item (cm/get-value (cm/get-submap (:choices gen-trace) :t0)))
   _ (assert-true ":t0 = constrained value" (= t0-val best-id))

   ;; Weight should be log-prob of constrained token
   _ (println (str "  Weight: " gen-weight))
   _ (println (str "  Score: " gen-score))
   _ (assert-close "weight ≈ log p(constrained token)"
                   best-lp gen-weight 0.01)

   ;; Score should still be full sequence log-prob
   _ (assert-true "score < 0" (< gen-score 0))
   _ (assert-true "score < weight (more terms)" (< gen-score gen-weight))

   ;; -----------------------------------------------------------------
   ;; 2.4 Generate — fully constrained (weight = score)
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.4 generate (fully constrained) ==")

   ;; Use the tokens from the simulate trace as full constraints
   full-constraints
   (reduce (fn [cm i]
             (let [v (cm/get-value (cm/get-submap choices (keyword (str "t" i))))]
               (cm/set-value cm (keyword (str "t" i)) v)))
           cm/EMPTY
           (range 5))

   full-result (p/generate llm-gf [prompt-ids 5] full-constraints)
   full-weight (mx/item (:weight full-result))
   full-score (mx/item (:score (:trace full-result)))

   _ (println (str "  Weight: " full-weight))
   _ (println (str "  Score: " full-score))
   _ (assert-close "fully constrained: weight = score"
                   full-score full-weight 0.001)

   ;; -----------------------------------------------------------------
   ;; 2.5 Assess — all tokens provided
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.5 assess ==")

   assess-result (p/assess llm-gf [prompt-ids 5] full-constraints)
   assess-weight (mx/item (:weight assess-result))

   _ (println (str "  Assess weight: " assess-weight))
   _ (assert-close "assess weight = generate score"
                   full-score assess-weight 0.001)

   ;; -----------------------------------------------------------------
   ;; 2.6 Reproducibility — same key → same tokens
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.6 reproducibility ==")

   keyed-gf (dyn/with-key llm-gf (rng/fresh-key 42))
   trace-a (p/simulate keyed-gf [prompt-ids 5])
   trace-b (p/simulate (dyn/with-key llm-gf (rng/fresh-key 42)) [prompt-ids 5])

   _ (assert-true "same key → same retval"
                  (= (:retval trace-a) (:retval trace-b)))
   _ (assert-close "same key → same score"
                   (mx/item (:score trace-a))
                   (mx/item (:score trace-b)) 0.001)

   ;; Different key → (very likely) different tokens
   trace-c (p/simulate (dyn/with-key llm-gf (rng/fresh-key 99)) [prompt-ids 5])
   _ (assert-true "different key → different retval"
                  (not= (:retval trace-a) (:retval trace-c)))

   ;; -----------------------------------------------------------------
   ;; 2.7 Importance sampling — varying weights
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.7 importance sampling ==")

   ;; Constrain :t2 (after free :t0, :t1) so context varies per particle,
   ;; giving different weights. This tests that IS actually works.
   is-constraints (cm/set-value cm/EMPTY :t2 (mx/scalar best-id mx/int32))
   is-weights
   (loop [k 0, ws []]
     (if (>= k 20)
       ws
       (let [r (p/generate llm-gf [prompt-ids 4] is-constraints)]
         (recur (inc k) (conj ws (mx/item (:weight r)))))))

   _ (println (str "  IS log-weights (first 5): "
                   (mapv #(.toFixed % 2) (take 5 is-weights))))
   _ (assert-true "IS weights finite" (every? js/isFinite is-weights))
   _ (assert-true "IS weights negative" (every? neg? is-weights))

   ;; With varying context, weights should not all be identical
   distinct-ws (count (distinct (mapv #(.toFixed % 4) is-weights)))
   _ (println (str "  Distinct weights: " distinct-ws "/20"))
   _ (assert-true "weights vary across particles" (> distinct-ws 1))

   ;; Marginal likelihood estimate should be finite
   log-ml-est (js/Math.log (/ (reduce + (map js/Math.exp is-weights))
                              (count is-weights)))
   _ (println (str "  Estimated log Z: " (.toFixed log-ml-est 4)))
   _ (assert-true "log Z is finite" (js/isFinite log-ml-est))

   ;; -----------------------------------------------------------------
   ;; 2.8 EOS handling
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.8 EOS handling ==")

   eos-id (llm/eos-token-id tok)
   eos-constraints (cm/set-value cm/EMPTY :t0 (mx/scalar eos-id mx/int32))
   eos-result (p/generate llm-gf [prompt-ids 10] eos-constraints)
   eos-trace (:trace eos-result)
   eos-choices (:choices eos-trace)

   ;; :t0 should be EOS
   _ (assert-true "EOS at :t0"
                  (= (mx/item (cm/get-value (cm/get-submap eos-choices :t0)))
                     eos-id))
   ;; Generation should stop — no :t1
   _ (assert-true "no :t1 after EOS"
                  (not (cm/has-value? (cm/get-submap eos-choices :t1))))
   ;; Retval should be prompt + 1 token (the EOS)
   _ (assert-true "retval = prompt + EOS"
                  (= (count (:retval eos-trace)) (+ (count prompt-ids) 1)))

   ;; -----------------------------------------------------------------
   ;; 2.9 Cached vs uncached equivalence
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.9 cached vs uncached ==")

   uncached-gf (llm-core/make-llm-gf-uncached m)

   ;; Same key → same tokens and scores from both paths
   cached-trace   (p/simulate (dyn/with-key llm-gf (rng/fresh-key 77)) [prompt-ids 5])
   uncached-trace (p/simulate (dyn/with-key uncached-gf (rng/fresh-key 77)) [prompt-ids 5])

   _ (assert-true "cached=uncached retval"
                  (= (:retval cached-trace) (:retval uncached-trace)))
   ;; bf16 cached vs uncached paths have different GPU accumulation order
   _ (assert-close "cached=uncached score"
                   (mx/item (:score cached-trace))
                   (mx/item (:score uncached-trace)) 0.3)

   ;; Fully constrained: same weight from both paths
   cached-gen   (p/generate (dyn/with-key llm-gf (rng/fresh-key 77))
                            [prompt-ids 5] full-constraints)
   uncached-gen (p/generate (dyn/with-key uncached-gf (rng/fresh-key 77))
                            [prompt-ids 5] full-constraints)

   _ (assert-close "cached=uncached weight (constrained)"
                   (mx/item (:weight cached-gen))
                   (mx/item (:weight uncached-gen)) 0.3)

   ;; -----------------------------------------------------------------
   ;; 2.10 Edge cases
   ;; -----------------------------------------------------------------
   _ (println "\n== 2.10 edge cases ==")

   zero-trace (p/simulate llm-gf [prompt-ids 0])
   _ (assert-true "max-tokens=0: retval = prompt"
                  (= (:retval zero-trace) prompt-ids))
   _ (assert-close "max-tokens=0: score = 0"
                   0.0 (mx/item (:score zero-trace)) 0.001)

   ;; max-tokens=1: single token generated
   one-trace (p/simulate llm-gf [prompt-ids 1])
   _ (assert-true "max-tokens=1: one trace site"
                  (cm/has-value? (cm/get-submap (:choices one-trace) :t0)))
   _ (assert-true "max-tokens=1: no :t1"
                  (not (cm/has-value? (cm/get-submap (:choices one-trace) :t1))))
   _ (assert-true "max-tokens=1: retval length"
                  (= (count (:retval one-trace)) (+ (count prompt-ids) 1)))]

  (println (str "\n== Phase 2: " @pass-count " passed, " @fail-count " failed ==")))
