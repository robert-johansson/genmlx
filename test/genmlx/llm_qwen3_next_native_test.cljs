;; @tier slow
;; genmlx-2sh6: end-to-end smoke for the native MoE forward wiring. The 80B
;; Qwen3-Coder-Next (config.json model_type "qwen3_next") has NO GenMLX-owned
;; CLJS forward — it routes to the native @genmlx/core Qwen35MoeModel
;; (forward / forwardWithCache / initCaches / resetCaches). On CUDA that native
;; forward is verified safe (mlx-2h4l); on Metal it SIGTRAPs and load-model
;; refuses it (covered by llm_moe_guard_test). So this test runs ONLY on a
;; non-Metal backend with the 80B present; everywhere else it SKIPs cleanly.
;;
;; What it asserts (the bean's VERIFY list):
;;   1. llm/load-model on qwen3_next yields a NATIVE model (:type :qwen3_next,
;;      not a CljsForwardModel) that passes assert-upstream-forward!.
;;   2. A tiny assess (code logprob): a direct next-token-logprobs over a code
;;      prompt is a finite, vocab-sized, normalized log-distribution; and the
;;      GFI consistency law assess(choices) == simulate score holds.
;;   3. A grammar-constrained simulate runs end-to-end and respects the grammar.
;; No SIGTRAP, no MLX_CUDA_DISABLE_MEMPOOL needed.
(ns genmlx.llm-qwen3-next-native-test
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.llm.grammar :as gram]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

(def ^:private metal? (mx/metal-is-available?))
(def ^:private home (.-HOME js/process.env))
(def ^:private repo-root
  (or (.-QWEN3_NEXT_DIR js/process.env)
      (str home "/code/mlx/models/Qwen3-Coder-Next-4bit")))

(defn- resolve-snapshot
  "Resolve the directory that actually holds config.json — either `dir` itself or,
   for a HuggingFace cache layout (dir/snapshots/<hash>/config.json), the snapshot
   subdir. Returns the model dir or nil if none has a config.json."
  [dir]
  (cond
    (not (.existsSync fs dir)) nil
    (.existsSync fs (str dir "/config.json")) dir
    (.existsSync fs (str dir "/snapshots"))
    (->> (.readdirSync fs (str dir "/snapshots"))
         (map #(str dir "/snapshots/" %))
         (filter #(.existsSync fs (str % "/config.json")))
         first)
    :else nil))

(def ^:private model-dir (resolve-snapshot repo-root))

(defn- finish []
  (println (str "\n== " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(println (str "\n== qwen3_next native forward smoke =="))
(println (str "  platform: " (if metal? "Metal" "CUDA/non-Metal")))
(println (str "  model-dir: " (or model-dir "<not found>")))

(cond
  metal?
  (do (println "  SKIP: Metal backend — native MoE refused here (see llm_moe_guard_test)")
      (finish))

  (nil? model-dir)
  (do (println (str "  SKIP: no qwen3_next checkpoint at " repo-root
                    " (set QWEN3_NEXT_DIR to override)"))
      (finish))

  :else
  (-> (pr/let [model-map (llm/load-model model-dir)]
        (let [{:keys [model tokenizer type]} model-map]
          ;; -------------------------------------------------------------
          ;; 1. Native model, correct type, passes the upstream-forward guard
          ;; -------------------------------------------------------------
          (println "\n-- 1. load + native forward surface --")
          (assert-true ":type is :qwen3_next" (= :qwen3_next type))
          (assert-true "model is NOT a CljsForwardModel (native instance)"
                       (not (llm/cljs-forward-model? model)))
          (assert-true "native instance exposes .forward"
                       (fn? (.-forward model)))
          (assert-true "native instance exposes .forwardWithCache"
                       (fn? (.-forwardWithCache model)))
          (assert-true "assert-upstream-forward! passes (returns nil, no throw)"
                       (nil? (llm/assert-upstream-forward! model)))

          (pr/let [prompt-raw (llm/encode tokenizer "def add(a, b):\n    return ")
                   prompt-ids (vec prompt-raw)]
            ;; -----------------------------------------------------------
            ;; 2a. tiny assess (code logprob) — direct next-token-logprobs
            ;; -----------------------------------------------------------
            (println "\n-- 2a. next-token-logprobs (code logprob) --")
            (let [lp (llm/next-token-logprobs model prompt-ids)
                  shp (mx/shape lp)
                  vocab (llm/vocab-size tokenizer)
                  argmax-id (mx/item (mx/argmax lp))
                  max-lp (mx/item (mx/amax lp))
                  ;; a normalized log-distribution: sum(exp(logprobs)) == 1
                  total-prob (mx/item (mx/sum (mx/exp lp)))]
              ;; The model's lm_head vocab (config vocab_size, e.g. 151936) is
              ;; PADDED beyond the tokenizer's real token count (vocab); logits
              ;; are 1-D over the model vocab, which is >= the tokenizer vocab.
              (assert-true "logprobs is 1-D over the model vocab (>= tokenizer vocab)"
                           (and (= 1 (count shp)) (>= (first shp) vocab)))
              (assert-true "max logprob is finite and <= 0"
                           (and (js/isFinite max-lp) (<= max-lp 1e-4)))
              (assert-true "sum of probs == 1 (normalized)"
                           (< (abs (- 1.0 total-prob)) 1e-2))
              (assert-true "argmax is a valid token id (within model vocab)"
                           (and (int? argmax-id) (<= 0 argmax-id) (< argmax-id (first shp))))
              (println (str "  argmax next token id=" argmax-id
                            " (\"" (llm/id->token tokenizer argmax-id) "\")"
                            "  max-logprob=" max-lp)))

            ;; -----------------------------------------------------------
            ;; 2b. GFI consistency: assess(choices) == simulate score
            ;; -----------------------------------------------------------
            (println "\n-- 2b. simulate/assess consistency (the GFI law) --")
            (let [gf (llm-core/make-llm-gf model-map)]
              (pr/let [tr (p/simulate gf [prompt-ids 3])
                       sc (mx/item (:score tr))
                       a  (p/assess gf [prompt-ids 3] (:choices tr))
                       w  (mx/item (:weight a))]
                (assert-true "simulate score is finite & negative"
                             (and (js/isFinite sc) (neg? sc)))
                (assert-true "assess weight == simulate score (GFI consistency)"
                             (< (abs (- sc w)) 0.05))
                (println (str "  score=" sc "  assess-weight=" w))

                ;; -------------------------------------------------------
                ;; 3. grammar-constrained simulate end-to-end
                ;; -------------------------------------------------------
                (println "\n-- 3. grammar-constrained simulate --")
                (pr/let [gprompt-raw (llm/encode tokenizer "Phone: ")
                         gprompt (vec gprompt-raw)]
                  (let [constraint (gram/compile-constraint tokenizer "\\d{3}-\\d{4}")
                        cgf (gram/constrain (llm-core/make-llm-gf model-map) constraint)
                        token-index (:token-index constraint)
                        eos-id (:eos-id constraint)
                        phone-re #"\d{3}-\d{4}"
                        decode-gen (fn [trace]
                                     (->> (subvec (:retval trace) (count gprompt))
                                          (remove #(= % eos-id))
                                          (map #(nth token-index %))
                                          (apply str)))]
                    (pr/let [ctrace (p/simulate cgf [gprompt 8])]
                      (let [text (decode-gen ctrace)
                            cscore (mx/item (:score ctrace))]
                        (assert-true "constrained simulate score is finite"
                                     (js/isFinite cscore))
                        (assert-true "constrained output matches grammar \\d{3}-\\d{4}"
                                     (some? (re-matches phone-re text)))
                        (println (str "  generated under grammar: \"" text "\""))
                        (finish))))))))) )
      (pr/catch
       (fn [e]
         (assert-true (str "load+forward should not throw — got: " (ex-message e)) false)
         (finish)))))
