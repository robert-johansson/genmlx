;; @tier heavy
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
(ns genmlx.llm.qwen3-next-native-test
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.llm.grammar :as gram]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [promesa.core :as pr]
            [genmlx.test-helpers :as h]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

(def ^:private metal? (mx/metal-is-available?))
(def ^:private model-spec {:org "mlx-community" :name "Qwen3-Coder-Next-4bit"})

(def ^:private env-dir
  ;; QWEN3_NEXT_DIR may name a model dir OR an HF-layout repo dir
  ;; (`<repo>/snapshots/<hash>`); an override is honoured even when incomplete,
  ;; so a wrong one fails loudly instead of skipping.
  (not-empty (.-QWEN3_NEXT_DIR js/process.env)))

(defn- resolve-override [dir]
  (let [snaps (str dir "/snapshots")]
    (or (when (h/complete-checkpoint? dir) dir)
        (when (.existsSync fs snaps)
          (->> (.readdirSync fs snaps)
               (map #(str snaps "/" %))
               (filter h/complete-checkpoint?)
               first))
        dir)))

(def ^:private model-dir
  ;; No revision is pinned — this test asserts native-forward WIRING, not
  ;; oracle-exact tokens (contrast qwen3_moe_layout_coherence_test, which must
  ;; stay revision-locked). The default is the first COMPLETE checkpoint across
  ;; each host's layout: a hardcoded ~/code/mlx/models default skipped on hosts
  ;; without it and scored PASS (genmlx-pc9o); its HF-hub-only replacement then
  ;; accepted Thor's config-only stub and FAILED in load-model (genmlx-5z51).
  (if env-dir
    (resolve-override env-dir)
    (first (filter h/complete-checkpoint? (h/checkpoint-candidates model-spec)))))

(defn- finish []
  (println (str "\n== " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(defn- skip-finish!
  "Call INSTEAD of `finish` on a skip branch, immediately after its `SKIP` print.
   run.sh scores SKIP only on a `SKIP` line within three lines above a
   `Ran 0 tests` summary (test/TESTING.md), and that classifier branch is
   reachable only for files that print such a summary — which a hand-rolled
   harness never does. Without this the skip fell through to `else status=PASS`,
   so a guard that loaded nothing was tallied green (genmlx-pc9o). Both lines
   are true: zero tests ran."
  []
  (println "Ran 0 tests containing 0 assertions.")
  (println (str "\n== " @pass " passed, " @fail " failed (skipped) ==")))

(println (str "\n== qwen3_next native forward smoke =="))
(println (str "  platform: " (if metal? "Metal" "CUDA/non-Metal")))
(println (str "  model-dir: " (or model-dir "<not found>")))

(cond
  metal?
  (do (println "  SKIP: Metal backend — native MoE refused here (see llm_moe_guard_test)")
      (skip-finish!))

  (nil? model-dir)
  (do (println (str "  SKIP: no complete qwen3_next checkpoint among "
                    (vec (h/checkpoint-candidates model-spec))
                    " (set QWEN3_NEXT_DIR to override)"))
      (skip-finish!))

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
                ;; RELATIVE band (genmlx-s3vo): the 4-bit MoE forward is not
                ;; run-to-run deterministic on this stack, so assess's
                ;; re-forward jitters against simulate's stored score.
                ;; Characterized 2026-07-07 over 10 pairs at 6 tokens: mean
                ;; |delta| 0.42, max 1.0, max rel (d/(1+|w|)) 0.122 — the old
                ;; 0.05 ABS band failed 9/10 pairs. Same llm_branched policy:
                ;; relative, never bit-exact.
                ;;
                ;; ATTRIBUTION (genmlx-cnhi, closed 2026-07-07): the jitter is
                ;; INHERENT to the current kernel set — kernel-level
                ;; nondeterminism (MoE gather_mm/reduction order), confirmed by
                ;; repeated assess on IDENTICAL choices spreading 0.125-0.625
                ;; nats in-process. Exonerated by direct probe: CUDA graphs
                ;; (spread unchanged with MLX_USE_CUDA_GRAPHS=0), cross-thread
                ;; lazy encoders (zero creation events during inference), MLX
                ;; kernel changes (none in the window), mlx-node forward
                ;; changes (none), toolchain (no nvcc/driver upgrades). The
                ;; apparent "10x growth vs June" was an artifact: June's
                ;; evidence was ONE 3-token sample under a 0.05 band; the July
                ;; numbers are a 25-sample 6-token characterization. A smaller
                ;; DETERMINISTIC per-choices offset (chunked-prefill vs
                ;; stepwise reduction order) stacks on the jitter. Observed max
                ;; rel to date 0.17 vs the 0.2 band — if this flakes, widen to
                ;; 0.3 rather than chasing determinism the kernels don't offer.
                (assert-true "assess weight ~ simulate score (GFI consistency, MoE-jitter relative band)"
                             (< (/ (abs (- sc w)) (+ 1.0 (abs w))) 0.2))
                (println (str "  score=" sc "  assess-weight=" w
                              "  rel-delta=" (.toFixed (/ (abs (- sc w)) (+ 1.0 (abs w))) 4)))

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
