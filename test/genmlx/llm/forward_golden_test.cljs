;; @tier slow
(ns genmlx.llm.forward-golden-test
  "Golden-output oracle for the LLM forward pass (the f6ov GATE — NOT the f6ov
   rewrite). Pins forward-pass(fixed token ids) -> argmax / top-5 token ids +
   logprob values against the CURRENT working forward, for each trusted model,
   so any future forward reimplementation that drifts is caught immediately
   (the silent-blow-up failure mode f6ov warns about).

   Also asserts the two forward code paths agree: the uncached `.forward`
   (forward-pass) and the cached `.forwardWithCache` (forward-prefill) must
   produce the same next-token distribution. This regression caught a real bug:
   forward-pass indexed the input length into a [1 1 vocab] last-position-only
   result, decoding garbage instead of the true next token.

   EXTERNAL ORACLE (the f6ov self-reference cure): this gate pins GenMLX against
   itself, so the pins below are cross-validated against an INDEPENDENT forward —
   Python mlx-lm — by scripts/llm_forward_xval_mlxlm.py (`npm run test:xval-llm`).
   That script is the source-of-truth's reference oracle; run it before a release
   (it needs Python mlx-lm + the checkpoints, so it is NOT in the bun gate; it
   skips cleanly if either is absent). Last confirmed: 18/18 vs mlx-lm 0.31.1 on
   qwen3.5-0.8b + 4b.

   Skips cleanly if a model directory is absent."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))
(defn assert= [label expected actual]
  (assert-true (str label " (=" (pr-str expected) ")") (= expected actual)))
(defn assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol) (do (swap! pass inc) (println (str "  PASS: " label " (diff=" (.toFixed d 6) ")")))
        (do (swap! fail inc) (println (str "  FAIL: " label " expected=" expected " actual=" actual))))))

(def prompt "The capital of France is")
(def expected-prompt-ids [760 6511 314 9338 369])
(def model-root (str (.-HOME js/process.env) "/.cache/models"))


;; ---------------------------------------------------------------------------
;; Goldens are BACKEND-CONDITIONAL (genmlx-z4hy / genmlx-nhvg).
;;
;; The invariants that hold on every backend — tokenizer ids, argmax token,
;; its decoded word, cached==uncached agreement — are pinned ONCE below.
;; The bf16 logprob VALUES and the top-5 membership are not portable and are
;; pinned per backend:
;;
;;   Metal (M4)  top-5 ids: 11751 279 7172  25 198
;;   CUDA (Thor) top-5 ids: 11751 303 279 7172 220
;;
;; Only three of five ids are common, so this is not merely tolerance noise —
;; several candidates sit within ~0.5 nats and the backends order them
;; differently. One golden set at tol 0.01 CANNOT hold on both: the measured
;; cross-kernel shift is 0.02-0.11, an order of magnitude above the tolerance.
;;
;; Metal values: captured 2026-07-27 on M4 at mlx fd21ea68c (genmlx-lr9c),
;; verified against the parity suites + the mlx-lm oracle argmax pin.
;; CUDA values:  captured 2026-07-27 on Thor sm_110 at mlx 135c5945e, same
;; prompt and load options (:cljs-forward? false :paged? false).
;;
;; Tight tol (0.01) is kept deliberately: WITHIN one backend the build is
;; bit-reproducible, so a forward rewrite must still reproduce its own
;; backend's values. An MLX-pin bump that shifts values is EXPECTED to re-pin
;; the AFFECTED backend after independent verification — that is this pin's
;; lifecycle. Re-pin one backend at a time; a change that moves BOTH is far
;; more likely a real forward regression than two coincident kernel bumps.
;; ---------------------------------------------------------------------------
(def golden
  [{:name "qwen3.5-0.8b" :dir "qwen3.5-0.8b-mlx-bf16"
    :argmax 11751 :argmax-decoded " Paris"
    :metal {:argmax-logprob -2.125
            :top5 [[11751 -2.125] [279 -2.3125] [7172 -2.5625] [25 -2.875] [198 -2.875]]}
    :cuda  {:argmax-logprob -1.8828125
            :top5 [[11751 -1.8828125] [303 -2.0625] [279 -2.5625] [7172 -2.5625] [220 -3.25]]}}
   {:name "qwen3.5-4b" :dir "qwen3.5-4b-mlx-bf16"
    :argmax 11751 :argmax-decoded " Paris"
    :metal {:argmax-logprob -0.62109375
            :top5 [[11751 -0.62109375] [7172 -2.75] [264 -3.125] [3750 -3.4375] [279 -3.625]]}
    ;; Absent on Thor, so never captured on CUDA. The model-presence check
    ;; below skips this entry there; if the checkpoint ever lands on a CUDA
    ;; box, capture rather than reuse the Metal numbers.
    :cuda  nil}])

(defn- backend-key [] (if (mx/metal-is-available?) :metal :cuda))

(defn- topk-from [lp k]
  (mx/eval! lp)
  (let [f32 (.toFloat32 lp)]
    (->> (range (.-length f32))
         (map (fn [i] [i (aget f32 i)]))
         (sort-by second >)
         (take k))))

(defn- log-softmax [logits]
  (mx/subtract logits (mx/logsumexp logits)))

(defn check-model [{:keys [name dir argmax argmax-decoded] :as entry}]
  (let [path (str model-root "/" dir)
        bk   (backend-key)
        {:keys [argmax-logprob top5]} (get entry bk)]
    (cond
      (not (.existsSync fs path))
      (do (println (str "\n== " name " — SKIP (absent: " path ") ==")) (pr/resolved nil))
      (nil? top5)
      (do (println (str "\n== " name " — SKIP (no " (clj->js bk)
                        " goldens captured; capture, do not reuse the other backend's) =="))
          (pr/resolved nil))
      ;; Explicitly pins the UPSTREAM/borrowed forward — its captured golden
      ;; values are upstream's, tol 0.01. The owned forward (now the default for
      ;; qwen3/qwen3_5) differs by bf16 cross-kernel noise (~0.12) and is guarded
      ;; separately by the parity tests + scripts/llm_forward_xval_mlxlm.py.
      ;; :paged? false — this suite pins the FLAT (Tier-1) forwardWithCache
      ;; seam; on Metal, v0.0.8 defaults VLM checkpoints to the block-paged
      ;; adapter, which refuses it (genmlx-lr9c; load policy: genmlx-eacv).
      :else
      (pr/let [m (llm/load-model path {:cljs-forward? false :paged? false})
               tok (:tokenizer m)
               ids-raw (llm/encode tok prompt false)
               ids (vec ids-raw)]
        (println (str "\n== " name " ==  prompt=" (pr-str prompt)))
        (assert= "tokenizer prompt-ids stable" expected-prompt-ids ids)
        ;; uncached path (forward-pass -> next-token-logprobs)
        (let [un-lp (llm/next-token-logprobs (:model m) ids)
              un-amax (mx/item (mx/argmax un-lp))]
          (llm/init-cache! (:model m))
          (let [ca-logits (llm/forward-prefill (:model m) ids)
                ca-lp (log-softmax ca-logits)
                ca-amax (mx/item (mx/argmax ca-lp))]
            (llm/reset-cache! (:model m))
            (pr/let [dec (llm/decode tok (js/Uint32Array.from #js [un-amax]))]
              ;; HARD argmax/Paris assertions (the f6ov gate)
              (assert= "uncached argmax token id" argmax un-amax)
              (assert= "uncached argmax decodes to expected word" argmax-decoded dec)
              (assert= "cached argmax == golden" argmax ca-amax)
              (assert= "cached == uncached argmax (paths agree)" un-amax ca-amax)
              (assert-close "argmax logprob pinned" argmax-logprob (mx/item (mx/index un-lp argmax)) 0.01)
              ;; path numeric agreement over the top of the distribution
              ;; The cached (chunked-prefill) and uncached (full-forward) kernels
              ;; differ slightly in bf16; argmax agreement above is the strong
              ;; invariant. This bounds gross divergence (the pre-fix garbage was
              ;; off by whole nats) while tolerating cross-kernel bf16 noise.
              (let [un-f32 (do (mx/eval! un-lp) (.toFloat32 un-lp))
                    ca-f32 (do (mx/eval! ca-lp) (.toFloat32 ca-lp))
                    max-diff (reduce max (map (fn [[i _]] (js/Math.abs (- (aget un-f32 i) (aget ca-f32 i)))) top5))]
                (println (str "    [info] cached-vs-uncached top5 max logprob diff = " (.toFixed max-diff 5)))
                (assert-true "cached vs uncached logprobs agree on top5 (<0.1, bf16 cross-kernel)" (< max-diff 0.1)))
              ;; pin top-5 ids + logprobs
              (let [tk (topk-from un-lp 5)]
                (assert= "top5 token ids pinned" (mapv first top5) (mapv first tk))
                (doseq [[[gi gv] [_ av]] (map vector top5 tk)]
                  (assert-close (str "top5 logprob id=" gi) gv av 0.01))))))))))

(pr/let [_ (check-model (first golden))
         _ (check-model (second golden))]
  (println (str "\n=== forward-golden: " @pass " PASS, " @fail " FAIL ===")))
