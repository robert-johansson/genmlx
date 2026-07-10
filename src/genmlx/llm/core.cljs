(ns genmlx.llm.core
  "LLM as a first-class generative function in GenMLX.

   An LLM generates text token-by-token. Each token is a trace site with
   address :t0, :t1, ..., :tN, sampled from categorical(logits). The
   standard handler system handles simulate/generate/update/regenerate
   automatically — no custom dispatch needed.

   Usage:
     (pr/let [m   (llm/load-model model-dir)
              gf  (make-llm-gf m)
              ids (vec (llm/encode (:tokenizer m) \"Hello\"))]
       (p/simulate gf [ids 20]))"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.llm.backend :as llm]
            [genmlx.choicemap :as cm]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- t-addr
  "Trace address for the i-th generated token: :t0, :t1, ..."
  [i]
  (keyword (str "t" i)))

;; ---------------------------------------------------------------------------
;; Masked-EOS algebra (genmlx-9uyg) — the [K]-lane freeze without handler
;; changes. An inactive (post-eos) lane's logits are replaced by a one-hot
;; pad row, so log_softmax gives lp(pad)=0 EXACTLY in f32, the Gumbel-max
;; sample is deterministically pad, and the handler's unconditional
;; score/weight accumulation adds zeros — the lane is frozen. Pinned by
;; llm_batched_mask_test.
;; ---------------------------------------------------------------------------

(defn pad-onehot-row
  "One-hot pad row over the vocabulary: 0 at pad-id, -1e9 elsewhere.
   log_softmax(row)[pad] = -log(1 + (V-1)·e^{-1e9}) = 0 exactly in f32."
  [vocab pad-id]
  (mx/where (mx/equal (mx/arange vocab) pad-id)
            (mx/scalar 0.0) (mx/scalar -1e9)))

(defn mask-inactive-logits
  "Per-lane logits selection: active lanes keep `logits` ([V] shared or
   [K V] per-lane), inactive lanes get `pad-row` [V]. active is a [K] mask
   (or scalar — shapes broadcast, so the scalar path reuses this unchanged)."
  [logits active pad-row]
  (mx/where (mx/expand-dims active -1) logits pad-row))

(defn advance-active
  "Monotone lane-liveness update: a lane dies when it samples eos and stays
   dead. `active` nil (first site) starts all-alive."
  [active tok eos-id]
  (let [alive (mx/logical-not (mx/equal tok eos-id))]
    (if active (mx/logical-and active alive) alive)))

(defn make-llm-gf
  "Create a generative function from a loaded LLM.

   model-map: {:model :tokenizer :type} from llm/load-model.

   Returns a DynamicGF that takes [prompt-ids max-tokens]:
     prompt-ids — vector of int token IDs (from llm/encode)
     max-tokens — maximum number of new tokens to generate

   Each generated token is a trace site :t0, :t1, ..., :tN with
   a categorical distribution over the vocabulary. Generation stops
   at EOS or max-tokens, whichever comes first.

   opts (genmlx-jq6l):
     :images — seq of image byte buffers closed over at construction: the gf
       then represents p(answer-tokens | images, prompt) on the OWNED VLM
       path. prompt-ids must carry one <|image_pad|> marker per image (encode
       a llm/render-chat prompt built with :images). Every GFI op re-runs the
       body and therefore the full vision prefill — the replay-oracle
       semantics; the expensive look is per-op, not amortized (branch-ledger
       amortization is the token-SMC/branched layer's job).

   Uses KV cache for O(n) generation instead of O(n²). The cache is
   initialized at the start of each gen body execution and reset at
   the end (including on early EOS exit).

   Not safe for concurrent execution on the same model — each concurrent
   path needs its own model instance. Not compatible with vsimulate/vgenerate
   (uses mx/item for EOS check, which requires scalar values) — use
   make-llm-gf-batched for the [K]-particle path (genmlx-9uyg)."
  ([model-map] (make-llm-gf model-map {}))
  ([model-map {:keys [images]}]
   (let [{:keys [model tokenizer]} model-map
         eos (llm/eos-token-id tokenizer)]
     (dyn/auto-key
      (gen [prompt-ids max-tokens]
           (if (zero? max-tokens)
             prompt-ids
             (do
               (llm/init-cache! model)
               (try
                 (let [logits (if (seq images)
                                (llm/forward-prefill model prompt-ids {:images images})
                                (llm/forward-prefill model prompt-ids))]
                   (loop [i 0, context prompt-ids, logits logits]
                     (if (>= i max-tokens)
                       context
                       (let [tok (trace (t-addr i) (dist/categorical logits))
                             tok-id (mx/item tok)]
                         (if (= tok-id eos)
                           (conj context tok-id)
                           (let [next-logits (llm/forward-step model tok-id)]
                             (recur (inc i) (conj context tok-id) next-logits)))))))
                 (finally
                   (llm/reset-cache! model))))))))))

(defn make-llm-gf-uncached
  "Like make-llm-gf but without KV cache. Recomputes full context at
   each token step — O(n²) but stateless. Useful for debugging or when
   the model doesn't support KV cache."
  [model-map]
  (let [{:keys [model tokenizer]} model-map
        eos (llm/eos-token-id tokenizer)]
    (dyn/auto-key
     (gen [prompt-ids max-tokens]
          (loop [i 0, context prompt-ids]
            (if (>= i max-tokens)
              context
              (let [logits (llm/forward-pass model context)
                    tok (trace (t-addr i) (dist/categorical logits))
                    tok-id (mx/item tok)]
                (if (= tok-id eos)
                  (conj context tok-id)
                  (recur (inc i) (conj context tok-id))))))))))

(defn make-llm-gf-batched
  "The [K]-particle LLM-GF (genmlx-9uyg, Route B): a DynamicGF over
   [prompt-ids max-tokens] whose body is vectorization-safe — no mx/item, no
   host control flow on sampled values — so dyn/vsimulate & dyn/vgenerate run
   K particles through ONE lockstep batched forward (B=1 shared prefill, the
   cache tiled to K on the first step; decode weight-traffic is shared across
   lanes). Owned forward (CljsForwardModel) only.

   Trace-shape contract (differs from make-llm-gf): sites :t0 … :t{max-1}
   are ALWAYS present with uniform [K] leaves. A lane that samples eos at
   site i traces and scores eos there (scalar semantics), then goes inactive:
   its later sites deterministically trace pad with logprob EXACTLY 0 (the
   masked-EOS algebra above), so its score/weight freeze — per-lane
   early-stop without per-lane control flow. The eos token itself is never
   fed to the forward.

   The binding law (L1, pinned by llm_batched_mask_test on a toy model and
   llm_batched_gf_test on a real checkpoint): for every lane k,
   vsimulate score[k] == scalar make-llm-gf assess of that lane's tokens
   truncated at eos.

   opts:
     :pad-id      — the frozen-lane filler token (default: tokenizer pad if
                    valid, else eos; ANY in-vocab id is correct — it only
                    needs lp 0 under the pad row and a harmless embedding).
     :hook        — stateful per-step logits middleware
                    {:init (fn [] state), :mask (fn [state logits i] logits'),
                     :advance (fn [state tok-K] state')} — all-MLX state; the
                    vectorized grammar (grammar/vectorized-hook) plugs in
                    here. Hook masking runs BEFORE the inactive-lane
                    override, so dead lanes stay frozen regardless.
     :check-every — host early-exit: every J sites, eval (mx/any active) and
                    stop when every lane is dead. Default OFF (it forces an
                    eval and makes the site count data-dependent); safe for
                    vsimulate-style unconstrained use only — never with
                    constraints on later sites.

   Scalar GFI ops on this gf work through broadcasting (shapes []), but run
   the full max-tokens loop (no early exit) — for scalar use, make-llm-gf
   remains the right constructor.

   Retval: {:tokens [K max-tokens] int matrix ([max-tokens] under scalar
   execution), :active the final lane-liveness mask, :prompt-ids}."
  ([model-map] (make-llm-gf-batched model-map {}))
  ([model-map {:keys [pad-id hook check-every]}]
   (let [{:keys [model tokenizer]} model-map
         _ (when-not (llm/cljs-forward-model? model)
             (throw (ex-info "make-llm-gf-batched requires the OWNED forward (CljsForwardModel) — load with {:cljs-forward? true} (or a supported family's smart default)."
                             {:genmlx/error :batched-gf-owned-only
                              :model-type (type model)})))
         eos     (llm/eos-token-id tokenizer)
         vocab   (get-in model [:fwd :config :vocab])
         pad     (or pad-id
                     (let [p (llm/pad-token-id tokenizer)]
                       (if (and (some? p) (>= p 0) (< p vocab)) p eos)))
         pad-row (pad-onehot-row vocab pad)
         pad-tok (mx/scalar pad mx/int32)
         ;; Site values can MIX shapes: a site constrained with a shared
         ;; scalar observation traces [], sampled sites trace [K] (and the
         ;; cache stays B=1 through a constrained prefix — the first sampled
         ;; token tiles it). Broadcast scalars up before stacking.
         stack-toks (fn [toks]
                      (let [k    (some #(let [sh (mx/shape %)]
                                          (when (pos? (count sh)) (first sh)))
                                       toks)
                            toks (if k
                                   (mapv #(if (pos? (count (mx/shape %)))
                                            % (mx/broadcast-to % [k]))
                                         toks)
                                   toks)
                            s    (mx/stack toks 0)]       ; [T K] / [T]
                        (if k (mx/transpose s [1 0]) s))) ; [K T]
         ;; mx/item on a bool array yields the NUMBER 0/1 — and (not 0) is
         ;; false in CLJS, so a bare (not (mx/item …)) can NEVER see death.
         ;; Found by llm_batched_checkevery_test (genmlx-lo6e D3).
         all-dead? (fn [active]
                     (let [v (mx/item (mx/any active))]
                       (or (false? v) (== 0 v))))]
     (dyn/auto-key
      (gen [prompt-ids max-tokens]
           (if (zero? max-tokens)
             {:tokens nil :active nil :prompt-ids prompt-ids}
             (do
               (llm/init-cache! model)
               (try
                 (loop [i      0
                        logits (llm/forward-prefill model prompt-ids)
                        active nil
                        hs     (when hook ((:init hook)))
                        toks   []]
                   (let [lg   (if hook ((:mask hook) hs logits i) logits)
                         lg   (if active (mask-inactive-logits lg active pad-row) lg)
                         tok  (trace (t-addr i) (dist/categorical lg))
                         act' (advance-active active tok eos)
                         hs'  (when hook ((:advance hook) hs tok))
                         done? (or (= (inc i) max-tokens)
                                   (and check-every
                                        (pos? (count (mx/shape tok)))
                                        (zero? (mod (inc i) check-every))
                                        (all-dead? act')))]
                     (if done?
                       {:tokens (stack-toks (conj toks tok)) :active act'
                        :prompt-ids prompt-ids}
                       ;; still-active lanes feed their sample; dead lanes
                       ;; (incl. just-eos'd) feed pad — eos is never fed.
                       (let [fed (mx/where act' tok pad-tok)]
                         (recur (inc i)
                                (if (pos? (count (mx/shape fed)))
                                  (llm/forward-step-batched model fed)
                                  (llm/forward-step model (mx/item fed)))
                                act' hs' (conj toks tok))))))
                 (finally
                   (llm/reset-cache! model))))))))))

(defn vtrace-lane-tokens
  "Host-extract per-lane token seqs from a batched vtrace: a vector of K
   vecs, each truncated AT its first eos (inclusive — the sequence scalar
   assess scores; trailing pads carry exactly 0 logprob and are dropped).
   Boundary fn (host sync); also works on a scalar trace (K=1)."
  [vtrace eos-id]
  (let [choices (:choices vtrace)
        cols (->> (range)
                  (map #(cm/get-submap choices (t-addr %)))
                  (take-while cm/has-value?)
                  (mapv #(let [v (mx/->clj (mx/astype (cm/get-value %) mx/int32))]
                           (if (sequential? v) (vec v) [v]))))
        k (apply max (map count cols))
        ;; scalar-constrained sites hold ONE shared value — fan it out
        cols (mapv #(if (= 1 (count %)) (vec (repeat k (first %))) %) cols)]
    (mapv (fn [l]
            (let [full (mapv #(nth % l) cols)
                  idx  (.indexOf (clj->js full) eos-id)]
              (if (neg? idx) full (subvec full 0 (inc idx)))))
          (range k))))

(defn decode-vtrace
  "Decode a batched vtrace's K lanes to text (eos and post-eos pads
   stripped). Returns a promise of a vector of K strings."
  [tokenizer vtrace]
  (let [eos (llm/eos-token-id tokenizer)]
    (pr/all
     (mapv (fn [toks]
             (let [txt (if (= eos (peek toks)) (pop toks) toks)]
               (llm/decode tokenizer (js/Uint32Array.from (clj->js txt)))))
           (vtrace-lane-tokens vtrace eos)))))

(defn decode-trace
  "Extract generated token IDs from a trace and decode to text.

   Collects token values from trace sites :t0, :t1, ... in order
   and decodes them to a string. Does NOT include prompt tokens —
   only the tokens generated by the LLM.

   Returns a promise (tokenizer decode is async)."
  [tokenizer trace]
  (let [choices (:choices trace)
        tokens (->> (range)
                    (map #(cm/get-submap choices (t-addr %)))
                    (take-while cm/has-value?)
                    (mapv (comp mx/item cm/get-value)))]
    (llm/decode tokenizer (js/Uint32Array.from (clj->js tokens)))))
