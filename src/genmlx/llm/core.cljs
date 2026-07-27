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
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dispatch :as dispatch]
            [genmlx.dynamic :as dyn]
            [genmlx.llm.backend :as llm]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
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
     :sweep-every — sweep dead MLX wrappers every N tokens inside the body's
       synchronous decode loop (default 32; 0/nil disables). Same finalizer-
       starvation exposure and fix as generate-text-raw+ (llm/sweep-tick!,
       genmlx-12w4/genmlx-nwsr); housekeeping only, results unchanged.
     :prefill-chunk — owned path only: run the prefill in n-token blocks with
       a materialize+sweep boundary per block (genmlx-nwsr). With :images this
       is the VLM decoder-prefill chunk (vlm-prefill :chunk); without, the
       chunked text prefill. nil (default) = single-slab prefill.

   Uses KV cache for O(n) generation instead of O(n²). The cache is
   initialized at the start of each gen body execution and reset at
   the end (including on early EOS exit).

   Not safe for concurrent execution on the same model — each concurrent
   path needs its own model instance. Not compatible with vsimulate/vgenerate
   (uses mx/item for EOS check, which requires scalar values) — use
   make-llm-gf-batched for the [K]-particle path (genmlx-9uyg)."
  ([model-map] (make-llm-gf model-map {}))
  ([model-map {:keys [images sweep-every prefill-chunk] :or {sweep-every 32}}]
   (let [{:keys [model tokenizer]} model-map
         eos (llm/eos-token-id tokenizer)]
     (dyn/auto-key
      (gen [prompt-ids max-tokens]
           (if (zero? max-tokens)
             prompt-ids
             (do
               (llm/init-cache! model)
               (try
                 (let [logits (if (or (seq images) prefill-chunk)
                                (llm/forward-prefill model prompt-ids
                                                     (cond-> {}
                                                       (seq images) (assoc :images images)
                                                       prefill-chunk (assoc :chunk prefill-chunk)))
                                (llm/forward-prefill model prompt-ids))]
                   (loop [i 0, context prompt-ids, logits logits]
                     (if (>= i max-tokens)
                       context
                       (let [tok (trace (t-addr i) (dist/categorical logits))
                             tok-id (mx/item tok)]
                         (if (= tok-id eos)
                           (conj context tok-id)
                           (do (llm/sweep-tick! i sweep-every)
                               (let [next-logits (llm/forward-step model tok-id)]
                                 (recur (inc i) (conj context tok-id) next-logits))))))))
                 (finally
                   (llm/reset-cache! model))))))))))

(defn with-slab-assess
  "Wrap an LLM GF (make-llm-gf) so p/assess scores all constrained tokens
   in ONE teacher-forcing forward over prompt++tokens
   (backend/forward-branch-scores) instead of replaying the body's
   per-token step recurrence.

   Assess with a fully-constrained choicemap needs no sampling: the slab
   computes the identical Σ log p(t_i | prompt, t_0..t_{i-1}) in one
   forward. Same compiled-vs-handler discipline as Layer 7 — the handler
   body stays ground truth, the slab is the optimization; they agree
   exactly in exact arithmetic and differ only by the documented bf16
   graph-shape drift (GDN chunk scan vs step recurrence — the ≤1.0/token
   bound measured by pi_assess_test law B). It is also the graph shape of
   pi-assess/session-scores, so walk == assess becomes tight instead of
   drift-bounded (genmlx-3n7b: on sm_120 the step-vs-slab drift exceeded
   the old 0.1 D law on a 2-token span).

   Semantics mirror the body's assess exactly: sites :t0.. consumed in
   order, stopping after an EOS token or at max-tokens; every consumed
   site must be constrained; retval = prompt ++ consumed tokens. Requires
   the OWNED forward (CljsForwardModel) — other model types fall back to
   the handler path, and all other GFI ops delegate to the base GF
   unchanged. Scoring runs on a fresh owned branch (disposed in finally),
   NOT the model-internal cache. Do not stack under grammar middleware:
   ::custom-dispatch outranks with-handler, so a grammar-masked assess
   must stay on the unwrapped GF."
  [gf model-map]
  (let [{:keys [model tokenizer]} model-map
        eos (llm/eos-token-id tokenizer)
        slab-assess
        (fn [args constraints]
          (let [[prompt-ids max-tokens] args
                prompt (vec prompt-ids)
                toks (loop [i 0, acc []]
                       (if (>= i max-tokens)
                         acc
                         (let [v (cm/get-value (cm/get-submap constraints (t-addr i)))]
                           (when (nil? v)
                             (throw (ex-info (str "with-slab-assess: site " (t-addr i)
                                                  " unconstrained — assess requires every visited site")
                                             {:genmlx/error :assess-missing-constraint
                                              :addr (t-addr i)})))
                           (let [id (if (number? v) v (mx/item v))
                                 acc' (conj acc id)]
                             (if (== id eos) acc' (recur (inc i) acc'))))))]
            (if (empty? toks)
              {:weight (mx/scalar 0.0) :retval prompt}
              (let [b (llm/owned-branch! model {:cache nil :offset 0})]
                (try
                  (let [scores (llm/forward-branch-scores model b (into prompt toks))
                        host   (vec (mx/->clj scores))
                        s      (count prompt)
                        span   (subvec host (dec s) (+ (dec s) (count toks)))]
                    {:weight (mx/scalar (reduce + 0.0 span))
                     :retval (into prompt toks)})
                  (finally (llm/dispose-branch! model b)))))))]
    (dispatch/with-dispatch gf
      (fn [op _gf2 args key opts]
        (if (and (= op :assess) (llm/cljs-forward-model? model))
          (slab-assess args (:constraints opts))
          (case op
            :simulate   (p/simulate   (dyn/with-key gf key) args)
            :generate   (p/generate   (dyn/with-key gf key) args (:constraints opts))
            :assess     (p/assess     (dyn/with-key gf key) args (:constraints opts))
            :update     (p/update     (dyn/with-key gf key) (:trace opts) (:constraints opts))
            :regenerate (p/regenerate (dyn/with-key gf key) (:trace opts) (:selection opts))
            :project    (p/project    (dyn/with-key gf key) (:trace opts) (:selection opts))
            :propose    (p/propose    (dyn/with-key gf key) args)))))))

(defn make-llm-gf-uncached
  "Like make-llm-gf but without KV cache. Recomputes full context at
   each token step — O(n²) but stateless. Useful for debugging or when
   the model doesn't support KV cache.

   opts: :sweep-every — in-loop dead-wrapper sweep every N tokens (default
   32; 0/nil disables), as in make-llm-gf (genmlx-nwsr)."
  ([model-map] (make-llm-gf-uncached model-map {}))
  ([model-map {:keys [sweep-every] :or {sweep-every 32}}]
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
                   (do (llm/sweep-tick! i sweep-every)
                       (recur (inc i) (conj context tok-id))))))))))))

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
     :temperature — sampling temperature (default nil = 1.0/raw logits).
                    Applied as a 1/T logit scale BEFORE the hook, so the
                    :hook slot stays free for grammar; -inf grammar masks
                    are unmoved by scaling. The gf's DISTRIBUTION is then
                    defined over the scaled logits (assess/score use the
                    same scaling — the L1 law holds per gf instance).
                    Must be > 0 (greedy is a scalar-path concern —
                    generate-text-raw+).
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
     :sweep-every — in-loop dead-wrapper sweep every N sites (default 32;
                    0/nil disables), as in make-llm-gf (genmlx-nwsr). Under
                    scalar execution (per-site mx/item) it bounds retention
                    exactly like the scalar loop; under vsimulate the loop
                    stays lazy so there is little to free, and the sweep is
                    near-free.

   Scalar GFI ops on this gf work through broadcasting (shapes []), but run
   the full max-tokens loop (no early exit) — for scalar use, make-llm-gf
   remains the right constructor.

   Retval: {:tokens [K max-tokens] int matrix ([max-tokens] under scalar
   execution), :active the final lane-liveness mask, :prompt-ids}."
  ([model-map] (make-llm-gf-batched model-map {}))
  ([model-map {:keys [pad-id temperature hook check-every sweep-every]
               :or {sweep-every 32}}]
   (let [{:keys [model tokenizer]} model-map
         _ (when-not (llm/cljs-forward-model? model)
             (throw (ex-info "make-llm-gf-batched requires the OWNED forward (CljsForwardModel) — load with {:cljs-forward? true} (or a supported family's smart default)."
                             {:genmlx/error :batched-gf-owned-only
                              :model-type (type model)})))
         _ (when (and temperature (not (pos? temperature)))
             (throw (ex-info "make-llm-gf-batched :temperature must be > 0 (greedy decoding is the scalar path's concern)."
                             {:genmlx/error :bad-temperature :temperature temperature})))
         inv-temp (when (and temperature (not= temperature 1.0))
                    (mx/scalar (/ 1.0 temperature)))
         eos     (llm/eos-token-id tokenizer)
         vocab   (get-in model [:fwd :config :vocab])
         pad     (or pad-id
                     (let [p (llm/pad-token-id tokenizer)]
                       (if (and (some? p) (>= p 0) (< p vocab)) p eos)))
         pad-row (pad-onehot-row vocab pad)
         pad-tok (mx/scalar pad mx/int32)
         ;; [K]-shaped (per-lane) vs [] (shared scalar) value?
         batched? (fn [a] (pos? (count (mx/shape a))))
         ;; Site values can MIX shapes: a site constrained with a shared
         ;; scalar observation traces [], sampled sites trace [K] (and the
         ;; cache stays B=1 through a constrained prefix — the first sampled
         ;; token tiles it). Broadcast scalars up before stacking.
         stack-toks (fn [toks]
                      (let [k    (some #(when (batched? %) (first (mx/shape %)))
                                       toks)
                            toks (if k
                                   (mapv #(if (batched? %)
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
                   (let [lg   (if inv-temp (mx/multiply logits inv-temp) logits)
                         lg   (if hook ((:mask hook) hs lg i) lg)
                         lg   (if active (mask-inactive-logits lg active pad-row) lg)
                         tok  (trace (t-addr i) (dist/categorical lg))
                         act' (advance-active active tok eos)
                         hs'  (when hook ((:advance hook) hs tok))
                         done? (or (= (inc i) max-tokens)
                                   (and check-every
                                        (batched? tok)
                                        (zero? (mod (inc i) check-every))
                                        (all-dead? act')))]
                     (if done?
                       {:tokens (stack-toks (conj toks tok)) :active act'
                        :prompt-ids prompt-ids}
                       ;; still-active lanes feed their sample; dead lanes
                       ;; (incl. just-eos'd) feed pad — eos is never fed.
                       (let [fed (mx/where act' tok pad-tok)]
                         (llm/sweep-tick! i sweep-every)
                         (recur (inc i)
                                (if (batched? fed)
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
                  pre  (count (take-while #(not= % eos-id) full))]
              (if (= pre (count full)) full (subvec full 0 (inc pre)))))
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

(defn generate-texts-batched
  "K sampled chat completions from ONE batched owned forward — the Route B
   (genmlx-9uyg) text-level API (genmlx-789s). Replaces K sequential
   generate-text-raw+ calls with a single shared-prefill, K-lane lockstep
   decode: measured K=8 decode ≈ 1.33× ONE scalar step's cost, i.e. ~6×
   throughput on best-of-K proposal generation.

   Prompt rendering matches generate-text-raw+ (render-chat: system + user,
   think-skip on the Qwen3/3.5 families); per-lane early stop is the
   masked-EOS algebra + a :check-every host early-exit (safe here —
   unconstrained sampling only).

   model-map: from load-model, OWNED forward required ({:cljs-forward? true}
   or a supported family's smart default).
   opts:
     :n            lanes / completions (default 4)
     :max-tokens   per-lane cap (default 256)
     :temperature  sampling temperature, must be > 0 (default 0.8; greedy
                   best-of-K is meaningless — lanes would be identical)
     :seed         PRNG seed; same seed => same K texts (default 1)
     :system-prompt (default \"You are a helpful assistant.\")
     :check-every  host early-exit period in sites (default 32; 0/nil off)
     :sweep-every  dead-wrapper sweep period (default 32)

   Returns a promise of {:texts [str ...K] :n-tokens [int ...K] :gen-ms n}
   — :n-tokens counts each lane's tokens up to and including its eos
   (trailing pads excluded); :gen-ms is wall-clock through the last decode
   step (text decode excluded, as in generate-text-raw+)."
  ([model-map prompt] (generate-texts-batched model-map prompt {}))
  ([{model-type :type :keys [model tokenizer] :as model-map} prompt
    {:keys [n max-tokens temperature seed system-prompt check-every sweep-every]
     :or {n 4 max-tokens 256 temperature 0.8 seed 1
          system-prompt "You are a helpful assistant."
          check-every 32 sweep-every 32}}]
   (let [t0 (.now js/Date)
         chat-str (llm/render-chat
                   [{:role "system" :content system-prompt}
                    {:role "user" :content prompt}]
                   {:think-skip? (contains? #{:qwen3 :qwen3_5 :qwen3_5_moe}
                                            model-type)})
         eos (llm/eos-token-id tokenizer)
         gf  (make-llm-gf-batched model-map
                                  (cond-> {:temperature temperature
                                           :sweep-every sweep-every}
                                    (and check-every (pos? check-every))
                                    (assoc :check-every check-every)))]
     (pr/let [ids-raw (llm/encode tokenizer chat-str true)]
       (let [prompt-ids (vec ids-raw)
             vt (dyn/vsimulate gf [prompt-ids max-tokens] n
                               (rng/fresh-key seed))
             lanes (vtrace-lane-tokens vt eos)
             gen-ms (- (.now js/Date) t0)]
         (pr/let [texts (decode-vtrace tokenizer vt)]
           {:texts (vec texts)
            :n-tokens (mapv count lanes)
            :gen-ms gen-ms}))))))

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
