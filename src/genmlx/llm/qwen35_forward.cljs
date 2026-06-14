(ns genmlx.llm.qwen35-forward
  "f6ov P6: a GenMLX-owned Qwen3.5 / Qwen3-Next *hybrid* forward pass in pure
   ClojureScript, composed over the genmlx.rs array primitives + GenMLX-owned
   weight loading (mx/load-safetensors). Reuses the P2-P4 scaffold from
   genmlx.llm.qwen3-forward (config/weights/layer-loop/KV-cache/parity gate);
   only the per-layer attention kind changes.

   Qwen3.5 is a HYBRID stack: most layers are GatedDeltaNet linear attention
   (Mamba-style gated delta recurrence + short causal depthwise conv1d) and
   every `full_attention_interval`-th layer is full softmax attention with
   PARTIAL RoPE and a sigmoid output gate. `layer_types` in config.json gives
   the per-layer kind.

   Authoritative reference = upstream mlx-node
   crates/mlx-core/src/models/qwen3_5 (the `use_kernel=false` ops path in
   gated_delta.rs is the correctness reference; we are NOT trying to match the
   fused Metal kernel bit-for-bit, only within bf16 cross-kernel tolerance, with
   argmax + top-5 ids exact). Golden oracle (llm_forward_golden_test) is the gate.

   DESIGN NOTES / decisions baked in here (all verified against the Rust source):
   - On-disk checkpoint pre-splits the GDN input projection into 4 tensors
     (in_proj_qkv / in_proj_z / in_proj_b / in_proj_a), NOT the fused
     in_proj_qkvz/in_proj_ba of the Rust struct. We use the split tensors.
   - Tensor prefix is `language_model.model.` (multimodal wrapper); tie_word_embeddings
     => logits = hidden @ embed_tokens^T (no lm_head tensor).
   - RMSNormGated = silu(gate) * rms_norm(value, weight, eps) — gate applied
     AFTER the norm (swiglu(gate, normed) per Activations::swiglu). This matches
     the upstream Rust, which is what produced the golden values.
   - Depthwise causal conv1d is composed as a sum of K shifted, weighted slices
     (cross-correlation, left-padded by K-1) — no conv primitive needed.
   - Full attention applies PARTIAL RoPE over the first rope_dims = head_dim *
     partial_rotary_factor channels, on the [B,H,T,D] layout (position axis),
     and gates the attention output by sigmoid of the second half of q_proj.

   Same 6-fn interface as qwen3-forward (load-model/forward/next-token-logits/
   prefill/step/init-cache) so genmlx.llm.forward / CljsForwardModel drive it."
  (:require [genmlx.mlx :as mx]
            ["fs" :as fs]))

;; ----------------------------------------------------------------------------
;; Config
;; ----------------------------------------------------------------------------

(defn load-config
  "Parse a Qwen3.5 config.json. Text-model fields live under `text_config`;
   RoPE params under `rope_parameters`. `layer_types` is the per-layer kind."
  [dir]
  (let [c    (js/JSON.parse (.readFileSync fs (str dir "/config.json") "utf8"))
        tc   (or (.-text_config c) c)
        rp   (or (.-rope_parameters tc) (.-rope_parameters c) tc)]
    {:hidden        (.-hidden_size tc)
     :n-layers      (.-num_hidden_layers tc)
     :n-heads       (.-num_attention_heads tc)
     :n-kv-heads    (.-num_key_value_heads tc)
     :head-dim      (.-head_dim tc)
     :intermediate  (.-intermediate_size tc)
     :vocab         (.-vocab_size tc)
     :eps           (.-rms_norm_eps tc)
     :rope-theta    (.-rope_theta rp)
     :partial       (.-partial_rotary_factor rp)
     :full-interval (.-full_attention_interval tc)
     :layer-types   (vec (.-layer_types tc))
     ;; linear-attention (GatedDeltaNet) dims
     :lin-k-heads   (.-linear_num_key_heads tc)
     :lin-v-heads   (.-linear_num_value_heads tc)
     :lin-k-dim     (.-linear_key_head_dim tc)
     :lin-v-dim     (.-linear_value_head_dim tc)
     :conv-k        (.-linear_conv_kernel_dim tc)
     :model-type    (.-model_type c)}))

(defn load-model
  "Load a Qwen3.5 checkpoint as {:config .. :weights {name -> MxArray}}."
  [dir]
  {:config  (load-config dir)
   :weights (mx/load-safetensors (str dir "/model.safetensors"))})

(def ^:private wp "language_model.model.")

;; ----------------------------------------------------------------------------
;; Small composed primitives
;; ----------------------------------------------------------------------------

(defn- slice-ax
  "Contiguous slice [start, stop) along `axis` (any axis), via gather. mx/slice
   is axis-0 only, so we use take-idx with an arange index — semantically a slice."
  [a axis start stop]
  (mx/take-idx a (mx/arange start stop) axis))

(defn- linear
  "HF Linear with no bias: y = x W^T (W is [out, in])."
  [x w]
  (mx/matmul x (mx/transpose w)))

(defn- rms-no-weight
  "RMSNorm over the last axis with no learnable weight (weight=1):
   x / sqrt(mean(x^2, -1) + eps). Implemented via the fast rms-norm with a
   ones weight (identical result; weight just multiplies by 1)."
  [x dim eps]
  (mx/rms-norm x (mx/ones [dim]) eps))

(defn- softplus
  "softplus(x) = log(1 + exp(x)). Inputs here (a + dt_bias) are bounded, so the
   naive log1p(exp) form is numerically safe and matches the stable form."
  [x]
  (mx/log1p (mx/exp x)))

(defn- causal-mask
  "Additive causal mask [seq seq]: 0 on/below the diagonal, large-negative above.
   Broadcasts to [1 H seq seq] inside SDPA."
  [seq dtype]
  (let [flat (vec (for [i (range seq) j (range seq)] (if (<= j i) 0.0 -1e9)))]
    (mx/astype (mx/array flat [seq seq]) dtype)))

(defn- mlp
  "SwiGLU MLP over the post-attention-normed hidden `hn`: down(silu(gate)·up)."
  [w prefix hn]
  (let [g (fn [s] (get w (str prefix s)))]
    (linear (mx/multiply (mx/silu (linear hn (g "gate_proj.weight")))
                         (linear hn (g "up_proj.weight")))
            (g "down_proj.weight"))))

;; ----------------------------------------------------------------------------
;; GatedDeltaNet linear-attention layer
;; ----------------------------------------------------------------------------

(defn- gdn-layer
  "GatedDeltaNet linear attention over the pre-attention-normed hidden
   `hn` [1 T hidden]. `ce` is the prior {:conv :rec} cache entry (or nil):
   :conv = last K-1 conv inputs [1 K-1 conv-dim], :rec = recurrent state
   [1 Hv Dv Dk]. Returns [out [1 T hidden] new-cache-entry].

   Mirrors gated_delta_net.rs::forward + gated_delta.rs (use_kernel=false)."
  [cfg w prefix hn T ce]
  (let [{:keys [eps lin-k-heads lin-v-heads lin-k-dim lin-v-dim conv-k]} cfg
        Hk      lin-k-heads
        Hv      lin-v-heads
        Dk      lin-k-dim
        Dv      lin-v-dim
        K       conv-k
        key-dim (* Hk Dk)
        val-dim (* Hv Dv)
        conv-dim (+ (* 2 key-dim) val-dim)            ; q + k + v channels
        g       (fn [s] (get w (str prefix s)))
        dtype   (mx/dtype hn)
        ;; --- input projections (pre-split on disk) ---
        qkv     (linear hn (g "in_proj_qkv.weight"))  ; [1 T conv-dim]
        z       (linear hn (g "in_proj_z.weight"))    ; [1 T val-dim]
        bb      (linear hn (g "in_proj_b.weight"))    ; [1 T Hv]
        aa      (linear hn (g "in_proj_a.weight"))    ; [1 T Hv]
        ;; --- depthwise causal conv1d over qkv (sum of K shifted weighted slices) ---
        conv-w  (g "conv1d.weight")                   ; [conv-dim K 1]
        pad     (if ce (:conv ce) (mx/zeros [1 (dec K) conv-dim] dtype))
        padded  (mx/concatenate [pad qkv] 1)          ; [1 T+K-1 conv-dim]
        conv-out (reduce (fn [acc j]
                           (let [wj  (mx/reshape (slice-ax conv-w 1 j (inc j)) [1 1 conv-dim])
                                 seg (slice-ax padded 1 j (+ j T))]   ; [1 T conv-dim]
                             (mx/add acc (mx/multiply seg wj))))
                         (mx/zeros [1 T conv-dim] dtype)
                         (range K))
        conv-out (mx/silu conv-out)
        ;; new conv-state = last K-1 timesteps of the conv input
        new-conv (slice-ax padded 1 T (+ T (dec K)))
        ;; --- split q/k/v and reshape to heads ---
        q (mx/reshape (slice-ax conv-out 2 0 key-dim)             [1 T Hk Dk])
        k (mx/reshape (slice-ax conv-out 2 key-dim (* 2 key-dim)) [1 T Hk Dk])
        v (mx/reshape (slice-ax conv-out 2 (* 2 key-dim) conv-dim) [1 T Hv Dv])
        ;; --- q/k RMSNorm (no weight) with inv-scale (matches Python/Rust exactly) ---
        inv (js/Math.pow Dk -0.5)
        q (mx/multiply (rms-no-weight q Dk 1e-6) (* inv inv))
        k (mx/multiply (rms-no-weight k Dk 1e-6) inv)
        ;; --- gates: beta = sigmoid(b); g = exp(-exp(A_log) * softplus(a + dt_bias)) ---
        ;; The SSM recurrence + gates run in float32 (config mamba_ssm_dtype:
        ;; "float32") to match the upstream fused kernel's f32 state accumulation;
        ;; projections/conv stay in the model dtype, and y is cast back below so
        ;; the residual stream remains bf16.
        beta   (mx/astype (mx/sigmoid bb) mx/float32)  ; [1 T Hv]
        a-log  (mx/astype (g "A_log") mx/float32)      ; [Hv]
        dt-bias (mx/astype (g "dt_bias") mx/float32)   ; [Hv]
        gg     (mx/exp (mx/multiply (mx/multiply (mx/exp a-log) -1.0)
                                    (softplus (mx/add (mx/astype aa mx/float32) dt-bias))))  ; [1 T Hv]
        ;; --- GQA: repeat q,k from Hk to Hv heads (np.repeat along head axis) ---
        rep (quot Hv Hk)
        q   (mx/astype (if (> rep 1) (mx/repeat-arr q rep 2) q) mx/float32)
        k   (mx/astype (if (> rep 1) (mx/repeat-arr k rep 2) k) mx/float32)
        v   (mx/astype v mx/float32)
        ;; --- gated delta recurrence (sequential over T), f32 accumulation ---
        init-state (if ce (:rec ce) (mx/zeros [1 Hv Dv Dk] mx/float32))
        [final-state outs]
        (reduce
         (fn [[st outs] t]
           (let [qt (mx/squeeze (slice-ax q  1 t (inc t)) [1])   ; [1 Hv Dk]
                 kt (mx/squeeze (slice-ax k  1 t (inc t)) [1])   ; [1 Hv Dk]
                 vt (mx/squeeze (slice-ax v  1 t (inc t)) [1])   ; [1 Hv Dv]
                 gt (mx/squeeze (slice-ax gg 1 t (inc t)) [1])   ; [1 Hv]
                 bt (mx/squeeze (slice-ax beta 1 t (inc t)) [1]) ; [1 Hv]
                 st (mx/multiply st (mx/reshape gt [1 Hv 1 1]))  ; decay state
                 k4 (mx/reshape kt [1 Hv 1 Dk])
                 kv-mem (mx/sum (mx/multiply st k4) [3] false)   ; [1 Hv Dv]
                 delta  (mx/multiply (mx/subtract vt kv-mem) (mx/reshape bt [1 Hv 1]))
                 st (mx/add st (mx/multiply k4 (mx/reshape delta [1 Hv Dv 1])))
                 q4 (mx/reshape qt [1 Hv 1 Dk])
                 yt (mx/sum (mx/multiply st q4) [3] false)]      ; [1 Hv Dv]
             [st (conj outs (mx/reshape yt [1 1 Hv Dv]))]))
         [init-state []] (range T))
        y (mx/astype (mx/concatenate outs 1) dtype)    ; [1 T Hv Dv], back to model dtype
        ;; --- gated RMSNorm: silu(z) * rms_norm(y, norm_weight[Dv], eps) ---
        z4 (mx/reshape z [1 T Hv Dv])
        y-norm (mx/multiply (mx/silu z4) (mx/rms-norm y (g "norm.weight") eps))
        y-flat (mx/reshape y-norm [1 T val-dim])]
    [(linear y-flat (g "out_proj.weight")) {:conv new-conv :rec final-state}]))

;; ----------------------------------------------------------------------------
;; Full softmax attention layer (partial RoPE + output gate)
;; ----------------------------------------------------------------------------

(defn- full-attn-layer
  "Qwen3.5 full attention over `hn` [1 T hidden]. q_proj is 2x width: first
   head-dim = query, second = output gate (sigmoid). PARTIAL RoPE over the first
   rope_dims of head_dim. `ce` = prior {:k :v} (post-RoPE) cache or nil; `offset`
   is the absolute position of the first new token. Returns [out new-cache-entry].

   Mirrors attention.rs::Qwen3_5Attention::forward."
  [cfg w prefix hn T ce offset mask]
  (let [{:keys [n-heads n-kv-heads head-dim eps rope-theta partial]} cfg
        H   n-heads
        Hkv n-kv-heads
        hd  head-dim
        rope-dims (js/Math.floor (* hd partial))
        scale (js/Math.pow hd -0.5)
        g   (fn [s] (get w (str prefix s)))
        qg  (mx/reshape (linear hn (g "q_proj.weight")) [1 T H (* 2 hd)])
        queries (slice-ax qg 3 0 hd)                          ; [1 T H hd]
        gate    (mx/reshape (slice-ax qg 3 hd (* 2 hd)) [1 T (* H hd)])
        keys    (mx/reshape (linear hn (g "k_proj.weight")) [1 T Hkv hd])
        values  (mx/reshape (linear hn (g "v_proj.weight")) [1 T Hkv hd])
        queries (mx/rms-norm queries (g "q_norm.weight") eps)
        keys    (mx/rms-norm keys    (g "k_norm.weight") eps)
        ;; transpose to [1 heads T hd], then partial RoPE at absolute offset
        q (-> (mx/transpose queries [0 2 1 3]) (mx/rope rope-dims false rope-theta 1.0 offset))
        k (-> (mx/transpose keys    [0 2 1 3]) (mx/rope rope-dims false rope-theta 1.0 offset))
        v (mx/transpose values [0 2 1 3])
        k-full (if ce (mx/concatenate [(:k ce) k] 2) k)        ; concat along seq
        v-full (if ce (mx/concatenate [(:v ce) v] 2) v)
        o (-> (mx/scaled-dot-product-attention q k-full v-full scale mask)
              (mx/transpose [0 2 1 3])                          ; [1 T H hd]
              (mx/reshape [1 T (* H hd)]))
        o (mx/multiply o (mx/sigmoid gate))]                   ; output gate
    [(linear o (g "o_proj.weight")) {:k k-full :v v-full}]))

;; ----------------------------------------------------------------------------
;; Forward
;; ----------------------------------------------------------------------------

(defn init-cache
  "Empty per-layer cache (one nil slot per layer; the slot shape differs by
   layer kind but the caller only threads it through)."
  [{:keys [config]}]
  (vec (repeat (:n-layers config) nil)))

(defn forward-cached
  "Run the forward over `token-ids` from absolute position `offset`, threading
   and extending the per-layer `cache`. Returns [logits new-cache] with logits
   [seq vocab]. A causal mask is built for multi-token chunks (prefill)."
  [{:keys [config weights]} token-ids cache offset]
  (let [{:keys [n-layers eps hidden vocab layer-types]} config
        T     (count token-ids)
        ids   (mx/array (vec token-ids) [T] mx/int32)
        embed (get weights (str wp "embed_tokens.weight"))
        dtype (mx/dtype embed)
        mask  (when (> T 1) (causal-mask T dtype))
        h0    (mx/reshape (mx/take-idx embed ids 0) [1 T hidden])
        [h new-cache]
        (reduce
         (fn [[h nc] i]
           (let [p  (str wp "layers." i ".")
                 hn (mx/rms-norm h (get weights (str p "input_layernorm.weight")) eps)
                 [a ce'] (if (= "linear_attention" (nth layer-types i))
                           (gdn-layer config weights (str p "linear_attn.") hn T (nth cache i))
                           (full-attn-layer config weights (str p "self_attn.") hn T
                                            (nth cache i) offset mask))
                 h1 (mx/add h a)
                 mn (mx/rms-norm h1 (get weights (str p "post_attention_layernorm.weight")) eps)]
             [(mx/add h1 (mlp weights (str p "mlp.") mn)) (conj nc ce')]))
         [h0 []] (range n-layers))
        hf (mx/rms-norm h (get weights (str wp "norm.weight")) eps)]
    [(mx/reshape (linear hf embed) [T vocab]) new-cache]))      ; tied lm_head

(defn forward
  "Uncached full forward over a token-id sequence. Returns logits [seq vocab]."
  [model token-ids]
  (first (forward-cached model token-ids (init-cache model) 0)))

(defn next-token-logits
  "Last-position logits [vocab] for the next token given a prompt token-id seq."
  [model token-ids]
  (mx/index (forward model token-ids) (dec (count token-ids))))

(defn prefill
  "Process the full prompt, populating the cache. Returns [last-logits cache]
   where last-logits is [vocab]."
  [model prompt-ids]
  (let [[logits cache] (forward-cached model (vec prompt-ids) (init-cache model) 0)]
    [(mx/index logits (dec (count prompt-ids))) cache]))

(defn step
  "Advance one token from `cache` (current length = `offset`). Returns
   [logits cache'] with logits [vocab]."
  [model cache offset token-id]
  (let [[logits cache'] (forward-cached model [token-id] cache offset)]
    [(mx/index logits 0) cache']))
