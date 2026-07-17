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
   - Tensor prefix is `language_model.model.` (multimodal wrapper). Logits:
     tied checkpoints (dense qwen3.5) use hidden @ embed_tokens^T; untied
     ones (Ornith / qwen3_5_moe) ship a real `language_model.lm_head` tensor.
   - qwen3_5_moe (genmlx-g6vk): the dense-MLP seam branches per layer to a
     sparse-MoE block — router softmax (precise f32, over ALL experts, BEFORE
     top-k), argsort top-k, renormalized weights, SwitchGLU via mx/gather-qmm
     over PACKED expert tensors, plus an always-on sigmoid-gated shared
     expert. Experts stay quantized end-to-end (spec §3): same kernel as the
     native path, third of the memory of dequantizing.
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
            [genmlx.llm.qwen3-forward :as q3]
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
     :mrope-section (some-> (.-mrope_section rp) vec)
     :partial       (.-partial_rotary_factor rp)
     :full-interval (.-full_attention_interval tc)
     :layer-types   (vec (.-layer_types tc))
     ;; linear-attention (GatedDeltaNet) dims
     :lin-k-heads   (.-linear_num_key_heads tc)
     :lin-v-heads   (.-linear_num_value_heads tc)
     :lin-k-dim     (.-linear_key_head_dim tc)
     :lin-v-dim     (.-linear_value_head_dim tc)
     :conv-k        (.-linear_conv_kernel_dim tc)
     ;; untied lm_head (Ornith ships one; dense qwen3.5 ties). Default = tied
     ;; when the field is absent — matching HF semantics for this family.
     :tie?          (not (false? (.-tie_word_embeddings tc)))
     ;; sparse-MoE dims (qwen3_5_moe; absent => dense MLP everywhere)
     :n-experts     (.-num_experts tc)
     :n-active      (.-num_experts_per_tok tc)
     :moe-inter     (.-moe_intermediate_size tc)
     :sparse-step   (or (.-decoder_sparse_step tc) 1)
     :mlp-only      (set (vec (or (.-mlp_only_layers tc) #js [])))
     :model-type    (.-model_type c)}))

(defn- moe-layer?
  "Layer i uses the sparse-MoE MLP (mirrors the upstream gating: every
   `decoder_sparse_step`-th layer, except `mlp_only_layers`, when experts
   exist)."
  [{:keys [n-experts sparse-step mlp-only]} i]
  (and (some? n-experts) (pos? n-experts)
       (pos? sparse-step)
       (zero? (mod (inc i) sparse-step))
       (not (contains? mlp-only i))))

(def ^:private wp "language_model.model.")

(defn- gdn-derived
  "Load-time derived per-GDN-layer constants (genmlx-t2cz L4): for each
   linear_attention layer prefix, :neg-a = -exp(f32 A_log) and :dt-f32 =
   f32 dt_bias — the weight-derived g-log inputs that otherwise rebuild 4
   lazy nodes per layer per decode step — plus a shared :ones-dk for the
   weightless q/k RMSNorm. Pure load-time value derivation (lazy views over
   immutable weights, built outside any tidy scope)."
  [cfg weights]
  (let [{:keys [lin-k-heads lin-v-heads lin-k-dim lin-v-dim conv-k]} cfg
        conv-dim (+ (* 2 lin-k-heads lin-k-dim) (* lin-v-heads lin-v-dim))]
    (into {:ones-dk (mx/ones [lin-k-dim])}
          (for [i     (range (:n-layers cfg))
                :when (= "linear_attention" (nth (:layer-types cfg) i))
                :let  [p (str wp "layers." i ".linear_attn.")]]
            [p {:neg-a   (mx/multiply (mx/exp (mx/astype (get weights (str p "A_log"))
                                                         mx/float32))
                                      -1.0)
                :dt-f32  (mx/astype (get weights (str p "dt_bias")) mx/float32)
                ;; [K conv-dim] layout of conv1d.weight for the T=1 decode
                ;; conv fast path (broadcast-multiply + K-axis sum)
                :conv-w2 (mx/transpose (mx/reshape (get weights (str p "conv1d.weight"))
                                                   [conv-dim conv-k])
                                       [1 0])}]))))

(defn load-model
  "Load a Qwen3.5 / qwen3_5_moe checkpoint as {:config .. :weights ..}.
   Affine-quantized checkpoints are dequantized at load (q3/load-weights —
   genmlx-9iqc/-vmks): the packed U32 [out, in/8] weights otherwise abort the
   forward at the first reshape (e.g. size T*(hidden/8) into (1,T,hidden)).

   MoE checkpoints (genmlx-g6vk, spec §3/§5.3): the `switch_mlp` expert
   tensors stay PACKED (weight+scales+biases) and are driven by mx/gather-qmm
   in the forward — dequantizing Ornith's 32B expert params would triple them
   to ~64 GB. Everything else dequantizes as usual. The experts' (bits,
   group-size) land in config :expert-qz; per-tensor overrides on switch_mlp
   tensors are rejected (none exist in any known checkpoint — the overrides
   cover routers/gates only)."
  [dir]
  (let [cfg  (load-config dir)
        moe? (and (:n-experts cfg) (pos? (:n-experts cfg)))
        qz   (q3/load-quantization dir)]
    (when (and moe? qz)
      (doseq [[base _] (:overrides qz)]
        (when (.includes base ".switch_mlp.")
          (throw (ex-info (str "qwen35 load-model: per-tensor quantization "
                               "override on packed expert tensor " base
                               " — gather-qmm runs experts at the GLOBAL "
                               "(bits, group-size); this checkpoint needs "
                               "per-projection expert quantization support.")
                          {:base base :quantization qz})))))
    (let [weights (q3/prepare-weight-transposes!
                   (q3/load-weights dir (when (and moe? qz)
                                          {:skip? #(.includes % ".switch_mlp.")})))
          cfg     (cond-> cfg
                    (and moe? qz) (assoc :expert-qz {:bits       (:bits qz)
                                                     :group-size (:group-size qz)}))]
      {:config  (assoc cfg :derived (gdn-derived cfg weights))
       :weights weights})))

;; ----------------------------------------------------------------------------
;; Small composed primitives
;; ----------------------------------------------------------------------------

(def ^:private END
  "'To the end' slice stop: mlx slice CLAMPS stops to the dim (numpy
   semantics), so no shape query is needed for open-ended axes."
  1073741824)

(defn- slice-ax
  "Contiguous slice [start, stop) along `axis` of an `nd`-dim array — ONE
   membrane call (mx/slice-nd), a VIEW at eval. Replaces the arange+take
   gather (3 NAPI crossings + a gather kernel per call — the largest single
   contributor to the decode per-op floor, genmlx-t2cz P0: ~570 gather
   slices per 35B decode step)."
  [a nd axis start stop]
  (mx/slice-nd a
               (assoc (vec (repeat nd 0)) axis start)
               (assoc (vec (repeat nd END)) axis stop)))

(defn- linear
  "HF Linear with no bias: y = x W^T (W is [out, in]). Wᵀ comes from the
   load-time transposed-view memo when `w` is a prepared weight
   (q3/transposed, genmlx-t2cz)."
  [x w]
  (mx/matmul x (q3/transposed w)))

(defn- softplus
  "softplus(x) = log(1 + exp(x)). Inputs here (a + dt_bias) are bounded, so the
   naive log1p(exp) form is numerically safe and matches the stable form."
  [x]
  (mx/log1p (mx/exp x)))

(defn- causal-mask
  "Additive causal mask [seq, prior+seq] for a block of `seq` new tokens on
   top of `prior` cached ones: row i attends to every cached column and to
   new columns j <= i. Broadcasts to [1 H seq prior+seq] inside SDPA.
   prior=0 gives the classic square causal mask."
  ([seq dtype] (causal-mask seq 0 dtype))
  ([seq prior dtype]
   (let [flat (vec (for [i (range seq) j (range (+ prior seq))]
                     (if (<= j (+ prior i)) 0.0 -1e9)))]
     (mx/astype (mx/array flat [seq (+ prior seq)]) dtype))))

(defn- mlp
  "SwiGLU MLP over the post-attention-normed hidden `hn`: down(silu(gate)·up)."
  [w prefix hn]
  (let [g (fn [s] (get w (str prefix s)))]
    (linear (mx/multiply (mx/silu (linear hn (g "gate_proj.weight")))
                         (linear hn (g "up_proj.weight")))
            (g "down_proj.weight"))))

(defn- switch-glu
  "One SwitchGLU projection through the PACKED [E, out, in-packed] expert
   tensor at `prefix` (+.scales/.biases), via mx/gather-qmm with per-row
   expert indices `idx`. x [.., 1, in] -> [.., k, 1, out] following mlx-lm's
   SwitchLinear broadcasting (idx [T k] against x [T 1 1 in])."
  [w prefix x idx {:keys [bits group-size]}]
  (let [g (fn [s] (get w (str prefix s)))]
    (mx/gather-qmm x (g "weight") (g "scales") (g "biases")
                   {:rhs-indices idx :bits bits :group-size group-size
                    :transpose true :sorted? false})))

(defn- moe-intervene
  "Interpretability intervention on router probs [BT E] (genmlx-4wsn):
     {:ablate #{ids}}   — zero those experts' routing probability: they can
                          never enter the top-k; renormalization
                          redistributes their share to the survivors.
     {:boost {id amt}}  — add amt to those experts' probability pre-top-k:
                          a continuous steering dial (probs are <=1, so
                          amt >= 1.0 pins the expert into every token's
                          top-k with dominant weight).
   Pure: builds the [1 E] adjustment rows from the spec and broadcasts."
  [probs {:keys [ablate boost]} E]
  (let [probs (if (seq ablate)
                (mx/multiply probs
                             (mx/array (mapv #(if (contains? ablate %) 0.0 1.0)
                                             (range E))
                                       [1 E] mx/float32))
                probs)]
    (if (seq boost)
      (mx/add probs
              (mx/array (mapv #(double (get boost % 0.0)) (range E))
                        [1 E] mx/float32))
      probs)))

(defn moe-mlp
  "Sparse-MoE MLP (qwen3_5_moe) over the post-attention-normed hidden `hn`
   [B T hidden]. Mirrors sparse_moe.rs / router.rs (genmlx-g6vk D1):

     router logits -> softmax over ALL experts in PRECISE f32 (before top-k)
     -> top-k by argsort (mx/topk returns values, not indices)
     -> renormalize the k weights (norm_topk_prob, default true)
     -> SwitchGLU: down(silu(gate(x,idx)) * up(x,idx)) via gather-qmm on the
        packed expert tensors (the same kernel the native path drives)
     -> weighted sum over the k experts
     -> + shared expert (dense SwiGLU, runs for EVERY token), sigmoid-gated
        by a per-token scalar.

   `sorted?` stays false: measured on this backend the flag is a performance
   hint, not a contract (gather_qmm_oracle_test Part 3); sorting tokens by
   expert is a later optimization.

   Batch (genmlx-9uyg): `hn` is [B T hidden]; routing/experts are row-wise,
   so the B·T rows flatten together and the batch axis costs nothing.
   `b` is threaded from forward-hidden (one shape query per forward,
   genmlx-t2cz).

   Interpretability seams (genmlx-4wsn; both nil in production):
     cfg :moe-intervene — {prefix-or-:all {:ablate #{ids} :boost {id amt}}}
       router-prob intervention applied pre-top-k (see moe-intervene).
     cfg :moe-tap — (fn [prefix top-idx topw]) called with THIS layer's
       lazy [BT k] top-expert indices + renormalized weights: the
       router-inspection tap. Caller-owned collection state (the nn.cljs
       caller-owned-state pattern); the owned forward stays pure with a
       nil tap."
  [cfg w prefix hn b T]
  (let [{:keys [hidden n-experts n-active expert-qz]} cfg
        BT     (* b T)
        g      (fn [s] (get w (str prefix s)))
        dtype  (mx/dtype hn)
        x      (mx/reshape hn [BT hidden])
        ;; --- router (gate.weight is dequantized at load) ---
        probs  (mx/softmax (mx/astype (linear x (g "gate.weight")) mx/float32) -1)
        probs  (if-let [iv (and (:moe-intervene cfg)
                                (or (get (:moe-intervene cfg) prefix)
                                    (get (:moe-intervene cfg) :all)))]
                 (moe-intervene probs iv n-experts)
                 probs)
        order  (mx/argsort probs -1)                              ; [BT E] ascending
        top    (mx/astype (slice-ax order 2 1 (- n-experts n-active) n-experts)
                          mx/uint32)                              ; [BT k]
        topw   (mx/take-along-axis probs (mx/astype top mx/int32) 1) ; [BT k] f32
        topw   (mx/divide topw (mx/sum topw [1] true))            ; renormalize
        _      (when-let [tap (:moe-tap cfg)] (tap prefix top topw))
        ;; --- experts: x [BT 1 1 hidden] x idx [BT k] -> [BT k 1 moe-inter] ---
        x4     (mx/reshape x [BT 1 1 hidden])
        gate-o (switch-glu w (str prefix "switch_mlp.gate_proj.") x4 top expert-qz)
        up-o   (switch-glu w (str prefix "switch_mlp.up_proj.")   x4 top expert-qz)
        expert (switch-glu w (str prefix "switch_mlp.down_proj.")
                           (mx/multiply (mx/silu gate-o) up-o) top expert-qz)
        expert (mx/reshape expert [BT n-active hidden])           ; [BT k hidden]
        moe-o  (mx/sum (mx/multiply expert
                                    (mx/reshape (mx/astype topw dtype)
                                                [BT n-active 1]))
                       [1] false)                                 ; [BT hidden]
        ;; --- shared expert: dense SwiGLU for every token, sigmoid-gated ---
        shared (mx/reshape (mlp w (str prefix "shared_expert.") x) [BT hidden])
        sgate  (mx/sigmoid (linear x (g "shared_expert_gate.weight")))  ; [BT 1]
        out    (mx/add moe-o (mx/multiply shared sgate))
        ;; cfg :moe-tap-inner — (fn [prefix label lazy-array]) over the expert
        ;; branch's intermediates (genmlx-mdet Phase-1.5: the L00.mlp-out
        ;; divergence with bit-identical routing pins the jitter INSIDE this
        ;; block; these taps name the op). Same caller-owned-state contract as
        ;; :moe-tap / :layer-tap; nil in production.
        _      (when-let [tap (:moe-tap-inner cfg)]
                 (tap prefix :gate-o gate-o)
                 (tap prefix :up-o up-o)
                 (tap prefix :expert-down expert)
                 (tap prefix :moe-sum moe-o)
                 (tap prefix :shared shared)
                 (tap prefix :sgate sgate))]
    (mx/reshape out [b T hidden])))

;; ----------------------------------------------------------------------------
;; GatedDeltaNet linear-attention layer
;; ----------------------------------------------------------------------------

(defn gdn-recur-steps
  "Token-serial gated-delta recurrence over q/k [B T Hv Dk], v [B T Hv Dv],
   gg/beta [B T Hv] (gg is the EXP-space gate), state [B Hv Dv Dk] — all f32.
   Returns [y [B T Hv Dv] state']. B is read from the shapes (the
   [K]-particle batch axis, genmlx-9uyg; at B=1 this is byte-identical to the
   pre-9uyg inline reduce). This is the CORRECTNESS REFERENCE for the fused
   mx/gated-delta-scan: gdn_scan_contract_test uses THIS fn as its per-step
   reference, so the production decode path and the scan gate cannot drift."
  [q k v gg beta init-state]
  (let [[b T Hv Dk] (mx/shape q)
        Dv (last (mx/shape v))
        [final-state outs]
        (reduce
         (fn [[st outs] t]
           (let [qt (mx/squeeze (slice-ax q  4 1 t (inc t)) [1])   ; [B Hv Dk]
                 kt (mx/squeeze (slice-ax k  4 1 t (inc t)) [1])   ; [B Hv Dk]
                 vt (mx/squeeze (slice-ax v  4 1 t (inc t)) [1])   ; [B Hv Dv]
                 gt (mx/squeeze (slice-ax gg 3 1 t (inc t)) [1])   ; [B Hv]
                 bt (mx/squeeze (slice-ax beta 3 1 t (inc t)) [1]) ; [B Hv]
                 st (mx/multiply st (mx/reshape gt [b Hv 1 1]))  ; decay state
                 k4 (mx/reshape kt [b Hv 1 Dk])
                 kv-mem (mx/sum (mx/multiply st k4) [3] false)   ; [B Hv Dv]
                 delta  (mx/multiply (mx/subtract vt kv-mem) (mx/reshape bt [b Hv 1]))
                 st (mx/add st (mx/multiply k4 (mx/reshape delta [b Hv Dv 1])))
                 q4 (mx/reshape qt [b Hv 1 Dk])
                 yt (mx/sum (mx/multiply st q4) [3] false)]      ; [B Hv Dv]
             [st (conj outs (mx/reshape yt [b 1 Hv Dv]))]))
         [init-state []] (range T))]
    [(mx/concatenate outs 1) final-state]))

(defn- gdn-layer
  "GatedDeltaNet linear attention over the pre-attention-normed hidden
   `hn` [B T hidden] (B read from the shape — the [K]-particle batch axis,
   genmlx-9uyg; B=1 for the scalar paths). `ce` is the prior {:conv :rec}
   cache entry (or nil): :conv = last K-1 conv inputs [B K-1 conv-dim],
   :rec = recurrent state [B Hv Dv Dk]. Returns [out [B T hidden]
   new-cache-entry].

   Mirrors gated_delta_net.rs::forward + gated_delta.rs (use_kernel=false).
   The recurrence routes by block size: T>1 (prefill chunks) takes ONE fused
   chunk-parallel mx/gated-delta-scan membrane call (the native CUDA prefill
   algorithm, BT=64 WY form; batch-ready by construction — genmlx-ps8a);
   T=1 (decode) takes ONE fused mx/gated-delta-step membrane call (the same
   math as one gdn-recur-steps iteration — genmlx-t2cz). gdn-recur-steps
   remains in source as the host-loop correctness reference; parity of both
   fused primitives against it is pinned by gdn_scan_contract_test.
   `b` is threaded from forward-hidden (one shape query per forward)."
  [cfg w prefix hn b T ce]
  (let [{:keys [eps lin-k-heads lin-v-heads lin-k-dim lin-v-dim conv-k derived]} cfg
        Hk      lin-k-heads
        Hv      lin-v-heads
        Dk      lin-k-dim
        Dv      lin-v-dim
        K       conv-k
        key-dim (* Hk Dk)
        val-dim (* Hv Dv)
        conv-dim (+ (* 2 key-dim) val-dim)            ; q + k + v channels
        g       (fn [s] (get w (str prefix s)))
        der     (get derived prefix)
        dtype   (mx/dtype hn)
        ;; --- input projections (pre-split on disk) ---
        qkv     (linear hn (g "in_proj_qkv.weight"))  ; [B T conv-dim]
        z       (linear hn (g "in_proj_z.weight"))    ; [B T val-dim]
        bb      (linear hn (g "in_proj_b.weight"))    ; [B T Hv]
        aa      (linear hn (g "in_proj_a.weight"))    ; [B T Hv]
        ;; --- depthwise causal conv1d over qkv (sum of K shifted weighted slices) ---
        conv-w  (g "conv1d.weight")                   ; [conv-dim K 1]
        pad     (if ce (:conv ce) (mx/zeros [b (dec K) conv-dim] dtype))
        padded  (mx/concatenate [pad qkv] 1)          ; [B T+K-1 conv-dim]
        conv-out (if (= T 1)
                   ;; T=1 (decode): padded is exactly [B K conv-dim] — the
                   ;; whole conv is one broadcast-multiply + K-axis sum
                   ;; (genmlx-t2cz L3; ~4 lazy ops instead of ~5 per tap).
                   (mx/sum (mx/multiply padded
                                        (or (:conv-w2 der)
                                            (mx/transpose (mx/reshape conv-w [conv-dim K])
                                                          [1 0])))
                           [1] true)                  ; [B 1 conv-dim]
                   (reduce (fn [acc j]
                             (let [wj  (mx/reshape (slice-ax conv-w 3 1 j (inc j)) [1 1 conv-dim])
                                   seg (slice-ax padded 3 1 j (+ j T))]   ; [B T conv-dim]
                               (mx/add acc (mx/multiply seg wj))))
                           (mx/zeros [b T conv-dim] dtype)
                           (range K)))
        conv-out (mx/silu conv-out)
        ;; new conv-state = last K-1 timesteps of the conv input
        new-conv (slice-ax padded 3 1 T (+ T (dec K)))
        ;; --- split q/k/v and reshape to heads ---
        q (mx/reshape (slice-ax conv-out 3 2 0 key-dim)             [b T Hk Dk])
        k (mx/reshape (slice-ax conv-out 3 2 key-dim (* 2 key-dim)) [b T Hk Dk])
        v (mx/reshape (slice-ax conv-out 3 2 (* 2 key-dim) conv-dim) [b T Hv Dv])
        ;; --- q/k RMSNorm (no weight) with inv-scale (matches Python/Rust exactly) ---
        inv (js/Math.pow Dk -0.5)
        ones-dk (or (:ones-dk derived) (mx/ones [Dk]))
        q (mx/multiply (mx/rms-norm q ones-dk 1e-6) (* inv inv))
        k (mx/multiply (mx/rms-norm k ones-dk 1e-6) inv)
        ;; --- gates: beta = sigmoid(b); g = exp(-exp(A_log) * softplus(a + dt_bias)) ---
        ;; The SSM recurrence + gates run in float32 (config mamba_ssm_dtype:
        ;; "float32") to match the upstream fused kernel's f32 state accumulation;
        ;; projections/conv stay in the model dtype, and y is cast back below so
        ;; the residual stream remains bf16. -exp(A_log) and f32 dt_bias are
        ;; load-time derived constants (genmlx-t2cz L4; per-call fallback for
        ;; models loaded without :derived, e.g. hand-built test models).
        beta   (mx/astype (mx/sigmoid bb) mx/float32)  ; [B T Hv]
        neg-a  (or (:neg-a der)
                   (mx/multiply (mx/exp (mx/astype (g "A_log") mx/float32)) -1.0)) ; [Hv]
        dt-bias (or (:dt-f32 der) (mx/astype (g "dt_bias") mx/float32))            ; [Hv]
        ;; Log-space decay gate, computed DIRECTLY — never (mx/log gg): strong
        ;; decay underflows exp-space gg to 0 and log(0) = -inf would NaN the
        ;; fused scan's in-chunk decay-diff (genmlx-ps8a).
        g-log  (mx/multiply neg-a
                            (softplus (mx/add (mx/astype aa mx/float32) dt-bias)))  ; [B T Hv]
        ;; --- GQA: repeat q,k from Hk to Hv heads (np.repeat along head axis) ---
        rep (quot Hv Hk)
        q   (mx/astype (if (> rep 1) (mx/repeat-arr q rep 2) q) mx/float32)
        k   (mx/astype (if (> rep 1) (mx/repeat-arr k rep 2) k) mx/float32)
        v   (mx/astype v mx/float32)
        ;; --- gated delta recurrence, f32 accumulation ---
        init-state (if ce (:rec ce) (mx/zeros [b Hv Dv Dk] mx/float32))
        [y-f32 final-state]
        (if (> T 1)
          ;; Multi-token block (prefill chunk): ONE fused chunk-parallel scan
          ;; (BT=64 WY form) instead of T host-loop iterations — the 45→17.6
          ;; ms/token lever (genmlx-ps8a). Same math as gdn-recur-steps
          ;; (parity pinned by gdn_scan_contract_test).
          (mx/gated-delta-scan q k v g-log beta init-state)
          ;; T=1 (decode): ONE fused per-step membrane call (genmlx-t2cz) —
          ;; same math as (gdn-recur-steps q k v (mx/exp g-log) beta st),
          ;; parity pinned by gdn_scan_contract_test.
          (mx/gated-delta-step q k v g-log beta init-state))
        y (mx/astype y-f32 dtype)    ; [B T Hv Dv], back to model dtype
        ;; --- gated RMSNorm: silu(z) * rms_norm(y, norm_weight[Dv], eps) ---
        z4 (mx/reshape z [b T Hv Dv])
        y-norm (mx/multiply (mx/silu z4) (mx/rms-norm y (g "norm.weight") eps))
        y-flat (mx/reshape y-norm [b T val-dim])]
    [(linear y-flat (g "out_proj.weight")) {:conv new-conv :rec final-state}]))

;; ----------------------------------------------------------------------------
;; Interleaved M-RoPE (VLM prefill — genmlx-w3og)
;; ----------------------------------------------------------------------------

(defn mrope-tables
  "Interleaved M-RoPE cos/sin tables [T rope-dims] from 3-axis position ids
   [[t…] [h…] [w…]] (host vectors). Mirrors MultimodalRoPE +
   apply_multimodal_rotary_pos_emb_interleaved's stride-3 per-frequency axis
   selector (mrope_section [th tw]-style, e.g. Ornith [11 11 10]): frequency
   slot j in the half-dim table takes axis h at j ≡ 1 (mod 3) below 3·sec_h,
   axis w at j ≡ 2 (mod 3) below 3·sec_w, else axis t; the doubled cos/sin
   repeats the selector (emb = concat[freqs freqs])."
  [pos-ids sec rope-dims theta]
  (let [half (quot rope-dims 2)
        T    (count (first pos-ids))
        inv  (mx/array (vec (for [i (range half)]
                              (/ 1.0 (js/Math.pow theta (/ (* 2 i) rope-dims)))))
                       [1 1 half] mx/float32)
        pos  (mx/astype (mx/array (vec (apply concat pos-ids)) [3 T 1] mx/int32)
                        mx/float32)
        fr   (mx/multiply pos inv)                    ; [3 T half]
        emb  (mx/concatenate [fr fr] 2)               ; [3 T rope-dims]
        sel-half (reduce (fn [s [dim limit off]]
                           (reduce #(assoc %1 %2 dim) s (range off (min limit half) 3)))
                         (vec (repeat half 0))
                         [[1 (* 3 (nth sec 1)) 1] [2 (* 3 (nth sec 2)) 2]])
        sel  (vec (map #(nth sel-half (mod % half)) (range rope-dims)))
        idx  (mx/astype (mx/broadcast-to (mx/array sel [1 1 rope-dims] mx/int32)
                                         [1 T rope-dims])
                        mx/int32)
        gsel (fn [a] (mx/reshape (mx/take-along-axis a idx 0) [1 1 T rope-dims]))]
    {:cos (gsel (mx/cos emb)) :sin (gsel (mx/sin emb)) :dims rope-dims}))

(defn- mrope-apply
  "Rotate the first rope-dims of x [1 H T head-dim] with the per-position
   cos/sin [1 1 T rope-dims]; pass the remaining dims through untouched."
  [x {:keys [cos sin dims]} head-dim]
  (let [xr   (slice-ax x 4 3 0 dims)
        xp   (slice-ax x 4 3 dims head-dim)
        h    (quot dims 2)
        x1   (slice-ax xr 4 3 0 h)
        x2   (slice-ax xr 4 3 h dims)
        rh   (mx/concatenate [(mx/multiply x2 -1.0) x1] 3)
        rot  (mx/astype (mx/add (mx/multiply xr cos) (mx/multiply rh sin))
                        (mx/dtype x))]
    (mx/concatenate [rot xp] 3)))

;; ----------------------------------------------------------------------------
;; Full softmax attention layer (partial RoPE + output gate)
;; ----------------------------------------------------------------------------

(defn- full-attn-layer
  "Qwen3.5 full attention over `hn` [B T hidden] (B read from the shape —
   the [K]-particle batch axis, genmlx-9uyg). q_proj is 2x width: first
   head-dim = query, second = output gate (sigmoid). PARTIAL RoPE over the first
   rope_dims of head_dim. `ce` = prior {:k :v} (post-RoPE, [B H T hd]) cache or
   nil; `offset` is the absolute position of the first new token (lockstep
   across lanes). Returns [out new-cache-entry].

   Mirrors attention.rs::Qwen3_5Attention::forward. `b` is threaded from
   forward-hidden (one shape query per forward, genmlx-t2cz)."
  [cfg w prefix hn b T ce offset mask mrope]
  (let [{:keys [n-heads n-kv-heads head-dim eps rope-theta partial]} cfg
        H   n-heads
        Hkv n-kv-heads
        hd  head-dim
        rope-dims (js/Math.floor (* hd partial))
        scale (js/Math.pow hd -0.5)
        g   (fn [s] (get w (str prefix s)))
        qg  (mx/reshape (linear hn (g "q_proj.weight")) [b T H (* 2 hd)])
        queries (slice-ax qg 4 3 0 hd)                          ; [B T H hd]
        gate    (mx/reshape (slice-ax qg 4 3 hd (* 2 hd)) [b T (* H hd)])
        keys    (mx/reshape (linear hn (g "k_proj.weight")) [b T Hkv hd])
        values  (mx/reshape (linear hn (g "v_proj.weight")) [b T Hkv hd])
        queries (mx/rms-norm queries (g "q_norm.weight") eps)
        keys    (mx/rms-norm keys    (g "k_norm.weight") eps)
        ;; transpose to [1 heads T hd], then rotate: 3-axis interleaved M-RoPE
        ;; over the partial dims (VLM prefill) or scalar-offset partial RoPE.
        rot (if mrope
              (fn [x] (mrope-apply x mrope hd))
              (fn [x] (mx/rope x rope-dims false rope-theta 1.0 offset)))
        q (rot (mx/transpose queries [0 2 1 3]))
        k (rot (mx/transpose keys    [0 2 1 3]))
        v (mx/transpose values [0 2 1 3])
        k-full (if ce (mx/concatenate [(:k ce) k] 2) k)        ; concat along seq
        v-full (if ce (mx/concatenate [(:v ce) v] 2) v)
        o (-> (mx/scaled-dot-product-attention q k-full v-full scale mask)
              (mx/transpose [0 2 1 3])                          ; [B T H hd]
              (mx/reshape [b T (* H hd)]))
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

(defn- forward-hidden
  "Layer loop over pre-built hidden states h0 [B T hidden] (B read from the
   shape — the [K]-particle batch axis, genmlx-9uyg): rotate with the 3-axis
   `mrope` tables when given (VLM prefill, B=1), else scalar RoPE at
   `offset`. `prior` = physical tokens already in the cache (mask width for a
   chunked prefill block; 0 for a from-scratch prefill). Returns
   [logits [B T vocab] new-cache]."
  [{:keys [config weights]} h0 T cache offset mrope prior]
  (let [{:keys [n-layers eps vocab layer-types]} config
        embed (get weights (str wp "embed_tokens.weight"))
        dtype (mx/dtype embed)
        b     (first (mx/shape h0))
        mask  (when (> T 1) (causal-mask T (or prior 0) dtype))
        ;; cfg :layer-tap — (fn [layer-idx kind label lazy-array]) called with
        ;; each sublayer's lazy output (:attn-out = attention/GDN branch,
        ;; :mlp-out = dense/MoE MLP branch, :final-norm). The sibling of
        ;; :moe-tap (genmlx-4wsn): caller-owned collection state, nil in
        ;; production, zero cost. Added for the genmlx-mdet determinism
        ;; bisect (bench/moe_determinism_bisect.cljs).
        tap   (:layer-tap config)
        [h new-cache]
        (reduce
         (fn [[h nc] i]
           (let [p  (str wp "layers." i ".")
                 kind (nth layer-types i)
                 hn (mx/rms-norm h (get weights (str p "input_layernorm.weight")) eps)
                 [a ce'] (if (= "linear_attention" kind)
                           (gdn-layer config weights (str p "linear_attn.") hn b T (nth cache i))
                           (full-attn-layer config weights (str p "self_attn.") hn b T
                                            (nth cache i) offset mask mrope))
                 _  (when tap (tap i kind :attn-out a))
                 h1 (mx/add h a)
                 mn (mx/rms-norm h1 (get weights (str p "post_attention_layernorm.weight")) eps)
                 m  (if (moe-layer? config i)
                      (moe-mlp config weights (str p "mlp.") mn b T)
                      (mlp weights (str p "mlp.") mn))
                 _  (when tap (tap i kind :mlp-out m))]
             [(mx/add h1 m) (conj nc ce')]))
         [h0 []] (range n-layers))
        hf (mx/rms-norm h (get weights (str wp "norm.weight")) eps)
        _  (when tap (tap -1 nil :final-norm hf))
        ;; tied: logits = h @ embed^T; untied (Ornith): a real lm_head tensor,
        ;; dequantized at load, OUTSIDE the `language_model.model.` prefix.
        lmw (if (:tie? config)
              embed
              (get weights "language_model.lm_head.weight"))]
    [(linear hf lmw) new-cache]))

(defn forward-cached
  "Run the forward over `token-ids` from absolute position `offset`, threading
   and extending the per-layer `cache`. Returns [logits new-cache] with logits
   [seq vocab]. A causal mask is built for multi-token chunks (prefill)."
  [{:keys [config weights] :as model} token-ids cache offset]
  (let [{:keys [hidden vocab]} config
        T     (count token-ids)
        ids   (mx/array (vec token-ids) [T] mx/int32)
        embed (get weights (str wp "embed_tokens.weight"))
        h0    (mx/reshape (mx/take-idx embed ids 0) [1 T hidden])
        [logits new-cache] (forward-hidden model h0 T cache offset nil offset)]
    [(mx/reshape logits [T vocab]) new-cache]))

(defn step-batched
  "Advance K lockstep lanes one token each from a [K …]-shaped `cache`
   (current length = `offset`, shared across lanes — the
   no-per-lane-early-stop contract, genmlx-9uyg). `tok` is a [K]-shaped int
   array of per-lane token ids. Returns [logits cache'] with logits [K vocab]."
  [{:keys [config weights] :as model} cache offset tok]
  (let [{:keys [hidden vocab]} config
        K     (first (mx/shape tok))
        embed (get weights (str wp "embed_tokens.weight"))
        h0    (mx/reshape (mx/take-idx embed (mx/astype tok mx/int32) 0)
                          [K 1 hidden])
        [logits new-cache] (forward-hidden model h0 1 cache offset nil offset)]
    [(mx/reshape logits [K vocab]) new-cache]))

(defn prefill-batched
  "Process K equal-length prompts in ONE [B T] forward over a fresh cache
   (batch-independence entry, genmlx-9uyg — the particle path itself prefills
   at B=1 and tiles). Returns [last-logits cache] with last-logits [B vocab]."
  [{:keys [config weights] :as model} prompts]
  (let [{:keys [hidden vocab]} config
        B (count prompts)
        T (count (first prompts))]
    (when-not (every? #(= T (count %)) prompts)
      (throw (ex-info "prefill-batched: prompts must be equal token length"
                      {:lengths (mapv count prompts)})))
    (let [ids   (mx/array (vec (apply concat prompts)) [B T] mx/int32)
          embed (get weights (str wp "embed_tokens.weight"))
          h0    (mx/take-idx embed ids 0)                        ; [B T hidden]
          [logits cache] (forward-hidden model h0 T (init-cache model) 0 nil 0)]
      [(mx/reshape (slice-ax logits 3 1 (dec T) T) [B vocab]) cache])))

(defn forward-embeds
  "VLM-prefill entry (genmlx-w3og): run the forward over pre-merged input
   embeddings h0 [1 T hidden] with 3-axis M-RoPE `position-ids`
   [[t…] [h…] [w…]] (host vectors, from the vision merge). Keys land in the
   cache at COMPRESSED M-RoPE positions; every subsequent forward-cached step
   must therefore rotate at `physical-offset + rope-delta` (the caller owns
   the delta — cf. the native genmlx-52mh continuation fix). `prior` =
   physical tokens already in `cache` (0 for the first chunk).
   Returns [logits new-cache]."
  [{:keys [config] :as model} h0 cache position-ids prior]
  (let [{:keys [head-dim partial rope-theta mrope-section vocab]} config
        T     (count (first position-ids))
        dims  (js/Math.floor (* head-dim partial))
        mrope (mrope-tables position-ids (or mrope-section [11 11 10]) dims rope-theta)
        [logits new-cache] (forward-hidden model h0 T cache 0 mrope prior)]
    [(mx/reshape logits [T vocab]) new-cache]))

(defn materialize-cache!
  "Force-evaluate every array in a per-layer cache (the {:k :v} / {:conv :rec}
   entries) — the chunked-prefill eval boundary. History: with the per-token
   host-loop GDN recurrence this was load-bearing (each chunk's live graph
   held ~2 MB × T × layers of f32 states; an unchunked ~630-token VLM prefill
   OOMed at >105 GB, genmlx-w3og). The fused GDN scan (genmlx-ps8a) removed
   that pressure — T=624 unchunked now holds >120 GB free — so this boundary
   is a cheap memory backstop, not a correctness requirement."
  [cache]
  (apply mx/materialize! (mapcat vals (remove nil? cache)))
  cache)

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

(defn prefill-chunked
  "Chunked prefill: process the prompt in `chunk`-token blocks through
   forward-cached (offset + prior-width mask make continuation exact), with
   materialize-cache! + mx/jsc-cleanup! at each block boundary. Same
   [last-logits cache] contract as prefill; mathematically identical —
   chunking is a memory boundary, not an approximation — but bf16
   accumulation order differs across block boundaries, so logits match the
   slab within cross-kernel tolerance (1-2 ULP; argmax + top-5 ids exact on
   the 0.8B probe, genmlx-nwsr), the same bar as the forward's own golden
   gate. Bounds the single-slab prefill
   transient (~10-18 GB at T2-prompt sizes on the 8-bit 35B-A3B) to one
   block's worth: without the boundary the whole-prompt graph evals as ONE
   slab and its dead intermediate wrappers survive until the sync caller
   yields (the same finalizer starvation as the decode loop, genmlx-12w4 /
   genmlx-nwsr). chunk <= 0 or >= prompt length degrades to the single-slab
   prefill."
  [model prompt-ids chunk]
  (let [ids (vec prompt-ids)
        n   (count ids)]
    (if (or (nil? chunk) (<= chunk 0) (>= chunk n))
      (prefill model ids)
      (loop [off 0, cache (init-cache model), last-logits nil]
        (if (>= off n)
          [last-logits cache]
          (let [block (subvec ids off (min n (+ off chunk)))
                [lg cache'] (forward-cached model block cache off)
                lg-last (mx/index lg (dec (count block)))]
            (materialize-cache! cache')
            (mx/materialize! lg-last)
            (mx/jsc-cleanup!)
            (recur (+ off (count block)) cache' lg-last)))))))

(defn step
  "Advance one token from `cache` (current length = `offset`). Returns
   [logits cache'] with logits [vocab]."
  [model cache offset token-id]
  (let [[logits cache'] (forward-cached model [token-id] cache offset)]
    [(mx/index logits 0) cache']))
