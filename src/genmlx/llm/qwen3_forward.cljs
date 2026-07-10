(ns genmlx.llm.qwen3-forward
  "f6ov P2: a GenMLX-owned Qwen3 (standard transformer) forward pass in pure
   ClojureScript, composed over the genmlx.rs fast:: primitives (mx/rms-norm,
   mx/rope, mx/scaled-dot-product-attention, mx/silu) and GenMLX-owned weight
   loading (mx/load-safetensors). Decoupled from upstream's per-model forward
   structs — the LLM forward becomes a value-level CLJS description.

   Proves the f6ov methodology by matching upstream's forward to parity, gated by
   the golden oracle. Qwen3.5's hybrid Mamba layers are a follow-up; the scaffold
   here (config/weights/layer-loop/parity) is shared.

   Weight layout: HF Linear weights are [out, in] (y = x W^T). Embeddings are
   tied (lm_head = embed_tokens^T). qk-norm (RMSNorm over head_dim) is applied
   after the head reshape, before RoPE — matching mlx-lm's Qwen3 attention."
  (:require [genmlx.mlx :as mx]
            ["fs" :as fs]))

(defn load-config [dir]
  (let [c (js/JSON.parse (.readFileSync fs (str dir "/config.json") "utf8"))]
    {:hidden       (.-hidden_size c)
     :n-layers     (.-num_hidden_layers c)
     :n-heads      (.-num_attention_heads c)
     :n-kv-heads   (.-num_key_value_heads c)
     :head-dim     (.-head_dim c)
     :intermediate (.-intermediate_size c)
     :vocab        (.-vocab_size c)
     :eps          (.-rms_norm_eps c)
     :rope-theta   (.-rope_theta c)
     :tie?         (.-tie_word_embeddings c)}))

;; ----------------------------------------------------------------------------
;; MLX affine-quantized checkpoints — dequantize at load (genmlx-9iqc/-vmks)
;;
;; Quantized checkpoints store each Linear/Embedding weight as THREE tensors:
;; `X.weight` U32 [out, in/(32/bits)] (values packed LSB-first per word —
;; mlx ops.cpp: out_el |= w_el << (k*bits)) plus `X.scales`/`X.biases`
;; [out, in/group_size]. Reading the packed U32 raw is what produced the
;; "Cannot reshape array of size T*(hidden/8) into (1,T,hidden)" abort. The
;; owned forward is written over full-precision tensors, so we dequantize the
;; whole map once at load (w = scale*q + bias with groups along the input
;; axis — mlx backend/cpu/quantized.cpp). Shared by the qwen3 and qwen3.5
;; family loaders.
;; ----------------------------------------------------------------------------

(defn load-quantization
  "Parse config.json's `quantization` block, or nil for a full-precision
   checkpoint. Per-tensor override sub-objects (mixed quantization — e.g.
   Ornith's 3-bit checkpoint keeps its routers at 8-bit) are parsed into
   :overrides {tensor-base-name {:bits b :group-size g}}; dequantize-weights
   resolves each tensor against them, falling back to the global scheme."
  [dir]
  (let [c (js/JSON.parse (.readFileSync fs (str dir "/config.json") "utf8"))
        q (.-quantization c)]
    (when q
      {:bits       (.-bits q)
       :group-size (.-group_size q)
       :mode       (or (.-mode q) "affine")
       :overrides  (reduce (fn [m k]
                             (let [v (unchecked-get q k)]
                               (if (object? v)
                                 (assoc m k {:bits       (.-bits v)
                                             :group-size (.-group_size v)})
                                 m)))
                           {} (js-keys q))})))

(defn dequantizable?
  "True if dequantize-weights implements this quantization: MLX `affine` mode
   with bits in {2,3,4,5,6,8} for the global scheme AND every per-tensor
   override. Power-of-two widths unpack via pure u32 arithmetic; the odd
   widths (3/5/6 — values straddle u32 words) go through the native
   mx/dequantize kernel (genmlx-q5uq/-dlvi). Mixed checkpoints are fine —
   group size is inferred per-tensor from the scales shape regardless.
   mxfp4/nvfp4/mxfp8 schemes must use the upstream forward."
  [{:keys [bits mode overrides]}]
  (let [ok #{2 3 4 5 6 8}]
    (and (= mode "affine")
         (contains? ok bits)
         (every? #(contains? ok (:bits %)) (vals overrides)))))

(defn dequantize-weights
  "Replace every packed-quantized weight in a load-safetensors map with its
   dequantized full-precision tensor (in the scales' dtype, e.g. bf16), and
   drop the folded-in `.scales`/`.biases` entries. Non-quantized tensors
   (norms, conv1d, A_log, dt_bias, ...) pass through untouched. A tensor is
   quantized iff its `.scales` sibling exists.

   Power-of-two widths unpack via pure uint32 arithmetic (floor-divide/
   remainder by 2^bits — exact for the full u32 range; the membrane has no
   bitwise ops), values identical to MLX's own dequantize (pinned by
   gather_qmm_oracle_test Part 1). Odd widths (3/5/6) straddle u32 words and
   go through the native mx/dequantize kernel instead (genmlx-dlvi). Each
   tensor's bit width comes from its :overrides entry when present, else the
   global :bits (group size is inferred from the scales shape either way).
   Each tensor is materialized here — load is an I/O boundary — so neither
   the packed sources nor the unpack graphs are retained.

   opts {:skip? (fn [base-name] ...)}: tensors whose base name matches stay
   PACKED — weight, .scales and .biases all pass through untouched. This is
   how the owned MoE forward keeps its 32B expert params packed for
   mx/gather-qmm instead of tripling them to bf16 (genmlx-g6vk, spec §5.3).

   Throws (naming the quantization) for schemes dequantizable? rejects."
  ([weights qz] (dequantize-weights weights qz nil))
  ([weights {:keys [bits overrides] :as qz} {:keys [skip?]}]
  (when-not (dequantizable? qz)
    (throw (ex-info (str "dequantize-weights: unsupported quantization " (pr-str qz)
                         " — the owned CLJS forward implements affine "
                         "bits 2/3/4/5/6/8 only (global and per-tensor "
                         "overrides). Load with {:cljs-forward? false} to "
                         "use the upstream forward.")
                    {:quantization qz})))
  (let [dequant
        (fn [nm base wq scales biases]
          (let [t-bits  (get-in overrides [base :bits] bits)
                p2?     (contains? #{2 4 8} t-bits)
                [out gcount] (mx/shape scales)
                in (if p2?
                     (* (quot 32 t-bits) (second (mx/shape wq)))
                     (quot (* 32 (second (mx/shape wq))) t-bits))
                gs (quot in gcount)]
            (when-not (= in (* gs gcount))
              (throw (ex-info (str "dequantize-weights: " nm " scales shape "
                                   (mx/shape scales) " does not tile its weight "
                                   (mx/shape wq) " at bits=" t-bits)
                              {:tensor nm :quantization qz})))
            (let [w (if p2?
                      ;; pure unpack: value k of each u32 word is original
                      ;; column j*pf+k (LSB-first)
                      (let [pf      (quot 32 t-bits)
                            divisor (mx/array [(js/Math.pow 2 t-bits)] [] mx/uint32)
                            parts (loop [cur wq k 0 acc []]
                                    (if (= k pf)
                                      acc
                                      (recur (mx/floor-divide cur divisor) (inc k)
                                             (conj acc (mx/remainder cur divisor)))))
                            q  (-> (mx/stack parts 2)   ; [out in/pf pf]
                                   (mx/reshape [out in])
                                   (mx/astype mx/float32))
                            s3 (mx/reshape (mx/astype scales mx/float32) [out gcount 1])
                            b3 (mx/reshape (mx/astype biases mx/float32) [out gcount 1])]
                        (-> (mx/reshape q [out gcount gs])
                            (mx/multiply s3)
                            (mx/add b3)
                            (mx/reshape [out in])
                            (mx/astype (mx/dtype scales))))
                      ;; odd widths: the native kernel owns the cross-word
                      ;; bit extraction
                      (mx/dequantize wq scales biases
                                     {:bits t-bits :group-size gs}))]
              (mx/materialize! w)
              w)))]
    (reduce-kv
     (fn [m k v]
       (let [base    (when (.endsWith k ".weight")
                       (subs k 0 (- (count k) (count ".weight"))))
             sb      (when (or (.endsWith k ".scales") (.endsWith k ".biases"))
                       (subs k 0 (- (count k) (count ".scales")))) ; same length
             skip-fn (or skip? (constantly false))]
         (cond
           ;; packed-by-request: weight AND its scales/biases pass through
           (and base (skip-fn base))  (assoc m k v)
           (and sb (skip-fn sb))      (assoc m k v)
           ;; folded into the dequantized weight below
           (or (.endsWith k ".scales") (.endsWith k ".biases")) m
           (and base (contains? weights (str base ".scales")))
           (assoc m k (dequant k base v (get weights (str base ".scales"))
                              (get weights (str base ".biases"))))
           :else (assoc m k v))))
     {} weights))))

(defn weight-files
  "Resolve a checkpoint's weight file paths: [model.safetensors] when the
   single-file layout exists, else the distinct shard files named by
   model.safetensors.index.json's weight_map (the HF sharded layout,
   genmlx-sbif). Single-file wins when both exist (some single-file
   checkpoints ship a redundant index). Throws when neither is present."
  [dir]
  (let [single (str dir "/model.safetensors")
        index  (str dir "/model.safetensors.index.json")]
    (cond
      (.existsSync fs single) [single]
      (.existsSync fs index)
      (let [wm (.-weight_map (js/JSON.parse (.readFileSync fs index "utf8")))]
        (->> (js-keys wm)
             (map #(unchecked-get wm %))
             distinct
             sort
             (mapv #(str dir "/" %))))
      :else
      (throw (ex-info (str "load-weights: neither model.safetensors nor "
                           "model.safetensors.index.json exists in " dir)
                      {:dir dir})))))

(defn load-weights
  "Load a checkpoint's weights as {name -> MxArray} — a single
   model.safetensors or all shards of an HF index.json layout, merged
   (see weight-files) — dequantizing at load when config.json declares a
   quantization (see dequantize-weights; opts {:skip?} selects tensors
   that stay packed)."
  ([dir] (load-weights dir nil))
  ([dir opts]
   (let [w  (reduce (fn [m f] (into m (mx/load-safetensors f))) {} (weight-files dir))
         qz (load-quantization dir)]
     (if qz (dequantize-weights w qz opts) w))))

(defn load-model
  "Load a Qwen3 checkpoint as {:config .. :weights {name -> MxArray}}."
  [dir]
  {:config (load-config dir)
   :weights (load-weights dir)})

(defn- causal-mask
  "Additive causal mask [seq seq]: 0 on/below the diagonal, large-negative above."
  [seq dtype]
  (let [flat (vec (for [i (range seq) j (range seq)] (if (<= j i) 0.0 -1e9)))]
    (mx/astype (mx/array flat [seq seq]) dtype)))

(defn- linear
  "HF Linear with no bias: y = x W^T (W is [out, in])."
  [x w]
  (mx/matmul x (mx/transpose w)))

(defn- attention
  "Self-attention for one layer over the pre-attention-normed hidden `hn`
   [seq hidden]. GQA via the fast SDPA (q has n-heads, k/v have n-kv-heads).
   `ce` is the prior {:k :v} cache entry (or nil); `offset` is the absolute
   position of the first new token (for RoPE + correct causal scope). Returns
   [attn-out new-cache-entry] — the new entry holds the post-RoPE k/v of the full
   context, so it threads straight into the next step."
  [{:keys [n-heads n-kv-heads head-dim eps rope-theta]} w prefix hn seq ce offset mask]
  (let [g     (fn [s] (get w (str prefix s)))
        scale (/ 1.0 (js/Math.sqrt head-dim))
        proj-heads (fn [name nh norm?]
                     (let [t (-> (linear hn (g (str name "_proj.weight")))
                                 (mx/reshape [1 seq nh head-dim])
                                 (mx/transpose [0 2 1 3]))]      ; [1 nh seq d]
                       (if norm?
                         (-> t (mx/rms-norm (g (str name "_norm.weight")) eps)
                               (mx/rope head-dim false rope-theta 1.0 offset))
                         t)))
        q (proj-heads "q" n-heads true)
        k (proj-heads "k" n-kv-heads true)
        v (proj-heads "v" n-kv-heads false)
        k-full (if ce (mx/concatenate [(:k ce) k] 2) k)          ; concat along seq axis
        v-full (if ce (mx/concatenate [(:v ce) v] 2) v)
        o (-> (mx/scaled-dot-product-attention q k-full v-full scale mask)
              (mx/transpose [0 2 1 3])                            ; [1 seq nh d]
              (mx/reshape [seq (* n-heads head-dim)]))]
    [(linear o (g "o_proj.weight")) {:k k-full :v v-full}]))

(defn- mlp
  "SwiGLU MLP over the post-attention-normed hidden `hn`: down(silu(gate)·up)."
  [w prefix hn]
  (let [g (fn [s] (get w (str prefix s)))]
    (linear (mx/multiply (mx/silu (linear hn (g "gate_proj.weight")))
                         (linear hn (g "up_proj.weight")))
            (g "down_proj.weight"))))

(defn init-cache
  "An empty per-layer KV cache (vector of nils, one slot per layer)."
  [{:keys [config]}]
  (vec (repeat (:n-layers config) nil)))

(defn forward-cached
  "Run the forward over `token-ids` starting at absolute position `offset`, using
   and extending the per-layer KV `cache` (vector of {:k :v} or nils). A causal
   mask is applied only for multi-token chunks (prefill); a single-token step
   attends to the whole cache unmasked. Returns [logits new-cache] where logits
   is [seq vocab]."
  [{:keys [config weights]} token-ids cache offset]
  (let [{:keys [n-layers eps]} config
        seq   (count token-ids)
        ids   (mx/array (vec token-ids) [seq] mx/int32)
        embed (get weights "model.embed_tokens.weight")
        dtype (mx/dtype embed)
        mask  (when (> seq 1) (causal-mask seq dtype))
        h0    (mx/take-idx embed ids 0)                          ; [seq hidden]
        [h new-cache]
        (reduce
         (fn [[h nc] layer]
           (let [p  (str "model.layers." layer ".")
                 hn (mx/rms-norm h (get weights (str p "input_layernorm.weight")) eps)
                 [a ce] (attention config weights (str p "self_attn.") hn seq
                                   (nth cache layer) offset mask)
                 h1 (mx/add h a)
                 mn (mx/rms-norm h1 (get weights (str p "post_attention_layernorm.weight")) eps)]
             [(mx/add h1 (mlp weights (str p "mlp.") mn)) (conj nc ce)]))
         [h0 []] (range n-layers))
        hf (mx/rms-norm h (get weights "model.norm.weight") eps)]
    [(linear hf embed) new-cache]))                              ; tied lm_head

(defn forward
  "Uncached full forward over a token-id sequence (offset 0, fresh cache).
   Returns logits [seq vocab]; take row (dec seq) for the next-token distribution."
  [model token-ids]
  (first (forward-cached model token-ids (init-cache model) 0)))

(defn next-token-logits
  "Last-position logits [vocab] for the next token given a prompt token-id seq."
  [model token-ids]
  (mx/index (forward model token-ids) (dec (count token-ids))))

(defn prefill
  "Process the full prompt, populating the KV cache. Returns [last-logits cache]
   where last-logits is [vocab] for the next token after the prompt."
  [model prompt-ids]
  (let [[logits cache] (forward-cached model (vec prompt-ids) (init-cache model) 0)]
    [(mx/index logits (dec (count prompt-ids))) cache]))

(defn step
  "Advance one token from `cache` (current length = `offset`). Returns
   [logits cache'] where logits is [vocab]. Constant work in sequence length."
  [model cache offset token-id]
  (let [[logits cache'] (forward-cached model [token-id] cache offset)]
    [(mx/index logits 0) cache']))
