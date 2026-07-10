(ns genmlx.llm.qwen35-vision-forward
  "genmlx-w3og (Ornith Phase 4): the GenMLX-owned qwen3.5-VL VISION tower in
   pure ClojureScript — patch embed, interpolated position embeddings, 2D
   rotary attention blocks under a block-diagonal image mask, and the 2x2
   spatial-merge projector. Consumes natively-preprocessed pixel patches
   (mx/vlm-preprocess — image decode/resize is I/O, the vision analogue of
   tokenization) and produces the vision features `[merged, out-hidden]` that
   the decoder scatters into `<|image_pad|>` slots.

   Authoritative reference = mlx-node crates/mlx-core/src/vision/* +
   models/qwen3_5/vision.rs (the exact op sequence the native encoder runs;
   parity gate llm_qwen35_vision_parity_test vs the native vlmVisionFeatures
   debug tap). Decisions mirrored from the Rust, all verified against source:
   - patch_embed.proj.weight is Conv3d-shaped [D, temporal, ph, pw, C]; the
     image processor duplicates the static frame across temporal, so the
     effective kernel is the SUM over the temporal axis, and (kernel==stride)
     conv2d over 16x16 patches is exactly a per-patch linear on the
     [ph, pw, C]-flattened pixels (persistence.rs collapse + spec 2.1d).
   - Learned position embeddings [2304=48x48, D] bilinear-interpolate
     (align_corners=false, clamped) to each image's [h, w] grid.
   - 2D RoPE: freqs dim head_dim/2 = 36, theta 10000, per-patch (row, col)
     ids gathered and concatenated -> [N, 36], cos/sin tiled x2 -> 72;
     x*cos + rotate_half(x)*sin, cast back to input dtype.
   - Blocks are pre-norm LayerNorm (weight AND bias, eps 1e-6), fused qkv
     with bias, SDPA at 72^-0.5 under a block-diagonal cu_seqlens mask (nil
     for a single image), gelu-tanh MLP (fc1/fc2 with bias).
   - Merger: LayerNorm -> [t, h/m, m, w/m, m, D] -> transpose (0 1 3 2 4 5)
     -> [t*(h/m)*(w/m), m*m*D] -> fc1 -> gelu-tanh -> fc2.

   LayerNorm/gelu are COMPOSED from membrane ops (mean/variance/sqrt/tanh) —
   the membrane stays thin (docs/membrane-coverage.md); float dtypes follow
   MLX promotion exactly as the native ops do (f32 pixels x bf16 weights)."
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.qwen35-forward :as q35]
            ["fs" :as fs]))

(defn load-vision-config
  "Parse config.json's vision_config (nil when the checkpoint has none)."
  [dir]
  (let [c  (js/JSON.parse (.readFileSync fs (str dir "/config.json") "utf8"))
        vc (.-vision_config c)]
    (when vc
      {:depth        (.-depth vc)
       :hidden       (.-hidden_size vc)
       :intermediate (.-intermediate_size vc)
       :heads        (.-num_heads vc)
       :patch        (.-patch_size vc)
       :temporal     (.-temporal_patch_size vc)
       :merge        (.-spatial_merge_size vc)
       :out-hidden   (.-out_hidden_size vc)
       :num-pos      (.-num_position_embeddings vc)
       :eps          1e-6})))

(def ^:private wp "vision_tower.")

(defn- lin
  "Linear with optional bias: y = x W^T (+ b)."
  [x w b]
  (let [y (mx/matmul x (mx/transpose w))]
    (if b (mx/add y b) y)))

(defn- layer-norm
  "LayerNorm over the last axis with weight AND bias, composed from membrane
   ops. Mirrors fast::layer_norm's precision: stats AND affine in f32, one
   cast back to the input dtype at the end."
  [x w b eps]
  (let [xf (mx/astype x mx/float32)
        mu (mx/mean xf [-1] true)
        v  (mx/variance xf [-1] true)
        n  (mx/divide (mx/subtract xf mu) (mx/sqrt (mx/add v eps)))]
    (mx/astype (mx/add (mx/multiply n (mx/astype w mx/float32))
                       (mx/astype b mx/float32))
               (mx/dtype x))))

(defn- gelu-tanh
  "gelu_pytorch_tanh: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))."
  [x]
  (let [c (js/Math.sqrt (/ 2 js/Math.PI))
        u (mx/multiply (mx/add x (mx/multiply (mx/multiply x (mx/multiply x x)) 0.044715)) c)]
    (mx/multiply (mx/multiply x 0.5) (mx/add (mx/tanh u) 1.0))))

(defn- rotate-half [x d]
  (let [h  (quot d 2)
        x1 (mx/take-idx x (mx/arange 0 h) 3)
        x2 (mx/take-idx x (mx/arange h d) 3)]
    (mx/concatenate [(mx/multiply x2 -1.0) x1] 3)))

(defn- apply-rope
  "x [1 N H D]; cos/sin [1 N 1 D]. x*cos + rotate_half(x)*sin, cast back."
  [x cos sin d]
  (mx/astype (mx/add (mx/multiply x cos) (mx/multiply (rotate-half x d) sin))
             (mx/dtype x)))

(defn- rot-freqs
  "2D rotary frequency table [N, head-dim/2] for the concatenated grids:
   per patch (row-id, col-id) gathered from freqs [max-size, head-dim/4]."
  [grids head-dim]
  (let [half (quot head-dim 2)
        quarter (quot half 2)
        theta 10000.0
        inv  (mx/array (vec (for [i (range quarter)]
                              (/ 1.0 (js/Math.pow theta (/ (* 2 i) half)))))
                       [1 quarter] mx/float32)
        max-size (reduce max 1 (mapcat (fn [[_ h w]] [h w]) grids))
        seq  (mx/reshape (mx/astype (mx/arange 0 max-size) mx/float32) [max-size 1])
        freqs (mx/multiply seq inv)                       ; [max, quarter]
        h-ids (vec (mapcat (fn [[t h w]]
                             (apply concat (repeat t (for [i (range (* h w))] (quot i w)))))
                           grids))
        w-ids (vec (mapcat (fn [[t h w]]
                             (apply concat (repeat t (for [i (range (* h w))] (mod i w)))))
                           grids))
        n    (count h-ids)
        hf   (mx/take-idx freqs (mx/array h-ids [n] mx/int32) 0)
        wf   (mx/take-idx freqs (mx/array w-ids [n] mx/int32) 0)]
    (mx/concatenate [hf wf] 1)))                          ; [N, half]

(defn- interpolate-pos
  "Bilinear-interpolate pos-embed [S*S, D] to [th*tw, D] (align_corners=false,
   floor/frac clamped — mirrors vision/interpolate.rs). Host-computed corner
   indices + weights (grids are tiny), 4 gathers + weighted sum."
  [pos-embed s th tw]
  (if (and (= th s) (= tw s))
    pos-embed
    (let [coords (fn [n-out n-in]
                   (vec (for [i (range n-out)]
                          (let [src (- (* (+ i 0.5) (/ n-in n-out)) 0.5)
                                i0  (min (max (js/Math.floor src) 0) (dec n-in))
                                i1  (min (inc i0) (dec n-in))
                                f   (min (max (- src i0) 0.0) 1.0)]
                            [i0 i1 f]))))
          ys (coords th s)
          xs (coords tw s)
          n  (* th tw)
          idx (fn [sel-y sel-x]
                (mx/array (vec (for [[y0 y1 _] ys [x0 x1 _] xs]
                                 (+ (* (sel-y [y0 y1]) s) (sel-x [x0 x1]))))
                          [n] mx/int32))
          wgt (fn [f]
                (mx/array (vec (for [[_ _ wy] ys [_ _ wx] xs] (f wy wx)))
                          [n 1] mx/float32))
          g   (fn [sy sx] (mx/take-idx pos-embed (idx sy sx) 0))]
      (mx/add
       (mx/add (mx/multiply (g first first)  (wgt (fn [wy wx] (* (- 1 wy) (- 1 wx)))))
               (mx/multiply (g first second) (wgt (fn [wy wx] (* (- 1 wy) wx)))))
       (mx/add (mx/multiply (g second first) (wgt (fn [wy wx] (* wy (- 1 wx)))))
               (mx/multiply (g second second) (wgt (fn [wy wx] (* wy wx)))))))))

(defn- block-mask
  "Additive block-diagonal mask [N N] from per-frame lengths (images don't
   attend across each other). nil when there is a single block."
  [lens dtype]
  (when (> (count lens) 1)
    (let [n (reduce + lens)
          starts (reductions + 0 lens)
          flat (vec (flatten
                     (for [i (range n)]
                       (let [bi (some (fn [[s e]] (when (and (>= i s) (< i e)) [s e]))
                                      (map vector starts (rest starts)))]
                         (for [j (range n)]
                           (if (and (>= j (first bi)) (< j (second bi))) 0.0 -1e9))))))]
      (mx/astype (mx/array flat [n n]) dtype))))

(defn vision-features
  "Run the owned vision tower: pixel-values [N 3 ps ps] (f32, from
   mx/vlm-preprocess) + grids [[t h w] ...] -> features [merged, out-hidden].
   `weights` is the checkpoint map (vision_tower.* keys, bf16)."
  [weights vcfg pixel-values grids]
  (let [{:keys [depth hidden heads patch merge eps num-pos]} vcfg
        g       (fn [s] (get weights (str wp s)))
        head-d  (quot hidden heads)
        scale   (/ 1.0 (js/Math.sqrt head-d))
        ;; --- patch embed: temporal-summed kernel as per-patch linear ---
        w5      (g "patch_embed.proj.weight")            ; [D t ph pw C]
        w2      (mx/reshape (mx/sum w5 [1] false) [hidden (* patch patch 3)])
        n-patch (first (mx/shape pixel-values))
        px      (-> pixel-values
                    (mx/transpose [0 2 3 1])             ; [N ph pw C]
                    (mx/reshape [n-patch (* patch patch 3)]))
        h0      (mx/add (mx/matmul px (mx/transpose w2)) (g "patch_embed.proj.bias"))
        ;; --- interpolated position embeddings per image ---
        s       (js/Math.round (js/Math.sqrt num-pos))
        pe      (g "pos_embed.weight")                   ; [S*S, D]
        pos     (mx/concatenate
                 (vec (mapcat (fn [[t h w]]
                                (repeat t (interpolate-pos pe s h w)))
                              grids))
                 0)
        h1      (mx/add h0 pos)
        ;; --- 2D rope tables + block mask ---
        fr      (rot-freqs grids head-d)                 ; [N half]
        cs      (mx/reshape (mx/concatenate [(mx/cos fr) (mx/cos fr)] 1) [1 n-patch 1 head-d])
        sn      (mx/reshape (mx/concatenate [(mx/sin fr) (mx/sin fr)] 1) [1 n-patch 1 head-d])
        lens    (vec (mapcat (fn [[t h w]] (repeat t (* h w))) grids))
        mask    (block-mask lens (mx/dtype h1))
        ;; --- encoder blocks ---
        hh
        (reduce
         (fn [h i]
           (let [p  (str "blocks." i ".")
                 hn (layer-norm h (g (str p "norm1.weight")) (g (str p "norm1.bias")) eps)
                 qkv (-> (lin hn (g (str p "attn.qkv.weight")) (g (str p "attn.qkv.bias")))
                         (mx/reshape [n-patch 3 heads head-d])
                         (mx/transpose [1 0 2 3]))       ; [3 N H hd]
                 pick (fn [j] (mx/reshape (mx/take-idx qkv (mx/array [j] [1] mx/int32) 0)
                                          [1 n-patch heads head-d]))
                 q  (apply-rope (pick 0) cs sn head-d)
                 k  (apply-rope (pick 1) cs sn head-d)
                 v  (pick 2)
                 tp (fn [x] (mx/transpose x [0 2 1 3]))  ; [1 H N hd]
                 o  (-> (mx/scaled-dot-product-attention (tp q) (tp k) (tp v) scale mask)
                        (mx/transpose [0 2 1 3])
                        (mx/reshape [n-patch hidden]))
                 h  (mx/add h (lin o (g (str p "attn.proj.weight")) (g (str p "attn.proj.bias"))))
                 mn (layer-norm h (g (str p "norm2.weight")) (g (str p "norm2.bias")) eps)
                 ff (lin (gelu-tanh (lin mn (g (str p "mlp.linear_fc1.weight"))
                                        (g (str p "mlp.linear_fc1.bias"))))
                         (g (str p "mlp.linear_fc2.weight"))
                         (g (str p "mlp.linear_fc2.bias")))]
             (mx/add h ff)))
         h1 (range depth))
        ;; --- merger (per image; spatial 2x2 merge -> MLP) ---
        outs
        (loop [gs grids start 0 acc []]
          (if-let [[t h w] (first gs)]
            (let [np   (* t h w)
                  x    (mx/take-idx hh (mx/arange start (+ start np)) 0)
                  xn   (layer-norm x (g "merger.norm.weight") (g "merger.norm.bias") eps)
                  hb   (quot h merge)
                  wb   (quot w merge)
                  xm   (-> xn
                           (mx/reshape [t hb merge wb merge hidden])
                           (mx/transpose [0 1 3 2 4 5])
                           (mx/reshape [(* t hb wb) (* merge merge hidden)]))
                  y    (lin (gelu-tanh (lin xm (g "merger.linear_fc1.weight")
                                           (g "merger.linear_fc1.bias")))
                            (g "merger.linear_fc2.weight")
                            (g "merger.linear_fc2.bias"))]
              (recur (rest gs) (+ start np) (conj acc y)))
            acc))]
    (if (= 1 (count outs)) (first outs) (mx/concatenate outs 0))))

;; ----------------------------------------------------------------------------
;; VLM prefill: expand pads, merge features into embeds, M-RoPE positions
;; ----------------------------------------------------------------------------

(def image-token-id 248056)

(defn- expand-image-pads
  "Expand each single <|image_pad|> marker to its image's merged token count
   (one marker per image, in order — the chat-template layout)."
  [tokens counts]
  (loop [ts tokens cs counts out []]
    (if-let [t (first ts)]
      (if (and (= t image-token-id) (seq cs))
        (recur (rest ts) (rest cs) (into out (repeat (first cs) image-token-id)))
        (recur (rest ts) cs (conj out t)))
      out)))

(defn- mrope-position-ids
  "3-axis M-RoPE position ids [[t…] [h…] [w…]] + rope-delta for the expanded
   token seq (mirrors get_rope_index): text advances all three axes together;
   each image block gets (t, h, w) spatial ids from a shared base; after an
   image the counter jumps to base + max(t,h,w). rope-delta = max-pos + 1 - T."
  [tokens merged-grids]
  (let [T (count tokens)]
    (loop [i 0 gs merged-grids pos 0 t [] h [] w []]
      (if (< i T)
        (if (= (nth tokens i) image-token-id)
          (let [[gt gh gw] (first gs)
                n (* gt gh gw)
                idxs (for [ti (range gt) hi (range gh) wi (range gw)] [ti hi wi])]
            (recur (+ i n) (rest gs)
                   (+ pos (max gt gh gw))
                   (into t (map #(+ pos (nth % 0)) idxs))
                   (into h (map #(+ pos (nth % 1)) idxs))
                   (into w (map #(+ pos (nth % 2)) idxs))))
          (recur (inc i) gs (inc pos) (conj t pos) (conj h pos) (conj w pos)))
        {:position-ids [t h w]
         :rope-delta (- (inc (reduce max 0 (concat t h w))) T)}))))

(defn vlm-prefill
  "The owned VLM prefill (genmlx-w3og): native preprocessing (mx/vlm-preprocess)
   -> OWNED vision tower -> expand <|image_pad|> markers -> scatter feature
   rows into the pad slots of the text embeddings -> 3-axis M-RoPE prefill
   through the owned qwen3_5(_moe) decoder.

   `model` = the fwd model map {:config :weights}; `vcfg` from
   load-vision-config; `images` = seq of byte buffers; `tokens` = chat-rendered
   prompt ids with ONE image_pad marker per image.

   The decoder prefill runs CHUNKED (`:chunk` opt, default 192 tokens), with
   the carry-over cache materialized between chunks. History: the default was
   48 when the GDN recurrence ran as a per-token host loop (~2 MB of live f32
   state per token per layer in one lazy graph; an unchunked ~630-token VLM
   prefill OOMed the box at >105 GB — genmlx-w3og). The fused GDN scan
   (genmlx-ps8a) removed that graph-size pressure — the ps8a chunk sweep
   measured 5.0 ms/token at 192 vs 5.9 at 48 on the 35B, with unchunked also
   safe — so 192 is now the measured optimum; the chunk boundary is retained
   as a cheap memory backstop.

   Returns {:logits [vocab] :cache :seq-len :rope-delta} — continue decoding
   with q35/step at offset (+ seq-len rope-delta) + relative step index
   (compressed M-RoPE positions; genmlx-52mh)."
  [{:keys [config weights] :as model} vcfg images tokens & [{:keys [chunk] :or {chunk 192}}]]
  (let [[pv grid-arr] (mx/vlm-preprocess images)
        grids   (mapv vec (mx/->clj grid-arr))
        m       (:merge vcfg)
        merged  (mapv (fn [[t h w]] [t (quot h m) (quot w m)]) grids)
        counts  (mapv (fn [[t h w]] (* t h w)) merged)
        expanded (expand-image-pads (vec tokens) counts)
        T       (count expanded)
        {:keys [position-ids rope-delta]} (mrope-position-ids expanded merged)
        ;; text embeddings with feature rows spliced into the pad runs
        embed   (get weights "language_model.model.embed_tokens.weight")
        feats   (mx/astype (vision-features weights vcfg pv grids) (mx/dtype embed))
        h0      (let [ids (mx/array expanded [T] mx/int32)
                      te  (mx/take-idx embed ids 0)          ; [T hidden]
                      runs (loop [i 0 segs [] fstart 0]      ; splice per segment
                             (if (< i T)
                               (if (= (nth expanded i) image-token-id)
                                 (let [j (loop [j i] (if (and (< j T) (= (nth expanded j) image-token-id)) (recur (inc j)) j))
                                       n (- j i)]
                                   (recur j (conj segs (mx/take-idx feats (mx/arange fstart (+ fstart n)) 0))
                                          (+ fstart n)))
                                 (let [j (loop [j i] (if (and (< j T) (not= (nth expanded j) image-token-id)) (recur (inc j)) j))]
                                   (recur j (conj segs (mx/take-idx te (mx/arange i j) 0)) fstart)))
                               segs))]
                  (mx/reshape (if (= 1 (count runs)) (first runs) (mx/concatenate runs 0))
                              [1 T (:hidden config)]))
        _ (mx/materialize! h0)
        h0-flat (mx/reshape h0 [T (:hidden config)])
        [last-logits cache]
        (loop [start 0 cache (q35/init-cache model) logits nil]
          (if (< start T)
            (let [n    (min chunk (- T start))
                  hs   (mx/reshape (mx/take-idx h0-flat (mx/arange start (+ start n)) 0)
                                   [1 n (:hidden config)])
                  pids (mapv #(subvec (vec %) start (+ start n)) position-ids)
                  [lg c] (q35/forward-embeds model hs cache pids start)
                  ;; materialize the kept row — an UNevaluated logits node
                  ;; would pin the whole chunk graph (GDN states included)
                  ;; across chunks and OOM exactly like the unchunked path
                  last-row (mx/index lg (dec n))]
              (mx/materialize! last-row)
              (q35/materialize-cache! c)
              (mx/force-gc!)
              (recur (+ start n) c last-row))
            [logits cache]))]
    {:logits last-logits
     :cache  cache
     :seq-len T
     :rope-delta rope-delta}))
