;; @tier slow
(ns genmlx.llm.batched-forward-gate-test
  "genmlx-9uyg GOLDEN GATE: the [B]-batched owned forward vs GenMLX's OWN B=1
   forward (never upstream). Written FIRST per the bean's mandate — the
   batched GDN/attention/MoE surgery has a silent-wrong-but-finite failure
   mode (no crash), so every batched path must reproduce the scalar path
   before it can land.

   Gates, per model (skip-if-absent, per-model owned load):
     G4  broadcast-cache value equality — every tiled lane of every layer
         entry is exactly the B=1 original (asserts the tile itself).
     G3  batched T>1 prefill — K equal-length DIFFERENT prompts through ONE
         [B T] forward vs per-prompt scalar prefill (batch independence at
         T>1: fused GDN scan, full-attn mask, MoE routing), plus one batched
         step continuing from the batched cache.
     G1  K-copies decode — B=1 prefill, lazy tile via forward-step-batched,
         K IDENTICAL tokens: every lane vs the scalar branch-replay
         reference; lanes mutually bit-identical (dense models — one kernel,
         same rows).
     G2  divergent lanes (the production shape) — K DISTINCT tokens, then
         lockstep predetermined per-lane greedy continuations for
         conv-K+2 steps vs per-lane owned branch replay: the cross-lane
         contamination detector for GDN recurrent+conv state, KV cache, and
         (on MoE checkpoints) routing.

   Assertion discipline: argmax exact per lane/step — except on
   dequantized-quantized checkpoints, where batched-vs-single-row reduction
   order can flip a SUB-ULP tie (measured on 0.8b: scalar top-3 lps
   [-2.625 -2.6875 -2.6875] round to a 3-way exact tie in the batched row;
   bf16 lp quantum 0.0625 here), so non-fp models accept argmax-exact OR a
   tie within tie-tol on the reference path's own logprobs. Top-5 ids exact
   only on full-precision checkpoints (same ULP-tie reason, genmlx-9iqc);
   top-5 logprob band 0.25 (the established cross-path band); cross-lane
   bit-identity on dense checkpoints only (an MoE gather-qmm lane would need
   the cnhi band)."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.forward :as fwd]
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

(def prompt "The capital of France is")
(def model-root (str (.-HOME js/process.env) "/.cache/models"))

(def models
  [{:name "qwen3-0.6b"   :dir "qwen3-0.6b-mlx-bf16"   :fp? true  :dense? true}
   {:name "qwen3.5-0.8b" :dir "qwen3.5-0.8b-mlx-bf16" :fp? false :dense? true}
   {:name "qwen3.5-4b"   :dir "qwen3.5-4b-mlx-bf16"   :fp? true  :dense? true}
   ;; 35B MoE (packed experts via gather-qmm): heavy — opt-in via
   ;; GENMLX_GATE_35B=1 under ~/genmlx-guarded-run.sh (Thor run discipline).
   ;; No cross-lane bit-identity, and the lp band sits at the cnhi jitter
   ;; CEILING (0.625 nats between IDENTICAL runs of the quantized gather-qmm
   ;; path — measured 2026-07-07, genmlx-cnhi; 0.3 flaked on one step of the
   ;; first gate run, 2026-07-10). On MoE the strong invariant is per-step
   ;; argmax/tie agreement; the band only bounds gross divergence.
   {:name "qwen3.5-35B-A3B-6bit (MoE)"
    :path (str (.-HOME js/process.env)
               "/code/mlx/models/Qwen3.5-35B-A3B-6bit/snapshots/"
               "b729d115bb2cfea696e390dd6bb898528c66b6e9")
    :fp? false :dense? false :band 0.65
    :opt-in "GENMLX_GATE_35B"}])

(def K 3)          ;; batch width for every gate
;; Every cross-path deviation measured in G2 is an EXACT multiple of the bf16
;; logprob quantum (genmlx-828j): the fixtures are bf16 and the compute dtype
;; comes from the embedding (qwen3_forward.cljs:403), so the whole forward and
;; the logits are bf16. Both constants below are therefore expressed in quanta
;; and deliberately placed BETWEEN lattice points — a strict `<` against a
;; lattice atom is a coin flip by construction.
(def lp-quantum 0.0625)

;; 4.5 quanta. The old 0.25 was simultaneously an exact lattice value AND the
;; observed full-vocab ceiling, so `< 0.25` operationally meant "<= 3 quanta".
;; Measured top-5 band over the 18 (step, lane) samples this gate asserts on:
;; {0.0 x2, 0.0625 x5, 0.125 x10, 0.25 x1} — nothing between 0.125 and 0.25 is
;; achievable, and the single 4-quantum outlier failed by EXACTLY 0.0.
;; NB the band is a coarse gross-divergence bound, not a safety margin: the
;; FULL-VOCAB cross-path max is 0.25-0.5 on every step even when everything is
;; correct. The argmax+tie predicate is the strong invariant.
(def band (* 4.5 lp-quantum))

;; 2.5 quanta. The old 0.13 accepted the measured 0.125 tie with 0.005 nats of
;; margin — 4% of a quantum. This robustly means "<= 2 quanta" and still
;; rejects any 3-quantum gap, which would be a genuine rank swap.
(def tie-tol (* 2.5 lp-quantum))

(defn- log-softmax [logits] (mx/subtract logits (mx/logsumexp logits)))

(defn- topk-ids [lp k]
  (mx/eval! lp)
  (let [f32 (.toFloat32 lp)]
    (->> (range (.-length f32))
         (map (fn [i] [i (aget f32 i)]))
         (sort-by second >)
         (take k)
         (mapv first))))

(defn- max-abs-diff
  ([a b] (mx/item (mx/amax (mx/abs (mx/subtract a b)))))
  ([a b ids]
   (mx/eval! a) (mx/eval! b)
   (let [af (.toFloat32 a) bf (.toFloat32 b)]
     (reduce max (map (fn [i] (js/Math.abs (- (aget af i) (aget bf i)))) ids)))))

(defn- lane
  "Row k of a [K vocab] logits block, as [vocab]."
  [logits-bk k]
  (mx/index logits-bk k))

(defn- tok-array [ids] (mx/array (vec ids) [(count ids)] mx/int32))

(defn- argmax-agrees?
  "Exact argmax agreement — or a sub-ULP tie: the REFERENCE path's logprobs of
   the two candidates differ by < tie-tol, so the flip is a rounding tie-break
   rather than divergence.

   The tie escape used to be gated on (not fp?), i.e. granted only to
   \"dequantized-quantized\" checkpoints. That gate was wrong and is removed
   (genmlx-828j). Measured on qwen3-0.6b, a :fp? true fixture:

     ref_lp[15344] = -1.750000   ref_lp[9625] = -1.875000   gap 0.125  (2 quanta)
     bat_lp[15344] = -1.750000   bat_lp[9625] = -1.750000   gap 0.000  (exact tie,
                                                  broken by mx/argmax on lower index)

   The tie is a property of the bf16 LOGIT LATTICE, not of weight provenance:
   :fp? distinguishes \"dequantized from a quantized file\" from \"stored bf16\",
   and neither computes logits at higher precision. Demanding exact argmax on a
   bf16 checkpoint demands that a B=1 GEMV and a B=3 GEMM agree bit-for-bit
   across a 2-quantum gap; arithmetic cannot supply that.

   This cannot mask a real bug. It is gated on the REFERENCE path's own gap, so
   a genuine batched defect — which moves logprobs far more than 2 quanta — is
   still caught here, and the top-5 band assertion bounds gross divergence
   independently."
  [ref-lp b-arg s-arg]
  (or (= b-arg s-arg)
      (< (js/Math.abs (- (mx/item (mx/index ref-lp s-arg))
                         (mx/item (mx/index ref-lp b-arg))))
         tie-tol)))

(defn- assert-argmaxes
  "Per-lane argmax agreement between batched logits [K V] and per-lane
   reference logits (seq of [V]), tie-aware (see argmax-agrees?)."
  [label blogits refs]
  (let [b (mapv #(mx/item (mx/argmax (lane blogits %))) (range (count refs)))
        s (mapv #(mx/item (mx/argmax %)) refs)]
    (assert-true (str label " (batched=" (pr-str b) " scalar=" (pr-str s) ")")
                 (every? (fn [k]
                           (argmax-agrees? (log-softmax (nth refs k))
                                           (nth b k) (nth s k)))
                         (range (count refs))))))

(defn- assert-band
  "Per-lane top-5 logprob band between batched logits [K V] and per-lane
   reference logits (seq of [V]), REPORTING the measured maximum.

   The four band assertions used to build their label from the CONSTANT and
   pass a bare boolean, so the log showed neither the measured deviation nor
   which lane tripped — an assertion that looks identical at 0.1875 and at
   0.375. Diagnosing genmlx-828j needed a GPU probe to recover a number the
   test already had in hand. Print it (assert-argmaxes prints both tuples,
   which is the only reason the sibling failure was diagnosable from the
   battery log at all)."
  [label blogits refs]
  (let [devs (mapv (fn [k]
                     (let [r (log-softmax (nth refs k))]
                       (max-abs-diff (log-softmax (lane blogits k)) r
                                     (topk-ids r 5))))
                   (range (count refs)))
        worst (reduce max devs)]
    (assert-true (str label " (max " (.toFixed worst 6) " < " band
                      ", per-lane " (pr-str (mapv #(.toFixed % 4) devs)) ")")
                 (< worst band))))

;; ---------------------------------------------------------------------------
;; G4: broadcast-cache value equality
;; ---------------------------------------------------------------------------

(defn- gate-broadcast-cache [fm ids]
  (println "  -- G4 broadcast-cache value equality --")
  (let [[_ cache] (fwd/prefill fm ids)
        bc (fwd/broadcast-cache cache K)
        checks
        (for [i     (range (count cache))
              :let  [orig (nth cache i) tiled (nth bc i)]
              :when (some? orig)
              [kk a] orig
              :let  [t (get tiled kk)]]
          (and (= K (first (mx/shape t)))
               (every? (fn [j] (zero? (max-abs-diff (mx/index t j) (mx/index a 0))))
                       (range K))))]
    (assert-true (str "every layer entry tiled to K=" K " with exact lane values")
                 (and (seq checks) (every? true? checks)))))

;; ---------------------------------------------------------------------------
;; G3: batched T>1 prefill (K equal-length different prompts) + one step
;; ---------------------------------------------------------------------------

(defn- gate-batched-prefill [fm ids spec]
  (println "  -- G3 batched T>1 prefill (K different prompts) --")
  ;; Same length by construction: swap one interior token id. The text is
  ;; meaningless — this is a numerical batch-independence gate.
  (let [{:keys [fp?]} spec
        band     (get spec :band band)
        variants (mapv #(assoc (vec ids) 3 %) [(nth ids 3) 5000 12345])
        refs     (mapv (fn [v] (first (fwd/prefill fm v))) variants)   ; [V] each
        [blogits bcache] (fwd/prefill-batched fm variants)             ; [K V]
        argmaxes (mapv #(mx/item (mx/argmax %)) refs)]
    (assert-argmaxes "per-lane argmax == per-prompt scalar prefill argmax"
                     blogits refs)
    (when fp?
      (assert= "per-lane top-5 ids == scalar top-5 ids (full-precision)"
               (mapv #(topk-ids (log-softmax %) 5) refs)
               (mapv #(topk-ids (log-softmax (lane blogits %)) 5) (range K))))
    (assert-band "per-lane top-5 logprob band" blogits refs)
    ;; one batched step from the batched cache vs scalar prefill+step per lane
    (let [T       (count ids)
          toks    argmaxes
          srefs   (mapv (fn [v t]
                          (let [[_ c] (fwd/prefill fm v)]
                            (first (fwd/step fm c T t))))
                        variants toks)
          [slogits _] (fwd/step-batched fm bcache T (tok-array toks))]
      (assert-argmaxes "post-prefill batched step argmax == scalar step argmax, per lane"
                       slogits srefs)
      (assert-band "post-prefill batched step top-5 band" slogits srefs))))

;; ---------------------------------------------------------------------------
;; G1: K-copies decode (tile + identical tokens)
;; ---------------------------------------------------------------------------

(defn- gate-k-copies [model ids spec]
  (println "  -- G1 K-copies decode --")
  (llm/init-cache! model)
  (let [{:keys [fp? dense?]} spec
        band   (get spec :band band)
        pf     (llm/forward-prefill model ids)
        tok    (mx/item (mx/argmax pf))
        br     (llm/branch-cache! model)
        ref    (llm/forward-branch model br tok)                        ; [V] scalar ref
        blogits (llm/forward-step-batched model (tok-array (repeat K tok)))]
    (llm/dispose-branch! model br)
    (assert-argmaxes (str "all " K " lanes argmax == scalar step argmax")
                     blogits (vec (repeat K ref)))
    (when fp?
      (assert= "per-lane top-5 ids == scalar top-5 (full-precision)"
               (vec (repeat K (topk-ids (log-softmax ref) 5)))
               (mapv #(topk-ids (log-softmax (lane blogits %)) 5) (range K))))
    (assert-band "per-lane top-5 logprob band" blogits (vec (repeat K ref)))
    (if dense?
      (assert-true "lanes mutually bit-identical (dense: one kernel, same rows)"
                   (every? #(zero? (max-abs-diff (lane blogits %) (lane blogits 0)))
                           (range 1 K)))
      (assert-true (str "lanes mutually within the MoE jitter band (cnhi, < " band ")")
                   (every? #(< (max-abs-diff (log-softmax (lane blogits %))
                                             (log-softmax (lane blogits 0))
                                             (topk-ids (log-softmax (lane blogits 0)) 5))
                               band)
                           (range 1 K))))
    (llm/reset-cache! model)))

;; ---------------------------------------------------------------------------
;; G2: divergent lanes, lockstep multi-step vs per-lane branch replay
;; ---------------------------------------------------------------------------

(defn- gate-divergent-lanes [model ids spec]
  (println "  -- G2 divergent lanes (production shape) --")
  (llm/init-cache! model)
  (let [{:keys [fp?]} spec
        band    (get spec :band band)
        n-steps 6                                   ; >= conv-K-1 + 2 (conv-k 4)
        pf      (llm/forward-prefill model ids)
        firsts  (vec (take K (topk-ids (log-softmax pf) (+ K 2))))
        ;; scalar reference: per lane, branch from the prefix and replay
        ;; greedily, recording fed tokens + logits at every step
        refs
        (mapv (fn [t0]
                (let [br (llm/branch-cache! model)
                      r  (loop [j 0 t t0 fed [] lgs []]
                           (let [lg (llm/forward-branch model br t)
                                 fed (conj fed t) lgs (conj lgs lg)]
                             (if (= (inc j) n-steps)
                               {:fed fed :logits lgs}
                               (recur (inc j) (mx/item (mx/argmax lg)) fed lgs))))]
                  (llm/dispose-branch! model br)
                  r))
              firsts)
        ;; batched: feed the SAME recorded per-lane tokens, lockstep
        steps
        (loop [j 0 out []]
          (if (= j n-steps)
            out
            (let [toks (mapv #(nth (:fed %) j) refs)]
              (recur (inc j)
                     (conj out (llm/forward-step-batched model (tok-array toks)))))))]
    (doseq [j (range n-steps)]
      (let [b (nth steps j)]
        (assert-argmaxes (str "step " j ": per-lane argmax == branch-replay argmax")
                         b (mapv #(nth (:logits %) j) refs))
        (assert-band (str "step " j ": per-lane top-5 logprob band")
                     b (mapv #(nth (:logits %) j) refs))))
    (when fp?
      (let [b (peek steps)]
        (assert= "final step: per-lane top-5 ids == branch-replay (full-precision)"
                 (mapv #(topk-ids (log-softmax (peek (:logits %))) 5) refs)
                 (mapv #(topk-ids (log-softmax (lane b %)) 5) (range K)))))
    (llm/reset-cache! model)))

;; ---------------------------------------------------------------------------

(defn check-model [{:keys [name dir path opt-in] :as spec}]
  (let [path (or path (str model-root "/" dir))]
    (cond
      (and opt-in (not= "1" (aget js/process.env opt-in)))
      (do (println (str "\n== " name " — SKIP (opt-in: set " opt-in "=1) =="))
          (pr/resolved nil))

      (not (.existsSync fs path))
      (do (println (str "\n== " name " — SKIP (absent: " path ") ==")) (pr/resolved nil))

      :else
      (pr/let [m   (llm/load-model path {:cljs-forward? true})
               tok (:tokenizer m)
               ids-raw (llm/encode tok prompt false)]
        (println (str "\n== " name " batched-forward golden gate ==  prompt=" (pr-str prompt)))
        (let [ids (vec ids-raw)
              fm  (:fwd (:model m))]
          (gate-broadcast-cache fm ids)
          (gate-batched-prefill fm ids spec)
          (gate-k-copies (:model m) ids spec)
          (gate-divergent-lanes (:model m) ids spec)
          (mx/force-gc!))))))

(pr/let [_ (check-model (nth models 0))
         _ (check-model (nth models 1))
         _ (check-model (nth models 2))
         _ (check-model (nth models 3))]
  (println (str "\n=== llm-batched-forward-gate: " @pass " PASS, " @fail " FAIL ==="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))
