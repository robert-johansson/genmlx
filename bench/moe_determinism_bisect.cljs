(ns bench.moe-determinism-bisect
  "genmlx-mdet Phase 1: bisect the quantized-MoE forward's run-to-run jitter
   by fingerprinting every sublayer output across repeated identical runs.

   Context (do not re-derive; bean genmlx-mdet): dense bf16 is
   bit-deterministic on both forwards; the jitter (0.2-2.6 nats, scaling with
   sequence length) is specific to the quantized-MoE expert path; gather_qmm
   is exonerated standalone; suspects are (1) fused SDPA reduction order,
   (2) the GDN f32 recurrence, (3) the router precise-f32 softmax. The jitter
   enters IN SITU — under concurrency/stream interleaving — so this harness
   runs the REAL forward and fingerprints sublayers via the :layer-tap /
   :moe-tap seams rather than testing kernels standalone.

   Fingerprints: per tapped array, two fixed-random projections + sum + absmax
   (f32 scalars, exact-bit compared across runs). A 1-ULP wobble anywhere in
   the tensor moves the projections with probability ~1.

   Modes (GENMLX_MDET_MODE, default \"baseline,lazy\"):
     baseline — no taps; fingerprint final logits only (does the jitter
                reproduce at all, untap-perturbed?)
     lazy     — taps collect lazy fingerprint scalars, ONE materialize! at
                the end of each run (minimal scheduling interference)
     eager    — each tap materializes its fingerprint immediately (a
                per-sublayer barrier; if this KILLS the jitter, the culprit
                is cross-stream scheduling of the fused graph, and barriers
                are the deterministic-mode lever)
     decode   — N greedy 32-step rollouts from one shared prefill cache;
                compares token sequences + per-step logit fingerprints

   Output: per-tap distinct-fingerprint counts across runs, first divergent
   tap in layer order, per-layer-kind summary; JSON to
   results/moe-determinism/.

   Usage (guarded, ONE GPU process):
     ~/genmlx-guarded-run.sh mdet-bisect bunx --bun nbb@1.4.208 \\
       bench/moe_determinism_bisect.cljs
   Env: GENMLX_MDET_MODEL (default Ornith-1.0-35B-8bit HF snapshot),
        GENMLX_MDET_T (default 512), GENMLX_MDET_N (default 10),
        GENMLX_MDET_MODE, GENMLX_MDET_DECODE_STEPS (default 32)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.qwen35-forward :as q35]
            [promesa.core :as pr]
            [clojure.string :as str]))

(def fs (js/require "fs"))

(def model-dir
  (or (aget (.-env js/process) "GENMLX_MDET_MODEL")
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-8bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))

(def target-t (let [v (aget (.-env js/process) "GENMLX_MDET_T")]
                (if v (js/parseInt v 10) 512)))
(def n-runs (let [v (aget (.-env js/process) "GENMLX_MDET_N")]
              (if v (js/parseInt v 10) 10)))
(def modes (set (str/split (or (aget (.-env js/process) "GENMLX_MDET_MODE")
                               "baseline,lazy")
                           #",")))
(def decode-steps (let [v (aget (.-env js/process) "GENMLX_MDET_DECODE_STEPS")]
                    (if v (js/parseInt v 10) 32)))

(def out-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (str (js/process.cwd) "/results/moe-determinism")))

(defn- write-json [filename data]
  (when-not (.existsSync fs out-dir) (.mkdirSync fs out-dir #js {:recursive true}))
  (.writeFileSync fs (str out-dir "/" filename)
                  (js/JSON.stringify (clj->js data) nil 2))
  (println (str "  wrote: " out-dir "/" filename)))

;; ---------------------------------------------------------------------------
;; Fingerprints: fixed projections, cached per shape, exact-bit compared
;; ---------------------------------------------------------------------------

(def ^:private projectors (atom {}))

(defn- projs-for [shape]
  (or (get @projectors shape)
      (let [r1 (rng/uniform (rng/fresh-key 1234567) shape)
            r2 (rng/uniform (rng/fresh-key 7654321) shape)]
        (mx/materialize! r1 r2)
        (swap! projectors assoc shape [r1 r2])
        [r1 r2])))

(defn- fingerprint
  "Four lazy f32 scalars characterizing `x` to the bit: two fixed projections,
   the sum, and the absmax. Any single-ULP change in x moves p1/p2 w.p. ~1."
  [x]
  (let [xf (mx/astype x mx/float32)
        [r1 r2] (projs-for (vec (mx/shape x)))]
    [(mx/sum (mx/multiply xf r1))
     (mx/sum (mx/multiply xf r2))
     (mx/sum xf)
     (mx/amax (mx/abs xf))]))

;; ---------------------------------------------------------------------------
;; One tapped prefill run
;; ---------------------------------------------------------------------------

(defn- run-prefill
  "One full T-token prefill with sublayer taps. Returns
   {:taps [[label [f1 f2 f3 f4]] ...] :ms elapsed} where the four numbers are
   realized JS floats. mode :baseline = no taps; :lazy = one materialize at
   the end; :eager = materialize per tap."
  [fm ids mode]
  (let [taps (atom [])                       ; [[label [lazy-scalars]] ...]
        tap! (fn [label arr]
               (let [fp (fingerprint arr)]
                 (when (= mode :eager) (apply mx/materialize! fp))
                 (swap! taps conj [label fp])))
        layer-tap (fn [i kind label arr]
                    (tap! (str (when (>= i 0) (str "L" (when (< i 10) "0") i "."))
                               (when kind (str (first (str/split kind #"_")) "."))
                               (name label))
                          arr))
        moe-tap (fn [prefix top topw]
                  (let [lbl (second (re-find #"layers\.(\d+)\." prefix))
                        lbl (str "L" (when (< (js/parseInt lbl 10) 10) "0") lbl)]
                    (tap! (str lbl ".router-topk-idx") (mx/astype top mx/float32))
                    (tap! (str lbl ".router-topk-w") topw)))
        ;; Phase-1.5 drill-down: fingerprint the expert branch's internals
        ;; (gate-o / up-o / expert-down / moe-sum / shared / sgate) so the
        ;; L00.mlp-out divergence resolves to a named op.
        inner-tap (fn [prefix label arr]
                    (let [lnum (second (re-find #"layers\.(\d+)\." prefix))
                          lbl (str "L" (when (< (js/parseInt lnum 10) 10) "0") lnum)]
                      (tap! (str lbl ".inner." (name label)) arr)))
        fm' (case mode
              :baseline fm
              :inner (assoc-in fm [:config :moe-tap-inner] inner-tap)
              (-> fm
                  (assoc-in [:config :layer-tap] layer-tap)
                  (assoc-in [:config :moe-tap] moe-tap)))
        t0 (js/performance.now)
        logits (q35/forward fm' ids)
        ;; production-identical eval of the FULL graph first (the tap scalars
        ;; hang off already-materialized nodes afterwards, so lazy-mode taps
        ;; perturb the production schedule as little as possible)
        _ (mx/materialize! logits)
        _ (tap! "logits" logits)
        _ (tap! "logits-last" (mx/index logits (dec (count ids))))
        _ (doseq [chunk (partition-all 64 (mapcat second @taps))]
            (apply mx/materialize! chunk))
        out (mapv (fn [[label fp]] [label (mapv mx/item fp)]) @taps)
        ms (- (js/performance.now) t0)]
    {:taps out :ms ms}))

(defn- analyze
  "Across runs (vector of {:taps ...}), count distinct fingerprint tuples per
   tap label (call order preserved). Returns [{:label :distinct :spread} ...]."
  [runs]
  (let [labels (mapv first (:taps (first runs)))]
    (mapv (fn [ti]
            (let [vals (mapv #(second (nth (:taps %) ti)) runs)
                  uniq (count (distinct vals))
                  ;; spread of the first projection across runs
                  p1s (mapv first vals)
                  spread (- (apply max p1s) (apply min p1s))]
              {:label (nth labels ti) :distinct uniq :spread spread}))
          (range (count labels)))))

(defn- report [tag results]
  (println (str "\n== " tag ": per-tap distinct fingerprints over " (count results) " runs =="))
  (let [ana (analyze results)
        divergent (filterv #(> (:distinct %) 1) ana)]
    (doseq [{:keys [label distinct spread]} ana
            :when (> distinct 1)]
      (println (str "  DIVERGES  " label "  distinct=" distinct
                    "  p1-spread=" (.toExponential spread 3))))
    (if (empty? divergent)
      (println "  ALL TAPS BIT-IDENTICAL across runs")
      (println (str "  first divergent tap (layer order): " (:label (first divergent)))))
    {:analysis ana :divergent (mapv :label divergent)}))

;; ---------------------------------------------------------------------------
;; Decode probe: N greedy rollouts from one shared prefill cache
;; ---------------------------------------------------------------------------

(defn- run-decode
  "One greedy `decode-steps` rollout from (shared) `cache`/`offset`/`tok0`.
   Returns {:toks [...] :fps [[4 floats] per step]}."
  [fm cache offset tok0]
  (loop [i 0 cache cache offset offset tok tok0 toks [] fps []]
    (if (= i decode-steps)
      {:toks toks :fps fps}
      (let [[lg cache'] (q35/forward-cached fm [tok] cache offset)
            lg0 (mx/index lg 0)
            fp (fingerprint lg0)
            _ (apply mx/materialize! (conj fp lg0))
            nxt (mx/item (mx/argmax lg0))]
        (q35/materialize-cache! cache')
        (recur (inc i) cache' (inc offset) (int nxt)
               (conj toks (int nxt)) (conj fps (mapv mx/item fp)))))))

;; ---------------------------------------------------------------------------
;; Main
;; ---------------------------------------------------------------------------

(if-not (and model-dir (.existsSync fs model-dir))
  (do (println "SKIP — no checkpoint at" (str model-dir)) (js/process.exit 1))
  (pr/let [mm (llm/load-model model-dir {:cljs-forward? true})
           {:keys [model tokenizer]} mm
           enc (llm/encode tokenizer
                           (apply str (repeat 120 "Probabilistic programs denote measures over traces, and inference is conditioning. ")))]
    (let [fm  (:fwd model)
          ids (vec (take target-t (vec enc)))
          T   (count ids)
          results (atom {})]
      (println "== genmlx-mdet MoE determinism bisect ==")
      (println "model:" model-dir)
      (println (str "T=" T " N=" n-runs " modes=" (pr-str modes)))
      ;; JIT warmup (discarded)
      (println "-- warmup --")
      (run-prefill fm (vec (take 64 ids)) :baseline)
      (mx/force-gc!)

      (doseq [mode [:baseline :lazy :eager :inner]
              :when (contains? modes (name mode))]
        (println (str "\n-- mode " (name mode) ": " n-runs " tapped prefill runs --"))
        (let [runs (vec (for [r (range n-runs)]
                          (let [out (run-prefill fm ids mode)]
                            (mx/force-gc!)
                            (println (str "  run " r ": " (.toFixed (:ms out) 0) " ms"))
                            out)))
              rep (report (name mode) runs)]
          (swap! results assoc mode
                 {:ms (mapv :ms runs)
                  :divergent (:divergent rep)
                  :analysis (:analysis rep)})))

      (when (contains? modes "decode")
        (println (str "\n-- mode decode: " n-runs " greedy " decode-steps
                      "-step rollouts from one shared prefill cache --"))
        (let [[last-logits cache] (q35/prefill fm ids)
              _ (mx/materialize! last-logits)
              _ (q35/materialize-cache! cache)
              tok0 (int (mx/item (mx/argmax last-logits)))
              rollouts (vec (for [r (range n-runs)]
                              (let [out (run-decode fm cache T tok0)]
                                (mx/force-gc!)
                                (println (str "  rollout " r ": "
                                              (pr-str (take 8 (:toks out))) "…"))
                                out)))
              tok-seqs (distinct (map :toks rollouts))
              step-fps (mapv (fn [i] (count (distinct (map #(nth (:fps %) i) rollouts))))
                             (range decode-steps))
              first-div (first (keep-indexed (fn [i c] (when (> c 1) i)) step-fps))]
          (println (str "  distinct greedy token sequences: " (count tok-seqs)))
          (println (str "  first step with divergent logit fingerprints: "
                        (if first-div first-div "NONE — all steps bit-identical")))
          (swap! results assoc :decode
                 {:distinct-token-seqs (count tok-seqs)
                  :first-divergent-step first-div
                  :per-step-distinct step-fps
                  :token-seqs (mapv :toks rollouts)})))

      (write-json "bisect.json"
                  {:bean "genmlx-mdet"
                   :model model-dir
                   :T T :n-runs n-runs :modes (vec modes)
                   :results @results})
      (js/process.exit 0))))
