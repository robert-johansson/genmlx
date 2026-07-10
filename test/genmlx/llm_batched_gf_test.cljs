;; @tier slow
(ns genmlx.llm-batched-gf-test
  "genmlx-9uyg end-to-end: make-llm-gf-batched runs K LLM particles through
   vsimulate/vgenerate in ONE forward pass on a real checkpoint, and the L1
   law holds against the scalar make-llm-gf:

     L1: for every lane k, vsimulate score[k] == scalar assess of that lane's
         tokens truncated at eos (band 0.25 — batched-vs-scalar matmul
         reduction order on a dequantized-bf16 checkpoint; trailing pad
         sites contribute exactly 0 by the mask algebra, pinned fast-tier).

   Also: vgenerate with a shared constrained first token (weight parity vs
   scalar generate), the Route-B pipeline (vsimulate -> decode -> oracle
   score -> select; parity with Route A's sequential best-of-K is
   operational — same K/oracle/selection, candidates from ONE pass), and
   grammar composition (the Route-B-only capability): a vectorized-hook
   batched DFA constrains generation in-loop, lanes decode to grammar-valid
   text, and lane scores match the scalar wrap-grammar assess."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llmc]
            [genmlx.llm.grammar :as gr]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))
(defn assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol)
      (do (swap! pass inc) (println (str "  PASS: " label " (diff=" (.toFixed d 6) ")")))
      (do (swap! fail inc) (println (str "  FAIL: " label " expected=" expected
                                         " actual=" actual " (diff=" (.toFixed d 6) ")"))))))

(def model-root (str (.-HOME js/process.env) "/.cache/models"))
(def K 4)
(def MAX-TOKENS 8)
(def band 0.25)

(defn- t-addr [i] (keyword (str "t" i)))

(defn- site-shape [vtrace i]
  (mx/shape (cm/get-value (cm/get-submap (:choices vtrace) (t-addr i)))))

(defn- assess-lane
  "Scalar make-llm-gf assess weight of one lane's token seq (incl. eos)."
  [scalar-gf prompt-ids toks]
  (let [choices (reduce (fn [c [i t]] (cm/set-value c (t-addr i) (mx/scalar t mx/int32)))
                        cm/EMPTY (map-indexed vector toks))]
    (mx/item (:weight (p/assess scalar-gf [prompt-ids MAX-TOKENS] choices)))))

;; ---------------------------------------------------------------------------

(defn- test-vsimulate-l1 [m prompt-ids]
  (println "  -- vsimulate K lanes in one pass + L1 law --")
  (let [gf     (llmc/make-llm-gf-batched m)
        scalar (llmc/make-llm-gf m)
        eos    (llm/eos-token-id (:tokenizer m))
        vt     (dyn/vsimulate gf [prompt-ids MAX-TOKENS] K (rng/fresh-key 42))
        scores (vec (mx/->clj (mx/astype (:score vt) mx/float32)))
        lanes  (llmc/vtrace-lane-tokens vt eos)]
    (assert-true (str "all " MAX-TOKENS " sites present with [K] leaves")
                 (every? #(= [K] (site-shape vt %)) (range MAX-TOKENS)))
    (assert-true "score is [K]" (= K (count scores)))
    (assert-true (str K " lanes extracted") (= K (count lanes)))
    (doseq [l (range K)]
      (assert-close (str "L1 lane " l ": vsimulate score == scalar assess ("
                         (count (nth lanes l)) " tokens)")
                    (assess-lane scalar prompt-ids (nth lanes l))
                    (nth scores l) band))
    ;; decode gives K strings (text boundary; eos/pad stripped)
    (pr/let [texts (llmc/decode-vtrace (:tokenizer m) vt)]
      (assert-true "decode-vtrace returns K strings"
                   (and (= K (count texts)) (every? string? texts)))
      (println (str "    lanes: " (pr-str (mapv #(subs % 0 (min 30 (count %))) texts))))
      {:vt vt :scores scores :lanes lanes :scalar scalar})))

(defn- test-vgenerate-constrained [m prompt-ids]
  (println "  -- vgenerate with a constrained first token --")
  (let [gf     (llmc/make-llm-gf-batched m)
        scalar (llmc/make-llm-gf m)
        ;; constrain :t0 to the model's own greedy pick (any in-vocab id works)
        _      (llm/init-cache! (:model m))
        tok    (mx/item (mx/argmax (llm/forward-prefill (:model m) prompt-ids)))
        _      (llm/reset-cache! (:model m))
        cons'  (cm/set-value cm/EMPTY (t-addr 0) (mx/scalar tok mx/int32))
        vt     (dyn/vgenerate gf [prompt-ids 2] cons' K (rng/fresh-key 7))
        w      (vec (mx/->clj (mx/astype (:weight vt) mx/float32)))
        sw     (mx/item (:weight (p/generate (dyn/auto-key scalar)
                                             [prompt-ids 2] cons')))]
    (assert-true "weight is [K] and finite"
                 (and (= K (count w)) (every? js/isFinite w)))
    (doseq [l (range K)]
      (assert-close (str "lane " l " weight == scalar generate weight (shared prefix)")
                    sw (nth w l) band))))

(defn- test-pipeline-select [{:keys [scores lanes scalar]} prompt-ids]
  (println "  -- Route-B pipeline: oracle-score -> select --")
  ;; Oracle = the scalar model evidence (the same oracle Route A's sequential
  ;; best-of-K uses); candidates came from ONE vsimulate pass.
  (let [oracle (mapv #(assess-lane scalar prompt-ids %) lanes)
        best   (apply max-key #(nth oracle %) (range K))
        chosen (apply max-key #(nth scores %) (range K))]
    (assert-true "batched-score selection agrees with oracle selection within band"
                 (<= (- (nth oracle best) (nth oracle chosen)) band))))

(defn- test-grammar-composition [m prompt-ids]
  (println "  -- grammar composition (vectorized-hook batched DFA) --")
  (let [tokz   (:tokenizer m)
        regex  "(yes|no|maybe)"
        c      (gr/compile-constraint tokz regex)
        vocab  (get-in (:fwd (:model m)) [:config :vocab])
        vt*    (gr/build-vtables c vocab)
        hook   (gr/vectorized-hook vt*)
        gf     (llmc/make-llm-gf-batched m {:hook hook})
        scalar (llmc/make-llm-gf m)
        eos    (llm/eos-token-id tokz)
        vtr    (dyn/vsimulate gf [prompt-ids 4] K (rng/fresh-key 11))
        scores (vec (mx/->clj (mx/astype (:score vtr) mx/float32)))
        lanes  (llmc/vtrace-lane-tokens vtr eos)]
    (pr/let [texts (llmc/decode-vtrace tokz vtr)]
      (doseq [l (range K)]
        (let [txt (nth texts l)
              st  (gr/dfa-advance-string (:dfa c) (get-in c [:dfa :start]) txt)]
          (assert-true (str "lane " l " text " (pr-str txt) " stays in the grammar")
                       (and (not= :dead st) (contains? (get-in c [:dfa :alive]) st)))))
      ;; score parity vs the scalar wrap-grammar path on the same tokens
      (let [g-scalar (gr/constrain scalar c)]
        (doseq [l (range K)]
          (let [choices (reduce (fn [cm' [i t]]
                                  (cm/set-value cm' (t-addr i) (mx/scalar t mx/int32)))
                                cm/EMPTY (map-indexed vector (nth lanes l)))
                w (mx/item (:weight (p/assess g-scalar [prompt-ids 4] choices)))]
            (assert-close (str "lane " l ": batched grammar score == scalar wrap-grammar assess")
                          w (nth scores l) band)))))))

;; ---------------------------------------------------------------------------

(defn- run-model [name dir prompt include-grammar?]
  (let [path (str model-root "/" dir)]
    (if-not (.existsSync fs path)
      (do (println (str "\n== " name " — SKIP (absent: " path ") ==")) (pr/resolved nil))
      (pr/let [m       (llm/load-model path {:cljs-forward? true})
               ids-raw (llm/encode (:tokenizer m) prompt false)]
        (println (str "\n== " name " batched LLM-GF ==  prompt=" (pr-str prompt)))
        (let [ids (vec ids-raw)]
          (pr/let [ctx (test-vsimulate-l1 m ids)]
            (test-vgenerate-constrained m ids)
            (test-pipeline-select ctx ids)
            (if include-grammar?
              (test-grammar-composition m ids)
              (pr/resolved nil))))))))

(pr/let [_ (run-model "qwen3.5-0.8b" "qwen3.5-0.8b-mlx-bf16"
                      "Answer with yes, no, or maybe: is the sky blue? Answer: " true)
         _ (run-model "qwen3-0.6b" "qwen3-0.6b-mlx-bf16"
                      "The capital of France is" false)]
  (println (str "\n=== llm-batched-gf: " @pass " PASS, " @fail " FAIL ===")))
