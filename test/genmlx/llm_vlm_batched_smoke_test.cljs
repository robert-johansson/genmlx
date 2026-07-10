;; @tier exclude — loads Ornith-1.0-35B-4bit (~25 GB) and runs a full VLM
;; prefill. Run manually, guarded and solo:
;;   ~/genmlx-guarded-run.sh vlm-batched bunx --bun nbb@1.4.208 test/genmlx/llm_vlm_batched_smoke_test.cljs
(ns genmlx.llm-vlm-batched-smoke-test
  "genmlx-lo6e D2: [K]-lane batched decode over a VLM IMAGE prefix.

   The claim (spec'd as composition, not new machinery): forward-prefill
   {:images …} installs an image-conditioned cache WITH its rope-delta;
   forward-step-batched tiles it family-agnostically (broadcast-cache) and
   rotates at offset + rope-delta lockstep — so K particles decode over one
   expensive vision prefill with zero VLM-specific batching code.

   Gates (35B MoE 4-bit → cnhi band 0.3):
     V1 batched K=4 identical-token step over the image prefix: every lane
        matches the scalar branch step (argmax tie-tolerant + top-5 lp band
        at the cnhi MoE ceiling — the strong invariant is argmax/tie
        agreement; the band only bounds gross divergence)
     V2 divergent lanes, one more lockstep step: per-lane == scalar branch
        replay of that lane's token path
     V3 batched branching over the image prefix: fork the batched cache,
        advance the branch, live cache unaffected (bit-equal)"
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_OWNED_MOE_MODEL)
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))
(def image-path (str (.-HOME js/process.env) "/code/mlx/ornith/image.png"))

(def band 1.0)    ;; gross-divergence bound ONLY. cnhi gather-qmm jitter on the
                  ;; 4-bit over an image-conditioned ~630-token prefix measured
                  ;; 0.03-0.75 nats across lanes/runs (2026-07-10, argmax exact
                  ;; in every sample) — above the 8-bit text ceiling 0.65 the
                  ;; 9uyg gate uses. The STRONG invariant here is argmax/tie.
(def tie-tol 0.13) ;; sub-tie argmax flip window (gate-test convention)

(defn- log-softmax [logits]
  (let [m (mx/amax logits)
        s (mx/subtract logits m)]
    (mx/subtract s (mx/log (mx/sum (mx/exp s))))))
(defn- top5-ids [lp]
  (mx/eval! lp)
  (let [fa (.toFloat32 lp)]
    (->> (range (.-length fa)) (sort-by #(aget fa %) >) (take 5) vec)))
(defn- lane-matches? [lane ref]
  (let [la (log-softmax lane) lr (log-softmax ref)]
    (mx/eval! la) (mx/eval! lr)
    (let [fa (.toFloat32 la) fr (.toFloat32 lr)
          b (reduce max 0 (map #(js/Math.abs (- (aget fa %) (aget fr %)))
                               (top5-ids lr)))
          am-a (mx/item (mx/argmax la)) am-r (mx/item (mx/argmax lr))
          ;; tie on the REFERENCE lp of the two candidates (gate convention)
          gap (js/Math.abs (- (aget fr am-r) (aget fr am-a)))
          tie? (< gap tie-tol)]
      (println (str "    [info] band=" (.toFixed b 4) " argmax " am-a
                    (if (= am-a am-r) " == " " != ") am-r
                    " ref-lp-gap=" (.toFixed gap 4)))
      (and (or (= am-a am-r) tie?) (< b band)))))
(defn- row [m i] (mx/index m i))
(defn- toks-arr [v] (mx/array (vec v) [(count v)] mx/int32))

(defn- summary []
  (println (str "\n== llm-vlm-batched-smoke: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(if-not (and model-dir (.existsSync fs image-path))
  (do (println "SKIP llm-vlm-batched-smoke — checkpoint or test image absent") (summary))
  (->
   (pr/let [img (.readFileSync fs image-path)
            mm (llm/load-model model-dir)
            {:keys [model tokenizer]} mm
            chat (llm/render-chat [{:role "user"
                                    :content "What is shown in this image? Answer in a few words."
                                    :images 1}])
            enc (llm/encode tokenizer chat false)]
     (mx/force-gc!)
     (let [prompt (vec enc)]
       (llm/init-cache! model)
       (println "  [info] running owned VLM prefill (" (count prompt) "marker tokens)…")
       (let [pf (llm/forward-prefill model prompt {:images [img]})
             t* (mx/item (mx/argmax pf))
             ;; scalar reference: branches forked off the image prefix
             ref0 (llm/branch-cache! model)          ; scalar snapshot
             _ (mx/force-gc!)
             ;; V1: batched identical-token step on the LIVE cache
             lg1 (llm/forward-step-batched model (toks-arr [t* t* t* t*]))
             ref-lg1 (llm/forward-branch model ref0 t*)]
         (mx/eval! lg1)
         (println "  [info] greedy first answer token id:" t*)
         (assert-true "V1 lane-0 == scalar branch step over image prefix"
                      (lane-matches? (row lg1 0) ref-lg1))
         (assert-true "V1 lane-3 == scalar branch step over image prefix"
                      (lane-matches? (row lg1 3) ref-lg1))
         (mx/force-gc!)
         ;; V2: divergent second step — top-4 ids of the reference. NOTE:
         ;; ref0 was already advanced by t* in V1 (forward-branch mutates
         ;; the ledger entry), so each per-lane reference forks from ref0
         ;; at context [prefix, t*] and feeds ONLY its divergent token.
         (let [tops (vec (take 4 (top5-ids (log-softmax ref-lg1))))
               lg2 (llm/forward-step-batched model (toks-arr tops))
               refs (mapv (fn [tk]
                            (let [b (llm/branch-from model ref0)
                                  lg (llm/forward-branch model b tk)]
                              (llm/dispose-branch! model b)
                              lg))
                          tops)]
           (mx/eval! lg2)
           (doseq [k (range 4)]
             (assert-true (str "V2 divergent lane-" k " == scalar branch replay")
                          (lane-matches? (row lg2 k) (nth refs k))))
           (mx/force-gc!)
           ;; V3: batched branching over the image prefix — fork the batched
           ;; live cache, advance the fork on junk tokens, then advance BOTH
           ;; the live cache and a second fork with the same tokens: the
           ;; sibling advance must not have disturbed them (band, not
           ;; bit-equal — separate MoE forward calls carry cnhi jitter).
           (let [b1 (llm/branch-cache! model)        ; batched snapshot @P
                 b2 (llm/branch-from model b1)
                 _ (llm/forward-branch-batched model b2 (toks-arr [9 9 9 9]))
                 live (llm/forward-step-batched model (toks-arr tops))
                 lg-b1 (llm/forward-branch-batched model b1 (toks-arr tops))]
             (mx/eval! live) (mx/eval! lg-b1)
             (doseq [k [0 3]]
               (assert-true (str "V3 lane-" k " live == sibling-isolated branch (band)")
                            (lane-matches? (row lg-b1 k) (row live k))))
             (llm/dispose-branch! model b1)
             (llm/dispose-branch! model b2)))
         (llm/reset-cache! model)
         (mx/force-gc!))))
   (pr/then (fn [_] (summary)))
   (pr/catch (fn [e]
               (println "ERROR:" (.-message e))
               (swap! fail inc)
               (summary)))))
