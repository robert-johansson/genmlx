;; @tier slow
(ns genmlx.llm.batched-branch-test
  "genmlx-lo6e D1: batched-lane branching + lane-axis resample on the OWNED
   forward. The 9uyg batched cache is a persistent value, so a batched fork
   is reference-holding (like the scalar ledger, genmlx-7f93) and the
   token-SMC resample is ONE gather per cache array over the lane axis
   (fwd/resample-cache-lanes) instead of per-branch fork/replay loops.

   Gates (dense-deterministic models -> exact bands):
     G1 batched fork isolation: advancing a batched branch leaves the live
        batched cache untouched (bit-equal next-step logits vs no-fork
        control); the branch's per-lane logits match scalar replay
     G2 resample-lanes! all<-lane-a then step == scalar replay of ancestor
        a (rows mutually bit-identical + match scalar reference)
     G3 identity resample is a no-op (bit-equal logits vs control)
     G4 fork-of-fork + lane shrink via resample-branch-lanes! (K'=2)
     G5 guards: scalar step on batched cache/branch refused; resample on a
        scalar cache refused; batch-width change refused

   Runs on the qwen3 dense 0.6b AND the qwen3.5 hybrid 0.8b (the GDN
   {:conv :rec} cache entries take a different resample path than {:k :v}).
   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_batched_branch_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.forward :as fwd]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def dense-dir
  (let [cands [(path/join (os/homedir) ".cache" "models" "qwen3-0.6b-mlx-bf16")
               (path/join (os/homedir) ".cache" "models" "qwen3-0.6b")]]
    (first (filter #(.existsSync fs (path/join % "tokenizer.json")) cands))))
(def hybrid-dir
  (let [d (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")]
    (when (.existsSync fs (path/join d "tokenizer.json")) d)))

(defn- max-abs-diff [a b]
  (mx/eval! a) (mx/eval! b)
  (let [fa (.toFloat32 a) fb (.toFloat32 b)]
    (reduce max 0 (map #(js/Math.abs (- (aget fa %) (aget fb %)))
                       (range (.-length fa))))))
(defn- log-softmax [logits]
  (let [m (mx/amax logits)
        s (mx/subtract logits m)]
    (mx/subtract s (mx/log (mx/sum (mx/exp s))))))
(defn- top5-ids [logits]
  (mx/eval! logits)
  (let [fa (.toFloat32 logits)]
    (->> (range (.-length fa)) (sort-by #(aget fa %) >) (take 5) vec)))
(defn- lane-matches?
  "Batched-lane vs scalar-replay agreement, the 9uyg gate-test methodology:
   compare LOG-SOFTMAX values; argmax exact OR a near-tie (the two paths'
   lp gap under the cross-batch-shape quantum); top-5 logprob band < 0.25
   over the reference's top-5 ids (full-vocab max-diff is dominated by bf16
   wobble on tail tokens across batch shapes)."
  [lane ref]
  (let [la (log-softmax lane) lr (log-softmax ref)]
    (mx/eval! la) (mx/eval! lr)
    (let [fa (.toFloat32 la) fr (.toFloat32 lr)
          band (reduce max 0 (map #(js/Math.abs (- (aget fa %) (aget fr %)))
                                  (top5-ids lr)))
          am-a (mx/item (mx/argmax la)) am-r (mx/item (mx/argmax lr))
          tie? (< (js/Math.abs (- (aget fa am-r) (aget fa am-a))) 0.1)]
      (and (or (= am-a am-r) tie?)
           (< band 0.25)))))
(defn- row [m i] (mx/index m i))
(defn- argmax-id [logits] (mx/item (mx/argmax logits)))
(defn- toks-arr [v] (mx/array (vec v) [(count v)] mx/int32))

(defn- scalar-replay
  "Scalar reference: prefill + the given token seq; returns final logits [vocab]."
  [model prompt-ids toks]
  (llm/init-cache! model)
  (let [_ (llm/forward-prefill model prompt-ids)]
    (reduce (fn [_ t] (llm/forward-step model t)) nil toks)))

(defn- run-family [label dir]
  (println (str "\n=== " label " (" dir ") ==="))
  (pr/let [m (llm/load-model dir)]
    (let [{:keys [model tokenizer]} m]
      (pr/let [ids-raw (llm/encode tokenizer "The quick brown fox" false)]
        (let [prompt (vec ids-raw)
              K 4
              toks1 [100 200 300 400]          ; distinct lane histories
              toks2 [111 222 333 444]
              tnext 55]

          ;; ---- G1: batched fork isolation + branch vs scalar replay ----
          (println "\n-- G1 batched fork isolation --")
          ;; control: no fork
          (llm/init-cache! model)
          (llm/forward-prefill model prompt)
          (llm/forward-step-batched model (toks-arr toks1))
          (let [control (llm/forward-step-batched model (toks-arr toks2))]
            (mx/eval! control)
            ;; with a fork advanced in between
            (llm/init-cache! model)
            (llm/forward-prefill model prompt)
            (llm/forward-step-batched model (toks-arr toks1))
            (let [id (llm/branch-cache! model)
                  br-lg (llm/forward-branch-batched model id (toks-arr [7 8 9 10]))
                  live  (llm/forward-step-batched model (toks-arr toks2))]
              (mx/eval! br-lg) (mx/eval! live)
              (assert-true "live cache unaffected by branch advance (bit-equal)"
                           (zero? (max-abs-diff control live)))
              ;; branch lanes vs scalar replay
              (assert-true "branch lane-1 == scalar replay (argmax+top5+band)"
                           (lane-matches? (row br-lg 1)
                                          (scalar-replay model prompt [(nth toks1 1) 8])))
              (llm/dispose-branch! model id)))

          ;; ---- G2: resample all<-lane-2 == scalar replay of ancestor 2 ----
          (println "\n-- G2 resample-lanes! ancestry --")
          (llm/init-cache! model)
          (llm/forward-prefill model prompt)
          (llm/forward-step-batched model (toks-arr toks1))
          (llm/resample-lanes! model [2 2 2 2])
          (let [lg  (llm/forward-step-batched model (toks-arr [tnext tnext tnext tnext]))
                ref (scalar-replay model prompt [(nth toks1 2) tnext])]
            (mx/eval! lg)
            (assert-true "rows mutually bit-identical after all<-2 resample"
                         (and (zero? (max-abs-diff (row lg 0) (row lg 1)))
                              (zero? (max-abs-diff (row lg 0) (row lg 3)))))
            (assert-true "resampled lane == scalar replay of ancestor (argmax+top5+band)"
                         (lane-matches? (row lg 0) ref))
            (assert-true "argmax exact vs scalar replay"
                         (= (argmax-id (row lg 0)) (argmax-id ref))))

          ;; ---- G3: identity resample is a no-op ----
          (println "\n-- G3 identity resample --")
          (llm/init-cache! model)
          (llm/forward-prefill model prompt)
          (llm/forward-step-batched model (toks-arr toks1))
          (let [control (llm/forward-step-batched model (toks-arr toks2))]
            (mx/eval! control)
            (llm/init-cache! model)
            (llm/forward-prefill model prompt)
            (llm/forward-step-batched model (toks-arr toks1))
            (llm/resample-lanes! model [0 1 2 3])
            (let [lg (llm/forward-step-batched model (toks-arr toks2))]
              (assert-true "identity resample -> bit-equal logits"
                           (zero? (max-abs-diff control lg)))))

          ;; ---- G4: fork-of-fork + lane shrink on a branch ----
          (println "\n-- G4 fork-of-fork + shrink --")
          (llm/init-cache! model)
          (llm/forward-prefill model prompt)
          (llm/forward-step-batched model (toks-arr toks1))
          (let [id  (llm/branch-cache! model)
                id2 (llm/branch-from model id)]
            (llm/resample-branch-lanes! model id2 [0 2])
            (let [lg (llm/forward-branch-batched model id2 (toks-arr [tnext tnext]))]
              (assert-true "shrunk branch logits shape [2 vocab]"
                           (= 2 (first (mx/shape lg))))
              (assert-true "shrunk lane 0 == scalar replay of ancestor 0 (argmax+top5+band)"
                           (lane-matches? (row lg 0)
                                          (scalar-replay model prompt [(nth toks1 0) tnext])))
              (assert-true "shrunk lane 1 == scalar replay of ancestor 2 (argmax+top5+band)"
                           (lane-matches? (row lg 1)
                                          (scalar-replay model prompt [(nth toks1 2) tnext]))))
            ;; width-change guard on the (still K=4) parent branch
            (assert-true "batch-width change on branch refused"
                         (try (llm/forward-branch-batched model id (toks-arr [1 2]))
                              false
                              (catch :default e
                                (= :batch-width-changed (:genmlx/error (ex-data e))))))
            (llm/dispose-branch! model id)
            (llm/dispose-branch! model id2))

          ;; ---- G5: guards ----
          (println "\n-- G5 guards --")
          (llm/init-cache! model)
          (llm/forward-prefill model prompt)
          (assert-true "resample on scalar cache refused"
                       (try (llm/resample-lanes! model [0])
                            false
                            (catch :default e
                              (= :resample-scalar-cache (:genmlx/error (ex-data e))))))
          (llm/forward-step-batched model (toks-arr toks1))
          (assert-true "scalar forward-step on batched cache refused"
                       (try (llm/forward-step model 5)
                            false
                            (catch :default e
                              (= :batched-cache-scalar-step (:genmlx/error (ex-data e))))))
          (let [id (llm/branch-cache! model)]
            (assert-true "scalar forward-branch on batched branch refused"
                         (try (llm/forward-branch model id 5)
                              false
                              (catch :default e
                                (= :batched-branch-scalar-step (:genmlx/error (ex-data e))))))
            (llm/dispose-branch! model id))
          (llm/reset-cache! model)
          (mx/force-gc!)
          true)))))

(defn- summary []
  (println (str "\n== llm-batched-branch: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(if-not (or dense-dir hybrid-dir)
  (println "SKIP llm-batched-branch — no local model")
  (-> (pr/do
        (if dense-dir (run-family "dense qwen3 0.6b" dense-dir) (pr/resolved nil))
        (if hybrid-dir (run-family "hybrid qwen3.5 0.8b" hybrid-dir) (pr/resolved nil)))
      (pr/then (fn [_] (summary)))
      (pr/catch (fn [e]
                  (println "ERROR:" (.-message e))
                  (swap! fail inc)
                  (summary)))))
