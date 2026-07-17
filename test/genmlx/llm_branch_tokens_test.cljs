;; @tier slow
(ns genmlx.llm-branch-tokens-test
  "genmlx-djw6: forward-branch-tokens — the multi-token branch advance the
   pi provider's delta-prefill rides (a tool-loop turn extends a session's
   committed prefix by its suffix; no cold replay). Gates, on the dense
   0.6b (per-token degraded path) AND the hybrid qwen3.5 0.8b (true
   multi-token continuation, prior-width mask):

     T1 fresh-branch advance == forward-prefill (argmax exact + top-5 band)
     T2 split advance (prefix then suffix) == one-shot advance on a sibling
        branch — THE delta-prefill correctness property
     T3 chunked == unchunked (memory boundary, not an approximation)
     T4 multi-token advance == per-token forward-branch replay
     T5 guards: empty token-ids throws; ledger offset advances by n

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_branch_tokens_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
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

(defn- model-dir [& names]
  (let [cands (map #(path/join (os/homedir) ".cache" "models" %) names)]
    (first (filter #(.existsSync fs (path/join % "tokenizer.json")) cands))))

(def dense-dir  (model-dir "qwen3-0.6b-mlx-bf16" "qwen3-0.6b"))
(def hybrid-dir (model-dir "qwen3.5-0.8b-mlx-bf16"))

(defn- mat [a] (mx/materialize! a) a)
(defn- amax* [a] (mx/item (mx/argmax a)))
(defn- topk-band [a b k]
  (mx/eval! a) (mx/eval! b)
  (let [fa (.toFloat32 a) fb (.toFloat32 b)
        ids (->> (range (.-length fa)) (sort-by #(aget fa %) >) (take k))]
    (reduce max 0 (map #(js/Math.abs (- (aget fa %) (aget fb %))) ids))))

(defn- fresh-branch!
  "A branch rooted at the EMPTY prefix (init-cache! first so branch-cache!
   forks a clean cell)."
  [model]
  (llm/init-cache! model)
  (llm/branch-cache! model))

(defn- run-suite [dir]
  (pr/let [mm (llm/load-model dir)
           {:keys [model tokenizer]} mm
           enc (llm/encode tokenizer "The capital of France is")]
    (let [prompt (vec enc)
          n      (count prompt)
          split  (max 1 (- n 2))
          p1     (subvec prompt 0 split)
          p2     (subvec prompt split)]
      (println "\n== forward-branch-tokens on" dir "(" n "tokens )")

      ;; T1: fresh-branch advance == prefill
      (llm/init-cache! model)
      (let [l-pref (mat (llm/forward-prefill model prompt))
            b      (fresh-branch! model)
            l-bt   (mat (llm/forward-branch-tokens model b prompt))
            band   (topk-band l-pref l-bt 5)]
        (assert-true "T1: argmax matches prefill" (= (amax* l-pref) (amax* l-bt)))
        (assert-true (str "T1: top-5 band " (.toExponential band 2) " < 0.3") (< band 0.3))
        (llm/dispose-branch! model b))

      ;; T2: split advance == one-shot advance (delta-prefill correctness)
      (let [b-full  (fresh-branch! model)
            b-split (fresh-branch! model)
            l-full  (mat (llm/forward-branch-tokens model b-full prompt))
            _       (llm/forward-branch-tokens model b-split p1)
            l-split (mat (llm/forward-branch-tokens model b-split p2))
            band    (topk-band l-full l-split 5)]
        (assert-true "T2: split argmax == one-shot argmax" (= (amax* l-full) (amax* l-split)))
        (assert-true (str "T2: top-5 band " (.toExponential band 2) " < 0.3") (< band 0.3))
        ;; T5b: ledger offsets agree after both routes
        (let [off-full  (:offset (llm/owned-branch-state model b-full))
              off-split (:offset (llm/owned-branch-state model b-split))]
          (assert-true (str "T5: both routes end at offset " n) (= n off-full off-split)))
        (llm/dispose-branch! model b-full)
        (llm/dispose-branch! model b-split))

      ;; T3: chunked == unchunked
      (let [b-slab  (fresh-branch! model)
            b-chunk (fresh-branch! model)
            l-slab  (mat (llm/forward-branch-tokens model b-slab prompt))
            l-chunk (mat (llm/forward-branch-tokens model b-chunk prompt {:chunk 3}))
            band    (topk-band l-slab l-chunk 5)]
        (assert-true "T3: chunked argmax == slab argmax" (= (amax* l-slab) (amax* l-chunk)))
        (assert-true (str "T3: top-5 band " (.toExponential band 2) " < 0.3") (< band 0.3))
        (llm/dispose-branch! model b-slab)
        (llm/dispose-branch! model b-chunk))

      ;; T4: multi-token advance == per-token forward-branch replay
      (let [b-multi (fresh-branch! model)
            b-step  (fresh-branch! model)
            l-multi (mat (llm/forward-branch-tokens model b-multi prompt))
            l-step  (mat (reduce (fn [_ tok] (llm/forward-branch model b-step tok))
                                 nil prompt))
            band    (topk-band l-multi l-step 5)]
        (assert-true "T4: argmax == per-token replay" (= (amax* l-multi) (amax* l-step)))
        (assert-true (str "T4: top-5 band " (.toExponential band 2) " < 0.3") (< band 0.3))
        (llm/dispose-branch! model b-multi)
        (llm/dispose-branch! model b-step))

      ;; T5: guards
      (let [b (fresh-branch! model)]
        (assert-true "T5: empty token-ids throws"
                     (try (llm/forward-branch-tokens model b []) false
                          (catch :default e
                            (= :empty-branch-tokens (:genmlx/error (ex-data e))))))
        (llm/dispose-branch! model b))
      (llm/reset-cache! model))))

(defn- summary []
  (println (str "\n== llm-branch-tokens: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(let [dirs (filter some? [dense-dir hybrid-dir])]
  (if (empty? dirs)
    (do (println "SKIP llm-branch-tokens — no local model checkpoints") (summary))
    (-> (reduce (fn [acc dir] (pr/then acc (fn [_] (run-suite dir))))
                (pr/resolved nil) dirs)
        (pr/then (fn [_] (summary)))
        (pr/catch (fn [e]
                    (println "ERROR:" (str e))
                    (swap! fail inc)
                    (summary))))))
