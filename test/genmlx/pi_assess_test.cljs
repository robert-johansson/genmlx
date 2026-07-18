;; @tier slow
(ns genmlx.pi-assess-test
  "genmlx-opwh (L3-C1) Modules B+C on the 0.8b: forward-branch-scores (the
   one-forward teacher-forcing scorer) and pi-assess (render-parity session
   replay + the GFI face). The test ADMINISTERS a tiny two-turn session with
   its own greedy loop (backend primitives only), writes it as a pi-shaped
   session JSONL, reads it back through pi-session, and scores it — the
   full L3-C1 path end to end.

     A  fixture round trip: generated turns -> JSONL -> path->messages
     B  LAW: forward-branch-scores == per-token forward-branch reference
        (chunked AND single-slab), and the [0]-shape 1-token edge
     C  session-scores: finite negative per-turn logprobs, the delta-prefill
        walk (turn 2 :cached covers the turn-1 render), greedy sanity
        (the scored span decodes to the administered reply)
     D  LAW: turn-assess weight (p/assess through make-llm-gf) matches the
        session-scores logprob per turn
     E  independence: a single-turn slice scores identically to the walk

   Run: bun run --bun nbb test/genmlx/pi_assess_test.cljs (guarded on Thor)"
  (:require [clojure.string :as str]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.pi-assess :as pa]
            [genmlx.llm.pi-session :as ps]
            [genmlx.mlx :as mx]
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
(defn assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol)
      (do (swap! pass inc) (println "  PASS" label (str "(|Δ| " (.toFixed d 5) ")")))
      (do (swap! fail inc)
          (println "  FAIL" label "expected" expected "actual" actual
                   "tol" tol)))))

(def model-dir
  (let [d (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")]
    (when (.existsSync fs (path/join d "tokenizer.json")) d)))

(defn- summary []
  (println (str "\n== pi-assess: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(def sys-text "You are a terse assistant.")

(defn- render
  "applyChatTemplate over converted messages; resolves to a token vector."
  [tokenizer msgs add-gen?]
  (pr/let [r (.applyChatTemplate tokenizer (ps/messages->js msgs)
                                 add-gen? js/undefined false)]
    (vec (js/Array.from r))))

(defn- greedy-turn
  "Administer one greedy turn from prompt-ids on the MODEL cache; returns
   the generated ids (eos excluded)."
  [model prompt-ids max-new eos]
  (llm/init-cache! model)
  (try
    (loop [logits (llm/forward-prefill model prompt-ids), gen []]
      (let [tok (mx/item (mx/argmax logits))]
        (cond
          (== tok eos)            gen
          (>= (count gen) max-new) gen
          :else (recur (llm/forward-step model tok) (conj gen tok)))))
    (finally (llm/reset-cache! model))))

(defn- decode-toks [tokenizer toks]
  (llm/decode tokenizer (js/Uint32Array.from (into-array toks))))

(defn- jl [o] (js/JSON.stringify (clj->js o)))

(defn- session-jsonl
  "The administered turns as a pi-shaped session file (the shape the real
   agent appends — verified against live files by pi_session_test)."
  [text1 text2]
  (str/join "\n"
            [(jl {:type "session" :version 3 :id "s-test"
                  :timestamp "2026-07-18T00:00:00.000Z" :cwd "/tmp/proj"})
             (jl {:type "message" :id "u1" :parentId nil :timestamp "t"
                  :message {:role "user"
                            :content "Reply with exactly one short English word."}})
             (jl {:type "message" :id "a1" :parentId "u1" :timestamp "t"
                  :message {:role "assistant" :stopReason "stop"
                            :usage {:input 1 :output 1 :totalTokens 2}
                            :content [{:type "text" :text text1}]}})
             (jl {:type "message" :id "u2" :parentId "a1" :timestamp "t"
                  :message {:role "user"
                            :content "Now reply with a different word."}})
             (jl {:type "message" :id "a2" :parentId "u2" :timestamp "t"
                  :message {:role "assistant" :stopReason "stop"
                            :usage {:input 1 :output 1 :totalTokens 2}
                            :content [{:type "text" :text text2}]}})]))

(defn- step-reference-scores
  "Per-token reference scorer: fresh branch, one forward-branch per token,
   log-softmax + gather at each step. The O(n)-steps ground truth for LAW B."
  [model ids]
  (let [b (llm/owned-branch! model {:cache nil :offset 0})]
    (try
      (loop [i 1
             ;; seed via forward-branch-tokens (nil-cache safe), then step
             logits (llm/forward-branch-tokens model b [(first ids)])
             out []]
        (if (>= i (count ids))
          out
          (let [lp  (mx/log-softmax logits -1)
                s   (mx/item (mx/index lp (nth ids i)))]
            (recur (inc i)
                   (llm/forward-branch model b (nth ids i))
                   (conj out s)))))
      (finally (llm/dispose-branch! model b)))))

(if-not model-dir
  (do (println "SKIP pi-assess — no qwen3.5-0.8b checkpoint") (summary))
  (->
   (pr/let [mm (llm/load-model model-dir {:cljs-forward? true})]
     (let [{:keys [model tokenizer]} mm
           eos  (llm/eos-token-id tokenizer)
           u1   {:role "user" :content "Reply with exactly one short English word."}
           u2   {:role "user" :content "Now reply with a different word."}
           sysm {:role "system" :content sys-text}]
       (pr/let [;; ---- administer two greedy turns -------------------------
                p1    (render tokenizer [sysm u1] true)
                gen1  (pr/resolved (greedy-turn model p1 24 eos))
                text1 (decode-toks tokenizer gen1)
                text1 (pr/resolved (str/trim text1))
                p2    (render tokenizer [sysm u1 {:role "assistant" :content text1} u2] true)
                gen2  (pr/resolved (greedy-turn model p2 24 eos))
                text2 (decode-toks tokenizer gen2)
                text2 (pr/resolved (str/trim text2))]
         (println "  administered replies:" (pr-str text1) "/" (pr-str text2))
         ;; ---- A: fixture round trip --------------------------------------
         (let [dir  (fs/mkdtempSync (path/join (os/tmpdir) "pi-assess-"))
               file (path/join dir "2026-07-18T00-00-00-000Z_s-test.jsonl")]
           (fs/writeFileSync file (session-jsonl text1 text2))
           (let [session (ps/read-session file)
                 msgs    (ps/path->messages (ps/leaf-path session)
                                            {:system-prompt sys-text})]
             (assert-true "A: round-tripped messages match the administration"
                          (= msgs [sysm u1
                                   {:role "assistant" :content text1} u2
                                   {:role "assistant" :content text2}]))
             (assert-true "A: assistant indices" (= [2 4] (ps/assistant-indices msgs)))
             ;; ---- B: the scorer law ---------------------------------------
             (pr/let [r1 (pa/render-turn tokenizer msgs 2 {})]
               (let [f1     (:full r1)
                     ref    (step-reference-scores model f1)
                     b1     (llm/owned-branch! model {:cache nil :offset 0})
                     sc-ch  (vec (mx/->clj (llm/forward-branch-scores
                                            model b1 f1 {:chunk 7})))
                     _      (llm/dispose-branch! model b1)
                     b2     (llm/owned-branch! model {:cache nil :offset 0})
                     sc-one (vec (mx/->clj (llm/forward-branch-scores
                                            model b2 f1 {:chunk 0})))
                     _      (llm/dispose-branch! model b2)
                     dmax   (fn [a b] (apply max (map #(js/Math.abs (- %1 %2)) a b)))]
                 (assert-true "B: score count is n-1"
                              (= (count ref) (count sc-ch) (count sc-one)
                                 (dec (count f1))))
                 (println "    max |chunked - stepwise| ="
                          (.toFixed (dmax sc-ch ref) 6)
                          "| max |slab - stepwise| ="
                          (.toFixed (dmax sc-one ref) 6))
                 ;; tolerance 1.0/token: the slab-prefill and step-decode
                 ;; graphs legitimately differ at bf16 precision (GDN chunk
                 ;; scan vs step recurrence); an indexing off-by-one would
                 ;; score the WRONG tokens and diverge by many nats. The
                 ;; exact law is D below (assess == walk, the GFI contract).
                 (assert-true "B: chunked matches the per-token reference (<=1.0/token; off-by-one = nats)"
                              (<= (dmax sc-ch ref) 1.0))
                 (assert-true "B: single slab matches the per-token reference (<=1.0/token)"
                              (<= (dmax sc-one ref) 1.0))
                 (assert-close "B: summed logprob chunked vs stepwise"
                               (reduce + 0.0 ref) (reduce + 0.0 sc-ch) 2.0)
                 (let [b4 (llm/owned-branch! model {:cache nil :offset 0})
                       again (vec (mx/->clj (llm/forward-branch-scores
                                             model b4 f1 {:chunk 7})))]
                   (assert-true "B: chunked rerun is bit-identical (determinism)"
                                (= sc-ch again))
                   (llm/dispose-branch! model b4))
                 (let [b3 (llm/owned-branch! model {:cache nil :offset 0})
                       z  (llm/forward-branch-scores model b3 [(first f1)])]
                   (assert-true "B: 1-token input -> [0]-shaped scores"
                                (= [0] (vec (mx/shape z))))
                   (llm/dispose-branch! model b3))
                 ;; ---- C: session-scores over the real fixture -------------
                 (pr/let [scores (pa/session-scores mm msgs {})]
                   (let [[t1 t2] scores]
                     (assert-true "C: one entry per assistant turn" (= 2 (count scores)))
                     (assert-true "C: logprobs finite and negative"
                                  (every? #(and (js/isFinite (:logprob %))
                                                (neg? (:logprob %))) scores))
                     (assert-true "C: n-tokens positive, per-token lengths agree"
                                  (every? #(and (pos? (:n-tokens %))
                                                (= (:n-tokens %)
                                                   (count (:per-token %)))) scores))
                     (assert-true "C: render parity held on both turns"
                                  (every? :parity? scores))
                     (println "    turn logprobs:" (mapv :logprob scores)
                              "| cached:" (mapv :cached scores))
                     (assert-true "C: turn-2 delta-prefills over the turn-1 render"
                                  (and (pos? (:cached t2))
                                       (> (:cached t2) (:n-tokens t1))))
                     (pr/let [span-text (decode-toks tokenizer (:tokens t1))]
                       (assert-true "C: scored span decodes to the administered reply"
                                    (str/includes? span-text text1))
                       ;; ---- D: the GFI law --------------------------------
                       (pr/let [ta1 (pa/turn-assess mm msgs 2 {})
                                ta2 (pa/turn-assess mm msgs 4 {})]
                         (assert-close "D: turn-1 assess weight == walked logprob"
                                       (:logprob t1) (mx/item (:weight ta1)) 0.1)
                         (assert-close "D: turn-2 assess weight == walked logprob"
                                       (:logprob t2) (mx/item (:weight ta2)) 0.1)
                         (assert-true "D: assess scored the same tokens"
                                      (= (:tokens ta1) (:tokens t1)))
                         ;; ---- E: walk == independent single-turn slice ----
                         (pr/let [solo (pa/session-scores
                                        mm (subvec msgs 0 3) {})]
                           (assert-close "E: sliced turn-1 == walked turn-1"
                                         (:logprob t1) (:logprob (first solo)) 1e-3)
                           (assert-true "E: images refused (typed error)"
                                        (try (pa/session-scores
                                              mm [{:role "user" :content "x"
                                                   :images [(js/Uint8Array. 1)]}])
                                             false
                                             (catch :default e
                                               (= :images-unsupported
                                                  (:genmlx/error (ex-data e))))))
                           (summary)))))))))))))
   (pr/catch (fn [e]
               (println "ERROR:" (str e))
               (when-let [d (ex-data e)] (println "  ex-data:" (pr-str d)))
               (swap! fail inc)
               (summary)))))
