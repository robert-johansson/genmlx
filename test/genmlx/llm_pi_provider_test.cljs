;; @tier slow
(ns genmlx.llm-pi-provider-test
  "genmlx-djw6: the pi turn engine end-to-end on the hybrid 0.8b — the same
   #js API genmlx-host.ts drives, exercised exactly as the TS shim will:
   JSON messages/config in, per-token delta callbacks, final JSON out.

     P1 loadModel resolves with model info; newSession mints ids
     P2 a greedy turn completes: finishReason stop|length, promptTokens > 0,
        text == concatenated non-reasoning deltas, cachedTokens 0
     P3 turn 2 with the turn-1 reply appended DELTA-PREFILLS: cachedTokens
        covers the committed prefix (no cold replay) — THE L2 gate
     P4 same-turn determinism at temp 0: re-running turn 1 on a fresh
        session reproduces the text byte-for-byte
     P5 error taxonomy: unknown session rejects with finishReason error;
        dispose is idempotent
     P6 incremental detok: detok-step chained over the REAL tokenizer on
        multi-byte text reproduces the full decode byte-for-byte, with no
        replacement char ever leaking into a flushed piece
     P7 :logit-mask seam (genmlx-3g0t entry): a mask that pins everything
        but EOS forces an empty stop turn; clearing it restores generation
     P8 K-lane seam (genmlx-maww entry): bestOfK > 1 is a typed error

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_pi_provider_test.cljs"
  (:require [genmlx.llm.pi-provider :as pp]
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

(def model-dir
  (let [d (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")]
    (when (.existsSync fs (path/join d "tokenizer.json")) d)))

(def engine pp/engine)

(defn- run-turn
  "Drive engine.turnStream; resolves to {:final <parsed> :deltas [parsed...]}."
  [sid messages config]
  (let [deltas (atom [])]
    (-> (.turnStream engine sid
                     (js/JSON.stringify (clj->js messages))
                     (js/JSON.stringify (clj->js config))
                     (fn [dj] (swap! deltas conj (js->clj (js/JSON.parse dj) :keywordize-keys true))))
        (pr/then (fn [fj] {:final (js->clj (js/JSON.parse fj) :keywordize-keys true)
                           :deltas @deltas})))))

(def msgs-1
  [{:role "system" :content "You are a terse assistant."}
   {:role "user" :content "Reply with exactly one short English word."}])

(def cfg-greedy
  {:temperature 0 :maxNewTokens 24 :reasoningEffort "none"})

(defn- run-detok-chain
  "Feed token ids one at a time through pp/detok-step (the engine's
   incremental streamer); resolves to {:text concat-of-pieces
   :clean? no-piece-ever-held-a-replacement-char}."
  [decode-fn ids]
  (let [n (count ids)]
    (pr/loop [i 0 pending [] acc "" clean? true]
      (if (= i n)
        (if (empty? pending)
          {:text acc :clean? clean?}
          (pr/let [tail (decode-fn pending)]
            {:text (str acc tail) :clean? clean?}))
        (pr/let [{:keys [piece pending]} (pp/detok-step decode-fn pending (nth ids i))]
          (pr/recur (inc i) pending (str acc piece)
                    (and clean? (not (re-find #"�" piece)))))))))

(defn- summary []
  (println (str "\n== llm-pi-provider: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(defn- seam-tests
  "P6-P8: incremental detok + the 3g0t/maww extension seams. `info` is the
   loadModel result (carries :eosTokenId)."
  [info]
  (let [eos (.-eosTokenId info)]
    (pr/let [;; P6: real-tokenizer incremental detok on multi-byte text
             core (js/require "@genmlx/core")
             tok  (.fromPretrained (.-Qwen3Tokenizer core)
                                   (str model-dir "/tokenizer.json"))
             s    "Grüße, 🌍! 漢字→λ done."
             ids-u32 (.encode tok s false)
             ids  (vec (js/Array.from ids-u32))
             decode-fn (fn [toks]
                         (.decode tok (js/Uint32Array.from (into-array toks))))
             full (.decode tok ids-u32)
             {:keys [text clean?]} (run-detok-chain decode-fn ids)]
      (assert-true "P6: detok-step chain reproduces the full decode" (= full text))
      (assert-true "P6: no replacement char ever flushed mid-char" clean?)
      (assert-true "P6: the probe string really was multi-token multi-byte"
                   (> (count ids) 6))
      ;; P7: logit-mask pins everything but EOS -> empty stop turn
      (let [sid3 (.newSession engine "{}")
            mask (fn [logits _gen]
                   (let [vocab (first (mx/shape logits))]
                     (mx/where (mx/equal (mx/arange 0 vocab 1) (mx/scalar eos))
                               logits (mx/scalar -1e30))))]
        (pp/set-logit-mask! sid3 mask)
        (pr/let [{fm :final} (run-turn sid3 msgs-1 cfg-greedy)]
          (assert-true "P7: masked turn stops immediately (EOS forced)"
                       (= "stop" (:finishReason fm)))
          (assert-true "P7: masked turn emits no text" (= "" (:text fm)))
          (assert-true "P7: masked turn sampled exactly the EOS" (= 1 (:numTokens fm)))
          (pp/set-logit-mask! sid3 nil)
          (pr/let [{fu :final} (run-turn sid3
                                         (assoc-in msgs-1 [1 :content]
                                                   "Reply with exactly one short French word.")
                                         cfg-greedy)]
            (assert-true "P7: cleared mask generates again"
                         (and (contains? #{"stop" "length"} (:finishReason fu))
                              (seq (:text fu))))
            (assert-true "P7: set-logit-mask! on unknown session throws"
                         (try (pp/set-logit-mask! "nope" nil) false
                              (catch :default _ true)))
            (.dispose engine sid3)
            ;; P8: the K-lane seam refuses bestOfK > 1 with a typed error
            (let [sid4 (.newSession engine "{}")]
              (pr/let [{fk :final} (run-turn sid4 msgs-1 (assoc cfg-greedy :bestOfK 2))]
                (assert-true "P8: bestOfK 2 -> finishReason error" (= "error" (:finishReason fk)))
                (assert-true "P8: error names the maww seam"
                             (boolean (re-find #"maww" (str (:errorMessage fk)))))
                (.dispose engine sid4)))))))))

(if-not model-dir
  (do (println "SKIP llm-pi-provider — no qwen3.5-0.8b checkpoint") (summary))
  (-> (pr/let [info (.loadModel engine model-dir)]
        (assert-true "P1: loadModel resolves with the path"
                     (= model-dir (.-path info)))
        (let [sid (.newSession engine "{}")]
          (assert-true "P1: newSession mints an id" (string? sid))
          (pr/let [{:keys [final deltas]} (run-turn sid msgs-1 cfg-greedy)]
            (println "    turn-1 text:" (pr-str (:text final)))
            (assert-true "P2: finishReason stop|length"
                         (contains? #{"stop" "length"} (:finishReason final)))
            (assert-true "P2: promptTokens > 0" (pos? (:promptTokens final)))
            (assert-true "P2: cachedTokens 0 on the first turn" (zero? (:cachedTokens final)))
            (assert-true "P2: text == concat of non-reasoning deltas"
                         (= (:text final)
                            (apply str (map :text (remove :isReasoning deltas)))))
            (assert-true "P2: numTokens > 0" (pos? (:numTokens final)))
            ;; P3: append the reply + a new user message -> delta prefill
            (let [msgs-2 (conj msgs-1
                               {:role "assistant" :content (:text final)}
                               {:role "user" :content "Now reply with a different word."})]
              (pr/let [{f2 :final} (run-turn sid msgs-2 cfg-greedy)]
                (println "    turn-2 text:" (pr-str (:text f2))
                         "| cached" (:cachedTokens f2) "of" (:promptTokens f2))
                (assert-true "P3: turn 2 completes" (contains? #{"stop" "length"} (:finishReason f2)))
                (assert-true "P3: DELTA PREFILL — cachedTokens covers the turn-1 render"
                             (>= (:cachedTokens f2) (:promptTokens final)))
                (assert-true "P3: cached < prompt (the suffix was new)"
                             (< (:cachedTokens f2) (:promptTokens f2)))
                ;; P4: fresh session, same turn -> byte-identical greedy text
                (let [sid2 (.newSession engine "{}")]
                  (pr/let [{f3 :final} (run-turn sid2 msgs-1 cfg-greedy)]
                    (assert-true "P4: temp-0 rerun is byte-identical"
                                 (= (:text final) (:text f3)))
                    (.dispose engine sid2)
                    ;; P5: error taxonomy + idempotent dispose
                    (pr/let [{fe :final} (run-turn "nope" msgs-1 cfg-greedy)]
                      (assert-true "P5: unknown session -> finishReason error"
                                   (= "error" (:finishReason fe)))
                      (assert-true "P5: error names the session"
                                   (boolean (re-find #"unknown session" (str (:errorMessage fe)))))
                      (.dispose engine sid)
                      (.dispose engine sid)
                      (assert-true "P5: double dispose no-ops" true)
                      (pr/let [_ (seam-tests info)]
                        (summary))))))))))
      (pr/catch (fn [e]
                  (println "ERROR:" (str e))
                  (swap! fail inc)
                  (summary)))))
