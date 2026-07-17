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

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_pi_provider_test.cljs"
  (:require [genmlx.llm.pi-provider :as pp]
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

(defn- summary []
  (println (str "\n== llm-pi-provider: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

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
                      (summary)))))))))
      (pr/catch (fn [e]
                  (println "ERROR:" (str e))
                  (swap! fail inc)
                  (summary)))))
