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
     P8 K-lane lockstep parity (genmlx-maww): a temp-0 bestOfK turn (no
        verifier) reproduces the scalar greedy text through the batched
        forward — and the K commit (prompt only) delta-prefills on turn 2
     P11 verifier callback seam (genmlx-maww): called once with all K
        candidates; winner/scores protocols; throw/timeout fallback to
        candidate 0; thinking/grammar composition guards
     P9 images (genmlx-5aah): an imageRefs turn runs the owned VLM prefill
        (marker tokens in the render; the model SEES the image — names the
        fixture's color at temp 0), a follow-up text turn delta-prefills
        over the image-conditioned branch, and a temp-0 rerun on a fresh
        session is byte-identical
     P10 per-argument grammar (genmlx-3g0t): a JSON-Schema pattern on a
        tool parameter compiles the toolset into the tool-call DFA — the
        constrained arm's argument MUST match where the unconstrained
        control overflows; hot-temperature emissions never produce an
        unparseable call; x-genmlx-grammar \"cljs\" (the reader leg) emits
        exactly one complete delimiter-opened CLJS form per argument

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_pi_provider_test.cljs"
  (:require [genmlx.codegen.eval :as ceval]
            [genmlx.llm.pi-provider :as pp]
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
  "Drive engine.turnStream; resolves to {:final <parsed> :deltas [parsed...]}.
   4-arity passes an images array (the 5aah non-JSON leg); 5-arity adds the
   maww verifier callback."
  ([sid messages config] (run-turn sid messages config nil))
  ([sid messages config images] (run-turn sid messages config images nil))
  ([sid messages config images verifier]
   (let [deltas (atom [])]
     (-> (.turnStream engine sid
                      (js/JSON.stringify (clj->js messages))
                      (js/JSON.stringify (clj->js config))
                      (fn [dj] (swap! deltas conj (js->clj (js/JSON.parse dj) :keywordize-keys true)))
                      (when images (into-array images))
                      verifier)
         (pr/then (fn [fj] {:final (js->clj (js/JSON.parse fj) :keywordize-keys true)
                            :deltas @deltas}))))))

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

(declare image-tests grammar-tests k-verifier-tests)

(defn- point-tool
  "Native ToolDefinition (properties as a JSON STRING, the wire shape
   toolsToDefinitions produces) — `props` = the properties map."
  [props]
  {:type "function"
   :function {:name "set_point"
              :description "Set a 2D point on the board."
              :parameters {:type "object"
                           :properties (js/JSON.stringify (clj->js props))
                           :required ["xy"]}}})

(def xy-plain {:xy {:type "string" :description "coordinates ROW,COL"}})
(def xy-pattern
  (assoc-in xy-plain [:xy :pattern] "-?[0-9]{1,3},-?[0-9]{1,3}"))

(def msgs-point
  [{:role "system" :content "You are a function-calling assistant."}
   {:role "user" :content "Use the set_point tool to set the point at row 999999, column 888888."}])

(def xy-re #"-?\d{1,3},-?\d{1,3}")

(defn- seam-tests
  "P6-P9: incremental detok + the 3g0t/maww extension seams + the 5aah
   image path. `info` is the loadModel result (carries :eosTokenId).
   Ends by calling image-tests -> summary."
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
            ;; P8: K-lane LOCKSTEP PARITY — temp-0 bestOfK reproduces the
            ;; scalar greedy text (batched forward == scalar forward through
            ;; the whole stack), and the prompt-only K commit delta-prefills
            (let [sidS (.newSession engine "{}")
                  sid4 (.newSession engine "{}")]
              (pr/let [{fs :final} (run-turn sidS msgs-1 cfg-greedy)
                       {fk :final} (run-turn sid4 msgs-1 (assoc cfg-greedy :bestOfK 3))]
                (.dispose engine sidS)
                (println "    K-turn text:" (pr-str (:text fk))
                         "| winner" (:winnerIndex fk) "of" (:candidateCount fk))
                (assert-true "P8: temp-0 K-turn text == scalar greedy text"
                             (= (:text fk) (:text fs)))
                (assert-true "P8: no-verifier fallback = candidate 0, K candidates generated"
                             (and (= 0 (:winnerIndex fk)) (= 3 (:candidateCount fk))
                                  (= 3 (:bestOfK fk))))
                (let [msgs-k2 (conj msgs-1
                                    {:role "assistant" :content (:text fk)}
                                    {:role "user" :content "Now reply with a different word."})]
                  (pr/let [{fk2 :final} (run-turn sid4 msgs-k2 cfg-greedy)]
                    (assert-true "P8: turn after a K-turn delta-prefills the whole prompt"
                                 (= (:cachedTokens fk2) (:promptTokens fk)))
                    (.dispose engine sid4)
                    (image-tests info)))))))))))

(def msgs-img
  [{:role "user"
    :content "What is the dominant color of this image? Reply with exactly one English word."
    :imageRefs [0]}])

(defn- image-tests
  "P9: the 5aah image path on the 0.8b VLM snapshot.

   The delta-prefill leg uses the mask seam to force an EMPTY turn-1 reply:
   the chat template TRIMS assistant content on re-render, so a whitespace-
   led sampled reply legitimately rebuilds (v1-parity accounting — same
   trim, same rebuild). An empty reply re-renders exactly, making the
   append property deterministic — and the follow-up answer can then only
   come from the image-conditioned KV BRANCH (the suffix prefill is pure
   text), which is the strongest form of the 'session remembers the image'
   acceptance."
  [info]
  (let [png (fs/readFileSync "test/genmlx/assets/red-64.png")
        eos (.-eosTokenId info)
        sidA (.newSession engine "{}")]
    (pr/let [{fi :final} (run-turn sidA msgs-img (assoc cfg-greedy :maxNewTokens 8) [png])]
      (println "    image-turn text:" (pr-str (:text fi)))
      (assert-true "P9: image turn completes"
                   (contains? #{"stop" "length"} (:finishReason fi)))
      (assert-true "P9: image turn is a full (uncached) VLM prefill"
                   (zero? (:cachedTokens fi)))
      (assert-true "P9: the model SEES the image (names the color)"
                   (boolean (re-find #"(?i)red" (str (:text fi)))))
      ;; temp-0 determinism across the VLM prefill
      (let [sidB (.newSession engine "{}")]
        (pr/let [{fb :final} (run-turn sidB msgs-img (assoc cfg-greedy :maxNewTokens 8) [png])]
          (assert-true "P9: temp-0 image-turn rerun is byte-identical"
                       (= (:text fi) (:text fb)))
          (.dispose engine sidB)
          (.dispose engine sidA)
          ;; delta-prefill leg: masked-EOS image turn -> empty reply ->
          ;; the extension property holds deterministically
          (let [sidC (.newSession engine "{}")
                mask (fn [logits _gen]
                       (let [vocab (first (mx/shape logits))]
                         (mx/where (mx/equal (mx/arange 0 vocab 1) (mx/scalar eos))
                                   logits (mx/scalar -1e30))))]
            (pp/set-logit-mask! sidC mask)
            (pr/let [{fc :final} (run-turn sidC msgs-img cfg-greedy [png])]
              (assert-true "P9: masked image turn commits the render only"
                           (and (= "stop" (:finishReason fc)) (= "" (:text fc))))
              (pp/set-logit-mask! sidC nil)
              (let [msgs-2 (conj msgs-img
                                 {:role "assistant" :content ""}
                                 {:role "user"
                                  :content "What color was the image? Reply with exactly one English word."})]
                (pr/let [{f2 :final} (run-turn sidC msgs-2 cfg-greedy [png])]
                  (println "    image-memory text:" (pr-str (:text f2))
                           "| cached" (:cachedTokens f2) "of" (:promptTokens f2))
                  (assert-true "P9: DELTA PREFILL over the image prefix (cached >= turn-1 render)"
                               (>= (:cachedTokens f2) (:promptTokens fc)))
                  (assert-true "P9: the CACHED image branch still answers the color"
                               (boolean (re-find #"(?i)red" (str (:text f2)))))
                  (.dispose engine sidC)
                  (grammar-tests))))))))))

(defn- stress-emissions
  "N hot-temperature turns, one fresh session each; resolves to the vector
   of {:calls :errors} per turn."
  [cfg n]
  (pr/loop [i 0 acc []]
    (if (= i n)
      acc
      (let [sid (.newSession engine "{}")]
        (pr/let [{f :final} (run-turn sid msgs-point cfg)]
          (.dispose engine sid)
          (pr/recur (inc i) (conj acc {:calls (:toolCalls f)
                                       :errors (:toolCallErrors f)})))))))

(defn- grammar-tests
  "P10: the 3g0t per-argument grammar leg on the 0.8b. The 0.8b at temp 0
   answers this prompt in prose without calling, so both arms run HOT
   (temp 1.0): the unconstrained control faithfully emits the requested
   off-pattern 999999 coordinates, the constrained arm CANNOT — the pair
   proves the mask engaged, not that the model behaved."
  []
  ;; maxNewTokens 192: at 64 a late-opening block can be truncated mid-tag
  ;; by the budget — a sampling artifact, not a grammar failure (verified:
  ;; 100 emissions at 192 = zero parse errors; 5 truncations at 64)
  (let [cfg-ctl (assoc cfg-greedy :temperature 1.0 :maxNewTokens 192
                       :tools [(point-tool xy-plain)])
        cfg-pat (assoc cfg-greedy :temperature 1.0 :maxNewTokens 192
                       :tools [(point-tool xy-pattern)])]
    (pr/let [ctl-runs (stress-emissions cfg-ctl 6)
             pat-runs (stress-emissions cfg-pat 6)]
      (let [ctl-xys   (keep #(get-in % [:arguments :xy]) (mapcat :calls ctl-runs))
            pat-calls (mapcat :calls pat-runs)
            pat-xys   (keep #(get-in % [:arguments :xy]) pat-calls)]
        (println "    control xys:    " (pr-str (vec ctl-xys)))
        (println "    constrained xys:" (pr-str (vec pat-xys)))
        (assert-true "P10: unconstrained control emits the requested OFF-pattern coords"
                     (boolean (some #(not (re-matches xy-re %)) ctl-xys)))
        (assert-true "P10: constrained turns produced calls (non-vacuous)"
                     (pos? (count pat-xys)))
        (assert-true "P10: zero unparseable calls under the grammar"
                     (every? #(empty? (:errors %)) pat-runs))
        (assert-true "P10: every constrained xy stays on-pattern"
                     (every? #(re-matches xy-re %) pat-xys))
        ;; reader-level :cljs leg (genmlx-3g0t): the argument is exactly one
        ;; delimiter-opened CLJS form, enforced byte-granularly at hot temp
        (let [filter-tool
              {:type "function"
               :function {:name "set_filter"
                          :description "Install a scene filter predicate."
                          :parameters {:type "object"
                                       :properties
                                       (js/JSON.stringify
                                        (clj->js {:code {:type "string"
                                                         :description "A ClojureScript predicate like (fn [scene] ...)"
                                                         :x-genmlx-grammar "cljs"}}))
                                       :required ["code"]}}}
              msgs-f [{:role "system" :content "You are a terse assistant."}
                      {:role "user" :content "Call the set_filter tool with a ClojureScript predicate that keeps scenes whose :x is under 10."}]
              cfg-cljs (assoc cfg-greedy :temperature 1.0 :maxNewTokens 320
                              :tools [filter-tool])]
          (pr/let [runs (pr/loop [i 0 acc []]
                          (if (= i 4)
                            acc
                            (let [sid (.newSession engine "{}")]
                              (pr/let [{f :final} (run-turn sid msgs-f cfg-cljs)]
                                (.dispose engine sid)
                                (pr/recur (inc i)
                                          (conj acc {:calls (:toolCalls f)
                                                     :errors (:toolCallErrors f)
                                                     :finish (:finishReason f)}))))))]
            (let [codes (keep #(get-in % [:arguments :code]) (mapcat :calls runs))
                  ;; a "length" turn can truncate a block mid-tag — that is the
                  ;; documented token-budget sampling artifact, not a grammar
                  ;; failure; completed turns must have zero errors
                  hard-errs (mapcat :errors (filter #(not= "length" (:finish %)) runs))]
              (println "    cljs codes:" (pr-str (vec codes)))
              (when (seq hard-errs)
                (println "    cljs HARD ERRORS:" (pr-str (vec hard-errs))
                         "finishes:" (pr-str (mapv :finish runs))))
              (assert-true "P10cljs: constrained turns produced calls (non-vacuous)"
                           (pos? (count codes)))
              (assert-true "P10cljs: zero unparseable calls on completed turns"
                           (empty? hard-errs))
              (assert-true "P10cljs: every emitted argument is exactly one complete CLJS form"
                           (every? #(= :complete (ceval/cljs-arg-status %)) codes))
              (assert-true "P10cljs: every argument opens with a delimiter"
                           (every? #(contains? #{"(" "[" "{"} (subs % 0 1)) codes))
              (k-verifier-tests))))))))

(defn- k-verifier-tests
  "P11: the maww verifier-callback seam on the 0.8b (hot lanes, K=4)."
  []
  (let [cfgk (assoc cfg-greedy :temperature 1.0 :maxNewTokens 48 :bestOfK 4)
        seen (atom [])
        v-winner (fn [cj]
                   (swap! seen conj (js->clj (js/JSON.parse cj) :keywordize-keys true))
                   (js/JSON.stringify #js {:winner 2}))
        sid (.newSession engine "{}")]
    (pr/let [{f :final} (run-turn sid msgs-1 cfgk nil v-winner)]
      (let [cands (:candidates (first @seen))]
        (assert-true "P11: verifier called exactly once with all K candidates"
                     (and (= 1 (count @seen)) (= 4 (count cands))))
        (assert-true "P11: candidates carry text + parsed toolCalls fields"
                     (every? #(and (contains? % :text) (contains? % :toolCalls)) cands))
        (assert-true "P11: winner-index protocol honored (final text == candidate 2)"
                     (and (= 2 (:winnerIndex f))
                          (= (:text f) (:text (nth cands 2))))))
      (.dispose engine sid)
      (let [sid2  (.newSession engine "{}")
            seen2 (atom nil)
            v-scores (fn [cj]
                       (reset! seen2 (js->clj (js/JSON.parse cj) :keywordize-keys true))
                       ;; ties at indices 1 and 3 -> lowest wins
                       (js/JSON.stringify #js {:scores #js [0.1 0.9 0.3 0.9]}))]
        (pr/let [{f2 :final} (run-turn sid2 msgs-1 cfgk nil v-scores)]
          (assert-true "P11: scores protocol = argmax, ties -> lowest index"
                       (and (= 1 (:winnerIndex f2))
                            (= (:text f2) (:text (nth (:candidates @seen2) 1)))))
          (.dispose engine sid2)
          (let [sid3 (.newSession engine "{}")
                v-throw (fn [_] (throw (js/Error. "verifier exploded")))]
            (pr/let [{f3 :final} (run-turn sid3 msgs-1 cfgk nil v-throw)]
              (assert-true "P11: throwing verifier -> fallback to candidate 0"
                           (and (contains? #{"stop" "length" "toolUse"} (:finishReason f3))
                                (= 0 (:winnerIndex f3))))
              (.dispose engine sid3)
              (let [sid5 (.newSession engine "{}")
                    v-hang (fn [_] (js/Promise. (fn [_ _])))]
                (pr/let [{f5 :final} (run-turn sid5 msgs-1
                                               (assoc cfgk :verifierTimeoutMs 150)
                                               nil v-hang)]
                  (assert-true "P11: hanging verifier -> timeout fallback to candidate 0"
                               (and (= 0 (:winnerIndex f5))
                                    (contains? #{"stop" "length" "toolUse"} (:finishReason f5))))
                  (.dispose engine sid5)
                  (let [sid6 (.newSession engine "{}")]
                    (pr/let [{fe :final} (run-turn sid6 msgs-1
                                                   (assoc cfgk :reasoningEffort "medium"))]
                      (assert-true "P11: bestOfK + thinking -> typed error"
                                   (and (= "error" (:finishReason fe))
                                        (boolean (re-find #"thinking" (str (:errorMessage fe))))))
                      (pr/let [{fg :final} (run-turn sid6 msgs-point
                                                     (assoc cfgk :tools [(point-tool xy-pattern)]))]
                        (assert-true "P11: bestOfK + grammar -> typed error"
                                     (and (= "error" (:finishReason fg))
                                          (boolean (re-find #"grammar" (str (:errorMessage fg))))))
                        (.dispose engine sid6)
                        (summary)))))))))))))

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
                      (seam-tests info)))))))))
      (pr/catch (fn [e]
                  (println "ERROR:" (str e))
                  (swap! fail inc)
                  (summary)))))
