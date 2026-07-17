(ns genmlx.llm.pi-provider
  "The pi coding-agent turn engine over the OWNED forward (genmlx-djw6, L2
   core): sessions are branch-ledger branches, turns are token-diff delta
   prefills + a native-parity decode loop. Loaded into the agent process by
   packages/agent/src/provider/genmlx/genmlx-host.ts via nbb's programmatic
   API — the file's LAST FORM is the #js engine object the host consumes.

   Seam contract (genmlx-host.ts GenmlxTurnEngine): JSON strings in/out plus
   one per-delta callback; promises where work is async. Messages arrive in
   the native ChatMessage shape (the v1 convert-messages output), config in
   the v1 buildChatConfig shape, so prompt render (applyChatTemplate — the
   SAME Rust renderer v1 uses) and sampling (genmlx.llm.sampling — the
   native sampler's semantics) give v1 parity by construction.

   Turn algorithm (spec §4; fork-ability by construction — item 10):
   1. Render P = applyChatTemplate(messages, tools, enableThinking).
   2. If P extends the session's committed prefix T: delta-prefill ONLY the
      suffix on the branch (forward-branch-tokens). cachedTokens = |T|.
   3. Else (edited/compacted history): dispose the branch, fresh branch,
      full prefill. A pi session FORK is branch-from — O(1), never re-prefill.
   4. Decode: per-token onDelta callbacks (think-aware, marker tokens never
      leak into text), stop on eos / maxNewTokens / abort / budget-forced
      </think>; the promise chain yields per token (finalizer breathing
      under Node, where jsc-cleanup! is a no-op — genmlx-12w4).
   5. Commit T' = P + fed tokens; final JSON carries the v1 ChatStreamFinal
      accounting fields (promptTokens/numTokens/reasoningTokens/cachedTokens).

   State discipline: ONE atom (the session registry + resident model), at
   the world-membrane edge like world/net.cljs — never read by pure code.
   Per-turn abort is a volatile in the session entry.

   Extension seams (reserved, no-ops today): :logit-mask (per-decode grammar
   hook, genmlx-3g0t), :images (genmlx-5aah), K-lane decode (genmlx-maww)."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.sampling :as samp]
            [genmlx.llm.toolcall :as tc]
            [genmlx.mlx.random :as rng]
            [promesa.core :as p]))

(defonce ^:private state*
  (atom {:model-map nil :model-path nil :sessions {} :next-session 1}))

(def ^:private sweep-every 32)

;; ---------------------------------------------------------------------------
;; model + session plumbing
;; ---------------------------------------------------------------------------

(defn- resident []
  (let [{:keys [model-map model-path]} @state*]
    (when-not model-map
      (throw (ex-info "pi-provider: no model loaded" {:genmlx/error :no-model})))
    (assoc model-map :path model-path)))

(defn- session! [sid]
  (or (get-in @state* [:sessions sid])
      (throw (ex-info (str "pi-provider: unknown session " sid)
                      {:genmlx/error :unknown-session :sid sid}))))

(defn- fresh-branch! [model]
  (llm/owned-branch! model {:cache nil :offset 0}))

(defn- load-model!* [path]
  (let [{:keys [model-path sessions]} @state*]
    (if (= path model-path)
      (p/resolved (clj->js {:path path :reused true}))
      (do
        ;; swap: dispose all session branches of the old resident first
        (when-let [old (:model-map @state*)]
          (doseq [[_ s] sessions]
            (llm/dispose-branch! (:model old) (:branch-id s))))
        (swap! state* assoc :model-map nil :model-path nil :sessions {})
        (p/let [mm (llm/load-model path {:cljs-forward? true})]
          (when-not (llm/cljs-forward-model? (:model mm))
            (throw (ex-info "pi-provider: checkpoint did not load on the owned forward"
                            {:genmlx/error :not-owned-forward :path path})))
          (swap! state* assoc :model-map mm :model-path path)
          (clj->js {:path path
                    :type (name (:type mm))
                    :eosTokenId (llm/eos-token-id (:tokenizer mm))}))))))

(defn- new-session* [_opts-json]
  (let [{:keys [model]} (resident)
        id (str "s" (:next-session @state*))
        branch (fresh-branch! model)]
    (swap! state* (fn [st]
                    (-> st
                        (update :next-session inc)
                        (assoc-in [:sessions id]
                                  {:branch-id branch
                                   :tokens []
                                   :busy? false
                                   :abort? (volatile! false)
                                   :key (rng/fresh-key)}))))
    id))

(defn- dispose* [sid]
  (when-let [s (get-in @state* [:sessions sid])]
    (when-let [mm (:model-map @state*)]
      (llm/dispose-branch! (:model mm) (:branch-id s)))
    (swap! state* update :sessions dissoc sid))
  nil)

(defn- abort* [sid]
  (when-let [s (get-in @state* [:sessions sid])]
    (vreset! (:abort? s) true))
  nil)

;; ---------------------------------------------------------------------------
;; turn machinery
;; ---------------------------------------------------------------------------

(defn- shared-prefix-len [a b]
  (let [n (min (count a) (count b))]
    (loop [i 0]
      (if (and (< i n) (== (nth a i) (nth b i)))
        (recur (inc i))
        i))))

(defn- reject-images! [messages]
  (doseq [m messages]
    (when-let [imgs (.-images m)]
      (when (pos? (.-length imgs))
        (throw (ex-info (str "pi-provider: image-bearing history — the genmlx "
                             "provider is text-only until genmlx-5aah lands; "
                             "run VLM turns on the mlx provider.")
                        {:genmlx/error :images-unsupported-until-5aah}))))))

(defn- enable-thinking? [config]
  ;; TemplateHonoring mirror (qwen3.5 family): none/low -> false, medium/high -> true
  (contains? #{"medium" "high"} (or (.-reasoningEffort config) "none")))

(defn- sampling-cfg [config]
  {:temperature        (.-temperature config)
   :top-k              (.-topK config)
   :top-p              (.-topP config)
   :min-p              (.-minP config)
   :repetition-penalty (.-repetitionPenalty config)
   :presence-penalty   (.-presencePenalty config)})

(defn- toolcalls-for-final
  "Parse qwen3_xml tool-call blocks out of the visible text into the
   ChatStreamFinal toolCalls shape (ok calls only; malformed blocks surface
   as errors on the JSON for the shim to coerce)."
  [text sid-counter]
  (let [{:keys [calls errors]} (tc/parse-tool-calls text)]
    {:toolCalls (vec (map-indexed
                      (fn [i {:keys [name args]}]
                        {:id (str "call_" sid-counter "_" (inc i))
                         :name name
                         :arguments args
                         :status "ok"})
                      calls))
     :toolCallErrors (vec errors)}))

(defn- finish-payload
  [{:keys [gen-text think-text raw-text finish-reason prompt-tokens sampled
           reasoning cached sid]}]
  (let [{:keys [toolCalls toolCallErrors]} (toolcalls-for-final gen-text sid)
        reason (if (and (= finish-reason "stop") (seq toolCalls))
                 "toolUse" finish-reason)]
    (js/JSON.stringify
     (clj->js {:text gen-text
               :thinking (when (seq think-text) think-text)
               :rawText raw-text
               :finishReason reason
               :toolCalls toolCalls
               :toolCallErrors toolCallErrors
               :promptTokens prompt-tokens
               :numTokens sampled
               :reasoningTokens reasoning
               :cachedTokens cached}))))

(defn- turn-stream* [sid messages-json config-json on-delta]
  (-> (p/let [{:keys [model tokenizer]} (resident)
              session (session! sid)
              _ (when (:busy? session)
                  (throw (ex-info (str "pi-provider: session " sid " has a turn in flight")
                                  {:genmlx/error :turn-in-flight :sid sid})))
              messages (js/JSON.parse messages-json)
              config   (js/JSON.parse config-json)
              _ (reject-images! messages)
              think?   (enable-thinking? config)
              tools    (.-tools config)
              rendered (.applyChatTemplate tokenizer messages true
                                           (or tools js/undefined) think?)]
        (vreset! (:abort? session) false)
        (swap! state* assoc-in [:sessions sid :busy?] true)
        (let [prompt      (vec rendered)
              committed   (:tokens session)
              shared      (shared-prefix-len committed prompt)
              append?     (and (= shared (count committed)) (< shared (count prompt)))
              branch      (:branch-id session)
              ;; non-append (edited/compacted/regenerated history): rebuild
              branch      (if (or append? (zero? (count committed)))
                            branch
                            (do (llm/dispose-branch! model branch)
                                (let [b (fresh-branch! model)]
                                  (swap! state* assoc-in [:sessions sid :branch-id] b)
                                  b)))
              cached      (if append? shared 0)
              suffix      (subvec prompt cached)
              _ (when (empty? suffix)
                  (throw (ex-info "pi-provider: rendered prompt does not extend the committed prefix"
                                  {:genmlx/error :empty-suffix :sid sid})))
              prefill-chunk (.-prefillChunk config)
              eos-id      (llm/eos-token-id tokenizer)
              think-start (llm/token->id tokenizer "<think>")
              think-end   (llm/token->id tokenizer "</think>")
              budget      (.-thinkingTokenBudget config)
              max-new     (or (.-maxNewTokens config) 512)
              scfg        (sampling-cfg config)
              logits0     (llm/forward-branch-tokens model branch suffix
                                                     (when (number? prefill-chunk)
                                                       {:chunk prefill-chunk}))
              ;; template-opened think block: the generation prompt ends inside
              ;; <think> (the opener may be followed by a newline token, so
              ;; sniff the tail rather than only the last token)
              in-think0   (boolean (and think? (some? think-start)
                                        (some #(= % think-start) (take-last 3 prompt))))
              t0          (js/Date.now)]
          (letfn [(decode-all [toks]
                    (llm/decode tokenizer (js/Uint32Array.from (into-array toks))))
                  (emit! [piece reasoning?]
                    (when (seq piece)
                      (on-delta (js/JSON.stringify
                                 (clj->js {:text piece :isReasoning reasoning?})))))
                  (finish [state reason]
                    (let [{:keys [fed sampled reasoning think-pieces text-pieces key]} state
                          gen-text   (apply str text-pieces)
                          think-text (apply str think-pieces)]
                      (swap! state* update-in [:sessions sid]
                             (fn [s] (-> s
                                         (assoc :tokens (into prompt fed))
                                         (assoc :key key)
                                         (assoc :busy? false))))
                      (finish-payload {:gen-text gen-text
                                       :think-text think-text
                                       :raw-text (str think-text gen-text)
                                       :finish-reason reason
                                       :prompt-tokens (count prompt)
                                       :sampled sampled
                                       :reasoning reasoning
                                       :cached cached
                                       :sid sid})))
                  (step [{:keys [i logits key gen fed sampled reasoning in-think?
                                 decoded-len] :as st}]
                    (cond
                      @(:abort? (session! sid)) (p/resolved (finish st "aborted"))
                      (>= i max-new)            (p/resolved (finish st "length"))
                      :else
                      (let [force-close? (and in-think? (number? budget) (some? think-end)
                                              (>= reasoning budget))
                            [tok key']   (if force-close?
                                           [think-end key]
                                           (samp/sample-token key logits scfg gen))]
                        (llm/sweep-tick! i sweep-every)
                        (if (= tok eos-id)
                          (p/resolved (finish (assoc st :sampled (inc sampled)) "stop"))
                          (let [gen'   (conj gen tok)
                                think-marker? (or (= tok think-start) (= tok think-end))
                                in-think?' (cond (= tok think-start) true
                                                 (= tok think-end)   false
                                                 :else in-think?)
                                reasoning' (if in-think? (inc reasoning) reasoning)]
                            ;; decode the FULL gen each step and emit the tail delta —
                            ;; multi-byte tokens detokenize correctly (HF-streamer trick)
                            (p/let [full (decode-all gen')]
                              (let [piece (subs full decoded-len)
                                    logits' (llm/forward-branch model branch tok)
                                    st' (-> st
                                            (assoc :i (inc i) :logits logits' :key key'
                                                   :gen gen' :fed (conj fed tok)
                                                   :sampled (inc sampled)
                                                   :reasoning reasoning'
                                                   :in-think? in-think?'
                                                   :decoded-len (count full))
                                            (update :think-pieces
                                                    #(if (and in-think? (not think-marker?))
                                                       (conj % piece) %))
                                            (update :text-pieces
                                                    #(if (and (not in-think?) (not think-marker?))
                                                       (conj % piece) %)))]
                                (when-not think-marker?
                                  (emit! piece in-think?))
                                (step st'))))))))]
            (p/let [result (step {:i 0 :logits logits0 :key (:key session)
                                  :gen [] :fed [] :sampled 0 :reasoning 0
                                  :in-think? in-think0 :decoded-len 0
                                  :think-pieces [] :text-pieces []})]
              (println (str "[pi-provider] turn " sid ": " (count prompt) " prompt ("
                            cached " cached), " (- (js/Date.now) t0) " ms"))
              result))))
      (p/catch
       (fn [err]
         ;; error terminal: same shape v1 emits; branch marked for rebuild
         (when-let [s (get-in @state* [:sessions sid])]
           (swap! state* update-in [:sessions sid]
                  #(assoc % :busy? false :tokens [])))
         (js/JSON.stringify
          (clj->js {:text "" :thinking nil :rawText ""
                    :finishReason "error"
                    :errorMessage (str (or (ex-message err) err))
                    :toolCalls [] :toolCallErrors []
                    :promptTokens 0 :numTokens 0
                    :reasoningTokens 0 :cachedTokens 0}))))))

;; ---------------------------------------------------------------------------
;; the engine object — def'd for tests, and re-stated as the namespace's
;; LAST FORM so nbb loadFile (genmlx-host.ts) returns it
;; ---------------------------------------------------------------------------

(def engine
  "The #js turn-engine API consumed by genmlx-host.ts (GenmlxTurnEngine)."
  #js {:loadModel  (fn [path] (load-model!* path))
       :newSession (fn [opts-json] (new-session* opts-json))
       :turnStream (fn [sid messages-json config-json on-delta]
                     (turn-stream* sid messages-json config-json on-delta))
       :abort      (fn [sid] (abort* sid))
       :dispose    (fn [sid] (dispose* sid))})

engine
