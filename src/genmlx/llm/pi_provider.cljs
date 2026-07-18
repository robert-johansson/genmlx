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

   Extension seams:
   - :logit-mask: set-logit-mask! installs a per-session
     (fn [logits gen-tokens] -> logits'|nil) applied before every sampling
     step — the external constraint hook.

   Best-of-K + verifier (genmlx-maww, LANDED): config.bestOfK > 1 decodes
   K candidate turns in lockstep on a FORK of the session branch — one
   batched forward per token — then calls turnStream's optional verifier
   callback ONCE with all K candidates and streams the winner. Fallback-
   to-first on verifier absence/throw/timeout/malformed answer; bestOfK 1
   is the scalar path unchanged. v1 scope: no thinking, no grammar
   composition (typed errors); candidates do not stream token-by-token.

   Per-argument grammar (genmlx-3g0t, LANDED): a tool parameter schema
   that declares a JSON-Schema `pattern` regex turns the whole declared
   toolset into a tool-call DFA (toolcall.cljs qwen3_xml dialect) applied
   at the same per-decode masking point — malformed tool calls and
   off-pattern argument values become UNREPRESENTABLE at sampling time.
   The DFA sees only the VISIBLE stream (think blocks skip it). The
   `x-genmlx-grammar: \"cljs\"` reader-level annotation is reserved
   (typed error until wired).

   Images (genmlx-5aah, LANDED): image bytes ride turnStream's optional
   5th arg (a JS array of Uint8Array — bytes never cross the JSON seam);
   messages carry imageRefs indices, reattached as msg.images before
   applyChatTemplate so the SAME Rust renderer emits the vision markers
   (render parity with v1 by construction). An image-bearing render
   REBUILDS the session branch through the owned VLM prefill
   (forward-prefill {:images ...} -> branch-cache!): vision tower once,
   image-conditioned branch after — subsequent text-only turns delta-
   prefill over it transparently (branch entries carry :rope-delta).
   cachedTokens is 0 on the image-arrival turn (an honest full prefill),
   and covers the whole image prefix on every later turn."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.sampling :as samp]
            [genmlx.llm.toolcall :as tc]
            [genmlx.mlx :as mx]
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
        (swap! state* assoc :model-map nil :model-path nil :sessions {}
               :grammar-cache {} :token-index nil)
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
                                   :logit-mask nil
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

(defn set-logit-mask!
  "Install (or clear, with nil) a session's per-decode logit-mask hook — the
   genmlx-3g0t grammar entry point. f = (fn [logits gen-tokens] -> logits');
   logits is the [vocab] MLX array for the NEXT token, gen-tokens the host
   vector of ids sampled so far this turn. A nil return leaves logits
   unchanged. Runs before EVERY sampling step; must be pure graph
   construction (masking via mx/where etc.) — no eval!/item inside."
  [sid f]
  (session! sid)
  (swap! state* assoc-in [:sessions sid :logit-mask] f)
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

(defn- reattach-images!
  "Rebind image bytes onto the JSON-parsed messages: each message's
   imageRefs (indices into `images-arr`, the turnStream 5th arg) becomes
   msg.images = [Uint8Array ...] — the shape the Rust ChatMessage takes,
   so applyChatTemplate renders the vision markers inline (genmlx-5aah).
   Returns all images in message order."
  [messages images-arr]
  (let [arr (or images-arr #js [])]
    (into []
          (mapcat (fn [m]
                    (when-let [refs (.-imageRefs m)]
                      (let [imgs (mapv #(aget arr %) (array-seq refs))]
                        (when (some nil? imgs)
                          (throw (ex-info "pi-provider: imageRefs points past the images array"
                                          {:genmlx/error :bad-image-ref})))
                        (set! (.-images m) (into-array imgs))
                        (js-delete m "imageRefs")
                        imgs))))
          (array-seq messages))))

(defn- images-key
  "Cheap identity key for an image sequence (count + per-image byte
   lengths) — enough to detect an image set change between renders."
  [images]
  (mapv #(.-byteLength %) images))

(def ^:private max-pending-detok
  "Cap on tokens held back awaiting a complete UTF-8 char (a char spans at
   most 4 bytes, so at most 4 single-byte tokens; 8 is safety margin —
   past it we flush even with a trailing replacement char)."
  8)

(defn detok-step
  "One incremental-detokenization step (the HF TextStreamer discipline):
   pending = token ids held since the last flushed piece, tok = the new
   token. Decodes ONLY pending+tok — O(1) per step instead of the O(n)
   full-generation re-decode — holding back when the tail is an incomplete
   multi-byte char (trailing U+FFFD). Valid because byte-level BPE decode
   is concatenative at char boundaries (verified per-tokenizer by the
   engine test). Returns a promise of {:piece string :pending vector}."
  [decode-fn pending tok]
  (let [pending' (conj pending tok)]
    (p/let [s (decode-fn pending')]
      (if (and (seq s)
               (== 0xFFFD (.charCodeAt s (dec (count s))))
               (< (count pending') max-pending-detok))
        {:piece "" :pending pending'}
        {:piece s :pending []}))))

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

;; ---------------------------------------------------------------------------
;; per-argument grammar constraints (genmlx-3g0t)
;;
;; The constraint declaration RIDES THE TOOL PARAMETER SCHEMA — the seam is
;; the contract: an extension annotates a string parameter and the provider
;; compiles the whole declared toolset into one tool-call DFA (toolcall.cljs
;; dialect: prose (block prose)*), applied through the same per-decode
;; logit-mask point set-logit-mask! exposes. Two annotations:
;;   - `pattern` (standard JSON Schema): a grammar.cljs regex for the
;;     argument VALUE — malformed argument text becomes UNREPRESENTABLE at
;;     sampling time.
;;   - `x-genmlx-grammar: "cljs"`: the reader-level (edamame) constraint
;;     (genmlx-3g0t): the argument value is exactly one complete,
;;     delimiter-opened ClojureScript form, enforced byte-granularly by
;;     toolcall/hybrid-masker. Any other x-genmlx-grammar value is a typed
;;     error, never a silent ignore.
;; Grammar activation is annotation-presence — a toolset with no annotated
;; params runs unconstrained (v1-parity decoding).
;; ---------------------------------------------------------------------------

(defn- tool-grammar-spec
  "Parse the native ToolDefinition[] (config.tools — `properties` arrives as
   a JSON string) into the compile-toolcall tools spec, or nil when no
   parameter declares a grammar annotation."
  [tools]
  (when (and tools (pos? (.-length tools)))
    (let [specs
          (mapv (fn [t]
                  (let [f     (.-function t)
                        praw  (some-> f .-parameters .-properties)
                        props (cond (string? praw) (js/JSON.parse praw)
                                    (some? praw)   praw
                                    :else          #js {})
                        names (js/Object.keys props)]
                    {:name (.-name f)
                     :params
                     (mapv (fn [k]
                             (let [prop (aget props k)
                                   xg   (aget prop "x-genmlx-grammar")
                                   pat  (.-pattern prop)]
                               (when (and (some? xg) (not= xg "cljs"))
                                 (throw (ex-info (str "pi-provider: parameter " k " of tool "
                                                      (.-name f) " declares unknown "
                                                      "x-genmlx-grammar " (pr-str xg)
                                                      " — supported: \"cljs\".")
                                                 {:genmlx/error :unknown-grammar-annotation
                                                  :tool (.-name f) :param k :value xg})))
                               (cond-> {:name k}
                                 (string? pat) (assoc :pattern pat)
                                 (= xg "cljs") (assoc :cljs true))))
                           names)}))
                (array-seq tools))]
      (when (some #(some (fn [p] (or (:pattern p) (:cljs p))) (:params %)) specs)
        specs))))

(defn- toolcall-constraint!
  "Compile (and cache) the tool-call grammar constraint for `specs` over the
   resident tokenizer. The vocab token-index is built once per resident
   model; compiled constraints are cached by spec value."
  [tokenizer specs]
  (or (get-in @state* [:grammar-cache specs])
      (let [tix (or (:token-index @state*)
                    (let [t (gram/build-token-index tokenizer)]
                      (swap! state* assoc :token-index t)
                      t))
            c   (tc/compile-toolcall tokenizer specs {:token-index tix})]
        (swap! state* assoc-in [:grammar-cache specs] c)
        c)))

(defn- grammar-masker
  "Per-turn closure over `constraint`: advances the DFA through the VISIBLE
   token stream (think blocks and markers never touch it) and masks logits
   for the next position. Small per-turn state->mask memo (prose loops sit
   in one state; the toolcall DFA is too large to precompute in full)."
  [constraint]
  (let [{:keys [dfa token-index]} constraint
        st* (volatile! {:state (:start dfa) :seen 0 :memo {}})]
    (fn [logits vis]
      (let [{:keys [state seen memo]} @st*
            state' (reduce (fn [s t]
                             (let [txt (nth token-index t nil)]
                               (if (seq txt) (gram/dfa-advance-string dfa s txt) s)))
                           state (subvec vis seen))
            memo'  (if (or (contains? memo state') (>= (count memo) 64))
                     memo
                     (assoc memo state' (gram/get-mask constraint state')))
            _      (vreset! st* {:state state' :seen (count vis) :memo memo'})]
        (gram/apply-mask (if-let [m (get memo' state')]
                           (assoc constraint :masks {state' m})
                           constraint)
                         state' logits)))))

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
           reasoning cached sid extra]}]
  (let [{:keys [toolCalls toolCallErrors]} (toolcalls-for-final gen-text sid)
        reason (if (and (= finish-reason "stop") (seq toolCalls))
                 "toolUse" finish-reason)]
    (js/JSON.stringify
     (clj->js (merge {:text gen-text
                      :thinking (when (seq think-text) think-text)
                      :rawText raw-text
                      :finishReason reason
                      :toolCalls toolCalls
                      :toolCallErrors toolCallErrors
                      :promptTokens prompt-tokens
                      :numTokens sampled
                      :reasoningTokens reasoning
                      :cachedTokens cached}
                     extra)))))

;; ---------------------------------------------------------------------------
;; best-of-K decode + verifier callback (genmlx-maww)
;;
;; SEAM CONTRACT v1: turnStream's 6th arg is an optional verifier callback
;;   (candidatesJson: string) => Promise<resultJson: string>
;; called ONCE per bestOfK>1 turn with ALL K candidates:
;;   {"candidates": [{"index", "text", "finishReason", "toolCalls",
;;                    "toolCallErrors"}, ...]}
;; and answering either {"winner": <index>} or {"scores": [K numbers]}
;; (argmax; ties -> lowest index). Degenerate semantics (the bean's
;; done-means): verifier ABSENT / THROWS / TIMES OUT (config
;; verifierTimeoutMs, default 10000) / answers MALFORMED -> candidate 0;
;; bestOfK 1 (or absent) -> the scalar path, byte-identical to the core
;; provider. K candidates come from ONE batched forward per token
;; (forward-branch-batched over a FORK of the session branch — the session's
;; scalar branch never advances, so the commit stays honest: committed = the
;; prompt the branch consumed; the next turn delta-prefills the winner's
;; re-rendered reply as an ordinary suffix). v1 scope: thinking and
;; per-argument grammar do not compose with K>1 (typed errors); candidate
;; turns do not stream token-by-token (the winner arrives as one delta).
;; ---------------------------------------------------------------------------

(defn- verifier-winner
  "Parse a verifier result JSON into a winner index over k candidates;
   malformed -> 0 (fallback-to-first)."
  [res-json k]
  (try
    (let [r      (js/JSON.parse res-json)
          w      (.-winner r)
          scores (.-scores r)]
      (cond
        (and (number? w) (>= w 0) (< w k) (== w (js/Math.floor w)))
        w

        (and scores (= k (.-length scores)))
        ;; argmax, ties -> lowest index (max-key keeps the LAST max, so
        ;; feed indices in reverse)
        (first (apply max-key second (reverse (map-indexed vector (vec scores)))))

        :else 0))
    (catch :default _ 0)))

(defn- call-verifier
  "Invoke the external verifier with the candidates; resolves to the winner
   index. Absent verifier, throw, timeout, or malformed answer -> 0."
  [verifier candidates timeout-ms]
  (if-not (fn? verifier)
    (p/resolved 0)
    (-> (p/race [(-> (p/resolved nil)
                     (p/then (fn [_] (verifier (js/JSON.stringify
                                                (clj->js {:candidates candidates})))))
                     (p/then (fn [res] (verifier-winner res (count candidates)))))
                 (p/delay timeout-ms ::timeout)])
        (p/then (fn [r] (if (= r ::timeout) 0 r)))
        (p/catch (fn [_] 0)))))

(defn- decode-k-lanes!
  "Advance K lockstep lanes on branch `fork-id` until every lane stops or
   max-new. The first step samples K tokens from the SHARED prompt logits
   (per-lane keys); each later step feeds a [K] token vector through ONE
   batched forward (the scalar fork tiles lazily inside
   forward-branch-batched). Finished lanes are fed EOS padding; their
   outputs are ignored. Resolves to {:lanes [{:gen :reason}] :aborted?}."
  [model fork-id logits0 k lane-keys scfg max-new eos-id abort?*]
  (p/loop [i 0
           lanes (mapv (fn [key] {:gen [] :key key :done false :reason "length"})
                       lane-keys)
           logits logits0
           scalar? true]
    (if (or @abort?* (every? :done lanes) (>= i max-new))
      {:lanes (mapv #(select-keys % [:gen :reason]) lanes)
       :aborted? (boolean @abort?*)}
      (let [sampled (mapv (fn [j {:keys [gen key done] :as lane}]
                            (if done
                              (assoc lane :tok eos-id)
                              (let [lg (if scalar? logits (mx/index logits j))
                                    [tok key'] (samp/sample-token key lg scfg gen)]
                                (if (= tok eos-id)
                                  (assoc lane :done true :reason "stop"
                                         :key key' :tok eos-id)
                                  (-> lane
                                      (assoc :key key' :tok tok)
                                      (update :gen conj tok))))))
                          (range k) lanes)]
        (llm/sweep-tick! i sweep-every)
        (if (every? :done sampled)
          (p/recur (inc i) (mapv #(dissoc % :tok) sampled) logits scalar?)
          (let [toks    (mapv :tok sampled)
                logits' (llm/forward-branch-batched
                         model fork-id (mx/array toks [k] mx/int32))]
            (p/recur (inc i) (mapv #(dissoc % :tok) sampled) logits' false)))))))

(defn- turn-stream* [sid messages-json config-json on-delta images-arr verifier]
  (-> (p/let [{:keys [model tokenizer]} (resident)
              session (session! sid)
              _ (when (:busy? session)
                  (throw (ex-info (str "pi-provider: session " sid " has a turn in flight")
                                  {:genmlx/error :turn-in-flight :sid sid})))
              messages (js/JSON.parse messages-json)
              config   (js/JSON.parse config-json)
              ;; 5aah: rebind image bytes (the non-JSON leg of the seam) onto
              ;; their messages so the Rust renderer emits vision markers.
              images   (reattach-images! messages images-arr)
              ;; maww: the K-lane (batched best-of-K) decode entry.
              best-of-k (or (.-bestOfK config) 1)
              k-mode?   (and (number? best-of-k) (> best-of-k 1))
              think?   (enable-thinking? config)
              _ (when (and k-mode? think?)
                  (throw (ex-info (str "pi-provider: bestOfK=" best-of-k " does not "
                                       "compose with thinking in v1 — set "
                                       "reasoningEffort none for best-of-K turns.")
                                  {:genmlx/error :best-of-k-thinking-unsupported})))
              tools    (.-tools config)
              rendered (.applyChatTemplate tokenizer messages true
                                           (or tools js/undefined) think?)]
        (vreset! (:abort? session) false)
        (swap! state* assoc-in [:sessions sid :busy?] true)
        (let [prompt      (vec (js/Array.from rendered))
              committed   (:tokens session)
              shared      (shared-prefix-len committed prompt)
              img-key     (images-key images)
              same-imgs?  (= img-key (or (:images-key session) []))
              append?     (and same-imgs?
                               (= shared (count committed))
                               (< shared (count prompt)))
              ;; a render whose image set changed (first image, tool-result
              ;; image, edited history) rebuilds through the VLM prefill
              vlm-build?  (and (seq images) (not append?))
              cached      (if append? shared 0)
              suffix      (subvec prompt cached)
              _ (when (and (not vlm-build?) (empty? suffix))
                  (throw (ex-info "pi-provider: rendered prompt does not extend the committed prefix"
                                  {:genmlx/error :empty-suffix :sid sid})))
              ;; bounded prefill transient by default (mirrors v1's paged
              ;; chunking); config.prefillChunk overrides, <=0 disables.
              ;; The VLM prefill has its own measured-optimal default (192)
              ;; — only an EXPLICIT config chunk overrides it there.
              prefill-chunk (or (.-prefillChunk config) 2048)
              eos-id      (llm/eos-token-id tokenizer)
              think-start (llm/token->id tokenizer "<think>")
              think-end   (llm/token->id tokenizer "</think>")
              budget      (.-thinkingTokenBudget config)
              max-new     (or (.-maxNewTokens config) 512)
              scfg        (sampling-cfg config)
              ;; 3g0t: annotated tool params -> the tool-call DFA, applied
              ;; per decode step (compiled once per toolset, cached). A
              ;; toolset with :cljs params gets the hybrid (DFA + reader)
              ;; masker; pattern-only toolsets keep the plain DFA masker.
              gmask       (when-let [specs (tool-grammar-spec tools)]
                            (let [c (toolcall-constraint! tokenizer specs)]
                              (if (:cljs-support c)
                                (tc/hybrid-masker c)
                                (grammar-masker c))))
              _ (when (and k-mode? gmask)
                  (throw (ex-info (str "pi-provider: bestOfK=" best-of-k " does not "
                                       "compose with per-argument grammar constraints "
                                       "in v1 (the batched-DFA vtables are the "
                                       "follow-up); drop the pattern annotations or "
                                       "run bestOfK 1.")
                                  {:genmlx/error :best-of-k-grammar-unsupported})))
              [branch logits0]
              (cond
                append?
                [(:branch-id session)
                 (llm/forward-branch-tokens model (:branch-id session) suffix
                                            (when (number? prefill-chunk)
                                              {:chunk prefill-chunk}))]

                vlm-build?
                ;; owned VLM prefill over the FULL prompt (vision tower ->
                ;; pad-slot scatter -> M-RoPE decoder), then fork the cache
                ;; cell into the session branch — later text-only turns
                ;; delta-prefill over the image-conditioned branch.
                (do (llm/dispose-branch! model (:branch-id session))
                    (let [lg (llm/forward-prefill model prompt
                                                  {:images images
                                                   :chunk (let [c (.-prefillChunk config)]
                                                            (when (number? c) c))})
                          b  (llm/branch-cache! model)]
                      (swap! state* assoc-in [:sessions sid :branch-id] b)
                      [b lg]))

                :else
                ;; text-only rebuild (edited/compacted/regenerated history)
                (let [b (if (zero? (count committed))
                          (:branch-id session)
                          (do (llm/dispose-branch! model (:branch-id session))
                              (let [b (fresh-branch! model)]
                                (swap! state* assoc-in [:sessions sid :branch-id] b)
                                b)))]
                  [b (llm/forward-branch-tokens model b suffix
                                                (when (number? prefill-chunk)
                                                  {:chunk prefill-chunk}))]))
              ;; template-opened think block: the generation prompt ends inside
              ;; <think> (the opener may be followed by a newline token, so
              ;; sniff the tail rather than only the last token)
              in-think0   (boolean (and think? (some? think-start)
                                        (some #(= % think-start) (take-last 3 prompt))))
              t0          (js/Date.now)]
          (letfn [(decode-toks [toks]
                    (llm/decode tokenizer (js/Uint32Array.from (into-array toks))))
                  (emit! [piece reasoning?]
                    (when (seq piece)
                      (on-delta (js/JSON.stringify
                                 (clj->js {:text piece :isReasoning reasoning?})))))
                  (record-piece [st piece]
                    ;; append piece under the CURRENT mode + emit it; pending
                    ;; tokens never straddle a mode switch (markers flush first)
                    (if (seq piece)
                      (do (emit! piece (:in-think? st))
                          (update st (if (:in-think? st) :think-pieces :text-pieces)
                                  conj piece))
                      st))
                  (flush-pending [{:keys [pending] :as st}]
                    ;; force out held-back tokens (mode switch / end of turn)
                    (if (empty? pending)
                      (p/resolved st)
                      (p/let [piece (decode-toks pending)]
                        (record-piece (assoc st :pending []) piece))))
                  (finish [state reason]
                    (p/let [{:keys [fed sampled reasoning think-pieces text-pieces key]}
                            (flush-pending state)]
                      (let [gen-text   (apply str text-pieces)
                            think-text (apply str think-pieces)]
                        (swap! state* update-in [:sessions sid]
                               (fn [s] (-> s
                                           (assoc :tokens (into prompt fed))
                                           (assoc :images-key img-key)
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
                                         :sid sid}))))
                  (step [{:keys [i logits key gen fed sampled reasoning in-think?
                                 pending vis] :as st}]
                    (let [sess (session! sid)]
                      (cond
                        @(:abort? sess) (finish st "aborted")
                        (>= i max-new)  (finish st "length")
                        :else
                        (let [force-close? (and in-think? (number? budget) (some? think-end)
                                                (>= reasoning budget))
                              ;; grammar mask (3g0t) over the VISIBLE stream,
                              ;; then the external :logit-mask hook
                              logits*      (let [lg (if (and gmask (not in-think?))
                                                      (gmask logits vis)
                                                      logits)]
                                             (if-let [mask (:logit-mask sess)]
                                               (or (mask lg gen) lg)
                                               lg))
                              [tok key']   (if force-close?
                                             [think-end key]
                                             (samp/sample-token key logits* scfg gen))]
                          (llm/sweep-tick! i sweep-every)
                          (if (= tok eos-id)
                            (finish (assoc st :sampled (inc sampled)) "stop")
                            (let [think-marker? (or (= tok think-start) (= tok think-end))
                                  base (assoc st
                                              :i (inc i) :key key'
                                              :gen (conj gen tok) :fed (conj fed tok)
                                              :sampled (inc sampled)
                                              :reasoning (if in-think? (inc reasoning) reasoning)
                                              :vis (if (or in-think? think-marker?)
                                                     vis (conj vis tok)))]
                              (if think-marker?
                                ;; marker: flush held text under the pre-marker
                                ;; mode, then switch — marker text never leaks
                                (p/let [st' (flush-pending base)]
                                  (step (assoc st'
                                               :logits (llm/forward-branch model branch tok)
                                               :in-think? (= tok think-start))))
                                ;; incremental detok: decode only pending+tok
                                (p/let [{:keys [piece] :as dt} (detok-step decode-toks pending tok)]
                                  (step (-> base
                                            (assoc :logits (llm/forward-branch model branch tok)
                                                   :pending (:pending dt))
                                            (record-piece piece)))))))))))
                  (commit-k! [key']
                    ;; the session branch never advanced past the prompt (the
                    ;; K lanes ran on a fork), so the honest commit is P alone
                    (swap! state* update-in [:sessions sid]
                           (fn [s] (-> s
                                       (assoc :tokens prompt)
                                       (assoc :images-key img-key)
                                       (assoc :key key')
                                       (assoc :busy? false)))))
                  (k-turn! []
                    (let [k         best-of-k
                          fork      (llm/branch-from model branch)
                          ks        (rng/split-n (:key session) (inc k))
                          lane-keys (vec (take k ks))
                          sess-key' (peek ks)]
                      (-> (p/let [{:keys [lanes aborted?]}
                                  (decode-k-lanes! model fork logits0 k lane-keys
                                                   scfg max-new eos-id
                                                   (:abort? session))]
                            (if aborted?
                              (do (commit-k! sess-key')
                                  (finish-payload {:gen-text "" :think-text ""
                                                   :raw-text ""
                                                   :finish-reason "aborted"
                                                   :prompt-tokens (count prompt)
                                                   :sampled 0 :reasoning 0
                                                   :cached cached :sid sid
                                                   :extra {:bestOfK k}}))
                              (p/let [texts (p/all (mapv #(decode-toks (:gen %)) lanes))]
                                (let [candidates
                                      (vec (map-indexed
                                            (fn [j text]
                                              (let [{:keys [toolCalls toolCallErrors]}
                                                    (toolcalls-for-final text sid)]
                                                {:index j :text text
                                                 :finishReason (:reason (nth lanes j))
                                                 :toolCalls toolCalls
                                                 :toolCallErrors toolCallErrors}))
                                            texts))]
                                  (p/let [widx (call-verifier
                                                verifier candidates
                                                (or (.-verifierTimeoutMs config) 10000))]
                                    (let [wc (nth candidates widx)
                                          wl (nth lanes widx)]
                                      (emit! (:text wc) false)
                                      (commit-k! sess-key')
                                      (finish-payload
                                       {:gen-text (:text wc) :think-text ""
                                        :raw-text (:text wc)
                                        :finish-reason (:finishReason wc)
                                        :prompt-tokens (count prompt)
                                        :sampled (+ (count (:gen wl))
                                                    (if (= "stop" (:reason wl)) 1 0))
                                        :reasoning 0
                                        :cached cached :sid sid
                                        :extra {:bestOfK k :winnerIndex widx
                                                :candidateCount (count candidates)}})))))))
                          ;; p/handle, not p/finally: a downstream p/catch (line
                          ;; ~698) after a p/finally teardown double-settles under
                          ;; nbb (genmlx-tb5f) — dispose on both arms, re-raise once.
                          (p/handle (fn [r e]
                                      (llm/dispose-branch! model fork)
                                      (if e (throw e) r))))))]
            (p/let [result (if k-mode?
                             (k-turn!)
                             (step {:i 0 :logits logits0 :key (:key session)
                                    :gen [] :fed [] :sampled 0 :reasoning 0
                                    :in-think? in-think0 :pending [] :vis []
                                    :think-pieces [] :text-pieces []}))]
              (println (str "[pi-provider] turn " sid ": " (count prompt) " prompt ("
                            cached " cached), " (- (js/Date.now) t0) " ms"
                            (when k-mode? (str " [bestOfK " best-of-k "]"))))
              result))))
      (p/catch
       (fn [err]
         ;; error terminal: same shape v1 emits. The branch may have
         ;; consumed tokens before the throw, so dispose + fresh (a stale
         ;; branch with reset bookkeeping would let the next turn APPEND a
         ;; full prompt after the dead prefix). :turn-in-flight must NOT
         ;; touch state — the live turn owns it.
         (js/console.error "[pi-provider] turn error:" err)
         (when (and (not= :turn-in-flight (:genmlx/error (ex-data err)))
                    (get-in @state* [:sessions sid]))
           (when-let [mm (:model-map @state*)]
             (llm/dispose-branch! (:model mm) (get-in @state* [:sessions sid :branch-id]))
             (swap! state* assoc-in [:sessions sid :branch-id]
                    (fresh-branch! (:model mm))))
           (swap! state* update-in [:sessions sid]
                  #(assoc % :busy? false :tokens [] :images-key [])))
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
       :turnStream (fn [sid messages-json config-json on-delta images verifier]
                     (turn-stream* sid messages-json config-json on-delta images verifier))
       :abort      (fn [sid] (abort* sid))
       :dispose    (fn [sid] (dispose* sid))})

engine
