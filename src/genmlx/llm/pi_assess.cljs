(ns genmlx.llm.pi-assess
  "ASSESS on administered pi sessions (genmlx-opwh, L3-C1): replay a real
   session's chat messages through the SAME renderer the provider used at
   administration time and score, per assistant turn, the log-probability
   the model assigns to the action it actually took.

   RENDER-PARITY scoring: the context each turn is scored against is the
   re-rendered history (pi-session/path->messages mirror + native
   applyChatTemplate) — thinking dropped, error turns dropped — exactly the
   context v1 itself replays for KV reuse. The scored tokens are the
   re-rendered assistant segment through its first EOS (<|im_end|>);
   template glue after EOS is fed to keep the cache honest but never
   scored. This is the same approximation family the deployment commits
   to; exact parity with administration additionally requires the deployed
   :system-prompt, :tools, and :enable-thinking? to be passed in opts.

   Two paths, one law:
   - session-scores: ONE owned-branch walk over the whole session,
     delta-prefilling each turn like the provider and scoring spans via
     backend/forward-branch-scores (the one-forward primitive).
   - turn-assess: the GFI face — p/assess of (make-llm-gf model-map) at the
     rendered prefix with the turn's tokens as a :t0.. choicemap
     (turn-choicemap = the JSONL->choicemap converter). The agent turn IS
     the LLM generative function at the rendered prompt; no new GF type.
   LAW: turn-assess weight == the session-scores :logprob for that turn
   (same math, different graph shape — the compiled-vs-handler discipline).

   Sync math, async events: rendering (applyChatTemplate) is a genuine
   tokenizer I/O boundary and returns promises; the scoring core
   score-rendered-turns is synchronous."
  (:require [genmlx.choicemap :as cm]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as lcore]
            [genmlx.llm.pi-session :as ps]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [promesa.core :as pr]))

(defn- shared-prefix-len [a b]
  (let [n (min (count a) (count b))]
    (loop [i 0]
      (if (and (< i n) (== (nth a i) (nth b i)))
        (recur (inc i))
        i))))

;; ---------------------------------------------------------------------------
;; rendering (async: applyChatTemplate is a tokenizer I/O boundary)
;; ---------------------------------------------------------------------------

(defn render-turn
  "Render assistant turn `k` of `messages` (a path->messages vector):
   resolves to {:index k :prefix P :full F :span [s e] :parity? bool} where
   P = applyChatTemplate(messages[0..k-1], addGen TRUE, tools, think?) —
   the generation prompt the model decoded from — and
   F = applyChatTemplate(messages[0..k], addGen FALSE, tools, think?) —
   the committed render including the turn. s = |shared-prefix(P, F)|
   (:parity? true iff P is a full prefix of F), e = one past the first
   EOS at/after s (template glue after <|im_end|> is never scored), or
   |F| when no EOS occurs."
  [tokenizer messages k {:keys [tools enable-thinking?]}]
  (let [eos (llm/eos-token-id tokenizer)
        tls (or tools js/undefined)
        th? (boolean enable-thinking?)]
    (pr/let [pre (.applyChatTemplate tokenizer
                                     (ps/messages->js (subvec messages 0 k))
                                     true tls th?)
             ful (.applyChatTemplate tokenizer
                                     (ps/messages->js (subvec messages 0 (inc k)))
                                     false tls th?)]
      (let [p (vec (js/Array.from pre))
            f (vec (js/Array.from ful))
            s (shared-prefix-len p f)
            e (loop [i s]
                (cond
                  (>= i (count f)) (count f)
                  (== (nth f i) eos) (inc i)
                  :else (recur (inc i))))]
        {:index k :prefix p :full f :span [s e] :parity? (= s (count p))}))))

(defn- render-turns
  "Render every assistant turn sequentially (one native template call at a
   time — the renderer is shared state); resolves to the vector of
   render-turn maps."
  [tokenizer messages ks opts]
  (pr/loop [ks (seq ks), acc []]
    (if-not ks
      acc
      (pr/let [r (render-turn tokenizer messages (first ks) opts)]
        (pr/recur (next ks) (conj acc r))))))

;; ---------------------------------------------------------------------------
;; the scoring walk (sync core)
;; ---------------------------------------------------------------------------

(defn- score-rendered-turns
  "Walk one owned branch over the rendered turns, provider-style: when a
   turn's full render extends the committed tokens, delta-prefill just the
   suffix (:cached = the reused prefix); otherwise dispose and rebuild
   from scratch (:cached 0) — the provider's own rule, value-identical.
   The branch is always disposed. Returns one map per turn."
  [model renders {:keys [chunk]}]
  (let [branch* (volatile! (llm/owned-branch! model {:cache nil :offset 0}))]
    (try
      (loop [rs (seq renders), committed [], out []]
        (if-not rs
          out
          (let [{:keys [index full span parity?]} (first rs)
                [s e]   span
                shared  (shared-prefix-len committed full)
                append? (and (= shared (count committed))
                             (< shared (count full))
                             (> s (count committed)))
                base    (if append?
                          (count committed)
                          (do (llm/dispose-branch! model @branch*)
                              (vreset! branch* (llm/owned-branch!
                                                model {:cache nil :offset 0}))
                              0))
                _ (when (<= s base)
                    (throw (ex-info "pi-assess: span starts at the origin — nothing conditions the first action token"
                                    {:genmlx/error :span-at-origin :turn index})))
                d       (subvec full base)
                scores  (llm/forward-branch-scores model @branch* d
                                                   (when chunk {:chunk chunk}))
                host    (vec (mx/->clj scores))
                ;; token at absolute position j scores at d-index (j - base - 1)
                per-tok (subvec host (- s base 1) (- e base 1))]
            (recur (next rs) full
                   (conj out {:index     index
                              :logprob   (reduce + 0.0 per-tok)
                              :n-tokens  (- e s)
                              :tokens    (subvec full s e)
                              :per-token per-tok
                              :cached    base
                              :parity?   parity?})))))
      (finally
        (llm/dispose-branch! model @branch*)))))

(defn session-scores
  "Per-turn log-probs of the actions actually taken in an administered
   session: `messages` is a pi-session/path->messages vector; resolves to
   one map per assistant turn —
   {:index :logprob :n-tokens :tokens :per-token :cached :parity?}.
   opts {:system-prompt handled upstream in path->messages; :tools —
   native ToolDefinition JS array, pass the DEPLOYED toolset for exact
   parity; :enable-thinking?; :chunk}. Image-bearing sessions are a typed
   error (VLM replay is a follow-up); zero assistant turns resolve to []."
  ([model-map messages] (session-scores model-map messages {}))
  ([model-map messages opts]
   (when (some :images messages)
     (throw (ex-info "pi-assess: image-bearing sessions are not scoreable in v1 (VLM replay is a follow-up)"
                     {:genmlx/error :images-unsupported})))
   (let [{:keys [model tokenizer]} model-map
         ks (ps/assistant-indices messages)]
     (if (empty? ks)
       (pr/resolved [])
       (pr/let [renders (render-turns tokenizer messages ks opts)]
         (score-rendered-turns model renders opts))))))

;; ---------------------------------------------------------------------------
;; the GFI face
;; ---------------------------------------------------------------------------

(defn turn-choicemap
  "Token ids -> the {:t0 .. :tN} choicemap of the LLM generative function —
   the JSONL->choicemap converter of the L3 epic."
  [token-ids]
  (apply cm/choicemap
         (mapcat (fn [i id] [(keyword (str "t" i)) (mx/scalar id mx/int32)])
                 (range) token-ids)))

(defn turn-assess
  "p/assess of assistant turn `k` through the GFI: the turn's rendered
   conditioning context (F[0..s), which equals the generation prompt when
   :parity?) becomes the LLM GF's prompt args and its span tokens the
   :t0.. choicemap. Resolves to {:weight <MLX scalar> :retval :tokens
   :span :parity?}. Uses the model-internal cache (make-llm-gf), not the
   branch ledger — do not interleave with a live provider turn.
   LAW: (mx/item :weight) == the session-scores :logprob for turn k."
  ([model-map messages k] (turn-assess model-map messages k {}))
  ([model-map messages k opts]
   (pr/let [{:keys [full span parity?]} (render-turn (:tokenizer model-map)
                                                     messages k opts)]
     (let [[s e]  span
           prompt (subvec full 0 s)
           toks   (subvec full s e)
           gf     (lcore/make-llm-gf model-map)
           res    (p/assess gf [prompt (count toks)] (turn-choicemap toks))]
       {:weight  (:weight res)
        :retval  (:retval res)
        :tokens  toks
        :span    span
        :parity? parity?}))))
