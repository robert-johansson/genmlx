(ns genmlx.llm.pi-edit
  "Trajectory EDITING on administered pi sessions (genmlx-5v23, L3-C2):
   'step 3 was wrong, resample from there' — pick a boundary inside an
   assistant turn, resample the suffix from the model, and rejoin pi's
   session tree as a real branch.

   GFI meaning: on an autoregressive token GF the dependence cone of site
   :t_b is exactly the suffix, so cone-restricted regenerate at a token
   boundary IS resample-from-boundary. The manual branch-ledger decode
   below is that operation without reconstructing the old trace (whose
   retained-prefix weight terms cancel to 0 by the retained-only law).

   Boundaries: nil (whole turn) | {:token i} (span-relative token index)
   | {:tool-call j} (0-indexed: resample from where the j-th complete
   <tool_call> block opens — 'the j-th action was wrong').

   REJOIN discipline (pi branchWithSummary shape, append-only): a
   branch_summary entry at the rejoin point (parentId = the edited turn's
   parent; fromId = the abandoned leaf) followed by the new assistant
   message entry. Default writing mode FORKS the session (pi forkFrom
   semantics: new file, header parentSession = source, original entry
   lines copied verbatim) so the administered artifact is never mutated;
   {:in-place? true} appends to the source instead — also legal pi
   semantics, since branching never deletes.

   The circle closes with C1: pa/session-scores runs unchanged on the
   edited session (assess-the-edit)."
  (:require [clojure.string :as str]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.pi-assess :as pa]
            [genmlx.llm.pi-provider :as pp]
            [genmlx.llm.pi-session :as ps]
            [genmlx.llm.sampling :as samp]
            [genmlx.llm.toolcall :as tc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [promesa.core :as pr]
            ["crypto" :as crypto]
            ["fs" :as fs]
            ["path" :as path]))

;; ---------------------------------------------------------------------------
;; boundaries (pure text machinery)
;; ---------------------------------------------------------------------------

(defn tool-call-spans
  "Char spans [{:start :end}] of COMPLETE <tool_call>\\n...</tool_call>
   blocks in `text` — the parse-tool-calls delimiter discipline; an
   unterminated block contributes no span."
  [text]
  (loop [from 0, spans []]
    (let [open (.indexOf text "<tool_call>\n" from)]
      (if (neg? open)
        spans
        (let [close (.indexOf text "</tool_call>" (+ open 12))]
          (if (neg? close)
            spans
            (recur (+ close 12)
                   (conj spans {:start open :end (+ close 12)}))))))))

(defn strip-tool-calls
  "`text` with complete tool-call block spans removed and trailing
   whitespace trimmed — the prose a pi assistant text part stores."
  [text]
  (loop [pos 0, spans (seq (tool-call-spans text)), acc ""]
    (if-not spans
      (str/trimr (str acc (subs text pos)))
      (let [{:keys [start end]} (first spans)]
        (recur end (next spans) (str acc (subs text pos start)))))))

(defn token-char-offsets
  "Cumulative decoded char length after each token (a vector as long as
   `tokens`), via the provider's incremental-detok discipline. A held-back
   partial char keeps the count at the last flush point, so char->token
   mapping ROUNDS DOWN across a hold; resolves to the vector."
  [tokenizer tokens]
  (let [decode-fn (fn [toks]
                    (llm/decode tokenizer
                                (js/Uint32Array.from (into-array toks))))]
    (pr/loop [i 0, pending [], total 0, acc []]
      (if (= i (count tokens))
        acc
        (pr/let [{:keys [piece pending]} (pp/detok-step decode-fn pending
                                                        (nth tokens i))]
          (let [total' (+ total (count piece))]
            (pr/recur (inc i) pending total' (conj acc total'))))))))

(defn- check-turn! [messages k]
  (let [msg (nth messages k nil)]
    (when-not (map? msg)
      (throw (ex-info (str "pi-edit: no message at index " k)
                      {:genmlx/error :bad-turn-index :turn k
                       :message-count (count messages)})))
    (when-not (= "assistant" (:role msg))
      (throw (ex-info (str "pi-edit: message " k " is a " (:role msg)
                           " turn — only assistant turns are editable")
                      {:genmlx/error :not-assistant :turn k})))
    (when-not (ps/message-entry-id msg)
      (throw (ex-info (str "pi-edit: message " k " is synthetic (no source "
                           "session entry) — not editable")
                      {:genmlx/error :not-editable :turn k})))
    msg))

(defn resolve-boundary
  "Resolve `boundary` for assistant turn `k` against its render; resolves
   to {:render <pa/render-turn map> :boundary-token b :keep-tokens :exact?}.
   b is span-relative; keep-tokens = span[0..b)."
  [tokenizer messages k boundary opts]
  (check-turn! messages k)
  (pr/let [render (pa/render-turn tokenizer messages k opts)]
    (let [[s e]     (:span render)
          span-toks (subvec (:full render) s e)]
      (cond
        (nil? boundary)
        {:render render :boundary-token 0 :keep-tokens [] :exact? true}

        (contains? boundary :token)
        (let [i (:token boundary)]
          (when-not (and (int? i) (<= 0 i (count span-toks)))
            (throw (ex-info (str "pi-edit: boundary token " i
                                 " outside the turn span (0.."
                                 (count span-toks) ")")
                            {:genmlx/error :bad-boundary-token
                             :token i :span-tokens (count span-toks)})))
          {:render render :boundary-token i
           :keep-tokens (subvec span-toks 0 i) :exact? true})

        (contains? boundary :tool-call)
        (let [j       (:tool-call boundary)
              eos     (llm/eos-token-id tokenizer)
              txt-tks (if (and (seq span-toks) (== eos (peek span-toks)))
                        (pop span-toks)
                        span-toks)]
          (pr/let [text (llm/decode tokenizer
                                    (js/Uint32Array.from (into-array txt-tks)))
                   offs (token-char-offsets tokenizer txt-tks)]
            (let [spans (tool-call-spans text)]
              (when-not (and (int? j) (< -1 j (count spans)))
                (throw (ex-info (str "pi-edit: turn " k " has "
                                     (count spans) " tool-call block(s); "
                                     "no block " j)
                                {:genmlx/error :no-such-tool-call
                                 :tool-call j :available (count spans)})))
              (let [cpos (:start (nth spans j))
                    b    (count (take-while #(<= % cpos) offs))]
                {:render render :boundary-token b
                 :keep-tokens (subvec span-toks 0 b)
                 :exact? (= cpos (if (pos? b) (nth offs (dec b)) 0))}))))

        :else
        (throw (ex-info (str "pi-edit: unknown boundary " (pr-str boundary))
                        {:genmlx/error :bad-boundary :boundary boundary}))))))

;; ---------------------------------------------------------------------------
;; resample (the regenerate)
;; ---------------------------------------------------------------------------

(defn resample-turn
  "Resample assistant turn `k` of `messages` from `boundary` onward.
   opts: sampling (:temperature default 0 — editing defaults to greedy —
   :top-k :top-p :min-p :repetition-penalty :presence-penalty), :max-new
   (default 512), :key (default rng/fresh-key), plus the render opts
   (:tools :enable-thinking?) and :chunk for the context prefill.

   Resolves to {:turn :source-entry-id :boundary-token :exact?
   :kept-tokens :sampled-tokens (incl. terminal eos) :new-tokens (kept ++
   sampled, eos excluded) :text :prose :tool-calls :tool-call-errors
   :finish-reason (stop|length, or toolUse when a stop turn carries
   calls — the provider's mapping) :suffix-logprob (temp-1 unpenalized
   model logprob of the sampled tokens, float32 softmax — comparable to
   forward-branch-scores) :usage {:input :output}}."
  ([model-map messages k boundary]
   (resample-turn model-map messages k boundary {}))
  ([model-map messages k boundary opts]
   (let [{:keys [model tokenizer]} model-map
         msg (check-turn! messages k)
         eid (ps/message-entry-id msg)]
     (when (some :images messages)
       (throw (ex-info "pi-edit: image-bearing sessions are not editable in v1 (VLM replay is a follow-up)"
                       {:genmlx/error :images-unsupported})))
     (pr/let [{:keys [render boundary-token keep-tokens exact?]}
              (resolve-boundary tokenizer messages k boundary opts)]
       (let [s       (first (:span render))
             prefix  (subvec (:full render) 0 s)
             ctx     (into prefix keep-tokens)
             eos     (llm/eos-token-id tokenizer)
             max-new (or (:max-new opts) 512)
             scfg    (merge {:temperature 0}
                            (select-keys opts [:temperature :top-k :top-p
                                               :min-p :repetition-penalty
                                               :presence-penalty]))
             branch  (llm/owned-branch! model {:cache nil :offset 0})]
         (-> (pr/let
              [decoded
               (let [logits0 (llm/forward-branch-tokens
                              model branch ctx
                              (when (:chunk opts) {:chunk (:chunk opts)}))
                     ;; sync decode loop: sample under scfg, score under the
                     ;; raw temp-1 distribution (the C1-comparable logprob)
                     {:keys [gen lp reason]}
                     (loop [i 0, logits logits0
                            key (or (:key opts) (rng/fresh-key))
                            gen [], lp 0.0]
                       (let [lsm (mx/log-softmax
                                  (mx/astype logits mx/float32) -1)
                             [tok key'] (samp/sample-token key logits scfg gen)
                             lp'  (+ lp (mx/item (mx/index lsm tok)))
                             gen' (conj gen tok)]
                         (llm/sweep-tick! i 32)
                         (cond
                           (== tok eos)             {:gen gen' :lp lp'
                                                     :reason "stop"}
                           (>= (count gen') max-new) {:gen gen' :lp lp'
                                                      :reason "length"}
                           :else (recur (inc i)
                                        (llm/forward-branch model branch tok)
                                        key' gen' lp'))))
                     new-toks (into (vec keep-tokens)
                                    (if (= reason "stop") (pop gen) gen))]
                 {:gen gen :lp lp :reason reason :new-toks new-toks})
               text (llm/decode tokenizer
                                (js/Uint32Array.from
                                 (into-array (:new-toks decoded))))]
              (let [{:keys [calls errors]} (tc/parse-tool-calls text)
                    reason (if (and (= "stop" (:reason decoded)) (seq calls))
                             "toolUse"
                             (:reason decoded))]
                {:turn k
                 :source-entry-id eid
                 :boundary-token boundary-token
                 :exact? exact?
                 :kept-tokens keep-tokens
                 :sampled-tokens (:gen decoded)
                 :new-tokens (:new-toks decoded)
                 :text text
                 :prose (strip-tool-calls text)
                 :tool-calls calls
                 :tool-call-errors errors
                 :finish-reason reason
                 :suffix-logprob (:lp decoded)
                 :usage {:input (count prefix)
                         :output (count (:gen decoded))}}))
             (pr/handle (fn [r e]
                          (llm/dispose-branch! model branch)
                          (if e (throw e) r)))))))))

;; ---------------------------------------------------------------------------
;; rejoin (pure plan + effectful write)
;; ---------------------------------------------------------------------------

(defn- fresh-id []
  (-> (crypto/randomUUID) (str/replace "-" "") (subs 0 8)))

(defn plan-edit
  "Build the two pi entries that rejoin `resample` into `session`'s tree:
   a branch_summary at the rejoin point (fromId = the abandoned leaf) and
   the new assistant message. Pure. opts {:summary :model :provider :api}
   override the scraped model_change identity. Returns {:entries [e1 e2]
   :leaf-id}."
  ([session resample] (plan-edit session resample {}))
  ([session resample {:keys [summary model provider api]}]
   (let [by-id (into {} (map (juxt :id identity)) (:entries session))
         src   (get by-id (:source-entry-id resample))]
     (when-not src
       (throw (ex-info (str "pi-edit: source entry " (:source-entry-id resample)
                            " not in this session")
                       {:genmlx/error :not-editable
                        :id (:source-entry-id resample)})))
     (let [leaf-id   (:id (peek (:entries session)))
           mc        (->> (ps/leaf-path session)
                          (filter #(= "model_change" (:type %)))
                          last)
           provider' (or provider (some-> mc :js .-provider) "genmlx")
           model'    (or model (some-> mc :js .-modelId) "unknown")
           now       (.toISOString (js/Date.))
           sum-id    (fresh-id)
           msg-id    (fresh-id)
           prose     (:prose resample)
           content   (into (if (or (seq prose) (empty? (:tool-calls resample)))
                             [{:type "text" :text prose}]
                             [])
                           (map-indexed
                            (fn [i c] {:type "toolCall"
                                       :id (str "call_edit_" (inc i))
                                       :name (:name c)
                                       :arguments (:args c)})
                            (:tool-calls resample)))
           in        (get-in resample [:usage :input] 0)
           out       (get-in resample [:usage :output] 0)]
       {:leaf-id msg-id
        :entries
        [{:type "branch_summary" :id sum-id :parentId (:parent-id src)
          :timestamp now :fromId leaf-id
          :summary (or summary
                       (str "genmlx pi-edit: resampled turn "
                            (:source-entry-id resample) " from token "
                            (:boundary-token resample) " ("
                            (count (:kept-tokens resample)) " kept)"))}
         {:type "message" :id msg-id :parentId sum-id :timestamp now
          :message {:role "assistant"
                    :api (or api provider')
                    :provider provider'
                    :model model'
                    :stopReason (:finish-reason resample)
                    :usage {:input in :output out
                            :cacheRead 0 :cacheWrite 0 :reasoning 0
                            :totalTokens (+ in out)
                            :cost {:input 0 :output 0 :cacheRead 0
                                   :cacheWrite 0 :total 0}}
                    :timestamp (js/Date.now)
                    :content content}}]}))))

(defn- entry->line [e] (js/JSON.stringify (clj->js e)))

(defn write-edit!
  "Write a planned edit. Default: a FORKED session file (new uuid,
   pi-shaped filename, header parentSession = source, original entry
   lines verbatim, then the new lines) in the source dir or :out-dir.
   {:in-place? true}: append the new lines to the source file. Returns
   the written file's path."
  [source-file plan {:keys [in-place? out-dir]}]
  (let [lines (map entry->line (:entries plan))]
    (if in-place?
      (let [text (.readFileSync fs source-file "utf8")
            sep  (if (str/ends-with? text "\n") "" "\n")]
        (fs/appendFileSync source-file
                           (str sep (str/join "\n" lines) "\n"))
        source-file)
      (let [text    (.readFileSync fs source-file "utf8")
            session (ps/parse-session text)
            entry-lines (->> (str/split-lines text)
                             (remove str/blank?)
                             rest)
            now     (.toISOString (js/Date.))
            uuid    (crypto/randomUUID)
            header  {:type "session"
                     :version (or (get-in session [:header :version]) 3)
                     :id uuid
                     :timestamp now
                     :cwd (get-in session [:header :cwd])
                     :parentSession source-file}
            fname   (str (str/replace now #"[:.]" "-") "_" uuid ".jsonl")
            out     (path/join (or out-dir (path/dirname source-file)) fname)]
        (fs/writeFileSync
         out
         (str (str/join "\n" (concat [(entry->line header)]
                                     entry-lines
                                     lines))
              "\n"))
        out))))

(defn edit-session!
  "The full C2 operation: read `session-file`, resample assistant turn `k`
   (a path->messages index, as scripts/session_scores.cljs prints) from
   `boundary`, and write the rejoined branch. opts = resample-turn opts ++
   plan-edit opts ++ {:system-prompt :in-place? :out-dir}. Resolves to
   {:file :leaf-id :entries :turn <resample result>}."
  ([model-map session-file k boundary]
   (edit-session! model-map session-file k boundary {}))
  ([model-map session-file k boundary opts]
   (let [session  (ps/read-session session-file)
         messages (ps/path->messages (ps/leaf-path session)
                                     {:system-prompt (:system-prompt opts)})]
     (pr/let [res (resample-turn model-map messages k boundary opts)]
       (let [plan (plan-edit session res opts)
             out  (write-edit! session-file plan opts)]
         {:file out
          :leaf-id (:leaf-id plan)
          :entries (:entries plan)
          :turn res})))))
