(ns genmlx.llm.pi-session
  "pi coding-agent session JSONL -> chat messages (genmlx-opwh, L3-C1).

   A pi session file is an append-only entry TREE: line 1 is the header
   {type \"session\" id timestamp cwd parentSession?}; every later line is
   one entry {type id parentId timestamp ...}. Appends always child the
   current leaf, so the LAST line is the live leaf; branching moves the
   leaf to an earlier entry without deleting anything. Files live under
   ~/.mlx-node/agent/sessions/<encoded-cwd>/.

   path->messages MIRRORS the v1 provider's pi-Context conversion
   (mlx-node packages/agent/src/provider/convert-messages.ts is the
   NORMATIVE reference — its byte-stable joins are what keep the replayed
   KV prefix stable at administration time): text parts \\n-joined,
   thinking blocks dropped, error/aborted assistant turns dropped (their
   tool calls untracked), orphaned tool calls repaired with synthetic
   error results, tool-result images hoisted onto a synthetic user
   message with fixed text. The mirror is what makes RENDER PARITY
   possible: a session replayed through path->messages + the same
   applyChatTemplate re-renders the token stream the provider committed.

   Entries keep the raw parsed JS object under :js — message payloads are
   consumed via interop so tool-call `arguments` re-stringify byte-stably
   (JSON.stringify preserves the original key order of the parse).

   Pure: no fs writes, no GPU, no model. read-session is the one fs READ."
  (:require [clojure.string :as str]
            ["fs" :as fs]))

;; ---------------------------------------------------------------------------
;; parsing
;; ---------------------------------------------------------------------------

(defn- parse-line [line lineno]
  (try
    (js/JSON.parse line)
    (catch :default e
      (throw (ex-info (str "pi-session: malformed JSONL at line " lineno
                           ": " (ex-message e))
                      {:genmlx/error :malformed-session-line :line lineno})))))

(defn- entry-map [o]
  {:type      (.-type o)
   :id        (.-id o)
   :parent-id (when-not (nil? (.-parentId o)) (.-parentId o))
   :timestamp (.-timestamp o)
   :js        o})

(defn parse-session
  "Parse a pi session file's text into {:header m :entries [m...]}.
   The first non-blank line must be the {type \"session\"} header (typed
   error :bad-session-header otherwise); blank lines are skipped; a
   malformed JSON line is a typed error naming its 1-based line number."
  [text]
  (let [numbered (->> (str/split-lines text)
                      (map-indexed (fn [i l] [(inc i) l]))
                      (remove (fn [[_ l]] (str/blank? l))))]
    (when (empty? numbered)
      (throw (ex-info "pi-session: empty session file"
                      {:genmlx/error :bad-session-header})))
    (let [[[hline htext]] [(first numbered)]
          h (parse-line htext hline)]
      (when-not (= "session" (.-type h))
        (throw (ex-info (str "pi-session: first line is not a session header "
                             "(type " (pr-str (.-type h)) ")")
                        {:genmlx/error :bad-session-header})))
      {:header {:id             (.-id h)
                :version        (.-version h)
                :timestamp      (.-timestamp h)
                :cwd            (.-cwd h)
                :parent-session (when-not (nil? (.-parentSession h))
                                  (.-parentSession h))}
       :entries (mapv (fn [[n l]] (entry-map (parse-line l n)))
                      (rest numbered))})))

(defn read-session
  "Read + parse a pi session JSONL file."
  [path]
  (parse-session (.readFileSync fs path "utf8")))

;; ---------------------------------------------------------------------------
;; the entry tree
;; ---------------------------------------------------------------------------

(defn leaf-path
  "The live path of `session`: entries root->leaf, where the leaf is the
   LAST entry in file order (pi appends child the current leaf, so the
   last line is always live). Typed error :broken-session-tree on a
   missing parent or a parent cycle."
  [session]
  (let [entries (:entries session)]
    (if (empty? entries)
      []
      (let [by-id (into {} (map (juxt :id identity)) entries)]
        (loop [e (peek entries), acc (list e), seen #{(:id e)}]
          (if-let [pid (:parent-id e)]
            (let [p (get by-id pid)]
              (cond
                (nil? p)
                (throw (ex-info (str "pi-session: entry " (:id e)
                                     " references missing parent " pid)
                                {:genmlx/error :broken-session-tree :id (:id e)}))
                (contains? seen (:id p))
                (throw (ex-info (str "pi-session: parent cycle at " (:id p))
                                {:genmlx/error :broken-session-tree :id (:id p)}))
                :else (recur p (conj acc p) (conj seen (:id p)))))
            (vec acc)))))))

(defn session-tree
  "The full entry tree as nested {:entry e :children [...]} nodes, roots
   first (an entry with a nil or unknown parent is a root). The C2/C3
   substrate: branch points are nodes with multiple children."
  [session]
  (let [entries (:entries session)
        ids     (into #{} (map :id) entries)
        kids    (group-by :parent-id entries)
        root?   (fn [e] (or (nil? (:parent-id e))
                            (not (contains? ids (:parent-id e)))))
        build   (fn build [e]
                  {:entry e
                   :children (mapv build (get kids (:id e) []))})]
    (mapv build (filter root? entries))))

;; ---------------------------------------------------------------------------
;; pi Context -> chat messages (the convert-messages.ts mirror)
;; ---------------------------------------------------------------------------

(def tool-image-hoist-text
  "Fixed text of the synthetic user message that carries tool-result images
   (byte-identical to convert-messages.ts TOOL_IMAGE_HOIST_TEXT — it is
   replayed on every turn, so any variation would change the token prefix)."
  "The image output of the preceding tool result is attached.")

(defn- split-parts
  "pi content parts -> {:text <\\n-joined non-image text> :images [Uint8Array]}
   (within-message interleaving is not preserved — text first, images after,
   matching the native Jinja serializer's ordering)."
  [parts]
  (let [xs (array-seq (or parts #js []))]
    {:text   (->> xs
                  (remove #(= "image" (.-type %)))
                  (map #(.-text %))
                  (str/join "\n"))
     :images (->> xs
                  (filter #(= "image" (.-type %)))
                  (mapv #(js/Uint8Array. (js/Buffer.from (.-data %) "base64"))))}))

(defn- convert-user [m]
  (let [c (.-content m)]
    (if (string? c)
      {:role "user" :content c}
      (let [{:keys [text images]} (split-parts c)]
        (cond-> {:role "user" :content text}
          (seq images) (assoc :images images))))))

(defn- convert-assistant [m]
  (let [c (.-content m)]
    (if (string? c)
      {:role "assistant" :content c}
      (let [parts (array-seq (or c #js []))
            text  (->> parts
                       (filter #(= "text" (.-type %)))
                       (map #(.-text %))
                       (str/join "\n"))
            tcs   (->> parts
                       (filter #(= "toolCall" (.-type %)))
                       (mapv (fn [p]
                               {:id        (.-id p)
                                :name      (.-name p)
                                :arguments (js/JSON.stringify (.-arguments p))})))]
        (cond-> {:role "assistant" :content text}
          (seq tcs) (assoc :toolCalls tcs))))))

(defn- convert-tool-result [m]
  (let [{:keys [text]} (split-parts (.-content m))]
    {:role       "tool"
     :content    text
     :toolCallId (.-toolCallId m)
     :isError    (boolean (.-isError m))}))

(defn path->messages
  "Convert a path of entries (leaf-path output) into the chat messages the
   provider would prime — the contextToChatMessages mirror. Only
   :type \"message\" entries render; other entry types are skipped, EXCEPT
   :type \"compaction\" which is a typed error (a compacted context cannot
   be re-rendered faithfully from entries alone). Custom message roles
   (bashExecution etc.) are skipped, mirroring convertMessage's closed
   switch. opts {:system-prompt s} prepends the system message pi holds
   outside the session file."
  ([entries] (path->messages entries {}))
  ([entries {:keys [system-prompt]}]
   (when-let [c (first (filter #(= "compaction" (:type %)) entries))]
     (throw (ex-info "pi-session: compaction entry on the path — re-rendering a compacted context from entries alone is unsupported (follow-up)."
                     {:genmlx/error :compaction-unsupported :id (:id c)})))
   (let [msgs (keep #(when (= "message" (:type %)) (.-message (:js %)))
                    entries)
         flush-orphans (fn [out pending seen]
                         (into out
                               (keep (fn [id]
                                       (when-not (contains? seen id)
                                         {:role "tool"
                                          :content "No result provided"
                                          :toolCallId id
                                          :isError true})))
                               pending))]
     (loop [ms      (seq msgs)
            out     (if system-prompt
                      [{:role "system" :content system-prompt}]
                      [])
            pending []
            seen    #{}]
       (if-not ms
         (flush-orphans out pending seen)
         (let [m (first ms)]
           (case (.-role m)
             "user"
             (recur (next ms)
                    (conj (flush-orphans out pending seen) (convert-user m))
                    [] #{})

             "assistant"
             (let [out' (flush-orphans out pending seen)]
               (if (contains? #{"error" "aborted"} (.-stopReason m))
                 ;; dropped: not rendered, and its tool calls are NOT tracked
                 (recur (next ms) out' [] #{})
                 (let [cm  (convert-assistant m)
                       ids (into [] (keep :id) (:toolCalls cm))]
                   (recur (next ms) (conj out' cm) ids #{}))))

             "toolResult"
             (let [cm     (convert-tool-result m)
                   images (:images (split-parts (.-content m)))
                   out'   (cond-> (conj out cm)
                            ;; hoist WITHOUT flushing orphan state — sibling
                            ;; calls from the same fan-out may still be pending
                            (seq images)
                            (conj {:role "user"
                                   :content tool-image-hoist-text
                                   :images images}))]
               (recur (next ms) out' pending
                      (conj seen (.-toolCallId m))))

             ;; custom roles (bashExecution, custom, branchSummary,
             ;; compactionSummary): skipped, state untouched
             (recur (next ms) out pending seen))))))))

(defn assistant-indices
  "Indices of the assistant messages in a path->messages vector."
  [messages]
  (into [] (keep-indexed (fn [i m] (when (= "assistant" (:role m)) i)))
        messages))

;; ---------------------------------------------------------------------------
;; native ChatMessage interop (the applyChatTemplate input shape)
;; ---------------------------------------------------------------------------

(defn message->js
  "One converted message -> the native ChatMessage JS shape (the
   convert-messages.ts output the Rust renderer takes)."
  [m]
  (let [o #js {:role (:role m) :content (:content m)}]
    (when-let [tcs (:toolCalls m)]
      (set! (.-toolCalls o)
            (into-array (map (fn [tc]
                               #js {:id        (:id tc)
                                    :name      (:name tc)
                                    :arguments (:arguments tc)})
                             tcs))))
    (when-let [id (:toolCallId m)]
      (set! (.-toolCallId o) id))
    (when (contains? m :isError)
      (set! (.-isError o) (:isError m)))
    (when-let [imgs (:images m)]
      (set! (.-images o) (into-array imgs)))
    o))

(defn messages->js [messages]
  (into-array (map message->js messages)))
