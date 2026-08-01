(ns genmlx.memory
  "The PERSISTENCE face of the Bun WORLD membrane (bean genmlx-gsoi, sibling of
   `genmlx.world.net`) — a thin, honest boundary that owns ONE side effect:
   durable read/write to a process-external store via SYNCHRONOUS `bun:sqlite`
   (WAL, prepared statements). Pure flow above; the store is the only mutable,
   externally-observable resource and it is fenced behind a blessed `with-store`
   scope (try/finally close), exactly like `with-server` fences a Bun.serve
   listener.

   WHY THIS — AND ONLY THIS — EARNS A NAMESPACE
   -------------------------------------------------------------------------
   Across-EPISODE persistence (within one process run) is FREE: don't reset the
   fold — the carried accumulator (belief / params / library / experience) is a
   threaded value, the same pattern as a Trace or a PRNG key one timescale up.
   Across-SESSION persistence (process death + restart) is the ONE thing that is
   NOT free: it is an irreducible side effect (a write to a durable store). That
   single effect is all `genmlx.memory` owns, by the same logic that gave the
   control layer its scheduler. De Houwer discipline: this names the MECHANISM
   (memory = retention), not the RELATION (learning = experience->behaviour,
   which the whole composition exhibits and is correctly NOT a module). Memory is
   dumb retention, pure-above, owning durable read/write only — no learning
   algorithm, no abstraction/library-growth, no planning lives here.

   SYNC, NOT ASYNC (the difference from net.cljs)
   -------------------------------------------------------------------------
   `bun:sqlite` is synchronous, so persistence is the SYNC half of the Bun
   membrane: a checkpoint at an episode/session boundary is an ordinary
   synchronous call, consistent with GenMLX's sync core (CLAUDE.md: 'sync math,
   async events'). There is ZERO promesa in this namespace. The network face is
   the async half (a request reaches an autonomous other); persistence writes
   your own values to your own disk, exactly, like eval! — so it stays sync.

   TWO KINDS OF CROSS-SESSION KEY (the addressing discipline)
   -------------------------------------------------------------------------
   Mirroring the GFI choicemap-address discipline (content = leaf-value
   identity; named = the address slot):

     1. CONTENT-ADDRESS  (`content-hash`, `put-content!`): sha256 of the
        CANONICAL serialization, for IMMUTABLE artifacts (Traces, particle
        snapshots, experience entries). Reproducible across process death with
        no registry. CRITICAL: the canonicalization recursively SORTS map keys
        before hashing — JSON.stringify / CLJS map iteration are insertion-order
        dependent, so without this two structurally-equal choicemaps would hash
        differently and a session-50 recall of a session-3 value would silently
        fail.
     2. NAMED-KEY  (\"ns/name\" string): for MUTABLE THREADED SLOTS (the
        parameter store, the current ConceptMemory, the carried accumulator).
        Overwritten each fold (upsert), so it needs a stable program name, not a
        content hash.

   GOTCHAS (empirically confirmed this session)
   -------------------------------------------------------------------------
   - WAL only engages on an ON-DISK database: `PRAGMA journal_mode=WAL` returns
     \"wal\" for a file path but \"memory\" for an in-memory (\":memory:\") db.
   - bun:sqlite BLOB columns return a typed-array-like; serialized payloads are
     stored in a TEXT column (the serializer already emits a JSON / EDN string).
   - records print with a reader tag `read-string` cannot parse, so EDN payloads
     (ConceptMemory, experience entries) are flattened to plain maps first
     (`deep-plain`); they restore as plain data, not as the original record type.

   Composes `genmlx.serialize` (the value<->data codecs) and reconstructs a
   `VectorizedTrace` for particle sets. Requires nothing from control/agents and
   is never itself a generative function."
  (:require [genmlx.serialize :as ser]
            [genmlx.trace :as tr]
            [genmlx.vectorized :as vect]
            [clojure.string :as str]
            [cljs.reader :as reader]))

(def ^:private sqlite
  (try (js/require "bun:sqlite") (catch :default _ nil)))

(def ^:private Database (some-> sqlite .-Database))

(def ^:private node-crypto
  (try (js/require "node:crypto") (catch :default _ nil)))

(def schema-version-v
  "The current durable schema version this build writes. `migrate!` is an honest
   no-op at v1; the seam exists so a future version can transform v1 stores."
  1)

(defn available?
  "True when the bun:sqlite + node:crypto backends this face needs are present
   (running under `bun`). Pure flow above can branch on this to skip cleanly
   off-Bun, mirroring `genmlx.world.net/available?`."
  []
  (boolean (and Database node-crypto)))

;; ===========================================================================
;; Canonical serialization + content hashing (THE load-bearing law)
;; ===========================================================================
;;
;; A content address must be a pure function of the LOGICAL value, independent
;; of how the map happened to be built. JSON.stringify and CLJS map iteration
;; are both insertion-order dependent, so we serialize through `canonical-str`,
;; which recursively sorts map keys (and sorts set elements). Two
;; structurally-equal values therefore produce byte-identical strings and
;; identical hashes — the property cross-session recall relies on.

(defn- kname
  "Stable string for a map key (keyword -> its name without the colon; any
   other key -> its string form)."
  [k]
  (if (keyword? k) (subs (str k) 1) (str k)))

;; ---------------------------------------------------------------------------
;; The NON-FINITE vocabulary — ONE definition, used by BOTH the content hash
;; and the durable payload codec (genmlx-94tu)
;; ---------------------------------------------------------------------------
;; `JSON.stringify` collapses EVERY non-finite number to `null`, and `mx/array`'s
;; `Float32Array.from` silently coerces `null` back to `0.0`. Measured
;; end-to-end on this codebase: a VectorizedTrace `:weight` of `[-Inf -2 +Inf]`
;; round-tripped through `save-particles!` / `restore-particles` as `[0 -2 0]`.
;; For an SMC log-weight that is maximally perverse — `-Inf` means "impossible
;; particle" and `0.0` means "log-weight 0", so after logsumexp normalization the
;; DEAD particle restores as the single BEST one.
;;
;; These three tokens were originally introduced for the content hash only
;; (`canonical-str`); the durable payload codec below now shares them rather
;; than inventing a second spelling.

;; Vocabulary is owned by `genmlx.serialize` (the payload codec). Named here
;; only so the hashing call sites below read the same as before.
(def ^:private nan-token     "#nan")
(def ^:private pos-inf-token "#+inf")
(def ^:private neg-inf-token "#-inf")

(def ^:private token->non-finite ser/non-finite-tokens)

(defn- non-finite-token
  "The stable token for a non-finite number. NaN/+Inf/-Inf get DISTINCT tokens
   so they never alias each other (or nil), which plain JSON does."
  [x]
  (cond (js/Number.isNaN x) nan-token
        (pos? x)            pos-inf-token
        :else               neg-inf-token))

(defn canonical-str
  "Order-independent canonical string for a CLJS data value: map keys sorted
   lexicographically, vectors/seqs kept in order, set elements sorted by their
   own canonical form, scalars escaped via JSON.stringify. Works on both parsed
   JSON (string keys) and EDN data (keyword keys). The basis of content
   addressing."
  [x]
  (cond
    (map? x)
    (str "{"
         (->> x
              (sort-by (comp kname key))
              (map (fn [[k v]] (str (js/JSON.stringify (kname k)) ":" (canonical-str v))))
              (str/join ","))
         "}")

    (set? x)
    ;; '#set' prefix keeps a set DISJOINT from a vector of the same elements —
    ;; both would otherwise render "[...]" and a set would falsely content-hash
    ;; equal to a vector (they are not =).
    (str "#set[" (str/join "," (sort (map canonical-str x))) "]")

    (or (vector? x) (seq? x))
    (str "[" (str/join "," (map canonical-str x)) "]")

    ;; '#kw' prefix keeps a keyword VALUE disjoint from the same-named string
    ;; value. NOTE: map KEYS deliberately collapse keyword<->string (see the
    ;; map branch above, via kname) so a JSON artifact ({"a":..}) and the EDN it
    ;; encodes ({:a ..}) address identically — JSON object keys are always
    ;; strings, so that cross-codec equivalence is required. VALUES carry no such
    ;; equivalence (serialize encodes a keyword value as a tagged map, never a
    ;; bare string), so distinguishing them here only removes a false collision.
    (keyword? x) (str "#kw" (js/JSON.stringify (kname x)))
    (string? x)  (js/JSON.stringify x)
    ;; JSON.stringify collapses EVERY non-finite to "null", aliasing NaN/±Inf
    ;; with each other and with nil; emit distinct tokens so a score of -Inf
    ;; (an impossible-event log-prob) never content-hashes equal to nil.
    ;; Same vocabulary the durable payload codec uses (`non-finite-token`).
    (number? x)  (if (js/isFinite x)
                   (js/JSON.stringify x)
                   (non-finite-token x))
    (boolean? x) (str x)
    (nil? x)     "null"
    :else        (js/JSON.stringify (clj->js x))))

(defn- sha256-hex
  [s]
  (-> (.createHash node-crypto "sha256")
      (.update s "utf8")
      (.digest "hex")))

(defn content-hash
  "Content address of a value: `\"sha256:<hex>\"` over its canonical
   serialization. Insertion-order independent and cross-process reproducible —
   the same logical value always yields the same key, with no registry."
  [value]
  (str "sha256:" (sha256-hex (canonical-str value))))

(defn content-hash-json
  "Content address of a JSON payload STRING (e.g. the output of
   `genmlx.serialize/save-choices`). Parses, canonicalizes, hashes — so
   whitespace and key order in the source JSON do not affect the key."
  [json-str]
  (content-hash (js->clj (js/JSON.parse json-str))))

;; ===========================================================================
;; Store lifecycle — the RESOURCE BOUNDARY (mirrors net.cljs's with-server)
;; ===========================================================================

(def ^:private create-objects-sql
  ;; payload is TEXT: the serializer emits a JSON / EDN string (BLOB would come
  ;; back as a typed-array-like). score_type records the trace score encoding
  ;; (:joint / :marginal / :collapsed) so restore can refuse a :collapsed trace.
  "CREATE TABLE IF NOT EXISTS objects (
     key        TEXT PRIMARY KEY,
     kind       TEXT NOT NULL,
     payload    TEXT NOT NULL,
     score_type TEXT,
     created_at INTEGER NOT NULL
   )")

(def ^:private create-meta-sql
  "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")

(def ^:private create-kind-index-sql
  "CREATE INDEX IF NOT EXISTS idx_objects_kind_created ON objects (kind, created_at)")

(defn- now-ms [] (js/Date.now))

(defn- meta-get [db k]
  (some-> (.get (.query db "SELECT v FROM meta WHERE k = ?") k) .-v))

(defn- meta-put! [db k v]
  (.run (.prepare db "INSERT INTO meta (k, v) VALUES (?, ?)
                      ON CONFLICT(k) DO UPDATE SET v = excluded.v")
        k (str v)))

(defn open-store
  "Open (creating if absent) a durable store at `path` and return a store map
   `{:db <Database> :path path}`. Sets WAL journaling (effective on-disk only),
   creates the `objects` + `meta` tables and the kind index, and stamps the
   schema version into `meta` on first creation.

   `path` may be a filesystem path (durable, across-process) or \":memory:\"
   (ephemeral, for tests that don't need persistence). Opts: `:read-only?`."
  ([path] (open-store path {}))
  ([path {:keys [read-only?]}]
   (when-not (available?)
     (throw (ex-info "genmlx.memory unavailable: bun:sqlite / node:crypto not present (run under bun)"
                     {:genmlx/error :memory-unavailable})))
   (let [db (if read-only?
              (Database. path #js {:readonly true})
              (Database. path))]
     (when-not read-only?
       (.run db "PRAGMA journal_mode = WAL")
       (.run db create-objects-sql)
       (.run db create-meta-sql)
       (.run db create-kind-index-sql)
       (when (nil? (meta-get db "schema_version"))
         (meta-put! db "schema_version" schema-version-v)))
     {:db db :path path})))

(defn close-store!
  "Close the underlying database, releasing the OS resource (and flushing the
   WAL). Idempotent: closing an already-closed store is a silent no-op in
   bun:sqlite (verified), so `with-store`'s finally may safely close a store the
   body already closed."
  [{:keys [db]}]
  (.close db))

(defn with-store
  "[blessed scope] Open the store at `path`, call `(f store)`, and GUARANTEE the
   database is closed afterwards — on success OR throw (try/finally). Returns
   the value of `(f store)`. This is the only place a store lifecycle should
   live; a leaked handle keeps a WAL file open. SYNCHRONOUS (no promesa): the
   sqlite backend is synchronous, so unlike `with-server` this does not return a
   promise."
  ([path f] (with-store path {} f))
  ([path opts f]
   (let [store (open-store path opts)]
     (try (f store)
          (finally (close-store! store))))))

(defn journal-mode
  "The store's active SQLite journal mode (\"wal\" on-disk, \"memory\" for an
   in-memory db). Exposed so callers/tests can confirm WAL engaged."
  [{:keys [db]}]
  (some-> (.get (.query db "PRAGMA journal_mode")) .-journal_mode))

(defn schema-version
  "The durable schema version recorded in the store's `meta` table."
  [{:keys [db]}]
  (some-> (meta-get db "schema_version") js/parseInt))

(defn migrate!
  "Versioning seam. At schema v1 this is an HONEST no-op: there is no prior
   durable format to transform. Returns `{:from v :to v :migrated false}`. A
   future schema bump implements the real transform here rather than silently
   reading an incompatible store."
  [store]
  (let [v (schema-version store)]
    {:from v :to schema-version-v :migrated false}))

;; ===========================================================================
;; Low-level key/value recall surface (sqlite-indexed)
;; ===========================================================================

(defn put!
  "Upsert one object. `payload` is a TEXT string (JSON or EDN). `score-type` is
   optional metadata (nil for non-trace kinds). Overwrites any existing row at
   `key` — the named-key (mutable threaded slot) discipline."
  ([store key kind payload] (put! store key kind payload nil (now-ms)))
  ([store key kind payload score-type] (put! store key kind payload score-type (now-ms)))
  ([{:keys [db]} key kind payload score-type created-at]
   (.run (.prepare db "INSERT INTO objects (key, kind, payload, score_type, created_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET
                         kind = excluded.kind,
                         payload = excluded.payload,
                         score_type = excluded.score_type,
                         created_at = excluded.created_at")
         key kind payload (when score-type (name score-type)) created-at)
   key))

(defn fetch
  "Fetch the raw row at `key` as `{:key :kind :payload :score-type :created-at}`,
   or nil if absent. (Named `fetch`, not `get`, to avoid shadowing core/get.)"
  [{:keys [db]} key]
  (when-let [row (.get (.query db "SELECT key, kind, payload, score_type, created_at
                                   FROM objects WHERE key = ?") key)]
    {:key        (.-key row)
     :kind       (.-kind row)
     :payload    (.-payload row)
     :score-type (some-> (.-score_type row) keyword)
     :created-at (.-created_at row)}))

(defn has?
  "True when an object exists at `key`."
  [{:keys [db]} key]
  (boolean (.get (.query db "SELECT 1 FROM objects WHERE key = ? LIMIT 1") key)))

(defn forget!
  "Delete the object at `key`. Returns true if a row was present and removed."
  [store key]
  (let [present (has? store key)]
    (.run (.prepare (:db store) "DELETE FROM objects WHERE key = ?") key)
    present))

(defn list-keys
  "All keys in the store, or all keys of one `kind`. Ascending by key."
  ([{:keys [db]}]
   (->> (.all (.query db "SELECT key FROM objects ORDER BY key ASC"))
        (mapv #(.-key %))))
  ([{:keys [db]} kind]
   (->> (.all (.query db "SELECT key FROM objects WHERE kind = ? ORDER BY key ASC")
              (name kind))
        (mapv #(.-key %)))))

(defn recent
  "The `n` most recently written keys of `kind`, newest first (by created_at)."
  [{:keys [db]} kind n]
  (->> (.all (.query db "SELECT key FROM objects WHERE kind = ?
                         ORDER BY created_at DESC, key DESC LIMIT ?")
             (name kind) n)
       (mapv #(.-key %))))

(defn count-objects
  "Total object count, or count of one `kind`."
  ([{:keys [db]}]
   (.-c (.get (.query db "SELECT count(*) c FROM objects"))))
  ([{:keys [db]} kind]
   (.-c (.get (.query db "SELECT count(*) c FROM objects WHERE kind = ?") (name kind)))))

;; ===========================================================================
;; Durable JSON payload codec — non-finite-SAFE (genmlx-94tu)
;; ===========================================================================
;;
;; Every JSON payload this namespace writes goes through `encode-payload`, and
;; every one it reads through `decode-payload`. They are exact inverses: a
;; non-finite number becomes its token (see the vocabulary above), and a genuine
;; string that could be MISTAKEN for a token is escaped by doubling its leading
;; '#', so the transform is total and lossless in both directions.
;;
;; `assert-no-null-number!` is the second half of the defence, and the one that
;; covers what this namespace cannot re-encode: payloads produced INSIDE
;; `genmlx.serialize` (`save-choices` / `save-trace` stringify internally) and
;; legacy rows written before this fix. A JSON `null` standing where a number
;; must be is not recoverable, so we refuse LOUDLY at both the write and the read
;; boundary rather than hand back a silent 0.0.

;; The tag/untag pair used to live here as a private copy. It now lives in
;; `genmlx.serialize` — the LOWER layer that owns the JSON codec — so there is
;; one implementation and this namespace cannot drift from the payload format
;; it reads (genmlx-94tu / genmlx-pif1). These aliases keep the call sites
;; below unchanged.
(def ^:private tag-non-finite   ser/tag-non-finite)
(def ^:private untag-non-finite ser/untag-non-finite)

(defn- mlx-leaf?
  "Is this a serialized MLX leaf (`{:type \"scalar\"|\"array\" :value ..}`)?"
  [m]
  (and (map? m) (contains? #{"scalar" "array"} (:type m))))

(def ^:private bare-number-keys
  "Payload keys whose value is a BARE number rather than an MLX leaf (a 0-d
   trace score in `genmlx-trace-v1`, a scalar weight). A null there is a nulled
   non-finite, not a legitimate absent value — unlike `:retval` / `:args`, which
   may genuinely be nil and are therefore NOT listed."
  #{:score :weight "score" "weight"})

(defn- null-in-value?
  "Is `v` nil, or does it contain a nil anywhere (an MLX leaf `:value` is a
   number or a nested vector of numbers — never nil)?"
  [v]
  (or (nil? v)
      (and (sequential? v) (boolean (some null-in-value? v)))))

(defn- find-null-number
  "Path to the first JSON null standing where a NUMBER must be, or nil."
  ([data] (find-null-number data []))
  ([data path]
   (cond
     (mlx-leaf? data)
     (when (null-in-value? (:value data)) (conj path :value))

     (map? data)
     (some (fn [[k v]]
             (if (and (contains? bare-number-keys k) (nil? v))
               (conj path k)
               (find-null-number v (conj path k))))
           data)

     (sequential? data)
     (some (fn [[i v]] (find-null-number v (conj path i)))
           (map-indexed vector data))

     :else nil)))

(defn- assert-no-null-number!
  "Throw if `data` (a PARSED payload) holds a JSON null where a number must be.
   JSON.stringify maps NaN/±Inf to null and mx/array's Float32Array.from coerces
   null to 0.0, so restoring such a payload would silently yield 0.0 — for an SMC
   log-weight, turning an impossible particle into the best one."
  [data key ctx]
  (when-let [p (find-null-number data)]
    (throw (ex-info (str "genmlx.memory: JSON null where a number must be, at path "
                         (pr-str p) " of the " ctx " payload at key " (pr-str key)
                         ". JSON.stringify nulls every NaN/±Inf and mx/array coerces"
                         " null to 0.0, so this value is unrecoverable and would"
                         " restore as a silent 0.0.")
                    {:genmlx/error :null-number-in-payload
                     :key key :context ctx :path p}))))

(defn- assert-json-no-null-number!
  "Same check against a JSON payload STRING. The `null` substring is a cheap gate
   so the common (clean) payload is never re-parsed."
  [json key ctx]
  (when (str/includes? json "null")
    (assert-no-null-number! (js->clj (js/JSON.parse json) :keywordize-keys true)
                            key ctx))
  json)

(defn- encode-payload
  "CLJS data -> non-finite-safe JSON payload string."
  [data]
  (js/JSON.stringify (clj->js (tag-non-finite data))))

(defn- decode-payload
  "Non-finite-safe JSON payload string -> CLJS data. Throws on an unrecoverable
   null-number (a legacy row written before the tokens existed)."
  [json key ctx]
  (let [data (untag-non-finite
               (js->clj (js/JSON.parse json) :keywordize-keys true))]
    (assert-no-null-number! data key ctx)
    data))

;; ===========================================================================
;; Typed save / restore (composes genmlx.serialize)
;; ===========================================================================

(defn- require-row [store key]
  (or (fetch store key)
      (throw (ex-info (str "genmlx.memory: no object at key " (pr-str key))
                      {:genmlx/error :key-absent :key key}))))

;; --- choices (named-key; re-generation reproduces the score) ---------------

(defn save-choices!
  "Persist a trace's choices (kind \"choices\") under `key`. Throws on a
   :collapsed trace (enumerate/exact): its choicemap is EMPTY, so re-generating
   from the stored choices cannot reproduce the collapsed score — persisting it
   would be a silent-empty-choices trap (consistent with `save-trace!`).
   Also throws if the JSON `genmlx.serialize` produced nulled a non-finite choice
   value (genmlx-94tu): this namespace does not own that stringify, so it refuses
   to store an unrecoverable artifact rather than let it restore as a silent 0.0.
   Returns `key`."
  [store key trace & {:keys [created-at]}]
  (when (= :collapsed (tr/score-type trace))
    (throw (ex-info "genmlx.memory: cannot persist choices of a :collapsed trace (empty choicemap)"
                    {:genmlx/error :collapsed-trace :key key})))
  (put! store key "choices"
        (assert-json-no-null-number! (ser/save-choices trace) key "choices")
        nil (or created-at (now-ms))))

(defn restore-choices
  "Load the ChoiceMap stored at `key`. Throws on a payload that nulled a
   non-finite value rather than restoring it as a silent 0.0."
  [store key]
  (let [json (:payload (require-row store key))]
    (assert-json-no-null-number! json key "choices")
    (ser/load-choices json)))

;; --- full trace (caller supplies the gen-fn; :collapsed refused) -----------

(defn save-trace!
  "Persist a full trace (kind \"trace\", with its score-type) under `key`.
   Throws on a :collapsed trace: enumerate/exact traces have an EMPTY choicemap
   and no recoverable choices, so persisting one would store a misleading
   artifact (restore re-generates from choices). Also throws if the JSON
   `genmlx.serialize` produced nulled a non-finite choice value or score
   (genmlx-94tu). Returns `key`."
  [store key trace & {:keys [created-at]}]
  (let [st (tr/score-type trace)]
    (when (= :collapsed st)
      (throw (ex-info "genmlx.memory: cannot persist a :collapsed trace (empty choicemap, nothing to re-generate)"
                      {:genmlx/error :collapsed-trace :key key})))
    (put! store key "trace"
          (assert-json-no-null-number! (ser/save-trace trace) key "trace")
          st (or created-at (now-ms)))))

(defn restore-trace
  "Reconstruct a full trace at `key` by re-generating from the saved choices and
   args with the caller-supplied `gen-fn` (gen-fns are never serialized). Refuses
   a :collapsed row, and a payload that nulled a non-finite value."
  [store key gen-fn]
  (let [{:keys [payload score-type]} (require-row store key)]
    (when (= :collapsed score-type)
      (throw (ex-info "genmlx.memory: stored trace is :collapsed and cannot be restored"
                      {:genmlx/error :collapsed-trace :key key})))
    (assert-json-no-null-number! payload key "trace")
    (ser/load-trace gen-fn payload)))

;; --- particle set (VectorizedTrace; no serialize path exists upstream) ------

(defn save-particles!
  "Persist a VectorizedTrace particle set (kind \"particles\") under `key`:
   the [N]-leaf choices plus the [N] score, weight, n-particles and args. The
   gen-fn is NOT serialized (supplied on restore). Non-finite numbers are written
   as tokens, NOT as JSON null (genmlx-94tu): an SMC `-Inf` log-weight must not
   restore as 0.0, which logsumexp would read as the BEST particle. Returns
   `key`."
  [store key vtrace & {:keys [created-at]}]
  (let [data {:format      "genmlx-particles-v1"
              :n-particles (:n-particles vtrace)
              :choices     (ser/choices->data (:choices vtrace))
              :score       (ser/value->data (:score vtrace))
              :weight      (ser/value->data (:weight vtrace))
              :args        (mapv ser/value->data (:args vtrace))}]
    (put! store key "particles"
          (encode-payload data)
          nil (or created-at (now-ms)))))

(defn restore-particles
  "Reconstruct a VectorizedTrace at `key` with the caller-supplied `gen-fn`.
   retval is not persisted (best-effort nil). Non-finite scores/weights/choices
   restore EXACTLY; a legacy payload that nulled one throws rather than silently
   returning 0.0."
  [store key gen-fn]
  (let [data    (decode-payload (:payload (require-row store key)) key "particles")
        choices (ser/data->choices (:choices data))
        score   (ser/data->value (:score data))
        weight  (ser/data->value (:weight data))
        args    (mapv ser/data->value (:args data))]
    (vect/->VectorizedTrace gen-fn args choices score weight (:n-particles data) nil)))

;; --- parameter store (named-key; overwrite semantics) ----------------------

(defn save-param-store!
  "Persist a learning parameter store `{:params {name -> MLX-array} :version n}`
   (kind \"param-store\") under `key`. Upserts — re-saving under the same key
   OVERWRITES, the mutable-threaded-slot discipline. Non-finite parameters are
   written as tokens, not JSON null (genmlx-94tu). Returns `key`."
  [store key param-store & {:keys [created-at]}]
  (put! store key "param-store"
        (encode-payload (ser/value->data param-store))
        nil (or created-at (now-ms))))

(defn restore-param-store
  "Load the parameter store stored at `key` (MLX arrays restored). Non-finite
   parameters restore EXACTLY; a legacy payload that nulled one throws rather
   than silently returning 0.0."
  [store key]
  (ser/data->value
    (decode-payload (:payload (require-row store key)) key "param-store")))

;; --- pure-EDN values (ConceptMemory, experience log, skill library) --------

(defn- deep-plain
  "Recursively convert records to plain maps so a value is EDN-printable AND
   reader-readable (records print with a tag `read-string` cannot parse).
   Preserves maps/vectors/sets/keywords/scalars."
  [v]
  (cond
    (record? v)     (deep-plain (into {} v))
    (map? v)        (update-vals v deep-plain)
    (set? v)        (into #{} (map deep-plain) v)
    (vector? v)     (mapv deep-plain v)
    (seq? v)        (mapv deep-plain v)
    :else           v))

(defn save-edn!
  "Persist an arbitrary pure-data value (no MLX arrays) as EDN under `key` with
   the given `kind` — for the experience log, a skill library, or any folded
   accumulator that is plain data. Records are flattened to maps first. Returns
   `key`."
  [store key kind value & {:keys [created-at]}]
  (put! store key (name kind) (pr-str (deep-plain value)) nil (or created-at (now-ms))))

(defn restore-edn
  "Load an EDN value stored by `save-edn!`. Records restore as PLAIN MAPS."
  [store key]
  (reader/read-string (:payload (require-row store key))))

(defn save-concept-memory!
  "Persist a sensorimotor `ConceptMemory` (kind \"concept-memory\") under `key`.
   It is pure data (no MLX arrays), so it round-trips as EDN; its `Implication`
   records and the index sets are flattened to plain maps/sets. Returns `key`."
  [store key concept-memory & {:keys [created-at]}]
  (apply save-edn! store key "concept-memory" concept-memory
         (when created-at [:created-at created-at])))

(defn restore-concept-memory
  "Load a ConceptMemory's field data at `key`. Returns the three index maps as
   PLAIN data (`{:by-key .. :by-precondition .. :by-consequent ..}`); the caller
   re-wraps with `sensorimotor/map->ConceptMemory` if it needs the record type."
  [store key]
  (restore-edn store key))

;; ===========================================================================
;; Content addressing (immutable artifacts; no registry)
;; ===========================================================================

(defn put-content!
  "Store an immutable JSON `payload` under its CONTENT address and return that
   key (`\"sha256:<hex>\"`). Storing the same logical payload twice — even built
   with different map key order — yields the same key and is idempotent. Use for
   Traces, particle snapshots, experience entries that should be addressed by
   value, not by name."
  [store kind payload & {:keys [score-type created-at]}]
  (let [key (content-hash-json payload)]
    ;; genmlx-94tu: a nulled non-finite is doubly bad here — it corrupts the
    ;; VALUE and collapses the content ADDRESS onto the nil-valued variant.
    (assert-json-no-null-number! payload key (name kind))
    (put! store key (name kind) payload score-type (or created-at (now-ms)))
    key))

(defn save-trace-content!
  "Persist a trace by CONTENT address (kind \"trace\"); returns the content key.
   Like `save-trace!` but the key is the canonical content hash, so an identical
   trace persisted again resolves to the same key. Refuses :collapsed traces."
  [store trace & {:keys [created-at]}]
  (let [st (tr/score-type trace)]
    (when (= :collapsed st)
      (throw (ex-info "genmlx.memory: cannot content-address a :collapsed trace"
                      {:genmlx/error :collapsed-trace})))
    (put-content! store "trace" (ser/save-trace trace)
                  :score-type st :created-at (or created-at (now-ms)))))

;; ===========================================================================
;; Session lifecycle — synchronous checkpoint of the carried accumulator
;; ===========================================================================
;;
;; A session is a durable store stamped with a session id; a snapshot is an
;; ATOMIC (single db.transaction) batch of writes — the all-or-nothing
;; checkpoint at an episode/session boundary. Because sqlite persists on disk,
;; the checkpoint survives process death: a later run reopens the same path and
;; restores. This is the across-SESSION effect (the one thing not free).

(defn open-session
  "Open (or create) the durable store at `path` for a named session, stamping
   `:session-id` and an open timestamp into `meta`. Returns the store."
  [path session-id]
  (let [store (open-store path)]
    (meta-put! (:db store) "session_id" session-id)
    (meta-put! (:db store) "session_opened_at" (now-ms))
    store))

(defn snapshot!
  "Atomically persist many objects in a SINGLE synchronous `db.transaction` —
   the session-boundary checkpoint. `writes` is a seq of maps
   `{:key :kind :payload :score-type? :created-at?}`. All-or-nothing: a throw
   inside rolls the whole batch back. Returns the number of objects written."
  [{:keys [db] :as store} writes]
  (let [writes (vec writes)
        tx (.transaction db
             (fn [ws]
               (doseq [{:keys [key kind payload score-type created-at]} ws]
                 (put! store key kind payload score-type (or created-at (now-ms))))))]
    (tx writes)
    (count writes)))

(defn restore-session
  "A manifest of what is recoverable from a (re)opened session store:
   `{:session-id .. :schema-version .. :opened-at .. :keys [..] :count n}`.
   The actual values are fetched with the typed `restore-*` functions. This is
   the across-process-death entry point: open the same path, read the manifest,
   restore the carried accumulator."
  [{:keys [db] :as store}]
  {:session-id     (meta-get db "session_id")
   :schema-version (schema-version store)
   :opened-at      (some-> (meta-get db "session_opened_at") js/parseInt)
   :keys           (list-keys store)
   :count          (count-objects store)})
