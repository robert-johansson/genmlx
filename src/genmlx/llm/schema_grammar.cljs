(ns genmlx.llm.schema-grammar
  "Malli schema -> EDN grammar for structured LLM generation (PURE / model-free).

   This is the schema-typed structured-output layer: a malli schema is a
   declarative contract for what an LLM should emit. We compile the schema to a
   regular expression over its *canonical EDN serialization*, which the existing
   regex->DFA->byte-mask machinery (grammar.cljs + bytes.cljs) turns into a
   constraint that forces the model to emit only conforming data.

   The whole namespace is pure and contains NO model/GPU calls — it is
   exhaustively unit-testable without an LLM. `genmlx.llm.structured` wires the
   output of `schema->regex` into byte-level constrained generation.

   Canonical EDN form (what the regex matches and `schema->canonical-str`
   emits): single-space separated, NO commas, double-quoted strings, keywords
   bare. e.g. {:name \"ann\" :age 30}, [1 2 3], :red, true, nil.

   Supported schema subset (v1):
     :int (+ :min>=0 drops the sign), :double/:float, :string (+ :max, [:re ..]),
     :boolean, :keyword, :nil/nil, :any-of [:= v], [:enum ...], [:maybe X],
     [:map [k S]...] (closed, fixed key order, required keys), [:vector X] /
     [:sequential X] (variable length, optional :min/:max), [:tuple ...],
     [:and X ...] (driven by the first child; the rest are validated post-hoc),
     [:or ...] (alternation).
   Unsupported forms throw a clear ex-info (no silent fallthrough): :optional map
   entries, recursive/registry refs, :multi, :map-of, open maps. These are
   deferred to a later schema-directed decoder (v1.1)."
  (:require [malli.core :as m]
            [clojure.string :as str]
            [edamame.core :as eda]))

;; ============================================================
;; Regex literal escaping
;; ============================================================

;; Chars the grammar.cljs regex reader treats as structural (must be escaped to
;; appear literally): \ [ ] ( ) . * + ? | { }  — all are escapable via `\\c`.
;; Note: : " space , - ^ are NOT structural outside a class and stay bare.
(def ^:private re-special #{"\\" "[" "]" "(" ")" "." "*" "+" "?" "|" "{" "}"})

(defn re-quote
  "Escape a literal string so it matches itself under the grammar.cljs regex
   reader. Only the structural metacharacters are backslash-escaped."
  [s]
  (->> s
       (map (fn [c] (let [c (str c)] (if (re-special c) (str "\\" c) c))))
       (apply str)))

;; ============================================================
;; Canonical EDN serialization (schema-agnostic, for literal values)
;; ============================================================

(declare canonical-str)

(defn- canonical-coll-str [open xs close]
  (str open (str/join " " (map canonical-str xs)) close))

(defn canonical-str
  "Serialize a value to its canonical EDN string (the form the regex matches).
   Schema-agnostic — used for [:= v] and [:enum ...] literal members, and as the
   leaf serializer. Maps are emitted in their natural seq order; for
   schema-ordered map serialization use `schema->canonical-str`."
  [v]
  (cond
    (map? v)     (str "{" (str/join " " (mapcat (fn [[k val]]
                                                  [(canonical-str k) (canonical-str val)])
                                                v)) "}")
    (vector? v)  (canonical-coll-str "[" v "]")
    (set? v)     (canonical-coll-str "#{" v "}")
    (list? v)    (canonical-coll-str "(" v ")")
    (keyword? v) (str v)
    (string? v)  (pr-str v)
    (boolean? v) (str v)
    (nil? v)     "nil"
    (number? v)  (str v)
    :else        (pr-str v)))

;; ============================================================
;; Leaf regex builders
;; ============================================================

;; Canonical EDN integers: no -0, no leading zeros. This guarantees the
;; constrained generation can only emit the SAME string the value serializes to
;; (so score(value) equals the generated trace density exactly).
(def ^:private int-regex-signed   "(0|-?[1-9][0-9]*)")
(def ^:private int-regex-unsigned "(0|[1-9][0-9]*)")
(def ^:private double-regex       "-?(0|[1-9][0-9]*)(\\.[0-9]+)?")
;; EDN keyword value: simple keyword name (v1 — no namespaces/special chars).
;; Hyphen is backslash-escaped so the class reader treats it as a literal, not
;; a range. Enums (the common keyword case) use exact alternation instead.
(def ^:private keyword-regex      ":[a-zA-Z][a-zA-Z0-9_\\-]*")
;; EDN string with no embedded quote/backslash (v1 limitation, documented).
(def ^:private string-regex       "\"[^\"]*\"")

(defn- int-regex [props]
  (if (and (number? (:min props)) (>= (:min props) 0))
    int-regex-unsigned
    int-regex-signed))

(defn- string-leaf-regex [props children]
  (cond
    ;; [:re pattern] — content matches the pattern, wrapped in EDN quotes.
    (seq children)
    (let [pat (first children)
          src (cond (string? pat) pat
                    (instance? js/RegExp pat) (.-source pat)
                    :else (str pat))]
      (str "\"" src "\""))
    (number? (:max props)) (str "\"[^\"]{0," (:max props) "}\"")
    :else string-regex))

(defn- repeat-quant
  "Quantifier `{lo,hi}` / `*` for a vector tail group, from optional min/max."
  [min* max*]
  (let [lo (max 0 (dec (or min* 1)))]
    (if (and max* (>= max* 1))
      (str "{" lo "," (dec max*) "}")
      (if (zero? lo) "*" (str "{" lo ",}")))))

;; ============================================================
;; schema -> regex
;; ============================================================

(declare schema->regex)

(defn- map-entry-regex [[k props vschema]]
  (when (:optional props)
    (throw (ex-info "structured-gen: :optional map entries unsupported (v1)"
                    {:key k})))
  (str (re-quote (canonical-str k)) " " (schema->regex vschema)))

(defn- map-regex [children]
  (str "\\{" (str/join " " (map map-entry-regex children)) "\\}"))

(defn- vector-regex [child props]
  (let [e (schema->regex child)
        q (repeat-quant (:min props) (:max props))]
    ;; head element then `( <e>){lo,hi}`, whole thing optional when min 0
    (if (and (:min props) (pos? (:min props)))
      (str "\\[" e "( " e ")" q "\\]")
      (str "\\[(" e "( " e ")" q ")?\\]"))))

(defn- tuple-regex [children]
  (str "\\[" (str/join " " (map schema->regex children)) "\\]"))

(defn- alt-regex [members->str members]
  (str "(" (str/join "|" (map members->str members)) ")"))

(defn schema->regex
  "Compile a malli schema to a regex (string) matching the canonical EDN
   serialization of any conforming value. Throws ex-info on unsupported forms."
  [schema]
  (let [s     (m/schema schema)
        t     (m/type s)
        props (m/properties s)
        ch    (m/children s)]
    (case t
      :int            (int-regex props)
      (:double :float) double-regex
      :string         (string-leaf-regex props ch)
      :boolean        "(true|false)"
      :keyword        keyword-regex
      (:nil)          "nil"
      :=              (re-quote (canonical-str (first ch)))
      :enum           (alt-regex #(re-quote (canonical-str %)) ch)
      :maybe          (str "(nil|" (schema->regex (first ch)) ")")
      :map            (map-regex ch)
      (:vector :sequential) (vector-regex (first ch) props)
      :tuple          (tuple-regex ch)
      :and            (schema->regex (first ch))
      :or             (alt-regex schema->regex ch)
      ;; predicates malli exposes as fn-schemas reach here by type
      (throw (ex-info "structured-gen: unsupported schema form"
                      {:type t :form (m/form s)})))))

;; ============================================================
;; schema -> canonical string (schema-ordered, for scoring a value)
;; ============================================================

(defn schema->canonical-str
  "Serialize `value` to the canonical EDN string the schema's regex matches,
   ordering map entries by SCHEMA order (not the value's seq order). This is the
   exact string a constrained generation of `value` would have produced, so it
   is the correct target for teacher-forced scoring."
  [schema value]
  (let [s  (m/schema schema)
        t  (m/type s)
        ch (m/children s)]
    (case t
      :map   (str "{" (str/join " "
                                (mapcat (fn [[k _ vschema]]
                                          [(canonical-str k)
                                           (schema->canonical-str vschema (get value k))])
                                        ch)) "}")
      (:vector :sequential)
      (str "[" (str/join " " (map #(schema->canonical-str (first ch) %) value)) "]")
      :tuple (str "[" (str/join " " (map schema->canonical-str ch value)) "]")
      :maybe (if (nil? value) "nil" (schema->canonical-str (first ch) value))
      :and   (schema->canonical-str (first ch) value)
      :or    (canonical-str value)
      ;; leaves
      (canonical-str value))))

;; ============================================================
;; Conditioning: specialize a schema by fixing some fields
;; ============================================================

(defn constrain-schema
  "Return a new schema with the fields named in `partial` (a map address->value,
   or a nested map mirroring the schema) replaced by [:= value], so that
   `schema->regex` of the result forces those fields. Only :map schemas support
   keyed conditioning; for other schemas a non-nil `partial` fixes the whole
   value to [:= partial]."
  [schema partial]
  (let [s  (m/schema schema)
        t  (m/type s)
        ch (m/children s)]
    (cond
      (and (= t :map) (map? partial))
      (into [:map]
            (map (fn [[k props vschema]]
                   (if (contains? partial k)
                     (let [pv (get partial k)]
                       [k props (if (map? pv) (constrain-schema vschema pv) [:= pv])])
                     [k props vschema])))
            ch)
      (some? partial) [:= partial]
      :else schema)))

;; ============================================================
;; Parse + validate generated EDN
;; ============================================================

(def ^:private eda-opts {:all true})

(defn parse-edn
  "Parse a canonical-EDN string into a ClojureScript value via edamame.
   Returns ::parse-error on failure."
  [s]
  (try (eda/parse-string s eda-opts)
       (catch :default _ ::parse-error)))

(defn validate
  "True iff `value` conforms to `schema` (malli)."
  [schema value]
  (m/validate (m/schema schema) value))

(defn parse-and-validate
  "Parse `s` and validate against `schema`. Returns {:ok? :value :error}.
   :error is :parse-error or :schema-mismatch."
  [schema s]
  (let [v (parse-edn s)]
    (cond
      (= v ::parse-error) {:ok? false :error :parse-error}
      (validate schema v)  {:ok? true :value v}
      :else                {:ok? false :value v :error :schema-mismatch})))

(defn supported?
  "True iff `schema` compiles under the v1 subset (no throw)."
  [schema]
  (try (some? (schema->regex schema)) (catch :default _ false)))
