;; @tier fast
;; genmlx-3g0t reader leg (model-free): the :cljs per-argument constraint —
;; cljs-arg-status semantics, cljs-value-chars byte masking, and the full
;; hybrid-masker driven over a fake token index: envelope (DFA) -> value
;; (reader, byte-granular, '<' representable) -> forced closing -> envelope,
;; with opener-straddle tokens masked and the final text parsing clean.
;;
;; Run: bunx --bun nbb@1.4.208 test/genmlx/llm_toolcall_cljs_test.cljs

(ns genmlx.llm-toolcall-cljs-test
  (:require [clojure.string :as str]
            [genmlx.codegen.eval :as ceval]
            [genmlx.llm.toolcall :as tc]
            [genmlx.mlx :as mx]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

;; -- 1. cljs-arg-status: exactly-one-form semantics ------------------------
(println "\n-- 1. cljs-arg-status --")
(assert-true "complete single form" (= :complete (ceval/cljs-arg-status "(fn [scene] (:x scene))")))
(assert-true "trailing ws still complete" (= :complete (ceval/cljs-arg-status "(f x) ")))
(assert-true "valid prefix incomplete" (= :incomplete (ceval/cljs-arg-status "(fn [scene")))
(assert-true "second form is INVALID (not first-form-blind)" (= :invalid (ceval/cljs-arg-status "(f) (g)")))
(assert-true "unmatched delimiter invalid" (= :invalid (ceval/cljs-arg-status "(f))")))
(assert-true "blank incomplete" (= :incomplete (ceval/cljs-arg-status "")))

;; -- 2. cljs-value-chars: byte-level mask ----------------------------------
(println "\n-- 2. cljs-value-chars --")
(assert-true "empty value admits only delimiter openers"
             (= #{"(" "[" "{"} (set (tc/cljs-value-chars ""))))
(assert-true "'<' representable mid-form (comparisons!)"
             (contains? (set (tc/cljs-value-chars "(fn [a b] (")) "<"))
(assert-true "')' allowed to complete"
             (contains? (set (tc/cljs-value-chars "(inc x")) ")"))
(assert-true "no closing-tag completion inside a string literal"
             (not (contains? (set (tc/cljs-value-chars "(str \"\n</parameter")) ">")))

;; -- 3. hybrid-masker over a fake token index ------------------------------
(println "\n-- 3. hybrid-masker (fake token index) --")

(def single-chars
  (into (mapv #(js/String.fromCharCode %) (range 32 127)) ["\n" "\t"]))

(def multi-tokens
  ["<tool_call>" "<function=" "set_point" ">\n" "<parameter=" "code"
   "</parameter>" "</function>" "</tool_call>" "\n</parameter>\n"
   "</parameter>\n" ">\n(fn"])   ; last one = the opener-straddle token

(def token-index
  ;; pad to put eos (empty text) at a known id
  (into (into single-chars multi-tokens) [""]))

(def eos-id (dec (count token-index)))
(def n-tok (count token-index))

(defn tid [t]
  (let [i (.indexOf (clj->js token-index) t)]
    (when (neg? i) (throw (ex-info "no such fake token" {:t t})))
    i))

(def mock-tokenizer #js {:getEosTokenId (fn [] eos-id)})

(def tools
  [{:name "set_point"
    :params [{:name "code" :cljs true}]}])

(def constraint (tc/compile-toolcall mock-tokenizer tools
                                     {:token-index token-index}))

(assert-true "compile-toolcall attaches :cljs-support"
             (some? (:cljs-support constraint)))
(assert-true "closing-ids at k=0 include the full closing token"
             (pos? (.-length (nth (:closing-ids (:cljs-support constraint)) 0))))

(def zeros (mx/zeros [n-tok]))

(defn allowed-ids
  "Run the masker at the current vis; return the set of token ids with
   finite masked logit."
  [masker vis]
  (let [out (mx/->clj (masker zeros vis))]
    (set (keep-indexed (fn [i v] (when (js/isFinite v) i)) out))))

(defn emit
  "Append token (by text) to vis after asserting the masker allows it."
  [masker vis t label]
  (let [ids (allowed-ids masker vis)
        i (tid t)]
    (assert-true (str label ": " (pr-str t) " allowed") (contains? ids i))
    (conj vis i)))

(let [masker (tc/hybrid-masker constraint)
      ;; envelope: open the block + function + the cljs param
      vis (reduce (fn [v t] (emit masker v t "envelope"))
                  []
                  ["<tool_call>" "\n" "<function=" "set_point" ">\n" "<parameter=" "code"])
      ;; STRADDLE check: just before the opener completes, the token that
      ;; completes it AND carries value bytes must be masked; the clean
      ;; opener-completing token allowed.
      ids-here (allowed-ids masker vis)
      _ (assert-true "straddle token ('>\\n(fn') masked at the opener boundary"
                     (not (contains? ids-here (tid ">\n(fn"))))
      vis (emit masker vis ">\n" "opener completes")
      ;; VALUE region: only ( [ { to start — envelope/multi tokens all masked
      ids-v0 (allowed-ids masker vis)
      _ (assert-true "value start admits exactly the delimiter openers"
                     (= ids-v0 (set (map tid ["(" "[" "{"]))))
      _ (assert-true "eos masked inside the value region"
                     (not (contains? ids-v0 eos-id)))
      ;; byte-granular emission of (fn [a b] (< a b)) — '<' representable
      vis (reduce (fn [v c] (emit masker v c "value byte"))
                  vis
                  (mapv str "(fn [a b] (< a b)"))
      ;; before the final ')': still :incomplete, ')' among allowed
      vis (emit masker vis ")" "value completes")
      ;; CLOSING: form complete -> only prefixes of "\n</parameter>\n"
      ids-c (allowed-ids masker vis)
      _ (assert-true "closing forced: full closing token allowed"
                     (contains? ids-c (tid "\n</parameter>\n")))
      _ (assert-true "closing forced: '\\n' (a prefix) allowed"
                     (contains? ids-c (tid "\n")))
      _ (assert-true "closing forced: value bytes now masked"
                     (not (contains? ids-c (tid "x"))))
      _ (assert-true "closing forced: eos masked"
                     (not (contains? ids-c eos-id)))
      vis (emit masker vis "\n</parameter>\n" "closing emitted")
      ;; back in ENVELOPE: the DFA (parked through the value) resumes exactly
      vis (emit masker vis "</function>" "envelope resumes")
      vis (emit masker vis "\n" "envelope newline")
      vis (emit masker vis "</tool_call>" "block closes")
      text (apply str (map #(nth token-index %) vis))
      parsed (tc/parse-tool-calls text tools)]
  (assert-true "final text: zero parse errors" (empty? (:errors parsed)))
  (assert-true "final text: one call, declared name"
               (and (= 1 (count (:calls parsed)))
                    (= "set_point" (:name (first (:calls parsed))))))
  (assert-true "argument value round-trips with '<' intact"
               (= "(fn [a b] (< a b))"
                  (get-in (first (:calls parsed)) [:args "code"])))
  (assert-true "argument value is exactly one complete CLJS form"
               (= :complete (ceval/cljs-arg-status
                             (get-in (first (:calls parsed)) [:args "code"])))))

;; -- 4. name-clash guard ----------------------------------------------------
(println "\n-- 4. clash guard --")
(assert-true "cljs+plain same name across tools throws typed error"
             (try (tc/compile-toolcall mock-tokenizer
                                       [{:name "a" :params [{:name "p" :cljs true}]}
                                        {:name "b" :params [{:name "p"}]}]
                                       {:token-index token-index})
                  false
                  (catch :default e
                    (= :cljs-param-name-clash (:genmlx/error (ex-data e))))))

(println (str "\n== llm-toolcall-cljs: " @pass " pass, " @fail " fail =="))
(when (pos? @fail) (js/process.exit 1))
