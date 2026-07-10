(ns genmlx.llm.grammar
  "Grammar-constrained LLM generation.

   Compiles regex patterns to DFAs for efficient token-level masking.
   At each token step, invalid tokens are masked to -inf in the logits,
   ensuring the LLM only generates text matching the constraint.

   Uses Instaparse to parse the regex syntax itself — dogfooding the
   grammar library for its own implementation.

   Architecture:
     regex string → (Instaparse) → AST → (Thompson) → NFA → (subset) → DFA
     DFA + token-index → per-state valid-token masks
     mask + logits → constrained categorical → GFI trace site

   LIMITATION — printable-ASCII alphabet (genmlx-6dax): the wildcard alphabet
   is chars 32–126 (see printable-chars below), so `.`, negated classes like
   [^\"], and \\D \\W \\S never match non-ASCII characters or control chars
   (including newline; \\s/\\n match explicitly listed whitespace only).
   Non-ASCII content is silently off-grammar: it cannot be generated, and
   scoring it yields -Inf / :off-grammar. Explicit literals outside ASCII are
   not supported by the alphabet either."
  (:require [instaparse.core :as insta]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.handler :as h]
            [genmlx.dispatch :as dispatch]
            [clojure.set :as set]))

;; ============================================================
;; Character sets
;; ============================================================

(defn- char-range
  "Range of single-char strings from `from` to `to` inclusive."
  [from to]
  (let [f (.charCodeAt (str from) 0)
        t (.charCodeAt (str to) 0)]
    (set (map #(js/String.fromCharCode %) (range f (inc t))))))

(def ^:private digit-chars (char-range "0" "9"))
(def ^:private lower-chars (char-range "a" "z"))
(def ^:private upper-chars (char-range "A" "Z"))
(def ^:private word-chars (into (into lower-chars upper-chars)
                                (conj digit-chars "_")))
(def ^:private space-chars #{" " "\t" "\n" "\r"})
;; The wildcard alphabet for `.`, negated classes ([^…]) and \D \W \S.
;; Printable ASCII ONLY (32–126): those constructs never match bytes >= 127
;; (any non-ASCII/UTF-8 text) or control chars incl. newline, so non-ASCII
;; content can be neither generated nor scored on-grammar (genmlx-6dax).
(def ^:private printable-chars (set (map #(js/String.fromCharCode %) (range 32 127))))

;; ============================================================
;; Regex parsing (Instaparse → AST)
;; ============================================================

;; Open-ended repeats `{n,}` have no upper bound in regex, but the NFA/DFA
;; construction needs a finite one. We truncate them to this many repetitions.
(def ^:private max-unbounded-repeat 999)

(def ^:private regex-grammar
  (insta/parser
    "regex = alt
     alt = cat (<'|'> cat)*
     cat = quant+
     quant = atom quantifier?
     <quantifier> = star | plus | opt | repeat-exact | repeat-range
     star = <'*'>
     plus = <'+'>
     opt = <'?'>
     repeat-exact = <'{'> number <'}'>
     repeat-range = <'{'> number <','> number? <'}'>
     number = #'[0-9]+'
     <atom> = lit | escape | dot | class | group
     group = <'('> regex <')'>
     dot = <'.'>
     lit = #'[^\\\\\\[\\]().*+?|{}^$]'
     escape = <'\\\\'> #'[dwsDWSntr^$\\\\.*+?|()\\[\\]{}]'
     class = <'['> neg? class-body <']'>
     neg = <'^'>
     class-body = class-item+
     <class-item> = range | class-char
     range = class-char <'-'> class-char
     class-char = #'[^\\]\\\\^-]' | class-escape
     class-escape = <'\\\\'> #'.'"))

(defn- strip-anchors
  "Remove a single leading ^ and trailing (unescaped) $ from a regex string.
   compile-regex is whole-string anchored, so the anchors are redundant —
   but users write them habitually, and they used to parse as LITERAL
   characters: \"^[a-z]+$\" compiled to a DFA requiring a literal '^', which
   dead-ended generation after 0 bytes with no error (genmlx-bgyu).
   Anchors anywhere ELSE are now a loud parse error (excluded from lit);
   match a literal ^ or $ by escaping it."
  [s]
  (let [s (if (.startsWith s "^") (subs s 1) s)]
    (if (and (.endsWith s "$")
             ;; unescaped: an even number of backslashes precedes the $
             (even? (count (re-find #"\\*$" (subs s 0 (dec (count s)))))))
      (subs s 0 (dec (count s)))
      s)))

(defn parse-regex
  "Parse a regex string into an AST.

   AST nodes:
     [:lit \"c\"]        — literal single-char string
     [:class #{chars}]  — character class (set of single-char strings)
     [:cat a b]         — concatenation
     [:alt a b]         — alternation
     [:star a]          — zero or more
     [:empty]           — empty string"
  [regex-str]
  (let [tree (regex-grammar (strip-anchors regex-str))]
    (when (insta/failure? tree)
      (throw (js/Error. (str "Invalid regex: " regex-str "\n" (pr-str tree)))))
    (insta/transform
      {:number     (fn [s] (js/parseInt s))
       :lit        (fn [c] [:lit c])
       :dot        (fn [] [:class printable-chars])
       :escape     (fn [c]
                     (case c
                       "d" [:class digit-chars]
                       "D" [:class (set/difference printable-chars digit-chars)]
                       "w" [:class word-chars]
                       "W" [:class (set/difference printable-chars word-chars)]
                       "s" [:class space-chars]
                       "S" [:class (set/difference printable-chars space-chars)]
                       "n" [:lit "\n"]
                       "t" [:lit "\t"]
                       "r" [:lit "\r"]
                       [:lit c]))
       ;; Escapes INSIDE character classes expand to the same char-sets as
       ;; top-level escapes — [\d] used to keep the literal letter d
       ;; (class-char returned the escaped char verbatim), silently masking
       ;; generation to the wrong language (genmlx-bgyu). Returns a SET for
       ;; the class escapes (class-body merges sets) or a single-char string
       ;; for literal escapes (\\ \] \- \. ...).
       :class-escape (fn [c]
                       (case c
                         "d" digit-chars
                         "D" (set/difference printable-chars digit-chars)
                         "w" word-chars
                         "W" (set/difference printable-chars word-chars)
                         "s" space-chars
                         "S" (set/difference printable-chars space-chars)
                         "n" "\n"
                         "t" "\t"
                         "r" "\r"
                         c))
       :class-char identity
       :range      (fn [from to]
                     ;; a class escape is a SET — meaningless as a range
                     ;; endpoint ([\d-x]); reject loudly rather than
                     ;; charCodeAt-ing garbage.
                     (when (or (set? from) (set? to))
                       (throw (js/Error.
                               "Invalid regex: character-class escape cannot be a range endpoint")))
                     (char-range from to))
       :class-body (fn [& items]
                     (reduce (fn [acc item]
                               (if (set? item) (into acc item) (conj acc item)))
                             #{} items))
       :neg        (fn [] :neg)
       :class      (fn [& args]
                     (if (= (first args) :neg)
                       [:class (set/difference printable-chars (second args))]
                       [:class (first args)]))
       :star       (fn [] :star)
       :plus       (fn [] :plus)
       :opt        (fn [] :opt)
       :repeat-exact (fn [n] [:repeat n n])
       :repeat-range (fn [n & [m]] [:repeat n (or m max-unbounded-repeat)])
       :quant      (fn [node & [q]]
                     (cond
                       (nil? q)     node
                       (= q :star)  [:star node]
                       (= q :plus)  [:cat node [:star node]]
                       (= q :opt)   [:alt node [:empty]]
                       (vector? q)  (let [[_ lo hi] q
                                          required (repeat lo node)
                                          optional (repeat (- hi lo) [:alt node [:empty]])
                                          parts (concat required optional)]
                                      (if (seq parts)
                                        (reduce (fn [a b] [:cat a b]) parts)
                                        ;; a{0} / a{0,0}: zero copies — matches the
                                        ;; empty string for this atom. Without this
                                        ;; (reduce ... ()) returned nil and crashed
                                        ;; NFA construction (genmlx-abw8).
                                        [:empty]))))
       :cat        (fn [& items]
                     (if (= 1 (count items))
                       (first items)
                       (reduce (fn [a b] [:cat a b]) items)))
       :alt        (fn [& items]
                     (if (= 1 (count items))
                       (first items)
                       (reduce (fn [a b] [:alt a b]) items)))
       :group      identity
       :regex      identity}
      tree)))

;; ============================================================
;; Thompson's construction (AST → NFA)
;; ============================================================

(defn- fresh-state [nfa]
  (let [id (:next-id nfa 0)]
    [id (assoc nfa :next-id (inc id))]))

(defn- fresh-states [nfa n]
  (let [start (:next-id nfa 0)]
    [(vec (range start (+ start n)))
     (assoc nfa :next-id (+ start n))]))

(defn- add-transition [nfa from input to]
  (update-in nfa [:transitions from] (fnil conj []) [input to]))

(defn- add-epsilon [nfa from to]
  (add-transition nfa from :epsilon to))

(defn ast->nfa
  "Convert regex AST to NFA via Thompson's construction."
  ([ast]
   (let [[s nfa] (fresh-states {:next-id 0 :transitions {}} 2)
         start (s 0) accept (s 1)
         nfa (ast->nfa nfa ast start accept)]
     (assoc nfa :start start :accept accept)))
  ([nfa ast start accept]
   (let [[tag a b] ast]
     (case tag
       :lit   (add-transition nfa start a accept)
       :class (add-transition nfa start a accept)
       :empty (add-epsilon nfa start accept)
       :cat   (let [[mid nfa] (fresh-state nfa)
                    nfa (ast->nfa nfa a start mid)]
                (ast->nfa nfa b mid accept))
       :alt   (let [[[s1 f1 s2 f2] nfa] (fresh-states nfa 4)
                    nfa (-> nfa
                            (add-epsilon start s1)
                            (add-epsilon start s2)
                            (ast->nfa a s1 f1)
                            (ast->nfa b s2 f2)
                            (add-epsilon f1 accept)
                            (add-epsilon f2 accept))]
                nfa)
       :star  (let [[[s1 f1] nfa] (fresh-states nfa 2)
                    nfa (-> nfa
                            (add-epsilon start s1)
                            (ast->nfa a s1 f1)
                            (add-epsilon f1 s1)
                            (add-epsilon start accept)
                            (add-epsilon f1 accept))]
                nfa)))))

;; ============================================================
;; Subset construction (NFA → DFA)
;; ============================================================

(defn- epsilon-closure
  "From a set of NFA states, follow all epsilon transitions."
  [nfa states]
  (loop [result (set states)
         worklist (vec states)]
    (if (empty? worklist)
      result
      (let [s (peek worklist)
            worklist (pop worklist)
            eps-targets (->> (get-in nfa [:transitions s])
                             (filter #(= :epsilon (first %)))
                             (map second)
                             (remove result))]
        (recur (into result eps-targets)
               (into worklist eps-targets))))))

(defn- nfa-inputs
  "Get all possible non-epsilon input characters from a set of NFA states."
  [nfa states]
  (let [all-transitions (mapcat #(get-in nfa [:transitions %]) states)]
    (->> all-transitions
         (map first)
         (remove #{:epsilon})
         (mapcat (fn [input] (if (set? input) input [input])))
         set)))

(defn- nfa-move
  "From a set of NFA states, follow transitions on a specific char."
  [nfa states ch]
  (set (for [s states
             [input target] (get-in nfa [:transitions s])
             :when (or (= input ch)
                       (and (set? input) (contains? input ch)))]
         target)))

(defn- nfa->dfa
  "Subset construction: NFA → DFA with integer states."
  [nfa]
  (let [start-set (epsilon-closure nfa #{(:start nfa)})
        accept-state (:accept nfa)]
    (loop [dfa {:start start-set
                :accept #{}
                :transitions {}
                :state-map {start-set 0}
                :next-id 1}
           worklist [start-set]]
      (if (empty? worklist)
        (let [{:keys [state-map]} dfa]
          {:start (state-map start-set)
           :accept (set (for [[ss id] state-map
                              :when (contains? ss accept-state)]
                          id))
           :transitions (into {}
                          (for [[[from-set ch] to-set] (:transitions dfa)]
                            [[(state-map from-set) ch] (state-map to-set)]))})
        (let [current (peek worklist)
              worklist (pop worklist)
              chars (nfa-inputs nfa current)
              [dfa worklist]
              (reduce
                (fn [[dfa wl] ch]
                  (let [moved (nfa-move nfa current ch)
                        target (epsilon-closure nfa moved)]
                    (if (empty? target)
                      [dfa wl]
                      (let [known? (contains? (:state-map dfa) target)
                            id (or (get-in dfa [:state-map target])
                                   (:next-id dfa))
                            dfa (-> (cond-> dfa
                                       (not known?)
                                       (-> (assoc-in [:state-map target] id)
                                           (update :next-id inc)))
                                     (assoc-in [:transitions [current ch]] target))]
                        [dfa (if known? wl (conj wl target))]))))
                [dfa worklist]
                chars)]
          (recur dfa worklist))))))

;; ============================================================
;; DFA operations
;; ============================================================

(defn dfa-advance
  "Advance DFA by one character. Returns new state or :dead."
  [dfa state ch]
  (get (:transitions dfa) [state ch] :dead))

(defn dfa-advance-string
  "Advance DFA through a string. Returns final state or :dead."
  [dfa state s]
  (reduce (fn [st i]
            (if (= st :dead)
              (reduced :dead)
              (dfa-advance dfa st (subs s i (inc i)))))
          state (range (count s))))

(defn dfa-accepts?
  "Does the DFA accept this string (starting from :start)?"
  [dfa s]
  (contains? (:accept dfa) (dfa-advance-string dfa (:start dfa) s)))

(defn- compute-alive-states
  "Compute set of states from which an accept state is reachable."
  [dfa]
  (let [rev (reduce (fn [acc [[from _ch] to]]
                      (update acc to (fnil conj #{}) from))
                    {} (:transitions dfa))]
    (loop [alive (:accept dfa)
           worklist (vec (:accept dfa))]
      (if (empty? worklist)
        alive
        (let [s (peek worklist)
              worklist (pop worklist)
              preds (remove alive (get rev s))]
          (recur (into alive preds)
                 (into worklist preds)))))))

(defn compile-regex
  "Compile a regex string to a DFA with precomputed reachability.

   Returns {:start :accept :transitions :alive} where:
     :start       — integer start state
     :accept      — set of accept states
     :transitions — {[state char] -> state}
     :alive       — set of states that can reach an accept state"
  [regex-str]
  (let [ast (parse-regex regex-str)
        nfa (ast->nfa ast)
        dfa (nfa->dfa nfa)]
    (assoc dfa :alive (compute-alive-states dfa))))

;; ============================================================
;; BPE token decoding
;; ============================================================

(defn- bpe-byte-decoder
  "Build the BPE byte → actual character mapping.
   BPE tokenizers use a bijective mapping from bytes to unicode chars
   to avoid control characters in the vocabulary."
  []
  (let [bs (concat (range 33 127) (range 161 173) (range 174 256))
        cs (vec bs)
        extra-bs (remove (set bs) (range 256))
        extra-cs (map #(+ 256 %) (range (count extra-bs)))]
    (merge (zipmap (map #(js/String.fromCharCode %) cs)
                   (map #(js/String.fromCharCode %) bs))
           (zipmap (map #(js/String.fromCharCode %) extra-cs)
                   (map #(js/String.fromCharCode %) extra-bs)))))

(defn- decode-bpe-token
  "Convert a BPE token string to actual text."
  [decoder tok-str]
  (when tok-str
    (apply str (map #(get decoder (str %) (str %)) tok-str))))

(defn build-token-index
  "Build a vector mapping token-id → decoded text string.
   The BPE byte encoding is decoded to actual characters so that
   DFA transitions work on real text."
  [tokenizer]
  (let [decoder (bpe-byte-decoder)
        vocab-size (.vocabSize tokenizer)]
    (mapv (fn [id]
            (decode-bpe-token decoder (.idToToken tokenizer id)))
          (range vocab-size))))

;; ============================================================
;; Token mask computation
;; ============================================================

(def ^:private neg-inf js/Number.NEGATIVE_INFINITY)

(defn compute-valid-mask
  "For a given DFA state, compute a logit mask over the vocabulary.
   Returns a Float32Array: 0.0 for valid tokens, -Infinity for invalid.

   A token is valid if advancing through all its chars from the given
   state doesn't reach :dead and the resulting state is alive or accept."
  [dfa state token-index]
  (let [n (count token-index)
        mask (js/Float32Array. n)
        alive (:alive dfa)]
    (.fill mask neg-inf)
    (dotimes [i n]
      (let [tok-str (nth token-index i)
            final (when (and tok-str (pos? (count tok-str)))
                    (dfa-advance-string dfa state tok-str))]
        (when (and (some? final)
                   (not= final :dead)
                   (contains? alive final))
          (aset mask i 0.0))))
    mask))

(defn precompute-masks
  "Precompute valid-token masks for every DFA state.
   Returns a map of {state -> Float32Array mask}.
   Trades memory for O(1) runtime lookup per token step."
  [dfa token-index]
  (into {} (for [state (:alive dfa)]
             [state (compute-valid-mask dfa state token-index)])))

;; ============================================================
;; Grammar constraint record
;; ============================================================

(defn compile-constraint
  "Compile a regex constraint for use with constrained LLM generation.

   Returns a constraint map:
     :dfa          — the compiled DFA
     :token-index  — decoded token strings per ID
     :eos-id       — the EOS token ID
     :masks        — precomputed masks (if DFA is small enough)

   If the DFA has <= max-precompute states, masks are precomputed
   for O(1) lookup. Otherwise computed on demand.

   opts map (optional):
     :token-index    — pre-built token index (from build-token-index)
     :max-precompute — max DFA states for mask precomputation (default 50)"
  ([tokenizer regex-str] (compile-constraint tokenizer regex-str {}))
  ([tokenizer regex-str opts]
   (let [dfa (compile-regex regex-str)
         token-index (or (:token-index opts) (build-token-index tokenizer))
         eos-id (.getEosTokenId tokenizer)
         max-pre (get opts :max-precompute 50)
         n-states (count (:alive dfa))
         masks (when (<= n-states max-pre)
                 (precompute-masks dfa token-index))]
     {:dfa dfa
      :token-index token-index
      :eos-id eos-id
      :masks masks})))

(defn get-mask
  "Get the logit mask for a DFA state, using precomputed cache if available."
  [{:keys [dfa token-index masks]} state]
  (or (get masks state)
      (compute-valid-mask dfa state token-index)))

(defn apply-mask
  "Apply a grammar mask to logits. Returns masked logits as MxArray.

   Handles size mismatch between vocab and model output dim by padding
   with -Infinity. Also handles EOS: EOS is valid only when the DFA
   is in an accept state.

   Uses .fromFloat32 directly (not mx/array, which doesn't handle
   JS TypedArrays — it would create a scalar instead of a vector)."
  [constraint dfa-state logits]
  (when (= dfa-state :dead)
    (throw (ex-info "apply-mask: DFA is in the :dead state — every token would be
masked to -inf and the categorical would sample NaN. This happens when
generation continues past a grammar-final token (e.g. EOS text advanced
through the DFA) or the prior text left the grammar's language."
                    {:dfa-state dfa-state})))
  (let [{:keys [dfa eos-id]} constraint
        src-mask (get-mask constraint dfa-state)
        logits-n (first (mx/shape logits))
        ;; Copy mask to avoid mutating precomputed cache. Truncate when the
        ;; tokenizer vocab exceeds the model's logits dim (.set with a longer
        ;; source throws RangeError); ids beyond logits-n cannot be sampled.
        mask-arr (let [buf (js/Float32Array. logits-n)]
                   (.fill buf neg-inf)
                   (.set buf (if (> (.-length src-mask) logits-n)
                               (.subarray src-mask 0 logits-n)
                               src-mask))
                   buf)]
    ;; EOS handling: valid only in accept states (bounds-check for model/vocab mismatch)
    (when (< eos-id logits-n)
      (aset mask-arr eos-id (if (contains? (:accept dfa) dfa-state) 0.0 neg-inf)))
    (let [mask-mx (.fromFloat32 mx/core mask-arr #js [logits-n])]
      (mx/add logits mask-mx))))

;; ============================================================
;; Grammar constraint middleware
;; ============================================================

(defn wrap-grammar
  "Wrap a handler transition with grammar constraints.

   Ring-style middleware (same pattern as wrap-analytical):
   intercepts categorical distributions, masks invalid tokens per
   the DFA state, delegates to base-transition, advances DFA.

   Grammar state lives in :grammar-state on the handler state map.
   Initialized from closure default on first access. Purely functional.

   Usage:
     (-> (make-llm-gf model-map)
         (dispatch/with-handler
           (wrap-grammar h/generate-transition constraint)))"
  [base-transition constraint]
  (let [{:keys [dfa token-index]} constraint
        init-dfa-state (:start dfa)]
    (fn [state addr dist]
      (if (= :categorical (:type dist))
        (let [dfa-state (get state :grammar-state init-dfa-state)
              logits (get-in dist [:params :logits])
              masked-logits (apply-mask constraint dfa-state logits)
              masked-dist (dist/categorical masked-logits)
              [value state'] (base-transition state addr masked-dist)
              tok-id (mx/item value)
              ;; EOS terminates generation; advancing the DFA through the EOS
              ;; LITERAL text would reach :dead and NaN any further site
              ;; (genmlx-xwxh). Keep the (accepting) state instead.
              new-dfa-state (if (= tok-id (:eos-id constraint))
                              dfa-state
                              (dfa-advance-string dfa dfa-state
                                                  (nth token-index tok-id "")))]
          [value (assoc state' :grammar-state new-dfa-state)])
        (base-transition state addr dist)))))

(def ^:private standard-transitions
  "Per-op standard handler transitions, mirrored from the dispatcher table.
   constrain wraps EACH op's own transition: a single generate-flavored
   transition serving every op silently ran generate semantics under
   update/regenerate (old-choices/selection ignored, wrong weights for
   token-MCMC; genmlx-xwxh)."
  {:simulate h/simulate-transition, :generate h/generate-transition
   :update h/update-transition,     :regenerate h/regenerate-transition
   :assess h/assess-transition,     :project h/project-transition
   :propose h/simulate-transition})

(defn constrain
  "Convenience: apply grammar constraint to a generative function.

   Returns a new GF where all categorical distributions are masked
   by the grammar — under EVERY GFI op, each running its own standard
   transition wrapped with the grammar middleware. Works with any GF
   that uses dist/categorical — not limited to LLMs.

   Equivalent to:
     (dispatch/with-handler gf
       {:simulate (wrap-grammar h/simulate-transition constraint)
        :generate (wrap-grammar h/generate-transition constraint)
        ...})"
  [gf constraint]
  (dispatch/with-handler gf
    (assoc (into {} (map (fn [[op t]] [op (wrap-grammar t constraint)]))
                 standard-transitions)
           ;; The retained-only GENERAL regenerate transition, grammar-masked, so
           ;; a structure-changing constrained move (e.g. resampling an early
           ;; token shifts EOS and changes the number of sites) routes through the
           ;; correct retained-only weight instead of the fast per-site
           ;; (new-score − old-score) − ratio (genmlx-fayo C8). The dispatcher
           ;; gates fast vs general; this supplies the general arm.
           :regenerate-general (wrap-grammar h/regenerate-transition-general constraint))))

;; ============================================================
;; Vectorized grammar (genmlx-9uyg) — batched-DFA tables + hook
;;
;; vsimulate/vgenerate BYPASS the dispatcher, so wrap-grammar cannot
;; intercept a [K]-lane LLM-GF. Instead the grammar becomes DATA: dense
;; [S V] tables (per-state logit mask + per-(state, token) transition), and
;; per-lane DFA state becomes a [K] int32 array driven by pure gathers —
;; the whole constraint stays on-GPU, one gather per step. Semantics mirror
;; wrap-grammar exactly: masked logits renormalize through categorical's
;; log-softmax, eos is valid only in accept states, and sampling eos HOLDS
;; the state (never advances through the eos literal — genmlx-xwxh).
;; ============================================================

(defn build-vtables
  "Compile a `compile-constraint` result into dense batched-DFA tables over
   a vocab of `vocab-size` logits:

     :mask-table  [S V] f32 — 0 for valid (state, token), -inf otherwise;
                  row r is exactly apply-mask's mask for state r (incl. the
                  eos-only-in-accept override and -inf beyond token-index).
     :trans-table [S V] int32 — dense row index after consuming the token;
                  invalid tokens route to the dead row, eos self-loops.
     :states      alive DFA states in row order (dense row -> DFA state).
     :start-row / :dead-row — dense indices. The dead row absorbs and keeps
                  eos at 0 so it can never NaN a score; live lanes cannot
                  reach it (only masked-out tokens transition there).

   Build cost is one get-mask + one dfa-advance-string per VALID
   (state, token) pair — the same S×V sweep precompute-masks already does."
  [{:keys [dfa token-index eos-id] :as constraint} vocab-size]
  (let [states (vec (sort (:alive dfa)))
        row-of (zipmap states (range))
        dead   (count states)
        S      (inc dead)
        V      vocab-size
        n      (min (count token-index) V)
        mask   (js/Float32Array. (* S V))
        trans  (js/Int32Array. (* S V))]
    (.fill mask neg-inf)
    (.fill trans dead)
    (doseq [[s r] (map vector states (range))]
      (let [src  (get-mask constraint s)
            base (* r V)]
        (dotimes [t n]
          (when (zero? (aget src t))
            (aset mask (+ base t) 0.0)
            (aset trans (+ base t)
                  (get row-of (dfa-advance-string dfa s (nth token-index t))
                      dead))))
        (when (< eos-id V)
          (aset mask (+ base eos-id)
                (if (contains? (:accept dfa) s) 0.0 neg-inf))
          (aset trans (+ base eos-id) r))))
    (when (< eos-id V)
      (aset mask (+ (* dead V) eos-id) 0.0))
    {:mask-table  (.fromFloat32 mx/core mask #js [S V])
     :trans-table (.fromInt32 mx/core trans #js [S V])
     :states      states
     :start-row   (get row-of (:start dfa))
     :dead-row    dead}))

(defn vectorized-hook
  "The make-llm-gf-batched :hook over build-vtables output. State = dense
   DFA row per lane: a scalar int32 before the first sample establishes K,
   a [K] int32 array after. :mask row-gathers the per-lane logit masks;
   :advance gathers next rows by (state, token) — all pure MLX graph ops."
  [{:keys [mask-table trans-table start-row]}]
  {:init (fn [] (mx/scalar start-row mx/int32))
   :mask (fn [st logits _i]
           (mx/add logits (mx/take-idx mask-table st 0)))
   :advance (fn [st tok]
              (let [tok  (mx/astype tok mx/int32)
                    rows (mx/take-idx trans-table st 0)]     ; [V] or [K V]
                (if (pos? (count (mx/shape st)))
                  (mx/squeeze (mx/take-along-axis rows (mx/expand-dims tok 1) 1)
                              [1])                            ; [K]
                  (mx/take-idx rows tok 0))))})               ; scalar st: [K]
