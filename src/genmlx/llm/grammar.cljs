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
     mask + logits → constrained categorical → GFI trace site"
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
(def ^:private printable-chars (set (map #(js/String.fromCharCode %) (range 32 127))))

;; ============================================================
;; Regex parsing (Instaparse → AST)
;; ============================================================

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
     lit = #'[^\\\\\\[\\]().*+?|{}]'
     escape = <'\\\\'> #'[dwsDWSntr\\\\.*+?|()\\[\\]{}]'
     class = <'['> neg? class-body <']'>
     neg = <'^'>
     class-body = class-item+
     <class-item> = range | class-char
     range = class-char <'-'> class-char
     class-char = #'[^\\]\\\\-]' | <'\\\\'> #'.'"))

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
  (let [tree (regex-grammar regex-str)]
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
       :class-char (fn [c] c)
       :range      (fn [from to] (char-range from to))
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
       :repeat-range (fn [n & [m]] [:repeat n (or m 999)])
       :quant      (fn [atom & [q]]
                     (cond
                       (nil? q)     atom
                       (= q :star)  [:star atom]
                       (= q :plus)  [:cat atom [:star atom]]
                       (= q :opt)   [:alt atom [:empty]]
                       (vector? q)  (let [[_ lo hi] q
                                          required (repeat lo atom)
                                          optional (repeat (- hi lo) [:alt atom [:empty]])]
                                      (reduce (fn [a b] [:cat a b])
                                              (concat required optional)))))
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
   (case (first ast)
     :lit   (add-transition nfa start (second ast) accept)
     :class (add-transition nfa start (second ast) accept)
     :empty (add-epsilon nfa start accept)
     :cat   (let [[_ a b] ast
                   [mid nfa] (fresh-state nfa)
                   nfa (ast->nfa nfa a start mid)]
              (ast->nfa nfa b mid accept))
     :alt   (let [[_ a b] ast
                   [[s1 f1 s2 f2] nfa] (fresh-states nfa 4)
                   nfa (-> nfa
                           (add-epsilon start s1)
                           (add-epsilon start s2)
                           (ast->nfa a s1 f1)
                           (ast->nfa b s2 f2)
                           (add-epsilon f1 accept)
                           (add-epsilon f2 accept))]
              nfa)
     :star  (let [[_ a] ast
                   [[s1 f1] nfa] (fresh-states nfa 2)
                   nfa (-> nfa
                           (add-epsilon start s1)
                           (ast->nfa a s1 f1)
                           (add-epsilon f1 s1)
                           (add-epsilon start accept)
                           (add-epsilon f1 accept))]
              nfa))))

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
                            dfa (cond-> dfa
                                  (not known?)
                                  (-> (assoc-in [:state-map target] id)
                                      (update :next-id inc))
                                  true
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

(def ^:private neg-inf-val
  "Large negative value for masking. Using -1e9 instead of -Infinity
   because MLX's NAPI crashes when creating arrays from Float32Arrays
   containing -Infinity values."
  -1000000.0)

(defn compute-valid-mask
  "For a given DFA state, compute a logit mask over the vocabulary.
   Returns a Float32Array: 0.0 for valid tokens, -1e6 for invalid.

   A token is valid if advancing through all its chars from the given
   state doesn't reach :dead and the resulting state is alive or accept."
  [dfa state token-index]
  (let [n (count token-index)
        mask (js/Float32Array. n)
        alive (:alive dfa)]
    (.fill mask neg-inf-val)
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
   for O(1) lookup. Otherwise computed on demand."
  ([tokenizer regex-str] (compile-constraint tokenizer regex-str 50))
  ([tokenizer regex-str max-precompute]
   (let [dfa (compile-regex regex-str)
         token-index (build-token-index tokenizer)
         eos-id (.getEosTokenId tokenizer)
         n-states (count (:alive dfa))
         masks (when (<= n-states max-precompute)
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

(def ^:private mlx-core (js/require "@mlx-node/core"))

(defn apply-mask
  "Apply a grammar mask to logits. Returns masked logits as MxArray.

   Handles size mismatch between vocab and model output dim by padding
   the mask with -1e6. Also handles EOS: EOS is valid only when the
   DFA is in an accept state.

   Note: we mutate a copy of the precomputed mask for EOS handling,
   then convert to MxArray via .fromFloat32 (not mx/array, which
   goes through clj->js and is too slow for 151K elements)."
  [constraint dfa-state logits]
  (let [{:keys [dfa eos-id]} constraint
        src-mask (get-mask constraint dfa-state)
        logits-n (first (mx/shape logits))
        ;; Copy mask to avoid mutating precomputed cache
        mask-arr (let [buf (js/Float32Array. logits-n)]
                   (.fill buf neg-inf-val)
                   (.set buf src-mask)
                   buf)]
    ;; EOS handling: valid only in accept states
    (aset mask-arr eos-id (if (contains? (:accept dfa) dfa-state) 0.0 neg-inf-val))
    (let [mask-mx (.fromFloat32 mlx-core mask-arr #js [logits-n])]
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
              tok-str (nth token-index tok-id "")
              new-dfa-state (dfa-advance-string dfa dfa-state tok-str)]
          [value (assoc state' :grammar-state new-dfa-state)])
        (base-transition state addr dist)))))

(defn constrain
  "Convenience: apply grammar constraint to a generative function.

   Returns a new GF where all categorical distributions are masked
   by the grammar. Works with any GF that uses dist/categorical —
   not limited to LLMs.

   Equivalent to:
     (dispatch/with-handler gf (wrap-grammar h/generate-transition constraint))"
  [gf constraint]
  (dispatch/with-handler gf (wrap-grammar h/generate-transition constraint)))

;; ============================================================
;; Instaparse validation (post-hoc)
;; ============================================================

(defn validate
  "Parse text against an Instaparse grammar string.
   Returns the parse tree on success, or a failure object."
  [grammar-str text]
  (let [parser (insta/parser grammar-str)]
    (parser text)))

(defn valid?
  "Does text match the given Instaparse grammar?"
  [grammar-str text]
  (not (insta/failure? (validate grammar-str text))))
