(ns genmlx.llm-bytes-test
  "Tests for byte-level token marginalization (Phase 5).

   Three sections:
   1. TokenByteTrie (pure, no model needed)
   2. Byte marginals (needs model)
   3. Byte-level generation (needs model)"
  (:require [genmlx.llm.bytes :as bytes]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [promesa.core :as pr]))

;; ============================================================
;; Test helpers
;; ============================================================

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [msg v]
  (if v
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn- assert-equal [msg expected actual]
  (assert-true (str msg " (expected " expected ", got " actual ")")
               (= expected actual)))

(defn- assert-close [msg expected actual tol]
  (assert-true (str msg " (expected ~" expected ", got " actual ")")
               (< (abs (- expected actual)) tol)))

(defn- report []
  (let [p @pass-count f @fail-count]
    (println (str "\n=== " p "/" (+ p f) " PASS ==="))
    (when (pos? f) (println (str "!!! " f " FAILURES !!!")))))

;; ============================================================
;; 1. TokenByteTrie tests (pure, no model)
;; ============================================================

;; Synthetic token-index: a vector where index = token ID, value = token string.
;; Token 0 = "a", 1 = "an", 2 = "and", 3 = "b", 4 = "be", 5 = "bee", 6 = "c"
(def ^:private synthetic-tokens ["a" "an" "and" "b" "be" "bee" "c"])

(println "\n== TokenByteTrie: build-byte-trie ==")

(let [trie (bytes/build-byte-trie synthetic-tokens)]

  ;; All 7 tokens have non-empty strings, so all should be reachable from root.
  (assert-equal "all token IDs reachable from root"
                7 (count (:all-token-ids trie)))

  ;; Root children should be the first bytes: "a", "b", "c"
  (assert-equal "root has 3 children" 3 (count (:children trie)))
  (assert-true "root child 'a' exists" (contains? (:children trie) "a"))
  (assert-true "root child 'b' exists" (contains? (:children trie) "b"))
  (assert-true "root child 'c' exists" (contains? (:children trie) "c"))

  ;; Disjoint partition: sum of reachable counts across root children == total
  ;; (root itself has no token-ids that terminate at root, so no empty-string tokens)
  (let [child-counts (map #(count (:all-token-ids %)) (vals (:children trie)))
        root-own (count (:token-ids trie))]
    (assert-equal "disjoint partition at root"
                  (count (:all-token-ids trie))
                  (+ (apply + child-counts) root-own)))

  ;; all-token-ids-arr is a sorted Int32Array
  (let [arr (:all-token-ids-arr trie)]
    (assert-true "all-token-ids-arr is Int32Array" (instance? js/Int32Array arr))
    (assert-equal "all-token-ids-arr length" 7 (.-length arr))
    (assert-true "all-token-ids-arr sorted"
                 (every? true?
                         (map #(<= (aget arr %) (aget arr (inc %)))
                              (range (dec (.-length arr)))))))

  ;; Token IDs at leaves
  ;; "c" -> token 6, leaf node
  (let [c-node (get-in trie [:children "c"])]
    (assert-true "c-node has token-id 6" (contains? (:token-ids c-node) 6))
    (assert-true "c-node is a leaf" (empty? (:children c-node))))

  ;; "and" -> token 2, should be a leaf (no longer tokens start with "and")
  (let [and-node (-> trie
                     (get-in [:children "a"])
                     (get-in [:children "n"])
                     (get-in [:children "d"]))]
    (assert-true "and-node has token-id 2" (contains? (:token-ids and-node) 2))
    (assert-true "and-node is a leaf (no children)" (empty? (:children and-node)))))

(println "\n== TokenByteTrie: trie-lookup ==")

(let [trie (bytes/build-byte-trie synthetic-tokens)]

  ;; Lookup "a" returns node for tokens starting with "a"
  (let [a-node (bytes/trie-lookup trie "a")]
    (assert-true "lookup 'a' not nil" (some? a-node))
    ;; "a" node should have all-token-ids = {0, 1, 2} (a, an, and)
    (assert-equal "lookup 'a' has 3 reachable tokens" 3 (count (:all-token-ids a-node))))

  ;; Lookup "an" returns node for "an"/"and"
  (let [an-node (bytes/trie-lookup trie "an")]
    (assert-true "lookup 'an' not nil" (some? an-node))
    ;; "an" node has token-ids {1} (token "an") and children for "d" -> token 2
    (assert-true "lookup 'an' has token-id 1" (contains? (:token-ids an-node) 1))
    (assert-equal "lookup 'an' has 2 reachable tokens" 2 (count (:all-token-ids an-node))))

  ;; Lookup "xyz" returns nil (no such path)
  (assert-true "lookup 'xyz' returns nil" (nil? (bytes/trie-lookup trie "xyz")))

  ;; Lookup "" returns root (reduce over empty range returns init)
  (let [root-node (bytes/trie-lookup trie "")]
    (assert-true "lookup '' returns trie root" (= trie root-node))))

(println "\n== TokenByteTrie: ambiguous and leaf nodes ==")

(let [trie (bytes/build-byte-trie synthetic-tokens)]

  ;; Ambiguous node: "a" has BOTH token-ids (token "a" = id 0) AND children ("n" path)
  (let [a-node (bytes/trie-lookup trie "a")]
    (assert-true "ambiguous node 'a' has token-ids" (contains? (:token-ids a-node) 0))
    (assert-true "ambiguous node 'a' has children" (pos? (count (:children a-node))))
    (assert-true "ambiguous node 'a' child 'n' exists" (contains? (:children a-node) "n")))

  ;; Pure leaf: "and" has token-ids but NO children
  (let [and-node (bytes/trie-lookup trie "and")]
    (assert-true "leaf 'and' has token-ids" (contains? (:token-ids and-node) 2))
    (assert-true "leaf 'and' has no children" (empty? (:children and-node))))

  ;; Pure leaf: "bee" has token-ids (5) but NO children
  (let [bee-node (bytes/trie-lookup trie "bee")]
    (assert-true "leaf 'bee' has token-ids" (contains? (:token-ids bee-node) 5))
    (assert-true "leaf 'bee' has no children" (empty? (:children bee-node)))))

(println "\n== TokenByteTrie: empty/nil token handling ==")

(let [;; Token-index with some nil/empty entries
      tokens-with-gaps [nil "" "ab" "cd" nil "ef"]
      trie (bytes/build-byte-trie tokens-with-gaps)]
  ;; Only tokens 2 ("ab"), 3 ("cd"), 5 ("ef") should be in the trie.
  (assert-equal "skips nil/empty tokens" 3 (count (:all-token-ids trie)))
  (assert-true "contains token 2" (contains? (:all-token-ids trie) 2))
  (assert-true "contains token 3" (contains? (:all-token-ids trie) 3))
  (assert-true "contains token 5" (contains? (:all-token-ids trie) 5)))

;; ============================================================
;; 2. Byte marginals (needs model)
;; ============================================================

(println "\n== Byte marginals (loading model...) ==")

(def home-dir (.-HOME (.-env js/process)))

(pr/let [model-map (llm/load-model (str home-dir "/.cache/models/qwen3-0.6b"))]
  (let [tokenizer (:tokenizer model-map)
        model (:model model-map)
        token-index (gram/build-token-index tokenizer)
        trie (bytes/build-byte-trie token-index)]

    (println "\n-- byte trie from real tokenizer --")

    (let [vocab (llm/vocab-size tokenizer)
          reachable (count (:all-token-ids trie))]
      (assert-true "real trie reachable > 100k" (> reachable 100000))
      (assert-true "real trie reachable <= vocab" (<= reachable vocab))
      (assert-true "root has many children (byte branching)" (> (count (:children trie)) 50)))

    (println "\n-- byte-logprobs at root --")

    ;; Get token logprobs from a simple prompt
    (let [prompt-ids [6939 25 220] ;; "Phone: "
          _ (llm/init-cache! model)
          logits (llm/forward-prefill model prompt-ids)
          log-probs (mx/subtract logits (mx/logsumexp logits))
          byte-lps (bytes/byte-logprobs trie log-probs)
          _ (llm/reset-cache! model)]

      ;; Returns a map with string keys (single chars)
      (assert-true "byte-logprobs returns a map" (map? byte-lps))
      (assert-true "keys are single-char strings"
                   (every? #(and (string? %) (= 1 (count %))) (keys byte-lps)))

      ;; All values are negative (log-probabilities)
      (assert-true "all values negative"
                   (every? neg? (vals byte-lps)))

      ;; exp of values sum to approximately 1.0
      ;; The sum accounts for all probability mass that goes through byte-
      ;; producing tokens. EOS and special tokens with empty strings are
      ;; excluded from the trie, so their mass is missing. The sum should
      ;; be close to but not exceed 1.0.
      (let [total (reduce + (map #(js/Math.exp %) (vals byte-lps)))]
        (assert-true "exp sum <= 1.0 (EOS mass excluded)" (<= total 1.001))
        (assert-true "exp sum > 0.95 (most mass in byte tokens)" (> total 0.95)))

      ;; Common ASCII bytes have reasonable probabilities (not -infinity)
      (let [common-chars ["a" "e" "t" "o" "1" "0" " "]]
        (assert-true "common bytes have finite logprobs"
                     (every? #(and (contains? byte-lps %)
                                   (js/isFinite (get byte-lps %)))
                             common-chars))))

    (println "\n-- byte-logprobs at mid-trie node --")

    ;; Navigate to a non-root node (e.g., the "t" subtree)
    (let [prompt-ids [6939 25 220]
          _ (llm/init-cache! model)
          logits (llm/forward-prefill model prompt-ids)
          log-probs (mx/subtract logits (mx/logsumexp logits))
          t-node (bytes/trie-lookup trie "t")
          _ (llm/reset-cache! model)]

      (assert-true "t-node exists" (some? t-node))

      (when t-node
        (let [child-lps (bytes/byte-logprobs t-node log-probs)]
          ;; Values are conditional log-probs
          (assert-true "mid-trie: returns a map" (map? child-lps))
          (assert-true "mid-trie: values are finite"
                       (every? js/isFinite (vals child-lps)))

          ;; exp of conditional values sum to at most 1.0
          ;; At non-leaf nodes with token-ids, the boundary probability (tokens
          ;; that terminate exactly at this node) is NOT included in the child
          ;; marginals. So exp sum < 1.0, and the gap is the boundary mass.
          ;; At nodes without own token-ids, the sum should be close to 1.0.
          (let [total (reduce + (map #(js/Math.exp %) (vals child-lps)))
                has-own-tokens (pos? (count (:token-ids t-node)))]
            (assert-true "mid-trie: exp sum <= 1.0" (<= total 1.001))
            (when has-own-tokens
              (println "  INFO: t-node has own tokens, boundary mass accounts for gap"))))))

    (println "\n-- monotone inclusion invariant --")

    ;; logsumexp of child marginals <= parent logsumexp
    ;; (child covers a subset of tokens, so its total mass <= parent's)
    (let [prompt-ids [6939 25 220]
          _ (llm/init-cache! model)
          logits (llm/forward-prefill model prompt-ids)
          log-probs (mx/subtract logits (mx/logsumexp logits))
          _ (llm/reset-cache! model)
          root-lps (bytes/byte-logprobs trie log-probs)
          root-lse (js/Math.log (reduce + (map #(js/Math.exp %) (vals root-lps))))]

      ;; Pick a child node ("t") and verify its total mass <= root total mass
      (let [t-node (bytes/trie-lookup trie "t")]
        (when t-node
          (let [child-lps (bytes/byte-logprobs t-node log-probs)
                child-lse (js/Math.log (reduce + (map #(js/Math.exp %) (vals child-lps))))]
            (assert-true "monotone: child logsumexp <= root logsumexp"
                         (<= child-lse (+ root-lse 1e-6)))))))

    ;; ============================================================
    ;; 3. Byte-level generation (needs model)
    ;; ============================================================

    (println "\n== Byte-level generation ==")

    (println "\n-- make-byte-llm-gf: unconstrained simulate --")

    (let [byte-gf (bytes/make-byte-llm-gf model-map)
          prompt-ids [6939 25 220]] ;; "Phone: "

      (pr/let [trace (p/simulate byte-gf [prompt-ids 20])]
        (let [retval (:retval trace)
              text (bytes/decode-byte-trace trace)
              score (mx/item (:score trace))]

          ;; retval is a vector of single-char strings
          (assert-true "retval is a vector" (vector? retval))
          (assert-true "retval length <= max-bytes" (<= (count retval) 20))
          (assert-true "retval elements are single-char strings"
                       (every? #(and (string? %) (= 1 (count %))) retval))

          ;; Score is finite and negative
          (assert-true "score is finite" (js/isFinite score))
          (assert-true "score is negative" (neg? score))

          ;; decode-byte-trace returns a non-empty string
          (assert-true "decoded text is a string" (string? text))
          (assert-true "decoded text is non-empty" (pos? (count text)))

          ;; Generated text is valid UTF-8 (apply str does not throw)
          (assert-true "text is valid (apply str works)" (string? (apply str retval)))

          (println "  generated text:" (pr-str text))

          (println "\n-- make-byte-llm-gf: zero max-bytes --")

          (pr/let [empty-trace (p/simulate byte-gf [prompt-ids 0])]
            (assert-equal "zero max-bytes: empty retval" [] (:retval empty-trace))
            (assert-equal "zero max-bytes: decoded empty" "" (bytes/decode-byte-trace empty-trace))

            (println "\n-- constrain-bytes: digit sequence regex --")

            (let [digit-gf (bytes/constrain-bytes model-map "[0-9]+")]

              (pr/let [ct (p/simulate digit-gf [prompt-ids 10])]
                (let [digit-text (bytes/decode-byte-trace ct)
                      digit-re #"[0-9]+"]

                  (assert-true "constrained: output matches digit regex"
                               (some? (re-matches digit-re digit-text)))
                  (assert-true "constrained: score is finite"
                               (js/isFinite (mx/item (:score ct))))

                  (println "  digit output:" (pr-str digit-text))

                  ;; Multiple simulates produce variety
                  (pr/let [ct1 (p/simulate digit-gf [prompt-ids 10])
                           ct2 (p/simulate digit-gf [prompt-ids 10])
                           ct3 (p/simulate digit-gf [prompt-ids 10])]
                    (let [texts (mapv bytes/decode-byte-trace [ct1 ct2 ct3])]
                      (assert-true "constrained: all match digit regex"
                                   (every? #(some? (re-matches digit-re %)) texts))
                      (assert-true "constrained: not all identical"
                                   (< 1 (count (set texts))))

                      (println "\n-- constrain-bytes: capitalized word regex --")

                      (let [cap-gf (bytes/constrain-bytes model-map "[A-Z][a-z]+")]

                        (pr/let [cap-trace (p/simulate cap-gf [prompt-ids 10])]
                          (let [cap-text (bytes/decode-byte-trace cap-trace)
                                cap-re #"[A-Z][a-z]+"]

                            (assert-true "capitalized: output matches regex"
                                         (some? (re-matches cap-re cap-text)))
                            (assert-true "capitalized: score is finite"
                                         (js/isFinite (mx/item (:score cap-trace))))

                            (println "  capitalized output:" (pr-str cap-text))

                            (println "\n-- constrain-bytes: compiled DFA input --")

                            ;; constrain-bytes also accepts a pre-compiled DFA
                            (let [dfa (gram/compile-regex "[0-9]{2}")
                                  dfa-gf (bytes/constrain-bytes model-map dfa)]

                              (pr/let [dfa-trace (p/simulate dfa-gf [prompt-ids 3])]
                                (let [dfa-text (bytes/decode-byte-trace dfa-trace)]
                                  (assert-true "dfa input: output matches [0-9]{2}"
                                               (some? (re-matches #"[0-9]{2}" dfa-text)))

                                  (println "  dfa input output:" (pr-str dfa-text))

                                  (println "\n-- constrain-bytes: early stop on stuck DFA --")

                                  (let [short-gf (bytes/constrain-bytes model-map "[ab]")]
                                    (pr/let [short-trace (p/simulate short-gf [prompt-ids 100])]
                                      (let [short-text (bytes/decode-byte-trace short-trace)]
                                        (assert-true "early stop: output is 'a' or 'b'"
                                                     (contains? #{"a" "b"} short-text))
                                        (assert-true "early stop: retval length 1"
                                                     (= 1 (count (:retval short-trace))))

                                        (println "  early stop output:" (pr-str short-text))

                                        (report)))))))))))))))))))))
