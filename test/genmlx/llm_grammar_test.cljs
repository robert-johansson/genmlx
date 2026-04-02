(ns genmlx.llm-grammar-test
  "Tests for grammar-constrained LLM generation (Phase 4).

   Three sections:
   1. DFA engine (pure, no model needed)
   2. Token masking (needs tokenizer)
   3. Constrained generation (needs model)"
  (:require [genmlx.llm.grammar :as gram]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [promesa.core :as pr]
            [instaparse.core :as insta]))

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
;; 1. DFA engine tests (pure, no model)
;; ============================================================

(println "\n== DFA: regex parsing ==")

(let [dfa (gram/compile-regex "ab")]
  (assert-true "ab accepts 'ab'" (gram/dfa-accepts? dfa "ab"))
  (assert-true "ab rejects 'a'" (not (gram/dfa-accepts? dfa "a")))
  (assert-true "ab rejects 'abc'" (not (gram/dfa-accepts? dfa "abc")))
  (assert-true "ab rejects ''" (not (gram/dfa-accepts? dfa ""))))

(let [dfa (gram/compile-regex "a|b")]
  (assert-true "a|b accepts 'a'" (gram/dfa-accepts? dfa "a"))
  (assert-true "a|b accepts 'b'" (gram/dfa-accepts? dfa "b"))
  (assert-true "a|b rejects 'ab'" (not (gram/dfa-accepts? dfa "ab")))
  (assert-true "a|b rejects 'c'" (not (gram/dfa-accepts? dfa "c"))))

(println "\n== DFA: quantifiers ==")

(let [dfa (gram/compile-regex "a*")]
  (assert-true "a* accepts ''" (gram/dfa-accepts? dfa ""))
  (assert-true "a* accepts 'a'" (gram/dfa-accepts? dfa "a"))
  (assert-true "a* accepts 'aaa'" (gram/dfa-accepts? dfa "aaa"))
  (assert-true "a* rejects 'b'" (not (gram/dfa-accepts? dfa "b"))))

(let [dfa (gram/compile-regex "a+")]
  (assert-true "a+ rejects ''" (not (gram/dfa-accepts? dfa "")))
  (assert-true "a+ accepts 'a'" (gram/dfa-accepts? dfa "a"))
  (assert-true "a+ accepts 'aaa'" (gram/dfa-accepts? dfa "aaa")))

(let [dfa (gram/compile-regex "a?")]
  (assert-true "a? accepts ''" (gram/dfa-accepts? dfa ""))
  (assert-true "a? accepts 'a'" (gram/dfa-accepts? dfa "a"))
  (assert-true "a? rejects 'aa'" (not (gram/dfa-accepts? dfa "aa"))))

(println "\n== DFA: character classes ==")

(let [dfa (gram/compile-regex "[a-z]+")]
  (assert-true "[a-z]+ accepts 'hello'" (gram/dfa-accepts? dfa "hello"))
  (assert-true "[a-z]+ rejects 'Hello'" (not (gram/dfa-accepts? dfa "Hello")))
  (assert-true "[a-z]+ rejects ''" (not (gram/dfa-accepts? dfa "")))
  (assert-true "[a-z]+ rejects '123'" (not (gram/dfa-accepts? dfa "123"))))

(let [dfa (gram/compile-regex "[A-Z][a-z]+")]
  (assert-true "Capitalized accepts 'Hello'" (gram/dfa-accepts? dfa "Hello"))
  (assert-true "Capitalized rejects 'hello'" (not (gram/dfa-accepts? dfa "hello")))
  (assert-true "Capitalized rejects 'H'" (not (gram/dfa-accepts? dfa "H"))))

(println "\n== DFA: escape sequences ==")

(let [dfa (gram/compile-regex "\\d+")]
  (assert-true "\\d+ accepts '123'" (gram/dfa-accepts? dfa "123"))
  (assert-true "\\d+ rejects 'abc'" (not (gram/dfa-accepts? dfa "abc"))))

(let [dfa (gram/compile-regex "\\w+@\\w+\\.\\w+")]
  (assert-true "email accepts 'user@host.com'" (gram/dfa-accepts? dfa "user@host.com"))
  (assert-true "email accepts 'a@b.c'" (gram/dfa-accepts? dfa "a@b.c"))
  (assert-true "email rejects '@host.com'" (not (gram/dfa-accepts? dfa "@host.com")))
  (assert-true "email rejects 'user@.com'" (not (gram/dfa-accepts? dfa "user@.com"))))

(println "\n== DFA: repeat quantifiers ==")

(let [dfa (gram/compile-regex "\\d{3}-\\d{4}")]
  (assert-true "phone accepts '123-4567'" (gram/dfa-accepts? dfa "123-4567"))
  (assert-true "phone rejects '12-4567'" (not (gram/dfa-accepts? dfa "12-4567")))
  (assert-true "phone rejects '1234-567'" (not (gram/dfa-accepts? dfa "1234-567")))
  (assert-true "phone rejects '123-456'" (not (gram/dfa-accepts? dfa "123-456")))
  (assert-true "phone rejects 'abc-defg'" (not (gram/dfa-accepts? dfa "abc-defg"))))

(println "\n== DFA: groups and complex patterns ==")

(let [dfa (gram/compile-regex "(hello|world)")]
  (assert-true "group accepts 'hello'" (gram/dfa-accepts? dfa "hello"))
  (assert-true "group accepts 'world'" (gram/dfa-accepts? dfa "world"))
  (assert-true "group rejects 'helloworld'" (not (gram/dfa-accepts? dfa "helloworld"))))

(let [dfa (gram/compile-regex "(ab)+")]
  (assert-true "(ab)+ accepts 'ab'" (gram/dfa-accepts? dfa "ab"))
  (assert-true "(ab)+ accepts 'abab'" (gram/dfa-accepts? dfa "abab"))
  (assert-true "(ab)+ rejects 'a'" (not (gram/dfa-accepts? dfa "a")))
  (assert-true "(ab)+ rejects 'aba'" (not (gram/dfa-accepts? dfa "aba"))))

(println "\n== DFA: advance and alive ==")

(let [dfa (gram/compile-regex "\\d{3}-\\d{4}")]
  (assert-equal "start state" (:start dfa) (gram/dfa-advance-string dfa (:start dfa) ""))
  (let [s1 (gram/dfa-advance-string dfa (:start dfa) "12")]
    (assert-true "after '12' not dead" (not= s1 :dead))
    (assert-true "after '12' alive" (contains? (:alive dfa) s1)))
  (let [s2 (gram/dfa-advance-string dfa (:start dfa) "123")]
    (assert-true "after '123' not dead" (not= s2 :dead))
    (assert-true "after '123' only '-' valid"
                 (and (not= :dead (gram/dfa-advance dfa s2 "-"))
                      (= :dead (gram/dfa-advance dfa s2 "5")))))
  (assert-equal "wrong char → dead" :dead (gram/dfa-advance dfa (:start dfa) "a")))

;; ============================================================
;; 2. Token masking tests (needs tokenizer)
;; ============================================================

(println "\n== Token masking (loading model...) ==")

(def home-dir (.-HOME (.-env js/process)))

(pr/let [model-map (llm/load-model (str home-dir "/.cache/models/qwen3-0.6b"))]
  (let [tokenizer (:tokenizer model-map)
        model (:model model-map)]

    (println "\n-- token index --")
    (let [token-index (gram/build-token-index tokenizer)]
      (assert-equal "token index size" (llm/vocab-size tokenizer) (count token-index))
      (assert-true "token 0 is '!'" (= "!" (nth token-index 0)))
      ;; BPE decoding: Ġ prefix → leading space
      (assert-true "BPE space decoding" (= " " (subs (nth token-index 9906) 0 1))))

    (println "\n-- mask computation --")
    (let [constraint (gram/compile-constraint tokenizer "\\d{3}-\\d{4}")
          dfa (:dfa constraint)
          token-index (:token-index constraint)]

      ;; At start: only single-digit tokens valid
      (let [mask (gram/get-mask constraint (:start dfa))
            valid-ids (filterv #(zero? (aget mask %)) (range (count token-index)))]
        (assert-equal "start: exactly 10 valid tokens" 10 (count valid-ids))
        (assert-true "start: valid tokens are digits"
                     (every? #(re-matches #"[0-9]" (nth token-index %)) valid-ids)))

      ;; After "123": only "-" valid
      (let [state (gram/dfa-advance-string dfa (:start dfa) "123")
            mask (gram/get-mask constraint state)
            valid-ids (filterv #(zero? (aget mask %)) (range (count token-index)))]
        (assert-equal "after '123': exactly 1 valid token" 1 (count valid-ids))
        (assert-equal "after '123': valid token is '-'" "-" (nth token-index (first valid-ids))))

      ;; Masks precomputed for small DFA
      (assert-true "masks precomputed" (some? (:masks constraint)))
      (assert-equal "mask count = alive states" (count (:alive dfa)) (count (:masks constraint))))

    ;; ============================================================
    ;; 3. Constrained generation tests (needs model)
    ;; ============================================================

    (let [constraint (gram/compile-constraint tokenizer "\\d{3}-\\d{4}")
          gf (gram/constrain (llm-core/make-llm-gf model-map) constraint)
          prompt-ids [6939 25 220] ;; "Phone: "
          token-index (:token-index constraint)
          eos-id (:eos-id constraint)
          phone-re #"\d{3}-\d{4}"
          decode-gen (fn [trace]
                       (let [gen-ids (->> (subvec (:retval trace) (count prompt-ids))
                                          (remove #(= % eos-id)))]
                         (apply str (map #(nth token-index %) gen-ids))))]

      (println "\n== Constrained simulate ==")

      ;; simulate produces valid phone numbers
      (pr/let [trace (p/simulate gf [prompt-ids 8])]
        (let [text (decode-gen trace)]
          (assert-true "simulate: output matches regex" (some? (re-matches phone-re text)))
          (assert-true "simulate: score is finite" (js/isFinite (mx/item (:score trace))))
          (assert-true "simulate: score is negative" (neg? (mx/item (:score trace))))

          ;; Multiple simulates produce variety
          (pr/let [t1 (p/simulate gf [prompt-ids 8])
                   t2 (p/simulate gf [prompt-ids 8])
                   t3 (p/simulate gf [prompt-ids 8])]
            (let [texts (mapv decode-gen [t1 t2 t3])]
              (assert-true "simulate: all match regex"
                           (every? #(some? (re-matches phone-re %)) texts))
              (assert-true "simulate: not all identical"
                           (< 1 (count (set texts))))

              (println "\n== Constrained generate ==")

              ;; generate: condition on first 3 tokens
              (let [obs (-> (cm/choicemap)
                            (cm/set-value :t0 (mx/scalar 19 mx/int32)) ;; "4"
                            (cm/set-value :t1 (mx/scalar 16 mx/int32)) ;; "1"
                            (cm/set-value :t2 (mx/scalar 20 mx/int32)))] ;; "5"
                (pr/let [{:keys [trace weight]} (p/generate gf [prompt-ids 8] obs)]
                  (let [text (decode-gen trace)]
                    (assert-true "generate: output matches regex" (some? (re-matches phone-re text)))
                    (assert-true "generate: starts with '415'" (clojure.string/starts-with? text "415"))
                    (assert-true "generate: weight is finite" (js/isFinite (mx/item weight)))
                    (assert-true "generate: score is finite" (js/isFinite (mx/item (:score trace))))

                    (println "\n== Instaparse validation ==")

                    (assert-true "valid? on matching string"
                                 (gram/valid? "S = #'[0-9]{3}-[0-9]{4}'" "123-4567"))
                    (assert-true "valid? rejects non-matching"
                                 (not (gram/valid? "S = #'[0-9]{3}-[0-9]{4}'" "abc")))
                    (assert-true "validate returns parse tree"
                                 (vector? (gram/validate "S = 'hello' <' '> 'world'" "hello world")))
                    (assert-true "validate returns failure on bad input"
                                 (insta/failure? (gram/validate "S = 'hello'" "goodbye")))

                    (println "\n== Different pattern: hex color ==")

                    (let [hex-constraint (gram/compile-constraint tokenizer "#[0-9a-f]{6}")
                          hex-gf (gram/constrain (llm-core/make-llm-gf model-map) hex-constraint)
                          hex-index (:token-index hex-constraint)
                          hex-eos (:eos-id hex-constraint)]
                      (pr/let [htrace (p/simulate hex-gf [prompt-ids 10])]
                        (let [htext (->> (subvec (:retval htrace) (count prompt-ids))
                                         (remove #(= % hex-eos))
                                         (map #(nth hex-index %))
                                         (apply str))]
                          (assert-true "hex: output matches #[0-9a-f]{6}"
                                       (some? (re-matches #"#[0-9a-f]{6}" htext)))
                          (println "  hex color:" htext)

                          (report))))))))))))))
