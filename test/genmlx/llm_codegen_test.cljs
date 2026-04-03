(ns genmlx.llm-codegen-test
  "Tests for code generation pipeline (Phase 7).

   Two sections:
   1. Pure tests (no model needed) — reader constraint, validation,
      chat template, code extraction, eval, verification
   2. Model tests (Qwen3-0.6B) — generate-cljs, synthesize-loop"
  (:require [genmlx.llm.codegen :as cg]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.llm.bytes :as bytes]
            [genmlx.protocols :as p]
            [edamame.core :as eda]
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
  (assert-true (str msg " (expected " (pr-str expected) ", got " (pr-str actual) ")")
               (= expected actual)))

(defn- report []
  (let [p @pass-count f @fail-count]
    (println (str "\n=== " p "/" (+ p f) " PASS ==="))
    (when (pos? f) (println (str "!!! " f " FAILURES !!!")))))

;; ============================================================
;; 1. Pure tests (no model)
;; ============================================================

;; -- 1.1 prefix-status --

(println "\n== prefix-status: progressive prefixes ==")

(let [code "(fn [x] (+ x 1))"
      n (count code)
      statuses (mapv #(cg/prefix-status (subs code 0 (inc %))) (range n))]
  (assert-true "all prefixes except last are :incomplete"
               (every? #(= :incomplete %) (butlast statuses)))
  (assert-equal "final prefix is :complete" :complete (last statuses)))

(println "\n== prefix-status: realistic transition fn ==")

(let [code "(fn [{:keys [x y]} action] (case action :up {:x x :y (dec y)} {:x x :y y}))"
      statuses (mapv #(cg/prefix-status (subs code 0 (inc %))) (range (count code)))]
  (assert-equal "final is :complete" :complete (last statuses))
  (assert-true "no intermediate is :complete"
               (not-any? #(= :complete %) (butlast statuses))))

(println "\n== prefix-status: invalid strings ==")

(assert-equal "')' alone is invalid" :invalid (cg/prefix-status ")"))
(assert-equal "'(+ ])' is invalid" :invalid (cg/prefix-status "(+ ])"))
(assert-equal "'(+ 1 2]]' is invalid" :invalid (cg/prefix-status "(+ 1 2]]"))

(println "\n== prefix-status: complete forms ==")

(assert-equal "(+ 1 2) complete" :complete (cg/prefix-status "(+ 1 2)"))
(assert-equal "#{1 2 3} complete" :complete (cg/prefix-status "#{1 2 3}"))
(assert-equal "(let [a 1] a) complete" :complete (cg/prefix-status "(let [a 1] a)"))

;; -- 1.2 valid-next-bytes --

(println "\n== valid-next-bytes ==")

(let [valid (cg/valid-next-bytes "(fn [x")]
  (assert-true "']' valid after '(fn [x'" (contains? valid "]"))
  (assert-true "' ' valid after '(fn [x'" (contains? valid " "))
  (assert-true "'a' valid after '(fn [x'" (contains? valid "a"))
  (assert-true "')' invalid after '(fn [x'" (not (contains? valid ")")))
  (assert-true "'}' invalid after '(fn [x'" (not (contains? valid "}"))))

;; -- 1.3 reader-constraint --

(println "\n== reader-constraint ==")

(let [rc (cg/reader-constraint "(+ 1")]
  (assert-true "' ' is :incomplete" (= :incomplete (get rc " ")))
  (assert-true "')' is :complete" (= :complete (get rc ")")))
  (assert-true "']' is excluded" (nil? (get rc "]"))))

;; -- 1.4 valid-cljs? --

(println "\n== valid-cljs? ==")

(assert-true "(+ 1 2) is valid" (cg/valid-cljs? "(+ 1 2)"))
(assert-true "(defn add [a b] (+ a b)) is valid"
             (cg/valid-cljs? "(defn add [a b] (+ a b))"))
(assert-true "(fn [{:keys [x y]} a] ...) is valid"
             (cg/valid-cljs? "(fn [{:keys [x y]} a] (case a :up {:x x :y (dec y)} {:x x :y y}))"))
(assert-true "'foo is valid" (cg/valid-cljs? "'foo"))
(assert-true "( is not valid" (not (cg/valid-cljs? "(")))
(assert-true "(defn is not valid" (not (cg/valid-cljs? "(defn")))
(assert-true "empty string is not valid" (not (cg/valid-cljs? "")))

;; -- 1.5 Form predicates --

(println "\n== form predicates ==")

(let [fn-form (eda/parse-string "(fn [x] (+ x 1))")
      defn-form (eda/parse-string "(defn add [a b] (+ a b))")
      trans-form (eda/parse-string "(fn [state action] (assoc state :x 1))")
      one-arg (eda/parse-string "(fn [x] x)")
      not-fn (eda/parse-string "(+ 1 2)")]
  (assert-true "fn-form? on fn" (cg/fn-form? fn-form))
  (assert-true "fn-form? rejects defn" (not (cg/fn-form? defn-form)))
  (assert-true "defn-form? on defn" (cg/defn-form? defn-form))
  (assert-true "defn-form? rejects fn" (not (cg/defn-form? fn-form)))
  (assert-true "transition-fn-form? on 2-arg fn" (cg/transition-fn-form? trans-form))
  (assert-true "transition-fn-form? rejects 1-arg fn" (not (cg/transition-fn-form? one-arg)))
  (assert-true "fn-form? rejects non-fn" (not (cg/fn-form? not-fn))))

;; -- 1.6 format-chat --

(println "\n== format-chat ==")

(let [result (cg/format-chat "sys" "usr")]
  (assert-true "contains system tag"
               (clojure.string/includes? result "<|im_start|>system\nsys<|im_end|>"))
  (assert-true "contains user tag"
               (clojure.string/includes? result "<|im_start|>user\nusr<|im_end|>"))
  (assert-true "contains assistant + think-skip"
               (clojure.string/includes? result "<|im_start|>assistant\n<think>\n\n</think>\n\n")))

;; -- 1.7 extract-code --

(println "\n== extract-code ==")

(assert-equal "fenced clojure"
              "(+ 1 2)"
              (cg/extract-code "Here is code:\n```clojure\n(+ 1 2)\n```\nDone."))

(assert-equal "fenced cljs"
              "(+ 1 2)"
              (cg/extract-code "```cljs\n(+ 1 2)\n```"))

(assert-equal "bare fence"
              "(+ 1 2)"
              (cg/extract-code "```\n(+ 1 2)\n```"))

(assert-equal "raw code (starts with paren)"
              "(+ 1 2)"
              (cg/extract-code "(+ 1 2)"))

(assert-equal "prefix text stripped"
              "(+ 1 2)"
              (cg/extract-code "The answer is: (+ 1 2)"))

(assert-equal "empty input" "" (cg/extract-code ""))
(assert-equal "nil input" "" (cg/extract-code nil))
(assert-equal "no code found" "" (cg/extract-code "no code here"))

;; -- 1.8 eval-cljs --

(println "\n== eval-cljs ==")

(let [r (cg/eval-cljs "(+ 1 2)")]
  (assert-equal "eval valid code" 3 (:result r)))

(let [r (cg/eval-cljs "(/ 1 0)")]
  (assert-true "eval division by zero returns result (Infinity in JS)"
               (some? (:result r))))

(let [r (cg/eval-cljs "(throw (js/Error. \"boom\"))")]
  (assert-true "eval throw returns error" (some? (:error r)))
  (assert-equal "error message" "boom" (:error r)))

(let [r (cg/eval-cljs "(defn")]
  (assert-true "eval parse error returns error" (some? (:error r))))

;; -- 1.9 eval-fn --

(println "\n== eval-fn ==")

(let [r (cg/eval-fn "(fn [a b] (+ a b))")]
  (assert-true "eval-fn returns fn" (some? (:fn r)))
  (assert-equal "fn works" 7 ((:fn r) 3 4)))

(let [r (cg/eval-fn "42")]
  (assert-true "eval-fn rejects non-fn" (some? (:error r)))
  (assert-equal "error message" "Result is not a function" (:error r)))

(let [r (cg/eval-fn "(defn")]
  (assert-true "eval-fn catches parse error" (some? (:error r))))

;; -- 1.10 verify-transition-fn --

(println "\n== verify-transition-fn ==")

(let [good-fn "(fn [{:keys [x y]} action] (case action :up {:x x :y (dec y)} :down {:x x :y (inc y)} :left {:x (dec x) :y y} :right {:x (inc x) :y y}))"
      transitions [{:state {:x 5 :y 5} :action :up    :expected {:x 5 :y 4}}
                   {:state {:x 5 :y 5} :action :down  :expected {:x 5 :y 6}}
                   {:state {:x 5 :y 5} :action :left  :expected {:x 4 :y 5}}
                   {:state {:x 5 :y 5} :action :right :expected {:x 6 :y 5}}]
      result (cg/verify-transition-fn good-fn transitions)]
  (assert-equal "good fn: accuracy 1.0" 1.0 (:accuracy result))
  (assert-equal "good fn: 4 correct" 4 (:correct result))
  (assert-equal "good fn: 0 failures" 0 (count (:failures result))))

(let [bad-fn "(fn [{:keys [x y]} action] (case action :up {:x x :y (dec y)} :down {:x x :y (inc y)} {:x x :y y}))"
      transitions [{:state {:x 5 :y 5} :action :up    :expected {:x 5 :y 4}}
                   {:state {:x 5 :y 5} :action :down  :expected {:x 5 :y 6}}
                   {:state {:x 5 :y 5} :action :left  :expected {:x 4 :y 5}}
                   {:state {:x 5 :y 5} :action :right :expected {:x 6 :y 5}}]
      result (cg/verify-transition-fn bad-fn transitions)]
  (assert-true "bad fn: accuracy < 1.0" (< (:accuracy result) 1.0))
  (assert-equal "bad fn: 2 correct" 2 (:correct result))
  (assert-equal "bad fn: 2 failures" 2 (count (:failures result)))
  (assert-true "failures have :index" (every? #(contains? % :index) (:failures result)))
  (assert-true "failure indices are 2 and 3"
               (= #{2 3} (set (map :index (:failures result))))))

(let [invalid-fn "(not valid clojure"
      transitions [{:state {:x 0} :action :up :expected {:x 1}}]
      result (cg/verify-transition-fn invalid-fn transitions)]
  (assert-equal "invalid fn: accuracy 0.0" 0.0 (:accuracy result))
  (assert-true "invalid fn: has error" (some? (:error result))))

;; -- 1.11 score-structure --

(println "\n== score-structure ==")

(let [good (eda/parse-string "(defn move [{:keys [x y]} action] (case action :up {:x x :y (dec y)} :down {:x x :y (inc y)} :left {:x (dec x) :y y} :right {:x (inc x) :y y}))" {:all true})
      bad (eda/parse-string "(defn move [{:keys [x y]} action] (case action :up (assoc x y 1) :down (assoc x y -1) :left (assoc x y -1) :right (assoc x y 1)))" {:all true})
      worse (eda/parse-string "(defn move [{:keys [x y]} action] (cond-> action :up (assoc-in [:y] (dec y)) :down (update-in [:y] inc)))" {:all true})
      good-score (cg/score-structure good)
      bad-score (cg/score-structure bad)
      worse-score (cg/score-structure worse)]
  (assert-true "good > bad" (> good-score bad-score))
  (assert-true "bad > worse" (> bad-score worse-score))
  (assert-true "good score positive" (pos? good-score))
  (assert-true "worse score <= 0" (<= worse-score 0)))

;; ============================================================
;; Summary of pure tests
;; ============================================================

(println "\n== Pure test summary ==")
(report)

;; ============================================================
;; 2. Model tests (Qwen3-0.6B)
;; ============================================================

(def model-dir (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b"))

(println "\n== Loading Qwen3-0.6B for model tests... ==")

(pr/let [model-map (llm/load-model model-dir)
         prepared (bytes/prepare (:tokenizer model-map))
         opts {:prepared prepared :max-bytes 200}]

  ;; -- 2.1 generate-cljs --
  (println "\n== generate-cljs ==")

  (pr/let [result (cg/generate-cljs model-map "Write a function that adds two numbers" opts)]
    (assert-true "generate-cljs: text non-empty" (pos? (count (:text result))))
    (assert-true "generate-cljs: code is string" (string? (:code result)))
    (assert-true "generate-cljs: has :valid? key" (contains? result :valid?))
    (assert-true "generate-cljs: has :text key" (some? (:text result)))
    (println "  Generated text:" (pr-str (:text result)))
    (println "  Extracted code:" (pr-str (:code result)))
    (println "  Valid?:" (:valid? result))
    (when (:valid? result)
      (assert-true "generate-cljs: valid code has :form" (some? (:form result)))))

  ;; -- 2.2 generate-cljs-n --
  (println "\n== generate-cljs-n ==")

  (pr/let [results (cg/generate-cljs-n model-map "Write (fn [x] (+ x 1))" 3 opts)]
    (assert-equal "generate-cljs-n: 3 candidates" 3 (count results))
    (assert-true "generate-cljs-n: sorted valid first"
                 (or (not (:valid? (last results)))
                     (every? :valid? results)))
    (println "  Candidates:")
    (doseq [[i r] (map-indexed vector results)]
      (println (str "    " i ": valid=" (:valid? r) " code=" (pr-str (subs (:code r) 0 (min 60 (count (:code r)))))))))

  ;; -- 2.3 synthesize-loop --
  (println "\n== synthesize-loop ==")

  (pr/let [transitions [{:state {:x 5 :y 5} :action :up    :expected {:x 5 :y 4}}
                         {:state {:x 5 :y 5} :action :down  :expected {:x 5 :y 6}}
                         {:state {:x 5 :y 5} :action :left  :expected {:x 4 :y 5}}
                         {:state {:x 5 :y 5} :action :right :expected {:x 6 :y 5}}]
           result (cg/synthesize-loop
                    model-map
                    "Write a ClojureScript transition function (fn [state action] ...) where state is {:x int :y int} and action is :up/:down/:left/:right. :up decrements y, :down increments y, :left decrements x, :right increments x."
                    transitions
                    (merge opts {:max-revisions 2}))]
    (assert-true "synthesize-loop: has :code" (string? (:code result)))
    (assert-true "synthesize-loop: has :accuracy" (number? (:accuracy result)))
    (assert-true "synthesize-loop: has :history" (vector? (:history result)))
    (assert-true "synthesize-loop: has :converged?" (contains? result :converged?))
    (println "  Accuracy:" (:accuracy result))
    (println "  Revisions:" (:revisions result))
    (println "  Converged?:" (:converged? result))
    (println "  Best code:" (pr-str (:code result))))

  ;; -- 2.4 revise --
  (println "\n== revise ==")

  (pr/let [bad-code "(fn [s a] s)"
           failures [{:state {:x 5 :y 5} :action :up
                      :expected {:x 5 :y 4} :actual {:x 5 :y 5}}]
           result (cg/revise model-map bad-code failures opts)]
    (assert-true "revise: produces output" (pos? (count (:text result))))
    (assert-true "revise: code differs from input"
                 (not= bad-code (:code result)))
    (println "  Revised code:" (pr-str (:code result))))

  ;; -- 2.5 generate-and-score --
  (println "\n== generate-and-score ==")

  (pr/let [gf (llm-core/make-llm-gf model-map)
           result (cg/generate-and-score model-map gf "Write (defn add [a b] (+ a b))"
                    {:temperature 0.3})]
    (assert-true "generate-and-score: has :code" (string? (:code result)))
    (assert-true "generate-and-score: has :valid?" (contains? result :valid?))
    (assert-true "generate-and-score: has :weight" (number? (:weight result)))
    (assert-true "generate-and-score: has :struct-score" (number? (:struct-score result)))
    (assert-true "generate-and-score: weight is negative" (neg? (:weight result)))
    (println "  Code:" (pr-str (subs (:code result) 0 (min 60 (count (:code result))))))
    (println "  Weight:" (:weight result) "Struct:" (:struct-score result)))

  ;; -- 2.6 generate-and-rank --
  (println "\n== generate-and-rank ==")

  (pr/let [gf (llm-core/make-llm-gf model-map)
           transitions [{:state {:x 5 :y 5} :action :up    :expected {:x 5 :y 4}}
                        {:state {:x 5 :y 5} :action :down  :expected {:x 5 :y 6}}]
           results (cg/generate-and-rank model-map gf
                     "Write (defn move [{:keys [x y]} action] (case action :up {:x x :y (dec y)} :down {:x x :y (inc y)}))"
                     3
                     {:temperature 0.7 :transitions transitions})]
    (assert-equal "generate-and-rank: 3 candidates" 3 (count results))
    (assert-true "generate-and-rank: sorted by combined desc"
                 (let [scores (mapv :combined results)]
                   (= scores (vec (sort > scores)))))
    (assert-true "generate-and-rank: all have :weight" (every? :weight results))
    (assert-true "generate-and-rank: all have :struct-score" (every? :struct-score results))
    (assert-true "generate-and-rank: all have :combined" (every? :combined results))
    (assert-true "generate-and-rank: all have :accuracy" (every? #(contains? % :accuracy) results))
    (println "  Candidates:")
    (doseq [[i r] (map-indexed vector results)]
      (println (str "    " i ": combined=" (int (:combined r))
                    " weight=" (int (:weight r))
                    " struct=" (:struct-score r)
                    " acc=" (:accuracy r)))))

  ;; Final report
  (println "\n== Final summary (pure + model) ==")
  (report))
