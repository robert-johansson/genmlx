;; @tier fast
(ns genmlx.llm.schema-grammar-test
  "Model-free, exhaustive tests for the pure malli-schema -> EDN-grammar layer
   (genmlx-xi71). Independent oracle: hand-written conforming / non-conforming
   EDN strings checked against the compiled DFA, plus hand-written expected
   regex strings. No LLM/GPU involved."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.llm.schema-grammar :as sg]
            [genmlx.llm.grammar :as grammar]))

;; ------------------------------------------------------------
;; DFA acceptance oracle: does the schema's regex accept string s exactly?
;; ------------------------------------------------------------
(defn dfa-accepts?
  "True iff walking the schema's DFA over every char of s consumes the whole
   string and ends in an accept state, never entering a dead state."
  [schema s]
  (let [dfa   (grammar/compile-regex (sg/schema->regex schema))
        alive (:alive dfa)
        acc   (:accept dfa)]
    (loop [state (:start dfa) i 0]
      (if (= i (count s))
        (contains? acc state)
        (let [ns (grammar/dfa-advance dfa state (subs s i (inc i)))]
          (if (contains? alive ns)
            (recur ns (inc i))
            false))))))

(defn accepts-all [schema strs]
  (doseq [s strs] (is (dfa-accepts? schema s) (str "should ACCEPT " (pr-str s)))))

(defn rejects-all [schema strs]
  (doseq [s strs] (is (not (dfa-accepts? schema s)) (str "should REJECT " (pr-str s)))))

;; ------------------------------------------------------------
;; schema -> regex string (independent oracle = hand-written regex)
;; ------------------------------------------------------------
(deftest schema-regex-shapes
  (testing "leaf regexes (canonical EDN numbers)"
    (is (= "(0|-?[1-9][0-9]*)" (sg/schema->regex :int)))
    (is (= "(0|[1-9][0-9]*)" (sg/schema->regex [:int {:min 0}])))
    (is (= "-?(0|[1-9][0-9]*)(\\.[0-9]+)?" (sg/schema->regex :double)))
    (is (= "(true|false)" (sg/schema->regex :boolean)))
    (is (= "\"[^\"]*\"" (sg/schema->regex :string)))
    (is (= "nil" (sg/schema->regex :nil))))
  (testing "literal / enum / maybe"
    (is (= "5" (sg/schema->regex [:= 5])))
    (is (= ":foo" (sg/schema->regex [:= :foo])))
    (is (= "\"hi\"" (sg/schema->regex [:= "hi"])))
    (is (= "(:red|:green|:blue)" (sg/schema->regex [:enum :red :green :blue])))
    (is (= "(1|2|3)" (sg/schema->regex [:enum 1 2 3])))
    (is (= "(nil|(0|-?[1-9][0-9]*))" (sg/schema->regex [:maybe :int]))))
  (testing "containers"
    (is (= "\\{:age (0|-?[1-9][0-9]*)\\}" (sg/schema->regex [:map [:age :int]])))
    (is (= "\\{:name \"[^\"]*\" :age (0|-?[1-9][0-9]*)\\}"
           (sg/schema->regex [:map [:name :string] [:age :int]])))
    (is (= "\\[((0|-?[1-9][0-9]*)( (0|-?[1-9][0-9]*))*)?\\]" (sg/schema->regex [:vector :int])))
    (is (= "\\[(0|-?[1-9][0-9]*) \"[^\"]*\"\\]" (sg/schema->regex [:tuple :int :string])))))

;; ------------------------------------------------------------
;; DFA semantics: accept conforming, reject non-conforming
;; ------------------------------------------------------------
(deftest dfa-int
  (accepts-all :int ["0" "42" "-7" "1234567"])
  (rejects-all :int ["" "3.5" "abc" "4-" "+3" "1 2"])
  (testing "canonical: no -0, no leading zeros"
    (rejects-all :int ["-0" "007" "00"]))
  (testing "unsigned :min>=0 forbids the minus"
    (accepts-all [:int {:min 0}] ["0" "9"])
    (rejects-all [:int {:min 0}] ["-3" "-0"])))

(deftest dfa-double
  (accepts-all :double ["3.14" "-0.5" "42" "-7"])
  (rejects-all :double ["3." ".5" "abc" "1.2.3"]))

(deftest dfa-boolean-nil-string-keyword
  (accepts-all :boolean ["true" "false"])
  (rejects-all :boolean ["True" "0" "yes"])
  (accepts-all :nil ["nil"])
  (rejects-all :nil ["null" "Nil"])
  (accepts-all :string ["\"\"" "\"ann\"" "\"with space\""])
  (rejects-all :string ["ann" "\"unterminated" "no-quotes"])
  (accepts-all :keyword [":red" ":a" ":multi-word"])
  (rejects-all :keyword ["red" ":1bad" ":"]))

(deftest dfa-enum-maybe
  (accepts-all [:enum :red :green :blue] [":red" ":green" ":blue"])
  (rejects-all [:enum :red :green :blue] [":blue!" "red" ":yellow" ""])
  (accepts-all [:maybe :int] ["nil" "42" "-3"])
  (rejects-all [:maybe :int] ["none" "3.5" "" "false"]))

(deftest dfa-map
  (let [s [:map [:name :string] [:age :int]]]
    (accepts-all s ["{:name \"ann\" :age 30}"
                    "{:name \"\" :age -5}"
                    "{:name \"a b\" :age 0}"])
    (rejects-all s ["{:age 30 :name \"ann\"}"          ; wrong key order
                    "{:name \"ann\"}"                   ; missing key
                    "{:name ann :age 30}"               ; unquoted string
                    "{:name \"ann\" :age 3.5}"          ; non-int
                    "{:name \"ann\", :age 30}"          ; comma (non-canonical)
                    "{ :name \"ann\" :age 30 }"         ; extra spaces
                    "{:name \"ann\" :age 30"]))         ; unterminated
  (testing "nested map"
    (let [s [:map [:person [:map [:name :string]]] [:n :int]]]
      (accepts-all s ["{:person {:name \"x\"} :n 3}"])
      (rejects-all s ["{:person {:name x} :n 3}" "{:person {} :n 3}"]))))

(deftest dfa-vector-tuple
  (accepts-all [:vector :int] ["[]" "[1]" "[1 2 3]" "[-4 5 6]"])
  (rejects-all [:vector :int] ["[1, 2]" "[1 2" "[a]" "[1.5]" "1 2 3"])
  (testing "bounded vector"
    (accepts-all [:vector {:min 1 :max 3} :int] ["[1]" "[1 2]" "[1 2 3]"])
    (rejects-all [:vector {:min 1 :max 3} :int] ["[]" "[1 2 3 4]"]))
  (testing "tuple is fixed-arity"
    (accepts-all [:tuple :int :string] ["[1 \"a\"]" "[-9 \"\"]"])
    (rejects-all [:tuple :int :string] ["[1]" "[1 \"a\" 2]" "[\"a\" 1]"])))

;; ------------------------------------------------------------
;; canonical serialization round-trips with schema order
;; ------------------------------------------------------------
(deftest canonical-serialization
  (testing "schema-ordered map serialization (NOT value seq order)"
    (let [s [:map [:name :string] [:age :int]]
          v {:age 30 :name "ann"}]                       ; value built age-first
      (is (= "{:name \"ann\" :age 30}" (sg/schema->canonical-str s v))
          "entries follow schema order regardless of value order")
      (is (dfa-accepts? s (sg/schema->canonical-str s v))
          "serialized value is accepted by its own schema's DFA")))
  (testing "leaf + composite canonical strings"
    (is (= ":red" (sg/canonical-str :red)))
    (is (= "\"x\"" (sg/canonical-str "x")))
    (is (= "[1 2 3]" (sg/canonical-str [1 2 3])))
    (is (= "{:a 1 :b 2}" (sg/canonical-str (array-map :a 1 :b 2)))))
  (testing "vector + tuple schema serialization"
    (is (= "[1 2 3]" (sg/schema->canonical-str [:vector :int] [1 2 3])))
    (is (= "[3 \"hi\"]" (sg/schema->canonical-str [:tuple :int :string] [3 "hi"])))
    (is (= "nil" (sg/schema->canonical-str [:maybe :int] nil)))
    (is (= "7" (sg/schema->canonical-str [:maybe :int] 7)))))

;; ------------------------------------------------------------
;; parse + validate
;; ------------------------------------------------------------
(deftest parse-validate
  (let [s [:map [:name :string] [:age :int]]]
    (is (= {:ok? true :value {:name "ann" :age 30}}
           (sg/parse-and-validate s "{:name \"ann\" :age 30}")))
    (is (= :parse-error (:error (sg/parse-and-validate s "{:name \"ann\" :age"))))
    (is (= :schema-mismatch (:error (sg/parse-and-validate s "{:name \"ann\" :age \"x\"}")))))
  (testing "round-trip: serialize then parse-and-validate"
    (let [s [:map [:tags [:vector :keyword]] [:n :int]]
          v {:tags [:a :b] :n 5}
          str* (sg/schema->canonical-str s v)]
      (is (dfa-accepts? s str*))
      (is (= {:ok? true :value v} (sg/parse-and-validate s str*))))))

;; ------------------------------------------------------------
;; conditioning: constrain-schema fixes fields
;; ------------------------------------------------------------
(deftest conditioning
  (let [s [:map [:name :string] [:age :int]]]
    (testing "fixing :age bakes a literal into the regex"
      (is (= "\\{:name \"[^\"]*\" :age 30\\}"
             (sg/schema->regex (sg/constrain-schema s {:age 30}))))
      (let [cs (sg/constrain-schema s {:age 30})]
        (accepts-all cs ["{:name \"ann\" :age 30}"])
        (rejects-all cs ["{:name \"ann\" :age 31}"])))
    (testing "fixing a nested field"
      (let [s2 [:map [:p [:map [:x :int]]] [:y :int]]
            cs (sg/constrain-schema s2 {:p {:x 1}})]
        (accepts-all cs ["{:p {:x 1} :y 9}"])
        (rejects-all cs ["{:p {:x 2} :y 9}"])))))

;; ------------------------------------------------------------
;; unsupported forms throw clearly (no silent fallthrough)
;; ------------------------------------------------------------
(deftest unsupported-forms-throw
  (is (not (sg/supported? [:map [:a {:optional true} :int]])) ":optional unsupported")
  (is (not (sg/supported? [:map-of :keyword :int])) ":map-of unsupported")
  (is (sg/supported? [:map [:a :int]]))
  (is (sg/supported? [:vector [:enum :x :y]]))
  (is (thrown? js/Error (sg/schema->regex [:map-of :keyword :int]))))

(cljs.test/run-tests)
