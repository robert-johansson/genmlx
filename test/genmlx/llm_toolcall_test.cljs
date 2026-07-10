;; @tier slow
(ns genmlx.llm-toolcall-test
  "genmlx-4tcd: grammar-constrained qwen3_xml tool calls.

   T1 (pure, model-free): the tool-call regex compiles to a DFA that
      accepts exactly the dialect — valid blocks (with prose around them,
      multiple blocks, schema-constrained values) accepted; malformed
      envelopes, undeclared names/params, off-pattern values rejected.
   T2 (pure): parse-tool-calls (the independent scanner oracle) agrees.
   T3 (0.8b, batched): K=4 lanes sampled through the VECTORIZED grammar
      hook at temperature 2.0 (the adversarial case that breaks
      parse-and-retry) — every lane's text parses, every call's
      function/params are declared, zero parse errors. K constrained
      candidates in ONE forward (genmlx-9uyg composition).
   T4 (0.8b, scalar): grammar/constrain + p/simulate — the middleware
      path produces parseable text too.

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_toolcall_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as core]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.toolcall :as tc]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def tools
  [{:name "get_weather"
    :params [{:name "location"}
             {:name "unit" :pattern "(celsius|fahrenheit)"}]}
   {:name "calculate"
    :params [{:name "expression" :pattern "[\\-+*/()0-9. ]+"}]}
   {:name "set_timer"
    :params [{:name "minutes" :schema :int}                 ; malli -> regex
             {:name "label" :schema [:enum "tea" "eggs"]}]}])

;; T3/T4 sampling variant: bounded value patterns + capped param count +
;; block-first, so a temperature-2.0 sample can NEVER exceed the token
;; budget mid-block (truncation would be a sampling artifact, not a
;; grammar failure — see tool-call-regex docstring).
(def tools-bounded
  [{:name "get_weather"
    :params [{:name "location" :pattern "[A-Za-z,. ]{1,12}"}
             {:name "unit" :pattern "(celsius|fahrenheit)"}]}
   {:name "calculate"
    :params [{:name "expression" :pattern "[\\-+*/()0-9. ]{1,12}"}]}])

(def valid-block
  (str "<tool_call>\n<function=get_weather>\n"
       "<parameter=location>\nParis\n</parameter>\n"
       "<parameter=unit>\ncelsius\n</parameter>\n"
       "</function>\n</tool_call>"))

(def valid-calc
  (str "<tool_call>\n<function=calculate>\n"
       "<parameter=expression>\n2 + 2 * 10\n</parameter>\n"
       "</function>\n</tool_call>"))

(println "\n-- T1: DFA accepts the dialect, rejects malformations --")
(let [dfa (gram/compile-regex (tc/tool-call-regex tools))]
  (assert-true "empty string (pure prose) accepted"
               (gram/dfa-accepts? dfa ""))
  (assert-true "plain prose accepted"
               (gram/dfa-accepts? dfa "I will check the weather now.\n"))
  (assert-true "bare valid block accepted"
               (gram/dfa-accepts? dfa valid-block))
  (assert-true "prose + block + prose accepted"
               (gram/dfa-accepts? dfa (str "Let me check.\n" valid-block "\nDone.")))
  (assert-true "two blocks with prose between accepted"
               (gram/dfa-accepts? dfa (str valid-block "\nand\n" valid-calc)))
  (assert-true "zero-parameter call accepted"
               (gram/dfa-accepts? dfa "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"))
  (assert-true "undeclared function rejected"
               (not (gram/dfa-accepts? dfa (str "<tool_call>\n<function=rm_rf>\n"
                                                "</function>\n</tool_call>"))))
  (assert-true "undeclared parameter rejected"
               (not (gram/dfa-accepts? dfa (str "<tool_call>\n<function=get_weather>\n"
                                                "<parameter=city>\nParis\n</parameter>\n"
                                                "</function>\n</tool_call>"))))
  (assert-true "cross-tool parameter rejected (expression on get_weather)"
               (not (gram/dfa-accepts? dfa (str "<tool_call>\n<function=get_weather>\n"
                                                "<parameter=expression>\n1+1\n</parameter>\n"
                                                "</function>\n</tool_call>"))))
  (assert-true "off-pattern value rejected (unit=kelvin)"
               (not (gram/dfa-accepts? dfa (str "<tool_call>\n<function=get_weather>\n"
                                                "<parameter=unit>\nkelvin\n</parameter>\n"
                                                "</function>\n</tool_call>"))))
  (assert-true "unclosed block rejected"
               (not (gram/dfa-accepts? dfa "<tool_call>\n<function=get_weather>\n")))
  (assert-true "missing newline after opener rejected"
               (not (gram/dfa-accepts? dfa (str "<tool_call><function=get_weather>\n"
                                                "</function>\n</tool_call>"))))
  (assert-true "prose cannot smuggle a bare '<'"
               (not (gram/dfa-accepts? dfa "a < b")))
  (assert-true "malli :int param accepted (set_timer minutes=-3)"
               (gram/dfa-accepts? dfa (str "<tool_call>\n<function=set_timer>\n"
                                           "<parameter=minutes>\n-3\n</parameter>\n"
                                           "</function>\n</tool_call>")))
  (assert-true "malli :int param rejects non-numeric value"
               (not (gram/dfa-accepts? dfa (str "<tool_call>\n<function=set_timer>\n"
                                                "<parameter=minutes>\nsoon\n</parameter>\n"
                                                "</function>\n</tool_call>"))))
  (assert-true "malli enum param accepted (label=tea) / rejected (label=x)"
               (and (gram/dfa-accepts? dfa (str "<tool_call>\n<function=set_timer>\n"
                                                "<parameter=label>\n\"tea\"\n</parameter>\n"
                                                "</function>\n</tool_call>"))
                    (not (gram/dfa-accepts? dfa (str "<tool_call>\n<function=set_timer>\n"
                                                     "<parameter=label>\n\"x\"\n</parameter>\n"
                                                     "</function>\n</tool_call>"))))))

(println "\n-- T2: independent parse oracle agrees --")
(let [{:keys [calls errors]} (tc/parse-tool-calls
                              (str "Checking.\n" valid-block "\n" valid-calc)
                              tools)]
  (assert-true "oracle finds 2 calls, 0 errors"
               (and (= 2 (count calls)) (empty? errors)))
  (assert-true "call 1 = get_weather{location Paris, unit celsius}"
               (= (first calls)
                  {:name "get_weather"
                   :args {"location" "Paris" "unit" "celsius"}
                   :param-order ["location" "unit"]}))
  (assert-true "call 2 = calculate{expression 2 + 2 * 10}"
               (= (get-in (second calls) [:args "expression"]) "2 + 2 * 10")))
(let [{:keys [errors]} (tc/parse-tool-calls
                        "<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>"
                        tools)]
  (assert-true "oracle flags undeclared function" (seq errors)))

(def model-dir
  (let [d (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")]
    (when (.existsSync fs (path/join d "tokenizer.json")) d)))

(defn- summary []
  (println (str "\n== llm-toolcall: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(if-not model-dir
  (do (println "SKIP T3/T4 — 0.8b absent") (summary))
  (->
   (pr/let [mm (llm/load-model model-dir)]
     (let [{:keys [model tokenizer]} mm
           vocab (get-in model [:fwd :config :vocab])
           prompt-str (str "You have these tools:\n"
                           "get_weather(location, unit=celsius|fahrenheit)\n"
                           "calculate(expression)\n"
                           "Call a tool to answer: what is the weather in Paris in celsius?\n"
                           "Answer with a <tool_call> block only.\n")
           t0 (js/Date.now)
           constraint (tc/compile-toolcall tokenizer tools-bounded
                                           {:block-first? true :max-params 2})
           t1 (js/Date.now)
           _ (println (str "  [info] constraint compiled in " (- t1 t0) " ms ("
                           (count (get-in constraint [:dfa :alive])) " alive states)"))
           vt-tables (gram/build-vtables constraint vocab)
           t2 (js/Date.now)
           _ (println (str "  [info] vtables built in " (- t2 t1) " ms"))
           base-hook (gram/vectorized-hook vt-tables)
           ;; adversarial high temperature: scale logits BEFORE masking —
           ;; the flattened distribution stress-tests the grammar
           inv-temp (mx/scalar 0.5)   ; T = 2.0
           hot-hook (assoc base-hook
                           :mask (fn [st logits i]
                                   ((:mask base-hook) st (mx/multiply logits inv-temp) i)))]
       (pr/let [ids (llm/encode tokenizer prompt-str false)]
         (let [prompt (vec ids)
               gf (core/make-llm-gf-batched mm {:hook hot-hook})
               K 4
               vt (dyn/vsimulate gf [prompt 96] K (rng/fresh-key 1234))]
           (pr/let [texts (core/decode-vtrace tokenizer vt)]
             (println "\n-- T3: K=4 constrained lanes at temperature 2.0 --")
             (doseq [[k txt] (map-indexed vector texts)]
               (let [{:keys [calls errors]} (tc/parse-tool-calls txt tools-bounded)]
                 (println (str "  lane " k ": " (count calls) " call(s), "
                               (count errors) " error(s)"
                               (when (seq calls)
                                 (str " — " (:name (first calls)) " "
                                      (pr-str (:args (first calls)))))))
                 (assert-true (str "lane " k " parses with zero errors")
                              (empty? errors))
                 (assert-true (str "lane " k " contains >= 1 tool call")
                              (>= (count calls) 1))
                 (assert-true (str "lane " k " calls only declared tools")
                              (every? #(contains? #{"get_weather" "calculate"} (:name %))
                                      calls))))
             ;; T4: the scalar middleware path (single-call shape: exactly
             ;; one block + trailing prose — a second block could truncate
             ;; at max-tokens, a sampling artifact, not a grammar failure)
             (println "\n-- T4: scalar grammar/constrain path --")
             (let [c-single (tc/compile-toolcall tokenizer tools-bounded
                                                 {:block-first? true :max-params 2
                                                  :max-calls 1
                                                  :token-index (:token-index constraint)})
                   gf-s (gram/constrain (core/make-llm-gf mm) c-single)
                   tr (p/simulate gf-s [prompt 96])]
               (pr/let [txt (core/decode-trace tokenizer tr)]
                 (let [{:keys [calls errors]} (tc/parse-tool-calls txt tools-bounded)]
                   (println (str "  scalar: " (count calls) " call(s), "
                                 (count errors) " error(s)"))
                   (assert-true "scalar sample parses with zero errors" (empty? errors))
                   (assert-true "scalar sample contains >= 1 tool call" (>= (count calls) 1)))
                 (mx/force-gc!)
                 true)))))))
   (pr/then (fn [_] (summary)))
   (pr/catch (fn [e]
               (println "ERROR:" (.-message e) "\n" (.-stack e))
               (swap! fail inc)
               (summary)))))
