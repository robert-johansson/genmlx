;; genmlx-4tcd flagship demo: grammar-constrained qwen3_xml tool calls on
;; Ornith-1.0-35B (owned forward) — K constrained candidates in ONE batched
;; forward (Route B, genmlx-9uyg), sampled HOT (temperature 1.5, the regime
;; where parse-and-retry breaks), every candidate guaranteed parseable.
;;
;; Run (guarded, ONE GPU process):
;;   ~/genmlx-guarded-run.sh toolcall-demo \
;;     bunx --bun nbb@1.4.208 examples/toolcall_demo_35b.cljs
(ns examples.toolcall-demo-35b
  (:require [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as core]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.toolcall :as tc]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_OWNED_MOE_MODEL)
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))

(def tools
  [{:name "get_weather"
    :params [{:name "location" :pattern "[A-Za-z,. ]{1,24}"}
             {:name "unit" :pattern "(celsius|fahrenheit)"}]}
   {:name "calculate"
    :params [{:name "expression" :pattern "[\\-+*/()0-9. ]{1,24}"}]}])

(def system-prompt
  (str "You are a helpful assistant with access to these functions:\n"
       "- get_weather(location, unit): current weather; unit is celsius or fahrenheit.\n"
       "- calculate(expression): evaluate an arithmetic expression.\n"
       "Call a function by responding EXACTLY in this format:\n"
       "<tool_call>\n<function=FUNCTION_NAME>\n<parameter=PARAM_NAME>\n"
       "value\n</parameter>\n</function>\n</tool_call>"))

(def question "What's the weather in Paris right now, in celsius?")

(def K 6)
(def temperature 1.5)

(if-not model-dir
  (println "SKIP — Ornith-1.0-35B-4bit not cached")
  (->
   (pr/let [mm (llm/load-model model-dir)]
     (let [{:keys [model tokenizer]} mm
           vocab (get-in model [:fwd :config :vocab])
           chat (llm/render-chat [{:role "system" :content system-prompt}
                                  {:role "user" :content question}])]
       (println (str "model: " model-dir))
       (println (str "owned forward: " (llm/cljs-forward-model? model)))
       (pr/let [ids (llm/encode tokenizer chat false)]
         (let [prompt (vec ids)
               t0 (js/Date.now)
               constraint (tc/compile-toolcall tokenizer tools
                                               {:block-first? true
                                                :max-params 2 :max-calls 1})
               _ (println (str "constraint: "
                               (count (get-in constraint [:dfa :alive]))
                               " DFA states, compiled in " (- (js/Date.now) t0) " ms"))
               t1 (js/Date.now)
               vtables (gram/build-vtables constraint vocab)
               _ (println (str "vtables ([S V] mask+trans over " vocab
                               " logits): " (- (js/Date.now) t1) " ms"))
               base-hook (gram/vectorized-hook vtables)
               inv-temp (mx/scalar (/ 1.0 temperature))
               hook (assoc base-hook
                           :mask (fn [st logits i]
                                   ((:mask base-hook) st (mx/multiply logits inv-temp) i)))
               gf (core/make-llm-gf-batched mm {:hook hook})
               t2 (js/Date.now)
               vt (dyn/vsimulate gf [prompt 96] K (rng/fresh-key 20260710))]
           (pr/let [texts (core/decode-vtrace tokenizer vt)]
             (println (str "\n" K " constrained candidates in ONE batched forward, "
                           "temperature " temperature ", "
                           (- (js/Date.now) t2) " ms:\n"))
             (let [scores (vec (mx/->clj (mx/astype (:score vt) mx/float32)))
                   results
                   (vec (for [[k txt] (map-indexed vector texts)]
                          (let [{:keys [calls errors]} (tc/parse-tool-calls txt tools)]
                            {:lane k :calls calls :errors errors :score (nth scores k)})))]
               (doseq [{:keys [lane calls errors score]} results]
                 (println (str "lane " lane " (score " (.toFixed score 1) "): "
                               (if (seq errors)
                                 (str "PARSE ERRORS: " (pr-str errors))
                                 (apply str
                                        (for [c calls]
                                          (str (:name c) "("
                                               (clojure.string/join ", "
                                                 (for [[p v] (:args c)] (str p "=" (pr-str v))))
                                               ") ")))))))
               (let [ok (count (filter #(and (empty? (:errors %)) (seq (:calls %))) results))]
                 (println (str "\n" ok "/" K " candidates parse cleanly ("
                               "malformed tool calls are UNREPRESENTABLE — "
                               "constrained at sampling time, not parsed after)"))
                 (let [best (apply max-key :score (filter #(empty? (:errors %)) results))]
                   (println (str "highest-score candidate: lane " (:lane best) " -> "
                                 (pr-str (first (:calls best))))))))
             (mx/force-gc!)
             (js/process.exit 0))))))
   (pr/catch (fn [e]
               (println "ERROR:" (.-message e) "\n" (.-stack e))
               (js/process.exit 1)))))
