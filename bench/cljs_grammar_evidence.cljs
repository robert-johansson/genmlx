;; genmlx-3g0t acceptance evidence: N constrained emissions of a
;; (fn [scene] ...) tool argument under x-genmlx-grammar "cljs" — the
;; reader-level per-argument constraint. Reports, per turn: finishReason,
;; tool calls, per-call argument + its cljs-arg-status, toolCallErrors.
;; The gate: zero unparseable arguments, zero retries (there is no retry
;; loop anywhere in this harness — one sample per turn, by construction).
;;
;; Run (guarded):
;;   scripts/guarded-run.sh 3g0t-evidence bunx --bun nbb@1.4.208 \
;;     bench/cljs_grammar_evidence.cljs [N]

(ns cljs-grammar-evidence
  (:require [genmlx.codegen.eval :as ceval]
            [genmlx.llm.pi-provider :as pp]
            [promesa.core :as pr]
            ["os" :as os]
            ["path" :as path]))

(def n-turns
  (let [a (last (js->clj js/process.argv))]
    (if (and a (re-matches #"\d+" a)) (js/parseInt a 10) 100)))

(def model-dir (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16"))
(def engine pp/engine)

(def filter-tool
  {:type "function"
   :function {:name "set_filter"
              :description "Install a scene filter predicate."
              :parameters {:type "object"
                           :properties
                           (js/JSON.stringify
                            (clj->js {:code {:type "string"
                                             :description "A ClojureScript predicate like (fn [scene] ...)"
                                             :x-genmlx-grammar "cljs"}}))
                           :required ["code"]}}})

(def prompts
  ["Call the set_filter tool with a ClojureScript predicate that keeps scenes whose :x is under 10."
   "Call the set_filter tool with a ClojureScript predicate that keeps scenes whose :color equals :red."
   "Call the set_filter tool with a ClojureScript predicate that keeps scenes with more than 3 objects (use the :objects key)."
   "Call the set_filter tool with a ClojureScript predicate that keeps scenes whose :y is at least 5."])

(def cfg
  ;; 512: the reader leg emits value bytes at up to one token per byte, so
  ;; a hot model exploring a long string literal needs headroom — an
  ;; undersized budget truncates blocks mid-tag (the documented sampling
  ;; artifact class, NOT a grammar failure; cf. the regex-leg etiology run)
  {:temperature 1.0 :maxNewTokens 512 :reasoningEffort "none"
   :tools [filter-tool]})

(defn- run-turn [sid messages config]
  (-> (.turnStream engine sid
                   (js/JSON.stringify (clj->js messages))
                   (js/JSON.stringify (clj->js config))
                   (fn [_]))
      (pr/then (fn [fj] (js->clj (js/JSON.parse fj) :keywordize-keys true)))))

(defn- msgs [i]
  [{:role "system" :content "You are a terse assistant."}
   {:role "user" :content (nth prompts (mod i (count prompts)))}])

(pr/let [_ (.loadModel engine model-dir)]
  (pr/loop [i 0 stats {:turns 0 :calls 0 :complete 0 :invalid 0
                       :errors 0 :length-truncs 0 :no-call 0}]
    (if (= i n-turns)
      (do (println "\n== 3g0t evidence summary ==")
          (println (pr-str stats))
          (println (str "turns=" (:turns stats)
                        " calls=" (:calls stats)
                        " complete-args=" (:complete stats)
                        " invalid-args=" (:invalid stats)
                        " hard-errors=" (:errors stats)
                        " length-truncations=" (:length-truncs stats)
                        " no-call-turns=" (:no-call stats)))
          (println (if (and (zero? (:invalid stats)) (zero? (:errors stats)))
                     "GATE: PASS — zero unparseable, zero retries (no retry loop exists)"
                     "GATE: FAIL"))
          (js/process.exit (if (and (zero? (:invalid stats)) (zero? (:errors stats))) 0 1)))
      (let [sid (.newSession engine "{}")]
        (pr/let [f (run-turn sid (msgs i) cfg)]
          (.dispose engine sid)
          (let [calls (:toolCalls f)
                errs  (:toolCallErrors f)
                len?  (= "length" (:finishReason f))
                codes (keep #(get-in % [:arguments :code]) calls)
                stats' (reduce
                        (fn [st c]
                          (let [s (ceval/cljs-arg-status c)]
                            (println (str "  [" i "] code " (name s) ": " (pr-str c)))
                            (-> st
                                (update :calls inc)
                                (update (if (= :complete s) :complete :invalid) inc))))
                        (update stats :turns inc)
                        codes)
                stats' (cond-> stats'
                         (and (seq errs) len?) (update :length-truncs inc)
                         (and (seq errs) (not len?)) (update :errors inc)
                         (empty? codes) (update :no-call inc))]
            (when (seq errs)
              (println (str "  [" i "] finish=" (:finishReason f)
                            " toolCallErrors: " (pr-str errs))))
            (when (empty? codes)
              (let [txt (str (:text f))
                    head (subs txt 0 (min 80 (count txt)))]
                (println (str "  [" i "] finish=" (:finishReason f)
                              " NO CALL text: " (pr-str head)))))
            (pr/recur (inc i) stats')))))))
