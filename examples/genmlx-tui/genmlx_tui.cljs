(ns genmlx-tui
  (:require ["ink" :refer [render Text Box Newline]]
            ["ink-text-input$default" :as TextInput]
            ["ink-spinner$default" :as Spinner]
            ["@mlx-node/lm" :as lm]
            [reagent.core :as r]))

;; --- State ---
(def state (r/atom {:status :loading
                    :input ""
                    :history []
                    :model nil
                    :generating false}))

(def model-path (str (.-HOME (.-env js/process))
                     "/.cache/models/Qwen3.6-35B-A3B-4bit"))

(def few-shot-context
  "You are a GenMLX expert. You write ClojureScript probabilistic models using the GenMLX API.

KEY RULES:
- (gen [args] body) is the top-level model macro
- (trace :keyword-address distribution) for random choices — addresses must be UNIQUE keywords
- In loops use (keyword (str \"name\" j)) for unique addresses
- Distributions: dist/gaussian, dist/bernoulli, dist/beta-dist, dist/gamma-dist, dist/exponential, dist/poisson, dist/uniform, dist/categorical, dist/delta
- MLX ops: mx/add, mx/multiply, mx/subtract, mx/divide, mx/exp, mx/log, mx/scalar, mx/where, mx/equal
- Use doseq with map-indexed for observation loops

EXAMPLE 1 — Coin with unknown bias:
(def coin-model
  (gen []
    (let [p (trace :p (dist/beta-dist 2 2))]
      (trace :y0 (dist/bernoulli p))
      (trace :y1 (dist/bernoulli p))
      p)))

EXAMPLE 2 — Bayesian linear regression:
(def linreg
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str \"y\" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

EXAMPLE 3 — Gaussian mixture:
(def gmm
  (gen [xs]
    (let [mu1    (trace :mu1 (dist/gaussian 0 10))
          mu2    (trace :mu2 (dist/gaussian 0 10))
          weight (trace :weight (dist/beta-dist 2 2))]
      (doseq [[j x] (map-indexed vector xs)]
        (let [z (trace (keyword (str \"z\" j)) (dist/bernoulli weight))
              mu (mx/where (mx/equal z (mx/scalar 1)) mu1 mu2)]
          (trace (keyword (str \"y\" j))
                 (dist/gaussian mu 1))))
      {:mu1 mu1 :mu2 mu2 :weight weight})))

Output ONLY the (def ... (gen ...)) form. No markdown. No explanation.")

;; --- LLM interaction ---
(defn generate-code [model prompt]
  (js/Promise.
    (fn [resolve reject]
      (let [messages #js [#js {:role "system" :content few-shot-context}
                          #js {:role "user" :content prompt}]
            config #js {:maxTokens 500
                        :temperature 0
                        :reasoningEffort "none"}
            stream (.chatStreamSessionStart model messages config)]
        (-> (js/Promise.resolve
              ((fn iter [text]
                 (-> (.next stream)
                     (.then (fn [result]
                              (if (.-done result)
                                (resolve text)
                                (let [event (.-value result)]
                                  (if (.-done event)
                                    (resolve text)
                                    (do
                                      (swap! state update :current-output
                                             #(str (or % "") (.-text event)))
                                      (iter (str text (.-text event)))))))))))
               ""))
            (.catch reject))))))

(defn validate-cljs [code]
  (try
    (cljs.reader/read-string code)
    true
    (catch :default _ false)))

(defn submit-prompt [prompt]
  (when (and (seq prompt) (not (:generating @state)))
    (let [model (:model @state)]
      (swap! state assoc
             :generating true
             :current-output ""
             :input "")
      (.resetCaches model)
      (-> (generate-code model prompt)
          (.then (fn [result]
                   (let [valid (validate-cljs result)]
                     (swap! state
                            #(-> %
                                 (assoc :generating false
                                        :current-output nil)
                                 (update :history conj
                                         {:prompt prompt
                                          :code result
                                          :valid valid}))))))
          (.catch (fn [err]
                    (swap! state
                           #(-> %
                                (assoc :generating false
                                       :current-output nil)
                                (update :history conj
                                        {:prompt prompt
                                         :code (str "Error: " (.-message err))
                                         :valid false})))))))))

;; --- UI Components ---
(defn header []
  [:> Box {:borderStyle "round"
           :borderColor "cyan"
           :paddingX 1
           :width 80}
   [:> Text {:bold true :color "cyan"}
    " GenMLX Lisp Machine "]
   [:> Text {:color "gray"} " | Qwen3.6-35B-A3B | "
    (str (count (:history @state)) " generations")]])

(defn code-block [{:keys [prompt code valid]}]
  [:> Box {:flexDirection "column" :marginBottom 1}
   [:> Text {:color "yellow" :bold true} (str "> " prompt)]
   [:> Box {:borderStyle "single"
            :borderColor (if valid "green" "red")
            :paddingX 1
            :width 78
            :flexDirection "column"}
    [:> Text {:color (if valid "white" "gray")} code]]
   [:> Text {:color (if valid "green" "red")}
    (if valid " valid cljs" " invalid")]])

(defn input-area []
  (let [{:keys [input generating]} @state]
    [:> Box {:flexDirection "column"}
     (when generating
       [:> Box {:marginBottom 1}
        [:> Spinner {:type "dots"}]
        [:> Text {:color "yellow"} " generating..."]])
     (when-let [partial (:current-output @state)]
       (when (and generating (seq partial))
         [:> Box {:borderStyle "single" :borderColor "yellow"
                  :paddingX 1 :width 78 :flexDirection "column"
                  :marginBottom 1}
          [:> Text {:color "gray"} partial]]))
     [:> Box
      [:> Text {:color "cyan" :bold true} "genmlx> "]
      [:> TextInput {:value input
                     :onChange #(swap! state assoc :input %)
                     :onSubmit submit-prompt
                     :placeholder "describe a probabilistic model..."}]]]))

(defn app []
  (let [{:keys [status history]} @state]
    [:> Box {:flexDirection "column" :padding 1}
     [header]
     [:> Newline]
     (if (= status :loading)
       [:> Box
        [:> Spinner {:type "dots"}]
        [:> Text {:color "yellow"} " Loading Qwen3.6-35B-A3B-4bit..."]]
       [:> Box {:flexDirection "column"}
        ;; Show last 3 results
        (for [entry (take-last 3 history)]
          ^{:key (:prompt entry)}
          [code-block entry])
        [input-area]])
     [:> Newline]
     [:> Text {:color "gray"} " ctrl+c to exit"]]))

;; --- Main ---
(def ink-instance (render (r/as-element [app])))

(-> (lm/loadModel model-path)
    (.then (fn [model]
             (swap! state assoc
                    :status :ready
                    :model model)
             (println "")))
    (.catch (fn [err]
              (swap! state assoc :status :error)
              (js/console.error "Failed to load model:" (.-message err))
              (js/process.exit 1))))
