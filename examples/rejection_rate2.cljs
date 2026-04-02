(ns rejection-rate2
  (:require [genmlx.llm.codegen :as cg]
            [genmlx.llm.backend :as llm]
            [promesa.core :as pr]))

(def model-dir (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-cljs"))
(def N 20)

(def transitions
  [{:state {:x 5 :y 5} :action :up    :expected {:x 5 :y 4}}
   {:state {:x 5 :y 5} :action :down  :expected {:x 5 :y 6}}
   {:state {:x 5 :y 5} :action :left  :expected {:x 4 :y 5}}
   {:state {:x 5 :y 5} :action :right :expected {:x 6 :y 5}}
   {:state {:x 0 :y 0} :action :up    :expected {:x 0 :y -1}}
   {:state {:x 3 :y 7} :action :left  :expected {:x 2 :y 7}}])

;; More specific prompt — guide toward case pattern
(def prompt
  "Write (defn move [{:keys [x y]} action] ...) using case on action. For :up return {:x x :y (dec y)}, for :down {:x x :y (inc y)}, for :left {:x (dec x) :y y}, for :right {:x (inc x) :y y}.")

(println (str "Generating " N " candidates (temp=0.7, max-tokens=300)...\n"))

(pr/let [m (llm/load-model model-dir)
         t0 (js/Date.now)
         results (pr/all (repeatedly N #(cg/generate-cljs m prompt {:temperature 0.7 :max-tokens 300})))
         elapsed (- (js/Date.now) t0)]

  (println (str "Generated " N " in " (/ elapsed 1000.0) "s\n"))

  (let [classified
        (mapv (fn [i r]
                (let [valid (:valid? r)
                      v (when valid (cg/verify-transition-fn (:code r) transitions))
                      accuracy (when v (:accuracy v))]
                  {:i i :valid valid :accuracy accuracy :code (:code r)}))
              (range) results)

        n-valid (count (filter :valid classified))
        n-correct (count (filter #(= 1 (:accuracy %)) classified))
        n-partial (count (filter #(and (:accuracy %) (pos? (:accuracy %)) (< (:accuracy %) 1)) classified))]

    (doseq [{:keys [i valid accuracy code]} classified]
      (println (str "  " (if (< i 10) (str " " i) i) ": "
                    (cond (= 1 accuracy)        "CORRECT "
                          (and accuracy (pos? accuracy)) (str "PARTIAL(" accuracy ") ")
                          valid                  "VALID   "
                          :else                  "INVALID ")
                    (pr-str (subs code 0 (min 70 (count code)))))))

    (println (str "\n=== Summary (" N " candidates) ==="))
    (println (str "  Valid ClojureScript: " n-valid "/" N " (" (int (* 100 (/ n-valid N))) "%)"))
    (println (str "  Correct behavior:    " n-correct "/" N " (" (int (* 100 (/ n-correct N))) "%)"))
    (println (str "  Partial accuracy:    " n-partial "/" N))

    (when (pos? n-correct)
      (println "\n=== Correct candidates ===")
      (doseq [{:keys [i code]} (filter #(= 1 (:accuracy %)) classified)]
        (println (str "\n-- Candidate " i " --"))
        (println code)))))
