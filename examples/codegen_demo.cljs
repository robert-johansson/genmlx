(ns codegen-demo
  "Demo: generate ClojureScript with a fine-tuned LLM."
  (:require [genmlx.llm.codegen :as cg]
            [genmlx.llm.backend :as llm]
            [promesa.core :as pr]))

(def model-dir (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-cljs"))

(pr/let [m (llm/load-model model-dir)]
  (println "Model loaded.\n")

  ;; 1. Generate a function
  (println "== Prompt: Write a function that doubles every element in a vector ==")
  (pr/let [r (cg/generate-cljs m "Write a ClojureScript function called double-all that doubles every element in a vector")]
    (println "Valid?:" (:valid? r))
    (println (:code r))
    (when (:valid? r)
      (println "=> (double-all [1 2 3 4 5])")
      (println "  " (pr-str (:result (cg/eval-cljs (str (:code r) "\n(double-all [1 2 3 4 5])")))))))

  ;; 2. Generate and verify a transition function
  (println "\n== Prompt: Write a grid movement transition function ==")
  (pr/let [r (cg/generate-cljs m "Write a ClojureScript function (fn [{:keys [x y]} action] ...) for grid movement. Actions are :up :down :left :right. Up decrements y, down increments y, left decrements x, right increments x. Return the new {:x :y} map.")]
    (println "Valid?:" (:valid? r))
    (println (:code r))
    (when (:valid? r)
      (let [v (cg/verify-transition-fn (:code r)
                [{:state {:x 5 :y 5} :action :up    :expected {:x 5 :y 4}}
                 {:state {:x 5 :y 5} :action :down  :expected {:x 5 :y 6}}
                 {:state {:x 5 :y 5} :action :left  :expected {:x 4 :y 5}}
                 {:state {:x 5 :y 5} :action :right :expected {:x 6 :y 5}}])]
        (println "Accuracy:" (:accuracy v) (str "(" (:correct v) "/" (:total v) ")")))))

  ;; 3. Generate 3 candidates
  (println "\n== 3 candidates: Write a function that sums a vector of numbers ==")
  (pr/let [results (cg/generate-cljs-n m "Write a ClojureScript function called sum-vec that sums all numbers in a vector" 3)]
    (doseq [[i r] (map-indexed vector results)]
      (println (str "\n  Candidate " i " (valid=" (:valid? r) "):"))
      (println " " (:code r))
      (when (:valid? r)
        (println "  => (sum-vec [10 20 30])")
        (println "    " (pr-str (:result (cg/eval-cljs (str (:code r) "\n(sum-vec [10 20 30])")))))))))
