(ns examples.msa-matching-demo
  "Matching-to-sample task via Model Synthesis Architecture.
   A subject sees a sample stimulus and must pick the matching one.
   We observe their response and infer perceptual ability."
  (:require [genmlx.llm.backend :as backend]
            [genmlx.llm.msa :as msa]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]))

(def task
  {:name        "matching-to-sample"
   :description "A subject does a matching-to-sample perceptual task. They have some perceptual ability (higher is better). The task has some difficulty level. Their probability of making a correct match depends on their ability relative to the difficulty."
   :variables   [:ability :difficulty :correct]
   :observations {:correct 1}
   :query       :ability})

(def model-dir
  (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-cljs"))

(pr/let [model-map (backend/load-model model-dir)
         _ (println "\n=== Matching-to-Sample via MSA ===")
         _ (println "Task:" (:description task))
         _ (println "Observed: correct=1 (subject matched correctly)")
         _ (println "Query: what is their perceptual ability?\n")

         _ (println "-- Generating 8 candidate models... --")
         result (msa/msa model-map task {:n 8 :particles 500 :temperature 0.6})

         {:keys [model posterior candidates]} result]

  (println "\n-- Candidates (ranked by log-likelihood) --")
  (doseq [[i c] (map-indexed vector candidates)]
    (println (str "  [" i "] weight=" (.toFixed (:weight c) 2)
                  "  " (pr-str (:dist-map c)))))

  (println "\n-- Best model --")
  (println "  code:" (:code model))
  (println "  weight:" (.toFixed (:weight model) 2))

  (println "\n-- Posterior over :ability --")
  (println "  mean:    " (.toFixed (:mean posterior) 3))
  (println "  variance:" (.toFixed (:variance posterior) 3))
  (println "  ESS:     " (.toFixed (:ess posterior) 1))
  (println "\nDone."))
