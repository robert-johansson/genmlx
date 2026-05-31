(ns examples.vlm-grid-gf
  "VLM-as-generative-function: end-to-end demo.

   Loads a Qwen3.6 VLM, classifies all 25 cells of a pre-rendered 5×5
   hiking-style gridworld, then runs a GenMLX gen-fn over the per-cell
   categorical structure. The VLM's per-cell observations become constraints
   when invoking via `p/generate`; the returned trace records each per-cell
   choice and the score reflects the prior probability of those observations.

   Prereqs (from earlier in the investigation):
     python3 ../genmlx-lab/dev/render_gridworld.py     # writes ../genmlx-lab/dev/gridworld_clean.png
     python3 ../genmlx-lab/dev/crop_gridworld.py       # writes ../genmlx-lab/dev/grid_cells/cell_r{r}_c{c}.png

   Run:
     bun run --bun nbb examples/vlm_grid_gf.cljs"
  (:require [genmlx.llm.vision :as vision]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            ["node:fs/promises" :as fs]
            [promesa.core :as pr]
            [clojure.string :as str]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/Qwen3.6-35B-A3B-4bit"))

(def cells-dir
  (str (.-HOME js/process.env) "/code/genmlx-lab/dev/grid_cells"))

(def cell-types
  [{:label "empty" :description "a blank white cell with thin gray borders"}
   {:label "wall"  :description "a solid dark/black cell"}
   {:label "agent" :description "a blue circle with the letter \"A\""}
   {:label "west"  :description "an orange cell with text \"West\""}
   {:label "hill"  :description "a yellow cell with text \"Hill\""}
   {:label "east"  :description "a green cell with text \"East\""}])

(def truth-grid
  [["empty" "empty" "empty" "empty" "empty"]
   ["empty" "wall"  "empty" "wall"  "empty"]
   ["west"  "empty" "hill"  "empty" "east"]
   ["empty" "wall"  "empty" "wall"  "empty"]
   ["empty" "empty" "agent" "empty" "empty"]])

(defn read-cell [r c]
  (pr/let [buf (fs/readFile (str cells-dir "/cell_r" r "_c" c ".png"))]
    (js/Uint8Array. (.-buffer buf) (.-byteOffset buf) (.-byteLength buf))))

(defn pad [s n]
  (let [s (str s)
        diff (- n (count s))]
    (if (pos? diff) (str s (apply str (repeat diff " "))) s)))

(defn print-grid [title grid]
  (println (str "--- " title " ---"))
  (doseq [row grid]
    (println (str/join " | " (map #(pad % 7) row))))
  (println))

(defn count-mismatches [a b]
  (count (for [r (range (count a))
               c (range (count (first a)))
               :when (not= (get-in a [r c]) (get-in b [r c]))]
           1)))

(defn run! []
  (println "Loading VLM...")
  (pr/let [t-load (.now js/performance)
           session (vision/load-vlm model-path)
           _ (println (str "Loaded in " (.toFixed (- (.now js/performance) t-load) 0) "ms"))
           _ (println "\nClassifying 25 cells (this is the slow async step)...")
           t-cls (.now js/performance)
           {:keys [labels]} (vision/classify-grid
                              session 5 5 read-cell cell-types
                              (fn [r c lbl dt]
                                (println (str "  (" r "," c ") "
                                              (.toFixed dt 0) "ms -> " lbl))))
           cls-ms (- (.now js/performance) t-cls)]
    (println (str "Classified in " (.toFixed cls-ms 0) "ms ("
                  (.toFixed (/ cls-ms 25) 0) "ms/cell avg)\n"))

    (print-grid "VLM observations" labels)

    (let [;; --- The sync GFI layer takes over here ---
          grid-gf (vision/make-grid-gf 5 5 cell-types)

          ;; Build a choicemap of constraints from the VLM observations
          constraint-map (vision/labels->constraints labels cell-types)
          constraints (cm/from-map constraint-map)

          ;; p/generate: the gen-fn samples from the prior, then conditions on
          ;; constraints. Score = log p(observations | uniform prior) summed.
          {:keys [trace weight]} (p/generate grid-gf [] constraints)

          ;; Read the trace's choices back into a label grid
          choices (:choices trace)
          back-idx-grid (vec (for [r (range 5)]
                               (vec (for [c (range 5)]
                                      (cm/get-choice
                                       choices
                                       [(vision/cell-addr r c)])))))
          back-grid (vision/idx-grid->labels back-idx-grid cell-types)

          ;; Expected score under uniform prior:
          ;; for each cell, log(1/K) where K = (count cell-types)
          k (count cell-types)
          expected (* 25 (Math/log (/ 1.0 k)))
          weight-num (try (mx/item weight) (catch :default _ js/NaN))]

      (print-grid "Reconstructed from trace" back-grid)
      (print-grid "Ground truth" truth-grid)

      (println (str "Score (log p(observations | uniform prior)): "
                    (.toFixed weight-num 4)))
      (println (str "Expected: " (.toFixed expected 4)
                    "  (= 25 × log(1/" k "))"))
      (println)
      (println (str "Mismatches vs VLM observations: "
                    (count-mismatches back-grid labels) "/25"))
      (println (str "Mismatches vs ground truth:     "
                    (count-mismatches back-grid truth-grid) "/25"))
      (println)
      (println "What this demonstrates:")
      (println "  - The VLM did the perception (async I/O, ~4.5s/cell).")
      (println "  - The gen-fn did the structural side (sync, instant).")
      (println "  - Each cell is a categorical trace site at :r{R}-c{C}.")
      (println "  - VLM observations entered as a choicemap of constraints.")
      (println "  - Score reflects how likely those observations are under")
      (println "    the gen-fn's prior. With a uniform prior, score is")
      (println "    25 × log(1/K). Replace `make-grid-gf`'s prior-logits to")
      (println "    encode structural knowledge (e.g. low prior on agent")
      (println "    appearing in row 0) and watch score discriminate."))))

(run!)
