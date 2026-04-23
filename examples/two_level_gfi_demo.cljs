(ns examples.two-level-gfi-demo
  "Phase 1: Two-level GFI — discover causal structure from synthetic data.

   Ground truth: X(t) causes Y(t+1), not the reverse.
   The system should recover this from data alone."
  (:require [genmlx.program :as prog]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [promesa.core :as pr]))

;; ============================================================
;; Step 1: Generate synthetic data
;; ============================================================

(println "\n━━━ Two-Level GFI: Phase 1 ━━━\n")
(println "Generating synthetic data...")
(println "  Ground truth: X→Y with beta=-0.5, Y does NOT cause X\n")

(def data
  (prog/generate-synthetic-data
    {:n-individuals 30
     :n-steps 10
     :ar-x 0.8
     :ar-y 0.5
     :beta-xy -0.5
     :beta-yx 0
     :sigma-x 1.0
     :sigma-y 1.5}))

(def transitions (prog/extract-transitions data))

(println (str "  " (count data) " individuals, "
              (count (first data)) " time points each"))
(println (str "  " (count transitions) " total transitions\n"))

(println "  Sample individual (first 4 time points):")
(doseq [pt (take 4 (first data))]
  (println (str "    x=" (.toFixed (:x pt) 2) "  y=" (.toFixed (:y pt) 2))))

;; ============================================================
;; Step 2: Inspect one generated model source
;; ============================================================

(println "\n━━━ Model source (x->y) ━━━\n")
(def structures (prog/enumerate-2var-structures :x :y))
(println (prog/build-transition-source [:x :y] (:edges (first structures))))

;; ============================================================
;; Step 3: Compilation check
;; ============================================================

(println "\n━━━ Compilation check ━━━\n")
(doseq [s structures]
  (let [source (prog/build-transition-source [:x :y] (:edges s))
        gf (prog/compile-model source)]
    (println (str "  " (:name s) ": " (if gf "OK" "FAILED")))))

;; ============================================================
;; Step 4: Score all structures
;; ============================================================

(println "\n━━━ Scoring (100 particles each) ━━━\n")
(def results (prog/compare-structures [:x :y] transitions {:n-particles 100}))

(println "\n━━━ Posterior ━━━\n")
(doseq [r results]
  (println (str "  " (:name r)
                "  log-ML=" (.toFixed (:log-ml r) 1)
                "  P=" (.toFixed (:posterior r) 4)
                "  (" (:description r) ")")))

(let [best (first results)]
  (println (str "\n  Best: " (:name best) "  Ground truth: x->y  Match: " (= "x->y" (:name best)))))

;; ============================================================
;; Step 5: Reversed ground truth
;; ============================================================

(println "\n━━━ Reversed ground truth (Y→X) ━━━\n")
(def data-rev
  (prog/generate-synthetic-data
    {:n-individuals 30 :n-steps 10
     :ar-x 0.8 :ar-y 0.5 :beta-xy 0 :beta-yx -0.5
     :sigma-x 1.0 :sigma-y 1.5}))
(def results-rev (prog/compare-structures [:x :y] (prog/extract-transitions data-rev) {:n-particles 100}))

(println)
(doseq [r results-rev]
  (println (str "  " (:name r) "  log-ML=" (.toFixed (:log-ml r) 1) "  P=" (.toFixed (:posterior r) 4))))
(let [best (first results-rev)]
  (println (str "\n  Best: " (:name best) "  Ground truth: y->x  Match: " (= "y->x" (:name best)))))

;; ============================================================
;; Step 6: FIM integration — score equations under coding model
;; ============================================================

(println "\n━━━ FIM: scoring transition equations under Qwen2.5-Coder ━━━\n")
(println "Loading model...")

(def home-dir (.-HOME (.-env js/process)))

(defn score-fim-candidate
  "Score a candidate equation under the FIM model.
   Returns a promise of the log-probability."
  [gf tokenizer fim-prompt-str equation-str]
  (pr/let [fim-ids-raw (llm/encode tokenizer fim-prompt-str false)
           fim-ids (vec fim-ids-raw)
           eq-ids-raw (llm/encode tokenizer equation-str false)
           eq-ids (vec eq-ids-raw)
           n-eq (count eq-ids)
           constraints (reduce (fn [m k]
                                 (cm/set-value m (keyword (str "t" k))
                                               (mx/scalar (nth eq-ids k) mx/int32)))
                               (cm/choicemap) (range n-eq))
           {:keys [weight]} (p/generate gf [fim-ids n-eq] constraints)]
    (mx/item weight)))

(pr/let
  [fim-model (llm/load-model (str home-dir "/.cache/models/qwen25-coder-3b-cljs-fused-v2"))
   _ (println "Qwen2.5-Coder-3B loaded.\n")

   gf (llm-core/make-llm-gf fim-model)
   tokenizer (:tokenizer fim-model)

   ;; Build FIM scaffold — fill X as pure AR, leave Y as hole
   scaffold (prog/build-scaffold [:x :y])
   partial (prog/fill-scaffold scaffold ["(mx/multiply ar-x x-prev)"])
   y-holes (prog/scaffold-holes partial)
   y-hole (first y-holes)
   fim-str (prog/fim-prompt (:prefix y-hole) (:suffix y-hole))

   _ (println "FIM context: ...code... (dist/gaussian <<<HOLE>>> sigma-y))")
   _ (println "Scoring 4 candidate mean expressions for Y:\n")

   ;; Score candidates
   s0 (score-fim-candidate gf tokenizer fim-str "(mx/multiply ar-y y-prev)")
   s1 (score-fim-candidate gf tokenizer fim-str "(mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-x->y x-prev))")
   s2 (score-fim-candidate gf tokenizer fim-str "(mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-y->x x-prev))")
   s3 (score-fim-candidate gf tokenizer fim-str "(mx/add (mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-x->y x-prev)) (mx/multiply beta-y->x x-prev))")]

  (let [fim-scores [{:label "AR only" :fim s0}
                    {:label "X→Y effect" :fim s1}
                    {:label "Y→X in Y eq" :fim s2}
                    {:label "both effects" :fim s3}]
        equations ["(mx/multiply ar-y y-prev)"
                   "(mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-x->y x-prev))"
                   "(mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-y->x x-prev))"
                   "(mx/add (mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-x->y x-prev)) (mx/multiply beta-y->x x-prev))"]]

    (println "FIM scores (code model's preference):")
    (doseq [{:keys [label fim]} (sort-by :fim > fim-scores)]
      (println (str "  " (.toFixed fim 2) "  " label)))

    ;; Score each via GFI against data
    (println "\nGFI scores (data likelihood):")
    (let [gfi-scores
          (mapv (fn [eq label]
                  (let [source (prog/fill-scaffold partial [eq])
                        model-gf (prog/compile-model source)]
                    (if model-gf
                      (let [lml (prog/score-model model-gf transitions [:x :y]
                                  {:n-particles 100})]
                        (println (str "  " (.toFixed lml 1) "  " label))
                        {:label label :gfi lml})
                      (do (println (str "  FAILED  " label))
                          {:label label :gfi ##-Inf}))))
                equations (map :label fim-scores))]

      (println "\nCombined (GFI + 0.1×FIM):")
      (let [combined (sort-by :combined >
                       (mapv (fn [g f]
                               (assoc g :fim (:fim f)
                                        :combined (+ (:gfi g) (* 0.1 (:fim f)))))
                             gfi-scores fim-scores))]
        (doseq [{:keys [label gfi fim combined]} combined]
          (println (str "  " (.toFixed combined 1)
                        "  GFI=" (.toFixed gfi 1)
                        "  FIM=" (.toFixed fim 1)
                        "  " label)))
        (println (str "\n  Best: " (:label (first combined))))))

    (println "\n━━━ Phase 1 + FIM complete ━━━")))
