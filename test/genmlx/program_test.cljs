(ns genmlx.program-test
  "Deterministic tests for genmlx.program — two-level GFI infrastructure.

   Tests every public function that doesn't require an LLM or randomness.
   Run: bun run --bun nbb test/genmlx/program_test.cljs"
  (:require [genmlx.program :as prog]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [clojure.string :as str]))

;; ============================================================
;; Test harness
;; ============================================================

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [msg v]
  (if v
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn- assert-equal [msg expected actual]
  (assert-true (str msg " (expected " (pr-str expected) ", got " (pr-str actual) ")")
               (= expected actual)))

(defn- assert-close [msg expected actual tol]
  (assert-true (str msg " (expected ~" expected ", got " actual ", tol=" tol ")")
               (< (js/Math.abs (- expected actual)) tol)))

(defn- report []
  (let [p @pass-count f @fail-count]
    (println (str "\n=== " p "/" (+ p f) " PASS ==="))
    (when (pos? f) (println (str "!!! " f " FAILURES !!!")))))

;; ============================================================
;; 1. generate-synthetic-data
;; ============================================================

(println "\n== generate-synthetic-data ==")

(let [data (prog/generate-synthetic-data
             {:n-individuals 3 :n-steps 5})]
  (assert-equal "returns vector" true (vector? data))
  (assert-equal "n-individuals" 3 (count data))
  (assert-equal "n-steps per individual" 5 (count (first data)))
  (assert-true "each point has :x" (every? :x (first data)))
  (assert-true "each point has :y" (every? :y (first data)))
  (assert-true ":x is a number" (number? (:x (ffirst data))))
  (assert-true ":y is a number" (number? (:y (ffirst data)))))

(let [data (prog/generate-synthetic-data
             {:n-individuals 1 :n-steps 2})]
  (assert-equal "minimal: 1 individual" 1 (count data))
  (assert-equal "minimal: 2 steps" 2 (count (first data))))

;; ============================================================
;; 2. extract-transitions
;; ============================================================

(println "\n== extract-transitions ==")

(let [data [[{:x 1 :y 10} {:x 2 :y 20} {:x 3 :y 30}]
            [{:x 4 :y 40} {:x 5 :y 50}]]
      transitions (prog/extract-transitions data)]
  (assert-equal "count: 2+1=3 transitions" 3 (count transitions))
  (assert-equal "first transition x-prev" 1 (:x-prev (first transitions)))
  (assert-equal "first transition y-prev" 10 (:y-prev (first transitions)))
  (assert-equal "first transition x-next" 2 (:x-next (first transitions)))
  (assert-equal "first transition y-next" 20 (:y-next (first transitions)))
  (assert-equal "last transition x-prev" 4 (:x-prev (last transitions)))
  (assert-equal "last transition x-next" 5 (:x-next (last transitions))))

(let [empty-transitions (prog/extract-transitions [[{:x 1 :y 2}]])]
  (assert-equal "single point: 0 transitions" 0 (count empty-transitions)))

;; ============================================================
;; 3. build-transition-source
;; ============================================================

(println "\n== build-transition-source ==")

;; Neither: no cross-effects
(let [source (prog/build-transition-source [:x :y] {["x" "y"] false ["y" "x"] false})]
  (assert-true "is a string" (string? source))
  (assert-true "starts with (fn" (str/starts-with? source "(fn"))
  (assert-true "contains ar-x" (str/includes? source "ar-x"))
  (assert-true "contains ar-y" (str/includes? source "ar-y"))
  (assert-true "contains sigma-x" (str/includes? source "sigma-x"))
  (assert-true "no beta in neither model" (not (str/includes? source "beta-")))
  (assert-true "contains doseq" (str/includes? source "doseq"))
  (assert-true "contains dist/gaussian" (str/includes? source "dist/gaussian")))

;; X->Y only
(let [source (prog/build-transition-source [:x :y] {["x" "y"] true ["y" "x"] false})]
  (assert-true "has beta-x->y" (str/includes? source "beta-x->y"))
  (assert-true "no beta-y->x" (not (str/includes? source "beta-y->x"))))

;; Y->X only
(let [source (prog/build-transition-source [:x :y] {["x" "y"] false ["y" "x"] true})]
  (assert-true "has beta-y->x" (str/includes? source "beta-y->x"))
  (assert-true "no beta-x->y" (not (str/includes? source "beta-x->y"))))

;; Both
(let [source (prog/build-transition-source [:x :y] {["x" "y"] true ["y" "x"] true})]
  (assert-true "has beta-x->y" (str/includes? source "beta-x->y"))
  (assert-true "has beta-y->x" (str/includes? source "beta-y->x")))

;; ============================================================
;; 4. compile-model
;; ============================================================

(println "\n== compile-model ==")

;; All 4 structures should compile
(doseq [s (prog/enumerate-2var-structures :x :y)]
  (let [source (prog/build-transition-source [:x :y] (:edges s))
        gf (prog/compile-model source)]
    (assert-true (str (:name s) " compiles") (some? gf))
    (assert-true (str (:name s) " is DynamicGF")
                 (instance? dyn/DynamicGF gf))))

;; Invalid source returns nil
(assert-equal "invalid source returns nil" nil
              (prog/compile-model "(this is not valid"))
(assert-equal "non-fn source returns nil" nil
              (prog/compile-model "42"))
(assert-equal "empty string returns nil" nil
              (prog/compile-model ""))

;; ============================================================
;; 5. Compiled models produce valid traces
;; ============================================================

(println "\n== compiled model traces ==")

(let [source (prog/build-transition-source [:x :y] {["x" "y"] true ["y" "x"] false})
      gf (prog/compile-model source)
      transitions [{:x-prev 1.0 :y-prev 2.0 :x-next 1.5 :y-next 2.5}
                   {:x-prev 1.5 :y-prev 2.5 :x-next 1.8 :y-next 2.2}]
      trace (p/simulate gf [transitions])]
  (assert-true "simulate produces trace" (some? trace))
  (assert-true "trace has choices" (some? (:choices trace)))
  (assert-true "score is finite" (js/isFinite (mx/item (:score trace))))
  ;; Should have trace sites: ar-x, ar-y, beta-x->y, sigma-x, sigma-y, x0, y0, x1, y1
  (assert-true "has :ar-x" (cm/has-value? (cm/get-submap (:choices trace) :ar-x)))
  (assert-true "has :ar-y" (cm/has-value? (cm/get-submap (:choices trace) :ar-y)))
  (assert-true "has :beta-x->y" (cm/has-value? (cm/get-submap (:choices trace) :beta-x->y)))
  (assert-true "has :sigma-x" (cm/has-value? (cm/get-submap (:choices trace) :sigma-x)))
  (assert-true "has :sigma-y" (cm/has-value? (cm/get-submap (:choices trace) :sigma-y)))
  (assert-true "has :x0" (cm/has-value? (cm/get-submap (:choices trace) :x0)))
  (assert-true "has :y0" (cm/has-value? (cm/get-submap (:choices trace) :y0)))
  (assert-true "has :x1" (cm/has-value? (cm/get-submap (:choices trace) :x1)))
  (assert-true "has :y1" (cm/has-value? (cm/get-submap (:choices trace) :y1))))

;; p/generate with constraints should work
(let [source (prog/build-transition-source [:x :y] {["x" "y"] false ["y" "x"] false})
      gf (prog/compile-model source)
      transitions [{:x-prev 1.0 :y-prev 2.0 :x-next 1.5 :y-next 2.5}]
      constraints (prog/build-constraints transitions [:x :y])
      {:keys [trace weight]} (p/generate gf [transitions] constraints)]
  (assert-true "generate returns trace" (some? trace))
  (assert-true "weight is finite" (js/isFinite (mx/item weight)))
  (assert-true "weight is negative" (neg? (mx/item weight)))
  ;; Constrained values should match
  (let [x0-val (mx/item (cm/get-value (cm/get-submap (:choices trace) :x0)))]
    (assert-close "x0 constrained to 1.5" 1.5 x0-val 0.001))
  (let [y0-val (mx/item (cm/get-value (cm/get-submap (:choices trace) :y0)))]
    (assert-close "y0 constrained to 2.5" 2.5 y0-val 0.001)))

;; ============================================================
;; 6. build-constraints
;; ============================================================

(println "\n== build-constraints ==")

(let [transitions [{:x-prev 0 :y-prev 0 :x-next 1.5 :y-next 2.5}
                   {:x-prev 0 :y-prev 0 :x-next 3.0 :y-next 4.0}]
      cm (prog/build-constraints transitions [:x :y])]
  (assert-true "is a choicemap" (some? cm))
  (assert-true "has :x0" (cm/has-value? (cm/get-submap cm :x0)))
  (assert-true "has :y0" (cm/has-value? (cm/get-submap cm :y0)))
  (assert-true "has :x1" (cm/has-value? (cm/get-submap cm :x1)))
  (assert-true "has :y1" (cm/has-value? (cm/get-submap cm :y1)))
  (assert-close ":x0 = 1.5" 1.5 (mx/item (cm/get-value (cm/get-submap cm :x0))) 0.001)
  (assert-close ":y1 = 4.0" 4.0 (mx/item (cm/get-value (cm/get-submap cm :y1))) 0.001))

;; ============================================================
;; 7. log-mean-exp
;; ============================================================

(println "\n== log-mean-exp ==")

(assert-close "log-mean-exp [0] = -log(1) = 0"
              0.0 (prog/log-mean-exp [0]) 0.001)

(assert-close "log-mean-exp [0 0] = -log(2) + log(2) = 0"
              0.0 (prog/log-mean-exp [0 0]) 0.001)

(assert-close "log-mean-exp [-1 -1] = -1 - log(1) = -1"
              -1.0 (prog/log-mean-exp [-1 -1]) 0.001)

;; log-mean-exp([a, a]) = a for any a
(assert-close "log-mean-exp [-100 -100] = -100"
              -100.0 (prog/log-mean-exp [-100 -100]) 0.001)

;; log-mean-exp([0, -inf]) = log(0.5) = -0.693
(assert-close "log-mean-exp [0 -Inf] = log(0.5)"
              (js/Math.log 0.5) (prog/log-mean-exp [0 ##-Inf]) 0.001)

;; All -Inf
(assert-equal "log-mean-exp all -Inf" ##-Inf
              (prog/log-mean-exp [##-Inf ##-Inf]))

;; Known values: log-mean-exp([0, 1]) = log((1 + e)/2) = log(1.8591) = 0.6201
(assert-close "log-mean-exp [0 1]"
              0.6201 (prog/log-mean-exp [0 1]) 0.01)

;; ============================================================
;; 8. enumerate-2var-structures
;; ============================================================

(println "\n== enumerate-2var-structures ==")

(let [structs (prog/enumerate-2var-structures :x :y)]
  (assert-equal "4 structures" 4 (count structs))
  (assert-equal "first is x->y" "x->y" (:name (nth structs 0)))
  (assert-equal "second is y->x" "y->x" (:name (nth structs 1)))
  (assert-equal "third is both" "both" (:name (nth structs 2)))
  (assert-equal "fourth is neither" "neither" (:name (nth structs 3)))
  ;; Edge structure
  (assert-equal "x->y has [x y]=true" true (get-in (nth structs 0) [:edges ["x" "y"]]))
  (assert-equal "x->y has [y x]=false" false (get-in (nth structs 0) [:edges ["y" "x"]]))
  (assert-equal "both has [x y]=true" true (get-in (nth structs 2) [:edges ["x" "y"]]))
  (assert-equal "both has [y x]=true" true (get-in (nth structs 2) [:edges ["y" "x"]]))
  ;; Each has description
  (assert-true "all have descriptions" (every? :description structs)))

;; Custom variable names
(let [structs (prog/enumerate-2var-structures :a :b)]
  (assert-equal "custom: first is a->b" "a->b" (:name (first structs))))

;; ============================================================
;; 9. compute-posterior
;; ============================================================

(println "\n== compute-posterior ==")

;; Equal scores → uniform posterior
(let [result (prog/compute-posterior [{:name "a" :log-ml -10}
                                      {:name "b" :log-ml -10}])]
  (assert-close "equal scores: a=0.5" 0.5 (:posterior (first result)) 0.001)
  (assert-close "equal scores: b=0.5" 0.5 (:posterior (second result)) 0.001))

;; Dominant score
(let [result (prog/compute-posterior [{:name "a" :log-ml 0}
                                      {:name "b" :log-ml -100}])]
  (assert-close "dominant: a~1.0" 1.0 (:posterior (first result)) 0.001)
  (assert-close "dominant: b~0.0" 0.0 (:posterior (second result)) 0.001))

;; Posteriors sum to 1
(let [result (prog/compute-posterior [{:name "a" :log-ml -5}
                                      {:name "b" :log-ml -7}
                                      {:name "c" :log-ml -6}])
      total (reduce + (map :posterior result))]
  (assert-close "posteriors sum to 1" 1.0 total 0.001))

;; Preserves other keys
(let [result (prog/compute-posterior [{:name "a" :log-ml -5 :extra 42}])]
  (assert-equal "preserves :extra" 42 (:extra (first result))))

;; ============================================================
;; 10. build-scaffold
;; ============================================================

(println "\n== build-scaffold ==")

(let [scaffold (prog/build-scaffold [:x :y])]
  (assert-true "is a string" (string? scaffold))
  (assert-true "starts with (fn" (str/starts-with? scaffold "(fn"))
  (assert-true "has 2 holes" (= 2 (count (re-seq #"<<<HOLE>>>" scaffold))))
  (assert-true "has ar-x binding" (str/includes? scaffold "ar-x"))
  (assert-true "has ar-y binding" (str/includes? scaffold "ar-y"))
  (assert-true "has beta-x->y binding" (str/includes? scaffold "beta-x->y"))
  (assert-true "has beta-y->x binding" (str/includes? scaffold "beta-y->x"))
  (assert-true "has sigma-x binding" (str/includes? scaffold "sigma-x"))
  (assert-true "has sigma-y binding" (str/includes? scaffold "sigma-y"))
  (assert-true "has hint comment" (str/includes? scaffold "e.g.")))

;; ============================================================
;; 11. scaffold-holes
;; ============================================================

(println "\n== scaffold-holes ==")

(let [scaffold (prog/build-scaffold [:x :y])
      holes (prog/scaffold-holes scaffold)]
  (assert-equal "finds 2 holes" 2 (count holes))
  (assert-equal "hole 0 index" 0 (:index (first holes)))
  (assert-equal "hole 1 index" 1 (:index (second holes)))
  (assert-true "prefix is non-empty" (pos? (count (:prefix (first holes)))))
  (assert-true "suffix is non-empty" (pos? (count (:suffix (first holes)))))
  ;; prefix + "<<<HOLE>>>" + suffix of first hole = full scaffold
  (let [h (first holes)]
    (assert-equal "prefix+hole+suffix = scaffold"
                  scaffold
                  (str (:prefix h) "<<<HOLE>>>" (:suffix h)))))

;; No holes
(let [holes (prog/scaffold-holes "no holes here")]
  (assert-equal "no holes found" 0 (count holes)))

;; ============================================================
;; 12. fill-scaffold
;; ============================================================

(println "\n== fill-scaffold ==")

(let [scaffold (prog/build-scaffold [:x :y])
      filled (prog/fill-scaffold scaffold ["EXPR_X" "EXPR_Y"])]
  (assert-true "no holes remain" (not (str/includes? filled "<<<HOLE>>>")))
  (assert-true "EXPR_X inserted" (str/includes? filled "EXPR_X"))
  (assert-true "EXPR_Y inserted" (str/includes? filled "EXPR_Y")))

;; Fill with real expressions and compile
(let [scaffold (prog/build-scaffold [:x :y])
      filled (prog/fill-scaffold scaffold
               ["(mx/multiply ar-x x-prev)"
                "(mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-x->y x-prev))"])
      gf (prog/compile-model filled)]
  (assert-true "filled scaffold compiles" (some? gf))
  (when gf
    (let [transitions [{:x-prev 1.0 :y-prev 2.0 :x-next 1.5 :y-next 2.5}]
          trace (p/simulate gf [transitions])]
      (assert-true "filled model simulates" (some? trace))
      (assert-true "filled model score finite" (js/isFinite (mx/item (:score trace)))))))

;; ============================================================
;; 13. fim-prompt
;; ============================================================

(println "\n== fim-prompt ==")

(let [result (prog/fim-prompt "PREFIX" "SUFFIX")]
  (assert-equal "fim format"
                "<|fim_prefix|>PREFIX<|fim_suffix|>SUFFIX<|fim_middle|>"
                result))

(let [result (prog/fim-prompt "" "")]
  (assert-equal "empty fim"
                "<|fim_prefix|><|fim_suffix|><|fim_middle|>"
                result))

;; ============================================================
;; 14. score-model (deterministic check: finite output)
;; ============================================================

(println "\n== score-model (basic) ==")

(let [source (prog/build-transition-source [:x :y] {["x" "y"] false ["y" "x"] false})
      gf (prog/compile-model source)
      transitions [{:x-prev 1.0 :y-prev 2.0 :x-next 1.5 :y-next 2.5}
                   {:x-prev 1.5 :y-prev 2.5 :x-next 1.8 :y-next 2.2}]
      lml (prog/score-model gf transitions [:x :y] {:n-particles 10})]
  (assert-true "log-ML is a number" (number? lml))
  (assert-true "log-ML is finite" (js/isFinite lml))
  (assert-true "log-ML is negative" (neg? lml)))

;; ============================================================
;; 15. score-model-analytical: basic properties
;; ============================================================

(println "\n== score-model-analytical (basic) ==")

(let [transitions [{:x-prev 1.0 :y-prev 2.0 :x-next 1.5 :y-next 2.5}
                   {:x-prev 1.5 :y-prev 2.5 :x-next 1.8 :y-next 2.2}
                   {:x-prev 1.8 :y-prev 2.2 :x-next 2.0 :y-next 2.0}]
      xy-edges  {["x" "y"] true  ["y" "x"] false}
      no-edges  {["x" "y"] false ["y" "x"] false}]
  ;; Returns a finite negative number
  (let [lml (prog/score-model-analytical transitions [:x :y] xy-edges {})]
    (assert-true "analytical log-ML is a number" (number? lml))
    (assert-true "analytical log-ML is finite" (js/isFinite lml))
    (assert-true "analytical log-ML is negative" (neg? lml)))
  ;; Deterministic: same inputs → same output
  (let [lml1 (prog/score-model-analytical transitions [:x :y] xy-edges {})
        lml2 (prog/score-model-analytical transitions [:x :y] xy-edges {})]
    (assert-true "analytical scoring is deterministic" (= lml1 lml2)))
  ;; Known sigma variant works
  (let [lml (prog/score-model-analytical transitions [:x :y] xy-edges
              {:sigma-x 1.0 :sigma-y 2.0})]
    (assert-true "known-sigma log-ML is finite" (js/isFinite lml)))
  ;; Different structures give different scores
  (let [lml-xy (prog/score-model-analytical transitions [:x :y] xy-edges {})
        lml-no (prog/score-model-analytical transitions [:x :y] no-edges {})]
    (assert-true "different structures give different scores" (not= lml-xy lml-no))))

;; ============================================================
;; 16. score-model-analytical: structure recovery with synthetic data
;; ============================================================

(println "\n== score-model-analytical (structure recovery) ==")

(let [var-names [:x :y]
      xy-edges  {["x" "y"] true  ["y" "x"] false}
      yx-edges  {["x" "y"] false ["y" "x"] true}
      both-edges {["x" "y"] true  ["y" "x"] true}
      no-edges  {["x" "y"] false ["y" "x"] false}
      all-edges [xy-edges yx-edges both-edges no-edges]
      names ["x->y" "y->x" "both" "neither"]
      score-all (fn [trans]
                  (mapv #(prog/score-model-analytical trans var-names % {}) all-edges))
      best-idx (fn [scores] (first (apply max-key second (map-indexed vector scores))))]

  ;; X->Y data: beta-xy = -0.3
  (let [data (prog/generate-synthetic-data {:beta-xy -0.3 :beta-yx 0
                                             :n-individuals 50 :n-steps 10})
        trans (prog/extract-transitions data)
        scores (score-all trans)]
    (assert-true "X->Y: correct structure ranked first"
                 (= 0 (best-idx scores)))
    (println (str "    scores: " (mapv #(.toFixed % 1) scores))))

  ;; Y->X data: beta-yx = 0.5
  (let [data (prog/generate-synthetic-data {:beta-xy 0 :beta-yx 0.5
                                             :n-individuals 50 :n-steps 10})
        trans (prog/extract-transitions data)
        scores (score-all trans)]
    (assert-true "Y->X: correct structure ranked first"
                 (= 1 (best-idx scores)))
    (println (str "    scores: " (mapv #(.toFixed % 1) scores))))

  ;; Both directions: beta-xy = -0.3, beta-yx = 0.5
  (let [data (prog/generate-synthetic-data {:beta-xy -0.3 :beta-yx 0.5
                                             :n-individuals 50 :n-steps 10})
        trans (prog/extract-transitions data)
        scores (score-all trans)]
    (assert-true "Both: correct structure ranked first"
                 (= 2 (best-idx scores)))
    (println (str "    scores: " (mapv #(.toFixed % 1) scores))))

  ;; Neither direction: independent AR(1) — use more data for reliable null detection
  (let [data (prog/generate-synthetic-data {:beta-xy 0 :beta-yx 0
                                             :n-individuals 100 :n-steps 10})
        trans (prog/extract-transitions data)
        scores (score-all trans)]
    (assert-true "Neither: correct structure ranked first"
                 (= 3 (best-idx scores)))
    (println (str "    scores: " (mapv #(.toFixed % 1) scores)))))

;; ============================================================
;; 17. score-model-analytical: known vs estimated sigma
;; ============================================================

(println "\n== score-model-analytical (known vs estimated sigma) ==")

(let [data (prog/generate-synthetic-data {:beta-xy -0.3 :beta-yx 0
                                           :sigma-x 1.0 :sigma-y 2.0
                                           :n-individuals 50 :n-steps 10})
      trans (prog/extract-transitions data)
      edges {["x" "y"] true ["y" "x"] false}
      lml-known (prog/score-model-analytical trans [:x :y] edges
                  {:sigma-x 1.0 :sigma-y 2.0})
      lml-est (prog/score-model-analytical trans [:x :y] edges {})]
  (assert-true "known-sigma score is finite" (js/isFinite lml-known))
  (assert-true "estimated-sigma score is finite" (js/isFinite lml-est))
  ;; Both should be similar (MLE ≈ true sigma with enough data)
  (assert-true "known vs estimated within 50 nats"
               (< (js/Math.abs (- lml-known lml-est)) 50))
  (println (str "    known: " (.toFixed lml-known 2)
                " estimated: " (.toFixed lml-est 2))))

;; ============================================================
;; 18. score-model-analytical: repeated runs always agree on winner
;; ============================================================

(println "\n== score-model-analytical (stability over 10 datasets) ==")

(let [var-names [:x :y]
      xy-edges  {["x" "y"] true  ["y" "x"] false}
      yx-edges  {["x" "y"] false ["y" "x"] true}
      both-edges {["x" "y"] true  ["y" "x"] true}
      no-edges  {["x" "y"] false ["y" "x"] false}
      all-edges [xy-edges yx-edges both-edges no-edges]
      best-idx (fn [scores] (first (apply max-key second (map-indexed vector scores))))
      correct-count
      (reduce
        (fn [n _]
          (let [data (prog/generate-synthetic-data {:beta-xy -0.3 :beta-yx 0
                                                     :n-individuals 50 :n-steps 10})
                trans (prog/extract-transitions data)
                scores (mapv #(prog/score-model-analytical trans var-names % {}) all-edges)]
            (if (= 0 (best-idx scores)) (inc n) n)))
        0
        (range 10))]
  (assert-true (str "X->Y correct in " correct-count "/10 runs")
               (= correct-count 10)))

;; ============================================================
;; 19. compare-structures with analytical scoring
;; ============================================================

(println "\n== compare-structures (analytical) ==")

(let [data (prog/generate-synthetic-data {:beta-xy -0.3 :beta-yx 0
                                           :n-individuals 50 :n-steps 10})
      trans (prog/extract-transitions data)
      results (prog/compare-structures [:x :y] trans)]
  (assert-true "compare-structures returns 4 results" (= 4 (count results)))
  (assert-true "results sorted by posterior" (>= (:posterior (first results))
                                                  (:posterior (second results))))
  (assert-true "best structure is x->y" (= "x->y" (:name (first results))))
  (assert-true "posteriors sum to ~1.0"
               (< (js/Math.abs (- 1.0 (reduce + (map :posterior results)))) 0.01))
  (assert-true "all have :log-ml" (every? :log-ml results))
  (assert-true "all have :source" (every? :source results))
  (println (str "    best: " (:name (first results))
                " P=" (.toFixed (:posterior (first results)) 4))))

;; ============================================================
;; 20. K-variable data generation and transition extraction
;; ============================================================

(println "\n== K-variable infrastructure ==")

(let [data (prog/generate-kvar-data [:a :b :c]
             {:cross {[:a :b] 0.5}
              :n-individuals 10 :n-steps 5})]
  (assert-true "generates correct number of individuals" (= 10 (count data)))
  (assert-true "correct steps per individual" (= 5 (count (first data))))
  (assert-true "each step has all variables" (every? #(and (contains? % :a)
                                                           (contains? % :b)
                                                           (contains? % :c))
                                                     (first data))))

(let [data (prog/generate-kvar-data [:a :b] {:n-individuals 3 :n-steps 4})
      trans (prog/extract-kvar-transitions data)]
  (assert-true "correct transition count" (= (* 3 3) (count trans)))
  (assert-true "transitions have :prev and :next" (and (:prev (first trans))
                                                        (:next (first trans)))))

;; ============================================================
;; 21. enumerate-all-structures
;; ============================================================

(println "\n== enumerate-all-structures ==")

(let [s2 (prog/enumerate-all-structures [:a :b])
      s3 (prog/enumerate-all-structures [:a :b :c])]
  (assert-equal "2 vars: 4 structures" 4 (count s2))
  (assert-equal "3 vars: 64 structures" 64 (count s3))
  (assert-true "includes independent" (some #(= "independent" (:name %)) s2))
  (assert-true "all have :edges" (every? :edges s3)))

;; ============================================================
;; 22. 3-variable structure discovery
;; ============================================================

(println "\n== 3-variable structure discovery ==")

(let [var-names [:sleep :exercise :mood]
      data (prog/generate-kvar-data var-names
             {:ar {:sleep 0.7 :exercise 0.3 :mood 0.5}
              :cross {[:exercise :mood] 0.5 [:sleep :mood] -0.4}
              :sigma {:sleep 1.0 :exercise 0.5 :mood 1.0}
              :n-individuals 60 :n-steps 10})
      trans (prog/extract-kvar-transitions data)
      result (prog/discover-structure var-names trans)
      best (:best result)
      marginals (:marginals result)]
  (assert-true "best structure contains exercise->mood"
               (contains? (:edges best) [:exercise :mood]))
  (assert-true "best structure contains sleep->mood"
               (contains? (:edges best) [:sleep :mood]))
  (assert-true "P(exercise->mood) > 0.9"
               (> (get marginals [:exercise :mood]) 0.9))
  (assert-true "P(sleep->mood) > 0.9"
               (> (get marginals [:sleep :mood]) 0.9))
  (assert-true "P(mood->exercise) < 0.1"
               (< (get marginals [:mood :exercise]) 0.1))
  (assert-true "64 structures scored"
               (= 64 (count (:ranked result))))
  (println (str "    best: " (:name best) " P=" (.toFixed (:posterior best) 4)
                " in " (:elapsed-ms result) "ms")))

;; ============================================================
;; 23. edge-marginals sum correctly
;; ============================================================

(println "\n== edge-marginals ==")

(let [var-names [:a :b :c]
      data (prog/generate-kvar-data var-names
             {:cross {[:a :b] 0.5} :n-individuals 30 :n-steps 10})
      trans (prog/extract-kvar-transitions data)
      scored (prog/score-all-structures trans var-names)
      marginals (prog/edge-marginals var-names scored)]
  (assert-equal "6 edges for 3 variables" 6 (count marginals))
  (assert-true "all marginals in [0,1+eps]"
               (every? #(and (>= % -0.001) (<= % 1.001)) (vals marginals)))
  (assert-true "P(a->b) is highest"
               (> (get marginals [:a :b])
                  (apply max (vals (dissoc marginals [:a :b]))))))

;; ============================================================
;; Summary
;; ============================================================

(mx/force-gc!)
(report)
