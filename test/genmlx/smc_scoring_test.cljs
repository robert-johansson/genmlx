;; @tier medium
(ns genmlx.smc-scoring-test
  "Tests for multi-particle IS in msa.cljs and scoring infrastructure in program.cljs.
   Run: bun run --bun nbb test/genmlx/smc_scoring_test.cljs"
  (:require [genmlx.program :as prog]
            [genmlx.llm.msa :as msa]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

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
;; 1. Sequential constraint construction
;; ============================================================

(println "\n== build-sequential-constraints ==")

(let [transitions [{:x-prev 1 :y-prev 2 :x-next 3 :y-next 4}
                   {:x-prev 3 :y-prev 4 :x-next 5 :y-next 6}]
      obs-seq (prog/build-sequential-constraints transitions [:x :y])]
  (assert-equal "returns 2 choicemaps" 2 (count obs-seq))
  ;; Step 0: only :x0 :y0
  (assert-true "step 0 has :x0"
               (cm/has-value? (cm/get-submap (first obs-seq) :x0)))
  (assert-true "step 0 has :y0"
               (cm/has-value? (cm/get-submap (first obs-seq) :y0)))
  (assert-close ":x0 = 3" 3.0
                (mx/item (cm/get-value (cm/get-submap (first obs-seq) :x0))) 0.001)
  (assert-close ":y0 = 4" 4.0
                (mx/item (cm/get-value (cm/get-submap (first obs-seq) :y0))) 0.001)
  ;; Step 1: cumulative — has :x0, :y0, :x1, :y1
  (assert-true "step 1 has :x0 (cumulative)"
               (cm/has-value? (cm/get-submap (second obs-seq) :x0)))
  (assert-true "step 1 has :x1"
               (cm/has-value? (cm/get-submap (second obs-seq) :x1)))
  (assert-true "step 1 has :y1"
               (cm/has-value? (cm/get-submap (second obs-seq) :y1)))
  (assert-close ":x1 = 5" 5.0
                (mx/item (cm/get-value (cm/get-submap (second obs-seq) :x1))) 0.001)
  (assert-close ":y1 = 6" 6.0
                (mx/item (cm/get-value (cm/get-submap (second obs-seq) :y1))) 0.001))

(let [transitions [{:x-prev 0 :y-prev 0 :x-next 1 :y-next 2}]
      obs-seq (prog/build-sequential-constraints transitions [:x :y])]
  (assert-equal "single transition -> 1 choicemap" 1 (count obs-seq)))

(let [transitions []
      obs-seq (prog/build-sequential-constraints transitions [:x :y])]
  (assert-equal "empty transitions -> empty seq" 0 (count obs-seq)))

;; ============================================================
;; 2. Param selection construction
;; ============================================================

(println "\n== param-selection ==")

(let [sel-xy (prog/param-selection [:x :y] {["x" "y"] true ["y" "x"] false})]
  (assert-true ":ar-x selected" (sel/selected? sel-xy :ar-x))
  (assert-true ":ar-y selected" (sel/selected? sel-xy :ar-y))
  (assert-true ":sigma-x selected" (sel/selected? sel-xy :sigma-x))
  (assert-true ":sigma-y selected" (sel/selected? sel-xy :sigma-y))
  (assert-true ":beta-x->y selected" (sel/selected? sel-xy :beta-x->y))
  (assert-true ":x0 not selected" (not (sel/selected? sel-xy :x0)))
  (assert-true ":y0 not selected" (not (sel/selected? sel-xy :y0))))

(let [sel-no (prog/param-selection [:x :y] {["x" "y"] false ["y" "x"] false})]
  (assert-true "no-edge: :ar-x selected" (sel/selected? sel-no :ar-x))
  (assert-true "no-edge: :sigma-y selected" (sel/selected? sel-no :sigma-y))
  (assert-true "no-edge: :beta-x->y NOT selected" (not (sel/selected? sel-no :beta-x->y)))
  (assert-true "no-edge: :beta-y->x NOT selected" (not (sel/selected? sel-no :beta-y->x))))

(let [sel-both (prog/param-selection [:x :y] {["x" "y"] true ["y" "x"] true})]
  (assert-true "both: :beta-x->y selected" (sel/selected? sel-both :beta-x->y))
  (assert-true "both: :beta-y->x selected" (sel/selected? sel-both :beta-y->x)))

;; ============================================================
;; 3. IS scoring produces finite results
;; ============================================================

(println "\n== score-model IS (basic) ==")

(let [data (prog/generate-synthetic-data {:beta-xy 0.5 :beta-yx 0
                                           :n-individuals 5 :n-steps 4})
      transitions (prog/extract-transitions data)
      source (prog/build-transition-source [:x :y] {["x" "y"] true ["y" "x"] false})
      gf (prog/compile-model source)]
  (assert-true "model compiles" (some? gf))
  (when gf
    (let [lml (prog/score-model gf transitions [:x :y] {:n-particles 50})]
      (assert-true "IS log-ML is a number" (number? lml))
      (assert-true "IS log-ML is finite" (js/isFinite lml))
      (assert-true "IS log-ML is negative" (neg? lml))
      (println (str "    IS log-ML: " (.toFixed lml 2))))))

(mx/force-gc!)

;; ============================================================
;; 4. IS vs analytical convergence
;; ============================================================

(println "\n== IS vs analytical convergence ==")

(let [xy-edges {["x" "y"] true ["y" "x"] false}
      data (prog/generate-synthetic-data {:beta-xy 0.5 :beta-yx 0
                                           :n-individuals 20 :n-steps 6})
      transitions (prog/extract-transitions data)
      lml-analytical (prog/score-model-analytical transitions [:x :y] xy-edges {})
      source (prog/build-transition-source [:x :y] xy-edges)
      gf (prog/compile-model source)]
  (assert-true "model compiles" (some? gf))
  (when gf
    (let [lml-is (prog/score-model gf transitions [:x :y] {:n-particles 200})]
      (println (str "    analytical: " (.toFixed lml-analytical 2)
                    "  IS(200): " (.toFixed lml-is 2)
                    "  diff: " (.toFixed (js/Math.abs (- lml-analytical lml-is)) 2)))
      (assert-true "IS is finite" (js/isFinite lml-is))
      (assert-true "IS is negative" (neg? lml-is)))))

(mx/force-gc!)

;; ============================================================
;; 5. Analytical structure recovery (verifies infrastructure)
;; ============================================================

(println "\n== structure recovery (analytical) ==")

(let [data (prog/generate-synthetic-data {:beta-xy 0.5 :beta-yx 0
                                           :n-individuals 30 :n-steps 8})
      transitions (prog/extract-transitions data)
      result (prog/compare-structures [:x :y] transitions {:scoring :analytical})]
  (assert-true "x->y ranked first"
               (= "x->y" (:name (first result))))
  (println (str "    best=" (:name (first result))
                " P=" (.toFixed (:posterior (first result)) 4))))

(mx/force-gc!)

;; ============================================================
;; 6. Multi-particle IS in msa.cljs
;; ============================================================

(println "\n== msa/score-model multi-particle ==")

(let [model (dyn/auto-key
              (gen [] (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                        (trace :y (dist/gaussian mu (mx/scalar 1))))))
      obs-cm (cm/choicemap :y (mx/scalar 5.0))
      {:keys [weight]} (p/generate model [] obs-cm)]
  (assert-true "direct p/generate weight is finite" (js/isFinite (mx/item weight)))
  (println (str "    direct weight: " (.toFixed (mx/item weight) 4))))

(let [model (dyn/auto-key
              (gen [] (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                        (trace :y (dist/gaussian mu (mx/scalar 1))))))
      score (msa/score-model model {:y 5.0} {:n-particles 50})]
  (assert-true "multi-particle score is finite" (js/isFinite score))
  (assert-true "multi-particle score is negative" (neg? score))
  (println (str "    score: " (if (js/isFinite score) (.toFixed score 4) score))))

;; 2-arg backward compat
(let [model (dyn/auto-key
              (gen [] (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                        (trace :y (dist/gaussian mu (mx/scalar 1))))))
      score (msa/score-model model {:y 5.0})]
  (assert-true "2-arg backward compat is finite" (js/isFinite score))
  (assert-true "2-arg backward compat is negative" (neg? score)))

;; ============================================================
;; 7. Multi-particle stability
;; ============================================================

(println "\n== msa/score-model stability ==")

(let [model (dyn/auto-key
              (gen [] (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
                        (trace :y (dist/gaussian mu (mx/scalar 1))))))
      scores (vec (for [_ (range 10)]
                    (msa/score-model model {:y 5.0} {:n-particles 50})))
      finite-scores (filterv js/isFinite scores)]
  (assert-true "at least 8/10 scores are finite" (>= (count finite-scores) 8))
  (when (>= (count finite-scores) 2)
    (let [mean-score (/ (reduce + finite-scores) (count finite-scores))
          variance (/ (reduce + (map #(* (- % mean-score) (- % mean-score)) finite-scores))
                      (count finite-scores))]
      (assert-true "variance is finite" (js/isFinite variance))
      (assert-true "variance < 5.0" (< variance 5.0))
      (println (str "    mean=" (.toFixed mean-score 4)
                    " var=" (.toFixed variance 4)
                    " (" (count finite-scores) "/10 finite)")))))

(mx/force-gc!)

;; ============================================================
;; Summary
;; ============================================================

(report)
