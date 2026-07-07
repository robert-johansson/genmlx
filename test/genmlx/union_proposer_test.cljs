;; @tier fast
(ns genmlx.union-proposer-test
  "genmlx-mo69: union-proposer's dedup key must not collapse distinct
   off-grammar LLM candidates. Every :llm-raw candidate carries :spec' =
   current spec (the fallback), so the render-only key made all raw
   candidates in a step share one key — only the first survived, and the
   harvest under-measured loop breadth. The key is now
   (or (:code c) (render (:spec' c))).

   Run: bunx --bun nbb@1.4.208 test/genmlx/union_proposer_test.cljs"
  (:require [genmlx.world.synth :as syn]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def empty-spec {:latents [] :obs []})

;; two DISTINCT off-grammar candidates: same fallback :spec', different :code
(def raw-a {:edit :llm-raw :spec' empty-spec :code "(fn [trace] (let [a 1] {:y a}))"})
(def raw-b {:edit :llm-raw :spec' empty-spec :code "(fn [trace] (let [b 2] {:y b}))"})
;; a duplicate of raw-a offered by a second proposer
(def raw-a2 (assoc raw-a :edit :llm-raw))
;; two render-keyed candidates (no :code) with identical spec' — must dedup
(def grid-1 {:edit :set-prior :spec' empty-spec})
(def grid-2 {:edit :set-prior :spec' empty-spec})

(println "\n-- distinct off-grammar candidates survive --")
(let [up (syn/union-proposer (fn [_ _] [raw-a raw-b]))
      out (up empty-spec {})]
  (assert-true (str "both raw candidates survive (got " (count out) ")")
               (= 2 (count out)))
  (assert-true "both codes present"
               (= #{(:code raw-a) (:code raw-b)} (set (map :code out)))))

(println "\n-- identical candidates still dedup --")
(let [up (syn/union-proposer (fn [_ _] [raw-a]) (fn [_ _] [raw-a2]))
      out (up empty-spec {})]
  (assert-true "identical :code across proposers dedups to one" (= 1 (count out))))

(let [up (syn/union-proposer (fn [_ _] [grid-1]) (fn [_ _] [grid-2]))
      out (up empty-spec {})]
  (assert-true "identical rendered specs (no :code) still dedup to one" (= 1 (count out))))

(println (str "\n== union-proposer: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
