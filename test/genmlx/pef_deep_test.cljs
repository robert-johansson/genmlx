;; @tier slow
(ns genmlx.pef-deep-test
  "genmlx-0bgi: PEF deep mode — the full sweep.

   PEF_MODELS env sets the model count (default 2000); PEF_SEED the master
   seed (default 20260707). Runs ALL registered pairs (P1-P6 + I4/I5) over
   the full-grammar profile, plus the I2 liveness gate at depth. Prints every
   failure as a paste-runnable repro (the artifact contract).

   Run: PEF_MODELS=2000 bunx --bun nbb@1.4.208 test/genmlx/pef_deep_test.cljs"
  (:require [genmlx.pef :as pef]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def ^:private n-models
  (let [v (.. js/process -env -PEF_MODELS)]
    (if v (js/parseInt v 10) 2000)))

(def ^:private seed
  (let [v (.. js/process -env -PEF_SEED)]
    (if v (js/parseInt v 10) 20260707)))

(println (str "\n== PEF deep mode: " n-models " models, seed " seed " =="))

;; Deep sweep is budgeted by pair cost: the statistical pairs (P5-fired, P6)
;; run hundreds of inference particles per model, so they sweep a slice while
;; the cheap exact pairs (P1-P4, I4, I5) cover every model.
(let [t0 (js/Date.now)
      exact (pef/run-pef {:seed seed :n-models n-models :profile pef/full-profile
                          :pairs [:p1-compiled-vs-handler :p2-regen-fast-vs-general
                                  :p3-prefix-vs-handler :p4-batched-vs-scalar
                                  :i4-discard-roundtrip :i5-score-type]})
      stat-n (max 20 (quot n-models 40))
      stat (pef/run-pef {:seed (inc seed) :n-models stat-n :profile pef/full-profile
                         :pairs [:p5-analytical]})
      p6 (pef/run-pef {:seed (+ seed 2) :n-models (max 5 (quot stat-n 4))
                       :profile pef/sequential-obs-profile
                       :pairs [:p6-smc-vs-is]})
      secs (/ (- (js/Date.now) t0) 1000.0)]
  (println (str "    exact pairs: " (:n-checks exact) " checks / " (:n-models exact) " models"))
  (println (str "    P5: " (:n-checks stat) " checks | P6: " (:n-checks p6) " checks"))
  (println (str "    total " (.toFixed secs 1) "s"))
  (doseq [f (concat (:failures exact) (:failures stat) (:failures p6))]
    (println "  FAILURE:" (pef/format-repro f)))
  (assert-true (str "deep exact sweep green (" n-models " models x P1-P4+I4+I5)")
               (:pass? exact))
  (assert-true (str "deep P5 sweep green (" stat-n " models)") (:pass? stat))
  (assert-true "deep P6 sweep green" (:pass? p6)))

(let [{:keys [fired total fraction]} (pef/compiled-liveness {:seed seed :n-models 100})]
  (println "    I2 liveness at depth:" fired "/" total)
  (assert-true (str "I2 liveness >= " pef/liveness-threshold " over 100 models")
               (>= fraction pef/liveness-threshold)))

(println (str "\n== pef deep: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
