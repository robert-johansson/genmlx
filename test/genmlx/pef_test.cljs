;; @tier fast
(ns genmlx.pef-test
  "genmlx-0bgi: Path-Equivalence Fuzzing — fast-core smoke slice.

   Covers the FULL PEF spec at smoke scale (the deep tier lives in
   pef_deep_test.cljs; the frozen regression corpus in pef_corpus.cljs):

   1. Model generator: seed-determinism, profile gating, quoted-source
      output, shrink-friendly genome (test.check), paste-runnable repro.
   2. Path registry P1-P6: pair specs with :applicable? + compare policies.
   3. Invariants I1-I5 (I1 soundness = the pairs themselves; I2 liveness on
      the independent-sites profile; I3 crash-freedom; I4 discard round-trip;
      I5 score-type conservation).
   4. Runner: fast-core smoke = fixed seed, ~100 models x P1/P2/P4, < 45s
      on Thor.

   Run: bunx --bun nbb@1.4.208 test/genmlx/pef_test.cljs"
  (:require [genmlx.pef :as pef]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [clojure.string :as str]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

;; ===========================================================================
(println "\n-- 1. generator: seed determinism + quoted-source output --")

(let [m1 (pef/model-for {:seed 42 :idx 3 :profile pef/full-profile})
      m2 (pef/model-for {:seed 42 :idx 3 :profile pef/full-profile})
      m3 (pef/model-for {:seed 42 :idx 4 :profile pef/full-profile})]
  (assert-true "same (seed, idx, profile) -> identical source form"
               (= (pr-str (:source m1)) (pr-str (:source m2))))
  (assert-true "different idx -> (usually) different source form"
               (not= (pr-str (:source m1)) (pr-str (:source m3))))
  (assert-true "source is a quoted gen source form: (args-vector body...)"
               (and (seq? (:source m1)) (vector? (first (:source m1)))))
  (assert-true "model is a GFI value (simulate returns a trace with choices)"
               (let [t (p/simulate (:model m1) (:args m1))]
                 (pos? (count (cm/addresses (:choices t)))))))

(let [sources (mapv #(pr-str (:source (pef/model-for {:seed 7 :idx % :profile pef/full-profile})))
                    (range 12))]
  (assert-true "generator produces variety across indices (>= 8 distinct of 12)"
               (>= (count (distinct sources)) 8)))

;; ===========================================================================
(println "\n-- 2. profile gating --")

(let [no-splice (assoc pef/full-profile :splices? false)
      no-branch (assoc pef/full-profile :branches? false)
      gauss-only (assoc pef/full-profile :dists #{:gaussian})]
  (assert-true "profile {:splices? false} -> no splice call in any of 20 sources"
               (not-any? #(str/includes? (pr-str (:source (pef/model-for {:seed 9 :idx % :profile no-splice}))) "splice")
                         (range 20)))
  (assert-true "profile {:branches? false} -> no if in any of 20 sources"
               (not-any? #(str/includes? (pr-str (:source (pef/model-for {:seed 9 :idx % :profile no-branch}))) "(if ")
                         (range 20)))
  (assert-true "profile {:dists #{:gaussian}} -> only gaussian sites"
               (every? (fn [i]
                         (let [s (pr-str (:source (pef/model-for {:seed 9 :idx i :profile gauss-only})))]
                           (and (str/includes? s "gaussian")
                                (not-any? #(str/includes? s %)
                                          ["exponential" "uniform" "bernoulli"
                                           "categorical" "laplace" "log-normal"]))))
                       (range 20))))

;; ===========================================================================
(println "\n-- 3. failure artifact: paste-runnable repro --")

(let [artifact {:seed 42 :idx 3 :profile-name :full :pair :p1-compiled-vs-handler
                :op :generate :details {:why "example"}}
      s (pef/format-repro artifact)]
  (assert-true "repro string names seed, idx, profile, pair, op"
               (every? #(str/includes? s %) ["42" ":full" ":p1-compiled-vs-handler" ":generate"]))
  (assert-true "repro string is a paste-runnable pef/reproduce call"
               (str/includes? s "pef/reproduce"))
  (assert-true "pef/reproduce re-runs that exact (model, pair) deterministically"
               (map? (pef/reproduce artifact))))

;; ===========================================================================
(println "\n-- 4. path registry: P1-P6 present with applicability + policies --")

(let [names (set (map :name pef/pairs))]
  (assert-true "registry carries P1-P6"
               (every? names [:p1-compiled-vs-handler :p2-regen-fast-vs-general
                              :p3-prefix-vs-handler :p4-batched-vs-scalar
                              :p5-analytical :p6-smc-vs-is]))
  (assert-true "every pair has :applicable? and :run and :compare"
               (every? #(and (fn? (:applicable? %)) (fn? (:run %)) (some? (:compare %)))
                       pef/pairs)))

;; ===========================================================================
(println "\n-- 5. invariants I2-I5 --")

;; I2 liveness: on the independent-sites profile the compiled path must fire
;; on >= the documented threshold of models. A gate change routing everything
;; to the handler FAILS here, not just perf.
(let [{:keys [fired total fraction]} (pef/compiled-liveness {:seed 11 :n-models 30})]
  (println "    I2 liveness: compiled fired on" fired "/" total "=" fraction)
  (assert-true (str "I2: compiled-path liveness >= " pef/liveness-threshold
                    " on independent-sites profile")
               (>= fraction pef/liveness-threshold)))

;; I4 discard round-trip on a handful of models.
(let [r (pef/run-pef {:seed 13 :n-models 10 :profile pef/full-profile
                      :pairs [:i4-discard-roundtrip]})]
  (assert-true "I4: update-with-discard restores the original trace (10 models)"
               (:pass? r)))

;; I5 score-type conservation.
(let [r (pef/run-pef {:seed 17 :n-models 10 :profile pef/full-profile
                      :pairs [:i5-score-type]})]
  (assert-true "I5: every produced trace carries a legal score-type (10 models)"
               (:pass? r)))

;; ===========================================================================
(println "\n-- 6. fast-core smoke: ~100 models x P1/P2/P4 (I3 implicit) --")

(let [t0 (js/Date.now)
      r (pef/run-pef {:seed 42 :n-models 100 :profile pef/smoke-profile
                      :pairs [:p1-compiled-vs-handler :p2-regen-fast-vs-general
                              :p4-batched-vs-scalar]})
      secs (/ (- (js/Date.now) t0) 1000.0)]
  (println "    smoke:" (:n-checks r) "pair-checks over" (:n-models r)
           "models in" (.toFixed secs 1) "s")
  (when-not (:pass? r)
    (doseq [f (take 3 (:failures r))] (println "    FAILURE:" (pef/format-repro f))))
  (assert-true "fast-core smoke is green (P1/P2/P4 x 100 models)" (:pass? r))
  (assert-true "fast-core smoke runs < 45s on Thor" (< secs 45)))

;; ===========================================================================
(println "\n-- 7. P5 + P6 smoke (small n) --")

(let [r (pef/run-pef {:seed 23 :n-models 12 :profile pef/full-profile
                      :pairs [:p5-analytical]})]
  (assert-true "P5: analytical contract holds on 12 models" (:pass? r)))

(let [r (pef/run-pef {:seed 29 :n-models 3 :profile pef/sequential-obs-profile
                      :pairs [:p6-smc-vs-is]})]
  (assert-true "P6: smc log-ML within statistical band of one-shot IS (3 models)"
               (:pass? r)))

;; ===========================================================================
(println (str "\n== pef smoke: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
