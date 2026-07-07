;; @tier fast
(ns genmlx.pef-corpus
  "genmlx-0bgi: the PEF regression corpus — frozen MINIMAL models from every
   found path-divergence bug, run through the same pair registry as the
   fuzzer. Each entry is the smallest source form exhibiting the shape that
   diverged in the 2026-07-06 audit (genmlx-ansg), recast as a permanent
   property. New PEF failures get shrunk and appended here.

   Open bugs deliberately NOT yet encoded (add on fix): genmlx-5a87 (Map
   update-with-diffs vs p/update), genmlx-dp60, genmlx-8mih.

   Run: bunx --bun nbb@1.4.208 test/genmlx/pef_corpus.cljs"
  (:require [genmlx.pef :as pef]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def ^:private core-pairs
  [:p1-compiled-vs-handler :p2-regen-fast-vs-general :p3-prefix-vs-handler
   :p4-batched-vs-scalar :p5-analytical :i4-discard-roundtrip :i5-score-type])

(defn- check! [label bundle pair-names]
  (let [r (pef/check-model bundle {:pair-names pair-names
                                   :key (rng/fresh-key 3141)})]
    (when-not (:pass? r)
      (doseq [f (:failures r)] (println "    divergence:" (pr-str f))))
    (assert-true label (:pass? r))))

;; ===========================================================================
(println "\n-- corpus: genmlx-dv66 — nested trace call inside dist-args --")
;; The audit shape: an inner trace call nested in an outer site's parameters.
;; Fast-regen once mis-handled the inner site's dependency.
(let [src '([] (let [v0 (trace :x0 (dist/gaussian (trace :n0 (dist/gaussian 0 1)) 1.0))
                     v1 (trace :x1 (dist/gaussian v0 0.5))]
                 v1))]
  (check! "dv66: nested-trace model agrees across all core pairs"
          {:model (pef/source->model src) :args [] :source src}
          core-pairs))

;; ===========================================================================
(println "\n-- corpus: genmlx-njzu — splice deps in BOTH directions --")
;; Site feeds splice arg; splice retval feeds a later site. The regenerate
;; fast-path gate must see both provenance directions.
(let [sub '([a] (let [w0 (trace :y0 (dist/gaussian a 1.0))] w0))
      src '([] (let [v0 (trace :x0 (dist/gaussian 0 1))
                     v1 (splice :sp0 pefsub/sub0 v0)
                     v2 (trace :x2 (dist/gaussian v1 0.7))]
                 v2))]
  (check! "njzu: splice-dep model agrees across all core pairs"
          {:model (pef/source->model src {'sub0 (pef/source->model sub)})
           :args [] :source src}
          core-pairs))

;; ===========================================================================
(println "\n-- corpus: genmlx-94qc — let REBINDING (same name, derived value) --")
;; The audit shape: a let name rebound to a derived value; the analytical /
;; compiled paths must track the rebound symbol, not the first binding.
(let [src '([] (let [v (trace :x0 (dist/gaussian 0 2))
                     v (mx/add (mx/multiply v 0.5) 1.0)
                     v1 (trace :x1 (dist/gaussian v 1.0))]
                 v1))]
  (check! "94qc: rebinding model agrees across all core pairs"
          {:model (pef/source->model src) :args [] :source src}
          core-pairs))

;; ===========================================================================
(println "\n-- corpus: genmlx-rmy7 — analytical vs handler, conjugate pair --")
;; gaussian-gaussian conjugacy: when the analytical path fires on the
;; constrained observation, generate's weight must equal the closed-form
;; marginal N(y; m0, s0^2 + sn^2) — checked against an INDEPENDENT oracle.
(let [src '([] (let [mu (trace :mu (dist/gaussian 0 2))
                     y  (trace :y (dist/gaussian mu 1.0))]
                 y))
      model (pef/source->model src)
      y-obs 1.5
      log-marginal (fn [y m0 s0 sn]
                     (let [v (+ (* s0 s0) (* sn sn))]
                       (- (* -0.5 (js/Math.log (* 2 js/Math.PI v)))
                          (/ (* (- y m0) (- y m0)) (* 2 v)))))
      obs (cm/set-choice cm/EMPTY [:y] (mx/scalar y-obs))
      w (:weight (p/generate (dyn/with-key model (rng/fresh-key 5)) [] obs))
      _ (mx/eval! w)
      w (mx/item w)
      oracle (log-marginal y-obs 0.0 2.0 1.0)]
  (println "    generate weight =" w "| closed-form marginal =" oracle)
  (assert-true "rmy7: constrained-obs generate weight equals the closed-form marginal"
               (< (js/Math.abs (- w oracle)) 1e-3))
  (check! "rmy7: conjugate model agrees across all core pairs"
          {:model model :args [] :source src}
          core-pairs))

;; ===========================================================================
(println "\n-- corpus: genmlx-uizc — Mix regenerate (same-index resample) --")
;; The audit bug: a same-index component resample failed to regenerate
;; selected INNER sites. Pinned via a gen wrapper that splices a Mix over two
;; gaussian kernels. QUARANTINE (genmlx-175y): running :p2 here exposed a
;; fast-vs-forced-general weight divergence on the spliced Mix — a NEW find
;; under investigation; :p2 re-arms when 175y closes.
(let [k1 (pef/source->model '([] (let [z (trace :z (dist/gaussian -2 0.5))] z)))
      k2 (pef/source->model '([] (let [z (trace :z (dist/gaussian 2 0.5))] z)))
      mixed (comb/mix-combinator [k1 k2] (mx/array [-0.7 -0.7]))
      src '([] (let [v0 (splice :mx0 pefsub/sub0)
                     v1 (trace :x1 (dist/gaussian v0 1.0))]
                 v1))]
  (check! "uizc: spliced-Mix model survives round-trip/score-type pairs (:p2 quarantined -> genmlx-175y)"
          {:model (pef/source->model src {'sub0 mixed}) :args [] :source src}
          [:i4-discard-roundtrip :i5-score-type]))

;; ===========================================================================
(println "\n-- corpus: genmlx-uxjm — SMC increments vs one-shot IS --")
;; The audit bug: plain-driver SMC update increments replaced prior-sampled
;; obs values, giving an invalid estimator. Pinned as the P6 statistical
;; property on a 3-site gaussian chain.
(let [src '([] (let [v0 (trace :x0 (dist/gaussian 0 1))
                     v1 (trace :x1 (dist/gaussian v0 1.0))
                     v2 (trace :x2 (dist/gaussian v1 1.0))]
                 v2))]
  (check! "uxjm: sequential SMC log-ML agrees with one-shot IS (statistical band)"
          {:model (pef/source->model src) :args [] :source src}
          [:p6-smc-vs-is]))

;; ===========================================================================
(println "\n-- corpus: PEF-FOUND #1 — M3 prefix must decline nested-in-dist-args --")
;; Found by the fuzzer's FIRST smoke run (seed 42 idx 7, smoke profile,
;; shrunk by feature bisection): the M3 prefix walker accepted a trace site
;; whose dist-args contain a NESTED trace call (the dv66 shape, prefix
;; edition). The inner site is invisible to the compiled prefix, whose
;; compiled dist-args then read a value that never exists — every simulate
;; crashed with a NAPI type error while strip-compiled ran fine. Fixed in
;; compiled.cljs walk-prefix-bindings/walk-prefix-forms (nested-gen guard).
(let [src '([] (let [v2 (trace :x2 (dist/gaussian (trace :n2 (dist/gaussian 0 0.3)) 1.0))
                     v3 (if (pos? (mx/item v2))
                          (trace :b3 (dist/gaussian -0.2 2.8))
                          (trace :b3f (dist/gaussian 0.01 2.0)))]
                 v3))
      model (pef/source->model src)]
  (assert-true "pef#1: nested-in-dist-args + suffix does NOT prefix-compile"
               (nil? (get-in model [:schema :compiled-prefix])))
  (assert-true "pef#1: the model simulates (handler path)"
               (some? (p/simulate (dyn/with-key model (rng/fresh-key 3)) [])))
  (check! "pef#1: model agrees across regen/round-trip/score-type pairs"
          {:model model :args [] :source src}
          [:p2-regen-fast-vs-general :i4-discard-roundtrip :i5-score-type]))

;; ===========================================================================
(println (str "\n== pef corpus: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
