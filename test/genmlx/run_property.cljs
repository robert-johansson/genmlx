(ns genmlx.run-property
  "Category runner: property-based tests only."
  (:require [cljs.test :as t]
            genmlx.test-helpers
            ;; All 28 property-based test files (defspec + related)
            ;; Data structures
            genmlx.selection-property-test
            genmlx.choicemap-property-test
            genmlx.trace-immutability-property-test
            ;; Core GFI + handler
            genmlx.handler-property-test
            genmlx.gfi-property-test
            ;; Distributions
            genmlx.dist-property-test
            genmlx.dist-normalization-property-test
            genmlx.support-property-test
            genmlx.entropy-property-test
            genmlx.location-scale-property-test
            genmlx.score-monotonicity-property-test
            ;; Combinators
            genmlx.combinator-property-test
            genmlx.combinator-extra-property-test
            genmlx.splice-property-test
            genmlx.branching-property-test
            ;; Inference
            genmlx.inference-property-test
            genmlx.mcmc-property-test
            genmlx.smc-property-test
            genmlx.vi-property-test
            genmlx.adev-property-test
            ;; Gradients + learning
            genmlx.gradient-mcmc-property-test
            genmlx.gradient-learning-property-test
            ;; Vectorized + compiled
            genmlx.vectorized-property-test
            genmlx.compiled-equiv-property-test
            genmlx.vmap-property-test
            ;; Edit, serialize, misc
            genmlx.edit-property-test
            genmlx.serialize-property-test
            genmlx.stack-unstack-property-test))

(defmethod t/report [:cljs.test/default :end-run-tests] [{:keys [test pass fail error]}]
  (println (str "\n" test " tests, "
                pass " assertions, "
                fail " failures, "
                error " errors."))
  (when (pos? (+ fail error))
    (js/process.exit 1)))

(t/run-all-tests #"genmlx\..*-property-test")
