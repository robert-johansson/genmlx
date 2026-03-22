(ns genmlx.runner
  "Full test suite runner for GenMLX.
   Requires all test namespaces and runs them via cljs.test.
   Exits with code 1 on any failure."
  (:require [cljs.test :as t]
            ;; Phase 0: Infrastructure
            genmlx.test-helpers
            genmlx.test-models
            ;; Phase 1: Distribution tests
            genmlx.dist-logprob-test
            genmlx.dist-symmetry-test
            genmlx.dist-boundary-test
            genmlx.dist-moments-test
            genmlx.dist-batch-test
            genmlx.dist-normalization-test
            genmlx.dist-gradient-test
            genmlx.dist-error-test
            ;; Phase 2: GFI protocol tests
            genmlx.gfi-simulate-test
            genmlx.gfi-generate-test
            genmlx.gfi-update-test
            genmlx.gfi-regenerate-test
            genmlx.gfi-assess-test
            genmlx.gfi-project-test
            genmlx.gfi-contracts-test
            ;; Phase 3: Handler tests
            genmlx.handler-purity-test
            genmlx.handler-transitions-test
            ;; Phase 4: Inference convergence tests
            genmlx.inference-is-test
            genmlx.inference-mh-test
            genmlx.inference-hmc-test
            genmlx.inference-smc-test
            genmlx.inference-vi-test
            genmlx.inference-adev-test
            genmlx.inference-agreement-test
            ;; Phase 5: Combinator tests
            genmlx.combinator-mask-test
            ;; Phase 6-7: Vectorized / compiled tests
            genmlx.vectorized-shape-test
            genmlx.vectorized-equivalence-test
            genmlx.compiled-equivalence-test
            ;; Phase 8: Data structure tests
            genmlx.trace-immutability-test
            ;; Phase 9: Property-based tests (defspec)
            genmlx.selection-property-test
            genmlx.choicemap-property-test
            genmlx.handler-property-test
            genmlx.gfi-property-test
            genmlx.combinator-property-test
            genmlx.combinator-extra-property-test
            genmlx.dist-property-test
            genmlx.dist-normalization-property-test
            genmlx.inference-property-test
            genmlx.mcmc-property-test
            genmlx.smc-property-test
            genmlx.vi-property-test
            genmlx.adev-property-test
            genmlx.gradient-mcmc-property-test
            genmlx.gradient-learning-property-test
            genmlx.vectorized-property-test
            genmlx.compiled-equiv-property-test
            genmlx.vmap-property-test
            genmlx.edit-property-test
            genmlx.splice-property-test
            genmlx.branching-property-test
            genmlx.serialize-property-test
            genmlx.stack-unstack-property-test
            ;; Phase 10: Certification / compatibility tests
            genmlx.level0-certification-test
            genmlx.l4-certification-test
            genmlx.gen-clj-compat-test
            genmlx.genjax-compat-test))

;; Override the end-run-tests report to print summary and exit 1 on failure.
(defmethod t/report [:cljs.test/default :end-run-tests] [{:keys [test pass fail error]}]
  (println (str "\n" test " tests, "
                pass " assertions, "
                fail " failures, "
                error " errors."))
  (when (pos? (+ fail error))
    (js/process.exit 1)))

(t/run-all-tests #"genmlx\..*-test")
