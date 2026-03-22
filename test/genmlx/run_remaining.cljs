(ns genmlx.run-remaining
  "Category runner: handler, combinator, vectorized, compiled, and data structure tests.
   Covers test namespaces not matched by dist/gfi/inference/property/certification runners."
  (:require [cljs.test :as t]
            genmlx.test-helpers
            genmlx.test-models
            ;; Handler tests
            genmlx.handler-purity-test
            genmlx.handler-transitions-test
            ;; Combinator tests
            genmlx.combinator-mask-test
            ;; Vectorized / compiled tests
            genmlx.vectorized-shape-test
            genmlx.vectorized-equivalence-test
            genmlx.compiled-equivalence-test
            ;; Data structure tests
            genmlx.trace-immutability-test))

(defmethod t/report [:cljs.test/default :end-run-tests] [{:keys [test pass fail error]}]
  (println (str "\n" test " tests, "
                pass " assertions, "
                fail " failures, "
                error " errors."))
  (when (pos? (+ fail error))
    (js/process.exit 1)))

(t/run-all-tests #"genmlx\.(handler|combinator|vectorized|compiled|trace).*-test")
