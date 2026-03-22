(ns genmlx.run-dist
  "Category runner: distribution tests only."
  (:require [cljs.test :as t]
            genmlx.test-helpers
            ;; Distribution unit tests
            genmlx.dist-logprob-test
            genmlx.dist-symmetry-test
            genmlx.dist-boundary-test
            genmlx.dist-moments-test
            genmlx.dist-batch-test
            genmlx.dist-normalization-test
            genmlx.dist-gradient-test
            genmlx.dist-error-test
            ;; Distribution property tests
            genmlx.dist-property-test
            genmlx.dist-normalization-property-test))

(defmethod t/report [:cljs.test/default :end-run-tests] [{:keys [test pass fail error]}]
  (println (str "\n" test " tests, "
                pass " assertions, "
                fail " failures, "
                error " errors."))
  (when (pos? (+ fail error))
    (js/process.exit 1)))

(t/run-all-tests #"genmlx\.dist.*-test")
