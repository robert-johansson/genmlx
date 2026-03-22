(ns genmlx.run-inference
  "Category runner: inference tests only."
  (:require [cljs.test :as t]
            genmlx.test-helpers
            genmlx.test-models
            ;; Inference unit tests
            genmlx.inference-is-test
            genmlx.inference-mh-test
            genmlx.inference-hmc-test
            genmlx.inference-smc-test
            genmlx.inference-vi-test
            genmlx.inference-adev-test
            genmlx.inference-agreement-test
            ;; Inference property tests
            genmlx.inference-property-test))

(defmethod t/report [:cljs.test/default :end-run-tests] [{:keys [test pass fail error]}]
  (println (str "\n" test " tests, "
                pass " assertions, "
                fail " failures, "
                error " errors."))
  (when (pos? (+ fail error))
    (js/process.exit 1)))

(t/run-all-tests #"genmlx\.inference.*-test")
