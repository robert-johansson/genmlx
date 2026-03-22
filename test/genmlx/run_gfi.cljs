(ns genmlx.run-gfi
  "Category runner: GFI protocol tests only."
  (:require [cljs.test :as t]
            genmlx.test-helpers
            genmlx.test-models
            ;; GFI unit tests
            genmlx.gfi-simulate-test
            genmlx.gfi-generate-test
            genmlx.gfi-update-test
            genmlx.gfi-regenerate-test
            genmlx.gfi-assess-test
            genmlx.gfi-project-test
            genmlx.gfi-contracts-test
            ;; GFI property tests
            genmlx.gfi-property-test))

(defmethod t/report [:cljs.test/default :end-run-tests] [{:keys [test pass fail error]}]
  (println (str "\n" test " tests, "
                pass " assertions, "
                fail " failures, "
                error " errors."))
  (when (pos? (+ fail error))
    (js/process.exit 1)))

(t/run-all-tests #"genmlx\.gfi.*-test")
