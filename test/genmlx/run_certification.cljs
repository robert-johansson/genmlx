(ns genmlx.run-certification
  "Category runner: certification and compatibility tests."
  (:require [cljs.test :as t]
            genmlx.test-helpers
            genmlx.test-models
            ;; Certification tests
            genmlx.level0-certification-test
            genmlx.l4-certification-test
            ;; Compatibility tests
            genmlx.gen-clj-compat-test
            genmlx.genjax-compat-test))

(defmethod t/report [:cljs.test/default :end-run-tests] [{:keys [test pass fail error]}]
  (println (str "\n" test " tests, "
                pass " assertions, "
                fail " failures, "
                error " errors."))
  (when (pos? (+ fail error))
    (js/process.exit 1)))

(t/run-all-tests #"genmlx\.(level0|l4|gen-clj|genjax)-.*-test")
