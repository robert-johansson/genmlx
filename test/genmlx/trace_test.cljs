(ns genmlx.trace-test
  "Tests for trace construction and field access."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]))

(deftest trace-field-access-test
  (testing "Trace fields are accessible via keyword lookup"
    (let [choices (cm/choicemap :x 1.0 :y 2.0)
          trace (tr/make-trace {:gen-fn :test :args [1 2] :choices choices
                                :retval 42 :score 0.5})]
      (is (= 42 (:retval trace)) "get-retval")
      (is (= 0.5 (:score trace)) "get-score")
      (is (= [1 2] (:args trace)) "get-args")
      (is (= :test (:gen-fn trace)) "get-gen-fn")
      (is (= choices (:choices trace)) "get-choices"))))

(cljs.test/run-tests)
