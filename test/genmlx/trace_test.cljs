(ns genmlx.trace-test
  (:require [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]))

(defn assert= [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" (pr-str expected))
        (println "    actual:  " (pr-str actual)))))

(println "\n=== Trace Tests ===\n")

(let [choices (cm/choicemap :x 1.0 :y 2.0)
      trace (tr/make-trace {:gen-fn :test :args [1 2] :choices choices
                            :retval 42 :score 0.5})]
  (assert= "get-retval" 42 (tr/get-retval trace))
  (assert= "get-score" 0.5 (tr/get-score trace))
  (assert= "get-args" [1 2] (tr/get-args trace))
  (assert= "get-gen-fn" :test (tr/get-gen-fn trace))
  (assert= "get-choices" choices (tr/get-choices trace)))

(println "\nAll trace tests complete.")
