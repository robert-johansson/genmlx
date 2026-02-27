(ns genmlx.verify-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.verify :as verify]
            [genmlx.trace :as tr]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" expected)
        (println "    actual:  " actual))))

(defn has-violation? [result type]
  (some #(= type (:type %)) (:violations result)))

(println "\n=== Validate-Gen-Fn Tests ===\n")

;; 1. Valid model
(println "-- valid model --")
(let [model (gen [mu]
              (dyn/trace :x (dist/gaussian mu 1))
              (dyn/trace :y (dist/gaussian 0 1)))
      result (verify/validate-gen-fn model [0.0])]
  (assert-true "valid model returns valid" (:valid? result))
  (assert-equal "no violations" 0 (count (:violations result)))
  (assert-true "trace returned" (instance? tr/Trace (:trace result))))

;; 2. Duplicate address
(println "\n-- duplicate address --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :x (dist/gaussian 0 1)))
      result (verify/validate-gen-fn model [])]
  (assert-true "duplicate detected as invalid" (not (:valid? result)))
  (assert-true "has duplicate-address violation" (has-violation? result :duplicate-address))
  (assert-equal "violation has addr" :x
    (:addr (first (filter #(= :duplicate-address (:type %)) (:violations result))))))

;; 3. Non-finite score (MLX scalar sigma bypasses JS number check)
(println "\n-- non-finite score --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 (mx/scalar 0.0))))
      result (verify/validate-gen-fn model [])]
  (assert-true "non-finite detected as invalid" (not (:valid? result)))
  (assert-true "has non-finite-score violation" (has-violation? result :non-finite-score)))

;; 4. Empty model
(println "\n-- empty model --")
(let [model (gen []
              (mx/scalar 42.0))
      result (verify/validate-gen-fn model [])]
  (assert-true "empty model is valid (warning only)" (:valid? result))
  (assert-true "has empty-model warning" (has-violation? result :empty-model))
  (assert-equal "warning severity" :warning
    (:severity (first (filter #(= :empty-model (:type %)) (:violations result))))))

;; 5. Source analysis for eval!/item
(println "\n-- materialization in body --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (mx/eval! x)
                (mx/item x)))
      result (verify/validate-gen-fn model [])]
  (assert-true "materialization model is valid (warning only)" (:valid? result))
  (assert-true "has materialization warning" (has-violation? result :materialization-in-body))
  (assert-true "at least 1 materialization warning"
    (>= (count (filter #(= :materialization-in-body (:type %)) (:violations result))) 1)))

;; 6. Model that throws
(println "\n-- model that throws --")
(let [model (gen []
              (throw (js/Error. "intentional error")))
      result (verify/validate-gen-fn model [])]
  (assert-true "throwing model is invalid" (not (:valid? result)))
  (assert-true "has execution-error" (has-violation? result :execution-error))
  (assert-true "no trace on error" (nil? (:trace result))))

;; 7. Conditional duplicate (needs multiple trials)
(println "\n-- conditional duplicate (multi-trial) --")
(let [model (gen []
              (let [flip (dyn/trace :flip (dist/bernoulli 0.5))]
                (mx/eval! flip)
                (if (> (mx/item flip) 0.5)
                  (do (dyn/trace :a (dist/gaussian 0 1))
                      (dyn/trace :a (dist/gaussian 0 1)))
                  (dyn/trace :b (dist/gaussian 0 1)))))
      ;; Single trial might miss the dup; many trials should catch it
      result (verify/validate-gen-fn model [] {:n-trials 20})]
  (assert-true "conditional dup caught with multi-trial"
    (has-violation? result :duplicate-address)))

;; 8. Multi-site valid model
(println "\n-- multi-site valid model --")
(let [model (gen [xs]
              (let [slope (dyn/trace :slope (dist/gaussian 0 10))
                    intercept (dyn/trace :intercept (dist/gaussian 0 10))]
                (doseq [[j x] (map-indexed vector xs)]
                  (dyn/trace (keyword (str "y" j))
                             (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                                    intercept) 1)))
                slope))
      result (verify/validate-gen-fn model [(mapv float [1 2 3 4 5])])]
  (assert-true "multi-site model is valid" (:valid? result))
  (assert-equal "no violations" 0 (count (:violations result)))
  (assert-true "trace returned" (instance? tr/Trace (:trace result))))

(println "\n=== All validate-gen-fn tests complete ===")
