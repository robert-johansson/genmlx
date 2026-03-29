(ns genmlx.well-formedness-test
  "Tests for DML well-formedness restrictions ([T] Ch. 2.2.1, p.63).

   Restrictions:
     1. Halts with probability 1 (#42)
     2. Addresses must be unique (#43 — already in verify.cljs)
     3. No external randomness (#44)
     4. No mutation (#45)
     5. No HOF passing of gen fns (#46)

   Each restriction has:
     - Source-analysis unit tests (crafted source forms)
     - Integration tests via validate-gen-fn
     - GFI law checks via gfi/check-law"
  (:require [cljs.test :refer [deftest is testing are]]
            [genmlx.verify :as verify]
            [genmlx.gfi :as gfi]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.trace :as tr])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- has-violation?
  "True if result contains a violation of the given type."
  [result type]
  (some #(= type (:type %)) (:violations result)))

(defn- violation-count
  "Number of violations of the given type."
  [result type]
  (count (filter #(= type (:type %)) (:violations result))))

(defn- make-model-with-source
  "Create a DynamicGF with a clean body-fn but a crafted source form.
   The body-fn always traces :x ~ N(0,1); the source form is what
   the static analysis inspects."
  [source]
  (dyn/auto-key
    (dyn/make-gen-fn
      (fn [rt]
        (let [trace (.-trace rt)]
          (trace :x (dist/gaussian 0 1))))
      source)))

;; ---------------------------------------------------------------------------
;; #44: No external randomness (DML restriction 3)
;; ---------------------------------------------------------------------------

(deftest clean-source-has-no-external-randomness
  (are [source]
    (empty? (verify/check-no-external-randomness source))
    '([] (let [x (trace :x (dist/gaussian 0 1))] x))
    '([] (trace :x (dist/uniform 0 1)))
    '([mu] (trace :x (dist/gaussian mu 1)))
    nil))

(deftest detects-rand
  (testing "bare rand"
    (let [violations (verify/check-no-external-randomness
                       '([] (let [x (rand)] (trace :y (dist/gaussian x 1)))))]
      (is (= 1 (count violations)))
      (is (= :external-randomness (:type (first violations))))))
  (testing "rand-int"
    (is (seq (verify/check-no-external-randomness
               '([] (let [n (rand-int 10)] (trace :y (dist/gaussian n 1))))))))
  (testing "rand-nth"
    (is (seq (verify/check-no-external-randomness
               '([] (let [x (rand-nth [1 2 3])] (trace :y (dist/gaussian x 1)))))))))

(deftest detects-js-math-random
  (let [violations (verify/check-no-external-randomness
                     '([] (let [x (js/Math.random)] (trace :y (dist/gaussian x 1)))))]
    (is (= 1 (count violations)))
    (is (= :external-randomness (:type (first violations))))))

(deftest detects-math-random
  (let [violations (verify/check-no-external-randomness
                     '([] (let [x (Math/random)] (trace :y (dist/gaussian x 1)))))]
    (is (= 1 (count violations)))
    (is (= :external-randomness (:type (first violations))))))

(deftest external-randomness-severity-is-warning
  (let [{:keys [severity]} (first (verify/check-no-external-randomness
                                    '([] (rand))))]
    (is (= :warning severity))))

;; ---------------------------------------------------------------------------
;; #45: No mutation (DML restriction 4)
;; ---------------------------------------------------------------------------

(deftest clean-source-has-no-mutation
  (are [source]
    (empty? (verify/check-no-mutation source))
    '([] (let [x (trace :x (dist/gaussian 0 1))] x))
    '([] (trace :x (dist/uniform 0 1)))
    nil))

(deftest detects-atom-and-swap
  (let [violations (verify/check-no-mutation
                     '([] (let [a (atom 0)] (swap! a inc) (trace :x (dist/gaussian @a 1)))))]
    (is (= 2 (count violations)))
    (is (every? #(= :mutation (:type %)) violations))))

(deftest detects-volatile-and-vreset
  (let [violations (verify/check-no-mutation
                     '([] (let [v (volatile! 0)] (vreset! v 1) (trace :x (dist/gaussian @v 1)))))]
    (is (= 2 (count violations)))
    (is (every? #(= :mutation (:type %)) violations))))

(deftest detects-set-bang
  (is (seq (verify/check-no-mutation
             '([] (do (set! js/window.foo 1) (trace :x (dist/gaussian 0 1))))))))

(deftest detects-aset
  (is (seq (verify/check-no-mutation
             '([] (let [arr (js/Array. 3)] (aset arr 0 42) (trace :x (dist/gaussian 0 1))))))))

(deftest detects-vswap
  (is (seq (verify/check-no-mutation
             '([] (let [v (volatile! 0)] (vswap! v inc)))))))

(deftest mutation-severity-is-warning
  (let [{:keys [severity]} (first (verify/check-no-mutation '([] (atom 0))))]
    (is (= :warning severity))))

;; ---------------------------------------------------------------------------
;; #46: No HOF gen fns (DML restriction 5)
;; ---------------------------------------------------------------------------

(deftest clean-source-has-no-hof-gen-fns
  (are [source]
    (empty? (verify/check-no-hof-gen-fns source))
    '([] (let [x (trace :x (dist/gaussian 0 1))] x))
    '([] (let [xs (map inc (range 5))] (trace :x (dist/gaussian 0 1))))
    nil))

(deftest detects-gen-fn-passed-to-map
  (let [violations (verify/check-no-hof-gen-fns
                     '([] (map (gen [i] (trace :x (dist/gaussian 0 1))) (range 5))))]
    (is (= 1 (count violations)))
    (is (= :hof-gen-fn (:type (first violations))))))

(deftest detects-gen-fn-passed-to-reduce
  (is (seq (verify/check-no-hof-gen-fns
             '([] (reduce (gen [acc x] (trace :x (dist/gaussian acc 1))) 0 xs))))))

(deftest detects-gen-fn-passed-to-filter
  (is (seq (verify/check-no-hof-gen-fns
             '([] (filter (gen [x] (trace :ok (dist/bernoulli 0.5))) xs))))))

(deftest detects-gen-fn-passed-to-mapv
  (is (seq (verify/check-no-hof-gen-fns
             '([] (mapv (gen [i] (trace :x (dist/gaussian 0 1))) (range 5)))))))

(deftest detects-gen-fn-passed-to-keep
  (is (seq (verify/check-no-hof-gen-fns
             '([] (keep (gen [x] (trace :x (dist/gaussian x 1))) xs))))))

(deftest hof-gen-fn-severity-is-warning
  (let [{:keys [severity]} (first (verify/check-no-hof-gen-fns
                                    '([] (map (gen [] (trace :x (dist/gaussian 0 1))) xs))))]
    (is (= :warning severity))))

(deftest non-gen-fn-in-hof-is-clean
  (testing "regular fn passed to map is not flagged"
    (is (empty? (verify/check-no-hof-gen-fns
                  '([] (map (fn [x] (* x 2)) (range 5)))))))
  (testing "symbol passed to map is not flagged"
    (is (empty? (verify/check-no-hof-gen-fns
                  '([] (map inc (range 5))))))))

;; ---------------------------------------------------------------------------
;; #42: Halts with probability 1 (DML restriction 1)
;; ---------------------------------------------------------------------------

(deftest clean-model-halts
  (let [model (gen [] (trace :x (dist/gaussian 0 1)))
        result (verify/validate-gen-fn model [])]
    (is (not (has-violation? result :non-termination)))))

(deftest throwing-model-detected-as-non-halting
  (testing "model that always throws gets non-termination warning"
    (let [model (dyn/auto-key
                  (dyn/make-gen-fn
                    (fn [_rt] (throw (js/Error. "infinite loop simulation")))
                    '([] (loop [] (recur)))))
          result (verify/validate-gen-fn model [])]
      (is (has-violation? result :non-termination)))))

;; ---------------------------------------------------------------------------
;; Integration: validate-gen-fn catches all DML violations
;; ---------------------------------------------------------------------------

(deftest validate-gen-fn-catches-external-randomness
  (let [model (make-model-with-source
                '([] (let [x (rand)] (trace :y (dist/gaussian x 1)))))
        result (verify/validate-gen-fn model [])]
    (is (:valid? result) "external randomness is warning, not error")
    (is (has-violation? result :external-randomness))))

(deftest validate-gen-fn-catches-mutation
  (let [model (make-model-with-source
                '([] (let [a (atom 0)] (swap! a inc) (trace :x (dist/gaussian @a 1)))))
        result (verify/validate-gen-fn model [])]
    (is (:valid? result) "mutation is warning, not error")
    (is (has-violation? result :mutation))))

(deftest validate-gen-fn-catches-hof-gen-fn
  (let [model (make-model-with-source
                '([] (map (gen [i] (trace :x (dist/gaussian 0 1))) (range 5))))
        result (verify/validate-gen-fn model [])]
    (is (:valid? result) "hof-gen-fn is warning, not error")
    (is (has-violation? result :hof-gen-fn))))

(deftest validate-gen-fn-clean-model-no-well-formedness-violations
  (let [model (gen [] (trace :x (dist/gaussian 0 1)))
        result (verify/validate-gen-fn model [])]
    (is (:valid? result))
    (is (not (has-violation? result :external-randomness)))
    (is (not (has-violation? result :mutation)))
    (is (not (has-violation? result :hof-gen-fn)))
    (is (not (has-violation? result :non-termination)))))

(deftest validate-gen-fn-multiple-violations-combined
  (testing "source with both mutation and external randomness"
    (let [model (make-model-with-source
                  '([] (let [a (atom 0)]
                         (reset! a (rand))
                         (trace :x (dist/gaussian @a 1)))))
          result (verify/validate-gen-fn model [])]
      (is (has-violation? result :mutation))
      (is (has-violation? result :external-randomness)))))

;; ---------------------------------------------------------------------------
;; GFI law integration
;; ---------------------------------------------------------------------------

(deftest gfi-law-no-external-randomness-clean
  (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
        {:keys [pass?]} (gfi/check-law :no-external-randomness model [])]
    (is pass?)))

(deftest gfi-law-no-external-randomness-dirty
  (let [model (make-model-with-source
                '([] (let [x (rand)] (trace :y (dist/gaussian x 1)))))
        {:keys [pass?]} (gfi/check-law :no-external-randomness model [])]
    (is (not pass?))))

(deftest gfi-law-no-mutation-clean
  (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
        {:keys [pass?]} (gfi/check-law :no-mutation model [])]
    (is pass?)))

(deftest gfi-law-no-mutation-dirty
  (let [model (make-model-with-source
                '([] (let [a (atom 0)] (swap! a inc))))
        {:keys [pass?]} (gfi/check-law :no-mutation model [])]
    (is (not pass?))))

(deftest gfi-law-no-hof-gen-fns-clean
  (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
        {:keys [pass?]} (gfi/check-law :no-hof-gen-fns model [])]
    (is pass?)))

(deftest gfi-law-no-hof-gen-fns-dirty
  (let [model (make-model-with-source
                '([] (map (gen [i] (trace :x (dist/gaussian 0 1))) (range 5))))
        {:keys [pass?]} (gfi/check-law :no-hof-gen-fns model [])]
    (is (not pass?))))

(deftest gfi-well-formedness-tag-runs-all-restrictions
  (testing "verify with :well-formedness tag includes all restriction laws"
    (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          report (gfi/verify model [] :tags [:well-formedness] :n-trials 1)
          law-names (set (map :name (:results report)))]
      (is (contains? law-names :halts-with-probability-one))
      (is (contains? law-names :no-external-randomness))
      (is (contains? law-names :no-mutation))
      (is (contains? law-names :no-hof-gen-fns))
      (is (:all-pass? report)))))

;; ---------------------------------------------------------------------------
;; Edge cases
;; ---------------------------------------------------------------------------

(deftest nil-source-passes-all-checks
  (testing "model without source metadata passes all source-analysis checks"
    (is (nil? (verify/check-no-external-randomness nil)))
    (is (nil? (verify/check-no-mutation nil)))
    (is (nil? (verify/check-no-hof-gen-fns nil)))))

(deftest nested-violations-detected
  (testing "violation nested inside let/if/when is still found"
    (is (seq (verify/check-no-mutation
               '([] (let [x 1]
                      (if (> x 0)
                        (let [a (atom 0)] (reset! a 1))
                        nil))))))
    (is (seq (verify/check-no-external-randomness
               '([] (when true
                      (let [inner (fn [] (rand))]
                        (trace :x (dist/gaussian 0 1))))))))))

(deftest multi-site-model-passes-well-formedness
  (let [model (gen [xs]
                (let [slope (trace :slope (dist/gaussian 0 10))
                      intercept (trace :intercept (dist/gaussian 0 10))]
                  (doseq [[j x] (map-indexed vector xs)]
                    (trace (keyword (str "y" j))
                           (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                                  intercept) 1)))
                  slope))
        result (verify/validate-gen-fn model [(mapv float [1 2 3])])]
    (is (:valid? result))
    (is (not (has-violation? result :external-randomness)))
    (is (not (has-violation? result :mutation)))
    (is (not (has-violation? result :hof-gen-fn)))))

(cljs.test/run-tests)
