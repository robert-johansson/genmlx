(ns genmlx.assess-propose-test
  "Assess and propose combinator tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.dist.core :as dc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Simple kernel used across tests
(def simple-kernel
  (dyn/auto-key
    (gen [x]
      (let [y (trace :y (dist/gaussian x 1))]
        (mx/eval! y)
        (mx/item y)))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest map-propose-test
  (testing "map propose"
    (let [mapped (comb/map-combinator simple-kernel)
          {:keys [choices weight retval]} (p/propose mapped [[1.0 2.0 3.0]])]
      (mx/eval! weight)
      (is (not= choices cm/EMPTY) "map propose returns choices")
      (is (number? (mx/item weight)) "map propose has weight")
      (is (= 3 (count retval)) "map propose has 3 retvals")))

  (testing "map assess"
    (let [mapped (comb/map-combinator simple-kernel)
          {:keys [choices weight]} (p/propose mapped [[1.0 2.0]])
          assess-result (p/assess mapped [[1.0 2.0]] choices)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "map assess weight matches propose")
      (is (= 2 (count (:retval assess-result))) "map assess has retval"))))

(deftest unfold-propose-test
  (testing "unfold propose"
    (let [step (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   (mx/eval! next)
                   (mx/item next))))
          unfold (comb/unfold-combinator step)
          {:keys [choices weight retval]} (p/propose unfold [3 0.0])]
      (mx/eval! weight)
      (is (not= choices cm/EMPTY) "unfold propose returns choices")
      (is (number? (mx/item weight)) "unfold propose has weight")
      (is (= 3 (count retval)) "unfold propose has 3 retvals")))

  (testing "unfold assess"
    (let [step (dyn/auto-key (gen [t state]
                 (let [next (trace :x (dist/gaussian state 0.1))]
                   (mx/eval! next)
                   (mx/item next))))
          unfold (comb/unfold-combinator step)
          {:keys [choices weight]} (p/propose unfold [3 0.0])
          assess-result (p/assess unfold [3 0.0] choices)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "unfold assess weight matches propose"))))

(deftest switch-propose-test
  (testing "switch propose"
    (let [b0 (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                       (mx/eval! x) (mx/item x))))
          b1 (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 10 1))]
                       (mx/eval! x) (mx/item x))))
          sw (comb/switch-combinator b0 b1)
          r0 (p/propose sw [0])
          r1 (p/propose sw [1])]
      (mx/eval! (:weight r0))
      (mx/eval! (:weight r1))
      (is (number? (mx/item (:weight r0))) "switch propose branch 0 has weight")
      (is (number? (mx/item (:weight r1))) "switch propose branch 1 has weight")
      (is (not= (:choices r0) cm/EMPTY) "switch propose branch 0 has choices")))

  (testing "switch assess"
    (let [b0 (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                       (mx/eval! x) (mx/item x))))
          sw (comb/switch-combinator b0)
          {:keys [choices weight]} (p/propose sw [0])
          assess-result (p/assess sw [0] choices)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "switch assess weight matches propose"))))

(deftest scan-propose-test
  (testing "scan propose"
    (let [kernel (dyn/auto-key (gen [carry input]
                   (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                     (mx/eval! x)
                     [(mx/item x) (mx/item x)])))
          scan (comb/scan-combinator kernel)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          {:keys [choices weight retval]} (p/propose scan [(mx/scalar 0.0) inputs])]
      (mx/eval! weight)
      (is (not= choices cm/EMPTY) "scan propose has choices")
      (is (number? (mx/item weight)) "scan propose has weight")
      (is (some? (:carry retval)) "scan propose has carry")
      (is (= 3 (count (:outputs retval))) "scan propose has 3 outputs")))

  (testing "scan assess"
    (let [kernel (dyn/auto-key (gen [carry input]
                   (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
                     (mx/eval! x)
                     [(mx/item x) (mx/item x)])))
          scan (comb/scan-combinator kernel)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          {:keys [choices weight]} (p/propose scan [(mx/scalar 0.0) inputs])
          assess-result (p/assess scan [(mx/scalar 0.0) inputs] choices)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "scan assess weight matches propose"))))

(deftest mask-propose-test
  (testing "mask propose"
    (let [masked (comb/mask-combinator simple-kernel)
          active (p/propose masked [true 5.0])
          inactive (p/propose masked [false 5.0])]
      (mx/eval! (:weight active))
      (mx/eval! (:weight inactive))
      (is (not= (:choices active) cm/EMPTY) "mask propose active has choices")
      (is (number? (mx/item (:weight active))) "mask propose active has weight")
      (is (= (:choices inactive) cm/EMPTY) "mask propose inactive empty choices")
      (is (= 0.0 (mx/item (:weight inactive))) "mask propose inactive zero weight")
      (is (nil? (:retval inactive)) "mask propose inactive nil retval")))

  (testing "mask assess"
    (let [masked (comb/mask-combinator simple-kernel)
          {:keys [choices weight]} (p/propose masked [true 5.0])
          assess-result (p/assess masked [true 5.0] choices)
          assess-inactive (p/assess masked [false 5.0] cm/EMPTY)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (mx/eval! (:weight assess-inactive))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "mask assess active weight matches propose")
      (is (= 0.0 (mx/item (:weight assess-inactive))) "mask assess inactive zero weight"))))

(deftest recurse-propose-test
  (testing "recurse propose"
    (let [rec (comb/recurse
                (fn [self]
                  (dyn/auto-key (gen [depth]
                    (let [x (trace :x (dist/gaussian 0 1))]
                      (mx/eval! x) (mx/item x))))))
          {:keys [choices weight retval]} (p/propose rec [0])]
      (mx/eval! weight)
      (is (not= choices cm/EMPTY) "recurse propose has choices")
      (is (number? (mx/item weight)) "recurse propose has weight")
      (is (number? retval) "recurse propose has retval")))

  (testing "recurse assess"
    (let [rec (comb/recurse
                (fn [self]
                  (dyn/auto-key (gen [depth]
                    (let [x (trace :x (dist/gaussian 0 1))]
                      (mx/eval! x) (mx/item x))))))
          {:keys [choices weight]} (p/propose rec [0])
          assess-result (p/assess rec [0] choices)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "recurse assess weight matches propose"))))

(deftest contramap-propose-test
  (testing "contramap propose"
    (let [cmapped (comb/contramap-gf simple-kernel identity)
          {:keys [choices weight retval]} (p/propose cmapped [5.0])]
      (mx/eval! weight)
      (is (not= choices cm/EMPTY) "contramap propose has choices")
      (is (number? (mx/item weight)) "contramap propose has weight")
      (is (number? retval) "contramap propose has retval")))

  (testing "contramap assess"
    (let [cmapped (comb/contramap-gf simple-kernel identity)
          {:keys [choices weight]} (p/propose cmapped [5.0])
          assess-result (p/assess cmapped [5.0] choices)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "contramap assess weight matches propose"))))

(deftest map-retval-propose-test
  (testing "map-retval propose"
    (let [mr (comb/map-retval simple-kernel (fn [v] (* v 2)))
          {:keys [choices weight retval]} (p/propose mr [5.0])]
      (mx/eval! weight)
      (is (not= choices cm/EMPTY) "map-retval propose has choices")
      (is (number? (mx/item weight)) "map-retval propose has weight")
      (is (number? retval) "map-retval propose retval is doubled")))

  (testing "map-retval assess"
    (let [mr (comb/map-retval simple-kernel (fn [v] (* v 2)))
          {:keys [choices weight]} (p/propose mr [5.0])
          assess-result (p/assess mr [5.0] choices)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "map-retval assess weight matches propose"))))

(deftest mix-propose-test
  (testing "mix propose"
    (let [c0 (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                       (mx/eval! x) (mx/item x))))
          c1 (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 10 1))]
                       (mx/eval! x) (mx/item x))))
          mix (comb/mix-combinator [c0 c1] (mx/log (mx/array [0.5 0.5])))
          {:keys [choices weight retval]} (p/propose mix [])]
      (mx/eval! weight)
      (is (not= choices cm/EMPTY) "mix propose has choices")
      (is (number? (mx/item weight)) "mix propose has weight")
      (is (some? (cm/get-choice choices [:component-idx])) "mix propose has component-idx")))

  (testing "mix assess"
    (let [c0 (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                       (mx/eval! x) (mx/item x))))
          c1 (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 10 1))]
                       (mx/eval! x) (mx/item x))))
          mix (comb/mix-combinator [c0 c1] (mx/log (mx/array [0.5 0.5])))
          {:keys [choices weight]} (p/propose mix [])
          assess-result (p/assess mix [] choices)]
      (mx/eval! weight)
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
          "mix assess weight matches propose"))))

(deftest cross-check-assess-vs-generate-test
  (testing "cross-check: assess vs generate score"
    (let [mapped (comb/map-combinator simple-kernel)
          {:keys [choices weight]} (p/propose mapped [[1.0 2.0]])
          gen-result (p/generate mapped [[1.0 2.0]] choices)
          assess-result (p/assess mapped [[1.0 2.0]] choices)]
      (mx/eval! (:score (:trace gen-result)))
      (mx/eval! (:weight assess-result))
      (is (h/close? (mx/item (:score (:trace gen-result)))
                    (mx/item (:weight assess-result)) 1e-5)
          "map: assess weight = generate trace score"))))

(cljs.test/run-tests)
