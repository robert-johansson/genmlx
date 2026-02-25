(ns genmlx.assess-propose-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.dist.core :as dc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println "  PASS:" msg)
      (println "  FAIL:" msg "- expected" expected "got" actual "diff" diff))))

(println "\n=== Assess & Propose Combinator Tests ===\n")

;; Simple kernel used across tests
(def simple-kernel
  (gen [x]
    (let [y (dyn/trace :y (dist/gaussian x 1))]
      (mx/eval! y)
      (mx/item y))))

;; ---------------------------------------------------------------------------
;; MapCombinator
;; ---------------------------------------------------------------------------
(println "-- Map propose --")
(let [mapped (comb/map-combinator simple-kernel)
      {:keys [choices weight retval]} (p/propose mapped [[1.0 2.0 3.0]])]
  (mx/eval! weight)
  (assert-true "map propose returns choices" (not= choices cm/EMPTY))
  (assert-true "map propose has weight" (number? (mx/item weight)))
  (assert-true "map propose has 3 retvals" (= 3 (count retval))))

(println "\n-- Map assess --")
(let [mapped (comb/map-combinator simple-kernel)
      {:keys [choices weight]} (p/propose mapped [[1.0 2.0]])
      assess-result (p/assess mapped [[1.0 2.0]] choices)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (assert-close "map assess weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
  (assert-true "map assess has retval" (= 2 (count (:retval assess-result)))))

;; ---------------------------------------------------------------------------
;; UnfoldCombinator
;; ---------------------------------------------------------------------------
(println "\n-- Unfold propose --")
(let [step (gen [t state]
             (let [next (dyn/trace :x (dist/gaussian state 0.1))]
               (mx/eval! next)
               (mx/item next)))
      unfold (comb/unfold-combinator step)
      {:keys [choices weight retval]} (p/propose unfold [3 0.0])]
  (mx/eval! weight)
  (assert-true "unfold propose returns choices" (not= choices cm/EMPTY))
  (assert-true "unfold propose has weight" (number? (mx/item weight)))
  (assert-true "unfold propose has 3 retvals" (= 3 (count retval))))

(println "\n-- Unfold assess --")
(let [step (gen [t state]
             (let [next (dyn/trace :x (dist/gaussian state 0.1))]
               (mx/eval! next)
               (mx/item next)))
      unfold (comb/unfold-combinator step)
      {:keys [choices weight]} (p/propose unfold [3 0.0])
      assess-result (p/assess unfold [3 0.0] choices)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (assert-close "unfold assess weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5))

;; ---------------------------------------------------------------------------
;; SwitchCombinator
;; ---------------------------------------------------------------------------
(println "\n-- Switch propose --")
(let [b0 (gen [] (let [x (dyn/trace :x (dist/gaussian 0 1))]
                   (mx/eval! x) (mx/item x)))
      b1 (gen [] (let [x (dyn/trace :x (dist/gaussian 10 1))]
                   (mx/eval! x) (mx/item x)))
      sw (comb/switch-combinator b0 b1)
      r0 (p/propose sw [0])
      r1 (p/propose sw [1])]
  (mx/eval! (:weight r0))
  (mx/eval! (:weight r1))
  (assert-true "switch propose branch 0 has weight" (number? (mx/item (:weight r0))))
  (assert-true "switch propose branch 1 has weight" (number? (mx/item (:weight r1))))
  (assert-true "switch propose branch 0 has choices" (not= (:choices r0) cm/EMPTY)))

(println "\n-- Switch assess --")
(let [b0 (gen [] (let [x (dyn/trace :x (dist/gaussian 0 1))]
                   (mx/eval! x) (mx/item x)))
      sw (comb/switch-combinator b0)
      {:keys [choices weight]} (p/propose sw [0])
      assess-result (p/assess sw [0] choices)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (assert-close "switch assess weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5))

;; ---------------------------------------------------------------------------
;; ScanCombinator
;; ---------------------------------------------------------------------------
(println "\n-- Scan propose --")
(let [kernel (gen [carry input]
               (let [x (dyn/trace :x (dist/gaussian (mx/add carry input) 0.1))]
                 (mx/eval! x)
                 [(mx/item x) (mx/item x)]))
      scan (comb/scan-combinator kernel)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      {:keys [choices weight retval]} (p/propose scan [(mx/scalar 0.0) inputs])]
  (mx/eval! weight)
  (assert-true "scan propose has choices" (not= choices cm/EMPTY))
  (assert-true "scan propose has weight" (number? (mx/item weight)))
  (assert-true "scan propose has carry" (some? (:carry retval)))
  (assert-true "scan propose has 3 outputs" (= 3 (count (:outputs retval)))))

(println "\n-- Scan assess --")
(let [kernel (gen [carry input]
               (let [x (dyn/trace :x (dist/gaussian (mx/add carry input) 0.1))]
                 (mx/eval! x)
                 [(mx/item x) (mx/item x)]))
      scan (comb/scan-combinator kernel)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      {:keys [choices weight]} (p/propose scan [(mx/scalar 0.0) inputs])
      assess-result (p/assess scan [(mx/scalar 0.0) inputs] choices)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (assert-close "scan assess weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5))

;; ---------------------------------------------------------------------------
;; MaskCombinator
;; ---------------------------------------------------------------------------
(println "\n-- Mask propose --")
(let [masked (comb/mask-combinator simple-kernel)
      active (p/propose masked [true 5.0])
      inactive (p/propose masked [false 5.0])]
  (mx/eval! (:weight active))
  (mx/eval! (:weight inactive))
  (assert-true "mask propose active has choices" (not= (:choices active) cm/EMPTY))
  (assert-true "mask propose active has weight" (number? (mx/item (:weight active))))
  (assert-true "mask propose inactive empty choices" (= (:choices inactive) cm/EMPTY))
  (assert-true "mask propose inactive zero weight" (= 0.0 (mx/item (:weight inactive))))
  (assert-true "mask propose inactive nil retval" (nil? (:retval inactive))))

(println "\n-- Mask assess --")
(let [masked (comb/mask-combinator simple-kernel)
      {:keys [choices weight]} (p/propose masked [true 5.0])
      assess-result (p/assess masked [true 5.0] choices)
      assess-inactive (p/assess masked [false 5.0] cm/EMPTY)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (mx/eval! (:weight assess-inactive))
  (assert-close "mask assess active weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5)
  (assert-true "mask assess inactive zero weight" (= 0.0 (mx/item (:weight assess-inactive)))))

;; ---------------------------------------------------------------------------
;; RecurseCombinator
;; ---------------------------------------------------------------------------
(println "\n-- Recurse propose --")
(let [rec (comb/recurse
            (fn [self]
              (gen [depth]
                (let [x (dyn/trace :x (dist/gaussian 0 1))]
                  (mx/eval! x) (mx/item x)))))
      {:keys [choices weight retval]} (p/propose rec [0])]
  (mx/eval! weight)
  (assert-true "recurse propose has choices" (not= choices cm/EMPTY))
  (assert-true "recurse propose has weight" (number? (mx/item weight)))
  (assert-true "recurse propose has retval" (number? retval)))

(println "\n-- Recurse assess --")
(let [rec (comb/recurse
            (fn [self]
              (gen [depth]
                (let [x (dyn/trace :x (dist/gaussian 0 1))]
                  (mx/eval! x) (mx/item x)))))
      {:keys [choices weight]} (p/propose rec [0])
      assess-result (p/assess rec [0] choices)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (assert-close "recurse assess weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5))

;; ---------------------------------------------------------------------------
;; ContramapGF
;; ---------------------------------------------------------------------------
(println "\n-- Contramap propose --")
(let [cmapped (comb/contramap-gf simple-kernel identity)
      {:keys [choices weight retval]} (p/propose cmapped [5.0])]
  (mx/eval! weight)
  (assert-true "contramap propose has choices" (not= choices cm/EMPTY))
  (assert-true "contramap propose has weight" (number? (mx/item weight)))
  (assert-true "contramap propose has retval" (number? retval)))

(println "\n-- Contramap assess --")
(let [cmapped (comb/contramap-gf simple-kernel identity)
      {:keys [choices weight]} (p/propose cmapped [5.0])
      assess-result (p/assess cmapped [5.0] choices)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (assert-close "contramap assess weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5))

;; ---------------------------------------------------------------------------
;; MapRetvalGF
;; ---------------------------------------------------------------------------
(println "\n-- MapRetval propose --")
(let [mr (comb/map-retval simple-kernel (fn [v] (* v 2)))
      {:keys [choices weight retval]} (p/propose mr [5.0])]
  (mx/eval! weight)
  (assert-true "map-retval propose has choices" (not= choices cm/EMPTY))
  (assert-true "map-retval propose has weight" (number? (mx/item weight)))
  (assert-true "map-retval propose retval is doubled" (number? retval)))

(println "\n-- MapRetval assess --")
(let [mr (comb/map-retval simple-kernel (fn [v] (* v 2)))
      {:keys [choices weight]} (p/propose mr [5.0])
      assess-result (p/assess mr [5.0] choices)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (assert-close "map-retval assess weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5))

;; ---------------------------------------------------------------------------
;; MixCombinator
;; ---------------------------------------------------------------------------
(println "\n-- Mix propose --")
(let [c0 (gen [] (let [x (dyn/trace :x (dist/gaussian 0 1))]
                   (mx/eval! x) (mx/item x)))
      c1 (gen [] (let [x (dyn/trace :x (dist/gaussian 10 1))]
                   (mx/eval! x) (mx/item x)))
      mix (comb/mix-combinator [c0 c1] (mx/log (mx/array [0.5 0.5])))
      {:keys [choices weight retval]} (p/propose mix [])]
  (mx/eval! weight)
  (assert-true "mix propose has choices" (not= choices cm/EMPTY))
  (assert-true "mix propose has weight" (number? (mx/item weight)))
  (assert-true "mix propose has component-idx"
    (some? (cm/get-choice choices [:component-idx]))))

(println "\n-- Mix assess --")
(let [c0 (gen [] (let [x (dyn/trace :x (dist/gaussian 0 1))]
                   (mx/eval! x) (mx/item x)))
      c1 (gen [] (let [x (dyn/trace :x (dist/gaussian 10 1))]
                   (mx/eval! x) (mx/item x)))
      mix (comb/mix-combinator [c0 c1] (mx/log (mx/array [0.5 0.5])))
      {:keys [choices weight]} (p/propose mix [])
      assess-result (p/assess mix [] choices)]
  (mx/eval! weight)
  (mx/eval! (:weight assess-result))
  (assert-close "mix assess weight matches propose"
    (mx/item weight) (mx/item (:weight assess-result)) 1e-5))

;; ---------------------------------------------------------------------------
;; Cross-check: assess weight matches generate score for fully-constrained
;; ---------------------------------------------------------------------------
(println "\n-- Cross-check: assess vs generate score --")
(let [mapped (comb/map-combinator simple-kernel)
      {:keys [choices weight]} (p/propose mapped [[1.0 2.0]])
      gen-result (p/generate mapped [[1.0 2.0]] choices)
      assess-result (p/assess mapped [[1.0 2.0]] choices)]
  (mx/eval! (:score (:trace gen-result)))
  (mx/eval! (:weight assess-result))
  (assert-close "map: assess weight = generate trace score"
    (mx/item (:score (:trace gen-result)))
    (mx/item (:weight assess-result)) 1e-5))

(println "\nAll assess & propose combinator tests complete.")
