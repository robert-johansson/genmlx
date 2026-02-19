(ns genmlx.smcp3-kernel-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== SMCP3 Kernel Tests ===\n")

;; Model: x ~ N(0, 1), y ~ N(x, 0.5)
(def model
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))]
      (mx/eval! x)
      (dyn/trace :y (dist/gaussian (mx/item x) 0.5))
      (mx/item x))))

;; Forward kernel: random-walk propose new :x near current :x
(def fwd-kernel
  (gen [choices]
    (let [cur-x (mx/realize (cm/get-choice choices [:x]))]
      (dyn/trace :x (dist/gaussian cur-x 0.5)))))

;; Backward kernel: symmetric (same structure)
(def bwd-kernel
  (gen [choices]
    (let [new-x (mx/realize (cm/get-choice choices [:x]))]
      (dyn/trace :x (dist/gaussian new-x 0.5)))))

;; Two observation steps (step 1 triggers kernel path)
(def obs-seq [(cm/choicemap :y (mx/scalar 3.0))
              (cm/choicemap :y (mx/scalar 3.0))])

;; ---------------------------------------------------------------------------
;; Test 1: Runs without error
;; ---------------------------------------------------------------------------
(println "-- SMCP3 with kernels runs --")
(let [result (smcp3/smcp3
               {:particles 10
                :forward-kernel fwd-kernel
                :backward-kernel bwd-kernel
                :key (rng/fresh-key)}
               model [] obs-seq)]
  (assert-true "smcp3 returns result" (some? result)))

;; ---------------------------------------------------------------------------
;; Test 2: Structure — correct keys and particle count
;; ---------------------------------------------------------------------------
(println "\n-- Structure --")
(let [n 10
      result (smcp3/smcp3
               {:particles n
                :forward-kernel fwd-kernel
                :backward-kernel bwd-kernel
                :key (rng/fresh-key)}
               model [] obs-seq)]
  (assert-true "result has :traces" (some? (:traces result)))
  (assert-true "result has :log-weights" (some? (:log-weights result)))
  (assert-true "result has :log-ml-estimate" (some? (:log-ml-estimate result)))
  (assert-true "correct particle count for traces" (= n (count (:traces result))))
  (assert-true "correct particle count for weights" (= n (count (:log-weights result)))))

;; ---------------------------------------------------------------------------
;; Test 3: Traces valid — each trace has :x (finite) and :y (matches obs)
;; ---------------------------------------------------------------------------
(println "\n-- Traces valid --")
(let [result (smcp3/smcp3
               {:particles 10
                :forward-kernel fwd-kernel
                :backward-kernel bwd-kernel
                :key (rng/fresh-key)}
               model [] obs-seq)
      traces (:traces result)
      all-x-finite (every? (fn [t]
                             (let [x (mx/realize (cm/get-choice (:choices t) [:x]))]
                               (js/isFinite x)))
                           traces)
      all-y-correct (every? (fn [t]
                              (let [y (mx/realize (cm/get-choice (:choices t) [:y]))]
                                (< (js/Math.abs (- y 3.0)) 1e-6)))
                            traces)]
  (assert-true "all traces have finite :x" all-x-finite)
  (assert-true "all traces have :y = 3.0" all-y-correct))

;; ---------------------------------------------------------------------------
;; Test 4: Weights finite
;; ---------------------------------------------------------------------------
(println "\n-- Weights finite --")
(let [result (smcp3/smcp3
               {:particles 10
                :forward-kernel fwd-kernel
                :backward-kernel bwd-kernel
                :key (rng/fresh-key)}
               model [] obs-seq)
      all-w-finite (every? (fn [w] (js/isFinite (mx/realize w)))
                           (:log-weights result))
      ml-finite (js/isFinite (mx/realize (:log-ml-estimate result)))]
  (assert-true "all log-weights finite" all-w-finite)
  (assert-true "log-ml-estimate finite" ml-finite))

;; ---------------------------------------------------------------------------
;; Test 5: Log-ML reasonable
;; ---------------------------------------------------------------------------
(println "\n-- Log-ML reasonable --")
(let [result (smcp3/smcp3
               {:particles 50
                :forward-kernel fwd-kernel
                :backward-kernel bwd-kernel
                :key (rng/fresh-key)}
               model [] obs-seq)
      log-ml (mx/realize (:log-ml-estimate result))]
  ;; For x~N(0,1), y|x~N(x,0.5), p(y=3) can be computed analytically:
  ;; marginal y ~ N(0, 1+0.25) = N(0, 1.25), so log p(y=3) ≈ -4.8
  ;; Two steps with same observation, so total around -5 to -15
  (assert-true "log-ML in plausible range [-30, 0]"
    (and (> log-ml -30) (< log-ml 0))))

;; ---------------------------------------------------------------------------
;; Test 6: Direct smcp3-init + smcp3-step with kernels
;; ---------------------------------------------------------------------------
(println "\n-- Direct smcp3-init + smcp3-step --")
(let [n 10
      key (rng/fresh-key)
      [k1 k2] (rng/split-n (rng/ensure-key key) 2)
      ;; Step 0: init
      init-result (smcp3/smcp3-init model [] (first obs-seq) nil n k1)
      ;; Step 1: step with kernels
      step-result (smcp3/smcp3-step
                    (:traces init-result)
                    (:log-weights init-result)
                    model (second obs-seq)
                    fwd-kernel bwd-kernel
                    n 0.5 nil k2)]
  (assert-true "init has :traces" (some? (:traces init-result)))
  (assert-true "init has :log-weights" (some? (:log-weights init-result)))
  (assert-true "init has :log-ml-increment" (some? (:log-ml-increment init-result)))
  (assert-true "step has :traces" (some? (:traces step-result)))
  (assert-true "step has :log-weights" (some? (:log-weights step-result)))
  (assert-true "step has :log-ml-increment" (some? (:log-ml-increment step-result)))
  (assert-true "step has :ess" (some? (:ess step-result)))
  (assert-true "step has :resampled?" (contains? step-result :resampled?))
  (assert-true "step traces count" (= n (count (:traces step-result))))
  (assert-true "step weights count" (= n (count (:log-weights step-result)))))

(println "\nAll SMCP3 kernel tests complete.")
