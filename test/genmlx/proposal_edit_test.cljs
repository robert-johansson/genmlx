(ns genmlx.proposal-edit-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.edit :as edit]
            [genmlx.inference.util :as u])
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

(println "\n=== ProposalEdit Tests ===\n")

;; Model: x ~ N(0, 10), y ~ N(x, 1)
(def model
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 10))]
      (mx/eval! x)
      (dyn/trace :y (dist/gaussian (mx/item x) 1))
      (mx/item x))))

;; Forward kernel: propose new :x ~ N(4.0, 0.5)
(def forward-gf
  (gen [choices]
    (dyn/trace :x (dist/gaussian 4.0 0.5))))

;; Backward kernel: score old :x under N(4.0, 0.5)
(def backward-gf
  (gen [choices]
    (dyn/trace :x (dist/gaussian 4.0 0.5))))

;; ---------------------------------------------------------------------------
;; Test 1: Structure — edit returns correct keys + backward-request
;; ---------------------------------------------------------------------------
(println "-- Structure --")
(let [obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace weight]} (p/generate model [] obs)
      edit-req (edit/proposal-edit forward-gf backward-gf)
      result (edit/edit model trace edit-req)]
  (assert-true "result has :trace" (some? (:trace result)))
  (assert-true "result has :weight" (some? (:weight result)))
  (assert-true "result has :discard" (some? (:discard result)))
  (assert-true "result has :backward-request" (some? (:backward-request result)))
  (assert-true "backward-request is ProposalEdit"
    (instance? edit/ProposalEdit (:backward-request result))))

;; ---------------------------------------------------------------------------
;; Test 2: Backward swap — forward/backward GFs are swapped
;; ---------------------------------------------------------------------------
(println "\n-- Backward swap --")
(let [obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate model [] obs)
      edit-req (edit/proposal-edit forward-gf backward-gf)
      result (edit/edit model trace edit-req)
      bwd-req (:backward-request result)]
  (assert-true "backward .forward-gf = original backward-gf"
    (identical? (:forward-gf bwd-req) backward-gf))
  (assert-true "backward .backward-gf = original forward-gf"
    (identical? (:backward-gf bwd-req) forward-gf)))

;; ---------------------------------------------------------------------------
;; Test 3: Weight correctness — manually verify weight components
;; ---------------------------------------------------------------------------
(println "\n-- Weight correctness --")
(let [obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate model [] obs)
      old-x (mx/realize (cm/get-choice (:choices trace) [:x]))
      ;; Run proposal edit
      edit-req (edit/proposal-edit forward-gf backward-gf)
      result (edit/edit model trace edit-req)
      edit-weight (mx/realize (:weight result))
      ;; Get new x from result trace
      new-x (mx/realize (cm/get-choice (:choices (:trace result)) [:x]))
      ;; Manually compute components:
      ;; fwd_score = log-prob(new_x | N(4, 0.5))
      fwd-score (mx/realize (dist/log-prob (dist/gaussian 4.0 0.5) (mx/scalar new-x)))
      ;; update_weight via p/update
      fwd-choices (cm/choicemap :x (mx/scalar new-x))
      {:keys [weight]} (p/update model trace fwd-choices)
      update-weight (mx/realize weight)
      ;; bwd_score = log-prob(old_x | N(4, 0.5))
      bwd-score (mx/realize (dist/log-prob (dist/gaussian 4.0 0.5) (mx/scalar old-x)))
      ;; Expected weight
      expected-weight (+ update-weight (- bwd-score fwd-score))]
  (assert-close "edit weight matches manual computation"
    expected-weight edit-weight 1e-4))

;; ---------------------------------------------------------------------------
;; Test 4: Data-dependent proposals — kernels read choices arg
;; ---------------------------------------------------------------------------
(println "\n-- Data-dependent proposals --")
(let [;; Forward: propose near current x (random walk)
      dep-forward (gen [choices]
                    (let [cur-x (mx/realize (cm/get-choice choices [:x]))]
                      (dyn/trace :x (dist/gaussian cur-x 0.5))))
      ;; Backward: score under same random-walk (symmetric)
      dep-backward (gen [choices]
                     (let [new-x (mx/realize (cm/get-choice choices [:x]))]
                       (dyn/trace :x (dist/gaussian new-x 0.5))))
      obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate model [] obs)
      edit-req (edit/proposal-edit dep-forward dep-backward)
      result (edit/edit model trace edit-req)
      w (mx/realize (:weight result))]
  (assert-true "data-dependent weight is finite" (js/isFinite w))
  (assert-true "result trace has :x" (some? (cm/get-value (cm/get-submap (:choices (:trace result)) :x))))
  (assert-true "result trace has :y" (some? (cm/get-value (cm/get-submap (:choices (:trace result)) :y)))))

;; ---------------------------------------------------------------------------
;; Test 5: MH loop — 100 iterations with ProposalEdit
;; ---------------------------------------------------------------------------
(println "\n-- MH loop with ProposalEdit --")
(let [dep-forward (gen [choices]
                    (let [cur-x (mx/realize (cm/get-choice choices [:x]))]
                      (dyn/trace :x (dist/gaussian cur-x 1.0))))
      dep-backward (gen [choices]
                     (let [new-x (mx/realize (cm/get-choice choices [:x]))]
                       (dyn/trace :x (dist/gaussian new-x 1.0))))
      obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate model [] obs)
      n-iter 100
      final-trace
      (loop [i 0 tr trace]
        (if (>= i n-iter)
          tr
          (let [edit-req (edit/proposal-edit dep-forward dep-backward)
                result (edit/edit model tr edit-req)
                log-alpha (mx/realize (:weight result))
                accept? (or (>= log-alpha 0)
                            (< (js/Math.log (js/Math.random)) log-alpha))]
            (recur (inc i) (if accept? (:trace result) tr)))))
      final-x (mx/realize (cm/get-choice (:choices final-trace) [:x]))]
  (assert-true "MH final x is finite" (js/isFinite final-x))
  ;; Posterior mean of x|y=5 with prior N(0,10) and likelihood N(x,1):
  ;; posterior ~ N(5*100/(100+1), ...) ≈ N(4.95, ...)
  ;; With 100 MH steps and wide proposal, just check x is in a reasonable range
  (assert-true "MH final x in plausible range [-10, 20]"
    (and (> final-x -10) (< final-x 20))))

(println "\nAll ProposalEdit tests complete.")
