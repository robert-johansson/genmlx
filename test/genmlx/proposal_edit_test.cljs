(ns genmlx.proposal-edit-test
  "Tests for ProposalEdit."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.edit :as edit]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Model: x ~ N(0, 10), y ~ N(x, 1)
(def model
  (dyn/auto-key (gen []
    (let [x (trace :x (dist/gaussian 0 10))]
      (mx/eval! x)
      (trace :y (dist/gaussian (mx/item x) 1))
      (mx/item x)))))

;; Forward kernel: propose new :x ~ N(4.0, 0.5)
(def forward-gf
  (dyn/auto-key (gen [choices]
    (trace :x (dist/gaussian 4.0 0.5)))))

;; Backward kernel: score old :x under N(4.0, 0.5)
(def backward-gf
  (dyn/auto-key (gen [choices]
    (trace :x (dist/gaussian 4.0 0.5)))))

(deftest proposal-edit-structure
  (testing "edit returns correct keys + backward-request"
    (let [obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace weight]} (p/generate model [] obs)
          edit-req (edit/proposal-edit forward-gf backward-gf)
          result (edit/edit model trace edit-req)]
      (is (some? (:trace result)) "result has :trace")
      (is (some? (:weight result)) "result has :weight")
      (is (some? (:discard result)) "result has :discard")
      (is (some? (:backward-request result)) "result has :backward-request")
      (is (instance? edit/ProposalEdit (:backward-request result))
          "backward-request is ProposalEdit"))))

(deftest proposal-edit-backward-swap
  (testing "forward/backward GFs are swapped"
    (let [obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace]} (p/generate model [] obs)
          edit-req (edit/proposal-edit forward-gf backward-gf)
          result (edit/edit model trace edit-req)
          bwd-req (:backward-request result)]
      (is (identical? (:forward-gf bwd-req) backward-gf)
          "backward .forward-gf = original backward-gf")
      (is (identical? (:backward-gf bwd-req) forward-gf)
          "backward .backward-gf = original forward-gf"))))

(deftest proposal-edit-weight-correctness
  (testing "manually verify weight components"
    (let [obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace]} (p/generate model [] obs)
          old-x (mx/realize (cm/get-choice (:choices trace) [:x]))
          edit-req (edit/proposal-edit forward-gf backward-gf)
          result (edit/edit model trace edit-req)
          edit-weight (mx/realize (:weight result))
          new-x (mx/realize (cm/get-choice (:choices (:trace result)) [:x]))
          fwd-score (mx/realize (dist/log-prob (dist/gaussian 4.0 0.5) (mx/scalar new-x)))
          fwd-choices (cm/choicemap :x (mx/scalar new-x))
          {:keys [weight]} (p/update model trace fwd-choices)
          update-weight (mx/realize weight)
          bwd-score (mx/realize (dist/log-prob (dist/gaussian 4.0 0.5) (mx/scalar old-x)))
          expected-weight (+ update-weight (- bwd-score fwd-score))]
      (is (h/close? expected-weight edit-weight 1e-4)
          "edit weight matches manual computation"))))

(deftest data-dependent-proposals
  (testing "kernels read choices arg"
    (let [dep-forward (dyn/auto-key (gen [choices]
                        (let [cur-x (mx/realize (cm/get-choice choices [:x]))]
                          (trace :x (dist/gaussian cur-x 0.5)))))
          dep-backward (dyn/auto-key (gen [choices]
                         (let [new-x (mx/realize (cm/get-choice choices [:x]))]
                           (trace :x (dist/gaussian new-x 0.5)))))
          obs (cm/choicemap :y (mx/scalar 5.0))
          {:keys [trace]} (p/generate model [] obs)
          edit-req (edit/proposal-edit dep-forward dep-backward)
          result (edit/edit model trace edit-req)
          w (mx/realize (:weight result))]
      (is (js/isFinite w) "data-dependent weight is finite")
      (is (some? (cm/get-value (cm/get-submap (:choices (:trace result)) :x)))
          "result trace has :x")
      (is (some? (cm/get-value (cm/get-submap (:choices (:trace result)) :y)))
          "result trace has :y"))))

(deftest mh-loop-with-proposal-edit
  (testing "100 iterations with ProposalEdit"
    (let [dep-forward (dyn/auto-key (gen [choices]
                        (let [cur-x (mx/realize (cm/get-choice choices [:x]))]
                          (trace :x (dist/gaussian cur-x 1.0)))))
          dep-backward (dyn/auto-key (gen [choices]
                         (let [new-x (mx/realize (cm/get-choice choices [:x]))]
                           (trace :x (dist/gaussian new-x 1.0)))))
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
      (is (js/isFinite final-x) "MH final x is finite")
      (is (and (> final-x -10) (< final-x 20))
          "MH final x in plausible range [-10, 20]"))))

(cljs.test/run-tests)
