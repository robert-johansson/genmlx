(ns genmlx.hmm-forward-test
  "HMM forward middleware tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as hnd]
            [genmlx.runtime :as rt]
            [genmlx.inference.hmm-forward :as hmm]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test setup: 2-state HMM
;; State 0: "low" emission ~ N(0, 1)
;; State 1: "high" emission ~ N(5, 1)
;; ---------------------------------------------------------------------------

(def log-trans
  (mx/array [[(js/Math.log 0.9) (js/Math.log 0.1)]
             [(js/Math.log 0.1) (js/Math.log 0.9)]]))

(def K 2)

(defn emission-log-probs [obs]
  "Compute [K]-shaped log p(obs | state=k) for Gaussian emissions."
  (let [lp0 (dc/dist-log-prob (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) obs)
        lp1 (dc/dist-log-prob (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)) obs)]
    (mx/stack [lp0 lp1])))

(deftest hmm-predict-test
  (testing "pure hmm-predict"
    (let [log-alpha (mx/array [(js/Math.log 1.0) (js/Math.log 1e-30)])
          predicted (hmm/hmm-predict log-alpha log-trans)]
      (mx/eval! predicted)
      (is (= [2] (mx/shape predicted)) "predicted is [2]-shaped")
      (let [probs (mx/exp predicted)
            p0 (mx/item (mx/index probs 0))
            p1 (mx/item (mx/index probs 1))]
        (is (h/close? 0.9 p0 0.02) "P(state 0) ~ 0.9")
        (is (h/close? 0.1 p1 0.02) "P(state 1) ~ 0.1")))))

(deftest hmm-update-test
  (testing "pure hmm-update"
    (let [log-alpha (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)])
          obs (mx/scalar 4.5)
          log-ep (emission-log-probs obs)
          mask (mx/scalar 1.0)
          {:keys [log-alpha ll]} (hmm/hmm-update log-alpha log-ep mask)]
      (mx/eval! log-alpha)
      (mx/eval! ll)
      (is (= [2] (mx/shape log-alpha)) "updated belief is [2]-shaped")
      (is (= [] (mx/shape ll)) "ll is scalar")
      (let [probs (mx/exp log-alpha)
            p1 (mx/item (mx/index probs 1))]
        (is (> p1 0.95) "state 1 posterior > 0.95")))))

(deftest hmm-missing-data-test
  (testing "missing data (mask=0)"
    (let [log-alpha (mx/array [(js/Math.log 0.7) (js/Math.log 0.3)])
          obs (mx/scalar 4.5)
          log-ep (emission-log-probs obs)
          mask (mx/scalar 0.0)
          {:keys [log-alpha ll]} (hmm/hmm-update log-alpha log-ep mask)]
      (mx/eval! log-alpha)
      (mx/eval! ll)
      (let [probs (mx/exp log-alpha)
            p0 (mx/item (mx/index probs 0))]
        (is (h/close? 0.7 p0 0.01) "belief unchanged with mask=0")
        (is (h/close? 0.0 (mx/item ll) 0.001) "LL = 0 with mask=0")))))

(deftest hmm-step-test
  (testing "hmm-step (predict + update)"
    (let [log-alpha (mx/array [(js/Math.log 1.0) (js/Math.log 1e-30)])
          obs (mx/scalar 4.8)
          log-ep (emission-log-probs obs)
          mask (mx/scalar 1.0)
          {:keys [log-alpha ll]} (hmm/hmm-step log-alpha log-trans log-ep mask)]
      (mx/eval! log-alpha)
      (mx/eval! ll)
      (let [probs (mx/exp log-alpha)
            p1 (mx/item (mx/index probs 1))]
        (is (> p1 0.5) "state 1 posterior > 0.5 after observation near 5")))))

(deftest hmm-sequence-marginal-ll-test
  (testing "sequence marginal LL: exact vs brute force"
    (let [obs-seq [(mx/scalar 0.2) (mx/scalar 0.1) (mx/scalar 4.8)]
          forward-ll
          (loop [t 0
                 log-alpha (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)])
                 acc-ll (mx/scalar 0.0)]
            (if (>= t (count obs-seq))
              acc-ll
              (let [predicted (if (zero? t)
                                log-alpha
                                (hmm/hmm-predict log-alpha log-trans))
                    log-ep (emission-log-probs (nth obs-seq t))
                    {:keys [log-alpha ll]} (hmm/hmm-update predicted log-ep (mx/scalar 1.0))]
                (recur (inc t) log-alpha (mx/add acc-ll ll)))))
          _ (mx/eval! forward-ll)
          mu [0.0 5.0]
          log-init [(js/Math.log 0.5) (js/Math.log 0.5)]
          trans-p [[0.9 0.1] [0.1 0.9]]
          log-probs
          (for [s0 [0 1] s1 [0 1] s2 [0 1]]
            (+ (nth log-init s0)
               (mx/item (dc/dist-log-prob
                 (dist/gaussian (mx/scalar (nth mu s0)) (mx/scalar 1.0))
                 (nth obs-seq 0)))
               (js/Math.log (get-in trans-p [s0 s1]))
               (mx/item (dc/dist-log-prob
                 (dist/gaussian (mx/scalar (nth mu s1)) (mx/scalar 1.0))
                 (nth obs-seq 1)))
               (js/Math.log (get-in trans-p [s1 s2]))
               (mx/item (dc/dist-log-prob
                 (dist/gaussian (mx/scalar (nth mu s2)) (mx/scalar 1.0))
                 (nth obs-seq 2)))))
          max-lp (apply max log-probs)
          brute-ll (+ max-lp (js/Math.log
                               (reduce + (map #(js/Math.exp (- % max-lp)) log-probs))))]
      (is (h/close? brute-ll (mx/item forward-ll) 0.01)
          "forward LL matches brute force"))))

(deftest hmm-batched-forward-test
  (testing "batched [P,K] forward"
    (let [P 50
          log-alpha (mx/multiply (mx/scalar (- (js/Math.log 2)))
                                 (mx/ones [P K]))
          obs-vals (mx/add (mx/multiply (rng/uniform (rng/fresh-key) [P])
                                        (mx/scalar 6.0))
                           (mx/scalar -0.5))
          log-ep (let [lp0 (dc/dist-log-prob (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) obs-vals)
                       lp1 (dc/dist-log-prob (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)) obs-vals)]
                   (mx/stack [lp0 lp1] -1))
          mask (mx/ones [P])
          {:keys [log-alpha ll]} (hmm/hmm-update log-alpha log-ep mask)]
      (mx/eval! log-alpha)
      (mx/eval! ll)
      (is (= [P K] (mx/shape log-alpha)) "batched belief is [P,K]-shaped")
      (is (= [P] (mx/shape ll)) "batched LL is [P]-shaped"))))

(def hmm-step-fn
  (gen [obs log-ep]
    (let [z-prev (mx/scalar 0 mx/int32)
          z (trace :z (hmm/hmm-latent (mx/index log-trans 0) z-prev))
          _ (trace :obs (hmm/hmm-obs log-ep (mx/scalar 1.0)))]
      z)))

(deftest hmm-handler-middleware-test
  (testing "handler middleware"
    (let [obs (mx/scalar 4.5)
          log-ep (emission-log-probs obs)
          constraints (cm/set-value cm/EMPTY :obs obs)
          result (hmm/hmm-generate hmm-step-fn [obs log-ep] constraints
                                   :z log-trans 0 K (rng/fresh-key))]
      (is (some? result) "hmm-generate returns result")
      (mx/eval! (or (:hmm-ll result) (mx/scalar 0.0)))
      (let [ll (mx/item (or (:hmm-ll result) (mx/scalar 0.0)))]
        (is (js/isFinite ll) "LL is finite"))
      (is (some? (:hmm-belief result)) "hmm-belief exists"))))

(def hmm-step-batched
  (gen [log-ep-batched]
    (let [z-prev (mx/scalar 0 mx/int32)
          z (trace :z (hmm/hmm-latent (mx/index log-trans 0) z-prev))
          _ (trace :obs (hmm/hmm-obs log-ep-batched (mx/scalar 1.0)))]
      z)))

(deftest hmm-fold-test
  (testing "hmm-fold over sequence"
    (let [true-states [0 0 1 1 1]
          obs-seq (mapv (fn [s]
                          (let [mu (if (zero? s) 0.0 5.0)]
                            (mx/scalar (+ mu (* 0.3 (- (rand) 0.5))))))
                        true-states)
          T (count obs-seq)
          context-fn (fn [t]
                       (let [obs (nth obs-seq t)
                             log-ep (emission-log-probs obs)]
                         {:args [obs log-ep]
                          :constraints (cm/set-value cm/EMPTY :obs obs)}))
          {:keys [ll belief]} (hmm/hmm-fold hmm-step-fn :z log-trans 0 K T context-fn)]
      (mx/eval! ll)
      (is (= [] (mx/shape ll)) "fold LL is scalar")
      (is (js/isFinite (mx/item ll)) "fold LL is finite")
      (let [final-probs (mx/exp belief)
            p1 (mx/item (mx/index final-probs 1))]
        (is (> p1 0.8) "final belief favors state 1")))))

(deftest hmm-batched-fold-test
  (testing "batched hmm-fold (P elements)"
    (let [P 30
          T 5
          context-fn (fn [t]
                       (let [obs (mx/multiply (rng/uniform (rng/fresh-key (* 1000 t)) [P])
                                              (mx/scalar 2.0))
                             lp0 (dc/dist-log-prob (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) obs)
                             lp1 (dc/dist-log-prob (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)) obs)
                             log-ep (mx/stack [lp0 lp1] -1)]
                         {:args [log-ep]
                          :constraints (cm/set-value cm/EMPTY :obs obs)}))
          {:keys [ll belief]} (hmm/hmm-fold hmm-step-batched :z log-trans P K T context-fn)]
      (mx/eval! ll)
      (is (= [P] (mx/shape ll)) "batched fold LL is [P]-shaped")
      (is (= [P K] (mx/shape belief)) "batched fold belief is [P,K]-shaped"))))

(deftest hmm-composable-middleware-test
  (testing "composable middleware (wrap-analytical)"
    (let [dispatch (hmm/make-hmm-dispatch :z log-trans)
          transition (ana/wrap-analytical hnd/generate-transition dispatch)]
      (is (fn? transition) "wrap-analytical returns function"))))

(cljs.test/run-tests)
