(ns genmlx.ekf-test
  "Tests for EKF middleware."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as hnd]
            [genmlx.runtime :as rt]
            [genmlx.inference.ekf :as ekf]
            [genmlx.inference.kalman :as kal]
            [genmlx.inference.hmm-forward :as hmm]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test dynamics
;; ---------------------------------------------------------------------------

(def rho 0.8)
(def noise-std 0.3)

(defn linear-f [z] (mx/multiply (mx/scalar rho) z))
(defn tanh-f [z] (mx/tanh (mx/multiply (mx/scalar rho) z)))

(def obs-loading 2.0)
(def obs-base 1.0)
(defn linear-h [z] (mx/add (mx/scalar obs-base) (mx/multiply (mx/scalar obs-loading) z)))
(defn sigmoid-h [z] (mx/sigmoid z))

(deftest ekf-linearize-linear-test
  (testing "ekf-linearize on linear f(z) = 0.8z"
    (let [z0 (mx/scalar 2.0)
          [f-z0 A] (ekf/ekf-linearize linear-f z0)]
      (mx/eval! f-z0)
      (mx/eval! A)
      (is (h/close? 1.6 (mx/item f-z0) 1e-5) "f(z0) = 0.8 * 2.0 = 1.6")
      (is (h/close? 0.8 (mx/item A) 1e-5) "A = 0.8 (constant Jacobian)"))))

(deftest ekf-linearize-nonlinear-test
  (testing "ekf-linearize on tanh(0.8z)"
    (let [z0 (mx/scalar 1.5)
          [f-z0 A] (ekf/ekf-linearize tanh-f z0)]
      (mx/eval! f-z0)
      (mx/eval! A)
      (let [expected-f (js/Math.tanh (* rho 1.5))
            tanh-val (js/Math.tanh (* rho 1.5))
            expected-A (* rho (- 1.0 (* tanh-val tanh-val)))]
        (is (h/close? expected-f (mx/item f-z0) 1e-4) "f(z0) = tanh(0.8 * 1.5)")
        (is (h/close? expected-A (mx/item A) 1e-4) "A = 0.8 * (1 - tanh^2(1.2))")))))

(deftest ekf-linearize-batched-test
  (testing "ekf-linearize batched [P]-shaped"
    (let [P 20
          z0 (mx/multiply (rng/uniform (rng/fresh-key) [P]) (mx/scalar 4.0))
          [f-z0 A] (ekf/ekf-linearize tanh-f z0)]
      (mx/eval! f-z0)
      (mx/eval! A)
      (is (= [P] (mx/shape f-z0)) "f(z0) is [P]-shaped")
      (is (= [P] (mx/shape A)) "A is [P]-shaped")
      (let [a-min (mx/item (mx/amin A))
            a-max (mx/item (mx/amax A))]
        (is (and (> a-min 0) (<= a-max 0.8001)) "Jacobian values in (0, 0.8]")))))

(deftest ekf-predict-linear-test
  (testing "ekf-predict matches kalman-predict (linear)"
    (let [P 10
          belief {:mean (mx/multiply (rng/uniform (rng/fresh-key) [P]) (mx/scalar 3.0))
                  :var  (mx/add (mx/scalar 0.5)
                                (mx/multiply (rng/uniform (rng/fresh-key 1) [P]) (mx/scalar 1.0)))}
          q (mx/scalar noise-std)
          ekf-result (ekf/ekf-predict belief linear-f q)
          kal-result (kal/kalman-predict belief (mx/scalar rho) q)]
      (mx/eval! (:mean ekf-result))
      (mx/eval! (:var ekf-result))
      (mx/eval! (:mean kal-result))
      (mx/eval! (:var kal-result))
      (let [mean-diff (mx/item (mx/amax (mx/abs (mx/subtract (:mean ekf-result) (:mean kal-result)))))
            var-diff (mx/item (mx/amax (mx/abs (mx/subtract (:var ekf-result) (:var kal-result)))))]
        (is (h/close? 0.0 mean-diff 1e-5) "means match")
        (is (h/close? 0.0 var-diff 1e-5) "variances match")))))

(deftest ekf-predict-nonlinear-test
  (testing "ekf-predict with tanh dynamics"
    (let [belief {:mean (mx/scalar 2.0) :var (mx/scalar 1.0)}
          q (mx/scalar noise-std)
          result (ekf/ekf-predict belief tanh-f q)]
      (mx/eval! (:mean result))
      (mx/eval! (:var result))
      (let [expected-mean (js/Math.tanh (* rho 2.0))
            tanh-val (js/Math.tanh (* rho 2.0))
            A (* rho (- 1.0 (* tanh-val tanh-val)))
            expected-var (+ (* A A 1.0) (* noise-std noise-std))]
        (is (h/close? expected-mean (mx/item (:mean result)) 1e-4) "mean = tanh(0.8*2)")
        (is (h/close? expected-var (mx/item (:var result)) 1e-4) "var = A^2*var + Q^2")))))

(deftest ekf-update-linear-test
  (testing "ekf-update matches kalman-update (linear h)"
    (let [belief {:mean (mx/scalar 1.5) :var (mx/scalar 0.8)}
          obs (mx/scalar 5.2)
          r (mx/scalar 0.5)
          mask (mx/scalar 1.0)
          ekf-result (ekf/ekf-update belief obs linear-h r mask)
          kal-result (kal/kalman-update belief obs (mx/scalar obs-base) (mx/scalar obs-loading) r mask)]
      (mx/eval! (get-in ekf-result [:belief :mean]))
      (mx/eval! (get-in ekf-result [:belief :var]))
      (mx/eval! (:ll ekf-result))
      (mx/eval! (get-in kal-result [:belief :mean]))
      (mx/eval! (get-in kal-result [:belief :var]))
      (mx/eval! (:ll kal-result))
      (let [mean-diff (js/Math.abs (- (mx/item (get-in ekf-result [:belief :mean]))
                                      (mx/item (get-in kal-result [:belief :mean]))))
            var-diff (js/Math.abs (- (mx/item (get-in ekf-result [:belief :var]))
                                     (mx/item (get-in kal-result [:belief :var]))))
            ll-diff (js/Math.abs (- (mx/item (:ll ekf-result))
                                    (mx/item (:ll kal-result))))]
        (is (h/close? 0.0 mean-diff 1e-4) "means match")
        (is (h/close? 0.0 var-diff 1e-4) "variances match")
        (is (h/close? 0.0 ll-diff 1e-4) "LLs match")))))

(deftest ekf-update-nonlinear-test
  (testing "ekf-update with sigmoid observation"
    (let [belief {:mean (mx/scalar 0.5) :var (mx/scalar 1.0)}
          obs (mx/scalar 0.7)
          r (mx/scalar 0.1)
          mask (mx/scalar 1.0)
          {:keys [belief ll]} (ekf/ekf-update belief obs sigmoid-h r mask)]
      (mx/eval! (:mean belief))
      (mx/eval! (:var belief))
      (mx/eval! ll)
      (is (js/isFinite (mx/item ll)) "LL is finite")
      (is (< (mx/item (:var belief)) 1.0) "variance decreased"))))

(deftest ekf-missing-data-test
  (testing "missing data (mask=0)"
    (let [belief {:mean (mx/scalar 1.0) :var (mx/scalar 0.5)}
          obs (mx/scalar 99.0)
          mask (mx/scalar 0.0)
          {:keys [belief ll]} (ekf/ekf-update belief obs sigmoid-h (mx/scalar 0.1) mask)]
      (mx/eval! (:mean belief))
      (mx/eval! (:var belief))
      (mx/eval! ll)
      (is (h/close? 1.0 (mx/item (:mean belief)) 1e-6) "mean unchanged")
      (is (h/close? 0.5 (mx/item (:var belief)) 1e-6) "var unchanged")
      (is (h/close? 0.0 (mx/item ll) 1e-6) "LL = 0"))))

(def ekf-step-fn
  (gen [obs-val obs-fn-arg]
    (let [z (trace :z (ekf/ekf-latent tanh-f (mx/scalar 0.0) (mx/scalar noise-std)))
          _ (trace :obs (ekf/ekf-obs obs-fn-arg z (mx/scalar 0.5) (mx/scalar 1.0)))]
      z)))

(deftest ekf-handler-middleware-test
  (testing "handler middleware"
    (let [obs (mx/scalar 0.6)
          constraints (cm/set-value cm/EMPTY :obs obs)
          result (ekf/ekf-generate ekf-step-fn [obs linear-h] constraints
                                   :z 1 (rng/fresh-key))]
      (is (some? result) "ekf-generate returns result")
      (mx/eval! (or (:ekf-ll result) (mx/scalar 0.0)))
      (let [ll (mx/item (or (:ekf-ll result) (mx/scalar 0.0)))]
        (is (js/isFinite ll) "LL is finite"))
      (is (some? (:ekf-belief result)) "ekf-belief exists")
      (let [{:keys [mean var]} (:ekf-belief result)]
        (mx/eval! mean)
        (mx/eval! var)))))

(deftest ekf-fold-test
  (testing "ekf-fold over sequence"
    (let [obs-seq [(mx/scalar 0.2) (mx/scalar 0.5) (mx/scalar 0.7)
                   (mx/scalar 0.75) (mx/scalar 0.78)]
          T (count obs-seq)
          context-fn (fn [t]
                       (let [obs (nth obs-seq t)]
                         {:args [obs linear-h]
                          :constraints (cm/set-value cm/EMPTY :obs obs)}))
          {:keys [ll belief]} (ekf/ekf-fold ekf-step-fn :z 1 T context-fn)]
      (mx/eval! ll)
      (is (= [1] (mx/shape ll)) "fold LL shape is [1]")
      (is (js/isFinite (mx/item ll)) "fold LL is finite")
      (let [{:keys [mean var]} belief]
        (mx/eval! mean)
        (mx/eval! var)
        (is (js/isFinite (mx/item mean)) "final belief mean is finite")
        (is (pos? (mx/item var)) "final belief var > 0")))))

(def ekf-step-batched
  (gen [obs-val obs-fn-arg]
    (let [z (trace :z (ekf/ekf-latent tanh-f (mx/scalar 0.0) (mx/scalar noise-std)))
          _ (trace :obs (ekf/ekf-obs obs-fn-arg z (mx/scalar 0.5) (mx/scalar 1.0)))]
      z)))

(deftest ekf-batched-fold-test
  (testing "batched [P] ekf-fold"
    (let [P 30
          T 5
          context-fn (fn [t]
                       (let [obs (mx/multiply (rng/uniform (rng/fresh-key (* 1000 t)) [P])
                                              (mx/scalar 2.0))]
                         {:args [obs linear-h]
                          :constraints (cm/set-value cm/EMPTY :obs obs)}))
          {:keys [ll belief]} (ekf/ekf-fold ekf-step-batched :z P T context-fn)]
      (mx/eval! ll)
      (is (= [P] (mx/shape ll)) "batched LL is [P]-shaped")
      (is (= [P] (mx/shape (:mean belief))) "batched belief mean is [P]-shaped")
      (is (= [P] (mx/shape (:var belief))) "batched belief var is [P]-shaped"))))

(def kalman-step-fn
  (gen [obs-val]
    (let [z (trace :z (kal/kalman-latent (mx/scalar rho) (mx/scalar 0.0) (mx/scalar noise-std)))
          _ (trace :obs (kal/kalman-obs (mx/scalar obs-base) (mx/scalar obs-loading) z
                                        (mx/scalar 0.5) (mx/scalar 1.0)))]
      z)))

(def ekf-linear-step-fn
  (gen [obs-val obs-fn-arg]
    (let [z (trace :z (ekf/ekf-latent linear-f (mx/scalar 0.0) (mx/scalar noise-std)))
          _ (trace :obs (ekf/ekf-obs obs-fn-arg z (mx/scalar 0.5) (mx/scalar 1.0)))]
      z)))

(deftest ekf-kalman-linear-equivalence-test
  (testing "linear equivalence: EKF vs Kalman"
    (let [P 10
          T 8
          obs-data (mapv (fn [t] (mx/multiply (rng/uniform (rng/fresh-key (* 100 t)) [P])
                                               (mx/scalar 3.0)))
                         (range T))
          kal-context (fn [t]
                        {:args [(nth obs-data t)]
                         :constraints (cm/set-value cm/EMPTY :obs (nth obs-data t))})
          kal-ll (kal/kalman-fold kalman-step-fn :z P T kal-context)
          ekf-context (fn [t]
                        {:args [(nth obs-data t) linear-h]
                         :constraints (cm/set-value cm/EMPTY :obs (nth obs-data t))})
          {:keys [ll]} (ekf/ekf-fold ekf-linear-step-fn :z P T ekf-context)]
      (mx/eval! kal-ll)
      (mx/eval! ll)
      (let [diff (mx/item (mx/amax (mx/abs (mx/subtract ll kal-ll))))]
        (is (h/close? 0.0 diff 1e-3) "EKF LL matches Kalman LL for linear dynamics")))))

(deftest ekf-hmm-composable-middleware-test
  (testing "composable middleware (EKF + HMM)"
    (let [log-trans (mx/array [[(js/Math.log 0.9) (js/Math.log 0.1)]
                                [(js/Math.log 0.1) (js/Math.log 0.9)]])
          ekf-dispatch (ekf/make-ekf-dispatch :z)
          hmm-dispatch (hmm/make-hmm-dispatch :regime log-trans)
          transition (ana/compose-middleware hnd/generate-transition ekf-dispatch hmm-dispatch)]
      (is (fn? transition) "compose-middleware returns function"))))

(deftest ekf-standard-handler-fallback-test
  (testing "standard handler fallback"
    (let [d (ekf/ekf-latent tanh-f (mx/scalar 1.0) (mx/scalar 0.3))
          sample (dc/dist-sample d (rng/fresh-key))
          lp (dc/dist-log-prob d sample)]
      (mx/eval! sample)
      (mx/eval! lp)
      (is (js/isFinite (mx/item sample)) "ekf-latent samples under standard handler")
      (is (js/isFinite (mx/item lp)) "ekf-latent scores under standard handler"))

    (let [d (ekf/ekf-obs sigmoid-h (mx/scalar 0.5) (mx/scalar 0.1) (mx/scalar 1.0))
          sample (dc/dist-sample d (rng/fresh-key 1))
          lp (dc/dist-log-prob d (mx/scalar 0.6))]
      (mx/eval! sample)
      (mx/eval! lp)
      (is (js/isFinite (mx/item sample)) "ekf-obs samples under standard handler")
      (is (js/isFinite (mx/item lp)) "ekf-obs scores under standard handler"))))

(cljs.test/run-tests)
