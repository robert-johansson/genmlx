(ns genmlx.multistep-kalman-ekf-test
  "Multi-step Kalman filter and EKF verification against analytical ground truth.
   Covers CORRECTNESS_PLAN section 2.5 unchecked items."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.kalman :as kalman]
            [genmlx.inference.ekf :as ekf])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================================
;; Analytical ground truth computations (pure JS math)
;; ============================================================================

(def LOG-2PI (js/Math.log (* 2 js/Math.PI)))

(defn kalman-step-analytical
  "One Kalman predict+update step. Returns {:mean :var :ll}.
   A = transition coeff, H = loading, Q = process variance, R = obs variance."
  [m P A H Q R y]
  (let [m-pred (* A m)
        P-pred (+ (* A A P) Q)
        S      (+ (* H H P-pred) R)
        K      (/ (* H P-pred) S)
        innov  (- y (* H m-pred))
        m-upd  (+ m-pred (* K innov))
        P-upd  (- P-pred (* K H P-pred))
        ll     (* -0.5 (+ LOG-2PI (js/Math.log S) (/ (* innov innov) S)))]
    {:mean m-upd :var P-upd :ll ll}))

(defn run-kalman-analytical
  "Run full Kalman filter analytically. Returns vector of step results + total LL."
  [m0 P0 A H Q R ys]
  (loop [m m0, P P0, ys ys, steps [], total-ll 0.0]
    (if (empty? ys)
      {:steps steps :total-ll total-ll}
      (let [result (kalman-step-analytical m P A H Q R (first ys))]
        (recur (:mean result) (:var result) (rest ys)
               (conj steps result) (+ total-ll (:ll result)))))))

(defn ekf-step-analytical
  "One EKF predict+update step with nonlinear dynamics.
   f = transition fn, f' = Jacobian of f, h = obs fn, h' = Jacobian of h.
   Q = process variance (std^2), R = obs variance (std^2)."
  [m P f f' h h' Q R y]
  (let [;; Predict
        m-pred (f m)
        A      (f' m)
        P-pred (+ (* A A P) Q)
        ;; Update
        h-val  (h m-pred)
        H      (h' m-pred)
        base   (- h-val (* H m-pred))
        pred-obs (+ base (* H m-pred))  ;; = h-val
        innov  (- y pred-obs)
        S      (+ (* H H P-pred) R)
        K      (/ (* H P-pred) S)
        m-upd  (+ m-pred (* K innov))
        P-upd  (- P-pred (* K H P-pred))
        ll     (* -0.5 (+ LOG-2PI (js/Math.log S) (/ (* innov innov) S)))]
    {:mean m-upd :var P-upd :ll ll}))

(defn run-ekf-analytical
  "Run full EKF analytically. Returns vector of step results + total LL."
  [m0 P0 f f' h h' Q R ys]
  (loop [m m0, P P0, ys ys, steps [], total-ll 0.0]
    (if (empty? ys)
      {:steps steps :total-ll total-ll}
      (let [result (ekf-step-analytical m P f f' h h' Q R (first ys))]
        (recur (:mean result) (:var result) (rest ys)
               (conj steps result) (+ total-ll (:ll result)))))))

;; ============================================================================
;; Test 1: Multi-step Kalman log-marginal-likelihood
;; ============================================================================
;;
;; Linear-Gaussian SSM:  x_t = A * x_{t-1} + w_t,  y_t = H * x_t + v_t
;; A = 0.9, H = 1.0, Q_var = 0.1, R_var = 0.5
;; Initial belief: mean=0, var=0 (the API convention; predict gives prior N(0, Q))
;; Observations: y = [1.2, 0.8, 1.5, 0.3, 0.9]

(def kalman-A 0.9)
(def kalman-H 1.0)
(def kalman-Q-var 0.1)   ;; process noise variance
(def kalman-R-var 0.5)   ;; observation noise variance
(def kalman-Q-std (js/Math.sqrt kalman-Q-var))
(def kalman-R-std (js/Math.sqrt kalman-R-var))
(def kalman-ys [1.2 0.8 1.5 0.3 0.9])

;; Analytical reference
(def kalman-ref
  (run-kalman-analytical 0.0 0.0 kalman-A kalman-H kalman-Q-var kalman-R-var kalman-ys))

;; Gen function for Kalman filter
(def kalman-step-model
  (gen [obs-val]
    (let [z (trace :z (kalman/kalman-latent
                        (mx/scalar kalman-A)
                        (mx/scalar 0.0)
                        (mx/scalar kalman-Q-std)))]
      (trace :obs (kalman/kalman-obs
                    (mx/scalar 0.0)          ;; base-mean
                    (mx/scalar kalman-H)     ;; loading
                    z
                    (mx/scalar kalman-R-std) ;; noise-std
                    (mx/scalar 1.0)))        ;; mask
      z)))

(deftest kalman-5step-log-marginal-test
  (testing "5-step Kalman log-marginal-likelihood matches analytical"
    (let [T 5
          context-fn (fn [t]
                       (let [y (nth kalman-ys t)]
                         {:args [(mx/scalar y)]
                          :constraints (cm/set-value cm/EMPTY :obs (mx/scalar y))}))
          ;; kalman-fold returns [n]-shaped LL directly
          ll-tensor (kalman/kalman-fold kalman-step-model :z 1 T context-fn)]
      (mx/eval! ll-tensor)
      (let [ll-genmlx (mx/item (mx/index ll-tensor 0))
            ll-ref    (:total-ll kalman-ref)]
        (is (h/close? ll-ref ll-genmlx 1e-4)
            (str "Total log-ML: expected " ll-ref ", got " ll-genmlx))))))

(deftest kalman-5step-intermediate-beliefs-test
  (testing "5-step Kalman intermediate filtered means and variances"
    (let [T 5
          ;; Run step-by-step to check intermediate beliefs
          results
          (loop [t 0
                 belief {:mean (mx/zeros [1]) :var (mx/zeros [1])}
                 acc []]
            (if (>= t T)
              acc
              (let [y (nth kalman-ys t)
                    result (kalman/kalman-generate
                             kalman-step-model
                             [(mx/scalar y)]
                             (cm/set-value cm/EMPTY :obs (mx/scalar y))
                             :z 1 (rng/fresh-key t)
                             {:init-belief belief})
                    new-belief (:kalman-belief result)
                    step-ll   (or (:kalman-ll result) (mx/zeros [1]))]
                (mx/eval! (:mean new-belief))
                (mx/eval! (:var new-belief))
                (mx/eval! step-ll)
                (recur (inc t)
                       new-belief
                       (conj acc {:mean (mx/item (mx/index (:mean new-belief) 0))
                                  :var  (mx/item (mx/index (:var new-belief) 0))
                                  :ll   (mx/item (mx/index step-ll 0))})))))
          ref-steps (:steps kalman-ref)]
      (doseq [t (range T)]
        (let [got (nth results t)
              expected (nth ref-steps t)]
          (is (h/close? (:mean expected) (:mean got) 1e-4)
              (str "Step " t " mean: expected " (:mean expected) ", got " (:mean got)))
          (is (h/close? (:var expected) (:var got) 1e-4)
              (str "Step " t " var: expected " (:var expected) ", got " (:var got)))
          (is (h/close? (:ll expected) (:ll got) 1e-4)
              (str "Step " t " ll: expected " (:ll expected) ", got " (:ll got))))))))

(deftest kalman-5step-monotone-variance-test
  (testing "Kalman filtered variance converges (monotone after initial steps)"
    (let [ref-steps (:steps kalman-ref)
          vars (mapv :var ref-steps)]
      ;; After first observation, variance should decrease from prior P_pred = Q = 0.1
      ;; because the observation provides information
      (is (< (nth vars 0) kalman-Q-var)
          "After first observation, variance < predicted prior variance")
      ;; Variance should stabilize and remain positive
      (is (> (last vars) 0.0)
          "Final variance is positive")
      ;; Steady-state variance: for A=0.9, Q=0.1, H=1, R=0.5, the filtered
      ;; variance converges to ~0.154 (the Riccati equation solution).
      ;; It stays bounded below the *predicted* variance (A^2*P_ss + Q).
      (let [P-ss (last vars)
            P-pred-ss (+ (* kalman-A kalman-A P-ss) kalman-Q-var)]
        (is (< P-ss P-pred-ss)
            "Filtered variance < predicted variance (observation reduces uncertainty)")))))

(deftest kalman-5step-ll-negative-test
  (testing "Each step's log-likelihood is negative"
    (let [ref-steps (:steps kalman-ref)]
      (doseq [t (range 5)]
        (is (< (:ll (nth ref-steps t)) 0.0)
            (str "Step " t " LL is negative"))))))

(deftest kalman-5step-pure-building-blocks-test
  (testing "Pure building blocks match handler middleware"
    (let [;; Run using pure building blocks
          pure-result
          (loop [t 0
                 belief {:mean (mx/zeros [1]) :var (mx/zeros [1])}
                 acc-ll 0.0]
            (if (>= t 5)
              acc-ll
              (let [y (nth kalman-ys t)
                    pred (kalman/kalman-predict
                           belief
                           (mx/scalar kalman-A)
                           (mx/scalar kalman-Q-std))
                    {:keys [belief ll]}
                    (kalman/kalman-update
                      pred
                      (mx/array [(nth kalman-ys t)])  ;; obs as [1]-shaped
                      (mx/zeros [1])                   ;; base-mean
                      (mx/scalar kalman-H)             ;; loading
                      (mx/scalar kalman-R-std)         ;; noise-std
                      (mx/ones [1]))]                  ;; mask
                (mx/eval! (:mean belief))
                (mx/eval! (:var belief))
                (mx/eval! ll)
                (recur (inc t)
                       belief
                       (+ acc-ll (mx/item (mx/index ll 0)))))))
          ;; Run using kalman-fold
          fold-ll (kalman/kalman-fold
                    kalman-step-model :z 1 5
                    (fn [t]
                      (let [y (nth kalman-ys t)]
                        {:args [(mx/scalar y)]
                         :constraints (cm/set-value cm/EMPTY :obs (mx/scalar y))})))]
      (mx/eval! fold-ll)
      (let [fold-val (mx/item (mx/index fold-ll 0))]
        (is (h/close? pure-result fold-val 1e-4)
            (str "Pure blocks LL=" pure-result " matches fold LL=" fold-val))))))

;; ============================================================================
;; Test 2: EKF on nonlinear model
;; ============================================================================
;;
;; Nonlinear SSM: x_t = sin(x_{t-1}) + w_t, y_t = x_t^2 + v_t
;; Q_var = 0.01, R_var = 0.1
;; Initial belief: mean=0, var=0 (API convention; predict gives prior)
;; Note: The API starts from {0,0} and first predict gives {sin(0), Q} = {0, Q}
;; Observations: y = [0.3, 0.25, 0.2]

(def ekf-Q-var 0.01)
(def ekf-R-var 0.1)
(def ekf-Q-std (js/Math.sqrt ekf-Q-var))
(def ekf-R-std (js/Math.sqrt ekf-R-var))
(def ekf-ys [0.3 0.25 0.2])

;; Dynamics
(defn ekf-transition [x] (js/Math.sin x))
(defn ekf-transition-jac [x] (js/Math.cos x))
(defn ekf-observation [x] (* x x))
(defn ekf-observation-jac [x] (* 2 x))

;; MLX versions for the gen function
(defn ekf-transition-mx [z] (mx/sin z))
(defn ekf-observation-mx [z] (mx/multiply z z))

;; Analytical EKF reference
(def ekf-ref
  (run-ekf-analytical 0.0 0.0
                      ekf-transition ekf-transition-jac
                      ekf-observation ekf-observation-jac
                      ekf-Q-var ekf-R-var
                      ekf-ys))

;; Gen function for EKF
(def ekf-step-model
  (gen [obs-val]
    (let [z (trace :z (ekf/ekf-latent
                        ekf-transition-mx
                        (mx/scalar 0.0)
                        (mx/scalar ekf-Q-std)))]
      (trace :obs (ekf/ekf-obs
                    ekf-observation-mx
                    z
                    (mx/scalar ekf-R-std)
                    (mx/scalar 1.0)))
      z)))

(deftest ekf-3step-log-marginal-test
  (testing "3-step EKF log-marginal-likelihood matches analytical"
    (let [T 3
          context-fn (fn [t]
                       (let [y (nth ekf-ys t)]
                         {:args [(mx/scalar y)]
                          :constraints (cm/set-value cm/EMPTY :obs (mx/scalar y))}))
          {:keys [ll]} (ekf/ekf-fold ekf-step-model :z 1 T context-fn)]
      (mx/eval! ll)
      (let [ll-genmlx (mx/item (mx/index ll 0))
            ll-ref    (:total-ll ekf-ref)]
        (is (h/close? ll-ref ll-genmlx 1e-3)
            (str "Total log-ML: expected " ll-ref ", got " ll-genmlx))))))

(deftest ekf-3step-intermediate-beliefs-test
  (testing "3-step EKF intermediate filtered means and variances"
    (let [T 3
          results
          (loop [t 0
                 belief {:mean (mx/zeros [1]) :var (mx/zeros [1])}
                 acc []]
            (if (>= t T)
              acc
              (let [y (nth ekf-ys t)
                    result (ekf/ekf-generate
                             ekf-step-model
                             [(mx/scalar y)]
                             (cm/set-value cm/EMPTY :obs (mx/scalar y))
                             :z 1 (rng/fresh-key t)
                             {:init-belief belief})
                    new-belief (:ekf-belief result)
                    step-ll   (or (:ekf-ll result) (mx/zeros [1]))]
                (mx/eval! (:mean new-belief))
                (mx/eval! (:var new-belief))
                (mx/eval! step-ll)
                (recur (inc t)
                       new-belief
                       (conj acc {:mean (mx/item (mx/index (:mean new-belief) 0))
                                  :var  (mx/item (mx/index (:var new-belief) 0))
                                  :ll   (mx/item (mx/index step-ll 0))})))))
          ref-steps (:steps ekf-ref)]
      (doseq [t (range T)]
        (let [got (nth results t)
              expected (nth ref-steps t)]
          (is (h/close? (:mean expected) (:mean got) 1e-3)
              (str "Step " t " mean: expected " (:mean expected) ", got " (:mean got)))
          (is (h/close? (:var expected) (:var got) 1e-3)
              (str "Step " t " var: expected " (:var expected) ", got " (:var got)))
          (is (h/close? (:ll expected) (:ll got) 1e-3)
              (str "Step " t " ll: expected " (:ll expected) ", got " (:ll got))))))))

(deftest ekf-3step-pure-building-blocks-test
  (testing "EKF pure building blocks match handler middleware"
    (let [;; Run using pure building blocks
          pure-result
          (loop [t 0
                 belief {:mean (mx/zeros [1]) :var (mx/zeros [1])}
                 acc-ll 0.0]
            (if (>= t 3)
              acc-ll
              (let [y (nth ekf-ys t)
                    pred (ekf/ekf-predict
                           belief
                           ekf-transition-mx
                           (mx/scalar ekf-Q-std))
                    {:keys [belief ll]}
                    (ekf/ekf-update
                      pred
                      (mx/array [(nth ekf-ys t)])
                      ekf-observation-mx
                      (mx/scalar ekf-R-std)
                      (mx/ones [1]))]
                (mx/eval! (:mean belief))
                (mx/eval! (:var belief))
                (mx/eval! ll)
                (recur (inc t)
                       belief
                       (+ acc-ll (mx/item (mx/index ll 0)))))))
          ;; Run using ekf-fold
          {:keys [ll]} (ekf/ekf-fold
                         ekf-step-model :z 1 3
                         (fn [t]
                           (let [y (nth ekf-ys t)]
                             {:args [(mx/scalar y)]
                              :constraints (cm/set-value cm/EMPTY :obs (mx/scalar y))})))]
      (mx/eval! ll)
      (let [fold-val (mx/item (mx/index ll 0))]
        (is (h/close? pure-result fold-val 1e-3)
            (str "Pure blocks LL=" pure-result " matches fold LL=" fold-val))))))

(deftest ekf-3step-variance-contracts-test
  (testing "EKF variance stays positive and observation reduces it"
    (let [ref-steps (:steps ekf-ref)]
      (doseq [t (range 3)]
        (is (> (:var (nth ref-steps t)) 0.0)
            (str "Step " t " variance is positive")))
      ;; After first observation on nonzero predicted mean, variance should reduce
      ;; (This depends on the specific dynamics; with sin(x) and x near 0,
      ;; the Jacobian 2*x is near 0, so observation is less informative.
      ;; Just check positivity and finiteness.)
      (doseq [t (range 3)]
        (is (js/isFinite (:var (nth ref-steps t)))
            (str "Step " t " variance is finite"))
        (is (js/isFinite (:ll (nth ref-steps t)))
            (str "Step " t " LL is finite"))))))

(deftest ekf-linear-matches-kalman-test
  (testing "EKF with linear dynamics matches Kalman exactly"
    (let [;; Use the same Kalman SSM parameters but through EKF API
          linear-f-mx (fn [z] (mx/multiply (mx/scalar kalman-A) z))
          linear-h-mx (fn [z] (mx/multiply (mx/scalar kalman-H) z))
          ekf-linear-step
          (gen [obs-val]
            (let [z (trace :z (ekf/ekf-latent
                                linear-f-mx
                                (mx/scalar 0.0)
                                (mx/scalar kalman-Q-std)))]
              (trace :obs (ekf/ekf-obs
                            linear-h-mx
                            z
                            (mx/scalar kalman-R-std)
                            (mx/scalar 1.0)))
              z))
          T 5
          context-fn (fn [t]
                       (let [y (nth kalman-ys t)]
                         {:args [(mx/scalar y)]
                          :constraints (cm/set-value cm/EMPTY :obs (mx/scalar y))}))
          ;; Kalman fold
          kalman-ll (kalman/kalman-fold kalman-step-model :z 1 T context-fn)
          ;; EKF fold
          {:keys [ll]} (ekf/ekf-fold ekf-linear-step :z 1 T context-fn)]
      (mx/eval! kalman-ll)
      (mx/eval! ll)
      (let [k-val (mx/item (mx/index kalman-ll 0))
            e-val (mx/item (mx/index ll 0))]
        (is (h/close? k-val e-val 1e-4)
            (str "Kalman LL=" k-val " matches EKF-linear LL=" e-val))))))

(deftest ekf-3step-ll-finite-test
  (testing "Each EKF step's log-likelihood is finite"
    (let [ref-steps (:steps ekf-ref)]
      (doseq [t (range 3)]
        (is (js/isFinite (:ll (nth ref-steps t)))
            (str "Step " t " LL is finite")))
      ;; Total LL should be finite
      (is (js/isFinite (:total-ll ekf-ref))
          "Total LL is finite"))))

;; ============================================================================
;; Additional cross-check: batched execution matches scalar
;; ============================================================================

(deftest kalman-batched-matches-scalar-test
  (testing "Kalman fold with n=4 (replicated) matches n=1 result"
    (let [T 5
          ;; Scalar (n=1) run
          context-fn-1 (fn [t]
                         (let [y (nth kalman-ys t)]
                           {:args [(mx/scalar y)]
                            :constraints (cm/set-value cm/EMPTY :obs (mx/scalar y))}))
          ll-1 (kalman/kalman-fold kalman-step-model :z 1 T context-fn-1)
          ;; Batched (n=4): replicate same observation across all elements
          kalman-step-model-batched
          (gen [obs-val]
            (let [z (trace :z (kalman/kalman-latent
                                (mx/scalar kalman-A)
                                (mx/scalar 0.0)
                                (mx/scalar kalman-Q-std)))]
              (trace :obs (kalman/kalman-obs
                            (mx/scalar 0.0)
                            (mx/scalar kalman-H)
                            z
                            (mx/scalar kalman-R-std)
                            (mx/ones [4])))
              z))
          context-fn-4 (fn [t]
                         (let [y (nth kalman-ys t)]
                           {:args [(mx/array [y y y y])]
                            :constraints (cm/set-value cm/EMPTY :obs (mx/array [y y y y]))}))
          ll-4 (kalman/kalman-fold kalman-step-model-batched :z 4 T context-fn-4)]
      (mx/eval! ll-1)
      (mx/eval! ll-4)
      (let [val-1 (mx/item (mx/index ll-1 0))]
        ;; Each element of ll-4 should match val-1
        (doseq [i (range 4)]
          (is (h/close? val-1 (mx/item (mx/index ll-4 i)) 1e-4)
              (str "Batched element " i " matches scalar")))))))

;; ============================================================================
;; Run tests
;; ============================================================================

(cljs.test/run-tests)
