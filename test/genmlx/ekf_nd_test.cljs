(ns genmlx.ekf-nd-test
  "Multi-dimensional EKF middleware tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as hnd]
            [genmlx.runtime :as rt]
            [genmlx.inference.ekf-nd :as ekf-nd]
            [genmlx.inference.ekf :as ekf]
            [genmlx.inference.kalman :as kal]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test dynamics and observations
;; ---------------------------------------------------------------------------

(def rho-a 0.8)
(def rho-b 0.6)
(def q 0.3)

(defn linear-fa [z] (mx/multiply (mx/scalar rho-a) z))
(defn linear-fb [z] (mx/multiply (mx/scalar rho-b) z))
(defn tanh-f [z] (mx/tanh (mx/multiply (mx/scalar rho-a) z)))

(deftest ekf-nd-latent-standard-handler-test
  (testing "ekf-nd-latent under standard handler"
    (let [d (ekf-nd/ekf-nd-latent tanh-f (mx/scalar 1.0) (mx/scalar q))
          sample (dc/dist-sample d (rng/fresh-key))
          lp (dc/dist-log-prob d sample)]
      (mx/eval! sample)
      (mx/eval! lp)
      (is (js/isFinite (mx/item sample)) "samples are finite")
      (is (js/isFinite (mx/item lp)) "log-prob is finite"))))

(deftest ekf-nd-obs-standard-handler-test
  (testing "ekf-nd-obs under standard handler"
    (let [obs-fn (fn [{:keys [za zb]}]
                   (mx/add (mx/multiply (mx/scalar 2.0) (mx/sigmoid za))
                           (mx/multiply (mx/scalar 1.0) (mx/sigmoid zb))))
          vals {:za (mx/scalar 0.5) :zb (mx/scalar -0.3)}
          d (ekf-nd/ekf-nd-obs obs-fn vals (mx/scalar 0.1) (mx/scalar 1.0))
          sample (dc/dist-sample d (rng/fresh-key 1))
          lp (dc/dist-log-prob d (mx/scalar 1.5))]
      (mx/eval! sample)
      (mx/eval! lp)
      (is (js/isFinite (mx/item sample)) "samples are finite")
      (is (js/isFinite (mx/item lp)) "log-prob is finite"))))

(deftest ekf-nd-predict-1d-match-test
  (testing "N=1 predict matches 1D EKF"
    (let [addrs [:z]
          means {:z (mx/scalar 1.5)}
          covs {[:z :z] (mx/scalar 0.8)}
          [val new-means new-covs] (ekf-nd/ekf-nd-predict-one addrs :z means covs tanh-f (mx/scalar q))
          belief {:mean (mx/scalar 1.5) :var (mx/scalar 0.8)}
          ekf1 (ekf/ekf-predict belief tanh-f (mx/scalar q))]
      (mx/eval! val)
      (mx/eval! (get new-means :z))
      (mx/eval! (get new-covs [:z :z]))
      (mx/eval! (:mean ekf1))
      (mx/eval! (:var ekf1))
      (let [mean-diff (js/Math.abs (- (mx/item val) (mx/item (:mean ekf1))))
            var-diff (js/Math.abs (- (mx/item (get new-covs [:z :z])) (mx/item (:var ekf1))))]
        (is (h/close? 0.0 mean-diff 1e-5) "means match")
        (is (h/close? 0.0 var-diff 1e-5) "variances match")))))

(deftest ekf-nd-predict-2d-cross-cov-test
  (testing "N=2 predict: cross-covariance"
    (let [addrs [:za :zb]
          means {:za (mx/scalar 1.0) :zb (mx/scalar -0.5)}
          covs {[:za :za] (mx/scalar 1.0)
                [:za :zb] (mx/scalar 0.3)
                [:zb :zb] (mx/scalar 0.8)}
          [_ means1 covs1] (ekf-nd/ekf-nd-predict-one addrs :za means covs linear-fa (mx/scalar q))
          [_ means2 covs2] (ekf-nd/ekf-nd-predict-one addrs :zb means1 covs1 linear-fb (mx/scalar q))]
      (mx/eval! (get means2 :za))
      (mx/eval! (get means2 :zb))
      (mx/eval! (get covs2 [:za :za]))
      (mx/eval! (get covs2 [:za :zb]))
      (mx/eval! (get covs2 [:zb :zb]))
      (let [exp-paa (+ (* rho-a rho-a 1.0) (* q q))
            exp-pbb (+ (* rho-b rho-b 0.8) (* q q))
            exp-pab (* rho-a rho-b 0.3)]
        (is (h/close? exp-paa (mx/item (get covs2 [:za :za])) 1e-5) "P_aa")
        (is (h/close? exp-pbb (mx/item (get covs2 [:zb :zb])) 1e-5) "P_bb")
        (is (h/close? exp-pab (mx/item (get covs2 [:za :zb])) 1e-5) "P_ab (cross-cov)")
        (is (h/close? (* rho-a 1.0) (mx/item (get means2 :za)) 1e-5) "mu_a = rho_a * 1.0")
        (is (h/close? (* rho-b -0.5) (mx/item (get means2 :zb)) 1e-5) "mu_b = rho_b * -0.5")))))

(deftest ekf-nd-update-1d-match-test
  (testing "N=1 update matches 1D EKF"
    (let [addrs [:z]
          means {:z (mx/scalar 1.5)}
          covs {[:z :z] (mx/scalar 0.8)}
          obs (mx/scalar 4.0)
          noise-std (mx/scalar 0.5)
          mask (mx/scalar 1.0)
          obs-fn (fn [{:keys [z]}] (mx/add (mx/scalar 1.0) (mx/multiply (mx/scalar 2.0) z)))
          nd-r (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)
          ekf-r (ekf/ekf-update {:mean (mx/scalar 1.5) :var (mx/scalar 0.8)}
                                 obs
                                 (fn [z] (mx/add (mx/scalar 1.0) (mx/multiply (mx/scalar 2.0) z)))
                                 noise-std mask)]
      (mx/eval! (get (:means nd-r) :z))
      (mx/eval! (get (:covs nd-r) [:z :z]))
      (mx/eval! (:ll nd-r))
      (mx/eval! (get-in ekf-r [:belief :mean]))
      (mx/eval! (get-in ekf-r [:belief :var]))
      (mx/eval! (:ll ekf-r))
      (let [mean-diff (js/Math.abs (- (mx/item (get (:means nd-r) :z))
                                      (mx/item (get-in ekf-r [:belief :mean]))))
            var-diff (js/Math.abs (- (mx/item (get (:covs nd-r) [:z :z]))
                                     (mx/item (get-in ekf-r [:belief :var]))))
            ll-diff (js/Math.abs (- (mx/item (:ll nd-r)) (mx/item (:ll ekf-r))))]
        (is (h/close? 0.0 mean-diff 1e-4) "means match")
        (is (h/close? 0.0 var-diff 1e-4) "variances match")
        (is (h/close? 0.0 ll-diff 1e-4) "LLs match")))))

(deftest ekf-nd-update-2d-multi-loading-test
  (testing "N=2 update: observation loads on both latents"
    (let [addrs [:za :zb]
          means {:za (mx/scalar 1.0) :zb (mx/scalar -0.5)}
          covs {[:za :za] (mx/scalar 1.0)
                [:za :zb] (mx/scalar 0.0)
                [:zb :zb] (mx/scalar 0.8)}
          obs (mx/scalar 2.0)
          noise-std (mx/scalar 0.3)
          mask (mx/scalar 1.0)
          obs-fn (fn [{:keys [za zb]}]
                   (mx/add (mx/multiply (mx/scalar 2.0) za)
                           (mx/multiply (mx/scalar 3.0) zb)))
          {:keys [means covs ll]} (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)]
      (mx/eval! (get means :za))
      (mx/eval! (get means :zb))
      (mx/eval! (get covs [:za :za]))
      (mx/eval! (get covs [:za :zb]))
      (mx/eval! (get covs [:zb :zb]))
      (mx/eval! ll)
      (let [pred (+ (* 2.0 1.0) (* 3.0 -0.5))
            innov (- 2.0 pred)
            PH-a 2.0
            PH-b 2.4
            S (+ (* 2.0 PH-a) (* 3.0 PH-b) 0.09)
            K-a (/ PH-a S)
            K-b (/ PH-b S)
            exp-za (+ 1.0 (* K-a innov))
            exp-zb (+ -0.5 (* K-b innov))]
        (is (h/close? exp-za (mx/item (get means :za)) 1e-3) "mu_a updated")
        (is (h/close? exp-zb (mx/item (get means :zb)) 1e-3) "mu_b updated")
        (is (js/isFinite (mx/item ll)) "LL is finite")
        (let [pab (mx/item (get covs [:za :zb]))]
          (is (not (zero? pab)) "cross-cov nonzero after multi-loading obs"))))))

(deftest ekf-nd-missing-data-test
  (testing "missing data (mask=0)"
    (let [addrs [:za :zb]
          means {:za (mx/scalar 1.0) :zb (mx/scalar -0.5)}
          covs {[:za :za] (mx/scalar 1.0)
                [:za :zb] (mx/scalar 0.3)
                [:zb :zb] (mx/scalar 0.8)}
          obs (mx/scalar 99.0)
          noise-std (mx/scalar 0.3)
          mask (mx/scalar 0.0)
          obs-fn (fn [{:keys [za zb]}]
                   (mx/add (mx/multiply (mx/scalar 2.0) za) zb))
          {:keys [means covs ll]} (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)]
      (mx/eval! (get means :za))
      (mx/eval! (get means :zb))
      (mx/eval! (get covs [:za :za]))
      (mx/eval! (get covs [:za :zb]))
      (mx/eval! (get covs [:zb :zb]))
      (mx/eval! ll)
      (is (h/close? 1.0 (mx/item (get means :za)) 1e-6) "za unchanged")
      (is (h/close? -0.5 (mx/item (get means :zb)) 1e-6) "zb unchanged")
      (is (h/close? 1.0 (mx/item (get covs [:za :za])) 1e-6) "P_aa unchanged")
      (is (h/close? 0.3 (mx/item (get covs [:za :zb])) 1e-6) "P_ab unchanged")
      (is (h/close? 0.8 (mx/item (get covs [:zb :zb])) 1e-6) "P_bb unchanged")
      (is (h/close? 0.0 (mx/item ll) 1e-6) "LL = 0"))))

(deftest ekf-nd-nonlinear-obs-test
  (testing "nonlinear observation (sigmoid)"
    (let [addrs [:za :zb]
          means {:za (mx/scalar 0.5) :zb (mx/scalar -0.3)}
          covs {[:za :za] (mx/scalar 1.0)
                [:za :zb] (mx/scalar 0.0)
                [:zb :zb] (mx/scalar 1.0)}
          obs (mx/scalar 1.5)
          noise-std (mx/scalar 0.2)
          mask (mx/scalar 1.0)
          obs-fn (fn [{:keys [za zb]}]
                   (mx/add (mx/multiply (mx/scalar 2.0) (mx/sigmoid za))
                           (mx/sigmoid zb)))
          {:keys [means covs ll]} (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)]
      (mx/eval! (get means :za))
      (mx/eval! (get means :zb))
      (mx/eval! ll)
      (is (not= 0.5 (mx/item (get means :za))) "za updated")
      (is (not= -0.3 (mx/item (get means :zb))) "zb updated")
      (is (js/isFinite (mx/item ll)) "LL is finite"))))

;; ---------------------------------------------------------------------------
;; Handler middleware
;; ---------------------------------------------------------------------------

(def step-2d
  (gen [obs-a-val obs-b-val]
    (let [za (trace :za (ekf-nd/ekf-nd-latent linear-fa (mx/scalar 0.0) (mx/scalar q)))
          zb (trace :zb (ekf-nd/ekf-nd-latent linear-fb (mx/scalar 0.0) (mx/scalar q)))
          latents {:za za :zb zb}
          _ (trace :obs-a (ekf-nd/ekf-nd-obs
                            (fn [{:keys [za]}]
                              (mx/add (mx/scalar 1.0) (mx/multiply (mx/scalar 2.0) za)))
                            latents (mx/scalar 0.5) (mx/scalar 1.0)))
          _ (trace :obs-b (ekf-nd/ekf-nd-obs
                            (fn [{:keys [za zb]}]
                              (mx/add (mx/multiply (mx/scalar 1.5) za)
                                      (mx/multiply (mx/scalar 0.8) zb)))
                            latents (mx/scalar 0.3) (mx/scalar 1.0)))]
      {:za za :zb zb})))

(deftest ekf-nd-generate-test
  (testing "handler middleware: ekf-nd-generate"
    (let [constraints (-> cm/EMPTY
                          (cm/set-value :obs-a (mx/scalar 3.0))
                          (cm/set-value :obs-b (mx/scalar 1.0)))
          result (ekf-nd/ekf-nd-generate step-2d [nil nil] constraints
                                         [:za :zb] 1 (rng/fresh-key))]
      (is (some? (:ekf-nd-means result)) "result has ekf-nd-means")
      (is (some? (:ekf-nd-covs result)) "result has ekf-nd-covs")
      (let [ll (or (:ekf-nd-ll result) (mx/scalar 0.0))]
        (mx/eval! ll)
        (is (js/isFinite (mx/item ll)) "LL is finite")))))

(deftest ekf-nd-fold-test
  (testing "ekf-nd-fold over 5 timesteps"
    (let [T 5
          obs-as [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 2.5) (mx/scalar 2.8) (mx/scalar 3.0)]
          obs-bs [(mx/scalar 0.5) (mx/scalar 0.8) (mx/scalar 1.2) (mx/scalar 1.0) (mx/scalar 1.3)]
          context-fn (fn [t]
                       {:args [nil nil]
                        :constraints (-> cm/EMPTY
                                         (cm/set-value :obs-a (nth obs-as t))
                                         (cm/set-value :obs-b (nth obs-bs t)))})
          {:keys [ll means covs]} (ekf-nd/ekf-nd-fold step-2d [:za :zb] 1 T context-fn)]
      (mx/eval! ll)
      (is (= [1] (mx/shape ll)) "fold LL shape is [1]")
      (is (js/isFinite (mx/item ll)) "fold LL is finite")
      (mx/eval! (get means :za))
      (mx/eval! (get means :zb))
      (is (js/isFinite (mx/item (get means :za))) "final za is finite")
      (is (js/isFinite (mx/item (get means :zb))) "final zb is finite"))))

(deftest ekf-nd-batched-fold-test
  (testing "batched [P] ekf-nd-fold"
    (let [P 20
          T 4
          context-fn (fn [t]
                       {:args [nil nil]
                        :constraints
                        (-> cm/EMPTY
                            (cm/set-value :obs-a
                              (mx/multiply (rng/uniform (rng/fresh-key (* 100 t)) [P])
                                           (mx/scalar 3.0)))
                            (cm/set-value :obs-b
                              (mx/multiply (rng/uniform (rng/fresh-key (* 100 (+ t 50))) [P])
                                           (mx/scalar 2.0))))})
          {:keys [ll means covs]} (ekf-nd/ekf-nd-fold step-2d [:za :zb] P T context-fn)]
      (mx/eval! ll)
      (is (= [P] (mx/shape ll)) "batched LL is [P]-shaped")
      (is (= [P] (mx/shape (get means :za))) "batched za mean is [P]-shaped")
      (is (= [P] (mx/shape (get means :zb))) "batched zb mean is [P]-shaped")
      (is (= [P] (mx/shape (get covs [:za :za]))) "batched P_aa is [P]-shaped")
      (is (= [P] (mx/shape (get covs [:za :zb]))) "batched P_ab is [P]-shaped"))))

;; ---------------------------------------------------------------------------
;; N=1 linear equivalence: ND EKF vs Kalman
;; ---------------------------------------------------------------------------

(def kalman-step
  (gen [obs-val]
    (let [z (trace :z (kal/kalman-latent (mx/scalar rho-a) (mx/scalar 0.0) (mx/scalar q)))
          _ (trace :obs (kal/kalman-obs (mx/scalar 1.0) (mx/scalar 2.0) z (mx/scalar 0.5) (mx/scalar 1.0)))]
      z)))

(def nd-step-1d
  (gen [obs-val]
    (let [z (trace :z (ekf-nd/ekf-nd-latent linear-fa (mx/scalar 0.0) (mx/scalar q)))
          _ (trace :obs (ekf-nd/ekf-nd-obs
                          (fn [{:keys [z]}]
                            (mx/add (mx/scalar 1.0) (mx/multiply (mx/scalar 2.0) z)))
                          {:z z} (mx/scalar 0.5) (mx/scalar 1.0)))]
      z)))

(deftest ekf-nd-kalman-linear-equivalence-test
  (testing "N=1 linear equivalence: ND EKF vs Kalman"
    (let [P 10
          T 6
          obs-data (mapv (fn [t] (mx/multiply (rng/uniform (rng/fresh-key (* 100 t)) [P])
                                               (mx/scalar 5.0)))
                         (range T))
          kal-context (fn [t]
                        {:args [(nth obs-data t)]
                         :constraints (cm/set-value cm/EMPTY :obs (nth obs-data t))})
          kal-ll (kal/kalman-fold kalman-step :z P T kal-context)
          nd-context (fn [t]
                       {:args [(nth obs-data t)]
                        :constraints (cm/set-value cm/EMPTY :obs (nth obs-data t))})
          {:keys [ll]} (ekf-nd/ekf-nd-fold nd-step-1d [:z] P T nd-context)]
      (mx/eval! kal-ll)
      (mx/eval! ll)
      (let [diff (mx/item (mx/amax (mx/abs (mx/subtract ll kal-ll))))]
        (is (h/close? 0.0 diff 1e-2) "ND EKF (N=1) matches Kalman LL")))))

;; ---------------------------------------------------------------------------
;; N=3: three latents with different timescales
;; ---------------------------------------------------------------------------

(def step-3d
  (gen [ctx]
    (let [za (trace :za (ekf-nd/ekf-nd-latent
                          (fn [z] (mx/multiply (mx/scalar 0.5) z))
                          (mx/scalar 0.0) (mx/scalar 1.0)))
          zb (trace :zb (ekf-nd/ekf-nd-latent
                          (fn [z] (mx/multiply (mx/scalar 0.8) z))
                          (mx/scalar 0.0) (mx/scalar 1.0)))
          zc (trace :zc (ekf-nd/ekf-nd-latent
                          (fn [z] (mx/multiply (mx/scalar 0.95) z))
                          (mx/scalar 0.0) (mx/scalar 1.0)))
          latents {:za za :zb zb :zc zc}
          _ (trace :obs1 (ekf-nd/ekf-nd-obs
                           (fn [{:keys [za zb]}]
                             (mx/add (mx/multiply (mx/scalar 3.0) za) zb))
                           latents (mx/scalar 1.0) (mx/scalar 1.0)))
          _ (trace :obs2 (ekf-nd/ekf-nd-obs
                           (fn [{:keys [zb zc]}]
                             (mx/add (mx/multiply (mx/scalar 2.0) zb)
                                     (mx/multiply (mx/scalar 1.5) zc)))
                           latents (mx/scalar 0.8) (mx/scalar 1.0)))]
      latents)))

(deftest ekf-nd-3d-fold-test
  (testing "N=3: three latents, different timescales"
    (let [T 8
          context-fn (fn [t]
                       {:args [nil]
                        :constraints (-> cm/EMPTY
                                         (cm/set-value :obs1 (mx/scalar (+ 1.0 (* 0.5 t))))
                                         (cm/set-value :obs2 (mx/scalar (+ 0.5 (* 0.3 t)))))})
          {:keys [ll means covs]} (ekf-nd/ekf-nd-fold step-3d [:za :zb :zc] 1 T context-fn)]
      (mx/eval! ll)
      (mx/eval! (get means :za))
      (mx/eval! (get means :zb))
      (mx/eval! (get means :zc))
      (mx/eval! (get covs [:za :zb]))
      (mx/eval! (get covs [:zb :zc]))
      (is (js/isFinite (mx/item ll)) "LL is finite")
      (is (js/isFinite (mx/item (get means :za))) "za is finite")
      (is (js/isFinite (mx/item (get means :zb))) "zb is finite")
      (is (js/isFinite (mx/item (get means :zc))) "zc is finite")
      (let [pab (mx/item (get covs [:za :zb]))
            pbc (mx/item (get covs [:zb :zc]))]
        (is (> (js/Math.abs pab) 1e-6) "P_ab nonzero (shared obs1)")
        (is (> (js/Math.abs pbc) 1e-6) "P_bc nonzero (shared obs2)")))))

(deftest ekf-nd-compose-middleware-test
  (testing "composability: compose-middleware"
    (let [ekf-dispatch (ekf-nd/make-multi-ekf-dispatch [:za :zb])
          transition (ana/compose-middleware hnd/generate-transition ekf-dispatch)]
      (is (fn? transition) "compose-middleware returns function"))))

(deftest ekf-nd-variance-decrease-test
  (testing "variance decreases after observation"
    (let [addrs [:za :zb]
          means {:za (mx/scalar 0.0) :zb (mx/scalar 0.0)}
          covs {[:za :za] (mx/scalar 2.0)
                [:za :zb] (mx/scalar 0.0)
                [:zb :zb] (mx/scalar 2.0)}
          obs (mx/scalar 1.0)
          noise-std (mx/scalar 0.5)
          mask (mx/scalar 1.0)
          obs-fn (fn [{:keys [za zb]}]
                   (mx/add (mx/multiply (mx/scalar 1.5) za)
                           (mx/multiply (mx/scalar 0.5) zb)))
          {:keys [covs]} (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)]
      (mx/eval! (get covs [:za :za]))
      (mx/eval! (get covs [:zb :zb]))
      (is (< (mx/item (get covs [:za :za])) 2.0) "P_aa decreased")
      (is (< (mx/item (get covs [:zb :zb])) 2.0) "P_bb decreased"))))

;; =========================================================================
;; Analytical Jacobian variant tests
;; =========================================================================

(deftest ekf-nd-latent-j-test
  (testing "ekf-nd-latent-j under standard handler"
    (let [d (ekf-nd/ekf-nd-latent-j tanh-f
              (fn [z] (mx/multiply (mx/scalar rho-a)
                        (mx/subtract (mx/scalar 1.0)
                          (mx/multiply (mx/tanh (mx/multiply (mx/scalar rho-a) z))
                                       (mx/tanh (mx/multiply (mx/scalar rho-a) z))))))
              (mx/scalar 1.0) (mx/scalar q))
          sample (dc/dist-sample d (rng/fresh-key 99))
          lp (dc/dist-log-prob d sample)]
      (mx/eval! sample)
      (mx/eval! lp)
      (is (js/isFinite (mx/item sample)) "samples are finite")
      (is (js/isFinite (mx/item lp)) "log-prob is finite"))))

(deftest ekf-nd-obs-j-test
  (testing "ekf-nd-obs-j under standard handler"
    (let [obs-fn (fn [{:keys [za zb]}]
                   (mx/add (mx/multiply (mx/scalar 2.0) za) zb))
          jac-fn (fn [_] {:za (mx/scalar 2.0) :zb (mx/scalar 1.0)})
          vals {:za (mx/scalar 0.5) :zb (mx/scalar -0.3)}
          d (ekf-nd/ekf-nd-obs-j obs-fn jac-fn vals (mx/scalar 0.3) (mx/scalar 1.0))
          sample (dc/dist-sample d (rng/fresh-key 99))
          lp (dc/dist-log-prob d (mx/scalar 1.0))]
      (mx/eval! sample)
      (mx/eval! lp)
      (is (js/isFinite (mx/item sample)) "samples are finite")
      (is (js/isFinite (mx/item lp)) "log-prob is finite"))))

(deftest ekf-nd-analytical-predict-match-test
  (testing "analytical predict matches auto-diff"
    (let [addrs [:za :zb]
          means {:za (mx/scalar 1.5) :zb (mx/scalar -0.8)}
          covs {[:za :za] (mx/scalar 1.0)
                [:za :zb] (mx/scalar 0.2)
                [:zb :zb] (mx/scalar 0.6)}
          fa tanh-f
          jac-fa (fn [z] (mx/multiply (mx/scalar rho-a)
                           (mx/subtract (mx/scalar 1.0)
                             (mx/multiply (mx/tanh (mx/multiply (mx/scalar rho-a) z))
                                          (mx/tanh (mx/multiply (mx/scalar rho-a) z))))))
          [_ m1-ad c1-ad] (ekf-nd/ekf-nd-predict-one addrs :za means covs fa (mx/scalar q))
          [_ m1-an c1-an] (ekf-nd/ekf-nd-predict-one-j addrs :za means covs fa jac-fa (mx/scalar q))]
      (mx/eval! (get m1-ad :za))
      (mx/eval! (get m1-an :za))
      (mx/eval! (get c1-ad [:za :za]))
      (mx/eval! (get c1-an [:za :za]))
      (mx/eval! (get c1-ad [:za :zb]))
      (mx/eval! (get c1-an [:za :zb]))
      (let [mean-diff (js/Math.abs (- (mx/item (get m1-ad :za)) (mx/item (get m1-an :za))))
            paa-diff (js/Math.abs (- (mx/item (get c1-ad [:za :za])) (mx/item (get c1-an [:za :za]))))
            pab-diff (js/Math.abs (- (mx/item (get c1-ad [:za :zb])) (mx/item (get c1-an [:za :zb]))))]
        (is (h/close? 0.0 mean-diff 1e-5) "means match")
        (is (h/close? 0.0 paa-diff 1e-5) "P_aa match")
        (is (h/close? 0.0 pab-diff 1e-5) "P_ab match")))))

(deftest ekf-nd-analytical-update-linear-test
  (testing "analytical update matches auto-diff (linear)"
    (let [addrs [:za :zb]
          means {:za (mx/scalar 1.0) :zb (mx/scalar -0.5)}
          covs {[:za :za] (mx/scalar 1.0)
                [:za :zb] (mx/scalar 0.3)
                [:zb :zb] (mx/scalar 0.8)}
          obs (mx/scalar 2.5)
          noise-std (mx/scalar 0.4)
          mask (mx/scalar 1.0)
          obs-fn (fn [{:keys [za zb]}]
                   (mx/add (mx/multiply (mx/scalar 2.0) za)
                           (mx/multiply (mx/scalar 3.0) zb)))
          jac-fn (fn [_] {:za (mx/scalar 2.0) :zb (mx/scalar 3.0)})
          ad-r (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)
          an-r (ekf-nd/ekf-nd-update-j addrs means covs obs obs-fn jac-fn noise-std mask)]
      (mx/eval! (get (:means ad-r) :za))
      (mx/eval! (get (:means an-r) :za))
      (mx/eval! (get (:means ad-r) :zb))
      (mx/eval! (get (:means an-r) :zb))
      (mx/eval! (:ll ad-r))
      (mx/eval! (:ll an-r))
      (let [za-diff (js/Math.abs (- (mx/item (get (:means ad-r) :za))
                                    (mx/item (get (:means an-r) :za))))
            zb-diff (js/Math.abs (- (mx/item (get (:means ad-r) :zb))
                                    (mx/item (get (:means an-r) :zb))))
            ll-diff (js/Math.abs (- (mx/item (:ll ad-r)) (mx/item (:ll an-r))))]
        (is (h/close? 0.0 za-diff 1e-5) "za match")
        (is (h/close? 0.0 zb-diff 1e-5) "zb match")
        (is (h/close? 0.0 ll-diff 1e-5) "LL match")))))

(deftest ekf-nd-analytical-update-sigmoid-test
  (testing "analytical update matches auto-diff (sigmoid)"
    (let [addrs [:za :zb]
          means {:za (mx/scalar 0.5) :zb (mx/scalar -0.3)}
          covs {[:za :za] (mx/scalar 1.0)
                [:za :zb] (mx/scalar 0.0)
                [:zb :zb] (mx/scalar 1.0)}
          obs (mx/scalar 1.5)
          noise-std (mx/scalar 0.2)
          mask (mx/scalar 1.0)
          obs-fn (fn [{:keys [za zb]}]
                   (mx/add (mx/multiply (mx/scalar 2.0) (mx/sigmoid za))
                           (mx/sigmoid zb)))
          jac-fn (fn [{:keys [za zb]}]
                   (let [sa (mx/sigmoid za)
                         sb (mx/sigmoid zb)
                         one (mx/scalar 1.0)]
                     {:za (mx/multiply (mx/scalar 2.0)
                            (mx/multiply sa (mx/subtract one sa)))
                      :zb (mx/multiply sb (mx/subtract one sb))}))
          ad-r (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)
          an-r (ekf-nd/ekf-nd-update-j addrs means covs obs obs-fn jac-fn noise-std mask)]
      (mx/eval! (get (:means ad-r) :za))
      (mx/eval! (get (:means an-r) :za))
      (mx/eval! (:ll ad-r))
      (mx/eval! (:ll an-r))
      (let [za-diff (js/Math.abs (- (mx/item (get (:means ad-r) :za))
                                    (mx/item (get (:means an-r) :za))))
            ll-diff (js/Math.abs (- (mx/item (:ll ad-r)) (mx/item (:ll an-r))))]
        (is (h/close? 0.0 za-diff 1e-4) "za match (sigmoid)")
        (is (h/close? 0.0 ll-diff 1e-4) "LL match (sigmoid)")))))

;; ---------------------------------------------------------------------------
;; Full fold: analytical -j variants match auto-diff
;; ---------------------------------------------------------------------------

(def step-2d-j
  (gen [obs-a-val obs-b-val]
    (let [za (trace :za (ekf-nd/ekf-nd-latent-j
                          linear-fa
                          (fn [_] (mx/scalar rho-a))
                          (mx/scalar 0.0) (mx/scalar q)))
          zb (trace :zb (ekf-nd/ekf-nd-latent-j
                          linear-fb
                          (fn [_] (mx/scalar rho-b))
                          (mx/scalar 0.0) (mx/scalar q)))
          latents {:za za :zb zb}
          _ (trace :obs-a (ekf-nd/ekf-nd-obs-j
                            (fn [{:keys [za]}]
                              (mx/add (mx/scalar 1.0) (mx/multiply (mx/scalar 2.0) za)))
                            (fn [_] {:za (mx/scalar 2.0)})
                            latents (mx/scalar 0.5) (mx/scalar 1.0)))
          _ (trace :obs-b (ekf-nd/ekf-nd-obs-j
                            (fn [{:keys [za zb]}]
                              (mx/add (mx/multiply (mx/scalar 1.5) za)
                                      (mx/multiply (mx/scalar 0.8) zb)))
                            (fn [_] {:za (mx/scalar 1.5) :zb (mx/scalar 0.8)})
                            latents (mx/scalar 0.3) (mx/scalar 1.0)))]
      {:za za :zb zb})))

(deftest ekf-nd-fold-j-match-test
  (testing "full fold: -j variants match auto-diff"
    (let [T 5
          obs-as [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 2.5) (mx/scalar 2.8) (mx/scalar 3.0)]
          obs-bs [(mx/scalar 0.5) (mx/scalar 0.8) (mx/scalar 1.2) (mx/scalar 1.0) (mx/scalar 1.3)]
          context-fn (fn [t]
                       {:args [nil nil]
                        :constraints (-> cm/EMPTY
                                         (cm/set-value :obs-a (nth obs-as t))
                                         (cm/set-value :obs-b (nth obs-bs t)))})
          ad-r (ekf-nd/ekf-nd-fold step-2d [:za :zb] 1 T context-fn)
          an-r (ekf-nd/ekf-nd-fold step-2d-j [:za :zb] 1 T context-fn)]
      (mx/eval! (:ll ad-r))
      (mx/eval! (:ll an-r))
      (let [ll-diff (js/Math.abs (- (mx/item (:ll ad-r)) (mx/item (:ll an-r))))]
        (is (h/close? 0.0 ll-diff 1e-3) "fold LL matches"))
      (mx/eval! (get (:means ad-r) :za))
      (mx/eval! (get (:means an-r) :za))
      (let [za-diff (js/Math.abs (- (mx/item (get (:means ad-r) :za))
                                    (mx/item (get (:means an-r) :za))))]
        (is (h/close? 0.0 za-diff 1e-3) "final za matches")))))

(deftest ekf-nd-partial-jacobian-test
  (testing "partial Jacobian (missing addr = zero)"
    (let [addrs [:za :zb :zc]
          means {:za (mx/scalar 1.0) :zb (mx/scalar 0.5) :zc (mx/scalar -0.3)}
          covs {[:za :za] (mx/scalar 1.0) [:za :zb] (mx/scalar 0.0)
                [:za :zc] (mx/scalar 0.0) [:zb :zb] (mx/scalar 1.0)
                [:zb :zc] (mx/scalar 0.0) [:zc :zc] (mx/scalar 1.0)}
          obs (mx/scalar 2.0)
          noise-std (mx/scalar 0.5)
          mask (mx/scalar 1.0)
          obs-fn (fn [{:keys [za]}] (mx/multiply (mx/scalar 3.0) za))
          jac-fn (fn [_] {:za (mx/scalar 3.0)})
          {:keys [means covs]} (ekf-nd/ekf-nd-update-j addrs means covs obs obs-fn jac-fn noise-std mask)]
      (mx/eval! (get means :za))
      (mx/eval! (get means :zb))
      (mx/eval! (get means :zc))
      (is (not= 1.0 (mx/item (get means :za))) "za updated")
      (is (h/close? 0.5 (mx/item (get means :zb)) 1e-6) "zb unchanged")
      (is (h/close? -0.3 (mx/item (get means :zc)) 1e-6) "zc unchanged"))))

(cljs.test/run-tests)
