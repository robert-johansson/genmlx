(ns genmlx.ekf-nd-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.ekf-nd :as ekf-nd]
            [genmlx.inference.ekf :as ekf]
            [genmlx.inference.kalman :as kal]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
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

(println "\n=== Multi-Dim EKF Middleware Tests ===\n")

;; ---------------------------------------------------------------------------
;; Test dynamics and observations
;; ---------------------------------------------------------------------------

(def rho-a 0.8)
(def rho-b 0.6)
(def q 0.3)

(defn linear-fa [z] (mx/multiply (mx/scalar rho-a) z))
(defn linear-fb [z] (mx/multiply (mx/scalar rho-b) z))
(defn tanh-f [z] (mx/tanh (mx/multiply (mx/scalar rho-a) z)))

;; -- 1. Standard handler fallback: ekf-nd-latent --
(println "-- 1. ekf-nd-latent under standard handler --")
(let [d (ekf-nd/ekf-nd-latent tanh-f (mx/scalar 1.0) (mx/scalar q))
      sample (dc/dist-sample d (rng/fresh-key))
      lp (dc/dist-log-prob d sample)]
  (mx/eval! sample)
  (mx/eval! lp)
  (assert-true "samples are finite" (js/isFinite (mx/item sample)))
  (assert-true "log-prob is finite" (js/isFinite (mx/item lp)))
  (println "  sample:" (.toFixed (mx/item sample) 4) "lp:" (.toFixed (mx/item lp) 4)))

;; -- 2. Standard handler fallback: ekf-nd-obs --
(println "\n-- 2. ekf-nd-obs under standard handler --")
(let [obs-fn (fn [{:keys [za zb]}]
               (mx/add (mx/multiply (mx/scalar 2.0) (mx/sigmoid za))
                       (mx/multiply (mx/scalar 1.0) (mx/sigmoid zb))))
      vals {:za (mx/scalar 0.5) :zb (mx/scalar -0.3)}
      d (ekf-nd/ekf-nd-obs obs-fn vals (mx/scalar 0.1) (mx/scalar 1.0))
      sample (dc/dist-sample d (rng/fresh-key 1))
      lp (dc/dist-log-prob d (mx/scalar 1.5))]
  (mx/eval! sample)
  (mx/eval! lp)
  (assert-true "samples are finite" (js/isFinite (mx/item sample)))
  (assert-true "log-prob is finite" (js/isFinite (mx/item lp)))
  (println "  sample:" (.toFixed (mx/item sample) 4) "lp:" (.toFixed (mx/item lp) 4)))

;; -- 3. N=1 predict matches 1D EKF --
(println "\n-- 3. N=1 predict matches 1D EKF --")
(let [addrs [:z]
      means {:z (mx/scalar 1.5)}
      covs {[:z :z] (mx/scalar 0.8)}
      [val new-means new-covs] (ekf-nd/ekf-nd-predict-one addrs :z means covs tanh-f (mx/scalar q))
      ;; Compare with 1D
      belief {:mean (mx/scalar 1.5) :var (mx/scalar 0.8)}
      ekf1 (ekf/ekf-predict belief tanh-f (mx/scalar q))]
  (mx/eval! val)
  (mx/eval! (get new-means :z))
  (mx/eval! (get new-covs [:z :z]))
  (mx/eval! (:mean ekf1))
  (mx/eval! (:var ekf1))
  (let [mean-diff (js/Math.abs (- (mx/item val) (mx/item (:mean ekf1))))
        var-diff (js/Math.abs (- (mx/item (get new-covs [:z :z])) (mx/item (:var ekf1))))]
    (assert-close "means match" 0.0 mean-diff 1e-5)
    (assert-close "variances match" 0.0 var-diff 1e-5)
    (println "  mean diff:" (.toFixed mean-diff 8) "var diff:" (.toFixed var-diff 8))))

;; -- 4. N=2 predict: cross-covariance scaling --
(println "\n-- 4. N=2 predict: cross-covariance --")
(let [addrs [:za :zb]
      means {:za (mx/scalar 1.0) :zb (mx/scalar -0.5)}
      covs {[:za :za] (mx/scalar 1.0)
            [:za :zb] (mx/scalar 0.3)
            [:zb :zb] (mx/scalar 0.8)}
      ;; Predict za
      [_ means1 covs1] (ekf-nd/ekf-nd-predict-one addrs :za means covs linear-fa (mx/scalar q))
      ;; Predict zb
      [_ means2 covs2] (ekf-nd/ekf-nd-predict-one addrs :zb means1 covs1 linear-fb (mx/scalar q))]
  (mx/eval! (get means2 :za))
  (mx/eval! (get means2 :zb))
  (mx/eval! (get covs2 [:za :za]))
  (mx/eval! (get covs2 [:za :zb]))
  (mx/eval! (get covs2 [:zb :zb]))
  ;; Expected: P_aa = rho_a^2 * 1.0 + q^2, P_bb = rho_b^2 * 0.8 + q^2
  ;; P_ab = rho_a * rho_b * 0.3
  (let [exp-paa (+ (* rho-a rho-a 1.0) (* q q))
        exp-pbb (+ (* rho-b rho-b 0.8) (* q q))
        exp-pab (* rho-a rho-b 0.3)]
    (assert-close "P_aa" exp-paa (mx/item (get covs2 [:za :za])) 1e-5)
    (assert-close "P_bb" exp-pbb (mx/item (get covs2 [:zb :zb])) 1e-5)
    (assert-close "P_ab (cross-cov)" exp-pab (mx/item (get covs2 [:za :zb])) 1e-5)
    (assert-close "mu_a = rho_a * 1.0" (* rho-a 1.0) (mx/item (get means2 :za)) 1e-5)
    (assert-close "mu_b = rho_b * -0.5" (* rho-b -0.5) (mx/item (get means2 :zb)) 1e-5)
    (println "  P_aa:" (.toFixed (mx/item (get covs2 [:za :za])) 4)
             "P_ab:" (.toFixed (mx/item (get covs2 [:za :zb])) 4)
             "P_bb:" (.toFixed (mx/item (get covs2 [:zb :zb])) 4))))

;; -- 5. N=1 update matches 1D EKF --
(println "\n-- 5. N=1 update matches 1D EKF --")
(let [addrs [:z]
      means {:z (mx/scalar 1.5)}
      covs {[:z :z] (mx/scalar 0.8)}
      obs (mx/scalar 4.0)
      noise-std (mx/scalar 0.5)
      mask (mx/scalar 1.0)
      ;; Linear obs: h(z) = 1.0 + 2.0 * z
      obs-fn (fn [{:keys [z]}] (mx/add (mx/scalar 1.0) (mx/multiply (mx/scalar 2.0) z)))
      nd-r (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)
      ;; 1D EKF comparison
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
    (assert-close "means match" 0.0 mean-diff 1e-4)
    (assert-close "variances match" 0.0 var-diff 1e-4)
    (assert-close "LLs match" 0.0 ll-diff 1e-4)
    (println "  diffs — mean:" (.toFixed mean-diff 8) "var:" (.toFixed var-diff 8)
             "ll:" (.toFixed ll-diff 8))))

;; -- 6. N=2 update: multi-loading observation --
(println "\n-- 6. N=2 update: observation loads on both latents --")
(let [addrs [:za :zb]
      means {:za (mx/scalar 1.0) :zb (mx/scalar -0.5)}
      covs {[:za :za] (mx/scalar 1.0)
            [:za :zb] (mx/scalar 0.0)
            [:zb :zb] (mx/scalar 0.8)}
      obs (mx/scalar 2.0)
      noise-std (mx/scalar 0.3)
      mask (mx/scalar 1.0)
      ;; Observation depends on both: h = 2*za + 3*zb
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
  ;; Verify manually: H = [2, 3]
  ;; PH_a = P_aa*H_a + P_ab*H_b = 1.0*2 + 0.0*3 = 2.0
  ;; PH_b = P_ab*H_a + P_bb*H_b = 0.0*2 + 0.8*3 = 2.4
  ;; S = H_a*PH_a + H_b*PH_b + R^2 = 2*2 + 3*2.4 + 0.09 = 11.29
  ;; innov = 2.0 - (2*1 + 3*(-0.5)) = 2.0 - 0.5 = 1.5
  ;; K_a = PH_a/S = 2.0/11.29 = 0.17714
  ;; K_b = PH_b/S = 2.4/11.29 = 0.21256
  (let [pred (+ (* 2.0 1.0) (* 3.0 -0.5))  ;; 0.5
        innov (- 2.0 pred)                    ;; 1.5
        PH-a 2.0
        PH-b 2.4
        S (+ (* 2.0 PH-a) (* 3.0 PH-b) 0.09)  ;; 11.29
        K-a (/ PH-a S)
        K-b (/ PH-b S)
        exp-za (+ 1.0 (* K-a innov))
        exp-zb (+ -0.5 (* K-b innov))]
    (assert-close "mu_a updated" exp-za (mx/item (get means :za)) 1e-3)
    (assert-close "mu_b updated" exp-zb (mx/item (get means :zb)) 1e-3)
    (assert-true "LL is finite" (js/isFinite (mx/item ll)))
    ;; Cross-covariance should now be nonzero (both latents observed together)
    (let [pab (mx/item (get covs [:za :zb]))]
      (assert-true "cross-cov nonzero after multi-loading obs" (not (zero? pab)))
      (println "  za:" (.toFixed (mx/item (get means :za)) 4)
               "zb:" (.toFixed (mx/item (get means :zb)) 4)
               "P_ab:" (.toFixed pab 6)
               "ll:" (.toFixed (mx/item ll) 4)))))

;; -- 7. Missing data (mask=0) --
(println "\n-- 7. Missing data (mask=0) --")
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
  (assert-close "za unchanged" 1.0 (mx/item (get means :za)) 1e-6)
  (assert-close "zb unchanged" -0.5 (mx/item (get means :zb)) 1e-6)
  (assert-close "P_aa unchanged" 1.0 (mx/item (get covs [:za :za])) 1e-6)
  (assert-close "P_ab unchanged" 0.3 (mx/item (get covs [:za :zb])) 1e-6)
  (assert-close "P_bb unchanged" 0.8 (mx/item (get covs [:zb :zb])) 1e-6)
  (assert-close "LL = 0" 0.0 (mx/item ll) 1e-6))

;; -- 8. Nonlinear obs: sigmoid Jacobian --
(println "\n-- 8. Nonlinear observation (sigmoid) --")
(let [addrs [:za :zb]
      means {:za (mx/scalar 0.5) :zb (mx/scalar -0.3)}
      covs {[:za :za] (mx/scalar 1.0)
            [:za :zb] (mx/scalar 0.0)
            [:zb :zb] (mx/scalar 1.0)}
      obs (mx/scalar 1.5)
      noise-std (mx/scalar 0.2)
      mask (mx/scalar 1.0)
      ;; h({za, zb}) = 2*sigmoid(za) + sigmoid(zb)
      obs-fn (fn [{:keys [za zb]}]
               (mx/add (mx/multiply (mx/scalar 2.0) (mx/sigmoid za))
                       (mx/sigmoid zb)))
      {:keys [means covs ll]} (ekf-nd/ekf-nd-update addrs means covs obs obs-fn noise-std mask)]
  (mx/eval! (get means :za))
  (mx/eval! (get means :zb))
  (mx/eval! ll)
  (assert-true "za updated" (not= 0.5 (mx/item (get means :za))))
  (assert-true "zb updated" (not= -0.3 (mx/item (get means :zb))))
  (assert-true "LL is finite" (js/isFinite (mx/item ll)))
  (println "  za:" (.toFixed (mx/item (get means :za)) 4)
           "zb:" (.toFixed (mx/item (get means :zb)) 4)
           "ll:" (.toFixed (mx/item ll) 4)))

;; -- 9. Handler middleware: ekf-nd-generate --
(println "\n-- 9. Handler middleware: ekf-nd-generate --")

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

(let [constraints (-> cm/EMPTY
                      (cm/set-value :obs-a (mx/scalar 3.0))
                      (cm/set-value :obs-b (mx/scalar 1.0)))
      result (ekf-nd/ekf-nd-generate step-2d [nil nil] constraints
                                     [:za :zb] 1 (rng/fresh-key))]
  (assert-true "result has ekf-nd-means" (some? (:ekf-nd-means result)))
  (assert-true "result has ekf-nd-covs" (some? (:ekf-nd-covs result)))
  (let [ll (or (:ekf-nd-ll result) (mx/scalar 0.0))]
    (mx/eval! ll)
    (assert-true "LL is finite" (js/isFinite (mx/item ll)))
    (println "  ekf-nd-ll:" (.toFixed (mx/item ll) 4)))
  (let [{:keys [mean]} (select-keys (:ekf-nd-means result) [:za :zb])]
    (mx/eval! (get (:ekf-nd-means result) :za))
    (mx/eval! (get (:ekf-nd-means result) :zb))
    (println "  belief za:" (.toFixed (mx/item (get (:ekf-nd-means result) :za)) 4)
             "zb:" (.toFixed (mx/item (get (:ekf-nd-means result) :zb)) 4))))

;; -- 10. ekf-nd-fold over sequence --
(println "\n-- 10. ekf-nd-fold over 5 timesteps --")
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
  (assert-true "fold LL shape is [1]" (= [1] (mx/shape ll)))
  (assert-true "fold LL is finite" (js/isFinite (mx/item ll)))
  (mx/eval! (get means :za))
  (mx/eval! (get means :zb))
  (assert-true "final za is finite" (js/isFinite (mx/item (get means :za))))
  (assert-true "final zb is finite" (js/isFinite (mx/item (get means :zb))))
  (println "  total LL:" (.toFixed (mx/item ll) 4)
           "za:" (.toFixed (mx/item (get means :za)) 4)
           "zb:" (.toFixed (mx/item (get means :zb)) 4)))

;; -- 11. Batched [P] fold --
(println "\n-- 11. Batched [P] ekf-nd-fold --")
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
  (assert-true "batched LL is [P]-shaped" (= [P] (mx/shape ll)))
  (assert-true "batched za mean is [P]-shaped" (= [P] (mx/shape (get means :za))))
  (assert-true "batched zb mean is [P]-shaped" (= [P] (mx/shape (get means :zb))))
  (assert-true "batched P_aa is [P]-shaped" (= [P] (mx/shape (get covs [:za :za]))))
  (assert-true "batched P_ab is [P]-shaped" (= [P] (mx/shape (get covs [:za :zb]))))
  (println "  LL shape:" (mx/shape ll) "mean LL:" (.toFixed (mx/item (mx/mean ll)) 4)))

;; -- 12. N=1 linear equivalence: ND EKF vs Kalman --
(println "\n-- 12. N=1 linear equivalence: ND EKF vs Kalman --")

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
    (assert-close "ND EKF (N=1) matches Kalman LL" 0.0 diff 1e-2)
    (println "  max LL diff:" (.toFixed diff 6)
             "kalman mean:" (.toFixed (mx/item (mx/mean kal-ll)) 4)
             "nd-ekf mean:" (.toFixed (mx/item (mx/mean ll)) 4))))

;; -- 13. N=3 fold: three latents with different timescales --
(println "\n-- 13. N=3: three latents, different timescales --")

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
          ;; obs1 loads on za + zb
          _ (trace :obs1 (ekf-nd/ekf-nd-obs
                           (fn [{:keys [za zb]}]
                             (mx/add (mx/multiply (mx/scalar 3.0) za) zb))
                           latents (mx/scalar 1.0) (mx/scalar 1.0)))
          ;; obs2 loads on zb + zc
          _ (trace :obs2 (ekf-nd/ekf-nd-obs
                           (fn [{:keys [zb zc]}]
                             (mx/add (mx/multiply (mx/scalar 2.0) zb)
                                     (mx/multiply (mx/scalar 1.5) zc)))
                           latents (mx/scalar 0.8) (mx/scalar 1.0)))]
      latents)))

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
  (assert-true "LL is finite" (js/isFinite (mx/item ll)))
  (assert-true "za is finite" (js/isFinite (mx/item (get means :za))))
  (assert-true "zb is finite" (js/isFinite (mx/item (get means :zb))))
  (assert-true "zc is finite" (js/isFinite (mx/item (get means :zc))))
  ;; Cross-covariances should build up from shared observations
  (let [pab (mx/item (get covs [:za :zb]))
        pbc (mx/item (get covs [:zb :zc]))]
    (assert-true "P_ab nonzero (shared obs1)" (> (js/Math.abs pab) 1e-6))
    (assert-true "P_bc nonzero (shared obs2)" (> (js/Math.abs pbc) 1e-6))
    (println "  LL:" (.toFixed (mx/item ll) 4)
             "za:" (.toFixed (mx/item (get means :za)) 4)
             "zb:" (.toFixed (mx/item (get means :zb)) 4)
             "zc:" (.toFixed (mx/item (get means :zc)) 4))
    (println "  P_ab:" (.toFixed pab 6) "P_bc:" (.toFixed pbc 6))))

;; -- 14. Composition with other middleware --
(println "\n-- 14. Composability: compose-middleware --")
(let [ekf-dispatch (ekf-nd/make-multi-ekf-dispatch [:za :zb])
      ;; Can compose with any other dispatch map
      transition (ana/compose-middleware h/generate-transition ekf-dispatch)]
  (assert-true "compose-middleware returns function" (fn? transition))
  (println "  Multi-dim EKF composed into middleware chain"))

;; -- 15. Variance decrease after observation --
(println "\n-- 15. Variance decreases after observation --")
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
  (assert-true "P_aa decreased" (< (mx/item (get covs [:za :za])) 2.0))
  (assert-true "P_bb decreased" (< (mx/item (get covs [:zb :zb])) 2.0))
  (println "  P_aa:" (.toFixed (mx/item (get covs [:za :za])) 4)
           "(was 2.0) P_bb:" (.toFixed (mx/item (get covs [:zb :zb])) 4) "(was 2.0)"))

;; =========================================================================
;; Analytical Jacobian variant tests
;; =========================================================================

;; -- 16. ekf-nd-latent-j: standard handler fallback --
(println "\n-- 16. ekf-nd-latent-j under standard handler --")
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
  (assert-true "samples are finite" (js/isFinite (mx/item sample)))
  (assert-true "log-prob is finite" (js/isFinite (mx/item lp)))
  (println "  sample:" (.toFixed (mx/item sample) 4) "lp:" (.toFixed (mx/item lp) 4)))

;; -- 17. ekf-nd-obs-j: standard handler fallback --
(println "\n-- 17. ekf-nd-obs-j under standard handler --")
(let [obs-fn (fn [{:keys [za zb]}]
               (mx/add (mx/multiply (mx/scalar 2.0) za) zb))
      jac-fn (fn [_] {:za (mx/scalar 2.0) :zb (mx/scalar 1.0)})
      vals {:za (mx/scalar 0.5) :zb (mx/scalar -0.3)}
      d (ekf-nd/ekf-nd-obs-j obs-fn jac-fn vals (mx/scalar 0.3) (mx/scalar 1.0))
      sample (dc/dist-sample d (rng/fresh-key 99))
      lp (dc/dist-log-prob d (mx/scalar 1.0))]
  (mx/eval! sample)
  (mx/eval! lp)
  (assert-true "samples are finite" (js/isFinite (mx/item sample)))
  (assert-true "log-prob is finite" (js/isFinite (mx/item lp)))
  (println "  sample:" (.toFixed (mx/item sample) 4) "lp:" (.toFixed (mx/item lp) 4)))

;; -- 18. Analytical predict matches auto-diff predict --
(println "\n-- 18. Analytical predict matches auto-diff --")
(let [addrs [:za :zb]
      means {:za (mx/scalar 1.5) :zb (mx/scalar -0.8)}
      covs {[:za :za] (mx/scalar 1.0)
            [:za :zb] (mx/scalar 0.2)
            [:zb :zb] (mx/scalar 0.6)}
      ;; tanh dynamics for za
      fa tanh-f
      jac-fa (fn [z] (mx/multiply (mx/scalar rho-a)
                       (mx/subtract (mx/scalar 1.0)
                         (mx/multiply (mx/tanh (mx/multiply (mx/scalar rho-a) z))
                                      (mx/tanh (mx/multiply (mx/scalar rho-a) z))))))
      ;; Auto-diff predict za
      [_ m1-ad c1-ad] (ekf-nd/ekf-nd-predict-one addrs :za means covs fa (mx/scalar q))
      ;; Analytical predict za
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
    (assert-close "means match" 0.0 mean-diff 1e-5)
    (assert-close "P_aa match" 0.0 paa-diff 1e-5)
    (assert-close "P_ab match" 0.0 pab-diff 1e-5)
    (println "  diffs — mean:" (.toFixed mean-diff 8)
             "P_aa:" (.toFixed paa-diff 8) "P_ab:" (.toFixed pab-diff 8))))

;; -- 19. Analytical update matches auto-diff update (linear obs) --
(println "\n-- 19. Analytical update matches auto-diff (linear) --")
(let [addrs [:za :zb]
      means {:za (mx/scalar 1.0) :zb (mx/scalar -0.5)}
      covs {[:za :za] (mx/scalar 1.0)
            [:za :zb] (mx/scalar 0.3)
            [:zb :zb] (mx/scalar 0.8)}
      obs (mx/scalar 2.5)
      noise-std (mx/scalar 0.4)
      mask (mx/scalar 1.0)
      ;; h = 2*za + 3*zb, Jacobian is constant
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
    (assert-close "za match" 0.0 za-diff 1e-5)
    (assert-close "zb match" 0.0 zb-diff 1e-5)
    (assert-close "LL match" 0.0 ll-diff 1e-5)
    (println "  diffs — za:" (.toFixed za-diff 8)
             "zb:" (.toFixed zb-diff 8) "ll:" (.toFixed ll-diff 8))))

;; -- 20. Analytical update matches auto-diff (sigmoid obs) --
(println "\n-- 20. Analytical update matches auto-diff (sigmoid) --")
(let [addrs [:za :zb]
      means {:za (mx/scalar 0.5) :zb (mx/scalar -0.3)}
      covs {[:za :za] (mx/scalar 1.0)
            [:za :zb] (mx/scalar 0.0)
            [:zb :zb] (mx/scalar 1.0)}
      obs (mx/scalar 1.5)
      noise-std (mx/scalar 0.2)
      mask (mx/scalar 1.0)
      ;; h = 2*sigmoid(za) + sigmoid(zb)
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
    (assert-close "za match (sigmoid)" 0.0 za-diff 1e-4)
    (assert-close "LL match (sigmoid)" 0.0 ll-diff 1e-4)
    (println "  diffs — za:" (.toFixed za-diff 8) "ll:" (.toFixed ll-diff 8))))

;; -- 21. Full fold: analytical -j variants match auto-diff --
(println "\n-- 21. Full fold: -j variants match auto-diff --")

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

(let [T 5
      obs-as [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 2.5) (mx/scalar 2.8) (mx/scalar 3.0)]
      obs-bs [(mx/scalar 0.5) (mx/scalar 0.8) (mx/scalar 1.2) (mx/scalar 1.0) (mx/scalar 1.3)]
      context-fn (fn [t]
                   {:args [nil nil]
                    :constraints (-> cm/EMPTY
                                     (cm/set-value :obs-a (nth obs-as t))
                                     (cm/set-value :obs-b (nth obs-bs t)))})
      ;; Auto-diff fold (test 10 used step-2d)
      ad-r (ekf-nd/ekf-nd-fold step-2d [:za :zb] 1 T context-fn)
      ;; Analytical fold
      an-r (ekf-nd/ekf-nd-fold step-2d-j [:za :zb] 1 T context-fn)]
  (mx/eval! (:ll ad-r))
  (mx/eval! (:ll an-r))
  (let [ll-diff (js/Math.abs (- (mx/item (:ll ad-r)) (mx/item (:ll an-r))))]
    (assert-close "fold LL matches" 0.0 ll-diff 1e-3)
    (println "  auto-diff LL:" (.toFixed (mx/item (:ll ad-r)) 4)
             "analytical LL:" (.toFixed (mx/item (:ll an-r)) 4)
             "diff:" (.toFixed ll-diff 8)))
  (mx/eval! (get (:means ad-r) :za))
  (mx/eval! (get (:means an-r) :za))
  (let [za-diff (js/Math.abs (- (mx/item (get (:means ad-r) :za))
                                (mx/item (get (:means an-r) :za))))]
    (assert-close "final za matches" 0.0 za-diff 1e-3)
    (println "  final za diff:" (.toFixed za-diff 8))))

;; -- 22. Jacobian-fn with missing addrs defaults to zero --
(println "\n-- 22. Partial Jacobian (missing addr = zero) --")
(let [addrs [:za :zb :zc]
      means {:za (mx/scalar 1.0) :zb (mx/scalar 0.5) :zc (mx/scalar -0.3)}
      covs {[:za :za] (mx/scalar 1.0) [:za :zb] (mx/scalar 0.0)
            [:za :zc] (mx/scalar 0.0) [:zb :zb] (mx/scalar 1.0)
            [:zb :zc] (mx/scalar 0.0) [:zc :zc] (mx/scalar 1.0)}
      obs (mx/scalar 2.0)
      noise-std (mx/scalar 0.5)
      mask (mx/scalar 1.0)
      ;; Only loads on za — zb, zc Jacobians should be zero
      obs-fn (fn [{:keys [za]}] (mx/multiply (mx/scalar 3.0) za))
      jac-fn (fn [_] {:za (mx/scalar 3.0)})  ;; zb, zc missing → zero
      {:keys [means covs]} (ekf-nd/ekf-nd-update-j addrs means covs obs obs-fn jac-fn noise-std mask)]
  (mx/eval! (get means :za))
  (mx/eval! (get means :zb))
  (mx/eval! (get means :zc))
  ;; Only za should update (no cross-cov initially)
  (assert-true "za updated" (not= 1.0 (mx/item (get means :za))))
  (assert-close "zb unchanged" 0.5 (mx/item (get means :zb)) 1e-6)
  (assert-close "zc unchanged" -0.3 (mx/item (get means :zc)) 1e-6)
  (println "  za:" (.toFixed (mx/item (get means :za)) 4)
           "zb:" (.toFixed (mx/item (get means :zb)) 4)
           "zc:" (.toFixed (mx/item (get means :zc)) 4)))

(println "\n=== Done ===")
