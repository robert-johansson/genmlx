(ns genmlx.ekf-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.ekf :as ekf]
            [genmlx.inference.kalman :as kal]
            [genmlx.inference.hmm-forward :as hmm]
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

(println "\n=== EKF Middleware Tests ===\n")

;; ---------------------------------------------------------------------------
;; Test dynamics
;; ---------------------------------------------------------------------------

(def rho 0.8)
(def noise-std 0.3)

;; Linear dynamics for equivalence testing
(defn linear-f [z] (mx/multiply (mx/scalar rho) z))

;; Nonlinear dynamics: saturating AR
(defn tanh-f [z] (mx/tanh (mx/multiply (mx/scalar rho) z)))

;; Linear observation for equivalence testing
(def obs-loading 2.0)
(def obs-base 1.0)
(defn linear-h [z] (mx/add (mx/scalar obs-base) (mx/multiply (mx/scalar obs-loading) z)))

;; Nonlinear observation: sigmoid mapping
(defn sigmoid-h [z] (mx/sigmoid z))

;; -- 1. ekf-linearize: linear function --
(println "-- 1. ekf-linearize on linear f(z) = 0.8z --")
(let [z0 (mx/scalar 2.0)
      [f-z0 A] (ekf/ekf-linearize linear-f z0)]
  (mx/eval! f-z0)
  (mx/eval! A)
  (assert-close "f(z0) = 0.8 * 2.0 = 1.6" 1.6 (mx/item f-z0) 1e-5)
  (assert-close "A = 0.8 (constant Jacobian)" 0.8 (mx/item A) 1e-5)
  (println "  f(z0):" (.toFixed (mx/item f-z0) 4) "A:" (.toFixed (mx/item A) 4)))

;; -- 2. ekf-linearize: nonlinear function --
(println "\n-- 2. ekf-linearize on tanh(0.8z) --")
(let [z0 (mx/scalar 1.5)
      [f-z0 A] (ekf/ekf-linearize tanh-f z0)]
  (mx/eval! f-z0)
  (mx/eval! A)
  (let [expected-f (js/Math.tanh (* rho 1.5))
        tanh-val (js/Math.tanh (* rho 1.5))
        expected-A (* rho (- 1.0 (* tanh-val tanh-val)))]
    (assert-close "f(z0) = tanh(0.8 * 1.5)" expected-f (mx/item f-z0) 1e-4)
    (assert-close "A = 0.8 * (1 - tanh²(1.2))" expected-A (mx/item A) 1e-4)
    (println "  f(z0):" (.toFixed (mx/item f-z0) 6) "A:" (.toFixed (mx/item A) 6))))

;; -- 3. ekf-linearize: batched [P]-shaped --
(println "\n-- 3. ekf-linearize batched [P]-shaped --")
(let [P 20
      z0 (mx/multiply (rng/uniform (rng/fresh-key) [P]) (mx/scalar 4.0))
      [f-z0 A] (ekf/ekf-linearize tanh-f z0)]
  (mx/eval! f-z0)
  (mx/eval! A)
  (assert-true "f(z0) is [P]-shaped" (= [P] (mx/shape f-z0)))
  (assert-true "A is [P]-shaped" (= [P] (mx/shape A)))
  ;; Verify A values are in (0, 0.8] — tanh derivative is in (0,1), scaled by 0.8
  (let [a-min (mx/item (mx/amin A))
        a-max (mx/item (mx/amax A))]
    (assert-true "Jacobian values in (0, 0.8]" (and (> a-min 0) (<= a-max 0.8001)))
    (println "  shapes: f" (mx/shape f-z0) "A" (mx/shape A)
             "A range:" (.toFixed a-min 4) "-" (.toFixed a-max 4))))

;; -- 4. ekf-predict matches kalman-predict for linear dynamics --
(println "\n-- 4. ekf-predict matches kalman-predict (linear) --")
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
    (assert-close "means match" 0.0 mean-diff 1e-5)
    (assert-close "variances match" 0.0 var-diff 1e-5)
    (println "  max mean diff:" (.toFixed mean-diff 8) "max var diff:" (.toFixed var-diff 8))))

;; -- 5. ekf-predict with nonlinear dynamics --
(println "\n-- 5. ekf-predict with tanh dynamics --")
(let [belief {:mean (mx/scalar 2.0) :var (mx/scalar 1.0)}
      q (mx/scalar noise-std)
      result (ekf/ekf-predict belief tanh-f q)]
  (mx/eval! (:mean result))
  (mx/eval! (:var result))
  (let [expected-mean (js/Math.tanh (* rho 2.0))
        tanh-val (js/Math.tanh (* rho 2.0))
        A (* rho (- 1.0 (* tanh-val tanh-val)))
        expected-var (+ (* A A 1.0) (* noise-std noise-std))]
    (assert-close "mean = tanh(0.8*2)" expected-mean (mx/item (:mean result)) 1e-4)
    (assert-close "var = A²·var + Q²" expected-var (mx/item (:var result)) 1e-4)
    (println "  mean:" (.toFixed (mx/item (:mean result)) 6)
             "var:" (.toFixed (mx/item (:var result)) 6))))

;; -- 6. ekf-update matches kalman-update for linear observation --
(println "\n-- 6. ekf-update matches kalman-update (linear h) --")
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
    (assert-close "means match" 0.0 mean-diff 1e-4)
    (assert-close "variances match" 0.0 var-diff 1e-4)
    (assert-close "LLs match" 0.0 ll-diff 1e-4)
    (println "  max diffs — mean:" (.toFixed mean-diff 8)
             "var:" (.toFixed var-diff 8) "ll:" (.toFixed ll-diff 8))))

;; -- 7. ekf-update with nonlinear observation --
(println "\n-- 7. ekf-update with sigmoid observation --")
(let [belief {:mean (mx/scalar 0.5) :var (mx/scalar 1.0)}
      obs (mx/scalar 0.7)
      r (mx/scalar 0.1)
      mask (mx/scalar 1.0)
      {:keys [belief ll]} (ekf/ekf-update belief obs sigmoid-h r mask)]
  (mx/eval! (:mean belief))
  (mx/eval! (:var belief))
  (mx/eval! ll)
  (assert-true "LL is finite" (js/isFinite (mx/item ll)))
  (assert-true "variance decreased" (< (mx/item (:var belief)) 1.0))
  (println "  mean:" (.toFixed (mx/item (:mean belief)) 6)
           "var:" (.toFixed (mx/item (:var belief)) 6)
           "ll:" (.toFixed (mx/item ll) 4)))

;; -- 8. Missing data (mask=0) --
(println "\n-- 8. Missing data (mask=0) --")
(let [belief {:mean (mx/scalar 1.0) :var (mx/scalar 0.5)}
      obs (mx/scalar 99.0)
      mask (mx/scalar 0.0)
      {:keys [belief ll]} (ekf/ekf-update belief obs sigmoid-h (mx/scalar 0.1) mask)]
  (mx/eval! (:mean belief))
  (mx/eval! (:var belief))
  (mx/eval! ll)
  (assert-close "mean unchanged" 1.0 (mx/item (:mean belief)) 1e-6)
  (assert-close "var unchanged" 0.5 (mx/item (:var belief)) 1e-6)
  (assert-close "LL = 0" 0.0 (mx/item ll) 1e-6)
  (println "  mean:" (.toFixed (mx/item (:mean belief)) 4)
           "var:" (.toFixed (mx/item (:var belief)) 4)
           "ll:" (.toFixed (mx/item ll) 4)))

;; -- 9. Handler middleware --
(println "\n-- 9. Handler middleware --")

(def ekf-step-fn
  (gen [obs-val obs-fn-arg]
    (let [z (trace :z (ekf/ekf-latent tanh-f (mx/scalar 0.0) (mx/scalar noise-std)))
          _ (trace :obs (ekf/ekf-obs obs-fn-arg z (mx/scalar 0.5) (mx/scalar 1.0)))]
      z)))

(let [obs (mx/scalar 0.6)
      constraints (cm/set-value cm/EMPTY :obs obs)
      result (ekf/ekf-generate ekf-step-fn [obs linear-h] constraints
                               :z 1 (rng/fresh-key))]
  (assert-true "ekf-generate returns result" (some? result))
  (mx/eval! (or (:ekf-ll result) (mx/scalar 0.0)))
  (let [ll (mx/item (or (:ekf-ll result) (mx/scalar 0.0)))]
    (assert-true "LL is finite" (js/isFinite ll))
    (println "  ekf-ll:" (.toFixed ll 4)))
  (assert-true "ekf-belief exists" (some? (:ekf-belief result)))
  (let [{:keys [mean var]} (:ekf-belief result)]
    (mx/eval! mean)
    (mx/eval! var)
    (println "  belief mean:" (.toFixed (mx/item mean) 4)
             "var:" (.toFixed (mx/item var) 4))))

;; -- 10. ekf-fold over sequence --
(println "\n-- 10. ekf-fold over sequence --")
(let [;; Observations from a saturating process near 0.8
      obs-seq [(mx/scalar 0.2) (mx/scalar 0.5) (mx/scalar 0.7)
               (mx/scalar 0.75) (mx/scalar 0.78)]
      T (count obs-seq)
      context-fn (fn [t]
                   (let [obs (nth obs-seq t)]
                     {:args [obs linear-h]
                      :constraints (cm/set-value cm/EMPTY :obs obs)}))
      {:keys [ll belief]} (ekf/ekf-fold ekf-step-fn :z 1 T context-fn)]
  (mx/eval! ll)
  (assert-true "fold LL shape is [1]" (= [1] (mx/shape ll)))
  (assert-true "fold LL is finite" (js/isFinite (mx/item ll)))
  (let [{:keys [mean var]} belief]
    (mx/eval! mean)
    (mx/eval! var)
    (assert-true "final belief mean is finite" (js/isFinite (mx/item mean)))
    (assert-true "final belief var > 0" (pos? (mx/item var)))
    (println "  total LL:" (.toFixed (mx/item ll) 4)
             "final mean:" (.toFixed (mx/item mean) 4)
             "final var:" (.toFixed (mx/item var) 4))))

;; -- 11. Batched [P] ekf-fold --
(println "\n-- 11. Batched [P] ekf-fold --")
(def ekf-step-batched
  (gen [obs-val obs-fn-arg]
    (let [z (trace :z (ekf/ekf-latent tanh-f (mx/scalar 0.0) (mx/scalar noise-std)))
          _ (trace :obs (ekf/ekf-obs obs-fn-arg z (mx/scalar 0.5) (mx/scalar 1.0)))]
      z)))

(let [P 30
      T 5
      context-fn (fn [t]
                   (let [obs (mx/multiply (rng/uniform (rng/fresh-key (* 1000 t)) [P])
                                          (mx/scalar 2.0))]
                     {:args [obs linear-h]
                      :constraints (cm/set-value cm/EMPTY :obs obs)}))
      {:keys [ll belief]} (ekf/ekf-fold ekf-step-batched :z P T context-fn)]
  (mx/eval! ll)
  (assert-true "batched LL is [P]-shaped" (= [P] (mx/shape ll)))
  (assert-true "batched belief mean is [P]-shaped" (= [P] (mx/shape (:mean belief))))
  (assert-true "batched belief var is [P]-shaped" (= [P] (mx/shape (:var belief))))
  (println "  LL shape:" (mx/shape ll) "mean LL:" (.toFixed (mx/item (mx/mean ll)) 4)
           "belief mean shape:" (mx/shape (:mean belief))))

;; -- 12. Linear equivalence: ekf-fold vs kalman-fold --
(println "\n-- 12. Linear equivalence: EKF vs Kalman --")

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

(let [P 10
      T 8
      ;; Shared observations
      obs-data (mapv (fn [t] (mx/multiply (rng/uniform (rng/fresh-key (* 100 t)) [P])
                                           (mx/scalar 3.0)))
                     (range T))
      ;; Kalman fold
      kal-context (fn [t]
                    {:args [(nth obs-data t)]
                     :constraints (cm/set-value cm/EMPTY :obs (nth obs-data t))})
      kal-ll (kal/kalman-fold kalman-step-fn :z P T kal-context)
      ;; EKF fold with linear dynamics
      ekf-context (fn [t]
                    {:args [(nth obs-data t) linear-h]
                     :constraints (cm/set-value cm/EMPTY :obs (nth obs-data t))})
      {:keys [ll]} (ekf/ekf-fold ekf-linear-step-fn :z P T ekf-context)]
  (mx/eval! kal-ll)
  (mx/eval! ll)
  (let [diff (mx/item (mx/amax (mx/abs (mx/subtract ll kal-ll))))]
    (assert-close "EKF LL matches Kalman LL for linear dynamics" 0.0 diff 1e-3)
    (println "  max LL diff:" (.toFixed diff 8)
             "kalman mean LL:" (.toFixed (mx/item (mx/mean kal-ll)) 4)
             "ekf mean LL:" (.toFixed (mx/item (mx/mean ll)) 4))))

;; -- 13. Composition: EKF + HMM via compose-middleware --
(println "\n-- 13. Composable middleware (EKF + HMM) --")
(let [log-trans (mx/array [[(js/Math.log 0.9) (js/Math.log 0.1)]
                            [(js/Math.log 0.1) (js/Math.log 0.9)]])
      ekf-dispatch (ekf/make-ekf-dispatch :z)
      hmm-dispatch (hmm/make-hmm-dispatch :regime log-trans)
      transition (ana/compose-middleware h/generate-transition ekf-dispatch hmm-dispatch)]
  (assert-true "compose-middleware returns function" (fn? transition))
  (println "  EKF + HMM composed into single transition"))

;; -- 14. Standard handler fallback --
(println "\n-- 14. Standard handler fallback --")
(let [;; ekf-latent under standard handler
      d (ekf/ekf-latent tanh-f (mx/scalar 1.0) (mx/scalar 0.3))
      sample (dc/dist-sample d (rng/fresh-key))
      lp (dc/dist-log-prob d sample)]
  (mx/eval! sample)
  (mx/eval! lp)
  (assert-true "ekf-latent samples under standard handler" (js/isFinite (mx/item sample)))
  (assert-true "ekf-latent scores under standard handler" (js/isFinite (mx/item lp)))
  (println "  sample:" (.toFixed (mx/item sample) 4) "log-prob:" (.toFixed (mx/item lp) 4)))

(let [;; ekf-obs under standard handler
      d (ekf/ekf-obs sigmoid-h (mx/scalar 0.5) (mx/scalar 0.1) (mx/scalar 1.0))
      sample (dc/dist-sample d (rng/fresh-key 1))
      lp (dc/dist-log-prob d (mx/scalar 0.6))]
  (mx/eval! sample)
  (mx/eval! lp)
  (assert-true "ekf-obs samples under standard handler" (js/isFinite (mx/item sample)))
  (assert-true "ekf-obs scores under standard handler" (js/isFinite (mx/item lp)))
  (println "  sample:" (.toFixed (mx/item sample) 4) "log-prob:" (.toFixed (mx/item lp) 4)))

(println "\n=== Done ===")
