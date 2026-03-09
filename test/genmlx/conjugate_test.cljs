(ns genmlx.conjugate-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.conjugate :as conj]
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

(def ^:private LOG-2PI 1.8378770664093453)

(println "\n=== Conjugate Prior Middleware Tests ===\n")

;; =========================================================================
;; Normal-Normal
;; =========================================================================

;; -- 1. NN pure update --
(println "-- 1. Normal-Normal pure update --")
(let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
      obs (mx/scalar 3.0)
      obs-var (mx/scalar 1.0)
      mask (mx/scalar 1.0)
      {:keys [posterior ll]} (conj/nn-update prior obs obs-var mask)]
  (mx/eval! (:mean posterior))
  (mx/eval! (:var posterior))
  (mx/eval! ll)
  ;; τ'² = 1/(1/100 + 1/1) = 1/1.01 ≈ 0.9901
  ;; m' = 0.9901 * (0/100 + 3/1) = 0.9901 * 3 ≈ 2.9703
  (assert-close "posterior mean ≈ 2.97" 2.9703 (mx/item (:mean posterior)) 0.01)
  (assert-close "posterior var ≈ 0.99" 0.9901 (mx/item (:var posterior)) 0.01)
  (assert-true "LL is finite" (js/isFinite (mx/item ll)))
  ;; Marginal: N(3 | 0, 101) => ll = -0.5*(log(2π) + log(101) + 9/101)
  (let [expected-ll (* -0.5 (+ LOG-2PI (js/Math.log 101.0) (/ 9.0 101.0)))]
    (assert-close "marginal LL correct" expected-ll (mx/item ll) 0.01))
  (println "  mean:" (.toFixed (mx/item (:mean posterior)) 4)
           "var:" (.toFixed (mx/item (:var posterior)) 4)
           "ll:" (.toFixed (mx/item ll) 4)))

;; -- 2. NN sequential updates --
(println "\n-- 2. NN sequential updates --")
(let [observations [2.8 3.1 2.9 3.3 2.7]
      obs-var (mx/scalar 1.0)
      mask (mx/scalar 1.0)
      final (reduce
              (fn [{:keys [posterior]} obs-val]
                (conj/nn-update posterior (mx/scalar obs-val) obs-var mask))
              {:posterior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}}
              observations)
      posterior (:posterior final)]
  (mx/eval! (:mean posterior))
  (mx/eval! (:var posterior))
  ;; After 5 observations from N(3, 1) with N(0, 100) prior:
  ;; τ'² = 1/(1/100 + 5/1) = 1/5.01 ≈ 0.1996
  ;; m' ≈ (0/100 + sum/1) * 0.1996 = 14.8 * 0.1996 ≈ 2.954
  (assert-close "posterior mean ≈ 2.95" 2.954 (mx/item (:mean posterior)) 0.05)
  (assert-close "posterior var ≈ 0.20" 0.1996 (mx/item (:var posterior)) 0.01)
  (println "  mean:" (.toFixed (mx/item (:mean posterior)) 4)
           "var:" (.toFixed (mx/item (:var posterior)) 4)))

;; -- 3. NN missing data --
(println "\n-- 3. NN missing data (mask=0) --")
(let [prior {:mean (mx/scalar 5.0) :var (mx/scalar 2.0)}
      {:keys [posterior ll]} (conj/nn-update prior (mx/scalar 99.0) (mx/scalar 1.0) (mx/scalar 0.0))]
  (mx/eval! (:mean posterior))
  (mx/eval! (:var posterior))
  (mx/eval! ll)
  (assert-close "mean unchanged" 5.0 (mx/item (:mean posterior)) 1e-6)
  (assert-close "var unchanged" 2.0 (mx/item (:var posterior)) 1e-6)
  (assert-close "LL = 0" 0.0 (mx/item ll) 1e-6)
  (println "  mean:" (.toFixed (mx/item (:mean posterior)) 4)
           "var:" (.toFixed (mx/item (:var posterior)) 4)
           "ll:" (.toFixed (mx/item ll) 4)))

;; -- 4. NN handler middleware --
(println "\n-- 4. NN handler middleware --")

(def nn-step
  (gen [obs-val]
    (let [mu (trace :mu (conj/nn-prior (mx/scalar 0.0) (mx/scalar 10.0)))
          _ (trace :y (conj/nn-obs :mu mu (mx/scalar 1.0) (mx/scalar 1.0)))]
      mu)))

(let [constraints (cm/set-value cm/EMPTY :y (mx/scalar 3.0))
      dispatches [(conj/make-nn-dispatch :mu)]
      result (conj/conjugate-generate nn-step [(mx/scalar 3.0)] constraints
                                      dispatches (rng/fresh-key))]
  (assert-true "conjugate-generate returns result" (some? result))
  (let [ll (:conjugate-ll result)]
    (mx/eval! (or ll (mx/scalar 0.0)))
    (assert-true "LL is finite" (js/isFinite (mx/item (or ll (mx/scalar 0.0)))))
    (println "  conjugate-ll:" (.toFixed (mx/item (or ll (mx/scalar 0.0))) 4)))
  (let [posteriors (:conjugate-posteriors result)
        post (get posteriors :mu)]
    (assert-true "posterior exists" (some? post))
    (when post
      (mx/eval! (:mean post))
      (mx/eval! (:var post))
      (println "  posterior mean:" (.toFixed (mx/item (:mean post)) 4)
               "var:" (.toFixed (mx/item (:var post)) 4)))))

;; -- 5. NN fold (online learning) --
(println "\n-- 5. NN fold (online learning over 5 observations) --")
(let [obs-data [2.8 3.1 2.9 3.3 2.7]
      T (count obs-data)
      context-fn (fn [t]
                   (let [obs (mx/scalar (nth obs-data t))]
                     {:args [obs]
                      :constraints (cm/set-value cm/EMPTY :y obs)}))
      dispatches [(conj/make-nn-dispatch :mu)]
      {:keys [ll posteriors]} (conj/conjugate-fold nn-step dispatches T context-fn)]
  (mx/eval! (or ll (mx/scalar 0.0)))
  (assert-true "fold LL is finite" (js/isFinite (mx/item (or ll (mx/scalar 0.0)))))
  (let [post (get posteriors :mu)]
    (when post
      (mx/eval! (:mean post))
      (mx/eval! (:var post))
      (assert-close "fold posterior mean ≈ 2.95" 2.954 (mx/item (:mean post)) 0.05)
      (assert-close "fold posterior var ≈ 0.20" 0.1996 (mx/item (:var post)) 0.01)
      (println "  total LL:" (.toFixed (mx/item (or ll (mx/scalar 0.0))) 4)
               "posterior mean:" (.toFixed (mx/item (:mean post)) 4)
               "var:" (.toFixed (mx/item (:var post)) 4)))))

;; =========================================================================
;; Beta-Binomial
;; =========================================================================

;; -- 6. BB pure update --
(println "\n-- 6. Beta-Binomial pure update --")
(let [prior {:alpha (mx/scalar 2.0) :beta (mx/scalar 2.0)}
      ;; Observe 3 successes, 1 failure
      data [1.0 1.0 1.0 0.0]
      mask (mx/scalar 1.0)
      final (reduce
              (fn [{:keys [posterior]} x]
                (conj/bb-update posterior (mx/scalar x) mask))
              {:posterior prior}
              data)
      posterior (:posterior final)]
  (mx/eval! (:alpha posterior))
  (mx/eval! (:beta posterior))
  ;; Beta(2+3, 2+1) = Beta(5, 3)
  (assert-close "posterior alpha = 5" 5.0 (mx/item (:alpha posterior)) 1e-5)
  (assert-close "posterior beta = 3" 3.0 (mx/item (:beta posterior)) 1e-5)
  (println "  alpha:" (.toFixed (mx/item (:alpha posterior)) 4)
           "beta:" (.toFixed (mx/item (:beta posterior)) 4)
           "mean p:" (.toFixed (/ 5.0 8.0) 4)))

;; -- 7. BB marginal LL --
(println "\n-- 7. BB marginal LL correctness --")
(let [prior {:alpha (mx/scalar 3.0) :beta (mx/scalar 2.0)}
      ;; p(x=1) = α/(α+β) = 3/5 = 0.6
      {:keys [ll]} (conj/bb-update prior (mx/scalar 1.0) (mx/scalar 1.0))]
  (mx/eval! ll)
  (assert-close "p(x=1) = α/(α+β) = 0.6" (js/Math.log 0.6) (mx/item ll) 1e-5)
  (println "  ll:" (.toFixed (mx/item ll) 6)
           "expected:" (.toFixed (js/Math.log 0.6) 6)))

(let [prior {:alpha (mx/scalar 3.0) :beta (mx/scalar 2.0)}
      ;; p(x=0) = β/(α+β) = 2/5 = 0.4
      {:keys [ll]} (conj/bb-update prior (mx/scalar 0.0) (mx/scalar 1.0))]
  (mx/eval! ll)
  (assert-close "p(x=0) = β/(α+β) = 0.4" (js/Math.log 0.4) (mx/item ll) 1e-5)
  (println "  ll:" (.toFixed (mx/item ll) 6)
           "expected:" (.toFixed (js/Math.log 0.4) 6)))

;; -- 8. BB handler middleware --
(println "\n-- 8. BB handler middleware --")

(def bb-step
  (gen [obs-val]
    (let [p (trace :p (conj/bb-prior (mx/scalar 2.0) (mx/scalar 2.0)))
          _ (trace :x (conj/bb-obs :p p (mx/scalar 1.0)))]
      p)))

(let [constraints (cm/set-value cm/EMPTY :x (mx/scalar 1.0))
      dispatches [(conj/make-bb-dispatch :p)]
      result (conj/conjugate-generate bb-step [(mx/scalar 1.0)] constraints
                                      dispatches (rng/fresh-key))]
  (let [post (get (:conjugate-posteriors result) :p)]
    (assert-true "BB posterior exists" (some? post))
    (when post
      (mx/eval! (:alpha post))
      (mx/eval! (:beta post))
      (assert-close "alpha = 3 after x=1" 3.0 (mx/item (:alpha post)) 1e-5)
      (assert-close "beta = 2 (unchanged)" 2.0 (mx/item (:beta post)) 1e-5)
      (println "  alpha:" (.toFixed (mx/item (:alpha post)) 4)
               "beta:" (.toFixed (mx/item (:beta post)) 4)))))

;; =========================================================================
;; Gamma-Poisson
;; =========================================================================

;; -- 9. GP pure update --
(println "\n-- 9. Gamma-Poisson pure update --")
(let [prior {:shape (mx/scalar 3.0) :rate (mx/scalar 1.0)}
      data [2.0 4.0 3.0 5.0 1.0]
      mask (mx/scalar 1.0)
      final (reduce
              (fn [{:keys [posterior]} x]
                (conj/gp-update posterior (mx/scalar x) mask))
              {:posterior prior}
              data)
      posterior (:posterior final)]
  (mx/eval! (:shape posterior))
  (mx/eval! (:rate posterior))
  ;; Gamma(3 + 15, 1 + 5) = Gamma(18, 6)
  (assert-close "posterior shape = 18" 18.0 (mx/item (:shape posterior)) 1e-4)
  (assert-close "posterior rate = 6" 6.0 (mx/item (:rate posterior)) 1e-4)
  (println "  shape:" (.toFixed (mx/item (:shape posterior)) 4)
           "rate:" (.toFixed (mx/item (:rate posterior)) 4)
           "mean λ:" (.toFixed (/ 18.0 6.0) 4)))

;; -- 10. GP marginal LL --
(println "\n-- 10. GP marginal LL vs NegBin --")
(let [prior {:shape (mx/scalar 3.0) :rate (mx/scalar 2.0)}
      obs (mx/scalar 4.0)
      {:keys [ll]} (conj/gp-update prior obs (mx/scalar 1.0))
      ;; NegBin(4 | r=3, p=2/3)
      nb-ll (dc/dist-log-prob (dist/neg-binomial (mx/scalar 3.0) (mx/scalar (/ 2.0 3.0)))
                              (mx/scalar 4.0))]
  (mx/eval! ll)
  (mx/eval! nb-ll)
  (assert-close "GP marginal matches NegBin" (mx/item nb-ll) (mx/item ll) 1e-4)
  (println "  GP ll:" (.toFixed (mx/item ll) 6)
           "NegBin ll:" (.toFixed (mx/item nb-ll) 6)))

;; -- 11. GP handler middleware --
(println "\n-- 11. GP handler middleware --")

(def gp-step
  (gen [obs-val]
    (let [lam (trace :lam (conj/gp-prior (mx/scalar 3.0) (mx/scalar 1.0)))
          _ (trace :x (conj/gp-obs :lam lam (mx/scalar 1.0)))]
      lam)))

(let [constraints (cm/set-value cm/EMPTY :x (mx/scalar 2.0))
      dispatches [(conj/make-gp-dispatch :lam)]
      result (conj/conjugate-generate gp-step [(mx/scalar 2.0)] constraints
                                      dispatches (rng/fresh-key))]
  (let [post (get (:conjugate-posteriors result) :lam)]
    (assert-true "GP posterior exists" (some? post))
    (when post
      (mx/eval! (:shape post))
      (mx/eval! (:rate post))
      (assert-close "shape = 5 after x=2" 5.0 (mx/item (:shape post)) 1e-5)
      (assert-close "rate = 2 after 1 obs" 2.0 (mx/item (:rate post)) 1e-5)
      (println "  shape:" (.toFixed (mx/item (:shape post)) 4)
               "rate:" (.toFixed (mx/item (:rate post)) 4)))))

;; =========================================================================
;; Cross-cutting tests
;; =========================================================================

;; -- 12. Batched [P]-shaped --
(println "\n-- 12. Batched [P]-shaped conjugate --")
(let [P 20
      prior {:mean (mx/zeros [P]) :var (mx/multiply (mx/scalar 100.0) (mx/ones [P]))}
      obs (mx/add (mx/scalar 3.0) (mx/multiply (rng/uniform (rng/fresh-key) [P]) (mx/scalar 0.5)))
      obs-var (mx/ones [P])
      mask (mx/ones [P])
      {:keys [posterior ll]} (conj/nn-update prior obs obs-var mask)]
  (mx/eval! (:mean posterior))
  (mx/eval! ll)
  (assert-true "posterior mean is [P]-shaped" (= [P] (mx/shape (:mean posterior))))
  (assert-true "LL is [P]-shaped" (= [P] (mx/shape ll)))
  (println "  posterior mean shape:" (mx/shape (:mean posterior))
           "mean of means:" (.toFixed (mx/item (mx/mean (:mean posterior))) 4)
           "LL shape:" (mx/shape ll)))

;; -- 13. Multiple conjugate priors composed --
(println "\n-- 13. Multiple conjugate priors composed --")

(def multi-step
  (gen [obs-y obs-x]
    (let [mu (trace :mu (conj/nn-prior (mx/scalar 0.0) (mx/scalar 10.0)))
          p (trace :p (conj/bb-prior (mx/scalar 1.0) (mx/scalar 1.0)))
          _ (trace :y (conj/nn-obs :mu mu (mx/scalar 1.0) (mx/scalar 1.0)))
          _ (trace :x (conj/bb-obs :p p (mx/scalar 1.0)))]
      mu)))

(let [constraints (-> cm/EMPTY
                      (cm/set-value :y (mx/scalar 5.0))
                      (cm/set-value :x (mx/scalar 1.0)))
      dispatches [(conj/make-nn-dispatch :mu)
                  (conj/make-bb-dispatch :p)]
      result (conj/conjugate-generate multi-step [(mx/scalar 5.0) (mx/scalar 1.0)]
                                      constraints dispatches (rng/fresh-key))]
  (let [posteriors (:conjugate-posteriors result)
        nn-post (get posteriors :mu)
        bb-post (get posteriors :p)]
    (assert-true "NN posterior exists" (some? nn-post))
    (assert-true "BB posterior exists" (some? bb-post))
    (when nn-post
      (mx/eval! (:mean nn-post))
      (println "  NN mean:" (.toFixed (mx/item (:mean nn-post)) 4)))
    (when bb-post
      (mx/eval! (:alpha bb-post))
      (mx/eval! (:beta bb-post))
      (assert-close "BB alpha = 2 after x=1" 2.0 (mx/item (:alpha bb-post)) 1e-5)
      (assert-close "BB beta = 1 (unchanged)" 1.0 (mx/item (:beta bb-post)) 1e-5)
      (println "  BB alpha:" (.toFixed (mx/item (:alpha bb-post)) 4)
               "beta:" (.toFixed (mx/item (:beta bb-post)) 4)))))

;; -- 14. Composition with Kalman --
(println "\n-- 14. Composable: conjugate + Kalman via compose-middleware --")
(let [nn-dispatch (conj/make-nn-dispatch :mu)
      kal-dispatch (kal/make-kalman-dispatch :z)
      transition (ana/compose-middleware h/generate-transition nn-dispatch kal-dispatch)]
  (assert-true "compose-middleware returns function" (fn? transition))
  (println "  Conjugate + Kalman composed into single transition"))

;; -- 15. Standard handler fallback --
(println "\n-- 15. Standard handler fallback --")
(let [;; nn-prior samples normally
      d (conj/nn-prior (mx/scalar 5.0) (mx/scalar 2.0))
      s (dc/dist-sample d (rng/fresh-key))
      lp (dc/dist-log-prob d s)]
  (mx/eval! s)
  (mx/eval! lp)
  (assert-true "nn-prior samples" (js/isFinite (mx/item s)))
  (assert-true "nn-prior scores" (js/isFinite (mx/item lp)))
  (println "  nn-prior sample:" (.toFixed (mx/item s) 4)
           "log-prob:" (.toFixed (mx/item lp) 4)))

(let [;; nn-obs scores normally
      d (conj/nn-obs :mu (mx/scalar 5.0) (mx/scalar 1.0) (mx/scalar 1.0))
      lp (dc/dist-log-prob d (mx/scalar 4.5))]
  (mx/eval! lp)
  (assert-true "nn-obs scores" (js/isFinite (mx/item lp)))
  (println "  nn-obs log-prob:" (.toFixed (mx/item lp) 4)))

(let [;; bb-obs scores normally
      d (conj/bb-obs :p (mx/scalar 0.7) (mx/scalar 1.0))
      lp (dc/dist-log-prob d (mx/scalar 1.0))]
  (mx/eval! lp)
  (assert-close "bb-obs scores Bernoulli" (js/Math.log 0.7) (mx/item lp) 1e-4)
  (println "  bb-obs log-prob:" (.toFixed (mx/item lp) 4)))

;; -- 16. GP fold --
(println "\n-- 16. GP fold (online rate learning) --")
(let [obs-data [2.0 3.0 1.0 4.0 2.0 3.0]
      T (count obs-data)
      context-fn (fn [t]
                   (let [obs (mx/scalar (nth obs-data t))]
                     {:args [obs]
                      :constraints (cm/set-value cm/EMPTY :x obs)}))
      dispatches [(conj/make-gp-dispatch :lam)]
      {:keys [ll posteriors]} (conj/conjugate-fold gp-step dispatches T context-fn)]
  (mx/eval! (or ll (mx/scalar 0.0)))
  (assert-true "GP fold LL is finite" (js/isFinite (mx/item (or ll (mx/scalar 0.0)))))
  (let [post (get posteriors :lam)]
    (when post
      (mx/eval! (:shape post))
      (mx/eval! (:rate post))
      ;; Gamma(3+15, 1+6) = Gamma(18, 7)
      (assert-close "fold shape = 18" 18.0 (mx/item (:shape post)) 1e-4)
      (assert-close "fold rate = 7" 7.0 (mx/item (:rate post)) 1e-4)
      (println "  total LL:" (.toFixed (mx/item (or ll (mx/scalar 0.0))) 4)
               "shape:" (.toFixed (mx/item (:shape post)) 4)
               "rate:" (.toFixed (mx/item (:rate post)) 4)
               "mean λ:" (.toFixed (/ 18.0 7.0) 4)))))

(println "\n=== Done ===")
