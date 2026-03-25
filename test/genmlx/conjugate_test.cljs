(ns genmlx.conjugate-test
  "Conjugate prior middleware tests.
   Normal-Normal, Beta-Bernoulli, Gamma-Poisson pure updates,
   handler middleware, fold, batched, composition."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as handler]
            [genmlx.runtime :as rt]
            [genmlx.inference.conjugate :as conj]
            [genmlx.inference.kalman :as kal]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private LOG-2PI 1.8378770664093453)

;; =========================================================================
;; Normal-Normal
;; =========================================================================

(deftest nn-pure-update
  (testing "NN pure update"
    (let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
          obs (mx/scalar 3.0)
          obs-var (mx/scalar 1.0)
          mask (mx/scalar 1.0)
          {:keys [posterior ll]} (conj/nn-update prior obs obs-var mask)]
      (mx/eval! (:mean posterior))
      (mx/eval! (:var posterior))
      (mx/eval! ll)
      (is (h/close? 2.9703 (mx/item (:mean posterior)) 0.01) "posterior mean ~ 2.97")
      (is (h/close? 0.9901 (mx/item (:var posterior)) 0.01) "posterior var ~ 0.99")
      (is (js/isFinite (mx/item ll)) "LL is finite")
      ;; Marginal: N(3 | 0, 101) => ll = -0.5*(log(2pi) + log(101) + 9/101)
      (let [expected-ll (* -0.5 (+ LOG-2PI (js/Math.log 101.0) (/ 9.0 101.0)))]
        (is (h/close? expected-ll (mx/item ll) 0.01) "marginal LL correct")))))

(deftest nn-sequential-updates
  (testing "NN sequential updates"
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
      (is (h/close? 2.954 (mx/item (:mean posterior)) 0.05) "posterior mean ~ 2.95")
      (is (h/close? 0.1996 (mx/item (:var posterior)) 0.01) "posterior var ~ 0.20"))))

(deftest nn-missing-data
  (testing "NN missing data (mask=0)"
    (let [prior {:mean (mx/scalar 5.0) :var (mx/scalar 2.0)}
          {:keys [posterior ll]} (conj/nn-update prior (mx/scalar 99.0) (mx/scalar 1.0) (mx/scalar 0.0))]
      (mx/eval! (:mean posterior))
      (mx/eval! (:var posterior))
      (mx/eval! ll)
      (is (h/close? 5.0 (mx/item (:mean posterior)) 1e-6) "mean unchanged")
      (is (h/close? 2.0 (mx/item (:var posterior)) 1e-6) "var unchanged")
      (is (h/close? 0.0 (mx/item ll) 1e-6) "LL = 0"))))

(deftest nn-handler-middleware
  (testing "NN handler middleware"
    (let [nn-step (gen [obs-val]
                    (let [mu (trace :mu (conj/nn-prior (mx/scalar 0.0) (mx/scalar 10.0)))
                          _ (trace :y (conj/nn-obs :mu mu (mx/scalar 1.0) (mx/scalar 1.0)))]
                      mu))
          constraints (cm/set-value cm/EMPTY :y (mx/scalar 3.0))
          dispatches [(conj/make-nn-dispatch :mu)]
          result (conj/conjugate-generate nn-step [(mx/scalar 3.0)] constraints
                                          dispatches (rng/fresh-key))]
      (is (some? result) "conjugate-generate returns result")
      (let [ll (:conjugate-ll result)]
        (mx/eval! (or ll (mx/scalar 0.0)))
        (is (js/isFinite (mx/item (or ll (mx/scalar 0.0)))) "LL is finite"))
      (let [posteriors (:conjugate-posteriors result)
            post (get posteriors :mu)]
        (is (some? post) "posterior exists")
        (when post
          (mx/eval! (:mean post))
          (mx/eval! (:var post)))))))

(deftest nn-fold-online-learning
  (testing "NN fold (online learning over 5 observations)"
    (let [nn-step (gen [obs-val]
                    (let [mu (trace :mu (conj/nn-prior (mx/scalar 0.0) (mx/scalar 10.0)))
                          _ (trace :y (conj/nn-obs :mu mu (mx/scalar 1.0) (mx/scalar 1.0)))]
                      mu))
          obs-data [2.8 3.1 2.9 3.3 2.7]
          T (count obs-data)
          context-fn (fn [t]
                       (let [obs (mx/scalar (nth obs-data t))]
                         {:args [obs]
                          :constraints (cm/set-value cm/EMPTY :y obs)}))
          dispatches [(conj/make-nn-dispatch :mu)]
          {:keys [ll posteriors]} (conj/conjugate-fold nn-step dispatches T context-fn)]
      (mx/eval! (or ll (mx/scalar 0.0)))
      (is (js/isFinite (mx/item (or ll (mx/scalar 0.0)))) "fold LL is finite")
      (let [post (get posteriors :mu)]
        (when post
          (mx/eval! (:mean post))
          (mx/eval! (:var post))
          (is (h/close? 2.954 (mx/item (:mean post)) 0.05) "fold posterior mean ~ 2.95")
          (is (h/close? 0.1996 (mx/item (:var post)) 0.01) "fold posterior var ~ 0.20"))))))

;; =========================================================================
;; Beta-Binomial
;; =========================================================================

(deftest bb-pure-update
  (testing "BB pure update"
    (let [prior {:alpha (mx/scalar 2.0) :beta (mx/scalar 2.0)}
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
      (is (h/close? 5.0 (mx/item (:alpha posterior)) 1e-5) "posterior alpha = 5")
      (is (h/close? 3.0 (mx/item (:beta posterior)) 1e-5) "posterior beta = 3"))))

(deftest bb-marginal-ll
  (testing "BB marginal LL correctness"
    ;; p(x=1) = alpha/(alpha+beta) = 3/5 = 0.6
    (let [prior {:alpha (mx/scalar 3.0) :beta (mx/scalar 2.0)}
          {:keys [ll]} (conj/bb-update prior (mx/scalar 1.0) (mx/scalar 1.0))]
      (mx/eval! ll)
      (is (h/close? (js/Math.log 0.6) (mx/item ll) 1e-5) "p(x=1) = alpha/(alpha+beta) = 0.6"))

    ;; p(x=0) = beta/(alpha+beta) = 2/5 = 0.4
    (let [prior {:alpha (mx/scalar 3.0) :beta (mx/scalar 2.0)}
          {:keys [ll]} (conj/bb-update prior (mx/scalar 0.0) (mx/scalar 1.0))]
      (mx/eval! ll)
      (is (h/close? (js/Math.log 0.4) (mx/item ll) 1e-5) "p(x=0) = beta/(alpha+beta) = 0.4"))))

(deftest bb-handler-middleware
  (testing "BB handler middleware"
    (let [bb-step (gen [obs-val]
                    (let [p (trace :p (conj/bb-prior (mx/scalar 2.0) (mx/scalar 2.0)))
                          _ (trace :x (conj/bb-obs :p p (mx/scalar 1.0)))]
                      p))
          constraints (cm/set-value cm/EMPTY :x (mx/scalar 1.0))
          dispatches [(conj/make-bb-dispatch :p)]
          result (conj/conjugate-generate bb-step [(mx/scalar 1.0)] constraints
                                          dispatches (rng/fresh-key))]
      (let [post (get (:conjugate-posteriors result) :p)]
        (is (some? post) "BB posterior exists")
        (when post
          (mx/eval! (:alpha post))
          (mx/eval! (:beta post))
          (is (h/close? 3.0 (mx/item (:alpha post)) 1e-5) "alpha = 3 after x=1")
          (is (h/close? 2.0 (mx/item (:beta post)) 1e-5) "beta = 2 (unchanged)"))))))

;; =========================================================================
;; Gamma-Poisson
;; =========================================================================

(deftest gp-pure-update
  (testing "GP pure update"
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
      (is (h/close? 18.0 (mx/item (:shape posterior)) 1e-4) "posterior shape = 18")
      (is (h/close? 6.0 (mx/item (:rate posterior)) 1e-4) "posterior rate = 6"))))

(deftest gp-marginal-ll
  (testing "GP marginal LL vs NegBin"
    (let [prior {:shape (mx/scalar 3.0) :rate (mx/scalar 2.0)}
          obs (mx/scalar 4.0)
          {:keys [ll]} (conj/gp-update prior obs (mx/scalar 1.0))
          ;; NegBin(4 | r=3, p=2/3)
          nb-ll (dc/dist-log-prob (dist/neg-binomial (mx/scalar 3.0) (mx/scalar (/ 2.0 3.0)))
                                  (mx/scalar 4.0))]
      (mx/eval! ll)
      (mx/eval! nb-ll)
      (is (h/close? (mx/item nb-ll) (mx/item ll) 1e-4) "GP marginal matches NegBin"))))

(deftest gp-handler-middleware
  (testing "GP handler middleware"
    (let [gp-step (gen [obs-val]
                    (let [lam (trace :lam (conj/gp-prior (mx/scalar 3.0) (mx/scalar 1.0)))
                          _ (trace :x (conj/gp-obs :lam lam (mx/scalar 1.0)))]
                      lam))
          constraints (cm/set-value cm/EMPTY :x (mx/scalar 2.0))
          dispatches [(conj/make-gp-dispatch :lam)]
          result (conj/conjugate-generate gp-step [(mx/scalar 2.0)] constraints
                                          dispatches (rng/fresh-key))]
      (let [post (get (:conjugate-posteriors result) :lam)]
        (is (some? post) "GP posterior exists")
        (when post
          (mx/eval! (:shape post))
          (mx/eval! (:rate post))
          (is (h/close? 5.0 (mx/item (:shape post)) 1e-5) "shape = 5 after x=2")
          (is (h/close? 2.0 (mx/item (:rate post)) 1e-5) "rate = 2 after 1 obs"))))))

;; =========================================================================
;; Cross-cutting tests
;; =========================================================================

(deftest batched-p-shaped
  (testing "Batched [P]-shaped conjugate"
    (let [P 20
          prior {:mean (mx/zeros [P]) :var (mx/multiply (mx/scalar 100.0) (mx/ones [P]))}
          obs (mx/add (mx/scalar 3.0) (mx/multiply (rng/uniform (rng/fresh-key) [P]) (mx/scalar 0.5)))
          obs-var (mx/ones [P])
          mask (mx/ones [P])
          {:keys [posterior ll]} (conj/nn-update prior obs obs-var mask)]
      (mx/eval! (:mean posterior))
      (mx/eval! ll)
      (is (= [P] (mx/shape (:mean posterior))) "posterior mean is [P]-shaped")
      (is (= [P] (mx/shape ll)) "LL is [P]-shaped"))))

(deftest multiple-conjugate-priors
  (testing "Multiple conjugate priors composed"
    (let [multi-step (gen [obs-y obs-x]
                       (let [mu (trace :mu (conj/nn-prior (mx/scalar 0.0) (mx/scalar 10.0)))
                             p (trace :p (conj/bb-prior (mx/scalar 1.0) (mx/scalar 1.0)))
                             _ (trace :y (conj/nn-obs :mu mu (mx/scalar 1.0) (mx/scalar 1.0)))
                             _ (trace :x (conj/bb-obs :p p (mx/scalar 1.0)))]
                         mu))
          constraints (-> cm/EMPTY
                          (cm/set-value :y (mx/scalar 5.0))
                          (cm/set-value :x (mx/scalar 1.0)))
          dispatches [(conj/make-nn-dispatch :mu)
                      (conj/make-bb-dispatch :p)]
          result (conj/conjugate-generate multi-step [(mx/scalar 5.0) (mx/scalar 1.0)]
                                          constraints dispatches (rng/fresh-key))]
      (let [posteriors (:conjugate-posteriors result)
            nn-post (get posteriors :mu)
            bb-post (get posteriors :p)]
        (is (some? nn-post) "NN posterior exists")
        (is (some? bb-post) "BB posterior exists")
        (when nn-post
          (mx/eval! (:mean nn-post)))
        (when bb-post
          (mx/eval! (:alpha bb-post))
          (mx/eval! (:beta bb-post))
          (is (h/close? 2.0 (mx/item (:alpha bb-post)) 1e-5) "BB alpha = 2 after x=1")
          (is (h/close? 1.0 (mx/item (:beta bb-post)) 1e-5) "BB beta = 1 (unchanged)"))))))

(deftest compose-conjugate-kalman
  (testing "Composable: conjugate + Kalman via compose-middleware"
    (let [nn-dispatch (conj/make-nn-dispatch :mu)
          kal-dispatch (kal/make-kalman-dispatch :z)
          transition (ana/compose-middleware handler/generate-transition nn-dispatch kal-dispatch)]
      (is (fn? transition) "compose-middleware returns function"))))

(deftest standard-handler-fallback
  (testing "Standard handler fallback"
    ;; nn-prior samples normally
    (let [d (conj/nn-prior (mx/scalar 5.0) (mx/scalar 2.0))
          s (dc/dist-sample d (rng/fresh-key))
          lp (dc/dist-log-prob d s)]
      (mx/eval! s)
      (mx/eval! lp)
      (is (js/isFinite (mx/item s)) "nn-prior samples")
      (is (js/isFinite (mx/item lp)) "nn-prior scores"))

    ;; nn-obs scores normally
    (let [d (conj/nn-obs :mu (mx/scalar 5.0) (mx/scalar 1.0) (mx/scalar 1.0))
          lp (dc/dist-log-prob d (mx/scalar 4.5))]
      (mx/eval! lp)
      (is (js/isFinite (mx/item lp)) "nn-obs scores"))

    ;; bb-obs scores normally
    (let [d (conj/bb-obs :p (mx/scalar 0.7) (mx/scalar 1.0))
          lp (dc/dist-log-prob d (mx/scalar 1.0))]
      (mx/eval! lp)
      (is (h/close? (js/Math.log 0.7) (mx/item lp) 1e-4) "bb-obs scores Bernoulli"))))

(deftest gp-fold-online-rate-learning
  (testing "GP fold (online rate learning)"
    (let [gp-step (gen [obs-val]
                    (let [lam (trace :lam (conj/gp-prior (mx/scalar 3.0) (mx/scalar 1.0)))
                          _ (trace :x (conj/gp-obs :lam lam (mx/scalar 1.0)))]
                      lam))
          obs-data [2.0 3.0 1.0 4.0 2.0 3.0]
          T (count obs-data)
          context-fn (fn [t]
                       (let [obs (mx/scalar (nth obs-data t))]
                         {:args [obs]
                          :constraints (cm/set-value cm/EMPTY :x obs)}))
          dispatches [(conj/make-gp-dispatch :lam)]
          {:keys [ll posteriors]} (conj/conjugate-fold gp-step dispatches T context-fn)]
      (mx/eval! (or ll (mx/scalar 0.0)))
      (is (js/isFinite (mx/item (or ll (mx/scalar 0.0)))) "GP fold LL is finite")
      (let [post (get posteriors :lam)]
        (when post
          (mx/eval! (:shape post))
          (mx/eval! (:rate post))
          ;; Gamma(3+15, 1+6) = Gamma(18, 7)
          (is (h/close? 18.0 (mx/item (:shape post)) 1e-4) "fold shape = 18")
          (is (h/close? 7.0 (mx/item (:rate post)) 1e-4) "fold rate = 7"))))))

(cljs.test/run-tests)
