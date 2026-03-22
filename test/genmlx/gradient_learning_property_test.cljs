(ns genmlx.gradient-learning-property-test
  "Property-based tests for gradients, parameter stores, optimizers,
   and training loops using test.check."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.gradients :as grad]
            [genmlx.learning :as learn]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Model and fixture pools
;; ---------------------------------------------------------------------------

(def model
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/add x y)))))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

;; Pools for gradient tests -- points at which to evaluate gradients
;; Avoid 0.0 for autodiff/numerical tests (division issues at exactly 0)
(def x-pool [0.5 1.0 -0.5 -1.0 2.0 -2.0])
(def gen-x (gen/elements x-pool))

;; Pools for parameter store tests
(def param-pool [1.0 2.0 -1.0 0.5 10.0])
(def gen-param-val (gen/elements param-pool))
(def name-pool [:a :b :c :x :y])
(def gen-param-name (gen/elements name-pool))

;; Pools for optimizer tests
(def lr-pool [0.01 0.05 0.1 0.5])
(def gen-lr (gen/elements lr-pool))
(def grad-pool [1.0 -1.0 2.0 -0.5 0.1])
(def gen-grad-val (gen/elements grad-pool))
(def init-pool [1.0 3.0 5.0 -3.0 10.0])
(def gen-init (gen/elements init-pool))

;; ---------------------------------------------------------------------------
;; Choice Gradients (2)
;; ---------------------------------------------------------------------------

(defspec choice-gradients-gradient-shape-is-scalar 50
  (prop/for-all [xv gen-x
                 yv gen-x]
    (let [{:keys [trace]} (p/generate model [] (cm/choicemap :x (mx/scalar xv) :y (mx/scalar yv)))
          grads (grad/choice-gradients model trace [:x :y])]
      (and (= [] (mx/shape (:x grads)))
           (= [] (mx/shape (:y grads)))))))

;; Models with varying means -- gradient at mode should always be 0
;; Note: choice-gradients uses mx/compile-fn which caches the computation graph,
;; so each test trial must create a fresh model to get correct gradients.
(def mu-pool [0.0 1.0 -1.0 2.5 -3.0])
(def gen-mu (gen/elements mu-pool))

(defspec choice-gradients-gradient-at-mode-near-0 50
  (prop/for-all [mu gen-mu]
    ;; For gaussian(mu,1), the gradient d(log p)/dx at x=mu is 0
    (let [mode-model (dyn/auto-key
                       (gen []
                         (let [x (trace :x (dist/gaussian mu 1))
                               y (trace :y (dist/gaussian mu 1))]
                           (mx/eval! x y)
                           (+ (mx/item x) (mx/item y)))))
          {:keys [trace]} (p/generate mode-model [] (cm/choicemap :x (mx/scalar mu) :y (mx/scalar mu)))
          grads (grad/choice-gradients mode-model trace [:x :y])
          gx (eval-weight (:x grads))
          gy (eval-weight (:y grads))]
      (and (close? 0.0 gx 0.1)
           (close? 0.0 gy 0.1)))))

;; ---------------------------------------------------------------------------
;; Score Function Gradients (3)
;; Using make-score-fn + mx/grad (avoids compile-fn zero-grad issue)
;; ---------------------------------------------------------------------------

(defspec make-score-fn-gradient-shape-matches-params 50
  (prop/for-all [xv gen-x
                 yv gen-x]
    (let [obs cm/EMPTY
          sf (u/make-score-fn model [] obs [:x :y])
          grad-fn (mx/grad sf)
          params (mx/array [xv yv])
          g (grad-fn params)]
      (= (mx/shape g) [2]))))

(defspec make-score-fn-gradient-correct-for-gaussian 50
  (prop/for-all [xv gen-x
                 yv gen-x]
    ;; For gaussian(0,1): d(log p)/dx = -x
    ;; With y constrained, grad w.r.t. x = -x
    (let [obs (cm/choicemap :y (mx/scalar yv))
          sf (u/make-score-fn model [] obs [:x])
          grad-fn (mx/grad sf)
          params (mx/array [xv])
          g (grad-fn params)]
      (mx/eval! g)
      (close? (- xv) (mx/item (mx/index g 0)) 0.05))))

(defspec make-score-fn-autodiff-near-numerical-gradient 30
  (prop/for-all [xv gen-x
                 yv gen-x]
    (let [obs (cm/choicemap :y (mx/scalar yv))
          sf (u/make-score-fn model [] obs [:x])
          grad-fn (mx/grad sf)
          params (mx/array [xv])
          g (grad-fn params)
          _ (mx/eval! g)
          ad-grad (mx/item (mx/index g 0))
          ;; Numerical gradient via finite differences
          eps 0.001
          score-fn (fn [x]
                     (let [{:keys [weight]} (p/generate model []
                                              (cm/choicemap :x (mx/scalar x) :y (mx/scalar yv)))]
                       (eval-weight weight)))
          s-plus (score-fn (+ xv eps))
          s-minus (score-fn (- xv eps))
          num-grad (/ (- s-plus s-minus) (* 2 eps))]
      (close? ad-grad num-grad 0.05))))

;; ---------------------------------------------------------------------------
;; Parameter Store (3)
;; ---------------------------------------------------------------------------

(defspec param-store-get-set-round-trip 50
  (prop/for-all [pname gen-param-name
                 pval gen-param-val]
    (let [store (learn/make-param-store)
          store (learn/set-param store pname (mx/scalar pval))
          v (learn/get-param store pname)]
      (close? pval (eval-weight v) 1e-6))))

(defspec param-store-param-names-includes-all-stored-keys 50
  (prop/for-all [v1 gen-param-val
                 v2 gen-param-val
                 v3 gen-param-val]
    (let [store (-> (learn/make-param-store)
                    (learn/set-param :a (mx/scalar v1))
                    (learn/set-param :b (mx/scalar v2))
                    (learn/set-param :c (mx/scalar v3)))
          names (set (learn/param-names store))]
      (and (contains? names :a)
           (contains? names :b)
           (contains? names :c)))))

(defspec param-store-params-array-array-params-round-trip 50
  (prop/for-all [v1 gen-param-val
                 v2 gen-param-val]
    (let [store (-> (learn/make-param-store)
                    (learn/set-param :x (mx/scalar v1))
                    (learn/set-param :y (mx/scalar v2)))
          names [:x :y]
          arr (learn/params->array store names)
          _ (mx/eval! arr)
          recovered (learn/array->params arr names)
          rx (eval-weight (:x recovered))
          ry (eval-weight (:y recovered))]
      (and (close? v1 rx 1e-6)
           (close? v2 ry 1e-6)))))

;; ---------------------------------------------------------------------------
;; Optimizers (3)
;; ---------------------------------------------------------------------------

(defspec sgd-step-moves-params-in-negative-gradient-direction 50
  (prop/for-all [init gen-init
                 gval gen-grad-val
                 lr gen-lr]
    (let [params (mx/array [init])
          grad-arr (mx/array [gval])
          new-params (learn/sgd-step params grad-arr lr)]
      (mx/eval! new-params)
      ;; Should move in negative gradient direction: init - lr*gval
      (close? (- init (* lr gval)) (mx/item (mx/index new-params 0)) 1e-6))))

(defspec sgd-step-magnitude-proportional-to-lr 50
  (prop/for-all [init gen-init
                 gval gen-grad-val]
    (let [params (mx/array [init])
          grad-arr (mx/array [gval])
          p1 (learn/sgd-step params grad-arr 0.1)
          p2 (learn/sgd-step params grad-arr 0.2)
          _ (mx/eval! p1 p2)
          delta1 (- init (mx/item (mx/index p1 0)))  ;; 0.1*gval
          delta2 (- init (mx/item (mx/index p2 0)))]  ;; 0.2*gval
      ;; delta2 should be 2x delta1
      (close? (* 2.0 delta1) delta2 1e-6))))

(defspec adam-loss-decreases-on-quadratic-20-steps 10
  (prop/for-all [init gen-init]
    ;; Minimize f(x) = x^2, gradient = 2x, starting at x=init
    (let [init-params (mx/array [init])
          result (learn/train
                   {:iterations 20 :optimizer :adam :lr 0.1}
                   (fn [params _key]
                     (let [loss (mx/sum (mx/square params))
                           grad (mx/multiply (mx/scalar 2.0) params)]
                       {:loss loss :grad grad}))
                   init-params)
          history (:loss-history result)]
      (< (last history) (first history)))))

;; ---------------------------------------------------------------------------
;; Training Loop (1)
;; ---------------------------------------------------------------------------

(defspec train-final-loss-less-than-initial-loss 10
  (prop/for-all [init gen-init
                 lr gen-lr]
    (let [init-params (mx/array [init])
          result (learn/train
                   {:iterations 20 :optimizer :adam :lr lr}
                   (fn [params _key]
                     (let [loss (mx/sum (mx/square params))
                           grad (mx/multiply (mx/scalar 2.0) params)]
                       {:loss loss :grad grad}))
                   init-params)
          history (:loss-history result)]
      (< (last history) (first history)))))

(t/run-tests)
