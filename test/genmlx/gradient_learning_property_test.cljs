(ns genmlx.gradient-learning-property-test
  "Property-based tests for gradients and training loops using test.check.
   Parameter store and optimizer arithmetic tests demoted to unit_test.cljs."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.gradients :as grad]
            [genmlx.learning :as learn]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- report-result [name result]
  (if (:pass? result)
    (do (vswap! pass-count inc)
        (println "  PASS:" name (str "(" (:num-tests result) " trials)")))
    (do (vswap! fail-count inc)
        (println "  FAIL:" name)
        (println "    seed:" (:seed result))
        (when-let [s (get-in result [:shrunk :smallest])]
          (println "    shrunk:" s)))))

(defn- check [name prop & {:keys [num-tests] :or {num-tests 50}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

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

;; Pools for gradient tests — points at which to evaluate gradients
;; Avoid 0.0 for autodiff/numerical tests (division issues at exactly 0)
(def x-pool [0.5 1.0 -0.5 -1.0 2.0 -2.0])
(def gen-x (gen/elements x-pool))

;; Pools for training loop test
(def lr-pool [0.01 0.05 0.1 0.5])
(def gen-lr (gen/elements lr-pool))
(def init-pool [1.0 3.0 5.0 -3.0 10.0])
(def gen-init (gen/elements init-pool))

(println "\n=== Gradient & Learning Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Choice Gradients (2)
;; ---------------------------------------------------------------------------

(println "-- choice gradients --")

(check "choice-gradients: gradient shape is scalar []"
  (prop/for-all [xv gen-x
                 yv gen-x]
    (let [{:keys [trace]} (p/generate model [] (cm/choicemap :x (mx/scalar xv) :y (mx/scalar yv)))
          grads (grad/choice-gradients model trace [:x :y])]
      (and (= [] (mx/shape (:x grads)))
           (= [] (mx/shape (:y grads)))))))

;; Models with varying means — gradient at mode should always be 0
;; Note: choice-gradients uses mx/compile-fn which caches the computation graph,
;; so each test trial must create a fresh model to get correct gradients.
(def mu-pool [0.0 1.0 -1.0 2.5 -3.0])
(def gen-mu (gen/elements mu-pool))

(check "choice-gradients: gradient at mode near 0"
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

(println "\n-- score function gradients --")

(check "make-score-fn: gradient shape matches params"
  (prop/for-all [xv gen-x
                 yv gen-x]
    (let [obs cm/EMPTY
          sf (u/make-score-fn model [] obs [:x :y])
          grad-fn (mx/grad sf)
          params (mx/array [xv yv])
          g (grad-fn params)]
      (= (mx/shape g) [2]))))

(check "make-score-fn: gradient correct for gaussian"
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

(check "make-score-fn: autodiff near numerical gradient"
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
      (close? ad-grad num-grad 0.05)))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; Training Loop (1)
;; ---------------------------------------------------------------------------

(println "\n-- training loop --")

(check "train: final loss < initial loss"
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
      (< (last history) (first history))))
  :num-tests 10)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Gradient & Learning Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
