(ns genmlx.gradient-learning-property-test
  "Property-based tests for gradients, parameter stores, optimizers,
   and training loops using test.check."
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
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))
          y (dyn/trace :y (dist/gaussian 0 1))]
      (mx/eval! x y)
      (+ (mx/item x) (mx/item y)))))

(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

(println "\n=== Gradient & Learning Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Choice Gradients (4)
;; ---------------------------------------------------------------------------

(println "-- choice gradients --")

(check "choice-gradients: returns gradients for all requested addresses"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate model [])
          addrs [:x :y]
          grads (grad/choice-gradients model trace addrs)]
      (and (contains? grads :x)
           (contains? grads :y)))))

(check "choice-gradients: gradients are finite"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate model [])
          grads (grad/choice-gradients model trace [:x :y])]
      (and (finite? (eval-weight (:x grads)))
           (finite? (eval-weight (:y grads)))))))

(check "choice-gradients: gradient shape is scalar []"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate model [])
          grads (grad/choice-gradients model trace [:x :y])]
      (and (= [] (mx/shape (:x grads)))
           (= [] (mx/shape (:y grads)))))))

(check "choice-gradients: gradient at mode near 0"
  (prop/for-all [_ (gen/return nil)]
    ;; For gaussian(0,1), the gradient d(log p)/dx at x=0 is 0
    (let [;; Build a trace constrained at the mode
          {:keys [trace]} (p/generate model [] (cm/choicemap :x (mx/scalar 0.0) :y (mx/scalar 0.0)))
          grads (grad/choice-gradients model trace [:x :y])
          gx (eval-weight (:x grads))
          gy (eval-weight (:y grads))]
      (and (close? 0.0 gx 0.1)
           (close? 0.0 gy 0.1)))))

;; ---------------------------------------------------------------------------
;; Score Function Gradients (4)
;; Using make-score-fn + mx/grad (avoids compile-fn zero-grad issue)
;; ---------------------------------------------------------------------------

(println "\n-- score function gradients --")

(check "make-score-fn: gradient is finite"
  (prop/for-all [_ (gen/return nil)]
    (let [obs (cm/choicemap :y (mx/scalar 1.0))
          sf (u/make-score-fn model [] obs [:x])
          grad-fn (mx/grad sf)
          params (mx/array [0.5])
          g (grad-fn params)]
      (mx/eval! g)
      (finite? (mx/item (mx/index g 0))))))

(check "make-score-fn: gradient shape matches params"
  (prop/for-all [_ (gen/return nil)]
    (let [obs cm/EMPTY
          sf (u/make-score-fn model [] obs [:x :y])
          grad-fn (mx/grad sf)
          params (mx/array [0.5 0.3])
          g (grad-fn params)]
      (= (mx/shape g) [2]))))

(check "make-score-fn: gradient correct for gaussian"
  (prop/for-all [_ (gen/return nil)]
    ;; For gaussian(0,1): d(log p)/dx = -x
    ;; At x=0.5 with y=1.0 constrained: grad w.r.t. x = -0.5
    (let [obs (cm/choicemap :y (mx/scalar 1.0))
          sf (u/make-score-fn model [] obs [:x])
          grad-fn (mx/grad sf)
          params (mx/array [0.5])
          g (grad-fn params)]
      (mx/eval! g)
      (close? -0.5 (mx/item (mx/index g 0)) 0.05))))

(check "make-score-fn: autodiff near numerical gradient"
  (prop/for-all [_ (gen/return nil)]
    (let [obs (cm/choicemap :y (mx/scalar 1.0))
          x-val 0.5
          sf (u/make-score-fn model [] obs [:x])
          grad-fn (mx/grad sf)
          params (mx/array [x-val])
          g (grad-fn params)
          _ (mx/eval! g)
          ad-grad (mx/item (mx/index g 0))
          ;; Numerical gradient via finite differences
          eps 0.001
          score-fn (fn [x]
                     (let [{:keys [weight]} (p/generate model []
                                              (cm/choicemap :x (mx/scalar x) :y (mx/scalar 1.0)))]
                       (eval-weight weight)))
          s-plus (score-fn (+ x-val eps))
          s-minus (score-fn (- x-val eps))
          num-grad (/ (- s-plus s-minus) (* 2 eps))]
      (close? ad-grad num-grad 0.05)))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; Parameter Store (4)
;; ---------------------------------------------------------------------------

(println "\n-- parameter store --")

(check "param-store: get/set round-trip"
  (prop/for-all [_ (gen/return nil)]
    (let [store (learn/make-param-store)
          store (learn/set-param store :w (mx/scalar 3.14))
          v (learn/get-param store :w)]
      (close? 3.14 (eval-weight v) 1e-6))))

(check "param-store: param-names includes all stored keys"
  (prop/for-all [_ (gen/return nil)]
    (let [store (-> (learn/make-param-store)
                    (learn/set-param :a (mx/scalar 1.0))
                    (learn/set-param :b (mx/scalar 2.0))
                    (learn/set-param :c (mx/scalar 3.0)))
          names (set (learn/param-names store))]
      (and (contains? names :a)
           (contains? names :b)
           (contains? names :c)))))

(check "param-store: params->array / array->params round-trip"
  (prop/for-all [_ (gen/return nil)]
    (let [store (-> (learn/make-param-store)
                    (learn/set-param :x (mx/scalar 1.5))
                    (learn/set-param :y (mx/scalar 2.5)))
          names [:x :y]
          arr (learn/params->array store names)
          _ (mx/eval! arr)
          recovered (learn/array->params arr names)
          rx (eval-weight (:x recovered))
          ry (eval-weight (:y recovered))]
      (and (close? 1.5 rx 1e-6)
           (close? 2.5 ry 1e-6)))))

(check "param-store: empty store has no names"
  (prop/for-all [_ (gen/return nil)]
    (let [store (learn/make-param-store)]
      (empty? (learn/param-names store)))))

;; ---------------------------------------------------------------------------
;; Optimizers (4)
;; ---------------------------------------------------------------------------

(println "\n-- optimizers --")

(check "SGD: step moves params in negative gradient direction"
  (prop/for-all [_ (gen/return nil)]
    (let [params (mx/array [5.0])
          grad-arr (mx/array [1.0])  ;; positive gradient
          lr 0.1
          new-params (learn/sgd-step params grad-arr lr)]
      (mx/eval! new-params)
      ;; Should move in negative direction: 5.0 - 0.1*1.0 = 4.9
      (close? 4.9 (mx/item (mx/index new-params 0)) 1e-6))))

(check "SGD: step magnitude proportional to lr"
  (prop/for-all [_ (gen/return nil)]
    (let [params (mx/array [5.0])
          grad-arr (mx/array [2.0])
          p1 (learn/sgd-step params grad-arr 0.1)
          p2 (learn/sgd-step params grad-arr 0.2)
          _ (mx/eval! p1 p2)
          delta1 (- 5.0 (mx/item (mx/index p1 0)))  ;; 0.1*2=0.2
          delta2 (- 5.0 (mx/item (mx/index p2 0)))]  ;; 0.2*2=0.4
      ;; delta2 should be 2x delta1
      (close? (* 2.0 delta1) delta2 1e-6))))

(check "Adam: loss decreases on quadratic (20 steps)"
  (prop/for-all [_ (gen/return nil)]
    ;; Minimize f(x) = x^2, gradient = 2x, starting at x=5
    (let [init-params (mx/array [5.0])
          result (learn/train
                   {:iterations 20 :optimizer :adam :lr 0.1}
                   (fn [params _key]
                     (let [loss (mx/sum (mx/square params))
                           grad (mx/multiply (mx/scalar 2.0) params)]
                       {:loss loss :grad grad}))
                   init-params)
          history (:loss-history result)]
      (< (last history) (first history))))
  :num-tests 10)

(check "Adam: state t increments, m not all zeros after step"
  (prop/for-all [_ (gen/return nil)]
    (let [params (mx/array [5.0])
          grad-arr (mx/array [1.0])
          state (learn/adam-init params)
          [_ new-state] (learn/adam-step params grad-arr state {})]
      (and (= 1 (:t new-state))
           (let [m (:m new-state)]
             (mx/eval! m)
             (not (== 0.0 (mx/item (mx/index m 0)))))))))

;; ---------------------------------------------------------------------------
;; Training Loop (4)
;; ---------------------------------------------------------------------------

(println "\n-- training loop --")

(check "train: produces loss-history of correct length"
  (prop/for-all [n (gen/elements [5 10 15 20])]
    (let [init-params (mx/array [3.0])
          result (learn/train
                   {:iterations n :optimizer :adam :lr 0.1}
                   (fn [params _key]
                     (let [loss (mx/sum (mx/square params))
                           grad (mx/multiply (mx/scalar 2.0) params)]
                       {:loss loss :grad grad}))
                   init-params)]
      (= n (count (:loss-history result)))))
  :num-tests 20)

(check "train: final loss < initial loss"
  (prop/for-all [_ (gen/return nil)]
    (let [init-params (mx/array [5.0])
          result (learn/train
                   {:iterations 20 :optimizer :adam :lr 0.1}
                   (fn [params _key]
                     (let [loss (mx/sum (mx/square params))
                           grad (mx/multiply (mx/scalar 2.0) params)]
                       {:loss loss :grad grad}))
                   init-params)
          history (:loss-history result)]
      (< (last history) (first history))))
  :num-tests 10)

(check "train: params move toward target (distance decreases)"
  (prop/for-all [_ (gen/return nil)]
    ;; Minimize (x-2)^2, target is x=2
    (let [init-params (mx/array [10.0])
          target 2.0
          result (learn/train
                   {:iterations 30 :optimizer :adam :lr 0.1}
                   (fn [params _key]
                     (let [diff (mx/subtract params (mx/scalar target))
                           loss (mx/sum (mx/square diff))
                           grad (mx/multiply (mx/scalar 2.0) diff)]
                       {:loss loss :grad grad}))
                   init-params)
          final-params (:params result)
          _ (mx/eval! final-params)
          final-val (mx/item (mx/index final-params 0))
          init-dist (js/Math.abs (- 10.0 target))
          final-dist (js/Math.abs (- final-val target))]
      (< final-dist init-dist)))
  :num-tests 10)

(check "train: SGD optimizer also works"
  (prop/for-all [_ (gen/return nil)]
    (let [init-params (mx/array [5.0])
          result (learn/train
                   {:iterations 20 :optimizer :sgd :lr 0.1}
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
