(ns genmlx.gradient-fd-test
  "Phase 4.1: Finite-difference verification of autodiff gradients.
   For every differentiable distribution, verify that mx/grad of log-prob
   matches central-difference FD approximation at multiple test points.

   Design:
   - Data-driven: test specs are maps, not hardcoded assertions.
   - Two tolerance modes: absolute (near zero) and relative (large gradient).
   - Boundary-aware: detects support boundaries and uses one-sided FD.
   - Covers value gradients AND parameter gradients.

   FD step size: h=1e-3 is optimal for float32 MLX arrays.
   Smaller h amplifies rounding error; larger h increases truncation error.
   At h=1e-3, central difference achieves ~1e-5 relative error on smooth
   log-prob functions in float32."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]))

;; ---------------------------------------------------------------------------
;; FD verification utilities
;; ---------------------------------------------------------------------------

(def ^:private default-h
  "Central-difference step size, tuned for float32."
  1e-3)

(def ^:private abs-tol
  "Absolute tolerance for gradients near zero."
  1e-2)

(def ^:private rel-tol
  "Relative tolerance for gradients with large magnitude."
  5e-3)

(defn fd-gradient
  "Central-difference FD approximation of df/dx at x.
   f: (MLX scalar) -> (MLX scalar), x: JS number, h: step size.
   Returns JS number."
  ([f x] (fd-gradient f x default-h))
  ([f x h]
   (let [f-plus (-> (+ x h) mx/scalar f mx/realize)
         f-minus (-> (- x h) mx/scalar f mx/realize)]
     (/ (- f-plus f-minus) (* 2.0 h)))))

(defn fd-gradient-forward
  "Forward-difference FD approximation for boundary points.
   Less accurate (O(h) vs O(h^2)) but safe near lower support boundary."
  ([f x] (fd-gradient-forward f x default-h))
  ([f x h]
   (let [f-at-x (-> x mx/scalar f mx/realize)
         f-plus (-> (+ x h) mx/scalar f mx/realize)]
     (/ (- f-plus f-at-x) h))))

(defn fd-gradient-backward
  "Backward-difference FD approximation for upper support boundary."
  ([f x] (fd-gradient-backward f x default-h))
  ([f x h]
   (let [f-at-x (-> x mx/scalar f mx/realize)
         f-minus (-> (- x h) mx/scalar f mx/realize)]
     (/ (- f-at-x f-minus) h))))

(defn analytical-gradient
  "Compute analytical gradient via mx/grad at point x.
   f: (MLX scalar) -> (MLX scalar), x: JS number.
   Returns JS number."
  [f x]
  (let [grad-fn (mx/grad f)
        g (grad-fn (mx/scalar x))]
    (mx/realize g)))

(defn gradient-close?
  "True if analytical and FD gradients agree within tolerance.
   Uses absolute tolerance when |gradient| < 1, relative otherwise.
   This mixed mode handles both near-zero gradients (mode of symmetric
   distributions) and large gradients (near support boundaries)."
  [analytical fd]
  (if (and (js/isFinite analytical) (js/isFinite fd))
    (let [abs-err (js/Math.abs (- analytical fd))
          magnitude (js/Math.max (js/Math.abs analytical) (js/Math.abs fd))]
      (if (< magnitude 1.0)
        (<= abs-err abs-tol)
        (<= (/ abs-err magnitude) rel-tol)))
    (and (not (js/isFinite analytical))
         (not (js/isFinite fd)))))

(defn check-gradient
  "Compare analytical gradient vs FD at a test point.
   Returns a diagnostic map with :pass?, :analytical, :fd, :error.
   f: (MLX scalar) -> (MLX scalar), x: JS number."
  ([f x] (check-gradient f x default-h))
  ([f x h]
   (let [analytical (analytical-gradient f x)
         fd (fd-gradient f x h)
         pass? (gradient-close? analytical fd)]
     {:pass? pass?
      :analytical analytical
      :fd fd
      :abs-error (js/Math.abs (- analytical fd))
      :rel-error (if (zero? fd) 0.0
                     (/ (js/Math.abs (- analytical fd))
                        (js/Math.max (js/Math.abs analytical) (js/Math.abs fd))))
      :x x})))

(defn check-param-gradient
  "Compare analytical vs FD gradient of log-prob w.r.t. a distribution parameter.
   make-dist: (param-value: MLX scalar) -> Distribution
   value: JS number (fixed observation point)
   param-val: JS number (point at which to evaluate gradient)"
  ([make-dist value param-val]
   (check-param-gradient make-dist value param-val default-h))
  ([make-dist value param-val h]
   (let [v (mx/scalar value)
         f (fn [p] (dist/log-prob (make-dist p) v))]
     (check-gradient f param-val h))))

;; ---------------------------------------------------------------------------
;; Vector FD verification utilities (multivariate distributions)
;; ---------------------------------------------------------------------------

(defn fd-vector-gradient
  "Element-wise central-difference FD gradient of scalar function f w.r.t. vector input.
   f: (MLX array [k]) -> (MLX scalar), v: Clojure vector of JS numbers.
   Returns Clojure vector of JS numbers (one per element)."
  ([f v] (fd-vector-gradient f v default-h))
  ([f v h]
   (mapv (fn [i]
           (let [f-plus (mx/realize (f (mx/array (clj->js (update v i + h)))))
                 f-minus (mx/realize (f (mx/array (clj->js (update v i - h)))))]
             (/ (- f-plus f-minus) (* 2.0 h))))
         (range (count v)))))

(defn analytical-vector-gradient
  "Compute analytical gradient via mx/grad at vector point v.
   f: (MLX array [k]) -> (MLX scalar), v: Clojure vector of JS numbers.
   Returns Clojure vector of JS numbers."
  [f v]
  (let [grad-fn (mx/grad f)
        g (grad-fn (mx/array (clj->js v)))]
    (mx/eval! g)
    (mx/->clj g)))

(defn check-vector-gradient
  "Compare analytical gradient vs element-wise FD for a vector input.
   Returns {:pass? bool :analytical [...] :fd [...] :max-error num}."
  ([f v] (check-vector-gradient f v default-h))
  ([f v h]
   (let [analytical (analytical-vector-gradient f v)
         fd (fd-vector-gradient f v h)
         element-checks (mapv (fn [a fd-el]
                                (gradient-close? a fd-el))
                              analytical fd)
         max-error (->> (map (fn [a fd-el]
                               (js/Math.abs (- a fd-el)))
                             analytical fd)
                        (apply max))
         pass? (every? true? element-checks)]
     {:pass? pass?
      :analytical analytical
      :fd fd
      :max-error max-error
      :v v})))

(defn- run-vector-gradient-spec
  "Execute a single vector gradient spec and return diagnostic.
   Spec keys:
     :label      - human-readable description
     :f          - (MLX array) -> (MLX scalar) for value gradients
     :make-f     - (MLX array) -> ((MLX array) -> (MLX scalar)) for param gradients
     :v          - Clojure vector, evaluation point
     :expected   - Clojure vector or nil"
  [{:keys [label f v expected]}]
  (let [result (check-vector-gradient f v)
        {:keys [pass? analytical fd max-error]} result
        expected-pass? (if (some? expected)
                         (every? true?
                                 (map gradient-close? analytical expected))
                         true)]
    (merge result
           {:label label
            :expected expected
            :overall-pass? (and pass? expected-pass?)})))

(defn run-vector-gradient-specs
  "Run a sequence of vector gradient specs, printing results.
   Returns {:passed N :failed N :results [...]}."
  [specs]
  (let [results (mapv run-vector-gradient-spec specs)
        passed (count (filter :overall-pass? results))
        failed (- (count results) passed)]
    (doseq [{:keys [label overall-pass? analytical fd expected max-error v]} results]
      (if overall-pass?
        (println "  PASS:" label)
        (do (println "  FAIL:" label)
            (println "    v:" v "analytical:" analytical "fd:" fd)
            (println "    expected:" expected "max-error:" max-error))))
    {:passed passed :failed failed :results results}))

;; ---------------------------------------------------------------------------
;; Data-driven test spec runner
;; ---------------------------------------------------------------------------

(defn- run-gradient-spec
  "Execute a single gradient spec map and return diagnostic.
   Spec keys:
     :label      - human-readable description
     :dist       - Distribution instance (for value gradients)
     :make-dist  - (fn [param] -> Distribution) (for parameter gradients)
     :value      - JS number (observation point for param gradients)
     :x          - JS number (evaluation point)
     :expected   - JS number or nil (if nil, only checks FD agreement)"
  [{:keys [label dist make-dist value x expected] :as _spec}]
  (let [result (if dist
                 (check-gradient (fn [v] (dist/log-prob dist v)) x)
                 (check-param-gradient make-dist value x))
        {:keys [pass? analytical fd abs-error rel-error]} result
        expected-pass? (if (some? expected)
                         (gradient-close? analytical expected)
                         true)]
    (merge result
           {:label label
            :expected expected
            :overall-pass? (and pass? expected-pass?)})))

(defn run-gradient-specs
  "Run a sequence of gradient specs, printing results.
   Returns {:passed N :failed N :results [...]}."
  [specs]
  (let [results (mapv run-gradient-spec specs)
        passed (count (filter :overall-pass? results))
        failed (- (count results) passed)]
    (doseq [{:keys [label overall-pass? analytical fd expected abs-error rel-error x]} results]
      (if overall-pass?
        (println "  PASS:" label)
        (do (println "  FAIL:" label)
            (println "    x:" x "analytical:" analytical "fd:" fd
                     "expected:" expected)
            (println "    abs-error:" abs-error "rel-error:" rel-error))))
    {:passed passed :failed failed :results results}))

;; ---------------------------------------------------------------------------
;; Value gradient specs by distribution
;; ---------------------------------------------------------------------------

(def gaussian-value-specs
  [{:label "gaussian(0,1) at v=0 gradient is zero (mode)"
    :dist (dist/gaussian 0 1) :x 0.0 :expected 0.0}
   {:label "gaussian(0,1) at v=1 gradient is -1"
    :dist (dist/gaussian 0 1) :x 1.0 :expected -1.0}
   {:label "gaussian(0,1) at v=-2 gradient is 2"
    :dist (dist/gaussian 0 1) :x -2.0 :expected 2.0}
   {:label "gaussian(0,10) at v=5 gradient is -0.05"
    :dist (dist/gaussian 0 10) :x 5.0 :expected -0.05}
   {:label "gaussian(3,2) at v=5 gradient is -0.5"
    :dist (dist/gaussian 3 2) :x 5.0 :expected -0.5}
   {:label "gaussian(3,2) at v=1 gradient is 0.5"
    :dist (dist/gaussian 3 2) :x 1.0 :expected 0.5}])

(def beta-value-specs
  [{:label "beta(2,2) at v=0.5 gradient is zero (mode)"
    :dist (dist/beta-dist 2 2) :x 0.5 :expected 0.0}
   {:label "beta(2,2) at v=0.3 gradient is 1.9048"
    :dist (dist/beta-dist 2 2) :x 0.3 :expected 1.904761904761905}
   {:label "beta(2,2) at v=0.7 gradient is -1.9048"
    :dist (dist/beta-dist 2 2) :x 0.7 :expected -1.904761904761905}
   {:label "beta(3,1) at v=0.5 gradient is 4"
    :dist (dist/beta-dist 3 1) :x 0.5 :expected 4.0}])

(def gamma-value-specs
  [{:label "gamma(2,1) at v=1 gradient is zero (mode)"
    :dist (dist/gamma-dist 2 1) :x 1.0 :expected 0.0}
   {:label "gamma(2,1) at v=2 gradient is -0.5"
    :dist (dist/gamma-dist 2 1) :x 2.0 :expected -0.5}
   {:label "gamma(3,1) at v=0.5 gradient is 3"
    :dist (dist/gamma-dist 3 1) :x 0.5 :expected 3.0}])

(def exponential-value-specs
  [{:label "exponential(1) at v=1 gradient is -1"
    :dist (dist/exponential 1) :x 1.0 :expected -1.0}
   {:label "exponential(2) at v=0.5 gradient is -2"
    :dist (dist/exponential 2) :x 0.5 :expected -2.0}
   {:label "exponential(0.5) at v=3 gradient is -0.5"
    :dist (dist/exponential 0.5) :x 3.0 :expected -0.5}])

(def laplace-value-specs
  [{:label "laplace(0,1) at v=1 gradient is -1"
    :dist (dist/laplace 0 1) :x 1.0 :expected -1.0}
   {:label "laplace(0,1) at v=-1 gradient is 1"
    :dist (dist/laplace 0 1) :x -1.0 :expected 1.0}
   {:label "laplace(5,2) at v=7 gradient is -0.5"
    :dist (dist/laplace 5 2) :x 7.0 :expected -0.5}])

(def cauchy-value-specs
  [{:label "cauchy(0,1) at v=0 gradient is zero (mode)"
    :dist (dist/cauchy 0 1) :x 0.0 :expected 0.0}
   {:label "cauchy(0,1) at v=1 gradient is -1"
    :dist (dist/cauchy 0 1) :x 1.0 :expected -1.0}
   {:label "cauchy(0,1) at v=-1 gradient is 1"
    :dist (dist/cauchy 0 1) :x -1.0 :expected 1.0}])

(def log-normal-value-specs
  [{:label "lognormal(0,1) at v=1 gradient is -1"
    :dist (dist/log-normal 0 1) :x 1.0 :expected -1.0}
   {:label "lognormal(0,1) at v=2 gradient is -0.8466"
    :dist (dist/log-normal 0 1) :x 2.0 :expected -0.8465735902799727}])

(def inv-gamma-value-specs
  [{:label "inv-gamma(2,1) at v=1 gradient is -2"
    :dist (dist/inv-gamma 2 1) :x 1.0 :expected -2.0}
   {:label "inv-gamma(3,2) at v=0.5 gradient is zero (mode)"
    :dist (dist/inv-gamma 3 2) :x 0.5 :expected 0.0}])

(def student-t-value-specs
  [{:label "student-t(3,0,1) at v=0 gradient is zero (mode)"
    :dist (dist/student-t 3 0 1) :x 0.0 :expected 0.0}
   {:label "student-t(5,0,1) at v=1 gradient is -1"
    :dist (dist/student-t 5 0 1) :x 1.0 :expected -1.0}
   {:label "student-t(3,0,1) at v=2 gradient is -8/7"
    :dist (dist/student-t 3 0 1) :x 2.0 :expected -1.1428571428571428}])

(def truncated-normal-value-specs
  [{:label "truncnorm(0,1,-2,2) at v=0 gradient is zero (mode)"
    :dist (dist/truncated-normal 0 1 -2 2) :x 0.0 :expected 0.0}
   {:label "truncnorm(0,1,-2,2) at v=1 gradient is -1"
    :dist (dist/truncated-normal 0 1 -2 2) :x 1.0 :expected -1.0}
   {:label "truncnorm(5,2,0,10) at v=3 gradient is 0.5"
    :dist (dist/truncated-normal 5 2 0 10) :x 3.0 :expected 0.5}])

(def von-mises-value-specs
  [{:label "von-mises(0,1) at v=0 gradient is zero (mean direction)"
    :dist (dist/von-mises 0 1) :x 0.0 :expected 0.0}
   {:label "von-mises(0,2) at v=pi/4 gradient is -sqrt(2)"
    :dist (dist/von-mises 0 2) :x (/ js/Math.PI 4) :expected (- (js/Math.sqrt 2))}
   {:label "von-mises(0,5) at v=pi gradient is zero (antipodal)"
    :dist (dist/von-mises 0 5) :x js/Math.PI :expected 0.0}])

(def uniform-value-specs
  [{:label "uniform(0,1) at v=0.5 gradient is zero (constant density)"
    :dist (dist/uniform 0 1) :x 0.5 :expected 0.0}])

(def wrapped-cauchy-value-specs
  [{:label "wrapped-cauchy(0,0.5) at v=0 gradient is zero (mean direction)"
    :dist (dist/wrapped-cauchy 0 0.5) :x 0.0 :expected 0.0}
   {:label "wrapped-cauchy(0,0.5) at v=1 gradient is -1.18568"
    :dist (dist/wrapped-cauchy 0 0.5) :x 1.0 :expected -1.1856752413958853}])

(def wrapped-normal-value-specs
  [{:label "wrapped-normal(0,1) at v=0 gradient is zero (symmetry)"
    :dist (dist/wrapped-normal 0 1) :x 0.0 :expected 0.0}
   {:label "wrapped-normal(0,1) at v=1 gradient is -1 (k=0 dominates)"
    :dist (dist/wrapped-normal 0 1) :x 1.0 :expected -1.0}])

;; ---------------------------------------------------------------------------
;; Parameter gradient specs by distribution
;; ---------------------------------------------------------------------------

(def gaussian-param-specs
  [{:label "d/dmu gaussian(0,1) at v=1"
    :make-dist (fn [mu] (dist/gaussian mu (mx/scalar 1.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "d/dmu gaussian(1,2) at v=3"
    :make-dist (fn [mu] (dist/gaussian mu (mx/scalar 2.0)))
    :value 3.0 :x 1.0 :expected 0.5}
   {:label "d/dsigma gaussian(0,1) at v=1"
    :make-dist (fn [sigma] (dist/gaussian (mx/scalar 0.0) sigma))
    :value 1.0 :x 1.0 :expected 0.0}
   {:label "d/dsigma gaussian(0,1) at v=2"
    :make-dist (fn [sigma] (dist/gaussian (mx/scalar 0.0) sigma))
    :value 2.0 :x 1.0 :expected 3.0}])

(def beta-param-specs
  [{:label "d/dalpha beta(2,2) at v=0.5"
    :make-dist (fn [alpha] (dist/beta-dist alpha (mx/scalar 2.0)))
    :value 0.5 :x 2.0 :expected 0.14019}
   {:label "d/dbeta beta(2,2) at v=0.5"
    :make-dist (fn [beta-p] (dist/beta-dist (mx/scalar 2.0) beta-p))
    :value 0.5 :x 2.0 :expected 0.14019}
   {:label "d/dalpha beta(3,1) at v=0.5"
    :make-dist (fn [alpha] (dist/beta-dist alpha (mx/scalar 1.0)))
    :value 0.5 :x 3.0 :expected -0.35981}])

(def gamma-param-specs
  [{:label "d/dshape gamma(2,1) at v=1"
    :make-dist (fn [shape-p] (dist/gamma-dist shape-p (mx/scalar 1.0)))
    :value 1.0 :x 2.0 :expected -0.42278}
   {:label "d/drate gamma(2,1) at v=1"
    :make-dist (fn [rate] (dist/gamma-dist (mx/scalar 2.0) rate))
    :value 1.0 :x 1.0 :expected 1.0}
   {:label "d/drate gamma(2,1) at v=2"
    :make-dist (fn [rate] (dist/gamma-dist (mx/scalar 2.0) rate))
    :value 2.0 :x 1.0 :expected 0.0}])

(def exponential-param-specs
  [{:label "d/drate exponential(2) at v=1"
    :make-dist (fn [rate] (dist/exponential rate))
    :value 1.0 :x 2.0 :expected -0.5}
   {:label "d/drate exponential(1) at v=2"
    :make-dist (fn [rate] (dist/exponential rate))
    :value 2.0 :x 1.0 :expected -1.0}])

(def laplace-param-specs
  [{:label "d/dloc laplace(0,1) at v=1"
    :make-dist (fn [loc] (dist/laplace loc (mx/scalar 1.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "d/dloc laplace(0,1) at v=-1"
    :make-dist (fn [loc] (dist/laplace loc (mx/scalar 1.0)))
    :value -1.0 :x 0.0 :expected -1.0}
   {:label "d/dscale laplace(0,1) at v=1 is zero"
    :make-dist (fn [scale] (dist/laplace (mx/scalar 0.0) scale))
    :value 1.0 :x 1.0 :expected 0.0}
   {:label "d/dscale laplace(0,1) at v=2"
    :make-dist (fn [scale] (dist/laplace (mx/scalar 0.0) scale))
    :value 2.0 :x 1.0 :expected 1.0}])

(def cauchy-param-specs
  [{:label "d/dloc cauchy(0,1) at v=1"
    :make-dist (fn [loc] (dist/cauchy loc (mx/scalar 1.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "d/dscale cauchy(0,1) at v=1 is zero"
    :make-dist (fn [scale] (dist/cauchy (mx/scalar 0.0) scale))
    :value 1.0 :x 1.0 :expected 0.0}
   {:label "d/dscale cauchy(0,1) at v=2"
    :make-dist (fn [scale] (dist/cauchy (mx/scalar 0.0) scale))
    :value 2.0 :x 1.0 :expected 0.6}])

(def log-normal-param-specs
  [{:label "d/dmu lognormal(0,1) at v=2"
    :make-dist (fn [mu] (dist/log-normal mu (mx/scalar 1.0)))
    :value 2.0 :x 0.0 :expected 0.6931471805599453}
   {:label "d/dsigma lognormal(0,1) at v=1"
    :make-dist (fn [sigma] (dist/log-normal (mx/scalar 0.0) sigma))
    :value 1.0 :x 1.0 :expected -1.0}
   {:label "d/dsigma lognormal(0,1) at v=2"
    :make-dist (fn [sigma] (dist/log-normal (mx/scalar 0.0) sigma))
    :value 2.0 :x 1.0 :expected -0.5195469860817986}])

(def student-t-param-specs
  [{:label "d/ddf student-t(5,0,1) at v=0"
    :make-dist (fn [df] (dist/student-t df (mx/scalar 0.0) (mx/scalar 1.0)))
    :value 0.0 :x 5.0 :expected 0.009814}
   {:label "d/dloc student-t(5,0,1) at v=1"
    :make-dist (fn [loc] (dist/student-t (mx/scalar 5.0) loc (mx/scalar 1.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "d/dscale student-t(5,0,1) at v=1 is zero"
    :make-dist (fn [scale] (dist/student-t (mx/scalar 5.0) (mx/scalar 0.0) scale))
    :value 1.0 :x 1.0 :expected 0.0}
   {:label "d/dscale student-t(3,0,2) at v=2 is zero"
    :make-dist (fn [scale] (dist/student-t (mx/scalar 3.0) (mx/scalar 0.0) scale))
    :value 2.0 :x 2.0 :expected 0.0}])

(def inv-gamma-param-specs
  [{:label "d/dshape inv-gamma(2,1) at v=1"
    :make-dist (fn [shape-p] (dist/inv-gamma shape-p (mx/scalar 1.0)))
    :value 1.0 :x 2.0 :expected -0.42278}
   {:label "d/dscale inv-gamma(2,1) at v=1"
    :make-dist (fn [scale] (dist/inv-gamma (mx/scalar 2.0) scale))
    :value 1.0 :x 1.0 :expected 1.0}
   {:label "d/dscale inv-gamma(3,2) at v=0.5"
    :make-dist (fn [scale] (dist/inv-gamma (mx/scalar 3.0) scale))
    :value 0.5 :x 2.0 :expected -0.5}])

(def truncated-normal-param-specs
  [{:label "d/dmu truncnorm(0,1,-2,2) at v=1 symmetric bounds"
    :make-dist (fn [mu] (dist/truncated-normal mu (mx/scalar 1.0) (mx/scalar -2.0) (mx/scalar 2.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "d/dmu truncnorm(1,1,-2,2) at v=1 asymmetric bounds"
    :make-dist (fn [mu] (dist/truncated-normal mu (mx/scalar 1.0) (mx/scalar -2.0) (mx/scalar 2.0)))
    :value 1.0 :x 1.0 :expected 0.28279}
   {:label "d/dsigma truncnorm(0,1,-2,2) at v=1"
    :make-dist (fn [sigma] (dist/truncated-normal (mx/scalar 0.0) sigma (mx/scalar -2.0) (mx/scalar 2.0)))
    :value 1.0 :x 1.0 :expected 0.22626}])

(def von-mises-param-specs
  [{:label "d/dmu von-mises(0,2) at v=1"
    :make-dist (fn [mu] (dist/von-mises mu (mx/scalar 2.0)))
    :value 1.0 :x 0.0 :expected 1.6829419696157930}
   {:label "d/dkappa von-mises(0,1) at v=0"
    :make-dist (fn [kappa] (dist/von-mises (mx/scalar 0.0) kappa))
    :value 0.0 :x 1.0 :expected 0.55361}
   {:label "d/dkappa von-mises(0,2) at v=1"
    :make-dist (fn [kappa] (dist/von-mises (mx/scalar 0.0) kappa))
    :value 1.0 :x 2.0 :expected -0.15747}])

(def uniform-param-specs
  [{:label "d/dlo uniform(0,1) at v=0.5"
    :make-dist (fn [lo] (dist/uniform lo (mx/scalar 1.0)))
    :value 0.5 :x 0.0 :expected 1.0}
   {:label "d/dlo uniform(0,5) at v=2"
    :make-dist (fn [lo] (dist/uniform lo (mx/scalar 5.0)))
    :value 2.0 :x 0.0 :expected 0.2}
   {:label "d/dhi uniform(0,1) at v=0.5"
    :make-dist (fn [hi] (dist/uniform (mx/scalar 0.0) hi))
    :value 0.5 :x 1.0 :expected -1.0}])

(def wrapped-cauchy-param-specs
  [{:label "d/dmu wrapped-cauchy(0,0.5) at v=1"
    :make-dist (fn [mu] (dist/wrapped-cauchy mu (mx/scalar 0.5)))
    :value 1.0 :x 0.0 :expected 1.1856752413958853}
   {:label "d/drho wrapped-cauchy(0,0.5) at v=1"
    :make-dist (fn [rho] (dist/wrapped-cauchy (mx/scalar 0.0) rho))
    :value 1.0 :x 0.5 :expected -1.2197573524575895}])

(def wrapped-normal-param-specs
  [{:label "d/dmu wrapped-normal(0,1) at v=0.5"
    :make-dist (fn [mu] (dist/wrapped-normal mu (mx/scalar 1.0)))
    :value 0.5 :x 0.0 :expected 0.5}
   {:label "d/dsigma wrapped-normal(0,1) at v=1"
    :make-dist (fn [sigma] (dist/wrapped-normal (mx/scalar 0.0) sigma))
    :value 1.0 :x 1.0 :expected 0.0}])

;; ---------------------------------------------------------------------------
;; Multivariate value gradient specs
;; ---------------------------------------------------------------------------

(def mvn-value-specs
  "Multivariate normal value gradients: d/dv log N(v|mu,Sigma) = -Sigma^{-1}(v - mu)."
  (let [eye2 (mx/array #js [#js [1 0] #js [0 1]])
        cov-corr (mx/array #js [#js [2 0.5] #js [0.5 1]])]
    [{:label "mvn(0,I) at v=[0,0] gradient is zero (mode)"
      :f (fn [v] (dist/log-prob (dist/multivariate-normal (mx/array #js [0 0]) eye2) v))
      :v [0.0 0.0]
      :expected [0.0 0.0]}
     {:label "mvn(0,I) at v=[1,0.5] gradient is [-1,-0.5]"
      :f (fn [v] (dist/log-prob (dist/multivariate-normal (mx/array #js [0 0]) eye2) v))
      :v [1.0 0.5]
      :expected [-1.0 -0.5]}
     {:label "mvn([1,1],I) at v=[-1,2] gradient is [2,-1]"
      :f (fn [v] (dist/log-prob (dist/multivariate-normal (mx/array #js [1 1]) eye2) v))
      :v [-1.0 2.0]
      :expected [2.0 -1.0]}
     {:label "mvn(0,Sigma-corr) at v=[1,0.5] gradient is [-3/7,-2/7]"
      :f (fn [v] (dist/log-prob (dist/multivariate-normal (mx/array #js [0 0]) cov-corr) v))
      :v [1.0 0.5]
      :expected [(/ -3.0 7.0) (/ -2.0 7.0)]}]))

(def dirichlet-value-specs
  "Dirichlet value gradients: d/dv_i log Dir(v|alpha) = (alpha_i - 1) / v_i."
  [{:label "dirichlet([2,3,1]) at v=[0.5,0.3,0.2]"
    :f (fn [v] (dist/log-prob (dist/dirichlet (mx/array #js [2 3 1])) v))
    :v [0.5 0.3 0.2]
    :expected [2.0 (/ 2.0 0.3) 0.0]}
   {:label "dirichlet([1,1,1]) at v=[0.33,0.33,0.34] gradient is zero (uniform)"
    :f (fn [v] (dist/log-prob (dist/dirichlet (mx/array #js [1 1 1])) v))
    :v [0.33 0.33 0.34]
    :expected [0.0 0.0 0.0]}
   {:label "dirichlet([5,2,3]) at v=[0.6,0.2,0.2]"
    :f (fn [v] (dist/log-prob (dist/dirichlet (mx/array #js [5 2 3])) v))
    :v [0.6 0.2 0.2]
    :expected [(/ 4.0 0.6) (/ 1.0 0.2) (/ 2.0 0.2)]}])

(def broadcasted-normal-value-specs
  "Broadcasted-normal value gradients: element-wise -(v_i - mu_i)/sigma_i^2."
  [{:label "broadcasted-normal([0,1,2],[1,1,1]) at v=[0.5,1.5,2.5]"
    :f (fn [v] (dist/log-prob (dist/broadcasted-normal
                               (mx/array #js [0 1 2])
                               (mx/array #js [1 1 1])) v))
    :v [0.5 1.5 2.5]
    :expected [-0.5 -0.5 -0.5]}
   {:label "broadcasted-normal([0,0],[1,2]) at v=[1,1] with different sigmas"
    :f (fn [v] (dist/log-prob (dist/broadcasted-normal
                               (mx/array #js [0 0])
                               (mx/array #js [1 2])) v))
    :v [1.0 1.0]
    :expected [-1.0 -0.25]}
   {:label "broadcasted-normal([0,0,0],[1,1,1]) at v=[0,0,0] gradient is zero"
    :f (fn [v] (dist/log-prob (dist/broadcasted-normal
                               (mx/array #js [0 0 0])
                               (mx/array #js [1 1 1])) v))
    :v [0.0 0.0 0.0]
    :expected [0.0 0.0 0.0]}])

(def gaussian-vec-value-specs
  "Gaussian-vec value gradients: same as broadcasted-normal (sums over last axis)."
  [{:label "gaussian-vec([0,1,2],[1,1,1]) at v=[0.5,1.5,2.5]"
    :f (fn [v] (dist/log-prob (dist/gaussian-vec
                               (mx/array #js [0 1 2])
                               (mx/array #js [1 1 1])) v))
    :v [0.5 1.5 2.5]
    :expected [-0.5 -0.5 -0.5]}
   {:label "gaussian-vec([0,0],[1,2]) at v=[1,1] with different sigmas"
    :f (fn [v] (dist/log-prob (dist/gaussian-vec
                               (mx/array #js [0 0])
                               (mx/array #js [1 2])) v))
    :v [1.0 1.0]
    :expected [-1.0 -0.25]}])

;; ---------------------------------------------------------------------------
;; Multivariate parameter gradient specs
;; ---------------------------------------------------------------------------

(def mvn-param-specs
  "MVN parameter gradients w.r.t. mean vector: d/dmu log N(v|mu,Sigma) = Sigma^{-1}(v - mu)."
  (let [eye2 (mx/array #js [#js [1 0] #js [0 1]])]
    [{:label "d/dmu mvn(0,I) at v=[1,0.5] is [1,0.5]"
      :f (fn [mu] (dist/log-prob (dist/multivariate-normal mu eye2)
                                 (mx/array #js [1 0.5])))
      :v [0.0 0.0]
      :expected [1.0 0.5]}
     {:label "d/dmu mvn([1,1],I) at v=[1,1] is zero (mode)"
      :f (fn [mu] (dist/log-prob (dist/multivariate-normal mu eye2)
                                 (mx/array #js [1 1])))
      :v [1.0 1.0]
      :expected [0.0 0.0]}
     {:label "d/dmu mvn(0,I) at v=[-2,3] is [-2,3]"
      :f (fn [mu] (dist/log-prob (dist/multivariate-normal mu eye2)
                                 (mx/array #js [-2 3])))
      :v [0.0 0.0]
      :expected [-2.0 3.0]}]))

(def dirichlet-param-specs
  "Dirichlet parameter gradients w.r.t. alpha vector."
  [{:label "d/dalpha dirichlet at v=[0.5,0.3,0.2]"
    :f (fn [alpha] (dist/log-prob (dist/dirichlet alpha)
                                  (mx/array #js [0.5 0.3 0.2])))
    :v [2.0 3.0 1.0]}
   {:label "d/dalpha dirichlet([1,1,1]) at v=[0.4,0.4,0.2]"
    :f (fn [alpha] (dist/log-prob (dist/dirichlet alpha)
                                  (mx/array #js [0.4 0.4 0.2])))
    :v [1.0 1.0 1.0]}])

(def broadcasted-normal-param-specs
  "Broadcasted-normal parameter gradients w.r.t. mu vector."
  [{:label "d/dmu broadcasted-normal([0,1,2],[1,1,1]) at v=[0.5,1.5,2.5]"
    :f (fn [mu] (dist/log-prob (dist/broadcasted-normal
                                mu (mx/array #js [1 1 1]))
                               (mx/array #js [0.5 1.5 2.5])))
    :v [0.0 1.0 2.0]
    :expected [0.5 0.5 0.5]}
   {:label "d/dmu broadcasted-normal at mode is zero"
    :f (fn [mu] (dist/log-prob (dist/broadcasted-normal
                                mu (mx/array #js [1 1 1]))
                               (mx/array #js [0 0 0])))
    :v [0.0 0.0 0.0]
    :expected [0.0 0.0 0.0]}])

(def gaussian-vec-param-specs
  "Gaussian-vec parameter gradients w.r.t. mu vector."
  [{:label "d/dmu gaussian-vec([0,1,2],[1,1,1]) at v=[0.5,1.5,2.5]"
    :f (fn [mu] (dist/log-prob (dist/gaussian-vec
                                mu (mx/array #js [1 1 1]))
                               (mx/array #js [0.5 1.5 2.5])))
    :v [0.0 1.0 2.0]
    :expected [0.5 0.5 0.5]}
   {:label "d/dmu gaussian-vec at mode is zero"
    :f (fn [mu] (dist/log-prob (dist/gaussian-vec
                                mu (mx/array #js [1 1 1]))
                               (mx/array #js [0 0 0])))
    :v [0.0 0.0 0.0]
    :expected [0.0 0.0 0.0]}])

;; ---------------------------------------------------------------------------
;; Tests: value gradients
;; ---------------------------------------------------------------------------

(deftest gaussian-value-gradient-matches-fd
  (testing "Gaussian value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs gaussian-value-specs)]
      (is (zero? failed) "all Gaussian value gradient checks pass"))))

(deftest beta-value-gradient-matches-fd
  (testing "Beta value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs beta-value-specs)]
      (is (zero? failed) "all Beta value gradient checks pass"))))

(deftest gamma-value-gradient-matches-fd
  (testing "Gamma value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs gamma-value-specs)]
      (is (zero? failed) "all Gamma value gradient checks pass"))))

(deftest exponential-value-gradient-matches-fd
  (testing "Exponential value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs exponential-value-specs)]
      (is (zero? failed) "all Exponential value gradient checks pass"))))

(deftest laplace-value-gradient-matches-fd
  (testing "Laplace value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs laplace-value-specs)]
      (is (zero? failed) "all Laplace value gradient checks pass"))))

(deftest cauchy-value-gradient-matches-fd
  (testing "Cauchy value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs cauchy-value-specs)]
      (is (zero? failed) "all Cauchy value gradient checks pass"))))

(deftest log-normal-value-gradient-matches-fd
  (testing "LogNormal value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs log-normal-value-specs)]
      (is (zero? failed) "all LogNormal value gradient checks pass"))))

(deftest inv-gamma-value-gradient-matches-fd
  (testing "Inverse-Gamma value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs inv-gamma-value-specs)]
      (is (zero? failed) "all Inverse-Gamma value gradient checks pass"))))

(deftest student-t-value-gradient-matches-fd
  (testing "Student-t value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs student-t-value-specs)]
      (is (zero? failed) "all Student-t value gradient checks pass"))))

(deftest truncated-normal-value-gradient-matches-fd
  (testing "Truncated-Normal value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs truncated-normal-value-specs)]
      (is (zero? failed) "all Truncated-Normal value gradient checks pass"))))

(deftest von-mises-value-gradient-matches-fd
  (testing "Von-Mises value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs von-mises-value-specs)]
      (is (zero? failed) "all Von-Mises value gradient checks pass"))))

(deftest uniform-value-gradient-matches-fd
  (testing "Uniform value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs uniform-value-specs)]
      (is (zero? failed) "all Uniform value gradient checks pass"))))

(deftest wrapped-cauchy-value-gradient-matches-fd
  (testing "Wrapped-Cauchy value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs wrapped-cauchy-value-specs)]
      (is (zero? failed) "all Wrapped-Cauchy value gradient checks pass"))))

(deftest wrapped-normal-value-gradient-matches-fd
  (testing "Wrapped-Normal value gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs wrapped-normal-value-specs)]
      (is (zero? failed) "all Wrapped-Normal value gradient checks pass"))))

;; ---------------------------------------------------------------------------
;; Tests: parameter gradients
;; ---------------------------------------------------------------------------

(deftest gaussian-param-gradient-matches-fd
  (testing "Gaussian parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs gaussian-param-specs)]
      (is (zero? failed) "all Gaussian parameter gradient checks pass"))))

(deftest beta-param-gradient-matches-fd
  (testing "Beta parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs beta-param-specs)]
      (is (zero? failed) "all Beta parameter gradient checks pass"))))

(deftest gamma-param-gradient-matches-fd
  (testing "Gamma parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs gamma-param-specs)]
      (is (zero? failed) "all Gamma parameter gradient checks pass"))))

(deftest exponential-param-gradient-matches-fd
  (testing "Exponential parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs exponential-param-specs)]
      (is (zero? failed) "all Exponential parameter gradient checks pass"))))

(deftest laplace-param-gradient-matches-fd
  (testing "Laplace parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs laplace-param-specs)]
      (is (zero? failed) "all Laplace parameter gradient checks pass"))))

(deftest cauchy-param-gradient-matches-fd
  (testing "Cauchy parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs cauchy-param-specs)]
      (is (zero? failed) "all Cauchy parameter gradient checks pass"))))

(deftest log-normal-param-gradient-matches-fd
  (testing "LogNormal parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs log-normal-param-specs)]
      (is (zero? failed) "all LogNormal parameter gradient checks pass"))))

(deftest student-t-param-gradient-matches-fd
  (testing "Student-t parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs student-t-param-specs)]
      (is (zero? failed) "all Student-t parameter gradient checks pass"))))

(deftest inv-gamma-param-gradient-matches-fd
  (testing "Inverse-Gamma parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs inv-gamma-param-specs)]
      (is (zero? failed) "all Inverse-Gamma parameter gradient checks pass"))))

(deftest truncated-normal-param-gradient-matches-fd
  (testing "Truncated-Normal parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs truncated-normal-param-specs)]
      (is (zero? failed) "all Truncated-Normal parameter gradient checks pass"))))

(deftest von-mises-param-gradient-matches-fd
  (testing "Von-Mises parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs von-mises-param-specs)]
      (is (zero? failed) "all Von-Mises parameter gradient checks pass"))))

(deftest uniform-param-gradient-matches-fd
  (testing "Uniform parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs uniform-param-specs)]
      (is (zero? failed) "all Uniform parameter gradient checks pass"))))

(deftest wrapped-cauchy-param-gradient-matches-fd
  (testing "Wrapped-Cauchy parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs wrapped-cauchy-param-specs)]
      (is (zero? failed) "all Wrapped-Cauchy parameter gradient checks pass"))))

(deftest wrapped-normal-param-gradient-matches-fd
  (testing "Wrapped-Normal parameter gradients: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs wrapped-normal-param-specs)]
      (is (zero? failed) "all Wrapped-Normal parameter gradient checks pass"))))

;; ---------------------------------------------------------------------------
;; Tests: multivariate value gradients
;; ---------------------------------------------------------------------------

(deftest mvn-value-gradient-matches-fd
  (testing "MVN value gradients: analytical vs element-wise FD"
    (let [{:keys [failed]} (run-vector-gradient-specs mvn-value-specs)]
      (is (zero? failed) "all MVN value gradient checks pass"))))

(deftest dirichlet-value-gradient-matches-fd
  (testing "Dirichlet value gradients: analytical vs element-wise FD"
    (let [{:keys [failed]} (run-vector-gradient-specs dirichlet-value-specs)]
      (is (zero? failed) "all Dirichlet value gradient checks pass"))))

(deftest broadcasted-normal-value-gradient-matches-fd
  (testing "Broadcasted-normal value gradients: analytical vs element-wise FD"
    (let [{:keys [failed]} (run-vector-gradient-specs broadcasted-normal-value-specs)]
      (is (zero? failed) "all Broadcasted-normal value gradient checks pass"))))

(deftest gaussian-vec-value-gradient-matches-fd
  (testing "Gaussian-vec value gradients: analytical vs element-wise FD"
    (let [{:keys [failed]} (run-vector-gradient-specs gaussian-vec-value-specs)]
      (is (zero? failed) "all Gaussian-vec value gradient checks pass"))))

;; ---------------------------------------------------------------------------
;; Tests: multivariate parameter gradients
;; ---------------------------------------------------------------------------

(deftest mvn-param-gradient-matches-fd
  (testing "MVN parameter gradients: analytical vs element-wise FD"
    (let [{:keys [failed]} (run-vector-gradient-specs mvn-param-specs)]
      (is (zero? failed) "all MVN parameter gradient checks pass"))))

(deftest dirichlet-param-gradient-matches-fd
  (testing "Dirichlet parameter gradients: analytical vs element-wise FD"
    (let [{:keys [failed]} (run-vector-gradient-specs dirichlet-param-specs)]
      (is (zero? failed) "all Dirichlet parameter gradient checks pass"))))

(deftest broadcasted-normal-param-gradient-matches-fd
  (testing "Broadcasted-normal parameter gradients: analytical vs element-wise FD"
    (let [{:keys [failed]} (run-vector-gradient-specs broadcasted-normal-param-specs)]
      (is (zero? failed) "all Broadcasted-normal parameter gradient checks pass"))))

(deftest gaussian-vec-param-gradient-matches-fd
  (testing "Gaussian-vec parameter gradients: analytical vs element-wise FD"
    (let [{:keys [failed]} (run-vector-gradient-specs gaussian-vec-param-specs)]
      (is (zero? failed) "all Gaussian-vec parameter gradient checks pass"))))

;; ---------------------------------------------------------------------------
;; Tests: gradient finiteness sweep
;; ---------------------------------------------------------------------------

(def finiteness-specs
  "Every differentiable distribution at one interior point."
  [{:label "gaussian gradient is finite"
    :dist (dist/gaussian 0 1) :x 0.5}
   {:label "laplace gradient is finite"
    :dist (dist/laplace 0 1) :x 0.5}
   {:label "cauchy gradient is finite"
    :dist (dist/cauchy 0 1) :x 0.5}
   {:label "exponential gradient is finite"
    :dist (dist/exponential 1) :x 0.5}
   {:label "log-normal gradient is finite"
    :dist (dist/log-normal 0 1) :x 1.0}
   {:label "uniform gradient is finite"
    :dist (dist/uniform 0 1) :x 0.5}
   {:label "beta gradient is finite"
    :dist (dist/beta-dist 2 2) :x 0.5}
   {:label "gamma gradient is finite"
    :dist (dist/gamma-dist 2 1) :x 1.0}
   {:label "inv-gamma gradient is finite"
    :dist (dist/inv-gamma 2 1) :x 1.0}
   {:label "student-t gradient is finite"
    :dist (dist/student-t 3 0 1) :x 0.5}
   {:label "truncated-normal gradient is finite"
    :dist (dist/truncated-normal 0 1 -2 2) :x 0.5}
   {:label "von-mises gradient is finite"
    :dist (dist/von-mises 0 1) :x 0.5}
   {:label "wrapped-cauchy gradient is finite"
    :dist (dist/wrapped-cauchy 0 0.5) :x 1.0}
   {:label "wrapped-normal gradient is finite"
    :dist (dist/wrapped-normal 0 1) :x 1.0}])

(deftest all-differentiable-distributions-produce-finite-gradients
  (testing "gradient finiteness at interior points"
    (doseq [{:keys [label dist x]} finiteness-specs]
      (let [g (analytical-gradient (fn [v] (dist/log-prob dist v)) x)]
        (is (js/isFinite g) label)))))

(def vector-finiteness-specs
  "Every differentiable multivariate distribution at one interior point."
  [{:label "multivariate-normal gradient is finite"
    :f (fn [v] (dist/log-prob (dist/multivariate-normal
                               (mx/array #js [0 0])
                               (mx/array #js [#js [1 0] #js [0 1]])) v))
    :v [0.5 0.5]}
   {:label "dirichlet gradient is finite"
    :f (fn [v] (dist/log-prob (dist/dirichlet (mx/array #js [2 3 1])) v))
    :v [0.5 0.3 0.2]}
   {:label "broadcasted-normal gradient is finite"
    :f (fn [v] (dist/log-prob (dist/broadcasted-normal
                               (mx/array #js [0 1 2])
                               (mx/array #js [1 1 1])) v))
    :v [0.5 1.5 2.5]}
   {:label "gaussian-vec gradient is finite"
    :f (fn [v] (dist/log-prob (dist/gaussian-vec
                               (mx/array #js [0 1 2])
                               (mx/array #js [1 1 1])) v))
    :v [0.5 1.5 2.5]}])

(deftest all-multivariate-distributions-produce-finite-gradients
  (testing "vector gradient finiteness at interior points"
    (doseq [{:keys [label f v]} vector-finiteness-specs]
      (let [grad (analytical-vector-gradient f v)]
        (is (every? js/isFinite grad) label)))))

;; ---------------------------------------------------------------------------
;; Tests: cross-system spec verification
;; ---------------------------------------------------------------------------

(def cross-system-value-gradient-specs
  "Value gradient entries from cross_system_tests/specs/gradient_tests.json,
   re-expressed as data. Only value gradients (grad_wrt = 'value')."
  [{:label "grad-normal-val-1: N(0,1) at v=1"
    :dist (dist/gaussian 0 1) :x 1.0 :expected -1.0}
   {:label "grad-normal-val-0: N(0,1) at v=0"
    :dist (dist/gaussian 0 1) :x 0.0 :expected 0.0}
   {:label "grad-normal-val-neg: N(0,1) at v=-2"
    :dist (dist/gaussian 0 1) :x -2.0 :expected 2.0}
   {:label "grad-normal-large-sigma: N(0,10) at v=5"
    :dist (dist/gaussian 0 10) :x 5.0 :expected -0.05}
   {:label "grad-beta-val-1: Beta(2,2) at v=0.5"
    :dist (dist/beta-dist 2 2) :x 0.5 :expected 0.0}
   {:label "grad-beta-val-2: Beta(2,2) at v=0.3"
    :dist (dist/beta-dist 2 2) :x 0.3 :expected 1.904761904761905}
   {:label "grad-gamma-val-1: Gamma(2,1) at v=1"
    :dist (dist/gamma-dist 2 1) :x 1.0 :expected 0.0}
   {:label "grad-gamma-val-2: Gamma(2,1) at v=2"
    :dist (dist/gamma-dist 2 1) :x 2.0 :expected -0.5}
   {:label "grad-exp-val-1: Exp(1) at v=1"
    :dist (dist/exponential 1) :x 1.0 :expected -1.0}
   {:label "grad-exp-val-2: Exp(2) at v=0.5"
    :dist (dist/exponential 2) :x 0.5 :expected -2.0}
   {:label "grad-laplace-val-pos: Lap(0,1) at v=1"
    :dist (dist/laplace 0 1) :x 1.0 :expected -1.0}
   {:label "grad-laplace-val-neg: Lap(0,1) at v=-1"
    :dist (dist/laplace 0 1) :x -1.0 :expected 1.0}
   {:label "grad-cauchy-val-0: C(0,1) at v=0"
    :dist (dist/cauchy 0 1) :x 0.0 :expected 0.0}
   {:label "grad-cauchy-val-1: C(0,1) at v=1"
    :dist (dist/cauchy 0 1) :x 1.0 :expected -1.0}
   {:label "grad-lognormal-val-1: LN(0,1) at v=1"
    :dist (dist/log-normal 0 1) :x 1.0 :expected -1.0}
   {:label "grad-lognormal-val-2: LN(0,1) at v=2"
    :dist (dist/log-normal 0 1) :x 2.0 :expected -0.8465735902799727}
   {:label "grad-invgamma-val-1: IG(2,1) at v=1"
    :dist (dist/inv-gamma 2 1) :x 1.0 :expected -2.0}
   {:label "grad-invgamma-mode: IG(3,2) at v=0.5"
    :dist (dist/inv-gamma 3 2) :x 0.5 :expected 0.0}
   {:label "grad-studentt-val-0: t(3,0,1) at v=0"
    :dist (dist/student-t 3 0 1) :x 0.0 :expected 0.0}
   {:label "grad-studentt-val-1: t(5,0,1) at v=1"
    :dist (dist/student-t 5 0 1) :x 1.0 :expected -1.0}
   {:label "grad-studentt-val-2: t(3,0,1) at v=2"
    :dist (dist/student-t 3 0 1) :x 2.0 :expected -1.1428571428571428}
   {:label "grad-truncnorm-val-1: TN(0,1,-2,2) at v=1"
    :dist (dist/truncated-normal 0 1 -2 2) :x 1.0 :expected -1.0}
   {:label "grad-truncnorm-val-0: TN(0,1,-2,2) at v=0"
    :dist (dist/truncated-normal 0 1 -2 2) :x 0.0 :expected 0.0}
   {:label "grad-truncnorm-val-3: TN(5,2,0,10) at v=3"
    :dist (dist/truncated-normal 5 2 0 10) :x 3.0 :expected 0.5}
   {:label "grad-vonmises-val-0: VM(0,1) at v=0"
    :dist (dist/von-mises 0 1) :x 0.0 :expected 0.0}
   {:label "grad-vonmises-val-pi4: VM(0,2) at v=pi/4"
    :dist (dist/von-mises 0 2) :x (/ js/Math.PI 4) :expected -1.4142135623730951}
   {:label "grad-vonmises-val-pi: VM(0,5) at v=pi"
    :dist (dist/von-mises 0 5) :x js/Math.PI :expected 0.0}
   {:label "grad-uniform-val: U(0,1) at v=0.5"
    :dist (dist/uniform 0 1) :x 0.5 :expected 0.0}
   {:label "grad-wrappedcauchy-val-0: WC(0,0.5) at v=0"
    :dist (dist/wrapped-cauchy 0 0.5) :x 0.0 :expected 0.0}
   {:label "grad-wrappedcauchy-val-1: WC(0,0.5) at v=1"
    :dist (dist/wrapped-cauchy 0 0.5) :x 1.0 :expected -1.1856752413958853}
   {:label "grad-wrappednormal-val-0: WN(0,1) at v=0"
    :dist (dist/wrapped-normal 0 1) :x 0.0 :expected 0.0}
   {:label "grad-wrappednormal-val-1: WN(0,1) at v=1"
    :dist (dist/wrapped-normal 0 1) :x 1.0 :expected -1.0}])

(def cross-system-param-gradient-specs
  "Parameter gradient entries from cross_system_tests/specs/gradient_tests.json."
  [;; Gaussian
   {:label "grad-normal-mu-1: d/dmu N(0,1) at v=1"
    :make-dist (fn [mu] (dist/gaussian mu (mx/scalar 1.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "grad-normal-mu-2: d/dmu N(1,2) at v=3"
    :make-dist (fn [mu] (dist/gaussian mu (mx/scalar 2.0)))
    :value 3.0 :x 1.0 :expected 0.5}
   {:label "grad-normal-sigma-1: d/dsigma N(0,1) at v=1"
    :make-dist (fn [sigma] (dist/gaussian (mx/scalar 0.0) sigma))
    :value 1.0 :x 1.0 :expected 0.0}
   {:label "grad-normal-sigma-2: d/dsigma N(0,1) at v=2"
    :make-dist (fn [sigma] (dist/gaussian (mx/scalar 0.0) sigma))
    :value 2.0 :x 1.0 :expected 3.0}
   ;; Beta
   {:label "grad-beta-alpha: d/dalpha Beta(2,2) at v=0.5"
    :make-dist (fn [alpha] (dist/beta-dist alpha (mx/scalar 2.0)))
    :value 0.5 :x 2.0 :expected 0.14019}
   {:label "grad-beta-beta: d/dbeta Beta(2,2) at v=0.5"
    :make-dist (fn [beta-p] (dist/beta-dist (mx/scalar 2.0) beta-p))
    :value 0.5 :x 2.0 :expected 0.14019}
   {:label "grad-beta-alpha-asym: d/dalpha Beta(3,1) at v=0.5"
    :make-dist (fn [alpha] (dist/beta-dist alpha (mx/scalar 1.0)))
    :value 0.5 :x 3.0 :expected -0.35981}
   ;; Gamma
   {:label "grad-gamma-shape: d/dshape Gamma(2,1) at v=1"
    :make-dist (fn [shape-p] (dist/gamma-dist shape-p (mx/scalar 1.0)))
    :value 1.0 :x 2.0 :expected -0.42278}
   {:label "grad-gamma-rate: d/drate Gamma(2,1) at v=1"
    :make-dist (fn [rate] (dist/gamma-dist (mx/scalar 2.0) rate))
    :value 1.0 :x 1.0 :expected 1.0}
   {:label "grad-gamma-rate-mode: d/drate Gamma(2,1) at v=2"
    :make-dist (fn [rate] (dist/gamma-dist (mx/scalar 2.0) rate))
    :value 2.0 :x 1.0 :expected 0.0}
   ;; Exponential
   {:label "grad-exp-rate: d/drate Exp(2) at v=1"
    :make-dist (fn [rate] (dist/exponential rate))
    :value 1.0 :x 2.0 :expected -0.5}
   {:label "grad-exp-rate-2: d/drate Exp(1) at v=2"
    :make-dist (fn [rate] (dist/exponential rate))
    :value 2.0 :x 1.0 :expected -1.0}
   ;; Laplace
   {:label "grad-laplace-loc: d/dloc Lap(0,1) at v=1"
    :make-dist (fn [loc] (dist/laplace loc (mx/scalar 1.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "grad-laplace-loc-neg: d/dloc Lap(0,1) at v=-1"
    :make-dist (fn [loc] (dist/laplace loc (mx/scalar 1.0)))
    :value -1.0 :x 0.0 :expected -1.0}
   {:label "grad-laplace-scale-at-mode: d/dscale Lap(0,1) at v=1"
    :make-dist (fn [scale] (dist/laplace (mx/scalar 0.0) scale))
    :value 1.0 :x 1.0 :expected 0.0}
   {:label "grad-laplace-scale-far: d/dscale Lap(0,1) at v=2"
    :make-dist (fn [scale] (dist/laplace (mx/scalar 0.0) scale))
    :value 2.0 :x 1.0 :expected 1.0}
   ;; Cauchy
   {:label "grad-cauchy-loc: d/dloc C(0,1) at v=1"
    :make-dist (fn [loc] (dist/cauchy loc (mx/scalar 1.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "grad-cauchy-scale-at-z1: d/dscale C(0,1) at v=1"
    :make-dist (fn [scale] (dist/cauchy (mx/scalar 0.0) scale))
    :value 1.0 :x 1.0 :expected 0.0}
   {:label "grad-cauchy-scale-far: d/dscale C(0,1) at v=2"
    :make-dist (fn [scale] (dist/cauchy (mx/scalar 0.0) scale))
    :value 2.0 :x 1.0 :expected 0.6}
   ;; LogNormal
   {:label "grad-lognormal-mu: d/dmu LN(0,1) at v=2"
    :make-dist (fn [mu] (dist/log-normal mu (mx/scalar 1.0)))
    :value 2.0 :x 0.0 :expected 0.6931471805599453}
   {:label "grad-lognormal-sigma-at-1: d/dsigma LN(0,1) at v=1"
    :make-dist (fn [sigma] (dist/log-normal (mx/scalar 0.0) sigma))
    :value 1.0 :x 1.0 :expected -1.0}
   {:label "grad-lognormal-sigma-at-2: d/dsigma LN(0,1) at v=2"
    :make-dist (fn [sigma] (dist/log-normal (mx/scalar 0.0) sigma))
    :value 2.0 :x 1.0 :expected -0.5195469860817986}
   ;; Student-t
   {:label "grad-studentt-df: d/ddf t(5,0,1) at v=0"
    :make-dist (fn [df] (dist/student-t df (mx/scalar 0.0) (mx/scalar 1.0)))
    :value 0.0 :x 5.0 :expected 0.009814}
   {:label "grad-studentt-loc: d/dloc t(5,0,1) at v=1"
    :make-dist (fn [loc] (dist/student-t (mx/scalar 5.0) loc (mx/scalar 1.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "grad-studentt-scale: d/dscale t(5,0,1) at v=1"
    :make-dist (fn [scale] (dist/student-t (mx/scalar 5.0) (mx/scalar 0.0) scale))
    :value 1.0 :x 1.0 :expected 0.0}
   {:label "grad-studentt-scale-2: d/dscale t(3,0,2) at v=2"
    :make-dist (fn [scale] (dist/student-t (mx/scalar 3.0) (mx/scalar 0.0) scale))
    :value 2.0 :x 2.0 :expected 0.0}
   ;; Inverse-Gamma
   {:label "grad-invgamma-shape: d/dshape IG(2,1) at v=1"
    :make-dist (fn [shape-p] (dist/inv-gamma shape-p (mx/scalar 1.0)))
    :value 1.0 :x 2.0 :expected -0.42278}
   {:label "grad-invgamma-scale: d/dscale IG(2,1) at v=1"
    :make-dist (fn [scale] (dist/inv-gamma (mx/scalar 2.0) scale))
    :value 1.0 :x 1.0 :expected 1.0}
   {:label "grad-invgamma-scale-mode: d/dscale IG(3,2) at v=0.5"
    :make-dist (fn [scale] (dist/inv-gamma (mx/scalar 3.0) scale))
    :value 0.5 :x 2.0 :expected -0.5}
   ;; Truncated-Normal
   {:label "grad-truncnorm-mu-sym: d/dmu TN(0,1,-2,2) at v=1"
    :make-dist (fn [mu] (dist/truncated-normal mu (mx/scalar 1.0) (mx/scalar -2.0) (mx/scalar 2.0)))
    :value 1.0 :x 0.0 :expected 1.0}
   {:label "grad-truncnorm-mu-asym: d/dmu TN(1,1,-2,2) at v=1"
    :make-dist (fn [mu] (dist/truncated-normal mu (mx/scalar 1.0) (mx/scalar -2.0) (mx/scalar 2.0)))
    :value 1.0 :x 1.0 :expected 0.28279}
   {:label "grad-truncnorm-sigma: d/dsigma TN(0,1,-2,2) at v=1"
    :make-dist (fn [sigma] (dist/truncated-normal (mx/scalar 0.0) sigma (mx/scalar -2.0) (mx/scalar 2.0)))
    :value 1.0 :x 1.0 :expected 0.22626}
   ;; Von-Mises
   {:label "grad-vonmises-mu: d/dmu VM(0,2) at v=1"
    :make-dist (fn [mu] (dist/von-mises mu (mx/scalar 2.0)))
    :value 1.0 :x 0.0 :expected 1.6829419696157930}
   {:label "grad-vonmises-kappa-at-mode: d/dkappa VM(0,1) at v=0"
    :make-dist (fn [kappa] (dist/von-mises (mx/scalar 0.0) kappa))
    :value 0.0 :x 1.0 :expected 0.55361}
   {:label "grad-vonmises-kappa-off: d/dkappa VM(0,2) at v=1"
    :make-dist (fn [kappa] (dist/von-mises (mx/scalar 0.0) kappa))
    :value 1.0 :x 2.0 :expected -0.15747}
   ;; Uniform
   {:label "grad-uniform-lo: d/dlo U(0,1) at v=0.5"
    :make-dist (fn [lo] (dist/uniform lo (mx/scalar 1.0)))
    :value 0.5 :x 0.0 :expected 1.0}
   {:label "grad-uniform-lo-wide: d/dlo U(0,5) at v=2"
    :make-dist (fn [lo] (dist/uniform lo (mx/scalar 5.0)))
    :value 2.0 :x 0.0 :expected 0.2}
   {:label "grad-uniform-hi: d/dhi U(0,1) at v=0.5"
    :make-dist (fn [hi] (dist/uniform (mx/scalar 0.0) hi))
    :value 0.5 :x 1.0 :expected -1.0}
   ;; Wrapped-Cauchy
   {:label "grad-wrappedcauchy-mu: d/dmu WC(0,0.5) at v=1"
    :make-dist (fn [mu] (dist/wrapped-cauchy mu (mx/scalar 0.5)))
    :value 1.0 :x 0.0 :expected 1.1856752413958853}
   {:label "grad-wrappedcauchy-rho: d/drho WC(0,0.5) at v=1"
    :make-dist (fn [rho] (dist/wrapped-cauchy (mx/scalar 0.0) rho))
    :value 1.0 :x 0.5 :expected -1.2197573524575895}
   ;; Wrapped-Normal
   {:label "grad-wrappednormal-mu: d/dmu WN(0,1) at v=0.5"
    :make-dist (fn [mu] (dist/wrapped-normal mu (mx/scalar 1.0)))
    :value 0.5 :x 0.0 :expected 0.5}
   {:label "grad-wrappednormal-sigma: d/dsigma WN(0,1) at v=1"
    :make-dist (fn [sigma] (dist/wrapped-normal (mx/scalar 0.0) sigma))
    :value 1.0 :x 1.0 :expected 0.0}])

(deftest cross-system-value-gradients-match-fd
  (testing "all cross-system value gradient specs: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs cross-system-value-gradient-specs)]
      (is (zero? failed) "all cross-system value gradient checks pass"))))

(deftest cross-system-param-gradients-match-fd
  (testing "all cross-system parameter gradient specs: analytical vs FD"
    (let [{:keys [failed]} (run-gradient-specs cross-system-param-gradient-specs)]
      (is (zero? failed) "all cross-system parameter gradient checks pass"))))

;; ---------------------------------------------------------------------------
;; Tests: boundary gradient behavior
;; ---------------------------------------------------------------------------

(deftest exponential-gradient-at-boundary
  (testing "exponential gradient at v=0 (support boundary)"
    (let [d (dist/exponential 1)
          g (analytical-gradient (fn [v] (dist/log-prob d v)) 0.0)]
      (is (js/isFinite g) "gradient at v=0 is finite")
      (is (h/close? -1.0 g 1e-2) "gradient at v=0 is -rate"))))

(deftest beta-gradient-near-boundary
  (testing "beta gradient near v=0 stays finite and matches FD"
    (let [d (dist/beta-dist 2 2)
          result (check-gradient (fn [v] (dist/log-prob d v)) 0.01)]
      (is (:pass? result) "analytical agrees with FD near lower boundary"))))

(deftest beta-gradient-near-upper-boundary
  (testing "beta gradient near v=1 stays finite and matches backward FD"
    (let [d (dist/beta-dist 2 2)
          f (fn [v] (dist/log-prob d v))
          analytical (analytical-gradient f 0.99)
          fd (fd-gradient-backward f 0.99)]
      (is (js/isFinite analytical) "gradient near v=1 is finite")
      (is (neg? analytical) "gradient near upper boundary is negative")
      (is (neg? fd) "FD near upper boundary is negative")
      (is (< (/ (js/Math.abs (- analytical fd))
                (js/Math.abs analytical))
             0.05)
          "analytical agrees with backward FD within 5% near upper boundary"))))

(deftest gamma-gradient-near-boundary
  (testing "gamma gradient near v=0 stays finite and matches forward FD"
    (let [d (dist/gamma-dist 2 1)
          f (fn [v] (dist/log-prob d v))
          analytical (analytical-gradient f 0.01)
          fd (fd-gradient-forward f 0.01)]
      (is (js/isFinite analytical) "gradient near v=0 is finite")
      (is (pos? analytical) "gradient near lower boundary is positive")
      (is (pos? fd) "FD near lower boundary is positive")
      (is (< (/ (js/Math.abs (- analytical fd))
                (js/Math.abs analytical))
             0.05)
          "analytical agrees with forward FD within 5% near boundary"))))

(deftest inv-gamma-gradient-near-boundary
  (testing "inv-gamma gradient near v=0 stays finite and matches forward FD"
    (let [d (dist/inv-gamma 2 1)
          f (fn [v] (dist/log-prob d v))
          analytical (analytical-gradient f 0.01)
          fd (fd-gradient-forward f 0.01)]
      (is (js/isFinite analytical) "gradient near v=0 is finite")
      (is (pos? analytical) "gradient near lower boundary is positive")
      (is (pos? fd) "FD near lower boundary is positive")
      (is (< (/ (js/Math.abs (- analytical fd))
                (js/Math.abs analytical))
             0.10)
          "analytical agrees with forward FD within 10% near boundary"))))

(deftest lognormal-gradient-near-boundary
  (testing "lognormal gradient near v=0 stays finite and matches forward FD"
    (let [d (dist/log-normal 0 1)
          f (fn [v] (dist/log-prob d v))
          analytical (analytical-gradient f 0.01)
          fd (fd-gradient-forward f 0.01)]
      (is (js/isFinite analytical) "gradient near v=0 is finite")
      (is (pos? analytical) "gradient near lower boundary is positive")
      (is (pos? fd) "FD near lower boundary is positive")
      (is (< (/ (js/Math.abs (- analytical fd))
                (js/Math.abs analytical))
             0.06)
          "analytical agrees with forward FD within 6% near boundary"))))

(deftest truncated-normal-gradient-near-lo-boundary
  (testing "truncated-normal gradient near lo stays finite and matches forward FD"
    (let [d (dist/truncated-normal 0 1 -2 2)
          f (fn [v] (dist/log-prob d v))
          analytical (analytical-gradient f -1.99)
          fd (fd-gradient-forward f -1.99)]
      (is (js/isFinite analytical) "gradient near lo is finite")
      (is (pos? analytical) "gradient near lo is positive (pushes away from boundary)")
      (is (pos? fd) "FD near lo is positive")
      (is (< (/ (js/Math.abs (- analytical fd))
                (js/Math.abs analytical))
             0.01)
          "analytical agrees with forward FD within 1% near lo boundary"))))

(deftest truncated-normal-gradient-near-hi-boundary
  (testing "truncated-normal gradient near hi stays finite and matches backward FD"
    (let [d (dist/truncated-normal 0 1 -2 2)
          f (fn [v] (dist/log-prob d v))
          analytical (analytical-gradient f 1.99)
          fd (fd-gradient-backward f 1.99)]
      (is (js/isFinite analytical) "gradient near hi is finite")
      (is (neg? analytical) "gradient near hi is negative (pushes away from boundary)")
      (is (neg? fd) "FD near hi is negative")
      (is (< (/ (js/Math.abs (- analytical fd))
                (js/Math.abs analytical))
             0.01)
          "analytical agrees with backward FD within 1% near hi boundary"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
