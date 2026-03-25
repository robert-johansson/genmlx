(ns genmlx.score-monotonicity-property-test
  "Property-based tests for importance weight monotonicity.
   For single-site models: constraining at the mode should yield a higher
   importance weight than constraining far from the mode. For multi-site
   models with subset constraints: fewer constraints should generally yield
   higher average weights."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-item [x]
  (mx/eval! x)
  (mx/item x))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- mean [xs]
  (/ (reduce + xs) (count xs)))

;; ---------------------------------------------------------------------------
;; Property 1: Gaussian at mode vs far from mode
;; For gaussian(mu, sigma), constraining at mu should give higher weight
;; than constraining at mu + 5*sigma (5 standard deviations away).
;; ---------------------------------------------------------------------------

(def gaussian-mode-specs
  [{:mu 0.0 :sigma 1.0 :label "N(0,1)"}
   {:mu 3.0 :sigma 0.5 :label "N(3,0.5)"}
   {:mu -2.0 :sigma 2.0 :label "N(-2,2)"}
   {:mu 0.0 :sigma 10.0 :label "N(0,10)"}
   {:mu 5.0 :sigma 0.1 :label "N(5,0.1)"}])

(defspec gaussian-mode-has-higher-weight-than-tail 50
  (prop/for-all [spec (gen/elements gaussian-mode-specs)]
    (let [mu (:mu spec)
          sigma (:sigma spec)
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian mu sigma))]
                    (mx/eval! x)
                    (mx/item x))))
          ;; Constrain at mode
          constraint-mode (cm/choicemap :x (mx/scalar mu))
          {:keys [weight]} (p/generate model [] constraint-mode)
          w-mode (eval-item weight)
          ;; Constrain 5 sigma away
          constraint-tail (cm/choicemap :x (mx/scalar (+ mu (* 5.0 sigma))))
          {:keys [weight]} (p/generate model [] constraint-tail)
          w-tail (eval-item weight)]
      (and (finite? w-mode)
           (finite? w-tail)
           (> w-mode w-tail)))))

;; ---------------------------------------------------------------------------
;; Property 2: Exponential near zero vs far from zero
;; For exponential(rate), constraining near 0 should give higher weight
;; than constraining at 10/rate (far in the tail).
;; ---------------------------------------------------------------------------

(def exponential-specs
  [{:rate 1.0 :label "Exp(1)"}
   {:rate 0.5 :label "Exp(0.5)"}
   {:rate 2.0 :label "Exp(2)"}
   {:rate 5.0 :label "Exp(5)"}])

(defspec exponential-near-zero-has-higher-weight 50
  (prop/for-all [spec (gen/elements exponential-specs)]
    (let [rate (:rate spec)
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/exponential rate))]
                    (mx/eval! x)
                    (mx/item x))))
          ;; Constrain near zero (mode of exponential)
          constraint-near (cm/choicemap :x (mx/scalar 0.01))
          {:keys [weight]} (p/generate model [] constraint-near)
          w-near (eval-item weight)
          ;; Constrain far in the tail
          constraint-far (cm/choicemap :x (mx/scalar (/ 10.0 rate)))
          {:keys [weight]} (p/generate model [] constraint-far)
          w-far (eval-item weight)]
      (and (finite? w-near)
           (finite? w-far)
           (> w-near w-far)))))

;; ---------------------------------------------------------------------------
;; Property 3: Laplace at mode vs tail
;; For laplace(loc, scale), constraining at loc should give higher weight
;; than constraining at loc + 5*scale.
;; ---------------------------------------------------------------------------

(def laplace-specs
  [{:loc 0.0 :scale 1.0 :label "Laplace(0,1)"}
   {:loc 2.0 :scale 0.5 :label "Laplace(2,0.5)"}
   {:loc -3.0 :scale 3.0 :label "Laplace(-3,3)"}])

(defspec laplace-mode-has-higher-weight-than-tail 50
  (prop/for-all [spec (gen/elements laplace-specs)]
    (let [loc (:loc spec)
          scale (:scale spec)
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/laplace loc scale))]
                    (mx/eval! x)
                    (mx/item x))))
          constraint-mode (cm/choicemap :x (mx/scalar loc))
          {:keys [weight]} (p/generate model [] constraint-mode)
          w-mode (eval-item weight)
          constraint-tail (cm/choicemap :x (mx/scalar (+ loc (* 5.0 scale))))
          {:keys [weight]} (p/generate model [] constraint-tail)
          w-tail (eval-item weight)]
      (and (finite? w-mode)
           (finite? w-tail)
           (> w-mode w-tail)))))

;; ---------------------------------------------------------------------------
;; Property 4: Uniform in-support vs just inside boundary
;; For uniform(lo, hi), constraining at the midpoint is equally likely
;; as constraining anywhere in [lo, hi]. But constraining outside support
;; should give -Inf weight.
;; ---------------------------------------------------------------------------

(def uniform-specs
  [{:lo 0.0 :hi 1.0 :label "U(0,1)"}
   {:lo -5.0 :hi 5.0 :label "U(-5,5)"}
   {:lo 10.0 :hi 20.0 :label "U(10,20)"}])

(defspec uniform-in-support-has-finite-weight 30
  (prop/for-all [spec (gen/elements uniform-specs)]
    (let [lo (:lo spec)
          hi (:hi spec)
          mid (/ (+ lo hi) 2.0)
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/uniform lo hi))]
                    (mx/eval! x)
                    (mx/item x))))
          constraint (cm/choicemap :x (mx/scalar mid))
          {:keys [weight]} (p/generate model [] constraint)
          w (eval-item weight)]
      (finite? w))))

(defspec uniform-outside-support-has-neg-inf-weight 30
  (prop/for-all [spec (gen/elements uniform-specs)]
    (let [lo (:lo spec)
          hi (:hi spec)
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/uniform lo hi))]
                    (mx/eval! x)
                    (mx/item x))))
          constraint (cm/choicemap :x (mx/scalar (+ hi 1.0)))
          {:keys [weight]} (p/generate model [] constraint)
          w (eval-item weight)]
      (= ##-Inf w))))

;; ---------------------------------------------------------------------------
;; Property 5: Beta distribution - mode has higher weight than boundary
;; For beta(a, b) with a,b > 1, mode = (a-1)/(a+b-2).
;; Constraining at mode should give higher weight than near 0 or 1.
;; ---------------------------------------------------------------------------

(def beta-specs
  [{:a 2.0 :b 2.0 :label "Beta(2,2)"}
   {:a 5.0 :b 2.0 :label "Beta(5,2)"}
   {:a 2.0 :b 5.0 :label "Beta(2,5)"}
   {:a 10.0 :b 10.0 :label "Beta(10,10)"}])

(defspec beta-mode-has-higher-weight-than-boundary 50
  (prop/for-all [spec (gen/elements beta-specs)]
    (let [a (:a spec)
          b (:b spec)
          mode (/ (- a 1.0) (- (+ a b) 2.0))
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/beta-dist a b))]
                    (mx/eval! x)
                    (mx/item x))))
          constraint-mode (cm/choicemap :x (mx/scalar mode))
          {:keys [weight]} (p/generate model [] constraint-mode)
          w-mode (eval-item weight)
          ;; Constrain near boundary (0.01)
          constraint-boundary (cm/choicemap :x (mx/scalar 0.01))
          {:keys [weight]} (p/generate model [] constraint-boundary)
          w-boundary (eval-item weight)]
      (and (finite? w-mode)
           (finite? w-boundary)
           (> w-mode w-boundary)))))

;; ---------------------------------------------------------------------------
;; Property 6: Adding constraints reduces average weight (statistical)
;; For a 2-site model, constraining 1 site should have higher AVERAGE weight
;; than constraining both sites, over many runs.
;;
;; E[weight | 1 constraint] >= E[weight | 2 constraints]
;; because with 1 constraint, the other site is sampled from the prior
;; (weight = log-prob of constrained site), while with 2 constraints,
;; both sites are fixed (weight = total log-prob).
;;
;; Actually, for prior-as-proposal: generate(model, args, constraints).weight
;; = sum of log-probs of constrained sites only. So more constraints means
;; the weight includes more log-prob terms. The sign depends on the values.
;;
;; Cleaner: For a single constrained value, weight = log p(x_constrained).
;; For two constrained values, weight = log p(x1) + log p(x2).
;; If the constrained values are reasonable, both log-probs are negative,
;; so more constraints -> more negative -> lower weight.
;; ---------------------------------------------------------------------------

(defspec more-constraints-lower-average-weight 30
  (prop/for-all [_dummy (gen/return nil)]
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    (mx/eval! x y)
                    (+ (mx/item x) (mx/item y)))))
          ;; Fixed reasonable constraint values (not at mode to ensure negative log-probs)
          x-val (mx/scalar 1.0)
          y-val (mx/scalar 1.0)
          ;; Weight with 1 constraint
          c1 (cm/choicemap :x x-val)
          w1 (eval-item (:weight (p/generate model [] c1)))
          ;; Weight with 2 constraints
          c2 (-> (cm/choicemap :x x-val)
                 (cm/set-value :y y-val))
          w2 (eval-item (:weight (p/generate model [] c2)))]
      ;; With 1 constraint: weight = log p(x=1) under N(0,1)
      ;; With 2 constraints: weight = log p(x=1) + log p(y=1) under N(0,1)
      ;; Since log p(y=1) < 0, w2 < w1
      (and (finite? w1)
           (finite? w2)
           (> w1 w2)))))

;; ---------------------------------------------------------------------------
;; Property 7: Gamma distribution - mode has higher weight than far tail
;; For gamma(shape, rate) with shape > 1, mode = (shape-1)/rate.
;; ---------------------------------------------------------------------------

(def gamma-specs
  [{:shape 2.0 :rate 1.0 :label "Gamma(2,1)"}
   {:shape 5.0 :rate 1.0 :label "Gamma(5,1)"}
   {:shape 3.0 :rate 2.0 :label "Gamma(3,2)"}
   {:shape 10.0 :rate 0.5 :label "Gamma(10,0.5)"}])

(defspec gamma-mode-has-higher-weight-than-tail 50
  (prop/for-all [spec (gen/elements gamma-specs)]
    (let [shape (:shape spec)
          rate (:rate spec)
          mode (/ (- shape 1.0) rate)
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gamma-dist shape rate))]
                    (mx/eval! x)
                    (mx/item x))))
          constraint-mode (cm/choicemap :x (mx/scalar mode))
          {:keys [weight]} (p/generate model [] constraint-mode)
          w-mode (eval-item weight)
          ;; Far in the tail: 10x the mode
          constraint-tail (cm/choicemap :x (mx/scalar (* 10.0 mode)))
          {:keys [weight]} (p/generate model [] constraint-tail)
          w-tail (eval-item weight)]
      (and (finite? w-mode)
           (finite? w-tail)
           (> w-mode w-tail)))))

;; ---------------------------------------------------------------------------
;; Property 8: Student-t at mode vs tail
;; student-t(df, loc, scale): mode is at loc.
;; ---------------------------------------------------------------------------

(def student-t-specs
  [{:df 3.0 :loc 0.0 :scale 1.0 :label "t(3,0,1)"}
   {:df 10.0 :loc 2.0 :scale 0.5 :label "t(10,2,0.5)"}
   {:df 5.0 :loc -1.0 :scale 2.0 :label "t(5,-1,2)"}])

(defspec student-t-mode-has-higher-weight-than-tail 50
  (prop/for-all [spec (gen/elements student-t-specs)]
    (let [df (:df spec)
          loc (:loc spec)
          scale (:scale spec)
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/student-t df loc scale))]
                    (mx/eval! x)
                    (mx/item x))))
          constraint-mode (cm/choicemap :x (mx/scalar loc))
          {:keys [weight]} (p/generate model [] constraint-mode)
          w-mode (eval-item weight)
          constraint-tail (cm/choicemap :x (mx/scalar (+ loc (* 10.0 scale))))
          {:keys [weight]} (p/generate model [] constraint-tail)
          w-tail (eval-item weight)]
      (and (finite? w-mode)
           (finite? w-tail)
           (> w-mode w-tail)))))

;; ---------------------------------------------------------------------------
;; Property 9: Cauchy at mode vs tail
;; cauchy(loc, scale): mode is at loc.
;; ---------------------------------------------------------------------------

(def cauchy-specs
  [{:loc 0.0 :scale 1.0 :label "Cauchy(0,1)"}
   {:loc 3.0 :scale 0.5 :label "Cauchy(3,0.5)"}
   {:loc -2.0 :scale 2.0 :label "Cauchy(-2,2)"}])

(defspec cauchy-mode-has-higher-weight-than-tail 50
  (prop/for-all [spec (gen/elements cauchy-specs)]
    (let [loc (:loc spec)
          scale (:scale spec)
          model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/cauchy loc scale))]
                    (mx/eval! x)
                    (mx/item x))))
          constraint-mode (cm/choicemap :x (mx/scalar loc))
          {:keys [weight]} (p/generate model [] constraint-mode)
          w-mode (eval-item weight)
          constraint-tail (cm/choicemap :x (mx/scalar (+ loc (* 20.0 scale))))
          {:keys [weight]} (p/generate model [] constraint-tail)
          w-tail (eval-item weight)]
      (and (finite? w-mode)
           (finite? w-tail)
           (> w-mode w-tail)))))

(t/run-tests)
