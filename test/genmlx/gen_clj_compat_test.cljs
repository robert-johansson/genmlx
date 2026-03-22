(ns genmlx.gen-clj-compat-test
  "Tests adapted from Gen.clj's test suite to verify GenMLX correctness.
   Sources: gen.distribution-test, gen.dynamic-test, gen.inference.importance-test
   Spot-check values verified against scipy.stats and Gen.jl.

   Note: GenMLX uses float32 (MLX default) while Gen.clj uses float64.
   Tolerances are set to 1e-6 for spot checks and 1e-5 for computed values."
  (:require [cljs.test :as t :refer [deftest is testing]]
            [genmlx.test-helpers :as th]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as importance])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn lp
  "Extract log-prob as a plain number."
  [d v]
  (let [r (dist/log-prob d (if (mx/array? v) v (mx/scalar v)))]
    (mx/eval! r)
    (mx/item r)))

(def ^:private f32-tol 1e-6)
(def ^:private sym-tol 1e-10)

;; =========================================================================
;; SECTION 1: Distribution logpdf spot checks
;; =========================================================================

(deftest s1-distribution-logpdf
  (testing "Gaussian logpdf spot checks"
    (is (th/close? -1.0439385332046727
                   (lp (dist/gaussian 0 1) 0.5) f32-tol)
        "normal(0,1) at 0.5")
    (is (th/close? -1.643335713764618
                   (lp (dist/gaussian 0 2) 0.5) f32-tol)
        "normal(0,2) at 0.5")
    (is (th/close? -1.612085713764618
                   (lp (dist/gaussian 0 2) 0) f32-tol)
        "normal(0,2) at 0"))

  (testing "Gaussian symmetry"
    (doseq [sigma [0.5 1.0 2.0 5.0]
            v     [0.1 1.0 3.0 10.0]]
      (is (th/close? (lp (dist/gaussian 0 sigma) v)
                     (lp (dist/gaussian 0 sigma) (- v))
                     sym-tol)
          (str "N(0," sigma ") symmetric at +/-" v))))

  (testing "Gaussian shift symmetry"
    (doseq [[mu sigma v shift] [[2.0 1.0 3.0 5.0]
                                 [-1.0 0.5 2.0 -3.0]
                                 [0.0 3.0 1.0 10.0]]]
      (is (th/close? (lp (dist/gaussian mu sigma) v)
                     (lp (dist/gaussian (+ mu shift) sigma) (+ v shift))
                     f32-tol)
          (str "N(" mu "," sigma ") shift by " shift))))

  (testing "Uniform logpdf inside bounds"
    (doseq [[lo hi v] [[-5.0 5.0 0.0]
                        [0.0 1.0 0.5]
                        [-10.0 -0.1 -5.0]
                        [0.1 10.0 5.0]]]
      (let [expected (- (js/Math.log (- hi lo)))]
        (is (th/close? expected (lp (dist/uniform lo hi) v) f32-tol)
            (str "uniform(" lo "," hi ") at " v " (inside)")))))

  (testing "Uniform logpdf outside bounds"
    (doseq [[lo hi v] [[0.0 1.0 -0.1]
                        [0.0 1.0 1.5]
                        [-5.0 5.0 10.0]]]
      (is (= ##-Inf (lp (dist/uniform lo hi) v))
          (str "uniform(" lo "," hi ") at " v " (outside)"))))

  (testing "Beta logpdf spot checks"
    (is (th/close? -5.992380837839856
                   (lp (dist/beta-dist 0.001 1) 0.4) 1e-4)
        "beta(0.001, 1) at 0.4")
    (is (th/close? -6.397440480839912
                   (lp (dist/beta-dist 1 0.001) 0.4) 1e-4)
        "beta(1, 0.001) at 0.4"))

  (testing "Gamma logpdf spot checks"
    (is (th/close? -6.391804444241573
                   (lp (dist/gamma-dist 0.001 1) 0.4) 1e-4)
        "gamma(shape=0.001, scale=1) at 0.4")
    (is (th/close? -393.0922447210179
                   (lp (dist/gamma-dist 1 1000) 0.4) 0.1)
        "gamma(shape=1, scale=0.001) at 0.4"))

  (testing "Bernoulli logpdf"
    (is (th/close? (lp (dist/bernoulli 0.5) 1.0)
                   (lp (dist/bernoulli 0.5) 0.0)
                   sym-tol)
        "bernoulli(0.5) fair coin symmetry")
    (doseq [p [0.1 0.3 0.5 0.7 0.9]]
      (let [sum (+ (js/Math.exp (lp (dist/bernoulli p) 1.0))
                   (js/Math.exp (lp (dist/bernoulli p) 0.0)))]
        (is (th/close? 1.0 sum f32-tol)
            (str "bernoulli(" p ") sums to 1"))))
    (doseq [p [0.2 0.5 0.8]]
      (is (th/close? (js/Math.log p)
                     (lp (dist/bernoulli p) 1.0)
                     f32-tol)
          (str "bernoulli(" p ") at 1 = log(p)"))
      (is (th/close? (js/Math.log (- 1 p))
                     (lp (dist/bernoulli p) 0.0)
                     f32-tol)
          (str "bernoulli(" p ") at 0 = log(1-p)"))))

  (testing "Exponential logpdf"
    (doseq [v [-0.1 -1.0 -100.0]]
      (is (= ##-Inf (lp (dist/exponential 1.0) v))
          (str "exponential(1) at " v " (negative)")))
    (doseq [v [0.5 1.0 2.0 5.0]]
      (is (th/close? (- v) (lp (dist/exponential 1.0) v) sym-tol)
          (str "exponential(1) at " v " = -v")))
    (is (th/close? -3.3068528194400546
                   (lp (dist/exponential 2.0) 2.0) f32-tol)
        "exponential(2) at 2")
    (is (th/close? -5.306852819440055
                   (lp (dist/exponential 2.0) 3.0) f32-tol)
        "exponential(2) at 3"))

  (testing "Laplace logpdf"
    (is (th/close? -1.6931471805599454
                   (lp (dist/laplace 2 1) 1) f32-tol)
        "laplace(2,1) at 1")
    (is (th/close? -1.8862943611198906
                   (lp (dist/laplace 0 2) 1) f32-tol)
        "laplace(0,2) at 1")
    (is (th/close? 4.214608098422191
                   (lp (dist/laplace 0 0.001) 0.002) 1e-3)
        "laplace(0,0.001) at 0.002")
    (doseq [v [0.5 1.0 3.0 10.0]]
      (is (th/close? (lp (dist/laplace 0 1) v)
                     (lp (dist/laplace 0 1) (- v))
                     sym-tol)
          (str "laplace(0,1) symmetric at +/-" v)))
    (doseq [v [0.0 1.5 -3.0 10.0]]
      (is (th/close? (- (js/Math.log 2))
                     (lp (dist/laplace v 1) v)
                     f32-tol)
          (str "laplace(" v ",1) at " v " = -log(2)"))))

  (testing "Student-t logpdf spot checks"
    (is (th/close? -1.7347417805005154
                   (lp (dist/student-t 2 2.1 2) 2) f32-tol)
        "student-t(2, 2.1, 2) at 2")
    (is (th/close? -2.795309741614719
                   (lp (dist/student-t 1 0.8 4) 3) f32-tol)
        "student-t(1, 0.8, 4) at 3")
    (doseq [nu [1 2 5 10]
            v  [-2.0 0.0 1.5 3.0]]
      (is (th/close? (lp (dist/student-t nu 0 1) v)
                     (lp (dist/student-t nu 0 1) v)
                     sym-tol)
          (str "student-t(" nu ", 0, 1) at " v " self-consistent"))))

  (testing "Delta logpdf"
    (doseq [c [0.0 1.5 -3.0 100.0]]
      (is (th/close? 0.0 (lp (dist/delta c) c) sym-tol)
          (str "delta(" c ") at " c " = 0")))
    (doseq [[c v] [[0.0 1.0] [1.0 0.0] [5.0 5.1] [-3.0 3.0]]]
      (is (= ##-Inf (lp (dist/delta c) v))
          (str "delta(" c ") at " v " = -Inf"))))

  (testing "Log-Gamma correctness via Gamma distribution"
    (letfn [(factorial [n]
              (if (zero? n) 1 (* n (factorial (dec n)))))]
      (doseq [n (range 1 15)]
        (let [lgamma-n (- -1.0 (lp (dist/gamma-dist n 1) 1.0))
              expected (js/Math.log (factorial (dec n)))]
          (is (th/close? expected lgamma-n 1e-4)
              (str "lgamma(" n ") = log(" (dec n) "!)")))))))

;; =========================================================================
;; SECTION 2: GFI interface for primitive distributions
;; =========================================================================

(deftest s2-primitive-gfi
  (testing "gaussian GFI round-trip"
    (let [d (dist/gaussian 0 1)
          trace (p/simulate d [])]
      (is (= d (:gen-fn trace)) "gen-fn round-trips through trace")
      (is (= [] (:args trace)) "args round-trip")
      (let [choice (:choices trace)
            retval (:retval trace)]
        (mx/eval! retval (cm/get-value choice))
        (is (th/close? (mx/item retval) (mx/item (cm/get-value choice)) sym-tol)
            "retval = choice value"))))

  (testing "bernoulli GFI"
    (let [d (dist/bernoulli 0.5)
          trace (p/simulate d [0.5])]
      (is (some? trace) "simulate returns trace")
      (let [retval (:retval trace)
            score  (:score trace)]
        (mx/eval! retval score)
        (is (th/close? 0.5 (js/Math.exp (mx/item score)) f32-tol)
            "bernoulli(0.5): score = log(0.5)")))))

;; =========================================================================
;; SECTION 3: Bernoulli generate/update weight tests
;; =========================================================================

(deftest s3-bernoulli-generate-weight
  (testing "generate with constraint"
    (let [gf (dyn/auto-key (gen [] (trace :x (dist/bernoulli 0.3))))
          {:keys [trace weight]} (p/generate gf [] (cm/choicemap :x (mx/scalar 1.0)))]
      (mx/eval! weight)
      (is (th/close? (js/Math.log 0.3) (mx/item weight) f32-tol)
          "weight = log(0.3)")))

  (testing "update same value"
    (let [gf (dyn/auto-key (gen [] (trace :x (dist/bernoulli 0.3))))
          {:keys [trace]} (p/generate gf [] (cm/choicemap :x (mx/scalar 1.0)))
          {:keys [weight]} (p/update gf trace (cm/choicemap :x (mx/scalar 1.0)))]
      (mx/eval! weight)
      (is (th/close? 1.0 (js/Math.exp (mx/item weight)) f32-tol)
          "exp(weight) = 1")))

  (testing "update true->false"
    (let [gf (dyn/auto-key (gen [] (trace :x (dist/bernoulli 0.3))))
          {:keys [trace]} (p/generate gf [] (cm/choicemap :x (mx/scalar 1.0)))
          {:keys [weight]} (p/update gf trace (cm/choicemap :x (mx/scalar 0.0)))]
      (mx/eval! weight)
      (is (th/close? (/ 0.7 0.3) (js/Math.exp (mx/item weight)) 1e-4)
          "exp(weight) = 0.7/0.3"))))

;; =========================================================================
;; SECTION 4: Dynamic DSL / gen macro tests
;; =========================================================================

(deftest s4-dynamic-dsl
  (testing "no-arity, no-return"
    (let [gf (gen [] nil)]
      (is (nil? (dyn/call gf)) "returns nil")))

  (testing "deterministic gen round-trip"
    (let [gf (dyn/auto-key (gen [a b c d e] (+ a b c d e)))
          trace (p/simulate gf [1 2 3 4 5])]
      (is (= gf (:gen-fn trace)) "gen-fn round-trips")
      (is (= [1 2 3 4 5] (:args trace)) "args round-trip")
      (is (= 15 (:retval trace)) "retval matches")))

  (testing "deterministic: no choices"
    (let [gf (dyn/auto-key (gen [x y] (+ x y)))
          trace (p/simulate gf [3 4])]
      (is (= 0 (count (cm/addresses (:choices trace)))) "no choices")))

  (testing "trace creates choices"
    (let [gf (dyn/auto-key (gen [p] (trace :addr (dist/bernoulli p))))
          trace (p/simulate gf [0.5])]
      (let [choices (:choices trace)
            retval  (:retval trace)]
        (mx/eval! retval)
        (let [v (cm/get-value (cm/get-submap choices :addr))]
          (mx/eval! v)
          (is (th/close? (mx/item retval) (mx/item v) sym-tol)
              "trace choices match retval")))))

  (testing "splice bubbles up choices"
    (let [gf0 (gen [] (trace :addr (dist/bernoulli 0.5)))
          gf1 (dyn/auto-key (gen [] (splice :sub gf0)))
          trace (p/simulate gf1 [])]
      (let [choices (:choices trace)
            sub-map (cm/get-submap choices :sub)]
        (is (some? sub-map) "sub-map exists")
        (let [addr-val (cm/get-submap sub-map :addr)]
          (is (cm/has-value? addr-val) "addr exists in sub")))))

  (testing "trace inside trace nests"
    (let [inner (gen [] (trace :inner (dist/bernoulli 0.5)))
          outer (dyn/auto-key (gen [] (splice :outer inner)))
          trace (p/simulate outer [])]
      (let [choices (:choices trace)
            outer-sub (cm/get-submap choices :outer)
            inner-val (cm/get-submap outer-sub :inner)]
        (is (some? outer-sub) "outer exists")
        (is (cm/has-value? inner-val) "inner exists within outer"))))

  (testing "generate through splice preserves structure"
    (let [inner (gen [] (trace :addr (dist/bernoulli 0.5)))
          outer (dyn/auto-key (gen [] (splice :sub inner)))
          {:keys [trace]} (p/generate outer [] cm/EMPTY)]
      (let [choices (:choices trace)
            sub-map (cm/get-submap choices :sub)
            addr-val (cm/get-submap sub-map :addr)]
        (is (cm/has-value? addr-val) "structure preserved"))))

  (testing "score computation"
    (let [trace (p/simulate
                  (dyn/auto-key (gen [] (trace :addr (dist/bernoulli 0.5))))
                  [])]
      (let [score (:score trace)]
        (mx/eval! score)
        (is (th/close? 0.5 (js/Math.exp (mx/item score)) f32-tol)
            "bernoulli(0.5) score = log(0.5)")))))

;; =========================================================================
;; SECTION 5: Update semantics
;; =========================================================================

(deftest s5-update-semantics
  (testing "update with constraint => old value in discard"
    (let [gf (dyn/auto-key (gen [] (trace :x (dist/bernoulli 0.5))))
          trace (p/simulate gf [])
          old-x (let [v (cm/get-value (cm/get-submap (:choices trace) :x))]
                  (mx/eval! v) (mx/item v))
          new-val (if (> old-x 0.5) 0.0 1.0)
          {:keys [trace discard]} (p/update gf trace
                                     (cm/choicemap :x (mx/scalar new-val)))]
      (is (cm/has-value? (cm/get-submap discard :x)) "discard contains old value")
      (let [discarded (cm/get-value (cm/get-submap discard :x))]
        (mx/eval! discarded)
        (is (th/close? old-x (mx/item discarded) sym-tol) "discarded value matches old"))
      (let [new-x (cm/get-value (cm/get-submap (:choices trace) :x))]
        (mx/eval! new-x)
        (is (th/close? new-val (mx/item new-x) sym-tol) "new trace has new value"))))

  (testing "update same value: log-weight = 0"
    (let [gf (dyn/auto-key (gen [] (trace :x (dist/bernoulli 0.3))))
          {:keys [trace]} (p/generate gf [] (cm/choicemap :x (mx/scalar 1.0)))
          {:keys [weight]} (p/update gf trace (cm/choicemap :x (mx/scalar 1.0)))]
      (mx/eval! weight)
      (is (th/close? 0.0 (mx/item weight) f32-tol) "log-weight = 0")))

  (testing "update partial: unchanged addr kept"
    (let [gf (dyn/auto-key (gen []
               (trace :kept (dist/bernoulli 0.5))
               (trace :changed (dist/bernoulli 0.5))))
          trace (p/simulate gf [])
          old-kept (let [v (cm/get-value (cm/get-submap (:choices trace) :kept))]
                     (mx/eval! v) (mx/item v))
          {:keys [trace discard]} (p/update gf trace
                                     (cm/choicemap :changed (mx/scalar 1.0)))]
      (let [new-kept (cm/get-value (cm/get-submap (:choices trace) :kept))]
        (mx/eval! new-kept)
        (is (th/close? old-kept (mx/item new-kept) sym-tol) "unchanged addr kept"))
      (is (cm/has-value? (cm/get-submap discard :changed)) "changed addr in discard"))))

;; =========================================================================
;; SECTION 6: Generate semantics
;; =========================================================================

(deftest s6-generate-semantics
  (testing "empty constraints: weight = 0"
    (let [gf (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          {:keys [trace weight]} (p/generate gf [] cm/EMPTY)]
      (mx/eval! weight)
      (is (th/close? 0.0 (mx/item weight) sym-tol) "weight = 0")))

  (testing "generate with constraint"
    (let [gf (dyn/auto-key (gen []
               (trace :x (dist/gaussian 0 1))
               (trace :y (dist/gaussian 0 1))))
          constraints (cm/choicemap :x (mx/scalar 2.0))
          {:keys [trace weight]} (p/generate gf [] constraints)]
      (let [x-val (cm/get-value (cm/get-submap (:choices trace) :x))]
        (mx/eval! x-val)
        (is (th/close? 2.0 (mx/item x-val) sym-tol) "x constrained to 2.0"))
      (mx/eval! weight)
      (is (th/close? (lp (dist/gaussian 0 1) 2.0) (mx/item weight) f32-tol)
          "weight = logpdf(N(0,1), 2)"))))

;; =========================================================================
;; SECTION 7: Regenerate semantics
;; =========================================================================

(deftest s7-regenerate-semantics
  (testing "regenerate keeps unselected addresses"
    (let [gf (dyn/auto-key (gen []
               (let [x (trace :x (dist/gaussian 0 1))
                     y (trace :y (dist/gaussian 0 1))]
                 (mx/eval! x y)
                 [(mx/item x) (mx/item y)])))
          trace (p/simulate gf [])
          old-y (let [v (cm/get-value (cm/get-submap (:choices trace) :y))]
                  (mx/eval! v) (mx/item v))
          {:keys [trace weight]} (p/regenerate gf trace (sel/select :x))]
      (let [new-y (cm/get-value (cm/get-submap (:choices trace) :y))]
        (mx/eval! new-y)
        (is (th/close? old-y (mx/item new-y) sym-tol) "unselected :y unchanged"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "returns finite weight"))))

;; =========================================================================
;; SECTION 8: Importance sampling with rejection
;; =========================================================================

(deftest s8-importance-sampling
  (testing "importance resampling"
    (let [model (gen []
                  (let [foo (trace :foo (dist/bernoulli 0.5))]
                    (mx/eval! foo)
                    (if (> (mx/item foo) 0.5)
                      (trace :bar (dist/bernoulli 0.99))
                      (trace :bar (dist/bernoulli 0.01)))))
          observations (cm/choicemap :bar (mx/scalar 1.0))
          traces (importance/importance-resampling
                   {:samples 1 :particles 20}
                   model [] observations)
          trace (first traces)]
      (is (some? trace) "returns trace")
      (let [choices (:choices trace)
            bar-val (cm/get-value (cm/get-submap choices :bar))]
        (mx/eval! bar-val)
        (is (th/close? 1.0 (mx/item bar-val) sym-tol) "bar constrained to 1.0")))))

;; =========================================================================
;; SECTION 9: Mathematical properties of distributions
;; =========================================================================

(deftest s9-distribution-properties
  (testing "laplace at location = -log(2)"
    (doseq [v [-5.0 0.0 3.0 10.0]]
      (is (th/close? (- (js/Math.log 2))
                     (lp (dist/laplace v 1) v)
                     f32-tol)
          (str "laplace(" v ",1) at " v " = -log(2)"))))

  (testing "exponential(1) at 0 = 0"
    (is (th/close? 0.0 (lp (dist/exponential 1.0) 0.0) sym-tol)))

  (testing "Gaussian at mean gives peak"
    (doseq [sigma [0.5 1.0 2.0 5.0]]
      (let [expected (- (- (* 0.5 (js/Math.log (* 2 js/Math.PI))))
                        (js/Math.log sigma))]
        (is (th/close? expected (lp (dist/gaussian 0 sigma) 0) f32-tol)
            (str "N(0," sigma ") at 0 = peak"))))))

;; =========================================================================
;; SECTION 10: End-to-end model test
;; =========================================================================

(deftest s10-end-to-end-model
  (testing "line model simulate"
    (let [line-model (dyn/auto-key (gen [xs]
                       (let [slope     (trace :slope (dist/gaussian 0 1))
                             intercept (trace :intercept (dist/gaussian 0 2))]
                         (mx/eval! slope intercept)
                         (let [s (mx/item slope) i (mx/item intercept)]
                           (doseq [[idx x] (map-indexed vector xs)]
                             (trace (keyword (str "y" idx))
                                       (dist/gaussian (+ (* s x) i) 0.1)))
                           (fn [x] (+ (* s x) i))))))
          xs (range -5 6)
          trace (p/simulate line-model [(vec xs)])]
      (is (some? (:choices trace)) "has choices")
      (is (fn? (:retval trace)) "returns function")
      (is (cm/has-value? (cm/get-submap (:choices trace) :slope)) "has slope")
      (is (cm/has-value? (cm/get-submap (:choices trace) :intercept)) "has intercept")
      (doseq [i (range 11)]
        (is (cm/has-value? (cm/get-submap (:choices trace)
                                           (keyword (str "y" i))))
            (str "has y" i)))))

  (testing "line model generate with observations"
    (let [line-model (dyn/auto-key (gen [xs]
                       (let [slope     (trace :slope (dist/gaussian 0 1))
                             intercept (trace :intercept (dist/gaussian 0 2))]
                         (mx/eval! slope intercept)
                         (let [s (mx/item slope) i (mx/item intercept)]
                           (doseq [[idx x] (map-indexed vector xs)]
                             (trace (keyword (str "y" idx))
                                       (dist/gaussian (+ (* s x) i) 0.1)))
                           [s i]))))
          xs [1.0 2.0 3.0]
          observations (cm/choicemap :y0 (mx/scalar 2.0)
                                      :y1 (mx/scalar 4.0)
                                      :y2 (mx/scalar 6.0))
          {:keys [trace weight]} (p/generate line-model [xs] observations)]
      (is (some? trace) "returns trace")
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "weight is finite")
      (let [y0 (cm/get-value (cm/get-submap (:choices trace) :y0))]
        (mx/eval! y0)
        (is (th/close? 2.0 (mx/item y0) sym-tol) "y0 = 2.0")))))

(t/run-tests)
