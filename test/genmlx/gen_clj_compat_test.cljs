(ns genmlx.gen-clj-compat-test
  "Tests adapted from Gen.clj's test suite to verify GenMLX correctness.
   Sources: gen.distribution-test, gen.dynamic-test, gen.inference.importance-test
   Spot-check values verified against scipy.stats and Gen.jl.

   Note: GenMLX uses float32 (MLX default) while Gen.clj uses float64.
   Tolerances are set to 1e-6 for spot checks and 1e-5 for computed values."
  (:require [cljs.test :as t :refer [deftest is testing]]
            [genmlx.test-helpers :as th]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
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
            v [0.1 1.0 3.0 10.0]]
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
            v [-2.0 0.0 1.5 3.0]]
      (is (th/close? (lp (dist/student-t nu 0 1) v)
                     (lp (dist/student-t nu 0 1) (- v))
                     sym-tol)
          (str "student-t(" nu ", 0, 1) symmetric at +/-" v))))

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
;; SECTION 1b: Cross-system logpdf spot checks (issue #11)
;; Reference values from cross_system_tests/specs/logprob_tests.json
;; Analytical ground truth verified against Gen.jl + scipy
;; =========================================================================

(deftest s1b-cross-system-logpdf
  (testing "Log-normal logpdf spot checks"
    (is (th/close? -0.9189385332046727
                   (lp (dist/log-normal 0 1) 1.0) f32-tol)
        "lognormal(0,1) at 1.0")
    (is (th/close? -1.1560965676592116
                   (lp (dist/log-normal 1 0.5) 2.5) f32-tol)
        "lognormal(1,0.5) at 2.5")
    (is (th/close? 0.02776211541962792
                   (lp (dist/log-normal 0 2) 0.1) f32-tol)
        "lognormal(0,2) at 0.1"))

  (testing "Log-normal edge cases"
    (is (th/close? -0.41893853320467267
                   (lp (dist/log-normal 0 1) 0.36787944117144233) 1e-5)
        "lognormal(0,1) at e^-1")
    (is (th/close? -6.9175645681733755
                   (lp (dist/log-normal 0 1) 0.01) 1e-4)
        "lognormal(0,1) at 0.01")
    (is (th/close? -16.127904940149563
                   (lp (dist/log-normal 0 1) 100.0) 1e-3)
        "lognormal(0,1) at 100"))

  (testing "Poisson logpdf spot checks"
    (is (th/close? -1.495922603223724
                   (lp (dist/poisson 3) 3) 1e-4)
        "poisson(3) at 3")
    (is (th/close? -1.0
                   (lp (dist/poisson 1) 0) f32-tol)
        "poisson(1) at 0")
    (is (th/close? -4.010033448735005
                   (lp (dist/poisson 5) 10) 1e-4)
        "poisson(5) at 10"))

  (testing "Poisson edge cases"
    (is (th/close? -0.1
                   (lp (dist/poisson 0.1) 0) f32-tol)
        "poisson(0.1) at 0")
    (is (th/close? -0.01
                   (lp (dist/poisson 0.01) 0) f32-tol)
        "poisson(0.01) at 0"))

  (testing "Categorical logpdf spot checks"
    (is (th/close? -1.0986122886681098
                   (lp (dist/categorical (mx/array [0.0 0.0 0.0])) 0) f32-tol)
        "categorical(uniform-3) at 0")
    (is (th/close? -1.0986122886681098
                   (lp (dist/categorical (mx/array [0.0 0.0 0.0])) 2) f32-tol)
        "categorical(uniform-3) at 2")
    (is (th/close? -0.00009079573746717529
                   (lp (dist/categorical (mx/array [10.0 0.0 0.0])) 0) 1e-5)
        "categorical(peaked-at-0) at 0")
    (is (th/close? -0.0949229564209606
                   (lp (dist/categorical (mx/array [-1.0 2.0 -1.0])) 1) 1e-5)
        "categorical(peaked-at-1) at 1"))

  (testing "Categorical edge cases"
    (is (th/close? 0.0
                   (lp (dist/categorical (mx/array [100.0 0.0 0.0])) 0) 1e-4)
        "categorical(extreme-peak) at 0")
    (is (th/close? -0.6931471805599453
                   (lp (dist/categorical (mx/array [0.0 0.0])) 1) f32-tol)
        "categorical(binary-uniform) at 1")
    (is (th/close? -2.302585092994046
                   (lp (dist/categorical (mx/array [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])) 5) f32-tol)
        "categorical(uniform-10) at 5"))

  (testing "Dirichlet logpdf spot checks"
    (is (th/close? 0.6931471805599454
                   (lp (dist/dirichlet (mx/array [1.0 1.0 1.0]))
                       (mx/array [0.5 0.3 0.2])) 1e-5)
        "dirichlet([1,1,1]) at [0.5,0.3,0.2]")
    (is (th/close? 2.3109698425284577
                   (lp (dist/dirichlet (mx/array [5.0 2.0 1.0]))
                       (mx/array [0.7 0.2 0.1])) 1e-4)
        "dirichlet([5,2,1]) at [0.7,0.2,0.1]")
    (is (th/close? 1.791759469228055
                   (lp (dist/dirichlet (mx/array [1.0 1.0 1.0 1.0]))
                       (mx/array [0.25 0.25 0.25 0.25])) 1e-4)
        "dirichlet([1,1,1,1]) at uniform"))

  (testing "Dirichlet edge cases"
    (is (th/close? 3.1890247329087273
                   (lp (dist/dirichlet (mx/array [10.0 10.0 10.0]))
                       (mx/array [0.333333 0.333333 0.333334])) 1e-3)
        "dirichlet(concentrated) at near-uniform")
    (is (th/close? 1.8687205103641835
                   (lp (dist/dirichlet (mx/array [1.0 2.0 3.0]))
                       (mx/array [0.1 0.3 0.6])) 1e-4)
        "dirichlet([1,2,3]) at [0.1,0.3,0.6]"))

  (testing "Cauchy logpdf spot checks"
    (is (th/close? -1.1447298858494002
                   (lp (dist/cauchy 0 1) 0) f32-tol)
        "cauchy(0,1) at 0")
    (is (th/close? -4.4028264238708825
                   (lp (dist/cauchy 0 1) 5) f32-tol)
        "cauchy(0,1) at 5")
    (is (th/close? -3.284796049345671
                   (lp (dist/cauchy 2 0.5) 0) f32-tol)
        "cauchy(2,0.5) at 0"))

  (testing "Cauchy edge cases"
    (is (th/close? -10.355170252825918
                   (lp (dist/cauchy 0 1) 100) 1e-4)
        "cauchy(0,1) at 100")
    (is (th/close? -1.8378770664093453
                   (lp (dist/cauchy 5 2) 5) f32-tol)
        "cauchy(5,2) at mode")
    (is (th/close? 3.460440300138691
                   (lp (dist/cauchy 0 0.01) 0) 1e-4)
        "cauchy(0,0.01) at 0 — narrow peak"))

  (testing "Inverse-gamma logpdf spot checks"
    (is (th/close? -1.0
                   (lp (dist/inv-gamma 2 1) 1.0) f32-tol)
        "inv-gamma(2,1) at 1")
    (is (th/close? 0.15888308335967238
                   (lp (dist/inv-gamma 3 2) 0.5) 1e-4)
        "inv-gamma(3,2) at 0.5")
    (is (th/close? -3.4188758248682003
                   (lp (dist/inv-gamma 1 1) 5.0) 1e-4)
        "inv-gamma(1,1) at 5"))

  (testing "Inverse-gamma edge cases"
    (is (th/close? -86.18448944203573
                   (lp (dist/inv-gamma 2 1) 0.01) 0.1)
        "inv-gamma(2,1) at 0.01")
    (is (th/close? -13.825510557964275
                   (lp (dist/inv-gamma 2 1) 100.0) 1e-3)
        "inv-gamma(2,1) at 100"))

  (testing "Geometric logpdf spot checks"
    (is (th/close? -0.6931471805599453
                   (lp (dist/geometric 0.5) 0) f32-tol)
        "geometric(0.5) at 0")
    (is (th/close? -2.2739976361421332
                   (lp (dist/geometric 0.3) 3) 1e-5)
        "geometric(0.3) at 3")
    (is (th/close? -2.407945608651872
                   (lp (dist/geometric 0.9) 1) 1e-5)
        "geometric(0.9) at 1"))

  (testing "Geometric edge cases"
    (is (th/close? -0.01005033585350145
                   (lp (dist/geometric 0.99) 0) 1e-5)
        "geometric(0.99) at 0")
    (is (th/close? -7.6246189861593985
                   (lp (dist/geometric 0.5) 10) 1e-4)
        "geometric(0.5) at 10"))

  (testing "Negative-binomial logpdf spot checks"
    (is (th/close? -1.9898293829901275
                   (lp (dist/neg-binomial 5 0.5) 3) 1e-4)
        "neg-binomial(5,0.5) at 3")
    (is (th/close? -0.6931471805599448
                   (lp (dist/neg-binomial 1 0.5) 0) 1e-5)
        "neg-binomial(1,0.5) at 0")
    (is (th/close? -6.221200803075214
                   (lp (dist/neg-binomial 10 0.3) 5) 1e-3)
        "neg-binomial(10,0.3) at 5")
    (is (th/close? -0.10536051565782584
                   (lp (dist/neg-binomial 1 0.9) 0) 1e-5)
        "neg-binomial(1,0.9) at 0")
    (is (th/close? -4.821258605251945
                   (lp (dist/neg-binomial 3 0.5) 10) 1e-3)
        "neg-binomial(3,0.5) at 10"))

  (testing "Binomial logpdf spot checks"
    (is (th/close? -2.14398006281693
                   (lp (dist/binomial 10 0.5) 3) 1e-4)
        "binomial(10,0.5) at 3")
    (is (th/close? -1.7833747196936622
                   (lp (dist/binomial 5 0.3) 0) 1e-4)
        "binomial(5,0.3) at 0")
    (is (th/close? -1.0536051565782625
                   (lp (dist/binomial 10 0.9) 10) 1e-4)
        "binomial(10,0.9) at 10"))

  (testing "Binomial edge cases"
    (is (th/close? -6.931471805599452
                   (lp (dist/binomial 10 0.5) 10) 1e-4)
        "binomial(10,0.5) at 10 — all heads")
    (is (th/close? -23.02585092994046
                   (lp (dist/binomial 10 0.9) 0) 1e-3)
        "binomial(10,0.9) at 0 — all tails"))

  (testing "Discrete-uniform logpdf spot checks"
    (is (th/close? -1.791759469228055
                   (lp (dist/discrete-uniform 1 6) 3) f32-tol)
        "discrete-uniform(1,6) at 3")
    (is (th/close? -2.302585092994046
                   (lp (dist/discrete-uniform 0 9) 5) f32-tol)
        "discrete-uniform(0,9) at 5")
    (is (th/close? -0.6931471805599453
                   (lp (dist/discrete-uniform 0 1) 0) f32-tol)
        "discrete-uniform(0,1) at 0")
    (is (th/close? -2.3978952727983707
                   (lp (dist/discrete-uniform -5 5) 0) f32-tol)
        "discrete-uniform(-5,5) at 0"))

  (testing "Discrete-uniform out of support"
    (is (= ##-Inf (lp (dist/discrete-uniform 1 6) 7))
        "discrete-uniform(1,6) at 7 — out of support"))

  (testing "Truncated-normal logpdf spot checks"
    (is (th/close? -0.5372233370917658
                   (lp (dist/truncated-normal 0 1 -1 1) 0.0) 1e-5)
        "truncnorm(0,1,-1,1) at 0")
    (is (th/close? -0.3507913526447274
                   (lp (dist/truncated-normal 0 1 0 100) 0.5) 1e-5)
        "truncnorm(0,1,0,100) at 0.5")
    (is (th/close? -1.5995887269107556
                   (lp (dist/truncated-normal 5 2 0 10) 5.0) 1e-4)
        "truncnorm(5,2,0,10) at 5")
    (is (th/close? -2.677370855424669
                   (lp (dist/truncated-normal 0 1 -2 2) 1.9) 1e-4)
        "truncnorm(0,1,-2,2) at 1.9")
    (is (th/close? 1.6111038107692761
                   (lp (dist/truncated-normal 0 1 -0.1 0.1) 0.0) 1e-3)
        "truncnorm(0,1,-0.1,0.1) at 0 — narrow window"))

  (testing "Von Mises logpdf spot checks"
    (is (th/close? -1.0737915351746041
                   (lp (dist/von-mises 0 1) 0.0) 1e-4)
        "von-mises(0,1) at 0")
    (is (th/close? -3.073791535174604
                   (lp (dist/von-mises 0 1) js/Math.PI) 1e-4)
        "von-mises(0,1) at pi")
    (is (th/close? 0.2191507503631165
                   (lp (dist/von-mises 0 10) 0.0) 1e-3)
        "von-mises(0,10) at 0 — concentrated")
    (is (th/close? -0.6618706300613839
                   (lp (dist/von-mises (/ js/Math.PI 2) 2) (/ js/Math.PI 2)) 1e-4)
        "von-mises(pi/2,2) at pi/2")
    (is (th/close? -1.629275755832911
                   (lp (dist/von-mises 0 0.5) 1.0) 1e-4)
        "von-mises(0,0.5) at 1"))

  (testing "MVN logpdf spot checks"
    (is (th/close? -1.8378770664093453
                   (lp (dist/multivariate-normal
                        (mx/array [0.0 0.0])
                        (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [0.0 0.0])) 1e-4)
        "mvn(0,I₂) at 0")
    (is (th/close? -3.694036030183455
                   (lp (dist/multivariate-normal
                        (mx/array [0.0 0.0])
                        (mx/array [[1.0 0.5] [0.5 1.0]]))
                       (mx/array [1.0 2.0])) 1e-4)
        "mvn(0,correlated) at [1,2]")
    (is (th/close? -9.756815599614018
                   (lp (dist/multivariate-normal
                        (mx/array [1.0 2.0 3.0])
                        (mx/array [[1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0]]))
                       (mx/array [0.0 0.0 0.0])) 1e-3)
        "mvn([1,2,3],I₃) at 0"))

  (testing "MVN edge cases"
    (is (th/close? -1.1390904103669408
                   (lp (dist/multivariate-normal
                        (mx/array [0.0 0.0])
                        (mx/array [[1.0 0.9] [0.9 1.0]]))
                       (mx/array [0.5 0.5])) 1e-4)
        "mvn(0,high-corr) at [0.5,0.5]")
    (is (th/close? -5.837877066409345
                   (lp (dist/multivariate-normal
                        (mx/array [1.0 1.0])
                        (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [3.0 3.0])) 1e-4)
        "mvn([1,1],I₂) at [3,3]"))

  (testing "Wishart logpdf spot checks"
    (is (th/close? -5.322783716197342
                   (lp (dist/wishart 5 (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [[1.0 0.0] [0.0 1.0]])) 1e-3)
        "wishart(5,I₂) at I₂")
    (is (th/close? -5.073583861388082
                   (lp (dist/wishart 5 (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [[2.0 0.5] [0.5 3.0]])) 1e-3)
        "wishart(5,I₂) at off-diagonal")
    (is (th/close? -14.32898923286907
                   (lp (dist/wishart 10 (mx/array [[2.0 1.0] [1.0 2.0]]))
                       (mx/array [[3.0 1.0] [1.0 2.0]])) 1e-2)
        "wishart(10,correlated) at pos-def")
    (is (th/close? -3.031024246969334
                   (lp (dist/wishart 3 (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [[0.5 0.0] [0.0 0.5]])) 1e-3)
        "wishart(3,I₂) at 0.5*I₂"))

  (testing "Inverse-Wishart logpdf spot checks"
    (is (th/close? -5.322783716197341
                   (lp (dist/inv-wishart 5 (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [[1.0 0.0] [0.0 1.0]])) 1e-3)
        "inv-wishart(5,I₂) at I₂")
    (is (th/close? -11.75436574413003
                   (lp (dist/inv-wishart 5 (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [[2.0 0.5] [0.5 3.0]])) 1e-2)
        "inv-wishart(5,I₂) at off-diagonal")
    (is (th/close? -0.37214116360966276
                   (lp (dist/inv-wishart 4 (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [[0.5 0.0] [0.0 0.5]])) 1e-3)
        "inv-wishart(4,I₂) at 0.5*I₂")
    (is (th/close? 2.7675711727617855
                   (lp (dist/inv-wishart 6 (mx/array [[1.0 0.0] [0.0 1.0]]))
                       (mx/array [[0.25 0.0] [0.0 0.25]])) 1e-3)
        "inv-wishart(6,I₂) at 0.25*I₂")))

;; =========================================================================
;; SECTION 1c: Distributions not in cross-system runner (issue #11)
;; Reference values computed analytically / verified against scipy
;; =========================================================================

(deftest s1c-remaining-distribution-logpdf
  (testing "Wrapped-Cauchy logpdf spot checks"
    (is (th/close? -0.73926477774123556
                   (lp (dist/wrapped-cauchy 0 0.5) 0.0) 1e-4)
        "wrapped-cauchy(0,0.5) at 0")
    (is (th/close? -2.9364893550774549
                   (lp (dist/wrapped-cauchy 0 0.5) js/Math.PI) 1e-4)
        "wrapped-cauchy(0,0.5) at pi")
    (is (th/close? 0.35934751092687245
                   (lp (dist/wrapped-cauchy 1 0.8) 1.0) 1e-3)
        "wrapped-cauchy(1,0.8) at mode")
    (is (th/close? -1.8178763997026757
                   (lp (dist/wrapped-cauchy 0 0.01) 0.0) 1e-4)
        "wrapped-cauchy(0,0.01) at 0 — near-uniform")
    (is (th/close? 3.4554277583152597
                   (lp (dist/wrapped-cauchy 0 0.99) 0.0) 1e-2)
        "wrapped-cauchy(0,0.99) at 0 — near-delta"))

  (testing "Wrapped-Normal logpdf spot checks"
    (is (th/close? -0.91893852785409669
                   (lp (dist/wrapped-normal 0 1) 0.0) 1e-5)
        "wrapped-normal(0,1) at 0")
    (is (th/close? -5.1605935531894058
                   (lp (dist/wrapped-normal 0 1) js/Math.PI) 1e-3)
        "wrapped-normal(0,1) at pi")
    (is (th/close? 1.3836465597893728
                   (lp (dist/wrapped-normal 0 0.1) 0.0) 1e-4)
        "wrapped-normal(0,0.1) at 0 — concentrated")
    (is (th/close? -1.5978044067644803
                   (lp (dist/wrapped-normal 1 2) 1.0) 1e-4)
        "wrapped-normal(1,2) at mode")
    (is (th/close? -1.8603455658559758
                   (lp (dist/wrapped-normal 0 1.5) (/ js/Math.PI 2)) 1e-4)
        "wrapped-normal(0,1.5) at pi/2"))

  (testing "Piecewise-uniform logpdf spot checks"
    (is (th/close? -0.6931471805599453
                   (lp (dist/piecewise-uniform (mx/array [0 1 2]) (mx/array [1 1])) 0.5) f32-tol)
        "piecewise-uniform([0,1,2],[1,1]) at 0.5")
    (is (th/close? -0.6931471805599453
                   (lp (dist/piecewise-uniform (mx/array [0 1 2]) (mx/array [1 1])) 1.5) f32-tol)
        "piecewise-uniform([0,1,2],[1,1]) at 1.5")
    (is (th/close? -1.3862943611198906
                   (lp (dist/piecewise-uniform (mx/array [0 1 2 3]) (mx/array [1 2 1])) 0.5) f32-tol)
        "piecewise-uniform 3-bin, low-weight bin")
    (is (th/close? -0.6931471805599453
                   (lp (dist/piecewise-uniform (mx/array [0 1 2 3]) (mx/array [1 2 1])) 1.5) f32-tol)
        "piecewise-uniform 3-bin, high-weight bin"))

  (testing "Piecewise-uniform out of support"
    (is (= ##-Inf (lp (dist/piecewise-uniform (mx/array [0 1 2]) (mx/array [1 1])) 5.0))
        "piecewise-uniform out of support"))

  (testing "Mixture logpdf spot checks"
    (is (th/close? -1.61208198711839
                   (lp (dc/mixture [(dist/gaussian 0 1) (dist/gaussian 5 1)]
                                   (mx/array [0.0 0.0])) 0) 1e-4)
        "mixture(N(0,1)+N(5,1), equal) at 0")
    (is (th/close? -4.0439385332046722
                   (lp (dc/mixture [(dist/gaussian 0 1) (dist/gaussian 5 1)]
                                   (mx/array [0.0 0.0])) 2.5) 1e-4)
        "mixture(N(0,1)+N(5,1), equal) at midpoint 2.5")
    (is (th/close? -1.024299048862499
                   (lp (dc/mixture [(dist/gaussian 0 1) (dist/gaussian 10 1)]
                                   (mx/array [(js/Math.log 0.9) (js/Math.log 0.1)])) 0) 1e-4)
        "mixture(0.9*N(0,1)+0.1*N(10,1)) at 0"))

  (testing "Product logpdf spot checks"
    (let [lp-prod (fn [d vals]
                    (let [r (dc/dist-log-prob d vals)] (mx/eval! r) (mx/item r)))]
      (is (th/close? -1.9189385332046727
                     (lp-prod (dc/product [(dist/gaussian 0 1) (dist/exponential 1)])
                              [(mx/scalar 0) (mx/scalar 1)]) 1e-4)
          "product(N(0,1), Exp(1)) at [0,1]")
      (is (th/close? -3.4189385332046727
                     (lp-prod (dc/product [(dist/gaussian 0 1) (dist/exponential 1)])
                              [(mx/scalar 1) (mx/scalar 2)]) 1e-4)
          "product(N(0,1), Exp(1)) at [1,2]")
      (is (th/close? -1.4006134771434051
                     (lp-prod (dc/product [(dist/gaussian 0 1) (dist/uniform 0 1) (dist/bernoulli 0.7)])
                              [(mx/scalar 0.5) (mx/scalar 0.3) (mx/scalar 1)]) 1e-4)
          "product(N(0,1), U(0,1), Bern(0.7)) at [0.5, 0.3, 1]")))

  (testing "IID logpdf spot checks"
    (is (th/close? -5.2568155996140185
                   (lp (dist/iid (dist/gaussian 0 1) 3)
                       (mx/array [0 1 2])) 1e-4)
        "iid(N(0,1), 3) at [0,1,2]")
    (is (th/close? -1.6137056388801092
                   (lp (dist/iid (dist/exponential 2) 2)
                       (mx/array [0.5 1.0])) 1e-4)
        "iid(Exp(2), 2) at [0.5,1.0]")
    (is (th/close? -0.90316541057890953
                   (lp (dist/iid (dist/gaussian 3 0.5) 4)
                       (mx/array [3 3 3 3])) 1e-4)
        "iid(N(3,0.5), 4) at [3,3,3,3] — all at mean")))

;; =========================================================================
;; SECTION 1d: Sample mean/variance statistical tests (issue #11)
;; Catches sampler bugs where log-prob is correct but samples are wrong.
;; N=10000, tolerances at ~5x expected standard error.
;; =========================================================================

(def ^:private N 10000)

(defn- sample-moments
  "Sample N values from dist, return [mean variance] as JS numbers.
   Casts to float32 for integer-valued distributions."
  [d n seed]
  (let [key (rng/fresh-key seed)
        samples (dc/dist-sample-n d key n)
        _ (mx/eval! samples)
        fs (if (= (mx/dtype samples) mx/float32) samples (mx/astype samples mx/float32))
        m (mx/mean fs)
        v (mx/mean (mx/square (mx/subtract fs m)))
        _ (mx/eval! m v)]
    [(mx/item m) (mx/item v)]))

(defn- circular-moments
  "Sample N values from circular dist, return [circular-mean mean-resultant-length]."
  [d n seed]
  (let [key (rng/fresh-key seed)
        samples (dc/dist-sample-n d key n)
        _ (mx/eval! samples)
        mc (mx/mean (mx/cos samples))
        ms (mx/mean (mx/sin samples))
        _ (mx/eval! mc ms)
        c (mx/item mc) s (mx/item ms)]
    [(js/Math.atan2 s c)
     (js/Math.sqrt (+ (* c c) (* s s)))]))

(deftest s1d-sample-moments
  (testing "Gaussian sample moments"
    (let [[m v] (sample-moments (dist/gaussian 2 3) N 100)]
      (is (th/close? 2.0 m 0.15) "gaussian(2,3) mean ≈ 2")
      (is (th/close? 9.0 v 0.6) "gaussian(2,3) var ≈ 9")))

  (testing "Uniform sample moments"
    (let [[m v] (sample-moments (dist/uniform 1 5) N 101)]
      (is (th/close? 3.0 m 0.1) "uniform(1,5) mean ≈ 3")
      (is (th/close? 1.333333 v 0.1) "uniform(1,5) var ≈ 4/3")))

  (testing "Bernoulli sample moments"
    (let [[m v] (sample-moments (dist/bernoulli 0.3) N 102)]
      (is (th/close? 0.3 m 0.03) "bernoulli(0.3) mean ≈ 0.3")
      (is (th/close? 0.21 v 0.03) "bernoulli(0.3) var ≈ 0.21")))

  (testing "Beta sample moments"
    (let [[m v] (sample-moments (dist/beta-dist 2 5) N 103)]
      (is (th/close? 0.285714 m 0.03) "beta(2,5) mean ≈ 2/7")
      (is (th/close? 0.025510 v 0.005) "beta(2,5) var ≈ 5/196")))

  (testing "Gamma sample moments"
    (let [[m v] (sample-moments (dist/gamma-dist 3 2) N 104)]
      (is (th/close? 1.5 m 0.06) "gamma(3,2) mean ≈ 1.5")
      (is (th/close? 0.75 v 0.06) "gamma(3,2) var ≈ 0.75")))

  (testing "Exponential sample moments"
    (let [[m v] (sample-moments (dist/exponential 4) N 105)]
      (is (th/close? 0.25 m 0.02) "exponential(4) mean ≈ 0.25")
      (is (th/close? 0.0625 v 0.008) "exponential(4) var ≈ 1/16")))

  (testing "Laplace sample moments"
    (let [[m v] (sample-moments (dist/laplace 1 2) N 106)]
      (is (th/close? 1.0 m 0.2) "laplace(1,2) mean ≈ 1")
      (is (th/close? 8.0 v 1.0) "laplace(1,2) var ≈ 8")))

  (testing "Log-normal sample moments"
    (let [[m v] (sample-moments (dist/log-normal 0 1) N 108)]
      (is (th/close? 1.6487 m 0.15) "log-normal(0,1) mean ≈ e^0.5")
      (is (th/close? 4.6708 v 1.5) "log-normal(0,1) var ≈ (e-1)*e")))

  (testing "Categorical sample moments"
    ;; p = [0.2, 0.3, 0.5], mean = 1.3, var = 0.61
    (let [logits (mx/array [(js/Math.log 0.2) (js/Math.log 0.3) (js/Math.log 0.5)])
          [m v] (sample-moments (dist/categorical logits) N 110)]
      (is (th/close? 1.3 m 0.06) "categorical([.2,.3,.5]) mean ≈ 1.3")
      (is (th/close? 0.61 v 0.06) "categorical([.2,.3,.5]) var ≈ 0.61")))

  (testing "Inverse-gamma sample moments (shape=10)"
    ;; shape=10 has finite kurtosis (=16), so variance estimate converges.
    ;; mean = scale/(shape-1) = 9/9 = 1, var = scale²/((shape-1)²(shape-2)) = 81/648 = 0.125
    (let [[m v] (sample-moments (dist/inv-gamma 10 9) N 111)]
      (is (th/close? 1.0 m 0.05) "inv-gamma(10,9) mean ≈ 1")
      (is (th/close? 0.125 v 0.03) "inv-gamma(10,9) var ≈ 1/8")))

  (testing "Geometric sample moments"
    (let [[m v] (sample-moments (dist/geometric 0.4) N 112)]
      (is (th/close? 1.5 m 0.15) "geometric(0.4) mean ≈ 1.5")
      (is (th/close? 3.75 v 0.5) "geometric(0.4) var ≈ 3.75")))

  (testing "Binomial sample moments"
    (let [[m v] (sample-moments (dist/binomial 10 0.3) N 114)]
      (is (th/close? 3.0 m 0.1) "binomial(10,0.3) mean ≈ 3")
      (is (th/close? 2.1 v 0.2) "binomial(10,0.3) var ≈ 2.1")))

  (testing "Discrete-uniform sample moments"
    (let [[m v] (sample-moments (dist/discrete-uniform 1 6) N 115)]
      (is (th/close? 3.5 m 0.1) "disc-uniform(1,6) mean ≈ 3.5")
      (is (th/close? 2.9167 v 0.25) "disc-uniform(1,6) var ≈ 35/12")))

  (testing "Von Mises circular moments"
    ;; R = I1(2)/I0(2) ≈ 0.6978
    (let [[cm R] (circular-moments (dist/von-mises 0 2) N 116)]
      (is (th/close? 0.0 cm 0.05) "von-mises(0,2) circular mean ≈ 0")
      (is (th/close? 0.6978 R 0.03) "von-mises(0,2) R ≈ I1(2)/I0(2)")))

  (testing "Wrapped-cauchy circular moments"
    ;; For wrapped-cauchy(mu, rho): R = rho exactly
    (let [[cm R] (circular-moments (dist/wrapped-cauchy 0 0.5) N 118)]
      (is (th/close? 0.0 cm 0.05) "wrapped-cauchy(0,0.5) circular mean ≈ 0")
      (is (th/close? 0.5 R 0.03) "wrapped-cauchy(0,0.5) R ≈ rho = 0.5")))

  (testing "Wrapped-normal circular moments"
    ;; R = exp(-sigma^2/2) = exp(-0.5) ≈ 0.6065
    (let [[cm R] (circular-moments (dist/wrapped-normal 0 1) N 117)]
      (is (th/close? 0.0 cm 0.05) "wrapped-normal(0,1) circular mean ≈ 0")
      (is (th/close? 0.6065 R 0.03) "wrapped-normal(0,1) R ≈ exp(-0.5)"))))

;; Separate deftest for distributions without native dist-sample-n*
;; (poisson, neg-binomial use sequential fallback — smaller N to avoid Metal buffer limit)

(deftest s1d2-sample-moments-sequential
  (testing "Student-t sample moments (df=5)"
    (let [[m v] (sample-moments (dist/student-t 5 1 2) N 107)]
      (is (th/close? 1.0 m 0.2) "student-t(5,1,2) mean ≈ 1")
      (is (th/close? 6.6667 v 1.5) "student-t(5,1,2) var ≈ 20/3")))

  (testing "Poisson sample moments (N=2000, no native batch)"
    (let [[m v] (sample-moments (dist/poisson 3.5) 2000 109)]
      (is (th/close? 3.5 m 0.25) "poisson(3.5) mean ≈ 3.5")
      (is (th/close? 3.5 v 0.6) "poisson(3.5) var ≈ 3.5")))

  (testing "Negative-binomial sample moments (N=2000, no native batch)"
    (let [[m v] (sample-moments (dist/neg-binomial 3 0.4) 2000 113)]
      (is (th/close? 4.5 m 0.5) "neg-binom(3,0.4) mean ≈ 4.5")
      (is (th/close? 11.25 v 2.5) "neg-binom(3,0.4) var ≈ 11.25"))))

;; =========================================================================
;; SECTION 1e: Normalization tests (issue #11)
;; Verify exp(log-prob) integrates/sums to 1 over the support.
;; Catches missing normalization constants, sign errors, wrong denominators.
;; =========================================================================

(defn- integrate-logpdf
  "Vectorized midpoint rule: exp(log-prob(xs)) summed in one MLX op."
  [d lo hi n-pts]
  (let [dx (/ (- hi lo) n-pts)
        xs (mx/array (mapv #(+ lo (* (+ % 0.5) dx)) (range n-pts)))
        lps (dist/log-prob d xs)
        integral (mx/multiply (mx/sum (mx/exp lps)) (mx/scalar dx))
        _ (mx/eval! integral)]
    (mx/item integral)))

(defn- sum-pmf
  "Vectorized: sum exp(log-prob(ks)) in one MLX op."
  [d ks]
  (let [arr (mx/array (vec ks))
        lps (dist/log-prob d arr)
        s (mx/sum (mx/exp lps))
        _ (mx/eval! s)]
    (mx/item s)))

(def ^:private norm-tol 0.005)

(deftest s1e-normalization
  (testing "Continuous distributions integrate to 1"
    (is (th/close? 1.0 (integrate-logpdf (dist/gaussian 0 1) -8 8 500) norm-tol)
        "gaussian(0,1)")
    (is (th/close? 1.0 (integrate-logpdf (dist/uniform 2 7) 1.5 7.5 500) norm-tol)
        "uniform(2,7)")
    (is (th/close? 1.0 (integrate-logpdf (dist/exponential 2) 0 20 500) norm-tol)
        "exponential(2)")
    (is (th/close? 1.0 (integrate-logpdf (dist/laplace 1 2) -20 22 500) norm-tol)
        "laplace(1,2)")
    (is (th/close? 1.0 (integrate-logpdf (dist/beta-dist 2 5) 0.001 0.999 500) norm-tol)
        "beta(2,5)")
    (is (th/close? 1.0 (integrate-logpdf (dist/gamma-dist 3 2) 0.001 20 1000) norm-tol)
        "gamma(3,2)")
    (is (th/close? 1.0 (integrate-logpdf (dist/log-normal 0 1) 0.001 30 1000) norm-tol)
        "log-normal(0,1)")
    (is (th/close? 1.0 (integrate-logpdf (dist/cauchy 0 1) -200 200 4000) norm-tol)
        "cauchy(0,1)")
    (is (th/close? 1.0 (integrate-logpdf (dist/student-t 3 0 1) -50 50 2000) norm-tol)
        "student-t(3,0,1)")
    (is (th/close? 1.0 (integrate-logpdf (dist/inv-gamma 2 1) 0.001 50 2000) norm-tol)
        "inv-gamma(2,1)")
    (is (th/close? 1.0 (integrate-logpdf (dist/truncated-normal 0 1 -2 2) -2 2 500) norm-tol)
        "truncated-normal(0,1,-2,2)"))

  (testing "Circular distributions integrate to 1 over [-π, π)"
    (is (th/close? 1.0 (integrate-logpdf (dist/von-mises 0 2) (- js/Math.PI) js/Math.PI 500) norm-tol)
        "von-mises(0,2)")
    (is (th/close? 1.0 (integrate-logpdf (dist/wrapped-cauchy 0 0.5) (- js/Math.PI) js/Math.PI 500) norm-tol)
        "wrapped-cauchy(0,0.5)")
    (is (th/close? 1.0 (integrate-logpdf (dist/wrapped-normal 0 1) (- js/Math.PI) js/Math.PI 500) norm-tol)
        "wrapped-normal(0,1)"))

  (testing "Discrete distributions sum to 1"
    (is (th/close? 1.0 (sum-pmf (dist/bernoulli 0.3) [0 1]) norm-tol)
        "bernoulli(0.3)")
    (is (th/close? 1.0 (sum-pmf (dist/poisson 5) (range 0 50)) norm-tol)
        "poisson(5)")
    (is (th/close? 1.0 (sum-pmf (dist/binomial 10 0.3) (range 0 11)) norm-tol)
        "binomial(10,0.3)")
    (is (th/close? 1.0 (sum-pmf (dist/geometric 0.4) (range 0 80)) norm-tol)
        "geometric(0.4)")
    (is (th/close? 1.0 (sum-pmf (dist/neg-binomial 3 0.4) (range 0 80)) norm-tol)
        "neg-binomial(3,0.4)")
    (is (th/close? 1.0 (sum-pmf (dist/discrete-uniform 1 6) (range 1 7)) norm-tol)
        "discrete-uniform(1,6)")
    (is (th/close? 1.0 (sum-pmf (dist/categorical (mx/array [1.0 2.0 3.0])) (range 0 3)) norm-tol)
        "categorical([1,2,3])")))

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
            score (:score trace)]
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
            retval (:retval trace)]
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
                                        (let [slope (trace :slope (dist/gaussian 0 1))
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
                                        (let [slope (trace :slope (dist/gaussian 0 1))
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
