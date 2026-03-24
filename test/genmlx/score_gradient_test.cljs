(ns genmlx.score-gradient-test
  "Phase 4.2: Model score gradient tests.
   Verifies gradients flow correctly through generative function models —
   through gen bodies, generate weight, update weight, and choice-gradients.

   All models are Gaussian (fully reparameterizable). Each autodiff gradient
   is verified against both analytical derivation and central-difference FD.

   Gradient derivations (score = log p(all choices | args)):

   single-model: score = log N(mu;0,10) + log N(y;mu,1)
     d/d(mu) = -mu/100 + (y - mu)

   linreg-model: score = log N(s;0,10) + log N(b;0,10) + log N(y; s*x+b, 1)
     d/d(s) = -s/100 + (y - s*x - b)*x
     d/d(b) = -b/100 + (y - s*x - b)

   chain-model: score = log N(z1;0,1) + log N(z2;z1,1) + log N(y;z2,0.5)
     d/d(z1) = -z1 + (z2 - z1) = z2 - 2*z1
     d/d(z2) = -(z2-z1) + (y-z2)/0.25 = z1 - 5*z2 + 4*y
     d/d(y)  = -(y - z2)/0.25"
  (:require [cljs.test :refer [deftest is testing are run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.gradients :as grad]
            [genmlx.gradient-fd-test :as fd]
            [genmlx.test-helpers :as th]))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def single-model
  "Single latent: mu ~ N(0,10), y ~ N(mu,1)."
  (dyn/auto-key
    (gen [y-val]
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (trace :y (dist/gaussian mu 1))
        mu))))

(def linreg-model
  "Linear regression: slope,intercept ~ N(0,10), y ~ N(slope*x + intercept, 1)."
  (dyn/auto-key
    (gen [x]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))))))

(def chain-model
  "Dependent chain: z1 ~ N(0,1), z2 ~ N(z1,1), y ~ N(z2,0.5)."
  (dyn/auto-key
    (gen []
      (let [z1 (trace :z1 (dist/gaussian 0 1))
            z2 (trace :z2 (dist/gaussian z1 1))]
        (trace :y (dist/gaussian z2 0.5))))))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- weight-of
  "Generate with full constraints, return realized weight."
  [model args constraints]
  (-> (p/generate model args constraints) :weight mx/realize))

;; ---------------------------------------------------------------------------
;; A. choice-gradients: FD + analytical verification
;; ---------------------------------------------------------------------------

(deftest single-model-choice-gradient-matches-fd
  (testing "d(score)/d(mu) at mu=2, y=3 equals 0.98 and matches FD"
    (let [{:keys [trace]} (p/generate single-model [(mx/scalar 3.0)]
                                      (cm/choicemap :mu (mx/scalar 2.0)
                                                    :y (mx/scalar 3.0)))
          {:keys [mu]} (grad/choice-gradients single-model trace [:mu])
          ad (mx/realize mu)
          ;; FD: vary mu in generate weight
          score-fn (fn [mu-arr]
                     (:weight (p/generate single-model [(mx/scalar 3.0)]
                                          (cm/choicemap :mu mu-arr
                                                        :y (mx/scalar 3.0)))))
          fd-val (fd/fd-gradient score-fn 2.0)
          expected 0.98]
      (is (fd/gradient-close? ad expected)
          "autodiff matches analytical: -mu/100 + (y - mu)")
      (is (fd/gradient-close? ad fd-val)
          "autodiff matches central-difference FD"))))

(deftest single-model-choice-gradient-at-posterior-mode
  (testing "d(score)/d(mu) vanishes at posterior mode"
    ;; Posterior mode: mu* = (0/100 + y/1) / (1/100 + 1/1) = y / 1.01
    ;; At y=2: mu* = 2/1.01 ~ 1.9802
    (let [y 2.0
          mu-star (/ y 1.01)
          {:keys [trace]} (p/generate single-model [(mx/scalar y)]
                                      (cm/choicemap :mu (mx/scalar mu-star)
                                                    :y (mx/scalar y)))
          {:keys [mu]} (grad/choice-gradients single-model trace [:mu])]
      (is (th/close? 0.0 (mx/realize mu) 1e-4)
          "gradient is zero at posterior mode"))))

(deftest linreg-choice-gradients-zero-residual
  (testing "zero residual: gradients are just prior terms"
    (let [{:keys [trace]} (p/generate linreg-model [(mx/scalar 1.0)]
                                      (cm/choicemap :slope (mx/scalar 2.0)
                                                    :intercept (mx/scalar 1.0)
                                                    :y (mx/scalar 3.0)))
          grads (grad/choice-gradients linreg-model trace [:slope :intercept])]
      (is (th/close? -0.02 (mx/realize (:slope grads)) 1e-4)
          "d/d(slope) = -slope/100 when residual is zero")
      (is (th/close? -0.01 (mx/realize (:intercept grads)) 1e-4)
          "d/d(intercept) = -intercept/100 when residual is zero"))))

(deftest linreg-choice-gradients-nonzero-residual
  (testing "nonzero residual: gradients include likelihood term"
    (let [{:keys [trace]} (p/generate linreg-model [(mx/scalar 1.0)]
                                      (cm/choicemap :slope (mx/scalar 1.5)
                                                    :intercept (mx/scalar 0.5)
                                                    :y (mx/scalar 3.0)))
          grads (grad/choice-gradients linreg-model trace [:slope :intercept])
          ;; d/d(slope) = -1.5/100 + (3 - 2)*1 = 0.985
          ;; d/d(intercept) = -0.5/100 + (3 - 2) = 0.995
          ad-slope (mx/realize (:slope grads))
          ad-intercept (mx/realize (:intercept grads))]
      (is (th/close? 0.985 ad-slope 1e-3)
          "d/d(slope) = -s/100 + residual*x")
      (is (th/close? 0.995 ad-intercept 1e-3)
          "d/d(intercept) = -b/100 + residual"))))

(deftest linreg-choice-gradients-match-fd
  (testing "linreg gradients match FD at nonzero residual"
    (let [{:keys [trace]} (p/generate linreg-model [(mx/scalar 1.0)]
                                      (cm/choicemap :slope (mx/scalar 1.5)
                                                    :intercept (mx/scalar 0.5)
                                                    :y (mx/scalar 3.0)))
          grads (grad/choice-gradients linreg-model trace [:slope :intercept])
          ;; FD for slope
          slope-fn (fn [s-arr]
                     (:weight (p/generate linreg-model [(mx/scalar 1.0)]
                                          (cm/choicemap :slope s-arr
                                                        :intercept (mx/scalar 0.5)
                                                        :y (mx/scalar 3.0)))))
          fd-slope (fd/fd-gradient slope-fn 1.5)
          ;; FD for intercept
          intercept-fn (fn [b-arr]
                         (:weight (p/generate linreg-model [(mx/scalar 1.0)]
                                              (cm/choicemap :slope (mx/scalar 1.5)
                                                            :intercept b-arr
                                                            :y (mx/scalar 3.0)))))
          fd-intercept (fd/fd-gradient intercept-fn 0.5)]
      (is (fd/gradient-close? (mx/realize (:slope grads)) fd-slope)
          "slope autodiff matches FD")
      (is (fd/gradient-close? (mx/realize (:intercept grads)) fd-intercept)
          "intercept autodiff matches FD"))))

(deftest chain-model-choice-gradients-analytical
  (testing "chain model gradients match analytical derivation"
    (are [z1 z2 y expected-z1 expected-z2]
      (let [{:keys [trace]} (p/generate chain-model []
                                        (cm/choicemap :z1 (mx/scalar z1)
                                                      :z2 (mx/scalar z2)
                                                      :y (mx/scalar y)))
            grads (grad/choice-gradients chain-model trace [:z1 :z2])]
        (and (th/close? expected-z1 (mx/realize (:z1 grads)) 1e-4)
             (th/close? expected-z2 (mx/realize (:z2 grads)) 1e-4)))
      ;; z1   z2   y    d/dz1=z2-2z1   d/dz2=z1-5z2+4y
      0.5  1.0  1.5  0.0              1.5
      1.0  0.5  2.0  -1.5             6.5
      0.0  0.0  1.0  0.0              4.0
      -1.0 1.0  0.0  3.0              -6.0)))

(deftest chain-model-choice-gradients-match-fd
  (testing "chain model choice-gradients match FD for all three sites"
    (let [{:keys [trace]} (p/generate chain-model []
                                      (cm/choicemap :z1 (mx/scalar 1.0)
                                                    :z2 (mx/scalar 0.5)
                                                    :y (mx/scalar 2.0)))
          grads (grad/choice-gradients chain-model trace [:z1 :z2 :y])
          base-cm (cm/choicemap :z1 (mx/scalar 1.0)
                                :z2 (mx/scalar 0.5)
                                :y (mx/scalar 2.0))]
      (doseq [[addr point] [[:z1 1.0] [:z2 0.5] [:y 2.0]]]
        (let [score-fn (fn [v-arr]
                         (:weight (p/generate chain-model []
                                              (cm/set-choice base-cm [addr] v-arr))))
              fd-val (fd/fd-gradient score-fn point)
              ad-val (mx/realize (get grads addr))]
          (is (fd/gradient-close? ad-val fd-val)
              (str "choice-gradient FD match for " addr)))))))

;; ---------------------------------------------------------------------------
;; B. score-gradient: FD verification
;; ---------------------------------------------------------------------------

(deftest score-gradient-single-model
  (testing "score-gradient returns correct score and gradient"
    (let [y-obs (cm/choicemap :y (mx/scalar 3.0))
          params (mx/array [2.0])
          {:keys [score grad]} (grad/score-gradient single-model [(mx/scalar 3.0)]
                                                    y-obs [:mu] params)
          grad-val (first (mx/realize-clj grad))]
      (is (th/close? -4.6605 (mx/realize score) 1e-3)
          "score equals log N(2;0,10) + log N(3;2,1)")
      (is (th/close? 0.98 grad-val 1e-3)
          "gradient matches analytical derivation"))))

(deftest score-gradient-linreg-matches-fd
  (testing "score-gradient for linreg matches element-wise FD"
    (let [x-val (mx/scalar 1.0)
          y-obs (cm/choicemap :y (mx/scalar 3.0))
          addrs [:slope :intercept]
          param-vals [1.5 0.5]
          {:keys [grad]} (grad/score-gradient linreg-model [x-val] y-obs
                                              addrs (mx/array param-vals))
          grad-vec (mx/realize-clj grad)
          ;; Element-wise FD
          fd-vec (mapv (fn [i]
                         (let [p+ (update param-vals i + 1e-3)
                               p- (update param-vals i - 1e-3)
                               w+ (weight-of linreg-model [x-val]
                                             (cm/choicemap :slope (mx/scalar (p+ 0))
                                                           :intercept (mx/scalar (p+ 1))
                                                           :y (mx/scalar 3.0)))
                               w- (weight-of linreg-model [x-val]
                                             (cm/choicemap :slope (mx/scalar (p- 0))
                                                           :intercept (mx/scalar (p- 1))
                                                           :y (mx/scalar 3.0)))]
                           (/ (- w+ w-) 2e-3)))
                       [0 1])]
      (is (fd/gradient-close? (grad-vec 0) (fd-vec 0))
          "slope gradient matches FD")
      (is (fd/gradient-close? (grad-vec 1) (fd-vec 1))
          "intercept gradient matches FD"))))

(deftest score-gradient-chain-model
  (testing "score-gradient for chain model with observations"
    (let [y-obs (cm/choicemap :y (mx/scalar 1.5))
          addrs [:z1 :z2]
          params (mx/array [0.5 1.0])
          {:keys [grad]} (grad/score-gradient chain-model [] y-obs addrs params)
          [g-z1 g-z2] (mx/realize-clj grad)]
      ;; d/dz1 = z2 - 2*z1 = 1.0 - 1.0 = 0.0
      ;; d/dz2 = z1 - 5*z2 + 4*y = 0.5 - 5.0 + 6.0 = 1.5
      (is (th/close? 0.0 g-z1 1e-4) "z1 gradient is zero at this point")
      (is (th/close? 1.5 g-z2 1e-4) "z2 gradient matches derivation"))))

;; ---------------------------------------------------------------------------
;; C. Generate weight gradient
;; ---------------------------------------------------------------------------

(deftest generate-weight-gradient-wrt-observation
  (testing "d(weight)/d(y) = -(y - mu) for Normal-Normal model"
    (are [mu y expected]
      (let [score-fn (fn [y-arr]
                       (:weight (p/generate single-model [(mx/scalar 3.0)]
                                            (cm/choicemap :mu (mx/scalar mu)
                                                          :y y-arr))))
            ad (fd/analytical-gradient score-fn y)
            fd-val (fd/fd-gradient score-fn y)]
        (and (fd/gradient-close? ad expected)
             (fd/gradient-close? ad fd-val)))
      ;; mu   y     d(w)/d(y) = -(y-mu)
      0.5  1.0   -0.5
      2.0  3.0   -1.0
      2.0  2.0    0.0
      -1.0 1.0   -2.0)))

(deftest generate-weight-gradient-wrt-latent
  (testing "d(weight)/d(mu) through generate for single-model"
    (let [score-fn (fn [mu-arr]
                     (:weight (p/generate single-model [(mx/scalar 3.0)]
                                          (cm/choicemap :mu mu-arr
                                                        :y (mx/scalar 3.0)))))
          ad (fd/analytical-gradient score-fn 2.0)
          fd-val (fd/fd-gradient score-fn 2.0)]
      (is (fd/gradient-close? ad 0.98)
          "d(weight)/d(mu) matches analytical 0.98")
      (is (fd/gradient-close? ad fd-val)
          "autodiff matches FD"))))

(deftest generate-weight-gradient-linreg
  (testing "d(weight)/d(y) for linreg at different x values"
    ;; weight = score. d(weight)/d(y) = -(y - (slope*x + intercept))
    (are [slope intercept x y expected]
      (let [score-fn (fn [y-arr]
                       (:weight (p/generate linreg-model [(mx/scalar x)]
                                            (cm/choicemap :slope (mx/scalar slope)
                                                          :intercept (mx/scalar intercept)
                                                          :y y-arr))))
            ad (fd/analytical-gradient score-fn y)]
        (th/close? expected ad 1e-3))
      ;; slope intercept x   y     d/dy = -(y - (s*x+b))
      2.0   1.0       1.0 3.0   0.0
      2.0   1.0       1.0 4.0   -1.0
      1.0   0.0       2.0 3.0   -1.0)))

;; ---------------------------------------------------------------------------
;; D. Update weight gradient
;; ---------------------------------------------------------------------------

(deftest update-weight-gradient-chain-model
  (testing "d(update_weight)/d(new_z2) for chain model"
    ;; d/d(z2_new) = z1 - 5*z2_new + 4*y
    (let [{:keys [trace]} (p/generate chain-model []
                                      (cm/choicemap :z1 (mx/scalar 0.5)
                                                    :z2 (mx/scalar 1.0)
                                                    :y (mx/scalar 1.0)))
          update-fn (fn [z2-arr]
                      (:weight (p/update chain-model trace
                                         (cm/choicemap :z2 z2-arr))))
          grad-fn (mx/grad update-fn)]
      (are [z2-new expected]
        (th/close? expected (mx/realize (grad-fn (mx/scalar z2-new))) 1e-3)
        ;; z2_new  d/dz2 = 0.5 - 5*z2 + 4.0
        0.3     3.0
        0.5     2.0
        0.8     0.5
        0.9     0.0))))

(deftest update-weight-gradient-matches-fd
  (testing "update weight gradient matches FD for chain model"
    (let [{:keys [trace]} (p/generate chain-model []
                                      (cm/choicemap :z1 (mx/scalar 0.5)
                                                    :z2 (mx/scalar 1.0)
                                                    :y (mx/scalar 1.0)))
          score-fn (fn [z2-arr]
                     (:weight (p/update chain-model trace
                                        (cm/choicemap :z2 z2-arr))))]
      (doseq [z2-new [0.2 0.5 0.8 1.2]]
        (let [ad (fd/analytical-gradient score-fn z2-new)
              fd-val (fd/fd-gradient score-fn z2-new)]
          (is (fd/gradient-close? ad fd-val)
              (str "update weight FD match at z2=" z2-new)))))))

(deftest update-weight-gradient-single-model
  (testing "update y observation in single-model"
    ;; Update y from old to new. d(update_weight)/d(y_new) = -(y_new - mu)
    (let [{:keys [trace]} (p/generate single-model [(mx/scalar 3.0)]
                                      (cm/choicemap :mu (mx/scalar 2.0)
                                                    :y (mx/scalar 3.0)))
          score-fn (fn [y-arr]
                     (:weight (p/update single-model trace
                                        (cm/choicemap :y y-arr))))
          ad-at-4 (fd/analytical-gradient score-fn 4.0)
          fd-at-4 (fd/fd-gradient score-fn 4.0)]
      ;; d(update_weight)/d(y_new) = -(y_new - mu) = -(4 - 2) = -2
      (is (th/close? -2.0 ad-at-4 1e-3)
          "d(update_weight)/d(y_new) = -(y_new - mu)")
      (is (fd/gradient-close? ad-at-4 fd-at-4)
          "update weight autodiff matches FD"))))

;; ---------------------------------------------------------------------------
;; E. Chain model: gradient propagation through dependent sites
;; ---------------------------------------------------------------------------

(deftest chain-gradient-propagates-z1-to-y
  (testing "z1 gradient includes indirect effect through z2 and y"
    ;; d(score)/d(z1) = z2 - 2*z1: the z1 gradient depends on z2,
    ;; confirming gradient flows through the z1->z2 dependency.
    (let [{:keys [trace]} (p/generate chain-model []
                                      (cm/choicemap :z1 (mx/scalar 1.0)
                                                    :z2 (mx/scalar 0.5)
                                                    :y (mx/scalar 2.0)))
          {:keys [z1 z2 y]} (grad/choice-gradients chain-model trace [:z1 :z2 :y])]
      (is (th/close? -1.5 (mx/realize z1) 1e-4)
          "d/dz1 = z2 - 2*z1 = 0.5 - 2.0 = -1.5")
      (is (th/close? 6.5 (mx/realize z2) 1e-4)
          "d/dz2 = z1 - 5*z2 + 4*y = 1.0 - 2.5 + 8.0 = 6.5")
      (is (th/close? -6.0 (mx/realize y) 1e-4)
          "d/dy = -(y - z2)/0.25 = -(2.0 - 0.5)*4 = -6.0"))))

(deftest chain-gradient-zero-crossings
  (testing "gradients vanish at posterior conditionals"
    ;; z2 gradient is zero when z1 - 5*z2 + 4*y = 0
    ;; At z1=0.5, y=1.0: z2* = (0.5 + 4.0)/5 = 0.9
    (let [z2-star (/ (+ 0.5 4.0) 5.0)
          {:keys [trace]} (p/generate chain-model []
                                      (cm/choicemap :z1 (mx/scalar 0.5)
                                                    :z2 (mx/scalar z2-star)
                                                    :y (mx/scalar 1.0)))
          {:keys [z2]} (grad/choice-gradients chain-model trace [:z2])]
      (is (th/close? 0.0 (mx/realize z2) 1e-4)
          "z2 gradient vanishes at conditional mode"))))

(deftest chain-gradient-fd-all-sites
  (testing "all three chain sites match FD simultaneously"
    (let [z1-val 0.3 z2-val 0.7 y-val 1.5
          {:keys [trace]} (p/generate chain-model []
                                      (cm/choicemap :z1 (mx/scalar z1-val)
                                                    :z2 (mx/scalar z2-val)
                                                    :y (mx/scalar y-val)))
          grads (grad/choice-gradients chain-model trace [:z1 :z2 :y])
          base-cm (cm/choicemap :z1 (mx/scalar z1-val)
                                :z2 (mx/scalar z2-val)
                                :y (mx/scalar y-val))]
      (doseq [[addr point] [[:z1 z1-val] [:z2 z2-val] [:y y-val]]]
        (let [score-fn (fn [v-arr]
                         (:weight (p/generate chain-model []
                                              (cm/set-choice base-cm [addr] v-arr))))
              fd-val (fd/fd-gradient score-fn point)
              ad-val (mx/realize (get grads addr))]
          (is (fd/gradient-close? ad-val fd-val)
              (str "chain gradient FD match for " (name addr)
                   " (ad=" ad-val " fd=" fd-val ")")))))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
