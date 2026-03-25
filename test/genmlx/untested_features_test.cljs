(ns genmlx.untested-features-test
  "Tests for features identified as untested or weakly tested:
   - Custom proposal MH
   - Involutive MCMC
   - Choice gradients (with direction checks)
   - Programmable VI objectives
   - Wake-sleep learning
   - Training loop"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.vi :as vi]
            [genmlx.gradients :as grad]
            [genmlx.learning :as learn])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest custom-proposal-mh
  (testing "symmetric proposal"
    (let [model (gen []
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (mx/eval! mu)
                    (let [mu-val (mx/item mu)]
                      (doseq [i (range 5)]
                        (trace (keyword (str "obs" i))
                                   (dist/gaussian mu-val 1)))
                      mu-val)))
          observations (reduce (fn [cm i]
                                  (cm/set-choice cm [(keyword (str "obs" i))]
                                                 (mx/scalar 5.0)))
                                cm/EMPTY (range 5))
          proposal (gen [current-choices]
                     (let [current-mu (mx/realize (cm/get-choice current-choices [:mu]))]
                       (trace :mu (dist/gaussian current-mu 0.5))))
          traces (mcmc/mh-custom
                   {:samples 200 :burn 100 :proposal-gf proposal}
                   model [] observations)
          mu-vals (mapv (fn [t]
                           (mx/realize (cm/get-choice (:choices t) [:mu])))
                         traces)
          mu-mean (/ (reduce + mu-vals) (count mu-vals))]
      (is (= 200 (count traces)) "custom MH: 200 samples")
      (is (h/close? 5.0 mu-mean 1.0) "custom MH: posterior mu near 5")
      (let [ar (:acceptance-rate (meta traces))]
        (is (some? ar) "custom MH: has acceptance rate")
        (is (> ar 0) "custom MH: acceptance rate > 0")))))

(deftest involutive-mcmc
  (testing "swap involution"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    (mx/eval! x)
                    (trace :y (dist/gaussian (mx/item x) 0.1))
                    (mx/item x)))
          observations (cm/choicemap :y (mx/scalar 2.0))
          proposal (gen [current-choices]
                     (trace :aux (dist/gaussian 0 0.5)))
          involution (fn [trace-cm aux-cm]
                       (let [x-val (mx/realize (cm/get-choice trace-cm [:x]))
                             aux-val (mx/realize (cm/get-choice aux-cm [:aux]))
                             new-x (+ x-val aux-val)
                             new-aux (- aux-val)]
                         [(cm/set-choice trace-cm [:x] (mx/scalar new-x))
                          (cm/set-choice aux-cm [:aux] (mx/scalar new-aux))]))
          traces (mcmc/involutive-mh
                   {:samples 200 :burn 100
                    :proposal-gf proposal :involution involution}
                   model [] observations)
          x-vals (mapv (fn [t]
                          (mx/realize (cm/get-choice (:choices t) [:x])))
                        traces)
          x-mean (/ (reduce + x-vals) (count x-vals))]
      (is (= 200 (count traces)) "involutive MH: 200 samples")
      (is (h/close? 2.0 x-mean 0.5) "involutive MH: posterior x near 2")
      (let [ar (:acceptance-rate (meta traces))]
        (is (> ar 0) "involutive MH: acceptance rate > 0")))))

(deftest choice-gradients-direction
  (testing "gradient direction for Gaussian"
    (let [model (gen [mu]
                  (trace :x (dist/gaussian mu 1)))
          constraints (cm/choicemap :x (mx/scalar 3.0))
          {:keys [trace]} (p/generate (dyn/auto-key model) [0] constraints)
          result (grad/choice-gradients model trace [:x])]
      (mx/eval! (:x result))
      (let [grad-val (mx/item (:x result))]
        (is (< grad-val 0) "gradient at x=3 is negative")
        (is (h/close? -3.0 grad-val 0.5) "gradient at x=3 ~ -3"))))

  (testing "score gradient direction check"
    (let [model (gen [mu]
                  (trace :obs (dist/gaussian mu 1)))
          result (grad/score-gradient model [0]
                   (cm/choicemap :obs (mx/scalar 5.0))
                   [:obs] (mx/array [5.0]))]
      (mx/eval! (:grad result) (:score result))
      (is (< (mx/item (:grad result)) 0) "score gradient is negative at obs=5")
      (is (h/close? -5.0 (mx/item (:grad result)) 0.5) "score gradient ~ -5")
      (is (js/isFinite (mx/item (:score result))) "score is finite"))))

(deftest programmable-vi-elbo
  (testing "ELBO learns posterior of simple model"
    (let [log-p (fn [z]
                  (let [z-scalar (mx/index z 0)]
                    (mx/add (dc/dist-log-prob (dist/gaussian 0 1) z-scalar)
                            (dc/dist-log-prob (dist/gaussian z-scalar 1) (mx/scalar 3.0)))))
          log-q (fn [z params]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)]
                    (dc/dist-log-prob (dist/gaussian mu sigma) (mx/index z 0))))
          sample-fn (fn [params key n]
                      (let [mu (mx/index params 0)
                            log-sigma (mx/index params 1)
                            sigma (mx/exp log-sigma)
                            eps (rng/normal (rng/ensure-key key) [n 1])]
                        (mx/add mu (mx/multiply sigma eps))))
          init-params (mx/array [0.0 0.0])
          result (vi/programmable-vi
                   {:iterations 200 :learning-rate 0.01 :n-samples 20
                    :objective :elbo}
                   log-p log-q sample-fn init-params)]
      (mx/eval! (:params result))
      (let [final-mu (mx/item (mx/index (:params result) 0))
            final-log-sigma (mx/item (mx/index (:params result) 1))
            final-sigma (js/Math.exp final-log-sigma)]
        (is (h/close? 1.5 final-mu 0.5) "prog VI ELBO: mu near 1.5")
        (is (< final-sigma 2.0) "prog VI ELBO: sigma reasonable")
        (let [losses (:loss-history result)
              first-loss (first losses)
              last-loss (last losses)]
          (is (< last-loss first-loss) "prog VI ELBO: loss decreased"))))))

(deftest programmable-vi-iwelbo
  (testing "IWELBO tighter bound"
    (let [log-p (fn [z]
                  (let [z-scalar (mx/index z 0)]
                    (mx/add (dc/dist-log-prob (dist/gaussian 0 1) z-scalar)
                            (dc/dist-log-prob (dist/gaussian z-scalar 1) (mx/scalar 3.0)))))
          log-q (fn [z params]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)]
                    (dc/dist-log-prob (dist/gaussian mu sigma) (mx/index z 0))))
          sample-fn (fn [params key n]
                      (let [mu (mx/index params 0)
                            log-sigma (mx/index params 1)
                            sigma (mx/exp log-sigma)
                            eps (rng/normal (rng/ensure-key key) [n 1])]
                        (mx/add mu (mx/multiply sigma eps))))
          init-params (mx/array [0.0 0.0])
          result (vi/programmable-vi
                   {:iterations 200 :learning-rate 0.01 :n-samples 20
                    :objective :iwelbo}
                   log-p log-q sample-fn init-params)]
      (mx/eval! (:params result))
      (let [final-mu (mx/item (mx/index (:params result) 0))
            losses (:loss-history result)]
        (is (> final-mu 0) "prog VI IWELBO: mu moved toward 1.5 (> 0)")
        (is (pos? (count losses)) "prog VI IWELBO: has loss history")))))

(deftest training-loop-sgd
  (testing "SGD: minimize x^2"
    (let [loss-grad-fn (fn [params _key]
                         (let [loss (mx/sum (mx/square params))
                               grad (mx/multiply (mx/scalar 2.0) params)]
                           {:loss loss :grad grad}))
          result (learn/train
                   {:iterations 100 :optimizer :sgd :lr 0.1}
                   loss-grad-fn (mx/array [5.0 3.0]))]
      (mx/eval! (:params result))
      (let [final (mx/->clj (:params result))]
        (is (every? #(< (js/Math.abs %) 0.1) final) "SGD: params near 0")
        (is (< (last (:loss-history result)) (first (:loss-history result))) "SGD: loss decreased")))))

(deftest training-loop-adam
  (testing "Adam: minimize (x-3)^2"
    (let [loss-grad-fn (fn [params _key]
                         (let [target (mx/array [3.0 -2.0])
                               diff (mx/subtract params target)
                               loss (mx/sum (mx/square diff))
                               grad (mx/multiply (mx/scalar 2.0) diff)]
                           {:loss loss :grad grad}))
          result (learn/train
                   {:iterations 200 :optimizer :adam :lr 0.1}
                   loss-grad-fn (mx/array [0.0 0.0]))]
      (mx/eval! (:params result))
      (let [final (mx/->clj (:params result))]
        (is (h/close? 3.0 (first final) 0.5) "Adam train: param 0 near 3")
        (is (h/close? -2.0 (second final) 0.5) "Adam train: param 1 near -2")))))

(deftest wake-sleep-learning
  (testing "guide learns to match model"
    (let [model (gen []
                  (let [z (trace :z (dist/gaussian 3 0.5))]
                    (mx/eval! z) (mx/item z)))
          guide (gen []
                  (let [z (trace :z (dist/gaussian 0 1))]
                    (mx/eval! z) (mx/item z)))
          init-params (mx/array [0.0])
          result (learn/wake-sleep
                   {:iterations 30 :wake-steps 1 :sleep-steps 1 :lr 0.05}
                   model guide [] cm/EMPTY [:z] init-params)]
      (mx/eval! (:params result))
      (let [final-mu (mx/item (:params result))]
        (is (>= final-mu 0) "wake-sleep: guide mu moved toward 3")
        (is (pos? (count (:wake-losses result))) "wake-sleep: has wake losses")
        (is (pos? (count (:sleep-losses result))) "wake-sleep: has sleep losses")))))

(deftest param-store-flatten-unflatten
  (testing "flatten/unflatten round-trip"
    (let [store (learn/make-param-store {:a (mx/scalar 1.0) :b (mx/scalar 2.0) :c (mx/scalar 3.0)})
          names [:a :b :c]
          flat (learn/params->array store names)
          _ (mx/eval! flat)
          unflat (learn/array->params flat names)]
      (is (h/close? 1.0 (mx/item (mx/index flat 0)) 1e-5) "flatten a")
      (is (h/close? 2.0 (mx/item (mx/index flat 1)) 1e-5) "flatten b")
      (is (h/close? 3.0 (mx/item (mx/index flat 2)) 1e-5) "flatten c")
      (mx/eval! (:a unflat))
      (is (h/close? 1.0 (mx/item (:a unflat)) 1e-5) "unflatten a"))))

(cljs.test/run-tests)
