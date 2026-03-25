(ns genmlx.amortized-test
  "Amortized inference tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.nn :as nn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec]
            [genmlx.inference.amortized :as amort]
            [genmlx.inference.importance :as is])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test model: z ~ N(0, 1), x|z ~ N(z, 0.5)
;; Analytical posterior: z|x ~ N(0.8*x, sqrt(0.2))
;; For x=3: posterior mean = 2.4, posterior std = 0.4472, log(std) = -0.805
;; ---------------------------------------------------------------------------

(def model
  (dyn/auto-key (gen [x]
    (let [z (trace :z (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      (trace :x (dist/gaussian z (mx/scalar 0.5)))
      z))))

;; Encoder: 1 input -> 2 outputs (mu, log-sigma)
(def encoder (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)]))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest make-elbo-loss-test
  (testing "make-elbo-loss produces finite loss"
    (let [loss-fn (amort/make-elbo-loss
                    encoder model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data]
                                       (cm/choicemap :x (mx/squeeze data))))]
      (let [data (mx/array [3.0])
            loss (loss-fn data)]
        (mx/eval! loss)
        (is (js/isFinite (mx/item loss)) "loss is finite")))))

(deftest training-reduces-loss-test
  (testing "training reduces loss"
    (let [train-encoder (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)])
          loss-fn (amort/make-elbo-loss
                    train-encoder model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data]
                                       (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_]
                          (let [trace (p/simulate model [(mx/scalar 0.0)])
                                z-val (mx/realize (cm/get-choice (:choices trace) [:z]))]
                            (mx/array [(+ (* 3.0 (- (js/Math.random) 0.5)) 1.5)])))
                        (range 20))
          losses (amort/train-proposal
                   train-encoder loss-fn dataset
                   :iterations 300 :lr 0.01)]
      (let [early-loss (nth losses 5)
            late-loss  (last losses)]
        (is (< late-loss early-loss) "training reduces loss")))))

(deftest encoder-learns-posterior-test
  (testing "encoder learns posterior"
    (let [posterior-encoder (nn/sequential [(nn/linear 1 32) (nn/relu) (nn/linear 32 2)])
          loss-fn (amort/make-elbo-loss
                    posterior-encoder model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data]
                                       (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 20))
          _losses (amort/train-proposal
                    posterior-encoder loss-fn dataset
                    :iterations 500 :lr 0.005)]
      (let [out (.forward posterior-encoder (mx/array [3.0]))
            _ (mx/eval! out)
            mu (mx/item (mx/index out 0))
            log-sig (mx/item (mx/index out 1))]
        (is (h/close? 2.4 mu 0.6) "encoder mu ~ 2.4")
        (is (h/close? -0.805 log-sig 0.6) "encoder log-sigma ~ -0.805")))))

(deftest neural-importance-sampling-test
  (testing "neural-importance-sampling"
    ;; Train a dedicated encoder
    (let [pe (nn/sequential [(nn/linear 1 32) (nn/relu) (nn/linear 32 2)])
          loss-fn (amort/make-elbo-loss
                    pe model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data]
                                       (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 20))
          _ (amort/train-proposal pe loss-fn dataset :iterations 500 :lr 0.005)
          net-gf (nn/nn->gen-fn pe)
          guide (gen [x]
                  (let [out (splice :enc net-gf (mx/reshape x [1]))
                        mu      (mx/index out 0)
                        log-sig (mx/index out 1)
                        sig     (mx/exp log-sig)]
                    (trace :z (dist/gaussian mu sig))))
          x-val (mx/scalar 3.0)
          obs (cm/choicemap :x x-val)
          {:keys [traces log-weights log-ml-estimate]}
          (amort/neural-importance-sampling
            {:samples 50}
            guide model [x-val] [x-val] obs)]
      (mx/eval! log-ml-estimate)
      (let [lml (mx/item log-ml-estimate)]
        (is (js/isFinite lml) "log-ML is finite")
        (is (= 50 (count traces)) "traces count = 50")
        (is (= 50 (count log-weights)) "log-weights count = 50")))))

(deftest neural-is-vs-prior-is-test
  (testing "neural IS vs prior IS"
    (let [pe (nn/sequential [(nn/linear 1 32) (nn/relu) (nn/linear 32 2)])
          loss-fn (amort/make-elbo-loss
                    pe model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data]
                                       (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 20))
          _ (amort/train-proposal pe loss-fn dataset :iterations 500 :lr 0.005)
          net-gf (nn/nn->gen-fn pe)
          guide (gen [x]
                  (let [out (splice :enc net-gf (mx/reshape x [1]))
                        mu      (mx/index out 0)
                        log-sig (mx/index out 1)
                        sig     (mx/exp log-sig)]
                    (trace :z (dist/gaussian mu sig))))
          x-val (mx/scalar 3.0)
          obs (cm/choicemap :x x-val)
          neural-result (amort/neural-importance-sampling
                          {:samples 200}
                          guide model [x-val] [x-val] obs)
          neural-lml (mx/realize (:log-ml-estimate neural-result))
          prior-result (is/importance-sampling
                         {:samples 200}
                         model [x-val] obs)
          prior-lml (mx/realize (:log-ml-estimate prior-result))
          true-lml -4.52]
      (is (h/close? true-lml neural-lml 1.0) "neural log-ML near true value")
      (is (>= neural-lml (- prior-lml 0.5)) "neural log-ML >= prior log-ML (better proposal)"))))

(deftest minibatch-backward-compat-test
  (testing "minibatch: batch-size=1 backward compat"
    (let [enc1 (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)])
          loss-fn (amort/make-elbo-loss
                    enc1 model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data] (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 10))
          losses (amort/train-proposal
                   enc1 loss-fn dataset
                   :iterations 50 :lr 0.01 :batch-size 1)]
      (is (= 50 (count losses)) "batch-size=1 returns 50 losses")
      (is (every? js/isFinite losses) "all losses are finite"))))

(deftest minibatch-training-test
  (testing "minibatch: batch training reduces loss"
    (let [enc-b (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)])
          loss-fn (amort/make-elbo-loss
                    enc-b model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data] (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 20))
          losses (amort/train-proposal
                   enc-b loss-fn dataset
                   :iterations 100 :lr 0.01 :batch-size 5)]
      (let [early-loss (nth losses 5)
            late-loss  (last losses)]
        (is (= 100 (count losses)) "batch training returns 100 losses")
        (is (< late-loss early-loss) "batch training reduces loss")))))

(deftest full-batch-mode-test
  (testing "minibatch: full-batch mode"
    (let [enc-fb (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)])
          loss-fn (amort/make-elbo-loss
                    enc-fb model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data] (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 10))
          losses (amort/train-proposal
                   enc-fb loss-fn dataset
                   :iterations 50 :lr 0.01 :batch-size 10)]
      (is (= 50 (count losses)) "full-batch returns 50 losses")
      (is (every? js/isFinite losses) "all losses are finite")
      (is (< (last losses) (nth losses 3)) "full-batch training reduces loss"))))

(deftest posterior-family-gaussian-test
  (testing "posterior family: explicit Gaussian matches default"
    (let [enc-g1 (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)])
          enc-g2 (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)])
          loss-fn-default (amort/make-elbo-loss
                            enc-g1 model [:z]
                            :model-args-fn (fn [data] [(mx/squeeze data)])
                            :observations-fn (fn [data] (cm/choicemap :x (mx/squeeze data))))
          loss-fn-explicit (amort/make-elbo-loss
                             enc-g2 model [:z]
                             :model-args-fn (fn [data] [(mx/squeeze data)])
                             :observations-fn (fn [data] (cm/choicemap :x (mx/squeeze data)))
                             :posterior-family amort/gaussian-posterior)
          data (mx/array [3.0])
          l1 (loss-fn-default data)
          l2 (loss-fn-explicit data)]
      (mx/eval! l1)
      (mx/eval! l2)
      (is (js/isFinite (mx/item l1)) "default loss is finite")
      (is (js/isFinite (mx/item l2)) "explicit Gaussian loss is finite"))))

(deftest posterior-family-log-normal-test
  (testing "posterior family: log-normal for positive latents"
    (let [positive-model
          (dyn/auto-key (gen [x]
            (let [z (trace :z (dist/gamma-dist (mx/scalar 2.0) (mx/scalar 1.0)))]
              (trace :x (dist/gaussian z (mx/scalar 0.5)))
              z)))
          enc-ln (nn/sequential [(nn/linear 1 32) (nn/relu) (nn/linear 32 2)])
          loss-fn (amort/make-elbo-loss
                    enc-ln positive-model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data] (cm/choicemap :x (mx/squeeze data)))
                    :posterior-family amort/log-normal-posterior)
          dataset (mapv (fn [_] (mx/array [(+ 1.0 (* 2.0 (js/Math.random)))])) (range 20))
          losses (amort/train-proposal
                   enc-ln loss-fn dataset
                   :iterations 300 :lr 0.005)]
      (let [early-loss (nth losses 5)
            late-loss  (last losses)]
        (is (< late-loss early-loss) "log-normal training reduces loss")
        (is (every? js/isFinite losses) "all losses are finite")))))

(deftest vectorized-nis-test
  (testing "vectorized NIS: basic functionality"
    (let [pe (nn/sequential [(nn/linear 1 32) (nn/relu) (nn/linear 32 2)])
          loss-fn (amort/make-elbo-loss
                    pe model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data]
                                       (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 20))
          _ (amort/train-proposal pe loss-fn dataset :iterations 500 :lr 0.005)
          net-gf (nn/nn->gen-fn pe)
          guide (gen [x]
                  (let [out (splice :enc net-gf (mx/reshape x [1]))
                        mu      (mx/index out 0)
                        log-sig (mx/index out 1)
                        sig     (mx/exp log-sig)]
                    (trace :z (dist/gaussian mu sig))))
          x-val (mx/scalar 3.0)
          obs (cm/choicemap :x x-val)
          {:keys [vtrace log-weights log-ml-estimate]}
          (amort/vectorized-neural-importance-sampling
            {:samples 50}
            guide model [x-val] [x-val] obs)]
      (mx/eval! log-ml-estimate)
      (let [lml (mx/item log-ml-estimate)
            w-shape (mx/shape log-weights)]
        (is (js/isFinite lml) "log-ML is finite")
        (is (instance? genmlx.vectorized/VectorizedTrace vtrace)
            "vtrace is VectorizedTrace")
        (is (= [50] (vec w-shape)) "log-weights shape is [50]")))))

(deftest vectorized-nis-log-ml-test
  (testing "vectorized NIS: log-ML close to sequential"
    (let [pe (nn/sequential [(nn/linear 1 32) (nn/relu) (nn/linear 32 2)])
          loss-fn (amort/make-elbo-loss
                    pe model [:z]
                    :model-args-fn (fn [data] [(mx/squeeze data)])
                    :observations-fn (fn [data]
                                       (cm/choicemap :x (mx/squeeze data))))
          dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 20))
          _ (amort/train-proposal pe loss-fn dataset :iterations 500 :lr 0.005)
          net-gf (nn/nn->gen-fn pe)
          guide (gen [x]
                  (let [out (splice :enc net-gf (mx/reshape x [1]))
                        mu      (mx/index out 0)
                        log-sig (mx/index out 1)
                        sig     (mx/exp log-sig)]
                    (trace :z (dist/gaussian mu sig))))
          x-val (mx/scalar 3.0)
          obs (cm/choicemap :x x-val)
          v-result (amort/vectorized-neural-importance-sampling
                     {:samples 200}
                     guide model [x-val] [x-val] obs)
          v-lml (mx/realize (:log-ml-estimate v-result))
          s-result (amort/neural-importance-sampling
                     {:samples 200}
                     guide model [x-val] [x-val] obs)
          s-lml (mx/realize (:log-ml-estimate s-result))
          true-lml -4.52]
      (is (h/close? true-lml v-lml 1.5) "vectorized log-ML near true value")
      (is (h/close? true-lml s-lml 1.5) "sequential log-ML near true value")
      (is (and (js/isFinite v-lml) (js/isFinite s-lml)) "both estimates are finite"))))

(cljs.test/run-tests)
