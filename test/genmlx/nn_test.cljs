(ns genmlx.nn-test
  "Neural network integration tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.nn :as nn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

(deftest layer-constructors
  (testing "Linear"
    (let [lin (nn/linear 3 1)]
      (is (map? lin) "Linear is a map")
      (is (fn? (:forward lin)) "Linear has :forward fn")
      (is (contains? (:params lin) :weight) "Linear has :weight param")
      (is (contains? (:params lin) :bias) "Linear has :bias param")
      (is (= [1 3] (mx/shape (:weight (:params lin)))) "weight shape [out, in]")
      (is (= [1] (mx/shape (:bias (:params lin)))) "bias shape [out]")))

  (testing "Linear no bias"
    (let [lin (nn/linear 3 1 :bias false)]
      (is (not (contains? (:params lin) :bias)) "no bias param")
      (is (= 1 (count (:params lin))) "only weight param")))

  (testing "Sequential"
    (let [seq-mod (nn/sequential [(nn/linear 4 8) (nn/relu) (nn/linear 8 1)])]
      (is (map? seq-mod) "Sequential is a map")
      (is (fn? (:forward seq-mod)) "Sequential has :forward fn")
      (is (= :sequential (:type seq-mod)) "Sequential type tag")
      ;; 4 param arrays: 0/weight, 0/bias, 2/weight, 2/bias
      (is (= 4 (count (:params seq-mod))) "Sequential has 4 params")))

  (testing "Activations"
    (let [r (nn/relu)
          g (nn/gelu)
          t (nn/tanh-act)
          s (nn/sigmoid-act)]
      (is (empty? (:params r)) "ReLU has no params")
      (is (empty? (:params g)) "GELU has no params")
      (is (empty? (:params t)) "Tanh has no params")
      (is (empty? (:params s)) "Sigmoid has no params")))

  (testing "LayerNorm"
    (let [ln (nn/layer-norm 8)]
      (is (contains? (:params ln) :gamma) "LayerNorm has gamma")
      (is (contains? (:params ln) :beta) "LayerNorm has beta")))

  (testing "Embedding"
    (let [emb (nn/embedding 100 32)]
      (is (contains? (:params emb) :weight) "Embedding has weight")
      (is (= [100 32] (mx/shape (:weight (:params emb)))) "Embedding weight shape"))))

(deftest forward-pass
  (testing "Linear forward"
    (let [lin (nn/linear 3 1)
          x (rng/normal (rng/fresh-key) [3])
          y ((:forward lin) x)]
      (mx/eval! y)
      (is (= [1] (mx/shape y)) "Linear(3,1) forward produces [1]-shaped output")))

  (testing "Sequential MLP forward"
    (let [mlp (nn/sequential [(nn/linear 4 8) (nn/relu) (nn/linear 8 2)])
          x (rng/normal (rng/fresh-key) [4])
          y ((:forward mlp) x)]
      (mx/eval! y)
      (is (= [2] (mx/shape y)) "Sequential MLP forward produces [2]-shaped output")))

  (testing "Batched input"
    (let [lin (nn/linear 3 1)
          x (rng/normal (rng/fresh-key) [5 3])
          y ((:forward lin) x)]
      (mx/eval! y)
      (is (= [5 1] (mx/shape y)) "Linear(3,1) batched [5,3] -> [5,1]"))))

(deftest neuralnet-gf-simulate
  (testing "NeuralNetGF simulate"
    (let [lin (nn/linear 3 1)
          gf (nn/nn->gen-fn lin)
          x (rng/normal (rng/fresh-key) [3])
          trace (p/simulate gf [x])]
      (mx/eval! (:retval trace) (:score trace))
      (is (= [1] (mx/shape (:retval trace))) "simulate retval shape = [1]")
      (is (h/close? 0.0 (mx/item (:score trace)) 1e-6) "simulate score = 0")
      (is (= (:choices trace) cm/EMPTY) "simulate choices are empty"))))

(deftest neuralnet-gf-generate
  (testing "NeuralNetGF generate"
    (let [lin (nn/linear 3 1)
          gf (nn/nn->gen-fn lin)
          x (rng/normal (rng/fresh-key) [3])
          {:keys [trace weight]} (p/generate gf [x] cm/EMPTY)]
      (mx/eval! (:retval trace) weight)
      (is (= [1] (mx/shape (:retval trace))) "generate retval shape = [1]")
      (is (h/close? 0.0 (mx/item weight) 1e-6) "generate weight = 0"))))

(deftest neuralnet-gf-assess-propose
  (testing "NeuralNetGF assess and propose"
    (let [lin (nn/linear 3 1)
          gf (nn/nn->gen-fn lin)
          x (rng/normal (rng/fresh-key) [3])]

      (let [{:keys [retval weight]} (p/assess gf [x] cm/EMPTY)]
        (mx/eval! retval weight)
        (is (= [1] (mx/shape retval)) "assess retval shape = [1]")
        (is (h/close? 0.0 (mx/item weight) 1e-6) "assess weight = 0"))

      (let [{:keys [choices weight retval]} (p/propose gf [x])]
        (mx/eval! retval weight)
        (is (= [1] (mx/shape retval)) "propose retval shape = [1]")
        (is (h/close? 0.0 (mx/item weight) 1e-6) "propose weight = 0")
        (is (= choices cm/EMPTY) "propose choices are empty")))))

(deftest neuralnet-gf-update-regenerate-project
  (testing "NeuralNetGF update/regenerate/project"
    (let [lin (nn/linear 3 1)
          gf (nn/nn->gen-fn lin)
          trace (p/simulate gf [(mx/ones [3])])]
      ;; update
      (let [{:keys [trace weight discard]} (p/update gf trace cm/EMPTY)]
        (mx/eval! weight)
        (is (h/close? 0.0 (mx/item weight) 1e-6) "update weight = 0")
        (is (= discard cm/EMPTY) "update discard is empty"))
      ;; regenerate
      (let [{:keys [trace weight]} (p/regenerate gf trace sel/all)]
        (mx/eval! weight)
        (is (h/close? 0.0 (mx/item weight) 1e-6) "regenerate weight = 0"))
      ;; project
      (let [w (p/project gf trace sel/all)]
        (mx/eval! w)
        (is (h/close? 0.0 (mx/item w) 1e-6) "project = 0")))))

(deftest has-argument-grads-test
  (testing "NeuralNetGF has-argument-grads"
    (let [gf (nn/nn->gen-fn (nn/linear 3 1))]
      (is (= [true] (p/has-argument-grads gf)) "NeuralNetGF has-argument-grads = [true]"))))

(deftest splice-into-model
  (testing "splice into model"
    (let [net (nn/linear 3 1)
          net-gf (nn/nn->gen-fn net)
          model (dyn/auto-key (gen [x]
                  (let [mu (splice :net net-gf x)]
                    (trace :y (dist/gaussian mu 1))
                    mu)))
          x (rng/normal (rng/fresh-key) [3])
          expected ((:forward net) x)
          _ (mx/eval! expected)
          obs (cm/choicemap :y expected)
          {:keys [trace weight]} (p/generate model [x] obs)]
      (mx/eval! (:retval trace) weight)
      (is (h/close? (mx/item expected) (mx/item (:retval trace)) 1e-5)
          "spliced model retval matches module forward")
      (is (h/close? -0.9189 (mx/item weight) 0.01)
          "generate weight = log gaussian(mu; mu, 1)"))))

(deftest gradient-flow
  (testing "value-and-grad computes gradients"
    (let [lin-ref (atom (nn/linear 3 1))
          vg (nn/value-and-grad lin-ref
                (fn [fwd x]
                  (let [y (fwd x)]
                    (mx/sum (mx/square y)))))
          x (rng/normal (rng/fresh-key) [3])
          [loss grads] (vg x)]
      (mx/eval! loss)
      (is (= [] (mx/shape loss)) "loss is a scalar")
      (is (>= (mx/item loss) 0.0) "loss is non-negative")
      (is (map? grads) "grads is a map")
      (let [w-grad (:weight grads)]
        (mx/eval! w-grad)
        (is (some? w-grad) "weight grad exists")
        (is (= [1 3] (mx/shape w-grad)) "weight grad has correct shape")
        (is (> (mx/item (mx/sum (mx/abs w-grad))) 1e-10) "weight grad is non-zero")))))

(deftest training-convergence
  (testing "fit y = 2x + 1"
    (let [lin-ref (atom (nn/linear 1 1))
          opt (nn/optimizer :adam 0.01)
          vg (nn/value-and-grad lin-ref
                (fn [fwd x]
                  (let [y-pred (fwd x)
                        y-true (mx/add (mx/multiply (mx/scalar 2.0) x)
                                       (mx/scalar 1.0))]
                    (mx/mean (mx/square (mx/subtract y-pred y-true))))))]

      (dotimes [_ 500]
        (let [x (mx/subtract (mx/multiply (rng/uniform (rng/fresh-key) [10 1])
                                          (mx/scalar 2.0))
                             (mx/scalar 1.0))]
          (nn/training-step! lin-ref opt vg x)))

      (let [w (:weight (:params @lin-ref))
            b (:bias (:params @lin-ref))]
        (mx/eval! w b)
        (let [w-val (mx/item (mx/squeeze w))
              b-val (mx/item (mx/squeeze b))]
          (is (h/close? 2.0 w-val 0.2) "weight converges to ~2.0")
          (is (h/close? 1.0 b-val 0.2) "bias converges to ~1.0"))))))

(cljs.test/run-tests)
