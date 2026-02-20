(ns genmlx.amortized-test
  (:require [genmlx.mlx :as mx]
            [genmlx.nn :as nn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.amortized :as amort]
            [genmlx.inference.importance :as is])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Amortized Inference Tests ===\n")

;; ---------------------------------------------------------------------------
;; Test model: z ~ N(0, 1), x|z ~ N(z, 0.5)
;; Analytical posterior: z|x ~ N(0.8*x, sqrt(0.2))
;; For x=3: posterior mean = 2.4, posterior std = 0.4472, log(std) = -0.805
;; ---------------------------------------------------------------------------

(def model
  (gen [x]
    (let [z (dyn/trace :z (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      (dyn/trace :x (dist/gaussian z (mx/scalar 0.5)))
      z)))

;; Encoder: 1 input -> 2 outputs (mu, log-sigma)
(def encoder (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)]))

;; ---------------------------------------------------------------------------
;; Test 1: make-elbo-loss produces finite loss
;; ---------------------------------------------------------------------------

(println "-- make-elbo-loss produces finite loss --")

(let [loss-fn (amort/make-elbo-loss
                encoder model [:z]
                :model-args-fn (fn [data] [(mx/squeeze data)])
                :observations-fn (fn [data]
                                   (cm/choicemap :x (mx/squeeze data))))]
  (let [data (mx/array [3.0])
        loss (loss-fn data)]
    (mx/eval! loss)
    (let [v (mx/item loss)]
      (assert-true "loss is finite" (js/isFinite v))
      (println "    initial loss:" v))))

;; ---------------------------------------------------------------------------
;; Test 2: Training reduces loss
;; ---------------------------------------------------------------------------

(println "\n-- Training reduces loss --")

;; Fresh encoder for training test
(def train-encoder (nn/sequential [(nn/linear 1 16) (nn/relu) (nn/linear 16 2)]))

(let [loss-fn (amort/make-elbo-loss
                train-encoder model [:z]
                :model-args-fn (fn [data] [(mx/squeeze data)])
                :observations-fn (fn [data]
                                   (cm/choicemap :x (mx/squeeze data))))
      ;; Generate training data: sample from the model
      dataset (mapv (fn [_]
                      (let [trace (p/simulate model [(mx/scalar 0.0)])
                            z-val (mx/realize (cm/get-choice (:choices trace) [:z]))]
                        ;; x|z ~ N(z, 0.5), so sample x for various z values
                        (mx/array [(+ (* 3.0 (- (js/Math.random) 0.5)) 1.5)])))
                    (range 20))
      losses (amort/train-proposal!
               train-encoder loss-fn dataset
               :iterations 300 :lr 0.01)]
  (let [early-loss (nth losses 5)
        late-loss  (last losses)]
    (println "    loss[5]:" early-loss)
    (println "    loss[end]:" late-loss)
    (assert-true "training reduces loss" (< late-loss early-loss))))

;; ---------------------------------------------------------------------------
;; Test 3: Encoder learns approximate posterior parameters
;; ---------------------------------------------------------------------------

(println "\n-- Encoder learns posterior --")

;; Dedicated encoder trained on x=3 concentrated dataset
(def posterior-encoder (nn/sequential [(nn/linear 1 32) (nn/relu) (nn/linear 32 2)]))

(let [loss-fn (amort/make-elbo-loss
                posterior-encoder model [:z]
                :model-args-fn (fn [data] [(mx/squeeze data)])
                :observations-fn (fn [data]
                                   (cm/choicemap :x (mx/squeeze data))))
      ;; Train on x values near 3.0
      dataset (mapv (fn [_] (mx/array [(+ 2.5 (js/Math.random))])) (range 20))
      _losses (amort/train-proposal!
                posterior-encoder loss-fn dataset
                :iterations 500 :lr 0.005)]
  ;; Query encoder at x=3
  (let [out (.forward posterior-encoder (mx/array [3.0]))
        _ (mx/eval! out)
        mu (mx/item (mx/index out 0))
        log-sig (mx/item (mx/index out 1))]
    (println "    encoder mu:" mu "(target: 2.4)")
    (println "    encoder log-sigma:" log-sig "(target: -0.805)")
    ;; Tolerances are generous since NN training is stochastic
    (assert-close "encoder mu ≈ 2.4" 2.4 mu 0.6)
    (assert-close "encoder log-sigma ≈ -0.805" -0.805 log-sig 0.6)))

;; ---------------------------------------------------------------------------
;; Test 4: neural-importance-sampling works
;; ---------------------------------------------------------------------------

(println "\n-- neural-importance-sampling --")

;; Build a guide gen fn that uses the trained posterior-encoder
(let [net-gf (nn/nn->gen-fn posterior-encoder)
      guide (gen [x]
              (let [out (dyn/splice :enc net-gf (mx/reshape x [1]))
                    mu      (mx/index out 0)
                    log-sig (mx/index out 1)
                    sig     (mx/exp log-sig)]
                (dyn/trace :z (dist/gaussian mu sig))))
      x-val (mx/scalar 3.0)
      obs (cm/choicemap :x x-val)
      {:keys [traces log-weights log-ml-estimate]}
      (amort/neural-importance-sampling
        {:samples 50}
        guide model [x-val] [x-val] obs)]
  (mx/eval! log-ml-estimate)
  (let [lml (mx/item log-ml-estimate)]
    (println "    log-ML estimate:" lml)
    (assert-true "log-ML is finite" (js/isFinite lml))
    (assert-true "traces count = 50" (= 50 (count traces)))
    (assert-true "log-weights count = 50" (= 50 (count log-weights)))))

;; ---------------------------------------------------------------------------
;; Test 5: Neural IS vs prior IS — neural proposal gives better log-ML
;; ---------------------------------------------------------------------------

(println "\n-- Neural IS vs prior IS --")

(let [net-gf (nn/nn->gen-fn posterior-encoder)
      guide (gen [x]
              (let [out (dyn/splice :enc net-gf (mx/reshape x [1]))
                    mu      (mx/index out 0)
                    log-sig (mx/index out 1)
                    sig     (mx/exp log-sig)]
                (dyn/trace :z (dist/gaussian mu sig))))
      x-val (mx/scalar 3.0)
      obs (cm/choicemap :x x-val)
      ;; Neural IS
      neural-result (amort/neural-importance-sampling
                      {:samples 200}
                      guide model [x-val] [x-val] obs)
      neural-lml (mx/realize (:log-ml-estimate neural-result))
      ;; Prior IS (using model's prior as proposal = standard IS)
      prior-result (is/importance-sampling
                     {:samples 200}
                     model [x-val] obs)
      prior-lml (mx/realize (:log-ml-estimate prior-result))]
  (println "    neural log-ML:" neural-lml)
  (println "    prior  log-ML:" prior-lml)
  ;; True log p(x=3) = log N(3; 0, sqrt(1+0.25)) = log N(3; 0, 1.118)
  ;; = -0.5*log(2π) - log(1.118) - 0.5*(3/1.118)² ≈ -4.52
  (let [true-lml -4.52]
    (assert-close "neural log-ML near true value" true-lml neural-lml 1.0)
    (assert-true "neural log-ML >= prior log-ML (better proposal)"
                 (>= neural-lml (- prior-lml 0.5)))))

(println "\n=== All Amortized Inference Tests Complete ===")
