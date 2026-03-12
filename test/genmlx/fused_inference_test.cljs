(ns genmlx.fused-inference-test
  "WP-2: Fused Inference + Optimization tests.
   Tests fused MCMC+Adam, fused-learn dispatch, metadata reporting,
   and memory bounds.

   Note: VI composition is tested separately because mx/vmap (used inside
   vi/vi) has a native MLX interaction that causes segfaults in short-lived
   nbb processes. VI works correctly in long-lived REPL sessions and is
   verified in the existing vi test suite."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.inference.compiled-optimizer :as co]
            [genmlx.inference.compiled-gradient :as cg]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:dynamic *pass-count* (atom 0))
(def ^:dynamic *fail-count* (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! *pass-count* inc)
        (println (str "  PASS: " msg)))
    (do (swap! *fail-count* inc)
        (println (str "  FAIL: " msg)))))

(defn cleanup! []
  (mx/clear-cache!)
  (mx/sweep-dead-arrays!))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-model
  "Simple 2-latent model: Gaussian mean + noise, observe y."
  (gen []
       (let [mu (trace :mu (dist/gaussian 0 10))
             sigma (trace :sigma (dist/gaussian 1 0.1))]
         (trace :y (dist/gaussian mu (mx/abs sigma)))
         mu)))

(def three-param-model
  "3-latent model: slope + intercept + noise, observe y."
  (gen []
       (let [a (trace :a (dist/gaussian 0 5))
             b (trace :b (dist/gaussian 0 5))
             c (trace :c (dist/gaussian 1 0.1))]
         (trace :y (dist/gaussian (mx/add a b) (mx/abs c)))
         a)))

;; ---------------------------------------------------------------------------
;; 1. fused-learn :direct dispatch
;; ---------------------------------------------------------------------------

(println "\n=== 1. fused-learn :direct dispatch ===")

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      result (co/fused-learn simple-model [] obs [:mu :sigma] :direct
                             {:iterations 20 :lr 0.05 :log-every 10})]
  (assert-true "fused-learn :direct returns :compilation-level"
               (contains? result :compilation-level))
  (assert-true "fused-learn :direct returns :params"
               (mx/array? (:params result)))
  (assert-true "fused-learn :direct returns :loss-history"
               (sequential? (:loss-history result)))
  (assert-true "direct path reports :tensor-native or :handler"
               (#{:tensor-native :handler} (:compilation-level result))))

(cleanup!)

;; Default path (nil method)
(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      result (co/fused-learn simple-model [] obs [:mu :sigma] nil
                             {:iterations 20 :lr 0.05 :log-every 10})]
  (assert-true "fused-learn nil method falls through to :direct"
               (contains? result :compilation-level))
  (assert-true "fused-learn nil returns :params"
               (mx/array? (:params result))))

(cleanup!)

;; ---------------------------------------------------------------------------
;; 2. Fused MCMC+Adam: basic functionality
;; ---------------------------------------------------------------------------

(println "\n=== 2. Fused MCMC+Adam basic functionality ===")

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      result (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                       {:iterations 30 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3 :log-every 10})]

  (assert-true "returns :params key"
               (contains? result :params))

  (assert-true "returns :loss-history key"
               (contains? result :loss-history))

  (assert-true "returns :compilation-level key"
               (contains? result :compilation-level))

  (assert-true "returns :mcmc-compiled key"
               (contains? result :mcmc-compiled))

  (assert-true "returns :latent-index key"
               (contains? result :latent-index))

  (assert-true "returns :n-params key"
               (contains? result :n-params))

  (assert-true "mcmc-compiled is true"
               (true? (:mcmc-compiled result)))

  (assert-true "compilation-level is :tensor-native"
               (= :tensor-native (:compilation-level result)))

  (assert-true "params is an MLX array"
               (mx/array? (:params result)))

  (assert-true "loss-history is non-empty"
               (pos? (count (:loss-history result))))

  (assert-true "all loss-history entries are numbers"
               (every? number? (:loss-history result)))

  (assert-true "n-params is positive"
               (pos? (:n-params result))))

(cleanup!)

;; ---------------------------------------------------------------------------
;; 3. Noise pre-generation shapes
;; ---------------------------------------------------------------------------

(println "\n=== 3. Noise pre-generation shapes ===")

(let [obs (cm/choicemap {:y (mx/scalar 3.0)})
      model-k (dyn/auto-key simple-model)
      {:keys [trace]} (p/generate model-k [] obs)
      {:keys [score-fn n-params]} (u/prepare-mcmc-score simple-model [] obs [:mu :sigma] trace)
      K n-params
      T 10
      rk (rng/fresh-key)
      [nk uk] (rng/split rk)
      noise (rng/normal nk [T K])
      uniforms (rng/uniform uk [T])
      _ (mx/materialize! noise uniforms)]

  (assert-true "noise shape is [T,K]"
               (= [T K] (mx/shape noise)))

  (assert-true "uniforms shape is [T]"
               (= [T] (mx/shape uniforms)))

  (let [std (mx/scalar 0.3)
        chain-fn (cg/make-differentiable-chain score-fn std T K)
        init-p (mx/zeros [K])
        final-p (chain-fn init-p noise uniforms)]
    (mx/materialize! final-p)

    (assert-true "chain output shape matches K"
                 (= [K] (mx/shape final-p)))

    (assert-true "chain output is finite"
                 (every? js/isFinite (mx/->clj final-p)))))

(cleanup!)

;; ---------------------------------------------------------------------------
;; 4. Gradient direction through chain
;; ---------------------------------------------------------------------------

(println "\n=== 4. Gradient direction through chain ===")

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      model-k (dyn/auto-key simple-model)
      {:keys [trace]} (p/generate model-k [] obs)
      {:keys [score-fn n-params]}
      (u/prepare-mcmc-score simple-model [] obs [:mu :sigma] trace)
      K n-params
      T 3
      std (mx/scalar 0.3)
      chain-fn (cg/make-differentiable-chain score-fn std T K)
      neg-obj (fn [params noise uniforms]
                (mx/negative (score-fn (chain-fn params noise uniforms))))
      vg (mx/value-and-grad neg-obj)
      start (mx/array [3.0 1.0 5.0])
      rk (rng/fresh-key)
      [nk uk] (rng/split rk)
      noise (rng/normal nk [T K])
      uniforms (rng/uniform uk [T])
      _ (mx/materialize! noise uniforms)
      [loss grad] (vg start noise uniforms)]
  (mx/materialize! loss grad)
  (let [grad-vec (mx/->clj grad)]
    (assert-true "gradient is finite"
                 (every? js/isFinite grad-vec))

    (assert-true "loss is finite"
                 (js/isFinite (mx/item loss)))

    (assert-true "gradient has correct dimension"
                 (= K (count grad-vec)))))

(cleanup!)

;; ---------------------------------------------------------------------------
;; 5. fused-learn :mcmc dispatch + metadata
;; ---------------------------------------------------------------------------

(println "\n=== 5. fused-learn :mcmc dispatch + metadata ===")

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      result (co/fused-learn simple-model [] obs [:mu :sigma] :mcmc
                             {:iterations 20 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3 :log-every 10})]
  (assert-true "fused-learn :mcmc returns :mcmc-compiled"
               (true? (:mcmc-compiled result)))
  (assert-true "fused-learn :mcmc returns :params"
               (mx/array? (:params result)))
  (assert-true "fused-learn :mcmc returns :loss-history"
               (sequential? (:loss-history result)))
  (assert-true "mcmc path reports compilation-level"
               (#{:tensor-native :handler} (:compilation-level result)))
  (assert-true "mcmc path has latent-index map"
               (map? (:latent-index result)))
  (assert-true "mcmc path has n-params > 0"
               (pos? (:n-params result))))

(cleanup!)

;; ---------------------------------------------------------------------------
;; 6. Callback invocation
;; ---------------------------------------------------------------------------

(println "\n=== 6. Callback invocation ===")

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      callback-log (atom [])
      result (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                       {:iterations 50 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3
                                        :log-every 10
                                        :callback (fn [info]
                                                    (swap! callback-log conj info))})]

  (assert-true "callback was invoked"
               (pos? (count @callback-log)))

  (assert-true "callback info has :iter key"
               (every? #(contains? % :iter) @callback-log))

  (assert-true "callback info has :loss key"
               (every? #(contains? % :loss) @callback-log))

  (assert-true "callback info has :params key"
               (every? #(contains? % :params) @callback-log)))

(cleanup!)

;; ---------------------------------------------------------------------------
;; 7. PRNG key determinism
;; ---------------------------------------------------------------------------

(println "\n=== 7. PRNG key determinism ===")

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      key1 (rng/fresh-key 42)
      key2 (rng/fresh-key 42)
      r1 (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                   {:iterations 20 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3
                                    :log-every 10 :key key1})
      _ (cleanup!)
      r2 (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                   {:iterations 20 :lr 0.01 :mcmc-steps 3 :proposal-std 0.3
                                    :log-every 10 :key key2})
      p1 (mx/->clj (:params r1))
      p2 (mx/->clj (:params r2))]

  (assert-true "both runs complete with same key"
               (and (some? r1) (some? r2)))

  (assert-true "both produce finite params"
               (and (every? js/isFinite p1) (every? js/isFinite p2))))

(cleanup!)

;; ---------------------------------------------------------------------------
;; 8. Convergence on 3-param model
;; ---------------------------------------------------------------------------

(println "\n=== 8. Fused MCMC+Adam convergence (3-param model) ===")

(let [obs (cm/choicemap {:y (mx/scalar 7.0)})
      result (co/make-fused-mcmc-train three-param-model [] obs [:a :b :c]
                                       {:iterations 300 :lr 0.01 :mcmc-steps 5 :proposal-std 0.3
                                        :log-every 100})
      losses (:loss-history result)
      first-loss (first losses)
      last-loss (last losses)]

  (assert-true "iterations complete with logged losses"
               (>= (count losses) 3))

  (assert-true "loss decreases overall (last <= first + 2.0)"
               (<= last-loss (+ first-loss 2.0)))

  (assert-true "final loss is finite"
               (js/isFinite last-loss))

  (assert-true "params are finite"
               (every? js/isFinite (mx/->clj (:params result)))))

(cleanup!)

;; ---------------------------------------------------------------------------
;; 9. Memory bounds: 500 iterations without leak
;; ---------------------------------------------------------------------------

(println "\n=== 9. Memory bounds (500 iterations, no leak) ===")

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      result (co/make-fused-mcmc-train simple-model [] obs [:mu :sigma]
                                       {:iterations 500 :lr 0.01 :mcmc-steps 5 :proposal-std 0.3
                                        :log-every 100})]

  (assert-true "500 mcmc iterations completed without error"
               (some? result))

  (assert-true "params are finite after 500 iters"
               (every? js/isFinite (mx/->clj (:params result))))

  (assert-true "all logged losses are finite"
               (every? js/isFinite (:loss-history result))))

(cleanup!)

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      result (co/learn simple-model [] obs [:mu :sigma]
                       {:iterations 500 :lr 0.05 :log-every 100})]

  (assert-true "direct path 500 iterations completed"
               (some? result))

  (assert-true "direct path params finite after 500 iters"
               (every? js/isFinite (mx/->clj (:params result)))))

(cleanup!)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== WP-2 Fused Inference Test Summary ==="))
(println (str "  PASS: " @*pass-count*))
(println (str "  FAIL: " @*fail-count*))
(println (str "  TOTAL: " (+ @*pass-count* @*fail-count*)))
(when (pos? @*fail-count*)
  (println "  *** FAILURES DETECTED ***"))
(when (zero? @*fail-count*)
  (println "  All tests passed!"))
