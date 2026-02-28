(ns genmlx.optimization-benchmark
  "Step 1/Step 9 benchmark for TODO_OPTIMIZATION.md.
   Measures all operations we care about across 3 model sizes.
   Run: bun run --bun nbb test/genmlx/optimization_benchmark.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.importance :as is]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Timing
;; ---------------------------------------------------------------------------

(defn bench
  "Run f with warmup, then measure `runs` executions. Returns median ms."
  [f {:keys [warmup runs] :or {warmup 2 runs 5}}]
  (dotimes [_ warmup] (f))
  (let [times (mapv (fn [_]
                      (let [start (js/Date.now)
                            _ (f)
                            end (js/Date.now)]
                        (- end start)))
                    (range runs))
        sorted (sort times)]
    (nth sorted (quot runs 2))))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; 2-site: 2 latents, 0 observations (simplest possible)
(def model-2
  (gen []
    (let [a (dyn/trace :a (dist/gaussian 0 10))
          b (dyn/trace :b (dist/gaussian a 1))]
      b)))

(def obs-2 (cm/choicemap :b (mx/scalar 3.0)))

;; 7-site: 2 latents, 5 observations (linear regression)
(def model-7
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def xs-7 [1.0 2.0 3.0 4.0 5.0])
(def obs-7
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.1 3.9 6.2 7.8 10.1])))

;; 20-site: 2 latents, 18 observations
(def model-20
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def xs-20 (mapv #(* 0.5 %) (range 18)))
(def obs-20
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector
            (mapv #(+ (* 2.0 %) 1.0 (* 0.3 (- (js/Math.random) 0.5))) xs-20))))

;; ---------------------------------------------------------------------------
;; Run benchmarks
;; ---------------------------------------------------------------------------

(println "\n=== GenMLX Optimization Benchmark (Step 1 / Step 9 Final) ===")
(println (str "  Runtime: " (if (exists? js/Bun) "Bun" "Node.js")))
(println (str "  Date: " (.toISOString (js/Date.))))
(println)

;; ---------------------------------------------------------------------------
;; 1. simulate
;; ---------------------------------------------------------------------------

(println "-- simulate --")

(let [s2 (bench #(let [t (p/simulate model-2 [])]
                   (mx/eval! (:score t))) {})
      s7 (bench #(let [t (p/simulate model-7 [xs-7])]
                   (mx/eval! (:score t))) {})
      s20 (bench #(let [t (p/simulate model-20 [xs-20])]
                    (mx/eval! (:score t))) {})]
  (println (str "  2-site:  " s2 "ms"))
  (println (str "  7-site:  " s7 "ms"))
  (println (str "  20-site: " s20 "ms")))

;; ---------------------------------------------------------------------------
;; 2. generate
;; ---------------------------------------------------------------------------

(println "\n-- generate --")

(let [g2 (bench #(let [{:keys [trace weight]} (p/generate model-2 [] obs-2)]
                   (mx/eval! (:score trace) weight)) {})
      g7 (bench #(let [{:keys [trace weight]} (p/generate model-7 [xs-7] obs-7)]
                   (mx/eval! (:score trace) weight)) {})
      g20 (bench #(let [{:keys [trace weight]} (p/generate model-20 [xs-20] obs-20)]
                    (mx/eval! (:score trace) weight)) {})]
  (println (str "  2-site:  " g2 "ms"))
  (println (str "  7-site:  " g7 "ms"))
  (println (str "  20-site: " g20 "ms")))

;; ---------------------------------------------------------------------------
;; 3. compiled-mh (200 steps)
;; ---------------------------------------------------------------------------

(println "\n-- compiled-mh (200 steps) --")

(let [m2 (bench #(mcmc/compiled-mh
                   {:samples 200 :burn 0 :addresses [:a]
                    :proposal-std 0.5}
                   model-2 [] obs-2) {:warmup 1 :runs 3})
      m7 (bench #(mcmc/compiled-mh
                   {:samples 200 :burn 0 :addresses [:slope :intercept]
                    :proposal-std 0.5}
                   model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      m20 (bench #(mcmc/compiled-mh
                    {:samples 200 :burn 0 :addresses [:slope :intercept]
                     :proposal-std 0.5}
                    model-20 [xs-20] obs-20) {:warmup 1 :runs 3})]
  (println (str "  2-site:  " m2 "ms  (" (.toFixed (/ m2 200.0) 2) " ms/step)"))
  (println (str "  7-site:  " m7 "ms  (" (.toFixed (/ m7 200.0) 2) " ms/step)"))
  (println (str "  20-site: " m20 "ms  (" (.toFixed (/ m20 200.0) 2) " ms/step)")))

;; ---------------------------------------------------------------------------
;; 4. mala (50 steps)
;; ---------------------------------------------------------------------------

(println "\n-- mala (50 steps) --")

(let [m2 (bench #(mcmc/mala
                   {:samples 50 :burn 0 :step-size 0.01
                    :addresses [:a]}
                   model-2 [] obs-2) {:warmup 1 :runs 3})
      m7 (bench #(mcmc/mala
                   {:samples 50 :burn 0 :step-size 0.01
                    :addresses [:slope :intercept]}
                   model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      m20 (bench #(mcmc/mala
                    {:samples 50 :burn 0 :step-size 0.01
                     :addresses [:slope :intercept]}
                    model-20 [xs-20] obs-20) {:warmup 1 :runs 3})]
  (println (str "  2-site:  " m2 "ms  (" (.toFixed (/ m2 50.0) 2) " ms/step)"))
  (println (str "  7-site:  " m7 "ms  (" (.toFixed (/ m7 50.0) 2) " ms/step)"))
  (println (str "  20-site: " m20 "ms  (" (.toFixed (/ m20 50.0) 2) " ms/step)")))

;; ---------------------------------------------------------------------------
;; 5. hmc (20 steps, L=10)
;; ---------------------------------------------------------------------------

(println "\n-- hmc (20 steps, L=10) --")

(let [h2 (bench #(mcmc/hmc
                   {:samples 20 :burn 0 :step-size 0.005
                    :leapfrog-steps 10 :addresses [:a]}
                   model-2 [] obs-2) {:warmup 1 :runs 3})
      h7 (bench #(mcmc/hmc
                   {:samples 20 :burn 0 :step-size 0.005
                    :leapfrog-steps 10 :addresses [:slope :intercept]}
                   model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      h20 (bench #(mcmc/hmc
                    {:samples 20 :burn 0 :step-size 0.005
                     :leapfrog-steps 10 :addresses [:slope :intercept]}
                    model-20 [xs-20] obs-20) {:warmup 1 :runs 3})]
  (println (str "  2-site:  " h2 "ms  (" (.toFixed (/ h2 20.0) 2) " ms/step)"))
  (println (str "  7-site:  " h7 "ms  (" (.toFixed (/ h7 20.0) 2) " ms/step)"))
  (println (str "  20-site: " h20 "ms  (" (.toFixed (/ h20 20.0) 2) " ms/step)")))

;; ---------------------------------------------------------------------------
;; 6. loop-compiled-mh (200 steps) — compiled vs eager
;; ---------------------------------------------------------------------------

(println "\n-- loop-compiled-mh (200 steps) --")

(let [m7-compiled (bench #(mcmc/compiled-mh
                            {:samples 200 :burn 0 :addresses [:slope :intercept]
                             :proposal-std 0.5 :compile? true}
                            model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      m7-eager (bench #(mcmc/compiled-mh
                         {:samples 200 :burn 0 :addresses [:slope :intercept]
                          :proposal-std 0.5 :compile? false}
                         model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      ratio (if (pos? m7-eager) (/ m7-eager m7-compiled) ##Inf)]
  (println (str "  7-site compiled: " m7-compiled "ms"))
  (println (str "  7-site eager:    " m7-eager "ms"))
  (println (str "  speedup: " (.toFixed ratio 2) "x")))

;; ---------------------------------------------------------------------------
;; 7. loop-compiled-mala (50 steps) — compiled vs eager
;; ---------------------------------------------------------------------------

(println "\n-- loop-compiled-mala (50 steps) --")

(let [m7-compiled (bench #(mcmc/mala
                            {:samples 50 :burn 0 :step-size 0.01
                             :addresses [:slope :intercept] :compile? true}
                            model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      m7-eager (bench #(mcmc/mala
                         {:samples 50 :burn 0 :step-size 0.01
                          :addresses [:slope :intercept] :compile? false}
                         model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      ratio (if (pos? m7-eager) (/ m7-eager m7-compiled) ##Inf)]
  (println (str "  7-site compiled: " m7-compiled "ms"))
  (println (str "  7-site eager:    " m7-eager "ms"))
  (println (str "  speedup: " (.toFixed ratio 2) "x")))

;; ---------------------------------------------------------------------------
;; 8. vectorized-mala N=10, N=50 (50 steps)
;; ---------------------------------------------------------------------------

(println "\n-- vectorized-mala (50 steps) --")

(let [scalar (bench #(mcmc/mala
                       {:samples 50 :burn 0 :step-size 0.01
                        :addresses [:slope :intercept]}
                       model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      v10 (bench #(mcmc/vectorized-mala
                    {:samples 50 :burn 0 :step-size 0.01 :n-chains 10
                     :addresses [:slope :intercept]}
                    model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      v50 (bench #(mcmc/vectorized-mala
                    {:samples 50 :burn 0 :step-size 0.01 :n-chains 50
                     :addresses [:slope :intercept]}
                    model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      ;; Effective speedup: vec gives N chains in one run vs N sequential runs
      eff-10 (if (pos? v10) (/ (* 10 scalar) v10) ##Inf)
      eff-50 (if (pos? v50) (/ (* 50 scalar) v50) ##Inf)]
  (println (str "  7-site scalar:       " scalar "ms (1 chain)"))
  (println (str "  7-site vec N=10:     " v10 "ms (10 chains)"))
  (println (str "  7-site vec N=50:     " v50 "ms (50 chains)"))
  (println (str "  effective speedup N=10: " (.toFixed eff-10 1) "x"))
  (println (str "  effective speedup N=50: " (.toFixed eff-50 1) "x")))

;; ---------------------------------------------------------------------------
;; 9. vectorized-hmc N=10 (20 steps, L=10)
;; ---------------------------------------------------------------------------

(println "\n-- vectorized-hmc (20 steps, L=10) --")

(let [scalar (bench #(mcmc/hmc
                       {:samples 20 :burn 0 :step-size 0.005
                        :leapfrog-steps 10 :addresses [:slope :intercept]}
                       model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      v10 (bench #(mcmc/vectorized-hmc
                    {:samples 20 :burn 0 :step-size 0.005
                     :leapfrog-steps 10 :n-chains 10
                     :addresses [:slope :intercept]}
                    model-7 [xs-7] obs-7) {:warmup 1 :runs 3})
      eff-10 (if (pos? v10) (/ (* 10 scalar) v10) ##Inf)]
  (println (str "  7-site scalar:     " scalar "ms (1 chain)"))
  (println (str "  7-site vec N=10:   " v10 "ms (10 chains)"))
  (println (str "  effective speedup N=10: " (.toFixed eff-10 1) "x")))

;; ---------------------------------------------------------------------------
;; 10. Score function call cost (isolated)
;; ---------------------------------------------------------------------------

(println "\n-- score-fn call cost (200 calls) --")

(let [score-fn-7 (u/make-score-fn model-7 [xs-7] obs-7 [:slope :intercept])
      compiled-7 (mx/compile-fn score-fn-7)
      params (mx/array [2.0 0.5])
      raw (bench #(dotimes [_ 200]
                    (let [s (score-fn-7 params)] (mx/eval! s)))
                 {:warmup 1 :runs 3})
      comp (bench #(dotimes [_ 200]
                     (let [s (compiled-7 params)] (mx/eval! s)))
                  {:warmup 1 :runs 3})]
  (println (str "  7-site raw score-fn:      " raw "ms (" (.toFixed (/ raw 200.0) 3) " ms/call)"))
  (println (str "  7-site compiled score-fn: " comp "ms (" (.toFixed (/ comp 200.0) 3) " ms/call)"))
  (println (str "  compile speedup: " (.toFixed (/ raw comp) 1) "x")))

;; ---------------------------------------------------------------------------
;; 11. vectorized IS N=100 (5-site model)
;; ---------------------------------------------------------------------------

(println "\n-- vectorized IS (N=100, 5-site model) --")

(let [is-model (gen []
                 (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                   (dyn/trace :obs1 (dist/gaussian mu 1))
                   (dyn/trace :obs2 (dist/gaussian mu 1))
                   (dyn/trace :obs3 (dist/gaussian mu 1))
                   mu))
      is-obs (cm/choicemap :obs1 (mx/scalar 3.0) :obs2 (mx/scalar 3.1) :obs3 (mx/scalar 2.9))

      seq-ms (bench #(let [r (is/importance-sampling {:samples 100} is-model [] is-obs)]
                       (mx/eval! (:log-ml-estimate r))) {:warmup 1 :runs 3})
      vec-ms (bench #(let [r (is/vectorized-importance-sampling {:samples 100} is-model [] is-obs)]
                       (mx/eval! (:log-ml-estimate r))) {:warmup 1 :runs 3})
      speedup (if (pos? vec-ms) (/ seq-ms vec-ms) ##Inf)]
  (println (str "  Sequential IS (100):   " seq-ms "ms"))
  (println (str "  Vectorized IS (100):   " vec-ms "ms"))
  (println (str "  Speedup: " (.toFixed speedup 1) "x")))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n=== Benchmark complete ===")
