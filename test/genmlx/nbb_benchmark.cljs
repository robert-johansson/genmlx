(ns genmlx.nbb-benchmark
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as random]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn now-ms []
  (js/Date.now))

(defn bench [label reps f]
  (let [_ (f) ;; warmup
        start (now-ms)]
    (dotimes [_ reps] (f))
    (let [elapsed (- (now-ms) start)
          per-op (/ elapsed reps)]
      (println (str "  " label ": " (.toFixed per-op 3) " ms/op (" reps " reps)")))))

;; ---- Benchmarks ----

(println "\n=== nbb GenMLX Benchmark ===\n")

;; 1. MLX array creation + eval
(println "--- Array creation ---")
(bench "scalar creation + eval" 1000
  (fn [] (let [a (mx/scalar 42.0)] (mx/eval! a) a)))

(bench "array [1000] creation + eval" 1000
  (fn [] (let [a (mx/ones [1000])] (mx/eval! a) a)))

(bench "random normal [100 100]" 500
  (fn [] (let [k (random/fresh-key 0)
               a (random/normal k [100 100])]
           (mx/eval! a) a)))

;; 2. MLX ops
(println "\n--- MLX ops ---")
(let [a (mx/ones [100 100])
      b (mx/ones [100 100])]
  (mx/eval! a b)
  (bench "matmul 100x100" 500
    (fn [] (let [c (mx/matmul a b)] (mx/eval! c) c)))
  (bench "4 elementwise ops + eval" 1000
    (fn [] (let [c (-> a (mx/add b) (mx/multiply b) (mx/subtract b) (mx/divide b))]
             (mx/eval! c) c))))

(let [a (doto (mx/random-normal [512 512]) mx/eval!)
      b (doto (mx/random-normal [512 512]) mx/eval!)]
  (bench "matmul 512x512" 100
    (fn [] (let [c (mx/matmul a b)] (mx/eval! c) c))))

;; 3. Distribution sampling + scoring
(println "\n--- Distributions ---")
(let [k (random/fresh-key 42)]
  (bench "gaussian sample + log-prob" 1000
    (fn [] (let [d (dist/gaussian 0 1)
                 v (dist/sample d k)
                 lp (dist/log-prob d v)]
             (mx/eval! lp) lp))))

(bench "10 gaussian samples + scores" 200
  (fn [] (let [k (random/fresh-key 99)]
           (dotimes [i 10]
             (let [d (dist/gaussian (mx/scalar (float i)) (mx/scalar 1.0))
                   v (dist/sample d k)
                   lp (dist/log-prob d v)]
               (mx/eval! lp))))))

;; 4. Gen model simulate
(println "\n--- Model execution ---")
(def simple-model
  (gen [x]
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))
          sigma (dyn/trace :sigma (dist/exponential 1))]
      (dyn/trace :y (dist/gaussian mu sigma))
      mu)))

(bench "simulate (3-site model)" 200
  (fn [] (let [k (random/fresh-key (rand-int 10000))
               t (p/simulate simple-model [(mx/scalar 1.0)] {:key k})]
           (mx/eval! (:score t)) t)))

;; 5. Gen model generate (with constraints)
(println "\n--- Generate (constrained) ---")
(bench "generate (3-site, 1 constrained)" 200
  (fn [] (let [k (random/fresh-key (rand-int 10000))
               obs (cm/choicemap {:y (mx/scalar 2.0)})
               result (p/generate simple-model [(mx/scalar 1.0)] obs {:key k})]
           (mx/eval! (:weight result)) result)))

;; 6. Larger model
(def line-model
  (gen [xs]
    (let [slope (dyn/trace :slope (dist/gaussian 0 10))
          intercept (dyn/trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (float x)))
                                          intercept) 1)))
      slope)))

(let [xs (vec (range 5))]
  (bench "simulate line model (7 sites)" 100
    (fn [] (let [k (random/fresh-key (rand-int 10000))
                 t (p/simulate line-model [xs] {:key k})]
             (mx/eval! (:score t)) t)))

  (let [obs (cm/choicemap (into {} (map (fn [j] [(keyword (str "y" j)) (mx/scalar (float j))]) (range 5))))]
    (bench "generate line model (7 sites, 5 obs)" 100
      (fn [] (let [k (random/fresh-key (rand-int 10000))
                   result (p/generate line-model [xs] obs {:key k})]
               (mx/eval! (:weight result)) result)))))

;; 7. Importance sampling + MH
(println "\n--- Inference ---")
(let [xs (vec (range 5))
      obs (cm/choicemap (into {} (map (fn [j] [(keyword (str "y" j)) (mx/scalar (float j))]) (range 5))))]
  (bench "importance sampling (10 particles)" 20
    (fn [] (is/importance-sampling {:samples 10} line-model [xs] obs)))
  (bench "MH (10 steps)" 20
    (fn [] (mcmc/mh {:samples 10 :burn 0 :thin 1} line-model [xs] obs))))

;; 9. Grad transform
(println "\n--- Transforms ---")
(bench "grad(sum(x*x))" 500
  (fn [] (let [f (fn [x] (mx/sum (mx/multiply x x)))
               grad-f (mx/grad f)
               x (mx/array #js [1 2 3 4 5])
               g (grad-f x)]
           (mx/eval! g) g)))

(bench "compile(grad(sum(x*x)))" 500
  (let [f (fn [x] (mx/sum (mx/multiply x x)))
        grad-f (mx/grad f)
        compiled (mx/compile-fn grad-f)]
    (fn [] (let [x (mx/array #js [1 2 3 4 5])
                 g (compiled x)]
             (mx/eval! g) g))))

(println "\n=== Done ===")
