;; @tier bench
(ns cost-per-particle
  "genmlx-2gu2 (under milestone genmlx-819v): scalar-vs-batched cost-per-particle
   sweep. MEASUREMENT ONLY — no assertions gate host speed; classification
   happens in the bean/docs write-up, not here.

   The diagnostic: for each path, sweep N and print wall-clock ms/particle.
     flat in N   => host-bound (per-particle host GFI calls dominate)
     falls in N  => amortizing (GPU-bound or close)

   Anchors (genmlx-im8n, must roughly reproduce or the harness is wrong):
     membrane micro-latency ~1 ms/eval; vgenerate N=3000 ~21 ms.

   Model: the normal-normal from inference_smc_test (1 latent + 5 obs sites),
   analytical path STRIPPED on both sides so scalar and batched both measure
   the handler path, not L3 conjugate elimination."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vect]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.smcp3 :as smcp3])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Model + data (mirrors inference_smc_test)
;; ---------------------------------------------------------------------------

(def model
  (dyn/auto-key
    (gen [ys]
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (doseq [[i _] (map-indexed vector ys)]
          (trace (keyword (str "y" i))
                 (dist/gaussian mu 1)))
        mu))))

;; Handler path on BOTH sides: raw p/generate would otherwise hit the L3
;; analytical dispatcher (gaussian-gaussian is conjugate) and measure
;; elimination, not per-particle host dispatch.
(def hmodel (dyn/strip-analytical-path model))

(def ys [2.8 3.1 2.9 3.2 3.0])
(def obs
  (apply cm/choicemap
         (mapcat (fn [[i y]] [(keyword (str "y" i)) (mx/scalar y)])
                 (map-indexed vector ys))))
(def obs-seq
  (mapv (fn [[i y]] (cm/choicemap (keyword (str "y" i)) (mx/scalar y)))
        (map-indexed vector ys)))

;; ---------------------------------------------------------------------------
;; Harness
;; ---------------------------------------------------------------------------

(defn time-ms [f]
  (let [t0 (js/performance.now)] (f) (- (js/performance.now) t0)))

(defn median [xs] (nth (vec (sort xs)) (quot (count xs) 2)))

(defn bench
  "Warm up once, then median of `reps` timed runs. Sweeps dead arrays between
   runs so one point's garbage doesn't bill the next."
  [reps f]
  (f)
  (mx/sweep-dead-arrays!)
  (median (mapv (fn [_] (let [t (time-ms f)] (mx/sweep-dead-arrays!) t))
                (range reps))))

(defn row [label n units total-ms]
  (println (str "  " label
                "  N=" (.padEnd (str n) 5)
                "  total " (.padStart (.toFixed total-ms 1) 9) " ms"
                "  |  " (.toFixed (/ total-ms units) 4) " ms/"
                (if (= units n) "particle" "unit"))))

;; reps: cheap points get 3, expensive scalar points get 1 (still warmed).
(defn reps-for [scalar? n] (if (and scalar? (>= n 1000)) 1 3))

;; ---------------------------------------------------------------------------
;; Anchor A: membrane micro-latency (~1 ms/eval expected)
;; ---------------------------------------------------------------------------

(println "\n== Anchor A: membrane micro-latency (tiny graph + eval!) ==")
(let [x (mx/scalar 1.5)
      k 200
      ms (bench 3 (fn [] (dotimes [_ k] (mx/eval! (mx/add x 0.5)))))]
  (row "eval-loop            " k k ms))

;; ---------------------------------------------------------------------------
;; Anchor B + Pair 1: p/generate xN (scalar) vs vgenerate N (batched)
;; ---------------------------------------------------------------------------

(println "\n== Pair 1: generate — scalar loop vs vgenerate ==")
(doseq [n [1 10 100 1000 3000]]
  (let [ms (bench (reps-for true n)
                  (fn []
                    (let [ks (rng/split-n (rng/fresh-key 7) n)]
                      (doseq [k ks]
                        (let [{:keys [weight]} (p/generate (dyn/with-key hmodel k) [ys] obs)]
                          (mx/item weight)))
                      (when (>= n 100) (mx/sweep-dead-arrays!)))))]
    (row "scalar  p/generate   " n n ms)))
(doseq [n [1 10 100 1000 3000]]
  (let [ms (bench 3
                  (fn []
                    (let [vt (dyn/vgenerate hmodel [ys] obs n (rng/fresh-key 7))]
                      (mx/item (vect/vtrace-log-ml-estimate vt)))))]
    (row "batched vgenerate    " n n ms)))

;; GPU-vs-host split for the batched path at N=3000: graph build (host)
;; vs the forcing eval (GPU). Corroborates "falls with N" = amortizing.
(println "\n-- vgenerate N=3000 build/eval split --")
(let [n 3000
      vt (atom nil)
      build-ms (bench 3 (fn [] (reset! vt (dyn/vgenerate hmodel [ys] obs n (rng/fresh-key 7)))))
      eval-ms  (bench 3 (fn [] (mx/item (vect/vtrace-log-ml-estimate @vt))))]
  (row "graph build (host)   " n n build-ms)
  (row "force eval  (GPU)    " n n eval-ms))

;; ---------------------------------------------------------------------------
;; Pair 2: importance-sampling vs vectorized-importance-sampling
;; (scalar path deep-materializes each trace — that IS its real cost)
;; ---------------------------------------------------------------------------

(println "\n== Pair 2: importance sampling — scalar vs vectorized ==")
(doseq [n [1 10 100 1000]]
  (let [ms (bench (reps-for true n)
                  (fn [] (mx/item (:log-ml-estimate
                                   (is/importance-sampling
                                    {:samples n :key (rng/fresh-key 11)}
                                    hmodel [ys] obs)))))]
    (row "scalar  importance   " n n ms)))
(doseq [n [1 10 100 1000 3000]]
  (let [ms (bench 3
                  (fn [] (mx/item (:log-ml-estimate
                                   (is/vectorized-importance-sampling
                                    {:samples n :key (rng/fresh-key 11)}
                                    hmodel [ys] obs)))))]
    (row "batched importance   " n n ms)))

;; ---------------------------------------------------------------------------
;; Pair 3: MH — C sequential scalar chains vs vmh N broadcast chains
;; (units = chain-steps: C*S resp. N*S; scalar expected flat, vmh falling)
;; ---------------------------------------------------------------------------

(println "\n== Pair 3: MH (10 steps) — scalar chains vs vmh broadcast ==")
(let [s 10]
  (doseq [c [1 5 20]]
    (let [ms (bench (reps-for true (* c 100))  ;; scalar chains are expensive; 1 rep at c>=10
                    (fn []
                      (dotimes [i c]
                        (let [trs (mcmc/mh {:samples s :key (rng/fresh-key (+ 20 i))}
                                           hmodel [ys] obs)]
                          (mx/item (:score (peek trs)))))))]
      (row "scalar  mh chains    " c (* c s) ms)))
  (doseq [n [1 10 100 1000 3000]]
    (let [ms (bench 3
                    (fn []
                      (let [vt  (dyn/vgenerate hmodel [ys] obs n (rng/fresh-key 23))
                            vt' (mcmc/vmh hmodel vt {:iters s :addresses [:mu]
                                                     :key (rng/fresh-key 24)})]
                        (mx/eval! (:score vt')))))]
      (row "batched vmh          " n (* n s) ms))))

;; ---------------------------------------------------------------------------
;; Pair 4: SMC (5 steps) — scalar smc vs vsmc
;; (units = particle-steps = N*5)
;; ---------------------------------------------------------------------------

(println "\n== Pair 4: SMC (5 steps) — scalar vs vectorized ==")
(doseq [n [1 10 100 500]]
  (let [ms (bench (reps-for true n)
                  (fn [] (mx/item (:log-ml-estimate
                                   (smc/smc {:particles n :key (rng/fresh-key 31)}
                                            hmodel [ys] obs-seq)))))]
    (row "scalar  smc          " n (* n 5) ms)))
(doseq [n [1 10 100 1000 3000]]
  (let [ms (bench 3
                  (fn [] (mx/item (:log-ml-estimate
                                   (smc/vsmc {:particles n :key (rng/fresh-key 31)}
                                             hmodel [ys] obs-seq)))))]
    (row "batched vsmc         " n (* n 5) ms)))

;; ---------------------------------------------------------------------------
;; SMCP3 (3 steps, standard-SMC kernels): NO batched counterpart exists.
;; The diagnostic here is curve shape alone — flat ms/particle-step = host-bound.
;; ---------------------------------------------------------------------------

(println "\n== SMCP3 (3 steps, no kernels) — scalar only, curve shape is the verdict ==")
(let [obs3 (subvec obs-seq 0 3)]
  (doseq [n [1 10 50 150]]
    (let [ms (bench (reps-for true n)
                    (fn [] (mx/item (:log-ml-estimate
                                     (smcp3/smcp3 {:particles n :key (rng/fresh-key 41)}
                                                  hmodel [ys] obs3)))))]
      (row "scalar  smcp3        " n (* n 3) ms))))

(println "\n== cost_per_particle done ==")
