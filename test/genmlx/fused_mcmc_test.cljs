(ns genmlx.fused-mcmc-test
  "Tests for fused MCMC: pre-generated noise, fused burn-in, fused collection,
   and fused burn+collect (M2-M4)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- assert-true [desc pred]
  (if pred
    (do (vswap! pass-count inc)
        (println "  PASS:" desc))
    (do (vswap! fail-count inc)
        (println "  FAIL:" desc))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (vswap! pass-count inc)
          (println "  PASS:" desc "(diff=" (.toFixed diff 4) ")"))
      (do (vswap! fail-count inc)
          (println "  FAIL:" desc "expected" expected "got" actual "diff" diff)))))

(defn- assert-shape [desc expected-shape array]
  (assert-true (str desc " shape=" expected-shape)
               (= expected-shape (mx/shape array))))

;; ---------------------------------------------------------------------------
;; Access private fns
;; ---------------------------------------------------------------------------

(def ^:private pgcn @(resolve 'genmlx.inference.mcmc/pre-generate-chain-noise))
(def ^:private write-row @(resolve 'genmlx.inference.mcmc/write-sample-row))
(def ^:private mfbi @(resolve 'genmlx.inference.mcmc/make-fused-burn-in))
(def ^:private mfc @(resolve 'genmlx.inference.mcmc/make-fused-collection))
(def ^:private mfbc @(resolve 'genmlx.inference.mcmc/make-fused-burn-and-collect))
(def ^:private mfmbc @(resolve 'genmlx.inference.mcmc/make-fused-mala-burn-and-collect))
(def ^:private mfhbc @(resolve 'genmlx.inference.mcmc/make-fused-hmc-burn-and-collect))

;; Standard normal score: -x^2/2
(def ^:private std-normal-score
  (fn [p] (mx/multiply (mx/scalar -0.5) (mx/sum (mx/multiply p p)))))

;; ---------------------------------------------------------------------------
;; 1. pre-generate-chain-noise
;; ---------------------------------------------------------------------------

(println "\n-- pre-generate-chain-noise --")

(let [{:keys [noise uniforms]} (pgcn (rng/fresh-key) 100 3)]
  (assert-shape "noise" [100 3] noise)
  (assert-shape "uniforms" [100] uniforms)
  (let [u-min (mx/item (mx/amin uniforms))
        u-max (mx/item (mx/amax uniforms))]
    (assert-true "uniforms in [0,1]" (and (>= u-min 0) (<= u-max 1)))))

;; ---------------------------------------------------------------------------
;; 2. write-sample-row
;; ---------------------------------------------------------------------------

(println "\n-- write-sample-row --")

(let [samples (mx/zeros [5 3])
      params (mx/array [1.0 2.0 3.0])
      idx (mx/astype (mx/array [2]) mx/int32)
      result (write-row samples params idx 5 3)]
  (assert-shape "write result" [5 3] result)
  (let [r (mx/->clj result)]
    (assert-true "row 0 unchanged" (= [0 0 0] (nth r 0)))
    (assert-true "row 2 written" (= [1 2 3] (nth r 2)))
    (assert-true "row 4 unchanged" (= [0 0 0] (nth r 4)))))

(let [samples (mx/zeros [3 2])
      s1 (write-row samples (mx/array [1.0 2.0]) (mx/astype (mx/array [0]) mx/int32) 3 2)
      s2 (write-row s1 (mx/array [3.0 4.0]) (mx/astype (mx/array [1]) mx/int32) 3 2)
      s3 (write-row s2 (mx/array [5.0 6.0]) (mx/astype (mx/array [2]) mx/int32) 3 2)]
  (let [r (mx/->clj s3)]
    (assert-true "sequential writes" (= [[1 2] [3 4] [5 6]] r))))

;; ---------------------------------------------------------------------------
;; 3. make-fused-burn-in
;; ---------------------------------------------------------------------------

(println "\n-- make-fused-burn-in --")

(let [n-burn 200
      n-params 2
      std (mx/scalar 0.5)
      burn-fn (mfbi n-burn std-normal-score std n-params)
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) n-burn n-params)
      result (burn-fn (mx/array [10.0 -10.0]) noise uniforms)]
  (mx/materialize! result)
  (assert-shape "burn-in result" [2] result)
  (let [final (mx/->clj result)
        dist-sq (reduce + (map #(* % %) final))]
    (assert-true "moved toward origin" (< dist-sq 200))))

;; ---------------------------------------------------------------------------
;; 4. make-fused-collection: thin=1
;; ---------------------------------------------------------------------------

(println "\n-- make-fused-collection thin=1 --")

(let [n-samples 100
      thin 1
      n-params 2
      std (mx/scalar 0.5)
      collect-fn (mfc n-samples thin std-normal-score std n-params)
      total-steps (* thin n-samples)
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
      result (collect-fn (mx/array [1.0 -1.0]) noise uniforms)]
  (mx/materialize! result)
  (assert-shape "collection result" [100 2] result)
  ;; Verify shape is correct and last row differs from first (chain moved)
  (let [samples-js (mx/->clj result)]
    (assert-true "first != last row" (not= (first samples-js) (last samples-js)))))

;; ---------------------------------------------------------------------------
;; 5. make-fused-collection: thin=3
;; ---------------------------------------------------------------------------

(println "\n-- make-fused-collection thin=3 --")

(let [n-samples 50
      thin 3
      n-params 2
      std (mx/scalar 0.5)
      collect-fn (mfc n-samples thin std-normal-score std n-params)
      total-steps (* thin n-samples)
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
      result (collect-fn (mx/array [1.0 -1.0]) noise uniforms)]
  (mx/materialize! result)
  (assert-shape "thin=3 collection" [50 2] result)
  (let [samples-js (mx/->clj result)]
    (assert-true "first != last (thin=3)" (not= (first samples-js) (last samples-js)))))

;; ---------------------------------------------------------------------------
;; 6. make-fused-burn-and-collect: correctness
;; ---------------------------------------------------------------------------

(println "\n-- fused burn+collect correctness --")

(let [n-burn 200
      n-samples 500
      thin 1
      n-params 1
      std (mx/scalar 0.5)
      chain-fn (mfbc n-burn n-samples thin std-normal-score std n-params)
      total-steps (+ n-burn (* thin n-samples))
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
      result (chain-fn (mx/array [5.0]) noise uniforms)]
  (mx/materialize! (aget result 0) (aget result 1))
  (assert-shape "final params" [1] (aget result 0))
  (assert-shape "samples" [500 1] (aget result 1))
  (let [samples-js (mx/->clj (aget result 1))
        vals (mapv first samples-js)
        mean (/ (reduce + vals) (count vals))
        variance (/ (reduce + (map #(* (- % mean) (- % mean)) vals)) (count vals))
        std-dev (js/Math.sqrt variance)]
    (assert-close "posterior mean ≈ 0" 0.0 mean 0.3)
    (assert-close "posterior std ≈ 1" 1.0 std-dev 0.3)))

;; ---------------------------------------------------------------------------
;; 7. Fused burn+collect with thin=2
;; ---------------------------------------------------------------------------

(println "\n-- fused burn+collect thin=2 --")

(let [n-burn 100
      n-samples 200
      thin 2
      n-params 2
      std (mx/scalar 0.5)
      chain-fn (mfbc n-burn n-samples thin std-normal-score std n-params)
      total-steps (+ n-burn (* thin n-samples))
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
      result (chain-fn (mx/array [3.0 -3.0]) noise uniforms)]
  (mx/materialize! (aget result 0) (aget result 1))
  (assert-shape "thin=2 samples" [200 2] (aget result 1))
  (let [samples-js (mx/->clj (aget result 1))
        non-zero (count (filter (fn [row] (not= [0 0] row)) samples-js))]
    (assert-true "all rows populated (thin=2)" (= 200 non-zero))))

;; ---------------------------------------------------------------------------
;; 8. Statistical validation: 2D standard normal
;; ---------------------------------------------------------------------------

(println "\n-- statistical validation: 2D N(0,I) --")

(let [chain-fn (mfbc 500 2000 1 std-normal-score (mx/scalar 0.5) 2)
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) 2500 2)
      result (chain-fn (mx/array [1.0 -1.0]) noise uniforms)
      _ (mx/materialize! (aget result 0) (aget result 1))
      samples-js (mx/->clj (aget result 1))
      x1 (mapv first samples-js)
      x2 (mapv second samples-js)
      mean1 (/ (reduce + x1) (count x1))
      mean2 (/ (reduce + x2) (count x2))
      var1 (/ (reduce + (map #(* (- % mean1) (- % mean1)) x1)) (count x1))
      var2 (/ (reduce + (map #(* (- % mean2) (- % mean2)) x2)) (count x2))]
  (assert-close "dim1 mean ≈ 0" 0.0 mean1 0.25)
  (assert-close "dim2 mean ≈ 0" 0.0 mean2 0.25)
  (assert-close "dim1 var ≈ 1" 1.0 var1 0.4)
  (assert-close "dim2 var ≈ 1" 1.0 var2 0.4))

;; ---------------------------------------------------------------------------
;; 9. Performance: fused execution (cached) is fast
;; ---------------------------------------------------------------------------

(println "\n-- performance --")

(let [n-params 2
      std (mx/scalar 0.5)
      chain-fn (mfbc 200 500 1 std-normal-score std n-params)
      ;; Warmup
      _ (let [{:keys [noise uniforms]} (pgcn (rng/fresh-key) 700 n-params)
              r (chain-fn (mx/zeros [n-params]) noise uniforms)]
          (mx/materialize! (aget r 0) (aget r 1)))
      ;; Time 5 cached executions
      t0 (.now js/Date)
      _ (dotimes [_ 5]
          (let [{:keys [noise uniforms]} (pgcn (rng/fresh-key) 700 n-params)
                r (chain-fn (mx/zeros [n-params]) noise uniforms)]
            (mx/materialize! (aget r 0) (aget r 1))))
      t1 (.now js/Date)
      fused-ms (/ (- t1 t0) 5.0)]
  (println "  fused avg:" fused-ms "ms per chain (200 burn + 500 samples)")
  (assert-true "fused < 200ms per chain" (< fused-ms 200)))

;; ---------------------------------------------------------------------------
;; 10. Model integration: linreg with fused collection
;; ---------------------------------------------------------------------------

(println "\n-- model integration: linreg --")

(def linreg-model
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [j (range (count xs))]
        (let [x (nth xs j)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x)) intercept) 1))))
      slope)))

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      model (dyn/auto-key linreg-model)
      {:keys [trace]} (p/generate model [xs] obs)
      {:keys [score-fn init-params n-params]}
      (u/prepare-mcmc-score model [xs] obs [:slope :intercept] trace)
      std (mx/scalar 0.3)
      ;; Fused burn+collect
      chain-fn (mfbc 1000 2000 1 score-fn std n-params)
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) 3000 n-params)
      result (chain-fn init-params noise uniforms)
      _ (mx/materialize! (aget result 0) (aget result 1))
      samples-js (mx/->clj (aget result 1))
      slopes (mapv first samples-js)
      intercepts (mapv second samples-js)
      mean-slope (/ (reduce + slopes) (count slopes))
      mean-int (/ (reduce + intercepts) (count intercepts))]
  (assert-shape "linreg samples" [2000 2] (aget result 1))
  (assert-close "posterior slope ≈ 2" 2.0 mean-slope 0.5)
  (assert-close "posterior intercept ≈ 1" 1.0 mean-int 1.5))

;; ---------------------------------------------------------------------------
;; 11. fused-mh public API
;; ---------------------------------------------------------------------------

(println "\n-- fused-mh public API --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      ;; First call (compiles)
      r1 (mcmc/fused-mh
           {:samples 500 :burn 300 :thin 1
            :addresses [:slope :intercept]
            :proposal-std 0.3 :key (rng/fresh-key)}
           linreg-model [xs] obs)]
  (assert-shape "fused-mh samples" [500 2] (:samples r1))
  (assert-shape "fused-mh final-params" [2] (:final-params r1))
  (assert-true "chain-fn returned" (some? (:chain-fn r1)))
  ;; Second call with cached chain-fn
  (let [r2 (mcmc/fused-mh
             {:samples 500 :burn 300 :thin 1
              :addresses [:slope :intercept]
              :proposal-std 0.3 :key (rng/fresh-key)
              :chain-fn (:chain-fn r1)}
             linreg-model [xs] obs)
        samples-js (mx/->clj (:samples r2))
        mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
    (assert-shape "cached fused-mh samples" [500 2] (:samples r2))
    (assert-close "cached posterior slope ≈ 2" 2.0 mean-slope 0.5)))

;; ---------------------------------------------------------------------------
;; 12. fused-mh with thin > 1
;; ---------------------------------------------------------------------------

(println "\n-- fused-mh thin=2 --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      result (mcmc/fused-mh
               {:samples 200 :burn 200 :thin 2
                :addresses [:slope :intercept]
                :proposal-std 0.3 :key (rng/fresh-key)}
               linreg-model [xs] obs)]
  (assert-shape "fused-mh thin=2" [200 2] (:samples result)))

;; ---------------------------------------------------------------------------
;; 13. fused-vectorized-mh
;; ---------------------------------------------------------------------------

(println "\n-- fused-vectorized-mh --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      result (mcmc/fused-vectorized-mh
               {:samples 200 :burn 200 :n-chains 4
                :addresses [:slope :intercept]
                :proposal-std 0.3 :key (rng/fresh-key)
                :device :cpu}
               linreg-model [xs] obs)]
  (assert-shape "vectorized samples [S,N,D]" [200 4 2] (:samples result))
  (assert-shape "vectorized final-params [N,D]" [4 2] (:final-params result))
  (assert-true "vectorized chain-fn returned" (some? (:chain-fn result)))
  ;; Statistical check: pool chains, compute mean slope
  (let [samples-js (mx/->clj (:samples result))
        all-slopes (for [s samples-js, chain s] (first chain))
        mean-slope (/ (reduce + all-slopes) (count all-slopes))]
    (assert-close "vectorized posterior slope ≈ 2" 2.0 mean-slope 0.5)))

;; ---------------------------------------------------------------------------
;; 14. fused-vectorized-mh cached reuse
;; ---------------------------------------------------------------------------

(println "\n-- fused-vectorized-mh cached --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      r1 (mcmc/fused-vectorized-mh
           {:samples 200 :burn 100 :n-chains 4
            :addresses [:slope :intercept]
            :proposal-std 0.3 :key (rng/fresh-key) :device :cpu}
           linreg-model [xs] obs)
      ;; Cached call
      t0 (.now js/Date)
      r2 (mcmc/fused-vectorized-mh
           {:samples 200 :burn 100 :n-chains 4
            :addresses [:slope :intercept]
            :proposal-std 0.3 :key (rng/fresh-key)
            :chain-fn (:chain-fn r1) :device :cpu}
           linreg-model [xs] obs)
      t1 (.now js/Date)]
  (assert-shape "cached vectorized samples" [200 4 2] (:samples r2))
  (println "  cached vectorized:" (- t1 t0) "ms")
  (assert-true "cached vectorized < 500ms" (< (- t1 t0) 500)))

;; ---------------------------------------------------------------------------
;; 15. fused-vectorized-mh with thin
;; ---------------------------------------------------------------------------

(println "\n-- fused-vectorized-mh thin=2 --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      result (mcmc/fused-vectorized-mh
               {:samples 100 :burn 100 :thin 2 :n-chains 4
                :addresses [:slope :intercept]
                :proposal-std 0.3 :key (rng/fresh-key) :device :cpu}
               linreg-model [xs] obs)]
  (assert-shape "vectorized thin=2" [100 4 2] (:samples result)))

;; ===========================================================================
;; MALA / HMC helpers
;; ===========================================================================

(def ^:private val-grad-normal (mx/value-and-grad std-normal-score))

;; neg-U(q) = -score(q) = 0.5 * sum(q^2) — potential energy for N(0,I)
(def ^:private neg-U-normal (fn [q] (mx/negative (std-normal-score q))))
(def ^:private grad-neg-U-normal (mx/grad neg-U-normal))

;; ---------------------------------------------------------------------------
;; 16. Fused MALA: shape correctness
;; ---------------------------------------------------------------------------

(println "\n-- fused MALA shape correctness --")

(let [n-burn 100 n-samples 200 thin 1 n-params 2
      step-size 0.1
      eps (mx/scalar step-size)
      half-eps2 (mx/scalar (* 0.5 step-size step-size))
      two-eps-sq (mx/scalar (* 2.0 step-size step-size))
      chain-fn (mfmbc n-burn n-samples thin val-grad-normal
                      eps half-eps2 two-eps-sq n-params)
      total-steps (+ n-burn (* thin n-samples))
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
      init-q (mx/array [5.0 -5.0])
      [init-s init-g] (val-grad-normal init-q)
      _ (mx/materialize! init-s init-g)
      result (chain-fn init-q init-s init-g noise uniforms)]
  (mx/materialize! (aget result 0) (aget result 1) (aget result 2)
                   (aget result 3) (aget result 4))
  (assert-shape "MALA final-q" [2] (aget result 0))
  (assert-shape "MALA samples" [200 2] (aget result 3))
  (assert-true "MALA accept-count scalar" (= [] (mx/shape (aget result 4)))))

;; ---------------------------------------------------------------------------
;; 17. Fused MALA: statistical validation (2D N(0,I))
;; ---------------------------------------------------------------------------

(println "\n-- fused MALA statistical validation --")

(let [n-burn 500 n-samples 2000 thin 1 n-params 2
      step-size 0.3
      eps (mx/scalar step-size)
      half-eps2 (mx/scalar (* 0.5 step-size step-size))
      two-eps-sq (mx/scalar (* 2.0 step-size step-size))
      chain-fn (mfmbc n-burn n-samples thin val-grad-normal
                      eps half-eps2 two-eps-sq n-params)
      total-steps (+ n-burn (* thin n-samples))
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
      init-q (mx/array [3.0 -3.0])
      [init-s init-g] (val-grad-normal init-q)
      _ (mx/materialize! init-s init-g)
      result (chain-fn init-q init-s init-g noise uniforms)]
  (mx/materialize! (aget result 0) (aget result 3))
  (let [samples-js (mx/->clj (aget result 3))
        x1 (mapv first samples-js)
        x2 (mapv second samples-js)
        mean1 (/ (reduce + x1) (count x1))
        mean2 (/ (reduce + x2) (count x2))
        var1 (/ (reduce + (map #(* (- % mean1) (- % mean1)) x1)) (count x1))
        var2 (/ (reduce + (map #(* (- % mean2) (- % mean2)) x2)) (count x2))]
    (assert-close "MALA dim1 mean ≈ 0" 0.0 mean1 0.3)
    (assert-close "MALA dim2 mean ≈ 0" 0.0 mean2 0.3)
    (assert-close "MALA dim1 var ≈ 1" 1.0 var1 0.5)
    (assert-close "MALA dim2 var ≈ 1" 1.0 var2 0.5)))

;; ---------------------------------------------------------------------------
;; 18. Fused MALA: thin > 1
;; ---------------------------------------------------------------------------

(println "\n-- fused MALA thin=2 --")

(let [n-burn 100 n-samples 100 thin 2 n-params 2
      step-size 0.2
      eps (mx/scalar step-size)
      half-eps2 (mx/scalar (* 0.5 step-size step-size))
      two-eps-sq (mx/scalar (* 2.0 step-size step-size))
      chain-fn (mfmbc n-burn n-samples thin val-grad-normal
                      eps half-eps2 two-eps-sq n-params)
      total-steps (+ n-burn (* thin n-samples))
      {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
      init-q (mx/array [2.0 -2.0])
      [init-s init-g] (val-grad-normal init-q)
      _ (mx/materialize! init-s init-g)
      result (chain-fn init-q init-s init-g noise uniforms)]
  (mx/materialize! (aget result 3))
  (assert-shape "MALA thin=2 samples" [100 2] (aget result 3)))

;; ---------------------------------------------------------------------------
;; 19. fused-mala public API
;; ---------------------------------------------------------------------------

(println "\n-- fused-mala public API --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      ;; Chain within fused limits: 300 burn + 400 samples = 700 steps, 1400 MALA ops < 1500
      result (mcmc/fused-mala
               {:samples 400 :burn 300 :thin 1
                :addresses [:slope :intercept]
                :step-size 0.1 :key (rng/fresh-key)}
               linreg-model [xs] obs)]
  (assert-shape "fused-mala samples" [400 2] (:samples result))
  (assert-shape "fused-mala final-params" [2] (:final-params result))
  (assert-true "fused-mala chain-fn returned" (some? (:chain-fn result)))
  (assert-true "fused-mala acceptance-rate in (0,1)"
               (let [ar (:acceptance-rate result)]
                 (and (> ar 0) (< ar 1))))
  (let [samples-js (mx/->clj (:samples result))
        mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
    (assert-close "fused-mala posterior slope ≈ 2" 2.0 mean-slope 0.5)))

;; ---------------------------------------------------------------------------
;; 20. fused-mala cached chain-fn
;; ---------------------------------------------------------------------------

(println "\n-- fused-mala cached --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      r1 (mcmc/fused-mala
           {:samples 200 :burn 500 :thin 1
            :addresses [:slope :intercept]
            :step-size 0.05 :key (rng/fresh-key)}
           linreg-model [xs] obs)
      r2 (mcmc/fused-mala
           {:samples 200 :burn 500 :thin 1
            :addresses [:slope :intercept]
            :step-size 0.05 :key (rng/fresh-key)
            :chain-fn (:chain-fn r1)}
           linreg-model [xs] obs)]
  (assert-shape "cached fused-mala samples" [200 2] (:samples r2)))

;; ---------------------------------------------------------------------------
;; 21. fused-mala acceptance rate
;; ---------------------------------------------------------------------------

(println "\n-- fused-mala acceptance rate --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      result (mcmc/fused-mala
               {:samples 200 :burn 500 :thin 1
                :addresses [:slope :intercept]
                :step-size 0.05 :key (rng/fresh-key)}
               linreg-model [xs] obs)
      ar (:acceptance-rate result)]
  (assert-true "MALA acceptance-rate > 0" (> ar 0))
  (assert-true "MALA acceptance-rate < 1" (< ar 1))
  (println "  MALA acceptance rate:" (.toFixed ar 3)))

;; ---------------------------------------------------------------------------
;; 22. Fused HMC: shape correctness
;; ---------------------------------------------------------------------------

(println "\n-- fused HMC shape correctness --")

(let [n-burn 50 n-samples 100 thin 1 n-params 2
      step-size 0.1 leapfrog-steps 10
      eps (mx/scalar step-size)
      half-eps (mx/scalar (* 0.5 step-size))
      half (mx/scalar 0.5)
      chain-fn (mfhbc n-burn n-samples thin neg-U-normal grad-neg-U-normal
                      eps half-eps half n-params leapfrog-steps)
      total-steps (+ n-burn (* thin n-samples))
      [k1 k2] (rng/split (rng/fresh-key))
      momentum (rng/normal k1 [total-steps n-params])
      uniforms (rng/uniform k2 [total-steps])
      _ (mx/materialize! momentum uniforms)
      result (chain-fn (mx/array [5.0 -5.0]) momentum uniforms)]
  (mx/materialize! (aget result 0) (aget result 1) (aget result 2))
  (assert-shape "HMC final-q" [2] (aget result 0))
  (assert-shape "HMC samples" [100 2] (aget result 1))
  (assert-true "HMC accept-count scalar" (= [] (mx/shape (aget result 2)))))

;; ---------------------------------------------------------------------------
;; 23. Fused HMC: statistical validation (2D N(0,I))
;; ---------------------------------------------------------------------------

(println "\n-- fused HMC statistical validation --")

(let [n-burn 200 n-samples 1000 thin 1 n-params 2
      step-size 0.1 leapfrog-steps 20
      eps (mx/scalar step-size)
      half-eps (mx/scalar (* 0.5 step-size))
      half (mx/scalar 0.5)
      chain-fn (mfhbc n-burn n-samples thin neg-U-normal grad-neg-U-normal
                      eps half-eps half n-params leapfrog-steps)
      total-steps (+ n-burn (* thin n-samples))
      [k1 k2] (rng/split (rng/fresh-key))
      momentum (rng/normal k1 [total-steps n-params])
      uniforms (rng/uniform k2 [total-steps])
      _ (mx/materialize! momentum uniforms)
      result (chain-fn (mx/array [3.0 -3.0]) momentum uniforms)]
  (mx/materialize! (aget result 0) (aget result 1))
  (let [samples-js (mx/->clj (aget result 1))
        x1 (mapv first samples-js)
        x2 (mapv second samples-js)
        mean1 (/ (reduce + x1) (count x1))
        mean2 (/ (reduce + x2) (count x2))
        var1 (/ (reduce + (map #(* (- % mean1) (- % mean1)) x1)) (count x1))
        var2 (/ (reduce + (map #(* (- % mean2) (- % mean2)) x2)) (count x2))]
    (assert-close "HMC dim1 mean ≈ 0" 0.0 mean1 0.25)
    (assert-close "HMC dim2 mean ≈ 0" 0.0 mean2 0.25)
    (assert-close "HMC dim1 var ≈ 1" 1.0 var1 0.4)
    (assert-close "HMC dim2 var ≈ 1" 1.0 var2 0.4)))

;; ---------------------------------------------------------------------------
;; 24. Fused HMC: thin > 1
;; ---------------------------------------------------------------------------

(println "\n-- fused HMC thin=2 --")

(let [n-burn 50 n-samples 100 thin 2 n-params 2
      step-size 0.1 leapfrog-steps 10
      eps (mx/scalar step-size)
      half-eps (mx/scalar (* 0.5 step-size))
      half (mx/scalar 0.5)
      chain-fn (mfhbc n-burn n-samples thin neg-U-normal grad-neg-U-normal
                      eps half-eps half n-params leapfrog-steps)
      total-steps (+ n-burn (* thin n-samples))
      [k1 k2] (rng/split (rng/fresh-key))
      momentum (rng/normal k1 [total-steps n-params])
      uniforms (rng/uniform k2 [total-steps])
      _ (mx/materialize! momentum uniforms)
      result (chain-fn (mx/array [2.0 -2.0]) momentum uniforms)]
  (mx/materialize! (aget result 1))
  (assert-shape "HMC thin=2 samples" [100 2] (aget result 1)))

;; ---------------------------------------------------------------------------
;; 25. fused-hmc public API
;; ---------------------------------------------------------------------------

(println "\n-- fused-hmc public API --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      result (mcmc/fused-hmc
               {:samples 200 :burn 200 :thin 1
                :addresses [:slope :intercept]
                :step-size 0.05 :leapfrog-steps 5
                :key (rng/fresh-key)}
               linreg-model [xs] obs)]
  (assert-shape "fused-hmc samples" [200 2] (:samples result))
  (assert-shape "fused-hmc final-params" [2] (:final-params result))
  (assert-true "fused-hmc chain-fn returned" (some? (:chain-fn result)))
  (assert-true "fused-hmc acceptance-rate in (0,1)"
               (let [ar (:acceptance-rate result)]
                 (and (> ar 0) (< ar 1))))
  (let [samples-js (mx/->clj (:samples result))
        mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
    (assert-close "fused-hmc posterior slope ≈ 2" 2.0 mean-slope 0.5)))

;; ---------------------------------------------------------------------------
;; 26. fused-hmc cached chain-fn
;; ---------------------------------------------------------------------------

(println "\n-- fused-hmc cached --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      r1 (mcmc/fused-hmc
           {:samples 100 :burn 100 :thin 1
            :addresses [:slope :intercept]
            :step-size 0.05 :leapfrog-steps 5
            :key (rng/fresh-key)}
           linreg-model [xs] obs)
      r2 (mcmc/fused-hmc
           {:samples 100 :burn 100 :thin 1
            :addresses [:slope :intercept]
            :step-size 0.05 :leapfrog-steps 5
            :key (rng/fresh-key)
            :chain-fn (:chain-fn r1)}
           linreg-model [xs] obs)]
  (assert-shape "cached fused-hmc samples" [100 2] (:samples r2)))

;; ---------------------------------------------------------------------------
;; 27. fused-hmc acceptance rate
;; ---------------------------------------------------------------------------

(println "\n-- fused-hmc acceptance rate --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      result (mcmc/fused-hmc
               {:samples 100 :burn 100 :thin 1
                :addresses [:slope :intercept]
                :step-size 0.05 :leapfrog-steps 5
                :key (rng/fresh-key)}
               linreg-model [xs] obs)
      ar (:acceptance-rate result)]
  (assert-true "HMC acceptance-rate > 0" (> ar 0))
  (assert-true "HMC acceptance-rate < 1" (< ar 1))
  (println "  HMC acceptance rate:" (.toFixed ar 3)))

;; ---------------------------------------------------------------------------
;; 28. fused-mh acceptance rate (M9 retrofit)
;; ---------------------------------------------------------------------------

(println "\n-- fused-mh acceptance rate --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      result (mcmc/fused-mh
               {:samples 200 :burn 200 :thin 1
                :addresses [:slope :intercept]
                :proposal-std 0.3 :key (rng/fresh-key)}
               linreg-model [xs] obs)
      ar (:acceptance-rate result)]
  (assert-true "MH acceptance-rate present" (some? ar))
  (assert-true "MH acceptance-rate > 0" (> ar 0))
  (assert-true "MH acceptance-rate < 1" (< ar 1))
  (println "  MH acceptance rate:" (.toFixed ar 3)))

;; ---------------------------------------------------------------------------
;; 29. fused-vectorized-mh acceptance rate (M9 retrofit)
;; ---------------------------------------------------------------------------

(println "\n-- fused-vectorized-mh acceptance rate --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      result (mcmc/fused-vectorized-mh
               {:samples 100 :burn 100 :n-chains 4
                :addresses [:slope :intercept]
                :proposal-std 0.3 :key (rng/fresh-key)
                :device :cpu}
               linreg-model [xs] obs)
      ar (:acceptance-rate result)]
  (assert-true "vectorized MH acceptance-rate present" (some? ar))
  (assert-shape "vectorized MH acceptance-rate shape [4]" [4] ar)
  (let [ar-js (mx/->clj ar)]
    (assert-true "vectorized MH all rates > 0" (every? pos? ar-js))
    (assert-true "vectorized MH all rates < 1" (every? #(< % 1) ar-js))
    (println "  vectorized MH acceptance rates:" (mapv #(.toFixed % 3) ar-js))))

;; ===========================================================================
;; Phase 1: Compilation Safety Tests
;; ===========================================================================

;; ---------------------------------------------------------------------------
;; 30. validate-compiled-result passes valid result
;; ---------------------------------------------------------------------------

(println "\n-- test 30: validate-compiled-result passes valid result --")

(let [vcr @(resolve 'genmlx.inference.mcmc/validate-compiled-result)]
  ;; A valid JS array should not throw
  (let [valid-result #js [(mx/array [1.0 2.0]) (mx/zeros [10 2]) (mx/scalar 5.0)]
        threw? (volatile! false)]
    (try
      (vcr valid-result "test" 100)
      (catch :default _e
        (vreset! threw? true)))
    (assert-true "valid result does not throw" (not @threw?))))

;; ---------------------------------------------------------------------------
;; 31. validate-compiled-result catches nil
;; ---------------------------------------------------------------------------

(println "\n-- test 31: validate-compiled-result catches nil --")

(let [vcr @(resolve 'genmlx.inference.mcmc/validate-compiled-result)]
  ;; nil result should throw with descriptive message
  (let [threw? (volatile! false)
        msg (volatile! "")]
    (try
      (vcr nil "fused-mh" 5000)
      (catch :default e
        (vreset! threw? true)
        (vreset! msg (str e))))
    (assert-true "nil result throws" @threw?)
    (assert-true "error mentions Metal graph"
                 (re-find #"Metal graph too large" @msg))))

;; ---------------------------------------------------------------------------
;; 32. safe-compile-chain catches failure
;; ---------------------------------------------------------------------------

(println "\n-- test 32: safe-compile-chain catches failure --")

(let [scc @(resolve 'genmlx.inference.mcmc/safe-compile-chain)]
  ;; A chain builder that throws should be caught and re-thrown with context
  (let [threw? (volatile! false)
        msg (volatile! "")]
    (try
      (scc (fn [] (throw (js/Error. "Metal resource limit exceeded")))
           "test-method" 10)
      (catch :default e
        (vreset! threw? true)
        (vreset! msg (str e))))
    (assert-true "throwing builder caught" @threw?)
    (assert-true "error is descriptive" (pos? (count @msg)))))

;; ===========================================================================
;; Phase 2: Graph Size Estimation & Auto-Fallback Tests
;; ===========================================================================

;; ---------------------------------------------------------------------------
;; 33. estimate-fused-ops correctness
;; ---------------------------------------------------------------------------

(println "\n-- test 33: estimate-fused-ops correctness --")

(if-let [efo (some-> (resolve 'genmlx.inference.mcmc/estimate-fused-ops) deref)]
  (do
    (assert-true "MH: 1 op per step"
                 (= 3000 (efo :mh 3000 {})))
    (assert-true "MALA: 2 ops per step"
                 (= 4000 (efo :mala 2000 {})))
    (assert-true "HMC: total * leapfrog"
                 (= 2000 (efo :hmc 100 {:leapfrog-steps 20}))))
  (do
    (println "  SKIP: estimate-fused-ops not yet implemented")
    (assert-true "estimate-fused-ops exists" false)))

;; ---------------------------------------------------------------------------
;; 34. can-fuse? decision logic
;; ---------------------------------------------------------------------------

(println "\n-- test 34: can-fuse? decision logic --")

(if-let [cf (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)]
  (do
    ;; Small MH chain with GFI score: should be fuseable
    (assert-true "small GFI MH fuseable"
                 (cf :mh 500 {}))
    ;; Large MH chain with GFI score: should NOT be fuseable
    (assert-true "large GFI MH not fuseable"
                 (not (cf :mh 10000 {})))
    ;; Large MH chain with tensor-native: should be fuseable
    (assert-true "large tensor-native MH fuseable"
                 (cf :mh 10000 {:tensor-native? true})))
  (do
    (println "  SKIP: can-fuse? not yet implemented")
    (assert-true "can-fuse? exists" false)))

;; ---------------------------------------------------------------------------
;; 35. fused-mh auto-fallback on large chain
;; ---------------------------------------------------------------------------

(println "\n-- test 35: fused-mh auto-fallback on large chain --")

(if (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)
  (let [xs [1.0 2.0 3.0 4.0 5.0]
        ys (mapv #(+ (* 2.0 %) 1.0) xs)
        obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
        ;; Request large chain: 5000 burn + 5000 samples = 10000 total
        ;; Exceeds GFI MH limit of 2500
        result (mcmc/fused-mh
                 {:samples 5000 :burn 5000 :thin 1
                  :addresses [:slope :intercept]
                  :proposal-std 0.3 :key (rng/fresh-key)}
                 linreg-model [xs] obs)]
    ;; Should return valid result (auto-fell back to block-compiled)
    (assert-true "fallback has :samples" (some? (:samples result)))
    (assert-true "fallback has :final-params" (some? (:final-params result)))
    ;; :chain-fn should be nil in fallback path
    (assert-true "fallback :chain-fn is nil" (nil? (:chain-fn result)))
    ;; Posterior should still be reasonable
    (let [samples-js (mx/->clj (:samples result))
          mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
      (assert-close "fallback posterior slope ~ 2" 2.0 mean-slope 0.5)))
  (do
    (println "  SKIP: auto-fallback not yet implemented")
    (assert-true "can-fuse? exists for fallback" false)))

;; ---------------------------------------------------------------------------
;; 36. fused-mh stays fused for small chain
;; ---------------------------------------------------------------------------

(println "\n-- test 36: fused-mh stays fused for small chain --")

(if (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)
  (let [xs [1.0 2.0 3.0 4.0 5.0]
        ys (mapv #(+ (* 2.0 %) 1.0) xs)
        obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
        ;; Small chain: 200 burn + 200 samples = 400 total, well within limits
        result (mcmc/fused-mh
                 {:samples 200 :burn 200 :thin 1
                  :addresses [:slope :intercept]
                  :proposal-std 0.3 :key (rng/fresh-key)}
                 linreg-model [xs] obs)]
    (assert-true "small chain :chain-fn is non-nil" (some? (:chain-fn result)))
    (assert-shape "small chain samples shape" [200 2] (:samples result)))
  (do
    (println "  SKIP: auto-fallback not yet implemented")
    (assert-true "can-fuse? exists for fused-small" false)))

;; ---------------------------------------------------------------------------
;; 37. fused-mala auto-fallback
;; ---------------------------------------------------------------------------

(println "\n-- test 37: fused-mala auto-fallback --")

(if (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)
  (let [xs [1.0 2.0 3.0 4.0 5.0]
        ys (mapv #(+ (* 2.0 %) 1.0) xs)
        obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
        ;; Request large chain to trigger fallback
        ;; MALA: 2 ops/step, so 5000 total => 10000 ops, exceeds GFI MALA limit of 1500
        result (mcmc/fused-mala
                 {:samples 3000 :burn 2000 :thin 1
                  :addresses [:slope :intercept]
                  :step-size 0.05 :key (rng/fresh-key)}
                 linreg-model [xs] obs)]
    (assert-true "MALA fallback has :samples" (some? (:samples result)))
    (assert-true "MALA fallback has :final-params" (some? (:final-params result)))
    (assert-true "MALA fallback :chain-fn is nil" (nil? (:chain-fn result)))
    (let [samples-js (mx/->clj (:samples result))
          mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
      (assert-close "MALA fallback posterior slope ~ 2" 2.0 mean-slope 0.5)))
  (do
    (println "  SKIP: auto-fallback not yet implemented")
    (assert-true "can-fuse? exists for MALA fallback" false)))

;; ---------------------------------------------------------------------------
;; 38. fused-hmc auto-fallback
;; ---------------------------------------------------------------------------

(println "\n-- test 38: fused-hmc auto-fallback --")

(if (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)
  (let [xs [1.0 2.0 3.0 4.0 5.0]
        ys (mapv #(+ (* 2.0 %) 1.0) xs)
        obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
        ;; HMC: total * leapfrog ops. 500 total * 20 leapfrog = 10000, exceeds GFI HMC limit of 8000
        result (mcmc/fused-hmc
                 {:samples 300 :burn 200 :thin 1
                  :addresses [:slope :intercept]
                  :step-size 0.05 :leapfrog-steps 20
                  :key (rng/fresh-key)}
                 linreg-model [xs] obs)]
    (assert-true "HMC fallback has :samples" (some? (:samples result)))
    (assert-true "HMC fallback has :final-params" (some? (:final-params result)))
    (assert-true "HMC fallback :chain-fn is nil" (nil? (:chain-fn result)))
    (let [samples-js (mx/->clj (:samples result))
          mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
      (assert-close "HMC fallback posterior slope ~ 2" 2.0 mean-slope 0.5)))
  (do
    (println "  SKIP: auto-fallback not yet implemented")
    (assert-true "can-fuse? exists for HMC fallback" false)))

;; ===========================================================================
;; Phase 3: Step-Size Adaptation Tests
;; ===========================================================================

;; ---------------------------------------------------------------------------
;; 39. fused-mala with adapt-step-size
;; ---------------------------------------------------------------------------

(println "\n-- test 39: fused-mala with adapt-step-size --")

(try
  (let [xs [1.0 2.0 3.0 4.0 5.0]
        ys (mapv #(+ (* 2.0 %) 1.0) xs)
        obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
        ;; Chain within fused limits: 500 burn + 200 samples = 700 steps, 1400 MALA ops < 1500
        result (mcmc/fused-mala
                 {:samples 200 :burn 500 :thin 1
                  :addresses [:slope :intercept]
                  :adapt-step-size true
                  :warmup-steps 200
                  :key (rng/fresh-key)}
                 linreg-model [xs] obs)]
    (assert-true "adapted MALA has :samples" (some? (:samples result)))
    (assert-true "adapted MALA has :acceptance-rate" (some? (:acceptance-rate result)))
    ;; Acceptance rate should be in reasonable range (adaptation should help)
    (let [ar (:acceptance-rate result)]
      (assert-true "adapted MALA acceptance-rate > 0.1" (> ar 0.1))
      (assert-true "adapted MALA acceptance-rate < 0.95" (< ar 0.95))
      (println "  adapted MALA acceptance rate:" (.toFixed ar 3)))
    ;; Posterior should converge
    (let [samples-js (mx/->clj (:samples result))
          mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
      (assert-close "adapted MALA posterior slope ~ 2" 2.0 mean-slope 0.5)))
  (catch :default e
    (println "  SKIP: adapt-step-size not yet implemented -" (str e))
    (assert-true "fused-mala adapt-step-size implemented" false)))

;; ---------------------------------------------------------------------------
;; 40. fused-hmc with adapt-step-size
;; ---------------------------------------------------------------------------

(println "\n-- test 40: fused-hmc with adapt-step-size --")

(try
  (let [xs [1.0 2.0 3.0 4.0 5.0]
        ys (mapv #(+ (* 2.0 %) 1.0) xs)
        obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
        result (mcmc/fused-hmc
                 {:samples 200 :burn 200 :thin 1
                  :addresses [:slope :intercept]
                  :adapt-step-size true
                  :leapfrog-steps 5
                  :key (rng/fresh-key)}
                 linreg-model [xs] obs)]
    (assert-true "adapted HMC has :samples" (some? (:samples result)))
    (assert-true "adapted HMC has :acceptance-rate" (some? (:acceptance-rate result)))
    (let [ar (:acceptance-rate result)]
      (assert-true "adapted HMC acceptance-rate > 0.1" (> ar 0.1))
      (assert-true "adapted HMC acceptance-rate < 0.95" (< ar 0.95))
      (println "  adapted HMC acceptance rate:" (.toFixed ar 3)))
    (let [samples-js (mx/->clj (:samples result))
          mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
      (assert-close "adapted HMC posterior slope ~ 2" 2.0 mean-slope 0.5)))
  (catch :default e
    (println "  SKIP: adapt-step-size not yet implemented -" (str e))
    (assert-true "fused-hmc adapt-step-size implemented" false)))

;; ---------------------------------------------------------------------------
;; 41. mala (non-fused) with adapt-step-size
;; ---------------------------------------------------------------------------

(println "\n-- test 41: mala (non-fused) with adapt-step-size --")

(try
  (let [xs [1.0 2.0 3.0 4.0 5.0]
        ys (mapv #(+ (* 2.0 %) 1.0) xs)
        obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
        result (mcmc/mala
                 {:samples 50 :burn 200 :thin 1
                  :addresses [:slope :intercept]
                  :adapt-step-size true
                  :key (rng/fresh-key)}
                 linreg-model [xs] obs)]
    ;; Non-fused mala returns a vector of parameter samples
    (assert-true "adapted mala returns samples" (vector? result))
    (assert-true "adapted mala has 50 samples" (= 50 (count result)))
    ;; Posterior should be reasonable
    (let [mean-slope (/ (reduce + (mapv first result)) (count result))]
      (assert-close "adapted mala posterior slope ~ 2" 2.0 mean-slope 1.0)))
  (catch :default e
    (println "  SKIP: mala adapt-step-size not yet implemented -" (str e))
    (assert-true "mala adapt-step-size implemented" false)))

;; ---------------------------------------------------------------------------
;; 42. adaptation does not change default behavior
;; ---------------------------------------------------------------------------

(println "\n-- test 42: adaptation does not change default behavior --")

(let [xs [1.0 2.0 3.0 4.0 5.0]
      ys (mapv #(+ (* 2.0 %) 1.0) xs)
      obs (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys))
      ;; Run fused-mala WITHOUT adapt-step-size — small chain within fuse limits
      ;; (GFI MALA limit: 1500 ops = 750 steps max)
      result (mcmc/fused-mala
               {:samples 300 :burn 300 :thin 1
                :addresses [:slope :intercept]
                :step-size 0.1 :key (rng/fresh-key)}
               linreg-model [xs] obs)]
  ;; Should behave exactly as before — same structure
  (assert-shape "no-adapt samples" [300 2] (:samples result))
  (assert-shape "no-adapt final-params" [2] (:final-params result))
  (assert-true "no-adapt chain-fn returned" (some? (:chain-fn result)))
  (assert-true "no-adapt acceptance-rate in (0,1)"
               (let [ar (:acceptance-rate result)]
                 (and (> ar 0) (< ar 1)))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n  Passed: " @pass-count))
(println (str "  Failed: " @fail-count))
(println (str "  Total:  " (+ @pass-count @fail-count)))
(if (zero? @fail-count)
  (println "\n  *** ALL FUSED MCMC TESTS PASS ***")
  (println "\n  *** SOME TESTS FAILED ***"))
