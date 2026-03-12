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
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n  Passed: " @pass-count))
(println (str "  Failed: " @fail-count))
(println (str "  Total:  " (+ @pass-count @fail-count)))
(if (zero? @fail-count)
  (println "\n  *** ALL FUSED MCMC TESTS PASS ***")
  (println "\n  *** SOME TESTS FAILED ***"))
