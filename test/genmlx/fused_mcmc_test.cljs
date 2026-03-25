(ns genmlx.fused-mcmc-test
  "Tests for fused MCMC: pre-generated noise, fused burn-in, fused collection,
   and fused burn+collect (M2-M4)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]))

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
;; Model
;; ---------------------------------------------------------------------------

(def linreg-model
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [j (range (count xs))]
        (let [x (nth xs j)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x)) intercept) 1))))
      slope)))

;; ---------------------------------------------------------------------------
;; MALA / HMC helpers
;; ---------------------------------------------------------------------------

(def ^:private val-grad-normal (mx/value-and-grad std-normal-score))
(def ^:private neg-U-normal (fn [q] (mx/negative (std-normal-score q))))
(def ^:private grad-neg-U-normal (mx/grad neg-U-normal))

;; ---------------------------------------------------------------------------
;; Helper for linreg observations
;; ---------------------------------------------------------------------------

(defn- linreg-obs [xs ys]
  (apply cm/choicemap (mapcat (fn [j y] [(keyword (str "y" j)) y]) (range) ys)))

(def ^:private xs [1.0 2.0 3.0 4.0 5.0])
(def ^:private ys (mapv #(+ (* 2.0 %) 1.0) xs))
(def ^:private obs (linreg-obs xs ys))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest pre-generate-chain-noise-test
  (testing "pre-generate-chain-noise"
    (let [{:keys [noise uniforms]} (pgcn (rng/fresh-key) 100 3)]
      (is (= [100 3] (mx/shape noise)) "noise shape=[100 3]")
      (is (= [100] (mx/shape uniforms)) "uniforms shape=[100]")
      (let [u-min (mx/item (mx/amin uniforms))
            u-max (mx/item (mx/amax uniforms))]
        (is (and (>= u-min 0) (<= u-max 1)) "uniforms in [0,1]")))))

(deftest write-sample-row-test
  (testing "write-sample-row"
    (let [samples (mx/zeros [5 3])
          params (mx/array [1.0 2.0 3.0])
          idx (mx/astype (mx/array [2]) mx/int32)
          result (write-row samples params idx 5 3)]
      (is (= [5 3] (mx/shape result)) "write result shape=[5 3]")
      (let [r (mx/->clj result)]
        (is (= [0 0 0] (nth r 0)) "row 0 unchanged")
        (is (= [1 2 3] (nth r 2)) "row 2 written")
        (is (= [0 0 0] (nth r 4)) "row 4 unchanged"))))

  (testing "sequential writes"
    (let [samples (mx/zeros [3 2])
          s1 (write-row samples (mx/array [1.0 2.0]) (mx/astype (mx/array [0]) mx/int32) 3 2)
          s2 (write-row s1 (mx/array [3.0 4.0]) (mx/astype (mx/array [1]) mx/int32) 3 2)
          s3 (write-row s2 (mx/array [5.0 6.0]) (mx/astype (mx/array [2]) mx/int32) 3 2)]
      (is (= [[1 2] [3 4] [5 6]] (mx/->clj s3)) "sequential writes"))))

(deftest make-fused-burn-in-test
  (testing "make-fused-burn-in"
    (let [n-burn 200
          n-params 2
          std (mx/scalar 0.5)
          burn-fn (mfbi n-burn std-normal-score std n-params)
          {:keys [noise uniforms]} (pgcn (rng/fresh-key) n-burn n-params)
          result (burn-fn (mx/array [10.0 -10.0]) noise uniforms)]
      (mx/materialize! result)
      (is (= [2] (mx/shape result)) "burn-in result shape=[2]")
      (let [final (mx/->clj result)
            dist-sq (reduce + (map #(* % %) final))]
        (is (< dist-sq 200) "moved toward origin")))))

(deftest make-fused-collection-thin1-test
  (testing "make-fused-collection thin=1"
    (let [n-samples 100
          thin 1
          n-params 2
          std (mx/scalar 0.5)
          collect-fn (mfc n-samples thin std-normal-score std n-params)
          total-steps (* thin n-samples)
          {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
          result (collect-fn (mx/array [1.0 -1.0]) noise uniforms)]
      (mx/materialize! result)
      (is (= [100 2] (mx/shape result)) "collection result shape=[100 2]")
      (let [samples-js (mx/->clj result)]
        (is (not= (first samples-js) (last samples-js)) "first != last row")))))

(deftest make-fused-collection-thin3-test
  (testing "make-fused-collection thin=3"
    (let [n-samples 50
          thin 3
          n-params 2
          std (mx/scalar 0.5)
          collect-fn (mfc n-samples thin std-normal-score std n-params)
          total-steps (* thin n-samples)
          {:keys [noise uniforms]} (pgcn (rng/fresh-key) total-steps n-params)
          result (collect-fn (mx/array [1.0 -1.0]) noise uniforms)]
      (mx/materialize! result)
      (is (= [50 2] (mx/shape result)) "thin=3 collection shape=[50 2]")
      (let [samples-js (mx/->clj result)]
        (is (not= (first samples-js) (last samples-js)) "first != last (thin=3)")))))

(deftest fused-burn-and-collect-correctness-test
  (testing "fused burn+collect correctness"
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
      (is (= [1] (mx/shape (aget result 0))) "final params shape=[1]")
      (is (= [500 1] (mx/shape (aget result 1))) "samples shape=[500 1]")
      (let [samples-js (mx/->clj (aget result 1))
            vals (mapv first samples-js)
            mean (/ (reduce + vals) (count vals))
            variance (/ (reduce + (map #(* (- % mean) (- % mean)) vals)) (count vals))
            std-dev (js/Math.sqrt variance)]
        (is (h/close? 0.0 mean 0.3) "posterior mean ~ 0")
        (is (h/close? 1.0 std-dev 0.3) "posterior std ~ 1")))))

(deftest fused-burn-collect-thin2-test
  (testing "fused burn+collect thin=2"
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
      (is (= [200 2] (mx/shape (aget result 1))) "thin=2 samples shape=[200 2]")
      (let [samples-js (mx/->clj (aget result 1))
            non-zero (count (filter (fn [row] (not= [0 0] row)) samples-js))]
        (is (= 200 non-zero) "all rows populated (thin=2)")))))

(deftest statistical-validation-2d-test
  (testing "statistical validation: 2D N(0,I)"
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
      (is (h/close? 0.0 mean1 0.25) "dim1 mean ~ 0")
      (is (h/close? 0.0 mean2 0.25) "dim2 mean ~ 0")
      (is (h/close? 1.0 var1 0.4) "dim1 var ~ 1")
      (is (h/close? 1.0 var2 0.4) "dim2 var ~ 1"))))

(deftest performance-test
  (testing "performance"
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
      (is (< fused-ms 200) "fused < 200ms per chain"))))

(deftest model-integration-linreg-test
  (testing "model integration: linreg"
    (let [model (dyn/auto-key linreg-model)
          {:keys [trace]} (p/generate model [xs] obs)
          {:keys [score-fn init-params n-params]}
          (u/prepare-mcmc-score model [xs] obs [:slope :intercept] trace)
          std (mx/scalar 0.3)
          chain-fn (mfbc 1000 2000 1 score-fn std n-params)
          {:keys [noise uniforms]} (pgcn (rng/fresh-key) 3000 n-params)
          result (chain-fn init-params noise uniforms)
          _ (mx/materialize! (aget result 0) (aget result 1))
          samples-js (mx/->clj (aget result 1))
          slopes (mapv first samples-js)
          intercepts (mapv second samples-js)
          mean-slope (/ (reduce + slopes) (count slopes))
          mean-int (/ (reduce + intercepts) (count intercepts))]
      (is (= [2000 2] (mx/shape (aget result 1))) "linreg samples shape=[2000 2]")
      (is (h/close? 2.0 mean-slope 0.5) "posterior slope ~ 2")
      (is (h/close? 1.0 mean-int 1.5) "posterior intercept ~ 1"))))

(deftest fused-mh-public-api-test
  (testing "fused-mh public API"
    (let [r1 (mcmc/fused-mh
               {:samples 500 :burn 300 :thin 1
                :addresses [:slope :intercept]
                :proposal-std 0.3 :key (rng/fresh-key)}
               linreg-model [xs] obs)]
      (is (= [500 2] (mx/shape (:samples r1))) "fused-mh samples shape=[500 2]")
      (is (= [2] (mx/shape (:final-params r1))) "fused-mh final-params shape=[2]")
      (is (some? (:chain-fn r1)) "chain-fn returned")

      (testing "cached chain-fn"
        (let [r2 (mcmc/fused-mh
                   {:samples 500 :burn 300 :thin 1
                    :addresses [:slope :intercept]
                    :proposal-std 0.3 :key (rng/fresh-key)
                    :chain-fn (:chain-fn r1)}
                   linreg-model [xs] obs)
              samples-js (mx/->clj (:samples r2))
              mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
          (is (= [500 2] (mx/shape (:samples r2))) "cached fused-mh samples shape=[500 2]")
          (is (h/close? 2.0 mean-slope 0.5) "cached posterior slope ~ 2"))))))

(deftest fused-mh-thin2-test
  (testing "fused-mh thin=2"
    (let [result (mcmc/fused-mh
                   {:samples 200 :burn 200 :thin 2
                    :addresses [:slope :intercept]
                    :proposal-std 0.3 :key (rng/fresh-key)}
                   linreg-model [xs] obs)]
      (is (= [200 2] (mx/shape (:samples result))) "fused-mh thin=2 shape=[200 2]"))))

(deftest fused-vectorized-mh-test
  (testing "fused-vectorized-mh"
    (let [result (mcmc/fused-vectorized-mh
                   {:samples 200 :burn 200 :n-chains 4
                    :addresses [:slope :intercept]
                    :proposal-std 0.3 :key (rng/fresh-key)
                    :device :cpu}
                   linreg-model [xs] obs)]
      (is (= [200 4 2] (mx/shape (:samples result))) "vectorized samples [S,N,D]")
      (is (= [4 2] (mx/shape (:final-params result))) "vectorized final-params [N,D]")
      (is (some? (:chain-fn result)) "vectorized chain-fn returned")
      (let [samples-js (mx/->clj (:samples result))
            all-slopes (for [s samples-js, chain s] (first chain))
            mean-slope (/ (reduce + all-slopes) (count all-slopes))]
        (is (h/close? 2.0 mean-slope 0.5) "vectorized posterior slope ~ 2")))))

(deftest fused-vectorized-mh-cached-test
  (testing "fused-vectorized-mh cached"
    (let [r1 (mcmc/fused-vectorized-mh
               {:samples 200 :burn 100 :n-chains 4
                :addresses [:slope :intercept]
                :proposal-std 0.3 :key (rng/fresh-key) :device :cpu}
               linreg-model [xs] obs)
          t0 (.now js/Date)
          r2 (mcmc/fused-vectorized-mh
               {:samples 200 :burn 100 :n-chains 4
                :addresses [:slope :intercept]
                :proposal-std 0.3 :key (rng/fresh-key)
                :chain-fn (:chain-fn r1) :device :cpu}
               linreg-model [xs] obs)
          t1 (.now js/Date)]
      (is (= [200 4 2] (mx/shape (:samples r2))) "cached vectorized samples shape")
      (is (< (- t1 t0) 500) "cached vectorized < 500ms"))))

(deftest fused-vectorized-mh-thin2-test
  (testing "fused-vectorized-mh thin=2"
    (let [result (mcmc/fused-vectorized-mh
                   {:samples 100 :burn 100 :thin 2 :n-chains 4
                    :addresses [:slope :intercept]
                    :proposal-std 0.3 :key (rng/fresh-key) :device :cpu}
                   linreg-model [xs] obs)]
      (is (= [100 4 2] (mx/shape (:samples result))) "vectorized thin=2 shape"))))

(deftest fused-mala-shape-test
  (testing "fused MALA shape correctness"
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
      (is (= [2] (mx/shape (aget result 0))) "MALA final-q shape=[2]")
      (is (= [200 2] (mx/shape (aget result 3))) "MALA samples shape=[200 2]")
      (is (= [] (mx/shape (aget result 4))) "MALA accept-count scalar"))))

(deftest fused-mala-statistical-test
  (testing "fused MALA statistical validation"
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
        (is (h/close? 0.0 mean1 0.3) "MALA dim1 mean ~ 0")
        (is (h/close? 0.0 mean2 0.3) "MALA dim2 mean ~ 0")
        (is (h/close? 1.0 var1 0.5) "MALA dim1 var ~ 1")
        (is (h/close? 1.0 var2 0.5) "MALA dim2 var ~ 1")))))

(deftest fused-mala-thin2-test
  (testing "fused MALA thin=2"
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
      (is (= [100 2] (mx/shape (aget result 3))) "MALA thin=2 samples shape=[100 2]"))))

(deftest fused-mala-public-api-test
  (testing "fused-mala public API"
    (let [result (mcmc/fused-mala
                   {:samples 400 :burn 300 :thin 1
                    :addresses [:slope :intercept]
                    :step-size 0.1 :key (rng/fresh-key)}
                   linreg-model [xs] obs)]
      (is (= [400 2] (mx/shape (:samples result))) "fused-mala samples shape=[400 2]")
      (is (= [2] (mx/shape (:final-params result))) "fused-mala final-params shape=[2]")
      (is (some? (:chain-fn result)) "fused-mala chain-fn returned")
      (is (let [ar (:acceptance-rate result)]
            (and (> ar 0) (< ar 1)))
          "fused-mala acceptance-rate in (0,1)")
      (let [samples-js (mx/->clj (:samples result))
            mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
        (is (h/close? 2.0 mean-slope 0.5) "fused-mala posterior slope ~ 2")))))

(deftest fused-mala-cached-test
  (testing "fused-mala cached"
    (let [r1 (mcmc/fused-mala
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
      (is (= [200 2] (mx/shape (:samples r2))) "cached fused-mala samples shape=[200 2]"))))

(deftest fused-mala-acceptance-rate-test
  (testing "fused-mala acceptance rate"
    (let [result (mcmc/fused-mala
                   {:samples 200 :burn 500 :thin 1
                    :addresses [:slope :intercept]
                    :step-size 0.05 :key (rng/fresh-key)}
                   linreg-model [xs] obs)
          ar (:acceptance-rate result)]
      (is (> ar 0) "MALA acceptance-rate > 0")
      (is (< ar 1) "MALA acceptance-rate < 1"))))

(deftest fused-hmc-shape-test
  (testing "fused HMC shape correctness"
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
      (is (= [2] (mx/shape (aget result 0))) "HMC final-q shape=[2]")
      (is (= [100 2] (mx/shape (aget result 1))) "HMC samples shape=[100 2]")
      (is (= [] (mx/shape (aget result 2))) "HMC accept-count scalar"))))

(deftest fused-hmc-statistical-test
  (testing "fused HMC statistical validation"
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
        (is (h/close? 0.0 mean1 0.25) "HMC dim1 mean ~ 0")
        (is (h/close? 0.0 mean2 0.25) "HMC dim2 mean ~ 0")
        (is (h/close? 1.0 var1 0.4) "HMC dim1 var ~ 1")
        (is (h/close? 1.0 var2 0.4) "HMC dim2 var ~ 1")))))

(deftest fused-hmc-thin2-test
  (testing "fused HMC thin=2"
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
      (is (= [100 2] (mx/shape (aget result 1))) "HMC thin=2 samples shape=[100 2]"))))

(deftest fused-hmc-public-api-test
  (testing "fused-hmc public API"
    (let [result (mcmc/fused-hmc
                   {:samples 200 :burn 200 :thin 1
                    :addresses [:slope :intercept]
                    :step-size 0.05 :leapfrog-steps 5
                    :key (rng/fresh-key)}
                   linreg-model [xs] obs)]
      (is (= [200 2] (mx/shape (:samples result))) "fused-hmc samples shape=[200 2]")
      (is (= [2] (mx/shape (:final-params result))) "fused-hmc final-params shape=[2]")
      (is (some? (:chain-fn result)) "fused-hmc chain-fn returned")
      (is (let [ar (:acceptance-rate result)]
            (and (> ar 0) (< ar 1)))
          "fused-hmc acceptance-rate in (0,1)")
      (let [samples-js (mx/->clj (:samples result))
            mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
        (is (h/close? 2.0 mean-slope 0.5) "fused-hmc posterior slope ~ 2")))))

(deftest fused-hmc-cached-test
  (testing "fused-hmc cached"
    (let [r1 (mcmc/fused-hmc
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
      (is (= [100 2] (mx/shape (:samples r2))) "cached fused-hmc samples shape=[100 2]"))))

(deftest fused-hmc-acceptance-rate-test
  (testing "fused-hmc acceptance rate"
    (let [result (mcmc/fused-hmc
                   {:samples 100 :burn 100 :thin 1
                    :addresses [:slope :intercept]
                    :step-size 0.05 :leapfrog-steps 5
                    :key (rng/fresh-key)}
                   linreg-model [xs] obs)
          ar (:acceptance-rate result)]
      (is (> ar 0) "HMC acceptance-rate > 0")
      (is (< ar 1) "HMC acceptance-rate < 1"))))

(deftest fused-mh-acceptance-rate-test
  (testing "fused-mh acceptance rate"
    (let [result (mcmc/fused-mh
                   {:samples 200 :burn 200 :thin 1
                    :addresses [:slope :intercept]
                    :proposal-std 0.3 :key (rng/fresh-key)}
                   linreg-model [xs] obs)
          ar (:acceptance-rate result)]
      (is (some? ar) "MH acceptance-rate present")
      (is (> ar 0) "MH acceptance-rate > 0")
      (is (< ar 1) "MH acceptance-rate < 1"))))

(deftest fused-vectorized-mh-acceptance-rate-test
  (testing "fused-vectorized-mh acceptance rate"
    (let [result (mcmc/fused-vectorized-mh
                   {:samples 100 :burn 100 :n-chains 4
                    :addresses [:slope :intercept]
                    :proposal-std 0.3 :key (rng/fresh-key)
                    :device :cpu}
                   linreg-model [xs] obs)
          ar (:acceptance-rate result)]
      (is (some? ar) "vectorized MH acceptance-rate present")
      (is (= [4] (mx/shape ar)) "vectorized MH acceptance-rate shape [4]")
      (let [ar-js (mx/->clj ar)]
        (is (every? pos? ar-js) "vectorized MH all rates > 0")
        (is (every? #(< % 1) ar-js) "vectorized MH all rates < 1")))))

(deftest validate-compiled-result-test
  (testing "validate-compiled-result passes valid result"
    (let [vcr @(resolve 'genmlx.inference.mcmc/validate-compiled-result)]
      (let [valid-result #js [(mx/array [1.0 2.0]) (mx/zeros [10 2]) (mx/scalar 5.0)]
            threw? (volatile! false)]
        (try
          (vcr valid-result "test" 100)
          (catch :default _e
            (vreset! threw? true)))
        (is (not @threw?) "valid result does not throw"))))

  (testing "validate-compiled-result catches nil"
    (let [vcr @(resolve 'genmlx.inference.mcmc/validate-compiled-result)]
      (let [threw? (volatile! false)
            msg (volatile! "")]
        (try
          (vcr nil "fused-mh" 5000)
          (catch :default e
            (vreset! threw? true)
            (vreset! msg (str e))))
        (is @threw? "nil result throws")
        (is (re-find #"Metal graph too large" @msg) "error mentions Metal graph")))))

(deftest safe-compile-chain-test
  (testing "safe-compile-chain catches failure"
    (let [scc @(resolve 'genmlx.inference.mcmc/safe-compile-chain)]
      (let [threw? (volatile! false)
            msg (volatile! "")]
        (try
          (scc (fn [] (throw (js/Error. "Metal resource limit exceeded")))
               "test-method" 10)
          (catch :default e
            (vreset! threw? true)
            (vreset! msg (str e))))
        (is @threw? "throwing builder caught")
        (is (pos? (count @msg)) "error is descriptive")))))

(deftest estimate-fused-ops-test
  (testing "estimate-fused-ops correctness"
    (if-let [efo (some-> (resolve 'genmlx.inference.mcmc/estimate-fused-ops) deref)]
      (do
        (is (= 3000 (efo :mh 3000 {})) "MH: 1 op per step")
        (is (= 4000 (efo :mala 2000 {})) "MALA: 2 ops per step")
        (is (= 2000 (efo :hmc 100 {:leapfrog-steps 20})) "HMC: total * leapfrog"))
      (is false "estimate-fused-ops exists"))))

(deftest can-fuse-decision-test
  (testing "can-fuse? decision logic"
    (if-let [cf (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)]
      (do
        (is (cf :mh 500 {}) "small GFI MH fuseable")
        (is (not (cf :mh 10000 {})) "large GFI MH not fuseable")
        (is (cf :mh 10000 {:tensor-native? true}) "large tensor-native MH fuseable"))
      (is false "can-fuse? exists"))))

(deftest fused-mh-auto-fallback-test
  (testing "fused-mh auto-fallback on large chain"
    (if (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)
      (let [result (mcmc/fused-mh
                     {:samples 5000 :burn 5000 :thin 1
                      :addresses [:slope :intercept]
                      :proposal-std 0.3 :key (rng/fresh-key)}
                     linreg-model [xs] obs)]
        (is (some? (:samples result)) "fallback has :samples")
        (is (some? (:final-params result)) "fallback has :final-params")
        (is (nil? (:chain-fn result)) "fallback :chain-fn is nil")
        (let [samples-js (mx/->clj (:samples result))
              mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
          (is (h/close? 2.0 mean-slope 0.5) "fallback posterior slope ~ 2")))
      (is false "can-fuse? exists for fallback"))))

(deftest fused-mh-stays-fused-small-chain-test
  (testing "fused-mh stays fused for small chain"
    (if (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)
      (let [result (mcmc/fused-mh
                     {:samples 200 :burn 200 :thin 1
                      :addresses [:slope :intercept]
                      :proposal-std 0.3 :key (rng/fresh-key)}
                     linreg-model [xs] obs)]
        (is (some? (:chain-fn result)) "small chain :chain-fn is non-nil")
        (is (= [200 2] (mx/shape (:samples result))) "small chain samples shape"))
      (is false "can-fuse? exists for fused-small"))))

(deftest fused-mala-auto-fallback-test
  (testing "fused-mala auto-fallback"
    (if (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)
      (let [result (mcmc/fused-mala
                     {:samples 3000 :burn 2000 :thin 1
                      :addresses [:slope :intercept]
                      :step-size 0.05 :key (rng/fresh-key)}
                     linreg-model [xs] obs)]
        (is (some? (:samples result)) "MALA fallback has :samples")
        (is (some? (:final-params result)) "MALA fallback has :final-params")
        (is (nil? (:chain-fn result)) "MALA fallback :chain-fn is nil")
        (let [samples-js (mx/->clj (:samples result))
              mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
          (is (h/close? 2.0 mean-slope 0.5) "MALA fallback posterior slope ~ 2")))
      (is false "can-fuse? exists for MALA fallback"))))

(deftest fused-hmc-auto-fallback-test
  (testing "fused-hmc auto-fallback"
    (if (some-> (resolve 'genmlx.inference.mcmc/can-fuse?) deref)
      (let [result (mcmc/fused-hmc
                     {:samples 300 :burn 200 :thin 1
                      :addresses [:slope :intercept]
                      :step-size 0.05 :leapfrog-steps 20
                      :key (rng/fresh-key)}
                     linreg-model [xs] obs)]
        (is (some? (:samples result)) "HMC fallback has :samples")
        (is (some? (:final-params result)) "HMC fallback has :final-params")
        (is (nil? (:chain-fn result)) "HMC fallback :chain-fn is nil")
        (let [samples-js (mx/->clj (:samples result))
              mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
          (is (h/close? 2.0 mean-slope 0.5) "HMC fallback posterior slope ~ 2")))
      (is false "can-fuse? exists for HMC fallback"))))

(deftest fused-mala-adapt-step-size-test
  (testing "fused-mala with adapt-step-size"
    (try
      (let [result (mcmc/fused-mala
                     {:samples 200 :burn 500 :thin 1
                      :addresses [:slope :intercept]
                      :adapt-step-size true
                      :warmup-steps 200
                      :key (rng/fresh-key)}
                     linreg-model [xs] obs)]
        (is (some? (:samples result)) "adapted MALA has :samples")
        (is (some? (:acceptance-rate result)) "adapted MALA has :acceptance-rate")
        (let [ar (:acceptance-rate result)]
          (is (> ar 0.1) "adapted MALA acceptance-rate > 0.1")
          (is (< ar 0.95) "adapted MALA acceptance-rate < 0.95"))
        (let [samples-js (mx/->clj (:samples result))
              mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
          (is (h/close? 2.0 mean-slope 0.5) "adapted MALA posterior slope ~ 2")))
      (catch :default _e
        (is false "fused-mala adapt-step-size implemented")))))

(deftest fused-hmc-adapt-step-size-test
  (testing "fused-hmc with adapt-step-size"
    (try
      (let [result (mcmc/fused-hmc
                     {:samples 200 :burn 200 :thin 1
                      :addresses [:slope :intercept]
                      :adapt-step-size true
                      :leapfrog-steps 5
                      :key (rng/fresh-key)}
                     linreg-model [xs] obs)]
        (is (some? (:samples result)) "adapted HMC has :samples")
        (is (some? (:acceptance-rate result)) "adapted HMC has :acceptance-rate")
        (let [ar (:acceptance-rate result)]
          (is (> ar 0.1) "adapted HMC acceptance-rate > 0.1")
          (is (< ar 0.95) "adapted HMC acceptance-rate < 0.95"))
        (let [samples-js (mx/->clj (:samples result))
              mean-slope (/ (reduce + (mapv first samples-js)) (count samples-js))]
          (is (h/close? 2.0 mean-slope 0.5) "adapted HMC posterior slope ~ 2")))
      (catch :default _e
        (is false "fused-hmc adapt-step-size implemented")))))

(deftest mala-non-fused-adapt-test
  (testing "mala (non-fused) with adapt-step-size"
    (try
      (let [result (mcmc/mala
                     {:samples 50 :burn 200 :thin 1
                      :addresses [:slope :intercept]
                      :adapt-step-size true
                      :key (rng/fresh-key)}
                     linreg-model [xs] obs)]
        (is (vector? result) "adapted mala returns samples")
        (is (= 50 (count result)) "adapted mala has 50 samples")
        (let [mean-slope (/ (reduce + (mapv first result)) (count result))]
          (is (h/close? 2.0 mean-slope 1.0) "adapted mala posterior slope ~ 2")))
      (catch :default _e
        (is false "mala adapt-step-size implemented")))))

(deftest adaptation-default-behavior-test
  (testing "adaptation does not change default behavior"
    (let [result (mcmc/fused-mala
                   {:samples 300 :burn 300 :thin 1
                    :addresses [:slope :intercept]
                    :step-size 0.1 :key (rng/fresh-key)}
                   linreg-model [xs] obs)]
      (is (= [300 2] (mx/shape (:samples result))) "no-adapt samples shape=[300 2]")
      (is (= [2] (mx/shape (:final-params result))) "no-adapt final-params shape=[2]")
      (is (some? (:chain-fn result)) "no-adapt chain-fn returned")
      (is (let [ar (:acceptance-rate result)]
            (and (> ar 0) (< ar 1)))
          "no-adapt acceptance-rate in (0,1)"))))

(cljs.test/run-tests)
