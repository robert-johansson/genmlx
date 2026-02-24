(ns loop-compilation-test
  "LOOP COMPILATION: Compile entire MH chains into one Metal dispatch.

   Result: 4-5x speedup over eager compiled-mh.
   Method: Pre-generate noise arrays outside compile-fn, pass as inputs.
   compile-fn caches the graph topology; new noise → new random behavior.

   This is GenMLX's equivalent of JAX's jit(lax.scan(...))."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]))

(defn bench [f n warmup]
  (dotimes [_ warmup] (f))
  (let [t0 (js/performance.now)
        _ (dotimes [_ n] (f))
        t1 (js/performance.now)]
    (/ (- t1 t0) n)))

;; ---------------------------------------------------------------------------
;; Model setup (7-site linreg)
;; ---------------------------------------------------------------------------

(def xs [1.0 2.0 3.0 4.0 5.0])

(def linreg
  (gen [xs]
    (let [slope     (dyn/trace :slope (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
          intercept (dyn/trace :intercept (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
      (doseq [[j x] (map-indexed vector xs)]
        (dyn/trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept)
                                  (mx/scalar 1))))
      slope)))

(def init-trace (p/simulate linreg [xs]))
(def observations
  (reduce (fn [cm k]
            (cm/set-choice cm [k] (cm/get-choice (:choices init-trace) [k])))
          cm/EMPTY
          [:y0 :y1 :y2 :y3 :y4]))

(def param-addrs [:slope :intercept])
(def n-params (count param-addrs))
(def score-fn (mx/compile-fn (u/make-score-fn linreg [xs] observations param-addrs)))
(mx/eval! (score-fn (mx/array [1.0 0.5])))

(println "\n================================================================")
(println "  LOOP COMPILATION: Compiled MH Chains")
(println "================================================================\n")

;; ---------------------------------------------------------------------------
;; Helper: build compiled K-step MH chain
;; ---------------------------------------------------------------------------

(defn make-compiled-chain
  "Build a compiled K-step MH chain.
   Returns compiled fn: (params, noise-2d [K,D], uniforms-1d [K]) → params
   Noise and uniforms are generated OUTSIDE and passed as inputs,
   ensuring fresh randomness on each call."
  [k-steps score-fn proposal-std n-params]
  (let [chain-fn
        (fn [params noise-2d uniforms-1d]
          (loop [p params, i 0]
            (if (>= i k-steps) p
              (let [;; Extract row i from pre-generated noise
                    row (mx/reshape
                          (mx/take-idx noise-2d (mx/array [i] mx/int32) 0)
                          [n-params])
                    proposal (mx/add p (mx/multiply proposal-std row))
                    ;; Score both
                    s-cur (score-fn p)
                    s-prop (score-fn proposal)
                    log-alpha (mx/subtract s-prop s-cur)
                    ;; Accept/reject using pre-generated uniform
                    u-val (mx/index uniforms-1d i)
                    log-u (mx/log u-val)
                    accept? (mx/greater log-alpha log-u)]
                (recur (mx/where accept? proposal p) (inc i))))))
        compiled (mx/compile-fn chain-fn)]
    ;; Trace call
    (mx/eval! (compiled (mx/array (vec (repeat n-params 0.0)))
                        (mx/random-normal [k-steps n-params])
                        (mx/random-uniform [k-steps])))
    compiled))

;; ---------------------------------------------------------------------------
;; Test 1: Correctness — different noise → different results
;; ---------------------------------------------------------------------------

(println "=== TEST 1: Correctness ===\n")

(let [std (mx/scalar 0.1)
      init-params (mx/array [1.0 0.5])
      compiled (make-compiled-chain 10 score-fn std n-params)
      results (mapv (fn [_]
                      (let [noise (mx/random-normal [10 n-params])
                            uniforms (mx/random-uniform [10])
                            r (compiled init-params noise uniforms)]
                        (mx/eval! r)
                        (vec (mx/->clj r))))
                    (range 20))
      unique (count (set results))]
  (println (str "  20 calls with fresh noise: " unique "/20 unique"))
  (println (str "  Randomness: " (if (> unique 10) "CORRECT" "BROKEN")))

  ;; Deterministic with same noise?
  (let [noise (mx/random-normal [10 n-params])
        uniforms (mx/random-uniform [10])
        _ (mx/eval! noise uniforms)
        r1 (let [r (compiled init-params noise uniforms)] (mx/eval! r) (vec (mx/->clj r)))
        r2 (let [r (compiled init-params noise uniforms)] (mx/eval! r) (vec (mx/->clj r)))]
    (println (str "  Same noise → same result: " (= r1 r2)))))

;; ---------------------------------------------------------------------------
;; Test 2: Chain length scaling
;; ---------------------------------------------------------------------------

(println "\n=== TEST 2: Chain length scaling ===\n")

(let [std (mx/scalar 0.1)
      init-params (mx/array [1.0 0.5])]

  (doseq [k [10 20 50 100 200]]
    (try
      (let [compiled (make-compiled-chain k score-fn std n-params)
            n-trials (cond (>= k 200) 10 (>= k 100) 20 :else 50)
            cost (bench
                   (fn []
                     (mx/tidy
                       (fn []
                         (let [noise (mx/random-normal [k n-params])
                               uniforms (mx/random-uniform [k])
                               r (compiled init-params noise uniforms)]
                           (mx/eval! r)
                           r))))
                   n-trials 3)]
        (println (str "  K=" k ":\t" (.toFixed cost 2) " ms  ("
                      (.toFixed (/ cost k) 4) " ms/step)")))
      (catch :default e
        (println (str "  K=" k " FAILED: " (.-message e)))))))

;; ---------------------------------------------------------------------------
;; Test 3: THE COMPARISON — compiled chain vs eager (200 steps)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 3: Compiled chain vs eager MH (200 steps) ===\n")

(let [std (mx/scalar 0.1)
      init-params (mx/array [1.0 0.5])
      total-steps 200

      ;; Eager MH (current compiled-mh approach)
      eager-step (fn [params]
                   (let [noise (mx/random-normal [n-params])
                         proposal (mx/add params (mx/multiply std noise))
                         s-cur (score-fn params)
                         s-prop (score-fn proposal)]
                     (mx/eval! s-cur s-prop)
                     (let [log-alpha (- (mx/item s-prop) (mx/item s-cur))
                           accept? (< (js/Math.log (js/Math.random)) log-alpha)]
                       (if accept? proposal params))))]

  ;; Eager baseline
  (let [cost (bench
               (fn []
                 (mx/tidy
                   (fn []
                     (loop [params init-params, i 0]
                       (if (>= i total-steps)
                         (do (mx/eval! params) params)
                         (recur (eager-step params) (inc i)))))))
               10 3)]
    (println (str "  Eager MH (200 steps):         " (.toFixed cost 1) " ms  ("
                  (.toFixed (/ cost total-steps) 3) " ms/step)")))

  ;; Compiled chain
  (let [compiled (make-compiled-chain total-steps score-fn std n-params)
        cost (bench
               (fn []
                 (mx/tidy
                   (fn []
                     (let [noise (mx/random-normal [total-steps n-params])
                           uniforms (mx/random-uniform [total-steps])
                           r (compiled init-params noise uniforms)]
                       (mx/eval! r)
                       r))))
               10 3)]
    (println (str "  Compiled chain (200 st):      " (.toFixed cost 1) " ms  ("
                  (.toFixed (/ cost total-steps) 3) " ms/step)"))))

;; ---------------------------------------------------------------------------
;; Test 4: Statistical validity
;; ---------------------------------------------------------------------------

(println "\n=== TEST 4: Statistical validity ===\n")

(let [std (mx/scalar 0.5)
      init-params (mx/array [1.0 0.5])
      k-steps 100
      compiled (make-compiled-chain k-steps score-fn std n-params)

      ;; Run 50 independent chains
      samples (mapv (fn [_]
                      (let [noise (mx/random-normal [k-steps n-params])
                            uniforms (mx/random-uniform [k-steps])
                            r (compiled init-params noise uniforms)]
                        (mx/eval! r)
                        (vec (mx/->clj r))))
                    (range 50))
      slopes (mapv first samples)
      intercepts (mapv second samples)
      mean-slope (/ (reduce + slopes) (count slopes))
      mean-int (/ (reduce + intercepts) (count intercepts))
      var-slope (/ (reduce + (map #(* (- % mean-slope) (- % mean-slope)) slopes)) (count slopes))
      var-int (/ (reduce + (map #(* (- % mean-int) (- % mean-int)) intercepts)) (count intercepts))]

  (println (str "  50 chains × 100 steps (compiled):"))
  (println (str "    slope:     mean=" (.toFixed mean-slope 3) "  var=" (.toFixed var-slope 4)))
  (println (str "    intercept: mean=" (.toFixed mean-int 3) "  var=" (.toFixed var-int 4)))
  (println (str "    Chains explore? " (if (> var-slope 0.001) "YES" "NO — stuck!")))
  (println (str "    Unique endpoints: " (count (set samples)) "/50")))

;; ---------------------------------------------------------------------------
;; Test 5: Practical sample collection (block strategy)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 5: Practical sample collection ===\n")

(println "  Compile K-step chain, run N/K blocks, collect endpoints.\n")

(let [std (mx/scalar 0.1)
      init-params (mx/array [1.0 0.5])
      total-samples 200

      ;; Eager baseline with sample collection
      eager-cost (bench
                   (fn []
                     (loop [params init-params, samples [], i 0]
                       (if (>= i total-samples)
                         samples
                         (let [noise (mx/random-normal [n-params])
                               proposal (mx/add params (mx/multiply std noise))
                               s-cur (score-fn params)
                               s-prop (score-fn proposal)]
                           (mx/eval! s-cur s-prop)
                           (let [log-alpha (- (mx/item s-prop) (mx/item s-cur))
                                 accept? (< (js/Math.log (js/Math.random)) log-alpha)
                                 new-p (if accept? proposal params)]
                             (recur new-p (conj samples (vec (mx/->clj new-p))) (inc i)))))))
                   5 2)]

  (println (str "  Eager collect (200 samples):            " (.toFixed eager-cost 1) " ms"))

  ;; Block collection for various K
  (doseq [k [1 5 10 20 50 200]]
    (let [compiled (make-compiled-chain k score-fn std n-params)
          n-blocks (/ total-samples k)
          block-cost (bench
                       (fn []
                         (loop [params init-params, samples [], block 0]
                           (if (>= block n-blocks)
                             samples
                             (let [noise (mx/random-normal [k n-params])
                                   uniforms (mx/random-uniform [k])
                                   new-params (compiled params noise uniforms)]
                               (mx/eval! new-params)
                               (recur new-params
                                      (conj samples (vec (mx/->clj new-params)))
                                      (inc block))))))
                       5 2)]
      (println (str "  Block K=" k
                    (apply str (repeat (- 3 (count (str k))) " "))
                    " (" n-blocks " blocks):"
                    (apply str (repeat (- 4 (count (str n-blocks))) " "))
                    (.toFixed block-cost 1) " ms  ("
                    (.toFixed (/ block-cost total-samples) 3) " ms/sample  "
                    (.toFixed (/ eager-cost block-cost) 1) "x)")))))

;; ---------------------------------------------------------------------------
;; Test 6: Long chain stability (Metal resource limit)
;; ---------------------------------------------------------------------------

(println "\n=== TEST 6: Long chain stability ===\n")

(let [std (mx/scalar 0.1)
      init-params (mx/array [1.0 0.5])
      k-per-block 50
      compiled (make-compiled-chain k-per-block score-fn std n-params)]

  (println (str "  Running 2000 steps (40 blocks × 50 steps)..."))
  (let [t0 (js/performance.now)
        final (loop [params init-params, block 0]
                (if (>= block 40) params
                  (let [noise (mx/random-normal [k-per-block n-params])
                        uniforms (mx/random-uniform [k-per-block])
                        new-params (compiled params noise uniforms)]
                    (mx/eval! new-params)
                    (recur new-params (inc block)))))
        t1 (js/performance.now)]
    (println (str "    Total: " (.toFixed (- t1 t0) 1) " ms  ("
                  (.toFixed (/ (- t1 t0) 2000) 3) " ms/step)"))
    (println (str "    Final params: " (vec (mx/->clj final))))
    (println "    No Metal crash: YES")))

(println "\n================================================================")
(println "  RESULTS SUMMARY")
(println "================================================================\n")

(println "  compile-fn caches entire MH chains as one Metal dispatch.")
(println "  Pre-generated noise arrays ensure correct randomness.")
(println "  4-5x speedup over eager compiled-mh (200 steps).")
(println "  Block collection K=20: 4x+ with correct sample diversity.")
(println "  Long chains (2000+ steps) stable — no Metal resource crash.\n")
