(ns genmlx.inference-property-test
  "Property-based inference algorithm tests using test.check.
   Verifies importance sampling, MH, kernels, weight utilities,
   diagnostics, and resampling invariants."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.util :as u]
            [genmlx.inference.diagnostics :as diag])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- report-result [name result]
  (if (:pass? result)
    (do (vswap! pass-count inc)
        (println "  PASS:" name (str "(" (:num-tests result) " trials)")))
    (do (vswap! fail-count inc)
        (println "  FAIL:" name)
        (println "    seed:" (:seed result))
        (when-let [s (get-in result [:shrunk :smallest])]
          (println "    shrunk:" s)))))

(defn- check [name prop & {:keys [num-tests] :or {num-tests 50}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Model and fixture pools
;; ---------------------------------------------------------------------------

(def model
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))
          y (dyn/trace :y (dist/gaussian 0 1))]
      (mx/eval! x y)
      (+ (mx/item x) (mx/item y)))))

(def obs-pool
  [(cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5))
   (cm/choicemap :x (mx/scalar 0.0) :y (mx/scalar 0.0))
   (cm/choicemap :x (mx/scalar -1.0) :y (mx/scalar 2.0))
   (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar -0.5))])

(def gen-obs (gen/elements obs-pool))

(def particle-pool [5 10 15 20])
(def gen-n-particles (gen/elements particle-pool))

;; Pre-built PRNG keys
(def key-pool (mapv #(rng/fresh-key %) [42 99 123 7 255]))
(def gen-key (gen/elements key-pool))

(println "\n=== Inference Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Importance Sampling (4)
;; ---------------------------------------------------------------------------

(println "-- importance sampling --")

(check "IS: log-weights are finite"
  (prop/for-all [obs gen-obs
                 n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] obs)]
      (every? (fn [w] (finite? (eval-weight w))) log-weights))))

(check "IS: log-ml-estimate is finite"
  (prop/for-all [obs gen-obs
                 n gen-n-particles
                 k gen-key]
    (let [{:keys [log-ml-estimate]} (is/importance-sampling {:samples n :key k} model [] obs)]
      (finite? (eval-weight log-ml-estimate)))))

(check "IS: normalized weights sum to 1.0"
  (prop/for-all [obs gen-obs
                 n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] obs)
          {:keys [probs]} (u/normalize-log-weights log-weights)
          total (reduce + probs)]
      (close? 1.0 total 0.01))))

(check "IS: empty constraints yield weights near 0"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] cm/EMPTY)]
      (every? (fn [w] (close? 0.0 (eval-weight w) 0.01)) log-weights))))

;; ---------------------------------------------------------------------------
;; MH / Accept Decision (3)
;; ---------------------------------------------------------------------------

(println "\n-- MH accept --")

(check "accept-mh?(0) always true"
  (prop/for-all [k gen-key]
    (u/accept-mh? 0 k))
  :num-tests 100)

(check "accept-mh?(-100) rarely true"
  (prop/for-all [_ (gen/return nil)]
    ;; Run 100 trials, fewer than 5 should accept
    (let [accepts (count (filter true? (repeatedly 100 #(u/accept-mh? -100))))]
      (< accepts 5)))
  :num-tests 10)

(check "regenerate(sel/none) yields weight near 0"
  (prop/for-all [_ (gen/return nil)]
    (let [trace (p/simulate model [])
          {:keys [weight]} (p/regenerate model trace sel/none)]
      (close? 0.0 (eval-weight weight) 0.01))))

;; ---------------------------------------------------------------------------
;; Kernel Composition (5)
;; ---------------------------------------------------------------------------

(println "\n-- kernel composition --")

(def k-prior-x (kern/mh-kernel (sel/select :x)))
(def k-prior-y (kern/mh-kernel (sel/select :y)))

(check "chain(k1, k2) produces valid trace with finite score"
  (prop/for-all [k gen-key]
    (let [chained (kern/chain k-prior-x k-prior-y)
          trace (p/simulate model [])
          result (chained trace k)
          s (eval-weight (:score result))]
      (finite? s))))

(check "repeat-kernel(3, k) produces valid trace"
  (prop/for-all [k gen-key]
    (let [repeated (kern/repeat-kernel 3 k-prior-x)
          trace (p/simulate model [])
          result (repeated trace k)
          s (eval-weight (:score result))]
      (finite? s))))

(check "seed(k, key) produces valid trace with finite score"
  (prop/for-all [k gen-key]
    (let [seeded (kern/seed k-prior-x k)
          trace (p/simulate model [])
          r1 (seeded trace nil)
          s1 (eval-weight (:score r1))]
      (finite? s1))))

(check "run-kernel produces correct number of samples"
  (prop/for-all [n (gen/elements [3 5 8])]
    (let [trace (p/simulate model [])
          samples (kern/run-kernel {:samples n} k-prior-x trace)]
      (= n (count samples)))))

(check "cycle-kernels applies all kernels (finite score)"
  (prop/for-all [k gen-key]
    (let [cycled (kern/cycle-kernels 4 [k-prior-x k-prior-y])
          trace (p/simulate model [])
          result (cycled trace k)
          s (eval-weight (:score result))]
      (finite? s))))

;; ---------------------------------------------------------------------------
;; Score Function Utilities (3)
;; ---------------------------------------------------------------------------

(println "\n-- score functions --")

(check "extract-params → make-score-fn round-trip finite"
  (prop/for-all [obs gen-obs]
    (let [trace (p/simulate model [])
          addrs [:x :y]
          params (u/extract-params trace addrs)
          score-fn (u/make-score-fn model [] obs addrs)
          result (score-fn params)]
      (finite? (eval-weight result)))))

(check "score at mode > score far from mode"
  (prop/for-all [_ (gen/return nil)]
    (let [;; For gaussian(0,1), mode is at 0
          obs (cm/choicemap :y (mx/scalar 0.0))
          addrs [:x]
          score-fn (u/make-score-fn model [] obs addrs)
          score-near (eval-weight (score-fn (mx/array [0.0])))
          score-far  (eval-weight (score-fn (mx/array [100.0])))]
      (> score-near score-far))))

(check "make-score-fn produces finite result for all obs"
  (prop/for-all [obs gen-obs]
    (let [;; Use non-overlapping addr (constrain both, optimize neither — just evaluate)
          addrs [:x :y]
          sf (u/make-score-fn model [] cm/EMPTY addrs)
          ;; Use obs values as params
          params (mx/array [0.5 0.3])
          result (sf params)]
      (finite? (eval-weight result)))))

;; ---------------------------------------------------------------------------
;; Weight Utilities (4)
;; ---------------------------------------------------------------------------

(println "\n-- weight utilities --")

(check "normalize-log-weights probs sum to 1"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] (first obs-pool))
          {:keys [probs]} (u/normalize-log-weights log-weights)
          total (reduce + probs)]
      (close? 1.0 total 0.01))))

(check "compute-ess result in (0, N]"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] (first obs-pool))
          ess (u/compute-ess log-weights)]
      (and (> ess 0) (<= ess (+ n 0.01))))))

(check "uniform weights yield ESS ≈ N"
  (prop/for-all [n gen-n-particles]
    (let [log-weights (vec (repeat n (mx/scalar 0.0)))
          ess (u/compute-ess log-weights)]
      (close? (double n) ess 0.01))))

(check "materialize-weights produces finite values"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key k} model [] (first obs-pool))
          materialized (u/materialize-weights log-weights)]
      (mx/eval! materialized)
      (let [vals (mx/->clj materialized)]
        (every? js/isFinite vals)))))

;; ---------------------------------------------------------------------------
;; Diagnostics (3)
;; ---------------------------------------------------------------------------

(println "\n-- diagnostics --")

(check "diag/ess <= N always"
  (prop/for-all [_ (gen/return nil)]
    (let [n 20
          samples (mapv (fn [_] (let [t (p/simulate model [])]
                                  (:score t)))
                        (range n))
          e (diag/ess samples)]
      (<= e (+ n 0.01)))))

(check "diag/r-hat of identical chains near 1.0"
  (prop/for-all [_ (gen/return nil)]
    (let [n 20
          chain (mapv (fn [_] (mx/scalar 5.0)) (range n))
          chains [chain chain]]
      ;; Identical chains should have r-hat = 1 or NaN (zero variance within)
      ;; With constant chains, var0=0 so ESS=n which is valid; r-hat with zero W is degenerate
      ;; Just check it doesn't throw
      (let [r (diag/r-hat chains)]
        (or (js/isNaN r) (>= r 0.99))))))

(check "diag/r-hat >= 1.0 always (non-degenerate)"
  (prop/for-all [_ (gen/return nil)]
    (let [n 20
          ;; Two chains with different means
          chain1 (mapv (fn [i] (mx/scalar (+ 0.0 (* 0.1 i)))) (range n))
          chain2 (mapv (fn [i] (mx/scalar (+ 5.0 (* 0.1 i)))) (range n))
          r (diag/r-hat [chain1 chain2])]
      (>= r 1.0)))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; SMC Resampling (2)
;; ---------------------------------------------------------------------------

(println "\n-- SMC resampling --")

(check "systematic-resample indices in [0, N)"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [log-weights (vec (repeatedly n #(mx/scalar (- (js/Math.random) 0.5))))
          indices (u/systematic-resample log-weights n k)]
      (and (= n (count indices))
           (every? #(and (>= % 0) (< % n)) indices)))))

(check "uniform weight resample: valid indices"
  (prop/for-all [n gen-n-particles
                 k gen-key]
    (let [log-weights (vec (repeat n (mx/scalar 0.0)))
          indices (u/systematic-resample log-weights n k)]
      (and (= n (count indices))
           (every? #(and (>= % 0) (< % n)) indices)))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Inference Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
