(ns agentmodels.ch01-intro
  "agentmodels.org Chapter 1 — \"Introduction / Probabilistic programming\" ported
   to GenMLX. The textbook opens by teaching that a probabilistic program is just
   ordinary code in which `flip`/`sample` draw random choices, and `Infer` turns
   the program into a distribution over its return value. This file ports the two
   canonical ch1 vignettes to the GenMLX GFI:

   1. The COIN-FLIP TASTER — `dist/flip` is the GenMLX `flip()`. We `p/simulate`
      the one-site model three times and map 0/1 -> T/H only at the print boundary
      (values stay MLX arrays inside the model, per the GenMLX discipline).

   2. GEOMETRIC TWO WAYS — the recursive `geometric` of ch1
      (`flip() ? 1 + geometric() : 0`, count the consecutive successes before the
      first failure) shown as (a) a hand-rolled RECURSIVE gen function made exact
      by enumeration, and (b) the idiomatic built-in `dist/geometric`. Both give
      the same PMF P(n=k) = 0.5^(k+1) and E[n] = 1, demonstrating that an explicit
      model and a primitive agree.

      WebPPL bounds the recursion with `maxExecutions`; for EXACT enumeration we
      bound it with an explicit `max-depth` argument instead (the all-continue
      tail lumps into n = max-depth, mass 0.5^max-depth). With max-depth = 12 the
      tail is ~2e-4, so P(n=0..3) match the closed form to float32.

   Reuse, zero engine change: genmlx.dist (flip / geometric), genmlx.inference.exact
   (exact-joint enumeration), the GFI (p/simulate, p/assess). Self-checking: run

     bun run --bun nbb examples/agentmodels/ch01_intro.cljs

   prints each marginal and asserts it against the analytic reference number;
   exits non-zero if any check fails."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.exact :as exact]
            [genmlx.protocols :as p]
            [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; tiny self-check harness (examples are runnable + self-asserting; no test fw)
;; ---------------------------------------------------------------------------

(def ^:private fails (atom 0))

(defn- ->num [v] (if (mx/array? v) (mx/item v) v))

(defn- check-close [label expected actual tol]
  (let [a  (->num actual)
        ok (<= (abs (- expected a)) tol)]
    (println (str (if ok "  ✓ " "  ✗ FAIL ") label
                  " — expected " expected ", got " (.toFixed a 6)))
    (when-not ok (swap! fails inc))
    ok))

(defn- check-true [label ok]
  (println (str (if ok "  ✓ " "  ✗ FAIL ") label))
  (when-not ok (swap! fails inc))
  ok)

;; ---------------------------------------------------------------------------
;; 1. Coin-flip taster — dist/flip is the GenMLX flip()
;; ---------------------------------------------------------------------------

(def coin
  "A one-site probabilistic program: a single fair coin. Its return value IS the
   drawn choice, so `Infer`-ing it (here: forward p/simulate) is a distribution
   over {0,1}."
  (gen [] (trace :c (dist/flip 0.5))))

(defn flip->HT
  "Map the 0/1 draw to T/H at the PRINT boundary only (values stay MLX inside)."
  [v]
  (if (= 1 (int (->num v))) "H" "T"))

(defn coin-taster
  "Three forward draws from `coin`, each as T/H. Returns the vector of faces."
  []
  (mapv (fn [_]
          (-> (p/simulate (dyn/auto-key coin) []) :retval flip->HT))
        (range 3)))

;; ---------------------------------------------------------------------------
;; 2a. Geometric, way one: a recursive gen function made exact by enumeration
;; ---------------------------------------------------------------------------
;;
;; agentmodels ch1: geometric = function(){ return flip() ? 1 + geometric() : 0 }
;; — at each step continue with prob 0.5, and n counts the consecutive continues
;; before the first stop. WebPPL caps the unbounded recursion with maxExecutions;
;; for exact tensor enumeration we cap it with an explicit `max-depth`. Encoded
;; without host control flow (which enumeration cannot branch on): n is the sum
;; of the cumulative product of the continue-flips, i.e. the length of the leading
;; run of 1s. This is a fixed-depth unroll, so every flip is a distinct enumeration
;; axis and `reached`/`n` broadcast across the full joint support.

(defn geometric-recursive
  "Depth-bounded recursive geometric as a gen function. n = number of consecutive
   `continue`s (fair flips landing 1) before the first stop; capped at max-depth."
  [max-depth]
  (gen []
    (loop [i 0, reached (mx/scalar 1.0), n (mx/scalar 0.0)]
      (if (>= i max-depth)
        n
        (let [c        (trace (keyword (str "c" i)) (dist/flip 0.5))
              ;; reached stays 1.0 only while every earlier flip continued
              continue (mx/multiply reached (.astype c mx/float32))]
          (recur (inc i) continue (mx/add n continue)))))))

(defn geometric-pmf-recursive
  "Exactly enumerate the recursive geometric and read off P(n=0..3) and E[n] by
   probability-weighting the (tensor-shaped) return value over the full joint."
  [max-depth]
  (let [{:keys [probs retval]} (exact/exact-joint (geometric-recursive max-depth) [] nil)
        rv (.astype retval mx/float32)
        pk (fn [k]
             (mx/item (mx/sum (mx/multiply probs
                                (.astype (mx/equal rv (mx/scalar (* 1.0 k))) mx/float32)))))]
    {:pmf  (mapv pk (range 4))
     :mean (mx/item (mx/sum (mx/multiply probs rv)))}))

;; ---------------------------------------------------------------------------
;; 2b. Geometric, way two: the idiomatic built-in dist/geometric
;; ---------------------------------------------------------------------------

(def geometric-builtin
  "The same distribution as a single primitive trace site."
  (gen [] (trace :n (dist/geometric 0.5))))

(defn geometric-pmf-builtin
  "Exact P(n=0..3) read directly from the primitive's log-prob via p/assess —
   no enumeration/normalization, so these are the exact 0.5^(k+1)."
  []
  (mapv (fn [k]
          (let [{:keys [weight]} (p/assess (dyn/auto-key geometric-builtin) []
                                           (cm/choicemap :n (mx/scalar k)))]
            (js/Math.exp (->num weight))))
        (range 4)))

;; ---------------------------------------------------------------------------
;; run + self-check
;; ---------------------------------------------------------------------------

(defn -main []
  (println "\n=== agentmodels.org Ch 1 — Introduction (GenMLX port) ===")

  (println "\n-- 1. coin-flip taster: three forward draws of dist/flip --")
  (let [faces (coin-taster)]
    (println (str "  faces: " faces))
    (check-true "each face is H or T" (every? #{"H" "T"} faces)))

  (println "\n-- 2a. geometric (recursive gen, exact enumeration; depth 12) --")
  (let [{:keys [pmf mean]} (geometric-pmf-recursive 12)]
    (doseq [k (range 4)]
      (check-close (str "P(n=" k ") = 0.5^" (inc k)) (js/Math.pow 0.5 (inc k)) (nth pmf k) 1e-5))
    (check-close "E[n] = 1" 1.0 mean 5e-3))

  (println "\n-- 2b. geometric (idiomatic dist/geometric, exact via p/assess) --")
  (let [pmf (geometric-pmf-builtin)]
    (doseq [k (range 4)]
      (check-close (str "P(n=" k ")") (js/Math.pow 0.5 (inc k)) (nth pmf k) 1e-5)))

  (println (str "\n" (if (zero? @fails)
                       "ALL CHECKS PASSED ✓"
                       (str @fails " CHECK(S) FAILED ✗"))))
  (when (pos? @fails) (js/process.exit 1)))

(-main)
