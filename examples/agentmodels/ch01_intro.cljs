(ns agentmodels.ch01-intro
  "agentmodels.org Chapter 1 — \"Introduction / Probabilistic programming\" ported
   to GenMLX: the coin-flip taster (`dist/flip`) and geometric two ways (a
   recursive gen function made exact by enumeration, and the built-in
   `dist/geometric`), both yielding P(n=k) = 0.5^(k+1) and E[n] = 1.

   Run: bun run --bun nbb examples/agentmodels/ch01_intro.cljs"
  ;; WebPPL bounds the recursive geometric with `maxExecutions`; for EXACT tensor
  ;; enumeration we bound it with an explicit `max-depth` instead (the all-continue
  ;; tail lumps into n = max-depth, mass 0.5^max-depth). With max-depth = 12 the
  ;; tail is ~2e-4, so P(n=0..3) match the closed form to float32. n is encoded
  ;; without host control flow (enumeration cannot branch on a value): it is the
  ;; length of the leading run of 1s, computed as a cumulative product of the
  ;; continue-flips broadcast across the full joint support.
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.exact :as exact]
            [genmlx.protocols :as p]
            [genmlx.gen :refer [gen]]
            [agentmodels.harness :as chk]))

;; mx/item does NOT accept plain JS numbers (it calls .item directly), so guard.
(defn- mx-val [v] (if (mx/array? v) (mx/item v) v))

;; ---------------------------------------------------------------------------
;; 1. Coin-flip taster — dist/flip is the GenMLX flip()
;; ---------------------------------------------------------------------------

(def coin
  "A one-site probabilistic program: a single fair coin. Its return value IS the
   drawn choice, so `Infer`-ing it (here: forward p/simulate) is a distribution
   over {0,1}."
  (gen [] (trace :c (dist/flip 0.5))))

(def ^:private coin* (dyn/auto-key coin))

(defn flip->HT
  "Map the 0/1 draw to T/H at the PRINT boundary only (values stay MLX inside)."
  [v]
  (if (= 1 (int (mx-val v))) "H" "T"))

(defn coin-taster
  "Three forward draws from `coin`, each as T/H. Returns the vector of faces."
  []
  (mapv (fn [_]
          (-> (p/simulate coin* []) :retval flip->HT))
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
              continue (mx/multiply reached (mx/astype c mx/float32))]
          (recur (inc i) continue (mx/add n continue)))))))

(defn geometric-pmf-recursive
  "Exactly enumerate the recursive geometric and read off P(n=0..3) and E[n] by
   probability-weighting the (tensor-shaped) return value over the full joint."
  [max-depth]
  (let [{:keys [probs retval]} (exact/exact-joint (geometric-recursive max-depth) [] nil)
        rv (mx/astype retval mx/float32)
        pk (fn [k]
             (mx/item (mx/sum (mx/multiply probs
                                (mx/astype (mx/equal rv (mx/scalar (* 1.0 k))) mx/float32)))))]
    {:pmf  (mapv pk (range 4))
     :mean (mx/item (mx/sum (mx/multiply probs rv)))}))

;; ---------------------------------------------------------------------------
;; 2b. Geometric, way two: the idiomatic built-in dist/geometric
;; ---------------------------------------------------------------------------

(def geometric-builtin
  "The same distribution as a single primitive trace site."
  (gen [] (trace :n (dist/geometric 0.5))))

(def ^:private geometric-builtin* (dyn/auto-key geometric-builtin))

(defn geometric-pmf-builtin
  "Exact P(n=0..3) read directly from the primitive's log-prob via p/assess —
   no enumeration/normalization, so these are the exact 0.5^(k+1)."
  []
  (mapv (fn [k]
          (let [{:keys [weight]} (p/assess geometric-builtin* []
                                           (cm/choicemap :n (mx/scalar k)))]
            (Math/exp (mx-val weight))))
        (range 4)))

;; ---------------------------------------------------------------------------
;; run + self-check
;; ---------------------------------------------------------------------------

(defn -main []
  (println "\n=== agentmodels.org Ch 1 — Introduction (GenMLX port) ===")

  (println "\n-- 1. coin-flip taster: three forward draws of dist/flip --")
  (let [faces (coin-taster)]
    (println (str "  faces: " faces))
    (chk/check-true "each face is H or T" (every? #{"H" "T"} faces)))

  (println "\n-- 2a. geometric (recursive gen, exact enumeration; depth 12) --")
  (let [{:keys [pmf mean]} (geometric-pmf-recursive 12)]
    (doseq [k (range 4)]
      (chk/check-close (str "P(n=" k ") = 0.5^" (inc k)) (Math/pow 0.5 (inc k)) (nth pmf k) 1e-5))
    (chk/check-close "E[n] = 1" 1.0 mean 5e-3))

  (println "\n-- 2b. geometric (idiomatic dist/geometric, exact via p/assess) --")
  (let [pmf (geometric-pmf-builtin)]
    (doseq [k (range 4)]
      (chk/check-close (str "P(n=" k ")") (Math/pow 0.5 (inc k)) (nth pmf k) 1e-5)))

  (chk/report!))

(-main)
