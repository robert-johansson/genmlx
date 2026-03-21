(ns rsa
  "Rational Speech Acts (RSA) — recursive pragmatic reasoning.

   A speaker wants to communicate about one of three referents:
     green_square (gs), green_circle (gc), pink_circle (pc)
   using one of four utterances:
     green, pink, square, round

   RSA recursively defines:
     L0: literal listener (uniform prior, filter by denotation)
     S1: pragmatic speaker (soft-max over L0)
     L1: pragmatic listener (Bayes over S1)
     ... and so on

   GenMLX computes every depth via exact enumeration — no sampling.
   Depth iteration uses Clojure's `iterate` — no cache atoms, just a lazy
   sequence of improving listener tables."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---- Domain setup ----
;; Utterances: green=0, pink=1, square=2, round=3
;; Referents:  green_sq=0, green_ci=1, pink_ci=2
;;
;; denotes(u, r) = 1 if utterance u truthfully describes referent r
;; Shape [3, 4]: row=referent, col=utterance

(def denotes
  (.astype (mx/array #js [#js [1 0 1 0]    ;; gs: green, square
                          #js [1 0 0 1]    ;; gc: green, round
                          #js [0 1 0 1]])  ;; pc: pink, round
           mx/float32))

;; Uniform prior over 3 referents (log-space)
(def uniform-3 (mx/log (mx/array #js [1/3 1/3 1/3])))

;; ---- RSA model ----

(defn make-rsa-model
  "Build an RSA listener model.
   L-prev: [4, 3] probability table from previous depth (nil for depth 0)."
  [beta L-prev]
  (gen []
    (let [r-true (trace :r-true (dist/categorical uniform-3))
          wpp (if (nil? L-prev)
                denotes
                (mx/multiply denotes
                  (mx/exp (mx/multiply (mx/scalar beta)
                                       (mx/transpose L-prev)))))
          u-said (trace :u-said (dist/categorical (mx/log wpp)))]
      r-true)))

;; ---- Depth iteration via iterate ----

(defn rsa-step
  "One RSA depth step. Pure function: L-prev table → L table.
   Given the previous listener table (or nil for depth 0), builds the model,
   runs exact enumeration, and returns the next listener table [4, 3]."
  [beta L-prev]
  (let [model (make-rsa-model beta L-prev)
        joint (exact/exact-joint model [] nil)]
    (exact/extract-table joint :u-said)))

(defn rsa-L
  "Compute RSA listener table P(r | u) at given depth.
   Returns [4, 3] tensor (rows=utterances, cols=referents).

   Pure functional — no mutation, just Clojure's `iterate`."
  [beta max-depth]
  (->> (rsa-step beta nil)
       (iterate (partial rsa-step beta))
       (drop max-depth)
       first))

(defn rsa-row
  "Extract row u from the [4, 3] RSA table as a vector of JS numbers."
  [table u]
  (let [row (mx/idx table u)]
    (mx/eval! row)
    (mapv #(mx/item (mx/slice row % (inc %))) (range 3))))

;; ---- Compute and display results ----

(println "Rational Speech Acts (RSA) — exact enumeration\n")
(println "Referents: gs=green_square  gc=green_circle  pc=pink_circle")
(println "Utterances: green  pink  square  round\n")

(defn print-table [depth]
  (let [table (rsa-L 1.0 depth)
        labels ["green " "pink  " "square" "round "]]
    (println (str "Depth " depth " — P(referent | utterance):"))
    (println "  utterance     gs       gc       pc")
    (doseq [u (range 4)]
      (let [[gs gc pc] (rsa-row table u)]
        (println (str "  " (nth labels u) "    "
                      (.toFixed gs 4) "   "
                      (.toFixed gc 4) "   "
                      (.toFixed pc 4)))))
    (println)))

(doseq [d (range 4)]
  (print-table d))

;; ---- Verification against memo reference values ----

(println "Verification against memo reference values:")

(defn approx= [a b tol] (<= (abs (- a b)) tol))

(defn check [name pred]
  (if pred
    (println (str "  PASS: " name))
    (println (str "  FAIL: " name))))

(defn check-close [name expected actual tol]
  (check (str name " (" (.toFixed actual 5) " ~ " expected ")")
         (approx= expected actual tol)))

;; Depth 0: literal listener
;; memo ref: green=[0.5, 0.5, 0], pink=[0, 0, 1], square=[1, 0, 0], round=[0, 0.5, 0.5]
(let [L0 (rsa-L 1.0 0)]
  (check-close "L0 green->gs" 0.5 (nth (rsa-row L0 0) 0) 1e-5)
  (check-close "L0 green->gc" 0.5 (nth (rsa-row L0 0) 1) 1e-5)
  (check-close "L0 green->pc" 0.0 (nth (rsa-row L0 0) 2) 1e-5)
  (check-close "L0 pink->pc"  1.0 (nth (rsa-row L0 1) 2) 1e-5)
  (check-close "L0 square->gs" 1.0 (nth (rsa-row L0 2) 0) 1e-5)
  (check-close "L0 round->gc" 0.5 (nth (rsa-row L0 3) 1) 1e-5)
  (check-close "L0 round->pc" 0.5 (nth (rsa-row L0 3) 2) 1e-5))

;; Depth 1: pragmatic listener
;; memo ref: green=[0.4302, 0.5698, 0], round=[0, 0.5698, 0.4302]
(let [L1 (rsa-L 1.0 1)]
  (check-close "L1 green->gs" 0.43022 (nth (rsa-row L1 0) 0) 1e-4)
  (check-close "L1 green->gc" 0.56977 (nth (rsa-row L1 0) 1) 1e-4)
  (check-close "L1 green->pc" 0.0     (nth (rsa-row L1 0) 2) 1e-5)
  (check-close "L1 pink->pc"  1.0     (nth (rsa-row L1 1) 2) 1e-5)
  (check-close "L1 round->gc" 0.56977 (nth (rsa-row L1 3) 1) 1e-4)
  (check-close "L1 round->pc" 0.43022 (nth (rsa-row L1 3) 2) 1e-4))

;; Depths 2-3: convergence
;; memo ref: L2 green=[0.4195, 0.5805, 0], L3 green=[0.4178, 0.5822, 0]
(let [L2 (rsa-L 1.0 2)
      L3 (rsa-L 1.0 3)]
  (check-close "L2 green->gs" 0.41947 (nth (rsa-row L2 0) 0) 1e-4)
  (check-close "L2 green->gc" 0.58053 (nth (rsa-row L2 0) 1) 1e-4)
  (check-close "L3 green->gs" 0.41780 (nth (rsa-row L3 0) 0) 1e-4)
  (check-close "L3 green->gc" 0.58220 (nth (rsa-row L3 0) 1) 1e-4)
  ;; Convergence: |d3-d2| < |d2-d1|
  (let [d1-gc (nth (rsa-row (rsa-L 1.0 1) 0) 1)
        d2-gc (nth (rsa-row L2 0) 1)
        d3-gc (nth (rsa-row L3 0) 1)]
    (check "convergence: |d3-d2| < |d2-d1|"
           (< (abs (- d3-gc d2-gc)) (abs (- d2-gc d1-gc))))))

(println "\nDone.")
