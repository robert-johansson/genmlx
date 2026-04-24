(ns genmlx.temperature-sampling-test
  "Unit tests for temperature-scaled categorical sampling.
   No LLM required — tests the sampling primitives directly."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [label v]
  (if v
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc) (println (str "  FAIL: " label)))))

(defn- assert-equal [label expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc) (println (str "  FAIL: " label " expected=" expected " actual=" actual)))))

;; ---------------------------------------------------------------------------
;; 1. Greedy determinism
;; ---------------------------------------------------------------------------
(println "\n== 1. Greedy determinism ==")

(let [logits (mx/array [1.0 5.0 2.0 0.5])]
  (dotimes [_ 5]
    (let [idx (mx/item (mx/argmax logits))]
      (assert-equal "argmax of [1 5 2 0.5] is index 1" 1 idx))))

;; ---------------------------------------------------------------------------
;; 2. Categorical with temperature=1.0 (peaked logits)
;; ---------------------------------------------------------------------------
(println "\n== 2. Categorical with temperature=1.0 (peaked logits) ==")

(let [logits (mx/array [0 0 0 10])
      temp (mx/scalar 1.0)
      scaled (mx/multiply logits (mx/scalar (/ 1.0 1.0)))
      rk (rng/fresh-key 123)
      results (loop [i 0, acc [], rk rk]
                (if (>= i 100)
                  acc
                  (let [[sk nk] (rng/split rk)
                        idx (mx/item (rng/categorical sk scaled))]
                    (recur (inc i) (conj acc idx) nk))))
      count-3 (count (filter #(= 3 %) results))]
  (println (str "  Index 3 chosen " count-3 "/100 times"))
  (assert-true (str "index 3 chosen >90% (got " count-3 ")") (> count-3 90)))

;; ---------------------------------------------------------------------------
;; 3. Temperature diversity
;; ---------------------------------------------------------------------------
(println "\n== 3. Temperature diversity ==")

(let [logits (mx/array [1 2 3 4])]
  ;; 3a: temperature=1.0 should produce diversity
  (let [inv-temp (mx/scalar (/ 1.0 1.0))
        scaled (mx/multiply logits inv-temp)
        rk (rng/fresh-key 456)
        results (loop [i 0, acc [], rk rk]
                  (if (>= i 200)
                    acc
                    (let [[sk nk] (rng/split rk)
                          idx (mx/item (rng/categorical sk scaled))]
                      (recur (inc i) (conj acc idx) nk))))
        unique (count (distinct results))]
    (println (str "  T=1.0: " unique " unique categories from " (frequencies results)))
    (assert-true (str "at least 3 categories at T=1.0 (got " unique ")") (>= unique 3)))

  ;; 3b: temperature=0.01 should collapse to argmax
  (let [inv-temp (mx/scalar (/ 1.0 0.01))
        scaled (mx/multiply logits inv-temp)
        rk (rng/fresh-key 789)
        results (loop [i 0, acc [], rk rk]
                  (if (>= i 200)
                    acc
                    (let [[sk nk] (rng/split rk)
                          idx (mx/item (rng/categorical sk scaled))]
                      (recur (inc i) (conj acc idx) nk))))
        count-3 (count (filter #(= 3 %) results))]
    (println (str "  T=0.01: index 3 chosen " count-3 "/200 times"))
    (assert-true (str "nearly all argmax at T=0.01 (got " count-3 "/200)") (>= count-3 195))))

;; ---------------------------------------------------------------------------
;; 4. Temperature scaling math (equal logits)
;; ---------------------------------------------------------------------------
(println "\n== 4. Temperature scaling math (equal logits) ==")

(let [logits (mx/array [0 0])
      inv-temp (mx/scalar (/ 1.0 1.0))
      scaled (mx/multiply logits inv-temp)
      rk (rng/fresh-key 321)
      results (loop [i 0, acc [], rk rk]
                (if (>= i 200)
                  acc
                  (let [[sk nk] (rng/split rk)
                        idx (mx/item (rng/categorical sk scaled))]
                    (recur (inc i) (conj acc idx) nk))))
      count-0 (count (filter #(= 0 %) results))
      count-1 (count (filter #(= 1 %) results))]
  (println (str "  Equal logits: index 0=" count-0 " index 1=" count-1))
  (assert-true (str "index 0 between 35-65% (got " count-0 "/200)") (and (>= count-0 70) (<= count-0 130)))
  (assert-true (str "index 1 between 35-65% (got " count-1 "/200)") (and (>= count-1 70) (<= count-1 130))))

;; ---------------------------------------------------------------------------
;; 5. Numerical stability (extreme logits)
;; ---------------------------------------------------------------------------
(println "\n== 5. Numerical stability (extreme logits) ==")

(let [logits (mx/array [-1000 0 1000])
      inv-temp (mx/scalar (/ 1.0 1.0))
      scaled (mx/multiply logits inv-temp)
      rk (rng/fresh-key 999)
      [sk _] (rng/split rk)
      idx (mx/item (rng/categorical sk scaled))]
  (assert-true "no NaN from extreme logits" (number? idx))
  (assert-true "not NaN" (not (js/isNaN idx)))
  (assert-equal "extreme logits -> index 2" 2 idx))

;; ---------------------------------------------------------------------------
;; 6. Seed reproducibility
;; ---------------------------------------------------------------------------
(println "\n== 6. Seed reproducibility ==")

(let [logits (mx/array [1 2 3 4])
      inv-temp (mx/scalar (/ 1.0 0.8))
      scaled (mx/multiply logits inv-temp)
      sample-seq (fn [seed]
                   (loop [i 0, acc [], rk (rng/fresh-key seed)]
                     (if (>= i 10)
                       acc
                       (let [[sk nk] (rng/split rk)
                             idx (mx/item (rng/categorical sk scaled))]
                         (recur (inc i) (conj acc idx) nk)))))
      seq1 (sample-seq 42)
      seq2 (sample-seq 42)
      seq3 (sample-seq 99)]
  (println (str "  Seed 42 run 1: " seq1))
  (println (str "  Seed 42 run 2: " seq2))
  (println (str "  Seed 99 run 1: " seq3))
  (assert-equal "same seed -> same sequence" seq1 seq2)
  (assert-true "different seed -> likely different sequence" (not= seq1 seq3)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------
(println (str "\n== Summary: " @pass-count " passed, " @fail-count " failed =="))
(when (pos? @fail-count)
  (js/process.exit 1))
