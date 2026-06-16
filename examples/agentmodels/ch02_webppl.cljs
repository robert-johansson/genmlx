(ns agentmodels.ch02-webppl
  "agentmodels.org Chapter 2 — \"WebPPL: a probabilistic programming language\"
   ported to GenMLX. Ch2 is the language primer: elementary random primitives
   (ERPs), conditioning, and `Infer`. This file ports its canonical vignettes to
   the GenMLX GFI, using exact tensor enumeration where the textbook uses
   `Infer({method:'enumerate'})`.

   Vignettes:
   1. ERP DRAWS — forward p/simulate of discrete (bernoulli, categorical) and
      continuous (gaussian, uniform) primitives; values map to readable output
      only at the print boundary.
   2. MULTIVARIATE GAUSSIAN — dist/multivariate-normal (the GenMLX
      `multivariateGaussian`), a 2-D draw with the given mean/covariance.
   3. BINOMIAL FROM THREE FLIPS — `binomial = flip()+flip()+flip()`, with the
      distribution over the SUM obtained two ways and shown to agree: by
      enumerating the three-flip model and probability-weighting its return
      value, and by the primitive dist/binomial. PMF = [1 3 3 1]/8.
   4. twoHeads / moreThanTwoHeads — CONDITIONING via the bernoulli-indicator
      trick + exact-posterior. P(first flip = H | total heads >= 2) = 0.75;
      evidence P(total heads >= 2) = 0.5; posterior over the total given >=2 is
      {2: 0.75, 3: 0.25}.
   5. STRUCTURED RETURNS + a forward positionDist — a model returning a joint
      structure, and a small discrete random-walk position distribution
      enumerated exactly.

   Reuse, zero engine change: genmlx.dist (bernoulli/categorical/gaussian/uniform/
   multivariate-normal/binomial), genmlx.inference.exact (exact-joint /
   exact-posterior), the GFI (p/simulate, p/assess). Self-checking: run

     bun run --bun nbb examples/agentmodels/ch02_webppl.cljs

   prints each result and asserts deterministic (enumerated) quantities against
   the analytic reference numbers; exits non-zero if any check fails. Random
   forward draws are asserted structurally (shape / support), never by value."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.exact :as exact]
            [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; tiny self-check harness
;; ---------------------------------------------------------------------------

(def ^:private fails (atom 0))

(defn- ->num [v] (if (mx/array? v) (mx/item v) v))

(defn- check-close [label expected actual tol]
  (let [a  (->num actual)
        ok (<= (js/Math.abs (- expected a)) tol)]
    (println (str (if ok "  ✓ " "  ✗ FAIL ") label
                  " — expected " expected ", got " (.toFixed a 6)))
    (when-not ok (swap! fails inc))
    ok))

(defn- check-true [label ok]
  (println (str (if ok "  ✓ " "  ✗ FAIL ") label))
  (when-not ok (swap! fails inc))
  ok)

;; ---------------------------------------------------------------------------
;; 1. ERP draws — forward sampling of discrete + continuous primitives
;; ---------------------------------------------------------------------------

(def erp-model
  "A handful of ERPs in one program, returned as a structured map. Forward
   p/simulate draws all of them jointly."
  (gen []
    (let [b (trace :bern (dist/bernoulli 0.5))
          c (trace :cat  (dist/categorical (mx/array [0.0 0.0 0.0]))) ; uniform over 3
          g (trace :norm (dist/gaussian 0.0 1.0))
          u (trace :unif (dist/uniform 0.0 1.0))]
      {:bern b :cat c :norm g :unif u})))

(defn erp-draw
  "One forward draw of the ERP bundle, as plain numbers (print boundary)."
  []
  (let [rv (dyn/call erp-model)]
    {:bern (int (->num (:bern rv)))
     :cat  (int (->num (:cat rv)))
     :norm (->num (:norm rv))
     :unif (->num (:unif rv))}))

;; ---------------------------------------------------------------------------
;; 2. multivariateGaussian — a 2-D draw
;; ---------------------------------------------------------------------------

(def mvn-model
  (gen []
    (trace :pos (dist/multivariate-normal (mx/array [0.0 0.0])
                                          (mx/array [[1.0 0.0] [0.0 1.0]])))))

(defn mvn-draw []
  (mx/->clj (dyn/call mvn-model)))

;; ---------------------------------------------------------------------------
;; 3. binomial from three flips — distribution over the SUM, two ways
;; ---------------------------------------------------------------------------

(def three-flips-sum
  "binomial = flip() + flip() + flip(); the program RETURNS the sum, so the
   distribution over the return value is Binomial(3, 0.5)."
  (gen []
    (let [a (trace :a (dist/flip 0.5))
          b (trace :b (dist/flip 0.5))
          c (trace :c (dist/flip 0.5))]
      (reduce mx/add (mapv #(mx/astype % mx/float32) [a b c])))))

(defn binomial-pmf-from-flips
  "Enumerate the three-flip program and read P(sum = 0..3) by probability-
   weighting the (tensor-shaped) return value across the full joint."
  []
  (let [{:keys [probs retval]} (exact/exact-joint three-flips-sum [] nil)
        rv (mx/astype retval mx/float32)]
    (mapv (fn [k]
            (mx/item (mx/sum (mx/multiply probs
                               (mx/astype (mx/equal rv (mx/scalar k)) mx/float32)))))
          (range 4))))

(def binomial-builtin (gen [] (trace :k (dist/binomial 3 0.5))))

(defn binomial-pmf-builtin
  "P(k = 0..3) read from the primitive via exact-posterior marginals."
  []
  (let [r (exact/exact-posterior binomial-builtin [] nil)]
    (mapv (fn [k] (get-in r [:marginals :k k])) (range 4))))

;; ---------------------------------------------------------------------------
;; 4. twoHeads / moreThanTwoHeads — conditioning via the bernoulli-indicator
;; ---------------------------------------------------------------------------
;;
;; agentmodels conditions with `condition(totalHeads >= 2)`. In GenMLX the exact
;; analogue is the bernoulli-indicator trick (see inference/exact.cljs): trace a
;; deterministic bernoulli whose probability is the 1/0 event mask, then observe
;; it = 1. Observing bernoulli(mask)=1 contributes log(mask): 0 where the event
;; holds, -inf where it does not, which is precisely a hard `condition`.

(def first-flip-given-ge2
  "Three fair flips; condition (via :ge2) on total heads >= 2; RETURN the first
   flip so its marginal is P(first = H | total >= 2)."
  (gen []
    (let [a     (trace :a (dist/flip 0.5))
          b     (trace :b (dist/flip 0.5))
          c     (trace :c (dist/flip 0.5))
          total (reduce mx/add (mapv #(mx/astype % mx/float32) [a b c]))
          mask  (mx/where (mx/greater-equal total (mx/scalar 2.0))
                          (mx/scalar 1.0) (mx/scalar 0.0))]
      (trace :ge2 (dist/bernoulli mask))
      a)))

(def total-given-ge2
  "Binomial(3,0.5) conditioned on k >= 2; its :k marginal is P(total | total>=2)."
  (gen []
    (let [k    (trace :k (dist/binomial 3 0.5))
          mask (mx/where (mx/greater-equal k (mx/scalar 2))
                         (mx/scalar 1.0) (mx/scalar 0.0))]
      (trace :ge2 (dist/bernoulli mask))
      k)))

(defn two-heads-results []
  (let [r-first (exact/exact-posterior first-flip-given-ge2 []
                                       (cm/choicemap :ge2 (mx/scalar 1)))
        r-total (exact/exact-posterior total-given-ge2 []
                                       (cm/choicemap :ge2 (mx/scalar 1)))]
    {:p-first-heads (get-in r-first [:marginals :a 1])
     ;; :log-ml is already a host number, so this is Math/exp (not mx/exp).
     :evidence      (js/Math.exp (:log-ml r-first)) ; P(total heads >= 2)
     :p-total-2     (get-in r-total [:marginals :k 2])
     :p-total-3     (get-in r-total [:marginals :k 3])
     :p-total-0     (get-in r-total [:marginals :k 0])}))

;; ---------------------------------------------------------------------------
;; 5. structured returns + forward positionDist (discrete random walk)
;; ---------------------------------------------------------------------------

(def position-walk
  "A 2-step ±1 random walk from 0: each step is a fair flip mapped to ±1, and the
   return value is the final position in {-2, 0, +2}. Enumerated exactly, this is
   agentmodels' forward positionDist in miniature."
  (gen []
    (letfn [(step [f] ; 1->+1, 0->-1
              (mx/subtract (mx/multiply (mx/astype f mx/float32) (mx/scalar 2.0))
                           (mx/scalar 1.0)))]
      (let [s1 (trace :s1 (dist/flip 0.5))
            s2 (trace :s2 (dist/flip 0.5))]
        (mx/add (step s1) (step s2))))))

(defn position-dist
  "P(final position = v) for v in {-2, 0, +2}."
  []
  (let [{:keys [probs retval]} (exact/exact-joint position-walk [] nil)
        rv (mx/astype retval mx/float32)
        pv (fn [v] (mx/item (mx/sum (mx/multiply probs
                              (mx/astype (mx/equal rv (mx/scalar v)) mx/float32)))))]
    {-2 (pv -2) 0 (pv 0) 2 (pv 2)}))

;; ---------------------------------------------------------------------------
;; run + self-check
;; ---------------------------------------------------------------------------

(defn -main []
  (println "\n=== agentmodels.org Ch 2 — WebPPL primer (GenMLX port) ===")

  (println "\n-- 1. ERP forward draws (bernoulli / categorical / gaussian / uniform) --")
  (let [d (erp-draw)]
    (println (str "  draw: " d))
    (check-true "bern in {0,1}"  (contains? #{0 1} (:bern d)))
    (check-true "cat in {0,1,2}" (contains? #{0 1 2} (:cat d)))
    (check-true "unif in [0,1)"  (<= 0.0 (:unif d) 1.0)))

  (println "\n-- 2. multivariateGaussian: one 2-D draw --")
  (let [v (mvn-draw)]
    (println (str "  pos: [" (.toFixed (nth v 0) 3) ", " (.toFixed (nth v 1) 3) "]"))
    (check-true "draw has 2 components" (= 2 (count v))))

  (println "\n-- 3. binomial from 3 flips: P(sum=k), two ways --")
  (let [from-flips (binomial-pmf-from-flips)
        builtin    (binomial-pmf-builtin)
        ref        [0.125 0.375 0.375 0.125]]
    (doseq [k (range 4)]
      (check-close (str "from-flips P(sum=" k ")") (nth ref k) (nth from-flips k) 1e-5)
      (check-close (str "dist/binomial P(k=" k ")") (nth ref k) (nth builtin k) 1e-5)))

  (println "\n-- 4. twoHeads / moreThanTwoHeads: conditioning --")
  (let [{:keys [p-first-heads evidence p-total-2 p-total-3 p-total-0]} (two-heads-results)]
    (check-close "P(first = H | total >= 2)" 0.75 p-first-heads 1e-5)
    (check-close "evidence P(total >= 2)"    0.5  evidence      1e-5)
    (check-close "P(total = 2 | total >= 2)" 0.75 p-total-2     1e-5)
    (check-close "P(total = 3 | total >= 2)" 0.25 p-total-3     1e-5)
    (check-close "P(total = 0 | total >= 2)" 0.0  p-total-0     1e-5))

  (println "\n-- 5. forward positionDist: 2-step ±1 walk, P(final position) --")
  (let [pd (position-dist)]
    (println (str "  P(-2)=" (.toFixed (pd -2) 3)
                  "  P(0)=" (.toFixed (pd 0) 3)
                  "  P(+2)=" (.toFixed (pd 2) 3)))
    (check-close "P(position = -2)" 0.25 (pd -2) 1e-5)
    (check-close "P(position =  0)" 0.5  (pd 0)  1e-5)
    (check-close "P(position = +2)" 0.25 (pd 2)  1e-5))

  (println (str "\n" (if (zero? @fails)
                       "ALL CHECKS PASSED ✓"
                       (str @fails " CHECK(S) FAILED ✗"))))
  (when (pos? @fails) (js/process.exit 1)))

(-main)
