;; @tier fast
(ns genmlx.degenerate-weights-test
  "genmlx-ng9t: all-(-Inf) particle populations must FAIL LOUDLY.

   Audit genmlx-ansg (VERIFIED) found: normalize-log-weights on all-(-Inf)
   input yielded probs [NaN NaN ...], and every resampler's floating-point
   exhaustion branch fires on a NaN cumsum — the population silently collapsed
   to the LAST particle with valid-looking traces (only the -Inf log-ML
   hinted). The guard lives in u/normalize-log-weights — the choke point all
   three resamplers (systematic in inference/util, residual + stratified in
   inference/smc) and compute-ess normalize through — and throws
   `:genmlx/error :degenerate-particles`.

   Also pins the audit's verified-working behavior that must NOT break:
   logsumexp([-Inf ...]) = -Inf without NaN, and a SINGLE -Inf-weight particle
   normalizes to prob 0 and is never resampled."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.util :as u]
            [genmlx.inference.smc :as smc]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(defn assert-close [label expected actual tol]
  (assert-true (str label " (" actual " ~ " expected ")")
               (and (js/isFinite actual)
                    (<= (js/Math.abs (- expected actual)) tol))))

(def ^:private neg-inf js/Number.NEGATIVE_INFINITY)

(defn- error-kind
  "Run f, returning the :genmlx/error of the thrown ex-info (or :no-throw)."
  [f]
  (try (f) :no-throw
       (catch :default e (or (:genmlx/error (ex-data e)) :other-throw))))

(defn- inf-weights [n]
  (vec (repeat n (mx/scalar neg-inf))))

;; ===========================================================================
(println "\n-- normalize-log-weights: degenerate populations throw --")

(assert-true "all-(-Inf) weights throw :degenerate-particles"
             (= :degenerate-particles
                (error-kind #(u/normalize-log-weights (inf-weights 4)))))

(assert-true "a NaN weight throws :nan-weights"
             (= :nan-weights
                (error-kind #(u/normalize-log-weights
                              [(mx/scalar 0.0) (mx/scalar js/NaN)]))))

(assert-true "a +Inf weight throws :infinite-weight"
             (= :infinite-weight
                (error-kind #(u/normalize-log-weights
                              [(mx/scalar 0.0) (mx/scalar js/Number.POSITIVE_INFINITY)]))))

(println "\n-- single -Inf among finite weights: behavior unchanged --")

(let [{:keys [probs]} (u/normalize-log-weights
                       [(mx/scalar 0.0) (mx/scalar neg-inf) (mx/scalar 0.0)])]
  (assert-close "single--Inf particle normalizes to prob 0" 0.0 (nth probs 1) 1e-9)
  (assert-close "remaining mass splits evenly (p0 = 0.5)" 0.5 (nth probs 0) 1e-6)
  (assert-close "remaining mass splits evenly (p2 = 0.5)" 0.5 (nth probs 2) 1e-6))

(assert-close "compute-ess with a single--Inf particle is finite and exact (2.0)"
              2.0
              (u/compute-ess [(mx/scalar 0.0) (mx/scalar neg-inf) (mx/scalar 0.0)])
              1e-6)

(println "\n-- resamplers --")

(assert-true "systematic-resample on all-(-Inf) throws :degenerate-particles"
             (= :degenerate-particles
                (error-kind #(u/systematic-resample (inf-weights 5) 5 (rng/fresh-key 7)))))

(let [indices (u/systematic-resample
               [(mx/scalar 0.0) (mx/scalar neg-inf) (mx/scalar 0.0)]
               50 (rng/fresh-key 11))]
  (assert-true "single--Inf particle is never resampled (50 draws)"
               (and (= 50 (count indices))
                    (not-any? #(= 1 %) indices)
                    (every? #{0 2} indices))))

(assert-true "compute-ess on all-(-Inf) throws :degenerate-particles"
             (= :degenerate-particles
                (error-kind #(u/compute-ess (inf-weights 3)))))

;; ===========================================================================
(println "\n-- do-not-break pin: logsumexp([-Inf ...]) = -Inf without NaN --")

(let [lse (mx/realize (mx/logsumexp (mx/array [neg-inf neg-inf neg-inf])))]
  (assert-true "logsumexp of all--Inf is -Inf (not NaN)"
               (and (= lse neg-inf) (not (js/isNaN lse)))))

;; ===========================================================================
;; End-to-end: an SMC run where every particle hits an impossible observation
;; must throw loudly at the next weight normalization, for every resample
;; method (all three route through the guarded normalize).
;; ===========================================================================
(println "\n-- SMC end-to-end: impossible observation fails loudly --")

(def ^:private step-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      ;; :y is a point mass at 1.0 — constraining it to anything else is
      ;; impossible (exact -Inf log-prob), independent of :x.
      (trace :y (dist/delta (mx/scalar 1.0)))
      x)))

(def ^:private impossible-obs
  [cm/EMPTY                                        ; t=0: init from prior
   (cm/set-choice cm/EMPTY [:y] (mx/scalar 2.0))   ; t=1: all weights -> -Inf
   cm/EMPTY])                                      ; t=2: ESS check must throw

(doseq [method [:systematic :residual :stratified]]
  (assert-true (str "smc with :resample-method " method
                    " throws :degenerate-particles on an impossible observation")
               (= :degenerate-particles
                  (error-kind #(smc/smc {:particles 8 :ess-threshold 0.5
                                         :resample-method method
                                         :key (rng/fresh-key 13)}
                                        step-model [] impossible-obs)))))

;; ===========================================================================
(println (str "\n== degenerate-weights: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
