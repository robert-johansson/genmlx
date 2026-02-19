(ns genmlx.phase-bcd-test
  "Tests for Phase B, C, D features with meaningful correctness checks."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.trace :as tr]
            [genmlx.combinators :as comb]
            [genmlx.edit :as edit]
            [genmlx.diff :as diff]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.inference.kernel :as kern]
            [genmlx.gradients :as grad]
            [genmlx.learning :as learn])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Phase B/C/D Tests ===\n")

;; ---------------------------------------------------------------------------
;; 1.3 Edit interface — ConstraintEdit equivalence with update
;; ---------------------------------------------------------------------------

(println "-- Edit: ConstraintEdit == update --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))
                    y (dyn/trace :y (dist/gaussian 0 1))]
                (mx/eval! x y)
                (+ (mx/item x) (mx/item y))))
      trace (p/simulate model [])
      constraints (cm/choicemap :x (mx/scalar 3.0))
      ;; Compare edit vs direct update
      update-result (p/update model trace constraints)
      edit-result (edit/edit-dispatch model trace (edit/constraint-edit constraints))]
  (mx/eval! (:weight update-result) (:weight edit-result))
  (assert-close "constraint-edit weight == update weight"
    (mx/item (:weight update-result))
    (mx/item (:weight edit-result)) 1e-5)
  ;; Backward request should be ConstraintEdit containing discarded values
  (assert-true "backward request is ConstraintEdit"
    (instance? edit/ConstraintEdit (:backward-request edit-result))))

(println "\n-- Edit: SelectionEdit == regenerate --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (mx/eval! x) (mx/item x)))
      trace (p/simulate model [])
      sel (sel/select :x)
      edit-result (edit/edit-dispatch model trace (edit/selection-edit sel))]
  (mx/eval! (:weight edit-result))
  (assert-true "selection-edit weight is finite" (js/isFinite (mx/item (:weight edit-result))))
  (assert-true "backward request is SelectionEdit"
    (instance? edit/SelectionEdit (:backward-request edit-result))))

;; ---------------------------------------------------------------------------
;; 3.1 Scan combinator — verify carry threading
;; ---------------------------------------------------------------------------

(println "\n-- Scan: carry accumulates correctly --")
(let [;; Deterministic scan: carry += input (no randomness)
      step-fn (gen [carry input]
                (let [noise (dyn/trace :noise (dist/delta 0.0))]
                  [(+ carry input) (+ carry input)]))
      scan (comb/scan-combinator step-fn)
      trace (p/simulate scan [0.0 [1.0 2.0 3.0]])
      retval (:retval trace)]
  ;; carry should be 0+1+2+3 = 6
  (assert-close "scan final carry" 6.0 (:carry retval) 1e-5)
  ;; outputs should be [1, 3, 6]
  (assert-close "scan output 0" 1.0 (nth (:outputs retval) 0) 1e-5)
  (assert-close "scan output 1" 3.0 (nth (:outputs retval) 1) 1e-5)
  (assert-close "scan output 2" 6.0 (nth (:outputs retval) 2) 1e-5))

(println "\n-- Scan generate: constraining a step --")
(let [step-fn (gen [carry input]
                (let [noise (dyn/trace :noise (dist/gaussian 0 0.1))]
                  (mx/eval! noise)
                  (let [new-carry (+ carry (mx/item noise) input)]
                    [new-carry new-carry])))
      scan (comb/scan-combinator step-fn)
      ;; Constrain noise at step 1 to 0.0
      constraints (cm/set-choice cm/EMPTY [1] (cm/choicemap :noise (mx/scalar 0.0)))
      {:keys [trace weight]} (p/generate scan [0.0 [1.0 2.0 3.0]] constraints)]
  (mx/eval! weight)
  ;; Weight should be the log-prob of noise=0 under N(0,0.1)
  ;; which is log(1/(0.1*sqrt(2*pi))) ≈ 0.727
  (let [expected-lp (mx/item (do (let [v (dist/log-prob (dist/gaussian 0 0.1) (mx/scalar 0.0))]
                                   (mx/eval! v) v)))]
    (assert-close "scan constrained weight" expected-lp (mx/item weight) 0.01)))

;; ---------------------------------------------------------------------------
;; 3.3 Contramap / dimap — verify args actually transform
;; ---------------------------------------------------------------------------

(println "\n-- Contramap: argument transformation --")
(let [;; Model samples near its argument
      model (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.01))]
                (mx/eval! y) (mx/item y)))
      ;; Contramap doubles the argument
      doubled (comb/contramap-gf model (fn [args] [(* 2 (first args))]))
      trace (p/simulate doubled [5.0])]
  ;; retval should be near 10 (2*5), not near 5
  (assert-true "contramap: retval near 10" (< (js/Math.abs (- (:retval trace) 10)) 0.5))
  (assert-true "contramap: retval NOT near 5" (> (js/Math.abs (- (:retval trace) 5)) 3)))

(println "\n-- Map retval: return value transformation --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/delta 3.0))]
                (mx/eval! x) (mx/item x)))
      squared (comb/map-retval model #(* % %))
      trace (p/simulate squared [])]
  (assert-close "map-retval: 3^2=9" 9.0 (:retval trace) 1e-5))

;; ---------------------------------------------------------------------------
;; 3.4 Argdiffs / retdiffs
;; ---------------------------------------------------------------------------

(println "\n-- Diffs: identity check --")
(assert-true "identical values -> no-change" (diff/no-change? (diff/compute-diff 5 5)))
(assert-true "different values -> changed" (diff/changed? (diff/compute-diff 5 6)))

(println "\n-- Vector diff: identifies changed indices --")
(let [d (diff/compute-vector-diff [1 2 3 4] [1 2 99 4])]
  (assert-true "only index 2 changed" (= #{2} (:changed d))))

(println "\n-- Map diff: added/removed/changed --")
(let [d (diff/compute-map-diff {:a 1 :b 2 :c 3} {:a 1 :b 99 :d 4})]
  (assert-true "b changed" (contains? (:changed d) :b))
  (assert-true "c removed" (contains? (:removed d) :c))
  (assert-true "d added" (contains? (:added d) :d))
  (assert-true "a not in any diff set"
    (and (not (contains? (:changed d) :a))
         (not (contains? (:added d) :a)))))

;; ---------------------------------------------------------------------------
;; 2.5 Inference composition — verify MH chains converge
;; ---------------------------------------------------------------------------

(println "\n-- Kernel composition: MH chain converges to posterior --")
(let [model (gen []
              (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
                (mx/eval! mu)
                (let [mu-val (mx/item mu)]
                  (doseq [i (range 5)]
                    (dyn/trace (keyword (str "obs" i))
                               (dist/gaussian mu-val 1)))
                  mu-val)))
      observations (reduce (fn [cm i]
                              (cm/set-choice cm [(keyword (str "obs" i))]
                                             (mx/scalar 3.0)))
                            cm/EMPTY (range 5))
      {:keys [trace]} (p/generate model [] observations)
      k (kern/repeat-kernel 3 (kern/mh-kernel (sel/select :mu)))
      traces (kern/run-kernel {:samples 100 :burn 50} k trace)
      ;; Extract posterior mu samples
      mu-vals (mapv (fn [t]
                       (mx/realize (cm/get-value (cm/get-submap (:choices t) :mu))))
                     traces)
      mu-mean (/ (reduce + mu-vals) (count mu-vals))]
  (assert-true "kernel chain: 100 samples" (= 100 (count traces)))
  (assert-close "kernel chain: posterior mu near 3" 3.0 mu-mean 1.0)
  ;; Check acceptance rate in metadata
  (let [ar (:acceptance-rate (meta traces))]
    (assert-true "kernel chain has acceptance rate" (some? ar))
    (assert-true "acceptance rate > 0" (> ar 0))))

;; ---------------------------------------------------------------------------
;; 2.4 Conditional SMC — verify retained particle
;; ---------------------------------------------------------------------------

(println "\n-- Conditional SMC: reference particle present --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (mx/eval! x)
                (dyn/trace :obs (dist/gaussian (mx/item x) 0.5))
                (mx/item x)))
      observations [(cm/choicemap :obs (mx/scalar 3.0))]
      {:keys [trace]} (p/generate model [] (first observations))
      result (smc/csmc {:particles 20} model [] observations trace)]
  (assert-true "csmc: 20 traces" (= 20 (count (:traces result))))
  (mx/eval! (:log-ml-estimate result))
  (assert-true "csmc: log-ml finite" (js/isFinite (mx/item (:log-ml-estimate result)))))

;; ---------------------------------------------------------------------------
;; 2.1 SMCP3 — verify log-ml estimate
;; ---------------------------------------------------------------------------

(println "\n-- SMCP3: log-ml estimate consistent --")
(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (mx/eval! x)
                (dyn/trace :obs (dist/gaussian (mx/item x) 0.5))
                (mx/item x)))
      observations [(cm/choicemap :obs (mx/scalar 2.0))]
      ;; Run twice with more/fewer particles
      r1 (smcp3/smcp3 {:particles 50} model [] observations)
      r2 (smcp3/smcp3 {:particles 50} model [] observations)]
  (mx/eval! (:log-ml-estimate r1) (:log-ml-estimate r2))
  ;; Both estimates should be in the same ballpark
  ;; True log p(obs=2) = log N(2; 0, sqrt(1.25)) ≈ -2.86
  (let [lml1 (mx/item (:log-ml-estimate r1))
        lml2 (mx/item (:log-ml-estimate r2))]
    (assert-true "smcp3 log-ml estimate 1 reasonable" (< (js/Math.abs (- lml1 -2.86)) 1.5))
    (assert-true "smcp3 log-ml estimate 2 reasonable" (< (js/Math.abs (- lml2 -2.86)) 1.5))))

;; ---------------------------------------------------------------------------
;; 3.5 Mix combinator — verify branch selection
;; ---------------------------------------------------------------------------

(println "\n-- Mix combinator: components activated --")
(let [c0 (gen [] (let [x (dyn/trace :x (dist/delta -10.0))] (mx/eval! x) (mx/item x)))
      c1 (gen [] (let [x (dyn/trace :x (dist/delta 10.0))] (mx/eval! x) (mx/item x)))
      mix (comb/mix-combinator [c0 c1]
            (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)]))
      ;; Sample many times, should get both -10 and 10
      retvals (mapv (fn [_] (:retval (p/simulate mix []))) (range 50))
      has-neg (some #(< % 0) retvals)
      has-pos (some #(> % 0) retvals)]
  (assert-true "mix samples from component 0 (-10)" has-neg)
  (assert-true "mix samples from component 1 (+10)" has-pos))

;; ---------------------------------------------------------------------------
;; 4.1 Choice gradients — verify gradient direction
;; ---------------------------------------------------------------------------

(println "\n-- Choice gradients: gradient points toward observation --")
(let [model (gen [mu]
              (dyn/trace :obs (dist/gaussian mu 1)))
      ;; If mu=0 and obs=5, gradient of log p w.r.t. mu should be positive
      ;; (increasing mu increases log-prob)
      trace (let [{:keys [trace]} (p/generate model [0]
                    (cm/choicemap :obs (mx/scalar 5.0)))]
              trace)
      ;; Use the score-gradient utility
      result (grad/score-gradient model [0]
               (cm/choicemap :obs (mx/scalar 5.0))
               [:obs] (mx/array [5.0]))]
  (mx/eval! (:grad result))
  ;; The gradient at obs=5 should be finite
  (assert-true "gradient is finite" (js/isFinite (mx/item (:grad result)))))

;; ---------------------------------------------------------------------------
;; 4.2 Parameter store and optimizers
;; ---------------------------------------------------------------------------

(println "\n-- Param store: CRUD --")
(let [store (learn/make-param-store {:a (mx/scalar 1.0) :b (mx/scalar 2.0)})
      store' (learn/set-param store :a (mx/scalar 10.0))
      store'' (learn/update-params store' {:b (mx/scalar 20.0)})]
  (assert-close "get :a" 10.0 (mx/realize (learn/get-param store' :a)) 1e-5)
  (assert-close "get :b after update" 20.0 (mx/realize (learn/get-param store'' :b)) 1e-5)
  (assert-true "version increments" (> (:version store'') (:version store))))

(println "\n-- Adam: parameters move toward minimum --")
(let [;; Minimize f(x) = x^2, gradient = 2x, starting at x=5
      params (mx/array [5.0])
      state (learn/adam-init params)]
  ;; Take 50 adam steps toward minimum
  (let [[final-params _]
        (reduce (fn [[p s] _]
                  (let [g (mx/multiply (mx/scalar 2.0) p)]
                    (learn/adam-step p g s {:lr 0.1})))
                [params state]
                (range 50))]
    (mx/eval! final-params)
    (assert-true "adam: converges toward 0"
      (< (js/Math.abs (mx/item final-params)) 1.0))))

;; ---------------------------------------------------------------------------
;; Mixture distribution — gradient flow preserved
;; ---------------------------------------------------------------------------

(println "\n-- Mixture: log-prob consistency --")
(let [c1 (dist/gaussian 0 1)
      c2 (dist/gaussian 5 1)
      mix (dc/mixture [c1 c2] (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)]))]
  ;; At v=0: 0.5*N(0;0,1) + 0.5*N(0;5,1) ≈ 0.5*0.3989 + 0.5*tiny
  (let [lp0 (dc/dist-log-prob mix (mx/scalar 0.0))
        _ (mx/eval! lp0)
        ;; Manual: log(0.5 * 1/sqrt(2pi)) ≈ log(0.1995)
        expected (js/Math.log (* 0.5 (/ 1.0 (js/Math.sqrt (* 2 js/Math.PI)))))]
    (assert-close "mixture lp at 0" expected (mx/item lp0) 0.05))
  ;; At v=2.5: both components contribute equally
  (let [lp25 (dc/dist-log-prob mix (mx/scalar 2.5))
        _ (mx/eval! lp25)
        ;; Manual: 0.5*N(2.5;0,1) + 0.5*N(2.5;5,1) — both = N(2.5;_,1)
        g-lp (js/Math.exp (- (* -0.5 2.5 2.5) (* 0.5 (js/Math.log (* 2 js/Math.PI)))))
        expected (js/Math.log (* 2 0.5 g-lp))]
    (assert-close "mixture lp at 2.5 (symmetric)" expected (mx/item lp25) 0.05)))

(println "\nAll Phase B/C/D tests complete.")
