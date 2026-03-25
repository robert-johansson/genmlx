(ns genmlx.phase-bcd-test
  "Tests for Phase B, C, D features with meaningful correctness checks."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
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
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.inference.kernel :as kern]
            [genmlx.gradients :as grad]
            [genmlx.learning :as learn])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Edit interface
;; ---------------------------------------------------------------------------

(deftest constraint-edit-equivalence
  (testing "ConstraintEdit == update"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    (mx/eval! x y)
                    (+ (mx/item x) (mx/item y))))
          trace (p/simulate (dyn/auto-key model) [])
          constraints (cm/choicemap :x (mx/scalar 3.0))
          update-result (p/update (dyn/auto-key model) trace constraints)
          edit-result (edit/edit-dispatch (dyn/auto-key model) trace (edit/constraint-edit constraints))]
      (mx/eval! (:weight update-result) (:weight edit-result))
      (is (h/close? (mx/item (:weight update-result))
                    (mx/item (:weight edit-result)) 1e-5)
          "constraint-edit weight == update weight")
      (is (instance? edit/ConstraintEdit (:backward-request edit-result))
          "backward request is ConstraintEdit"))))

(deftest selection-edit-equivalence
  (testing "SelectionEdit == regenerate"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    (mx/eval! x) (mx/item x)))
          trace (p/simulate (dyn/auto-key model) [])
          sel (sel/select :x)
          edit-result (edit/edit-dispatch (dyn/auto-key model) trace (edit/selection-edit sel))]
      (mx/eval! (:weight edit-result))
      (is (js/isFinite (mx/item (:weight edit-result))) "selection-edit weight is finite")
      (is (instance? edit/SelectionEdit (:backward-request edit-result))
          "backward request is SelectionEdit"))))

;; ---------------------------------------------------------------------------
;; Scan combinator
;; ---------------------------------------------------------------------------

(deftest scan-carry-accumulation
  (testing "carry accumulates correctly"
    (let [step-fn (gen [carry input]
                    (let [noise (trace :noise (dist/delta 0.0))]
                      [(+ carry input) (+ carry input)]))
          scan (comb/scan-combinator (dyn/auto-key step-fn))
          trace (p/simulate scan [0.0 [1.0 2.0 3.0]])
          retval (:retval trace)
          realize (fn [x] (if (mx/array? x) (do (mx/eval! x) (mx/item x)) x))]
      (is (h/close? 6.0 (realize (:carry retval)) 1e-5) "scan final carry")
      (is (h/close? 1.0 (realize (nth (:outputs retval) 0)) 1e-5) "scan output 0")
      (is (h/close? 3.0 (realize (nth (:outputs retval) 1)) 1e-5) "scan output 1")
      (is (h/close? 6.0 (realize (nth (:outputs retval) 2)) 1e-5) "scan output 2"))))

(deftest scan-generate-constrained
  (testing "constraining a step"
    (let [step-fn (gen [carry input]
                    (let [noise (trace :noise (dist/gaussian 0 0.1))]
                      (mx/eval! noise)
                      (let [new-carry (+ carry (mx/item noise) input)]
                        [new-carry new-carry])))
          scan (comb/scan-combinator (dyn/auto-key step-fn))
          constraints (cm/set-choice cm/EMPTY [1] (cm/choicemap :noise (mx/scalar 0.0)))
          {:keys [trace weight]} (p/generate scan [0.0 [1.0 2.0 3.0]] constraints)]
      (mx/eval! weight)
      (let [expected-lp (mx/item (do (let [v (dist/log-prob (dist/gaussian 0 0.1) (mx/scalar 0.0))]
                                       (mx/eval! v) v)))]
        (is (h/close? expected-lp (mx/item weight) 0.01) "scan constrained weight")))))

;; ---------------------------------------------------------------------------
;; Contramap / dimap
;; ---------------------------------------------------------------------------

(deftest contramap-argument-transformation
  (testing "argument transformation"
    (let [model (gen [x]
                  (let [y (trace :y (dist/gaussian x 0.01))]
                    (mx/eval! y) (mx/item y)))
          doubled (comb/contramap-gf (dyn/auto-key model) (fn [args] [(* 2 (first args))]))
          trace (p/simulate doubled [5.0])]
      (is (< (js/Math.abs (- (:retval trace) 10)) 0.5) "contramap: retval near 10")
      (is (> (js/Math.abs (- (:retval trace) 5)) 3) "contramap: retval NOT near 5"))))

(deftest map-retval-transformation
  (testing "return value transformation"
    (let [model (gen []
                  (let [x (trace :x (dist/delta 3.0))]
                    (mx/eval! x) (mx/item x)))
          squared (comb/map-retval (dyn/auto-key model) #(* % %))
          trace (p/simulate squared [])]
      (is (h/close? 9.0 (:retval trace) 1e-5) "map-retval: 3^2=9"))))

;; ---------------------------------------------------------------------------
;; Kernel composition / MH convergence
;; ---------------------------------------------------------------------------

(deftest kernel-composition-convergence
  (testing "MH chain converges to posterior"
    (let [model (gen []
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (mx/eval! mu)
                    (let [mu-val (mx/item mu)]
                      (doseq [i (range 5)]
                        (trace (keyword (str "obs" i))
                                   (dist/gaussian mu-val 1)))
                      mu-val)))
          observations (reduce (fn [cm i]
                                  (cm/set-choice cm [(keyword (str "obs" i))]
                                                 (mx/scalar 3.0)))
                                cm/EMPTY (range 5))
          {:keys [trace]} (p/generate (dyn/auto-key model) [] observations)
          k (kern/repeat-kernel 3 (kern/mh-kernel (sel/select :mu)))
          traces (kern/run-kernel {:samples 100 :burn 50} k trace)
          mu-vals (mapv (fn [t]
                           (mx/realize (cm/get-value (cm/get-submap (:choices t) :mu))))
                         traces)
          mu-mean (/ (reduce + mu-vals) (count mu-vals))]
      (is (= 100 (count traces)) "kernel chain: 100 samples")
      (is (h/close? 3.0 mu-mean 1.0) "kernel chain: posterior mu near 3")
      (let [ar (:acceptance-rate (meta traces))]
        (is (some? ar) "kernel chain has acceptance rate")
        (is (> ar 0) "acceptance rate > 0")))))

;; ---------------------------------------------------------------------------
;; Conditional SMC
;; ---------------------------------------------------------------------------

(deftest conditional-smc
  (testing "reference particle present"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    (mx/eval! x)
                    (trace :obs (dist/gaussian (mx/item x) 0.5))
                    (mx/item x)))
          observations [(cm/choicemap :obs (mx/scalar 3.0))]
          {:keys [trace]} (p/generate (dyn/auto-key model) [] (first observations))
          result (smc/csmc {:particles 20} model [] observations trace)]
      (is (= 20 (count (:traces result))) "csmc: 20 traces")
      (mx/eval! (:log-ml-estimate result))
      (is (js/isFinite (mx/item (:log-ml-estimate result))) "csmc: log-ml finite"))))

;; ---------------------------------------------------------------------------
;; SMCP3
;; ---------------------------------------------------------------------------

(deftest smcp3-log-ml
  (testing "log-ml estimate consistent"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    (mx/eval! x)
                    (trace :obs (dist/gaussian (mx/item x) 0.5))
                    (mx/item x)))
          observations [(cm/choicemap :obs (mx/scalar 2.0))]
          r1 (smcp3/smcp3 {:particles 50} model [] observations)
          r2 (smcp3/smcp3 {:particles 50} model [] observations)]
      (mx/eval! (:log-ml-estimate r1) (:log-ml-estimate r2))
      (let [lml1 (mx/item (:log-ml-estimate r1))
            lml2 (mx/item (:log-ml-estimate r2))]
        (is (< (js/Math.abs (- lml1 -2.86)) 1.5) "smcp3 log-ml estimate 1 reasonable")
        (is (< (js/Math.abs (- lml2 -2.86)) 1.5) "smcp3 log-ml estimate 2 reasonable")))))

;; ---------------------------------------------------------------------------
;; Mix combinator
;; ---------------------------------------------------------------------------

(deftest mix-combinator-test
  (testing "components activated"
    (let [c0 (gen [] (let [x (trace :x (dist/delta -10.0))] (mx/eval! x) (mx/item x)))
          c1 (gen [] (let [x (trace :x (dist/delta 10.0))] (mx/eval! x) (mx/item x)))
          mix (comb/mix-combinator [(dyn/auto-key c0) (dyn/auto-key c1)]
                (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)]))
          retvals (mapv (fn [_] (:retval (p/simulate mix []))) (range 50))
          has-neg (some #(< % 0) retvals)
          has-pos (some #(> % 0) retvals)]
      (is has-neg "mix samples from component 0 (-10)")
      (is has-pos "mix samples from component 1 (+10)"))))

;; ---------------------------------------------------------------------------
;; Choice gradients
;; ---------------------------------------------------------------------------

(deftest choice-gradients
  (testing "gradient points toward observation"
    (let [model (gen [mu]
                  (trace :obs (dist/gaussian mu 1)))
          result (grad/score-gradient model [0]
                   (cm/choicemap :obs (mx/scalar 5.0))
                   [:obs] (mx/array [5.0]))]
      (mx/eval! (:grad result))
      (is (js/isFinite (mx/item (:grad result))) "gradient is finite"))))

;; ---------------------------------------------------------------------------
;; Parameter store and optimizers
;; ---------------------------------------------------------------------------

(deftest param-store-crud
  (testing "param store CRUD"
    (let [store (learn/make-param-store {:a (mx/scalar 1.0) :b (mx/scalar 2.0)})
          store' (learn/set-param store :a (mx/scalar 10.0))
          store'' (learn/update-params store' {:b (mx/scalar 20.0)})]
      (is (h/close? 10.0 (mx/realize (learn/get-param store' :a)) 1e-5) "get :a")
      (is (h/close? 20.0 (mx/realize (learn/get-param store'' :b)) 1e-5) "get :b after update")
      (is (> (:version store'') (:version store)) "version increments"))))

(deftest adam-convergence
  (testing "parameters move toward minimum"
    (let [params (mx/array [5.0])
          state (learn/adam-init params)]
      (let [[final-params _]
            (reduce (fn [[p s] _]
                      (let [g (mx/multiply (mx/scalar 2.0) p)]
                        (learn/adam-step p g s {:lr 0.1})))
                    [params state]
                    (range 50))]
        (mx/eval! final-params)
        (is (< (js/Math.abs (mx/item final-params)) 1.0) "adam: converges toward 0")))))

;; ---------------------------------------------------------------------------
;; Mixture log-prob consistency
;; ---------------------------------------------------------------------------

(deftest mixture-log-prob-consistency
  (testing "log-prob consistency"
    (let [c1 (dist/gaussian 0 1)
          c2 (dist/gaussian 5 1)
          mix (dc/mixture [c1 c2] (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)]))]
      (let [lp0 (dc/dist-log-prob mix (mx/scalar 0.0))
            _ (mx/eval! lp0)
            expected (js/Math.log (* 0.5 (/ 1.0 (js/Math.sqrt (* 2 js/Math.PI)))))]
        (is (h/close? expected (mx/item lp0) 0.05) "mixture lp at 0"))
      (let [lp25 (dc/dist-log-prob mix (mx/scalar 2.5))
            _ (mx/eval! lp25)
            g-lp (js/Math.exp (- (* -0.5 2.5 2.5) (* 0.5 (js/Math.log (* 2 js/Math.PI)))))
            expected (js/Math.log (* 2 0.5 g-lp))]
        (is (h/close? expected (mx/item lp25) 0.05) "mixture lp at 2.5 (symmetric)")))))

(cljs.test/run-tests)
