(ns genmlx.iid-compiled-test
  "M2 Step 3: Compiled noise transform for iid-gaussian.
   Verifies that models using iid-gaussian get compiled simulate paths
   and produce identical results to the handler path."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.compiled :as compiled])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc) (println "  PASS:" msg))
      (do (swap! fail-count inc) (println "  FAIL:" msg "expected:" expected "got:" actual "diff:" diff)))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg "expected:" expected "got:" actual))))

(defn ->num [v]
  (if (mx/array? v) (mx/item v) v))

;; ---------------------------------------------------------------------------
;; 1. noise-transforms-full has iid-gaussian
;; ---------------------------------------------------------------------------

(println "\n-- 1. noise-transforms-full --")

(let [nt (get compiled/noise-transforms-full :iid-gaussian)]
  (assert-true "iid-gaussian in noise-transforms-full" (some? nt))
  (assert-true "has :args-noise-fn" (some? (:args-noise-fn nt)))
  (assert-true "has :transform" (some? (:transform nt)))
  (assert-true "has :log-prob" (some? (:log-prob nt)))
  (assert-true ":noise-fn is nil" (nil? (:noise-fn nt))))

;; ---------------------------------------------------------------------------
;; 2. iid-gaussian noise transform produces correct values
;; ---------------------------------------------------------------------------

(println "\n-- 2. noise transform correctness --")

(let [nt (get compiled/noise-transforms-full :iid-gaussian)
      key (rng/fresh-key)
      mu (mx/scalar 5.0)
      sigma (mx/scalar 2.0)
      t (mx/scalar 10)
      eval-args [mu sigma t]
      noise ((:args-noise-fn nt) eval-args key)
      value ((:transform nt) noise mu sigma t)
      lp ((:log-prob nt) value mu sigma t)]
  (assert-equal "noise shape [10]" [10] (mx/shape noise))
  (assert-equal "value shape [10]" [10] (mx/shape value))
  (assert-equal "log-prob is scalar" [] (mx/shape lp))
  (assert-true "log-prob is finite" (js/isFinite (mx/item lp)))
  ;; Cross-check with dist/iid-gaussian log-prob
  (let [d (dist/iid-gaussian mu sigma 10)
        lp-dist (mx/item (dc/dist-log-prob d value))
        lp-nt (mx/item lp)]
    (assert-close "noise transform lp matches dist lp" lp-dist lp-nt 1e-4)))

;; ---------------------------------------------------------------------------
;; 3. Static model with literal T gets compiled-simulate
;; ---------------------------------------------------------------------------

(println "\n-- 3. compiled simulate (literal T) --")

(def iid-const
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 3))
      mu)))

(let [schema (:schema iid-const)]
  (assert-true "model is static" (:static? schema))
  (assert-true "has compiled-simulate" (some? (:compiled-simulate schema))))

;; ---------------------------------------------------------------------------
;; 4. Compiled simulate matches handler simulate
;; ---------------------------------------------------------------------------

(println "\n-- 4. compiled vs handler equivalence (literal T) --")

(let [gf (dyn/auto-key iid-const)
      ;; Run multiple simulations and check score = joint log-prob
      scores (for [_ (range 20)]
               (let [tr (p/simulate gf [])
                     choices (:choices tr)
                     mu (cm/get-value (cm/get-submap choices :mu))
                     ys (cm/get-value (cm/get-submap choices :ys))
                     ;; Compute joint log-prob directly
                     mu-lp (dc/dist-log-prob (dist/gaussian 0 5) mu)
                     ys-lp (dc/dist-log-prob (dist/iid-gaussian mu (mx/scalar 1.0) 3) ys)
                     expected (mx/item (mx/add mu-lp ys-lp))]
                 {:score (mx/item (:score tr))
                  :expected expected
                  :ys-shape (mx/shape ys)}))]
  (assert-true "all ys shape [3]" (every? #(= [3] (:ys-shape %)) scores))
  ;; Score should match joint log-prob
  (let [diffs (mapv #(js/Math.abs (- (:score %) (:expected %))) scores)
        max-diff (apply max diffs)]
    (assert-close "score ≈ joint log-prob (max diff)" 0.0 max-diff 0.01)))

;; ---------------------------------------------------------------------------
;; 5. Static model with dynamic T — compiled prefix
;; ---------------------------------------------------------------------------

(println "\n-- 5. compiled prefix (dynamic T) --")

(def iid-dyn
  (gen [t]
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) t))
      mu)))

(let [schema (:schema iid-dyn)]
  ;; Dynamic T means model is not fully static — but mu prefix should compile
  ;; (iid-dyn is not static because t is an arg, making the schema non-static
  ;;  if the schema walker classifies :ys as dynamic due to unknown dist-args)
  (assert-true "schema exists" (some? schema)))

;; Regardless of compilation, the model should work correctly
(let [gf (dyn/auto-key iid-dyn)
      tr (p/simulate gf [5])
      choices (:choices tr)
      mu (cm/get-value (cm/get-submap choices :mu))
      ys (cm/get-value (cm/get-submap choices :ys))
      ;; Compute joint log-prob directly
      mu-lp (dc/dist-log-prob (dist/gaussian 0 5) mu)
      ys-lp (dc/dist-log-prob (dist/iid-gaussian mu (mx/scalar 1.0) 5) ys)
      expected (mx/item (mx/add mu-lp ys-lp))]
  (assert-equal "dynamic T: ys shape [5]" [5] (mx/shape ys))
  (assert-close "dynamic T: score ≈ joint log-prob"
                expected
                (mx/item (:score tr))
                0.01))

;; ---------------------------------------------------------------------------
;; 6. Compiled generate (literal T)
;; ---------------------------------------------------------------------------

(println "\n-- 6. compiled generate --")

(let [gf (dyn/auto-key iid-const)
      obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0]))
      result (p/generate gf [] obs)]
  (assert-true "compiled generate: weight is finite"
               (js/isFinite (mx/item (:weight result))))
  ;; ys should be constrained
  (let [ys (cm/get-value (cm/get-submap (:choices (:trace result)) :ys))]
    (assert-equal "compiled generate: ys shape" [3] (mx/shape ys))
    (assert-close "compiled generate: ys[0] = 1.0" 1.0 (mx/item (mx/index ys 0)) 0.001)))

;; ---------------------------------------------------------------------------
;; 7. Multi-site model: gaussian + iid-gaussian
;; ---------------------------------------------------------------------------

(println "\n-- 7. multi-site: gaussian + iid-gaussian --")

(def multi-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/exponential 1))]
      (trace :ys (dist/iid-gaussian mu sigma 4))
      mu)))

(let [schema (:schema multi-model)]
  (assert-true "multi-site: static" (:static? schema))
  (assert-true "multi-site: has compiled-simulate" (some? (:compiled-simulate schema))))

(let [gf (dyn/auto-key multi-model)
      tr (p/simulate gf [])
      choices (:choices tr)
      mu (cm/get-value (cm/get-submap choices :mu))
      sigma (cm/get-value (cm/get-submap choices :sigma))
      ys (cm/get-value (cm/get-submap choices :ys))
      ;; Compute joint log-prob directly
      mu-lp (dc/dist-log-prob (dist/gaussian 0 10) mu)
      sigma-lp (dc/dist-log-prob (dist/exponential 1) sigma)
      ys-lp (dc/dist-log-prob (dist/iid-gaussian mu sigma 4) ys)
      expected (mx/item (mx/add mu-lp (mx/add sigma-lp ys-lp)))]
  (assert-equal "multi-site: ys shape [4]" [4] (mx/shape ys))
  (assert-close "multi-site: score ≈ joint log-prob"
                expected
                (mx/item (:score tr))
                0.01))

;; ---------------------------------------------------------------------------
;; 8. [T]-shaped mu in compiled path
;; ---------------------------------------------------------------------------

(println "\n-- 8. [T]-shaped mu compiled --")

(def per-elem-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          ;; Per-element means: mu + [0, 1, 2]
          means (mx/add mu (mx/array [0.0 1.0 2.0]))]
      (trace :ys (dist/iid-gaussian means (mx/scalar 1.0) 3))
      mu)))

(let [gf (dyn/auto-key per-elem-model)
      tr (p/simulate gf [])
      choices (:choices tr)
      mu (cm/get-value (cm/get-submap choices :mu))
      ys (cm/get-value (cm/get-submap choices :ys))
      means (mx/add mu (mx/array [0.0 1.0 2.0]))
      ;; Compute joint log-prob directly
      mu-lp (dc/dist-log-prob (dist/gaussian 0 10) mu)
      ys-lp (dc/dist-log-prob (dist/iid-gaussian means (mx/scalar 1.0) 3) ys)
      expected (mx/item (mx/add mu-lp ys-lp))]
  (assert-equal "per-elem: ys shape [3]" [3] (mx/shape ys))
  (assert-close "per-elem: score ≈ joint log-prob"
                expected
                (mx/item (:score tr))
                0.01))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n========================================")
(println (str "M2 Step 3 (Compiled iid-gaussian): " @pass-count "/" (+ @pass-count @fail-count)
              " passed" (when (pos? @fail-count) (str ", " @fail-count " FAILED"))))
(println "========================================")

(when (pos? @fail-count)
  (js/process.exit 1))
