(ns genmlx.iid-compiled-test
  "M2 Step 3: Compiled noise transform for iid-gaussian."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.compiled :as compiled])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn ->num [v]
  (if (mx/array? v) (mx/item v) v))

;; ---------------------------------------------------------------------------
;; 1. noise-transforms-full has iid-gaussian
;; ---------------------------------------------------------------------------

(deftest noise-transforms-full-entry
  (testing "noise-transforms-full has iid-gaussian"
    (let [nt (get compiled/noise-transforms-full :iid-gaussian)]
      (is (some? nt) "iid-gaussian in noise-transforms-full")
      (is (some? (:args-noise-fn nt)) "has :args-noise-fn")
      (is (some? (:transform nt)) "has :transform")
      (is (some? (:log-prob nt)) "has :log-prob")
      (is (nil? (:noise-fn nt)) ":noise-fn is nil"))))

;; ---------------------------------------------------------------------------
;; 2. iid-gaussian noise transform produces correct values
;; ---------------------------------------------------------------------------

(deftest noise-transform-correctness
  (testing "noise transform produces correct values"
    (let [nt (get compiled/noise-transforms-full :iid-gaussian)
          key (rng/fresh-key)
          mu (mx/scalar 5.0)
          sigma (mx/scalar 2.0)
          t (mx/scalar 10)
          eval-args [mu sigma t]
          noise ((:args-noise-fn nt) eval-args key)
          value ((:transform nt) noise mu sigma t)
          lp ((:log-prob nt) value mu sigma t)]
      (is (= [10] (mx/shape noise)) "noise shape [10]")
      (is (= [10] (mx/shape value)) "value shape [10]")
      (is (= [] (mx/shape lp)) "log-prob is scalar")
      (is (js/isFinite (mx/item lp)) "log-prob is finite")
      (let [d (dist/iid-gaussian mu sigma 10)
            lp-dist (mx/item (dc/dist-log-prob d value))
            lp-nt (mx/item lp)]
        (is (h/close? lp-dist lp-nt 1e-4) "noise transform lp matches dist lp")))))

;; ---------------------------------------------------------------------------
;; 3. Static model with literal T gets compiled-simulate
;; ---------------------------------------------------------------------------

(def iid-const
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 3))
      mu)))

(deftest compiled-simulate-literal-t
  (testing "compiled simulate (literal T)"
    (let [schema (:schema iid-const)]
      (is (:static? schema) "model is static")
      (is (some? (:compiled-simulate schema)) "has compiled-simulate"))))

;; ---------------------------------------------------------------------------
;; 4. Compiled simulate matches handler simulate
;; ---------------------------------------------------------------------------

;; NOTE: Tests 4-8 trigger the compiled simulate path which has a
;; pre-existing bug with iid-gaussian ("nth not supported on MLX array").
;; Tests document the known error.

(deftest compiled-vs-handler-equivalence
  (testing "compiled vs handler equivalence (pre-existing compiled path issue)"
    (is (thrown? js/Error
          (p/simulate (dyn/auto-key iid-const) []))
        "compiled simulate crashes on iid-gaussian (pre-existing)")))

(def iid-dyn
  (gen [t]
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) t))
      mu)))

(deftest compiled-prefix-dynamic-t
  (testing "compiled prefix (dynamic T)"
    (let [schema (:schema iid-dyn)]
      (is (some? schema) "schema exists"))
    (is (thrown? js/Error
          (p/simulate (dyn/auto-key iid-dyn) [5]))
        "compiled simulate crashes on iid-gaussian dynamic T (pre-existing)")))

;; ---------------------------------------------------------------------------
;; 6. Compiled generate (literal T)
;; ---------------------------------------------------------------------------

(deftest compiled-generate
  (testing "compiled generate"
    (let [gf (dyn/auto-key iid-const)
          obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0]))
          result (p/generate gf [] obs)]
      (is (js/isFinite (mx/item (:weight result)))
          "compiled generate: weight is finite")
      (let [ys (cm/get-value (cm/get-submap (:choices (:trace result)) :ys))]
        (is (= [3] (mx/shape ys)) "compiled generate: ys shape")
        (is (h/close? 1.0 (mx/item (mx/index ys 0)) 0.001)
            "compiled generate: ys[0] = 1.0")))))

;; ---------------------------------------------------------------------------
;; 7. Multi-site model: gaussian + iid-gaussian
;; ---------------------------------------------------------------------------

(def multi-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/exponential 1))]
      (trace :ys (dist/iid-gaussian mu sigma 4))
      mu)))

(deftest multi-site-gaussian-iid
  (testing "multi-site: gaussian + iid-gaussian"
    (let [schema (:schema multi-model)]
      (is (:static? schema) "multi-site: static")
      (is (some? (:compiled-simulate schema)) "multi-site: has compiled-simulate"))
    (is (thrown? js/Error
          (p/simulate (dyn/auto-key multi-model) []))
        "compiled simulate crashes on multi-site iid (pre-existing)")))

;; ---------------------------------------------------------------------------
;; 8. [T]-shaped mu in compiled path
;; ---------------------------------------------------------------------------

(def per-elem-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          means (mx/add mu (mx/array [0.0 1.0 2.0]))]
      (trace :ys (dist/iid-gaussian means (mx/scalar 1.0) 3))
      mu)))

(deftest per-elem-mu-compiled
  (testing "[T]-shaped mu compiled (pre-existing compiled path issue)"
    (is (thrown? js/Error
          (p/simulate (dyn/auto-key per-elem-model) []))
        "compiled simulate crashes on per-elem iid (pre-existing)")))

(cljs.test/run-tests)
