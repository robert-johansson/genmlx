(ns genmlx.wp6-regenerate-test
  "WP-6 tests: compiled regenerate for DynamicGF."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Non-assert helpers (kept)
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip ALL compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf)
                 :compiled-simulate :compiled-generate
                 :compiled-update :compiled-assess :compiled-project
                 :compiled-regenerate
                 :compiled-prefix :compiled-prefix-addrs
                 :compiled-prefix-generate :compiled-prefix-update
                 :compiled-prefix-assess :compiled-prefix-project
                 :compiled-prefix-regenerate)]
    (assoc gf :schema schema)))

(defn make-trace-via-generate
  "Create a deterministic trace by calling generate with all sites constrained."
  [gf args constraints]
  (let [gf-h (force-handler gf)
        {:keys [trace]} (p/generate (dyn/with-key gf-h (rng/fresh-key 1)) args constraints)]
    trace))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def model-indep
  (gen [mu]
    (let [x (trace :x (dist/gaussian mu 1))
          y (trace :y (dist/gaussian 0 2))]
      (mx/add x y))))

(def model-dep
  (gen [mu]
    (let [x (trace :x (dist/gaussian mu 1))
          y (trace :y (dist/gaussian x 0.5))]
      y)))

(def model-chain
  (gen [a]
    (let [x (trace :x (dist/gaussian a 1))
          y (trace :y (dist/gaussian x 0.5))
          z (trace :z (dist/gaussian (mx/add x y) 1))]
      z)))

(def model-multi
  (gen []
    (let [g (trace :g (dist/gaussian 0 1))
          u (trace :u (dist/uniform 0 1))
          e (trace :e (dist/exponential 2))]
      g)))

(def model-beta
  (dyn/auto-key
    (gen [] (trace :x (dist/beta-dist 2 5)))))

(def m4-branch
  (gen [flag]
    (let [x (if flag
              (trace :x (dist/gaussian 0 1))
              (trace :x (dist/gaussian 5 2)))
          y (trace :y (dist/gaussian x 0.5))]
      y)))

(def m3-loop
  (gen [n]
    (let [x (trace :x (dist/gaussian 0 1))]
      (doseq [i (range (mx/item n))]
        (trace (keyword (str "y" i)) (dist/gaussian x 0.5)))
      x)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest prerequisites-test
  (testing "Compilation check"
    (is (some? (:compiled-regenerate (:schema model-indep)))
        "model-indep has :compiled-regenerate")
    (is (some? (:compiled-regenerate (:schema model-dep)))
        "model-dep has :compiled-regenerate")
    (is (some? (:compiled-regenerate (:schema model-chain)))
        "model-chain has :compiled-regenerate")
    (is (nil? (:compiled-regenerate (:schema model-beta)))
        "model-beta has NO :compiled-regenerate")
    (is (some? (:compiled-regenerate (:schema m4-branch)))
        "m4-branch has :compiled-regenerate")
    (is (some? (:compiled-prefix-regenerate (:schema m3-loop)))
        "m3-loop has :compiled-prefix-regenerate")))

(deftest select-all-test
  (testing "Select all on independent model"
    (let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 42)
          gf-c (dyn/with-key model-indep key)
          gf-h (dyn/with-key (force-handler model-indep) key)
          result-c (p/regenerate gf-c trace sel/all)
          result-h (p/regenerate gf-h trace sel/all)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "select-all, indep: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "select-all, indep: weight matches handler")
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-5)
          "select-all, indep: weight = 0 (independent sites)")))

  (testing "Select all on dependent model"
    (let [trace (make-trace-via-generate model-dep [(mx/scalar 2.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 43)
          gf-c (dyn/with-key model-dep key)
          gf-h (dyn/with-key (force-handler model-dep) key)
          result-c (p/regenerate gf-c trace sel/all)
          result-h (p/regenerate gf-h trace sel/all)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "select-all, dep: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "select-all, dep: weight matches handler")
      (is (js/isFinite (mx/item (:retval (:trace result-c))))
          "select-all, dep: retval is finite"))))

(deftest select-none-test
  (testing "Select none"
    (let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 44)
          gf-c (dyn/with-key model-indep key)
          gf-h (dyn/with-key (force-handler model-indep) key)
          result-c (p/regenerate gf-c trace sel/none)
          result-h (p/regenerate gf-h trace sel/none)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "select-none: score matches handler")
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-5)
          "select-none: weight = 0")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-6)
          "select-none: choices :x unchanged")
      (is (h/close? 0.5 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))) 1e-6)
          "select-none: choices :y unchanged")
      (is (h/close? 1.5 (mx/item (:retval (:trace result-c))) 1e-5)
          "select-none: retval = x + y = 1.5"))))

(deftest partial-selection-test
  (testing "Independent model: select :x only"
    (let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 50)
          gf-c (dyn/with-key model-indep key)
          gf-h (dyn/with-key (force-handler model-indep) key)
          result-c (p/regenerate gf-c trace (sel/select :x))
          result-h (p/regenerate gf-h trace (sel/select :x))]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "indep, select :x: score matches handler")
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-5)
          "indep, select :x: weight = 0 (independent)")
      (is (h/close? 0.5 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))) 1e-6)
          "indep, select :x: :y unchanged")))

  (testing "Independent model: select :y only"
    (let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 51)
          gf-c (dyn/with-key model-indep key)
          gf-h (dyn/with-key (force-handler model-indep) key)
          result-c (p/regenerate gf-c trace (sel/select :y))
          result-h (p/regenerate gf-h trace (sel/select :y))]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "indep, select :y: score matches handler")
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-5)
          "indep, select :y: weight = 0 (independent)")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-6)
          "indep, select :y: :x unchanged")))

  (testing "Dependent model: select :x"
    (let [trace (make-trace-via-generate model-dep [(mx/scalar 2.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 52)
          gf-c (dyn/with-key model-dep key)
          gf-h (dyn/with-key (force-handler model-dep) key)
          result-c (p/regenerate gf-c trace (sel/select :x))
          result-h (p/regenerate gf-h trace (sel/select :x))]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "dep, select :x: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "dep, select :x: weight matches handler")
      (is (h/close? 0.5 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))) 1e-6)
          "dep, select :x: :y unchanged")
      (is (js/isFinite (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))))
          "dep, select :x: :x resampled (finite)")))

  (testing "Chain model: select :x"
    (let [trace (make-trace-via-generate model-chain [(mx/scalar 0.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5) :z (mx/scalar 2.0)))
          key (rng/fresh-key 53)
          gf-c (dyn/with-key model-chain key)
          gf-h (dyn/with-key (force-handler model-chain) key)
          result-c (p/regenerate gf-c trace (sel/select :x))
          result-h (p/regenerate gf-h trace (sel/select :x))]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "chain, select :x: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "chain, select :x: weight matches handler")
      (is (h/close? 0.5 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))) 1e-6)
          "chain, select :x: :y unchanged")
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z))) 1e-6)
          "chain, select :x: :z unchanged"))))

(deftest multi-distribution-test
  (testing "Multi-distribution model"
    (let [trace (make-trace-via-generate model-multi []
                  (cm/choicemap :g (mx/scalar 0.5) :u (mx/scalar 0.3) :e (mx/scalar 0.1)))
          key (rng/fresh-key 60)]
      (testing "select :g"
        (let [gf-c (dyn/with-key model-multi key)
              gf-h (dyn/with-key (force-handler model-multi) key)
              result-c (p/regenerate gf-c trace (sel/select :g))
              result-h (p/regenerate gf-h trace (sel/select :g))]
          (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
              "multi, select :g: score matches handler")
          (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
              "multi, select :g: weight matches handler")))

      (testing "select :u"
        (let [gf-c (dyn/with-key model-multi key)
              gf-h (dyn/with-key (force-handler model-multi) key)
              result-c (p/regenerate gf-c trace (sel/select :u))
              result-h (p/regenerate gf-h trace (sel/select :u))]
          (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
              "multi, select :u: score matches handler")
          (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
              "multi, select :u: weight matches handler")))

      (testing "select all"
        (let [gf-c (dyn/with-key model-multi key)
              gf-h (dyn/with-key (force-handler model-multi) key)
              result-c (p/regenerate gf-c trace sel/all)
              result-h (p/regenerate gf-h trace sel/all)]
          (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
              "multi, select all: score matches handler")
          (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
              "multi, select all: weight matches handler"))))))

(deftest m4-branch-models-test
  (testing "flag=true, select :x"
    (let [trace (make-trace-via-generate m4-branch [true]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 70)
          gf-c (dyn/with-key m4-branch key)
          gf-h (dyn/with-key (force-handler m4-branch) key)
          result-c (p/regenerate gf-c trace (sel/select :x))
          result-h (p/regenerate gf-h trace (sel/select :x))]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "m4, flag=true, select :x: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "m4, flag=true, select :x: weight matches handler")))

  (testing "flag=false, select :x"
    (let [trace (make-trace-via-generate m4-branch [false]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 71)
          gf-c (dyn/with-key m4-branch key)
          gf-h (dyn/with-key (force-handler m4-branch) key)
          result-c (p/regenerate gf-c trace (sel/select :x))
          result-h (p/regenerate gf-h trace (sel/select :x))]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "m4, flag=false, select :x: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "m4, flag=false, select :x: weight matches handler")))

  (testing "flag=true, select all"
    (let [trace (make-trace-via-generate m4-branch [true]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 72)
          gf-c (dyn/with-key m4-branch key)
          gf-h (dyn/with-key (force-handler m4-branch) key)
          result-c (p/regenerate gf-c trace sel/all)
          result-h (p/regenerate gf-h trace sel/all)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "m4, select all: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "m4, select all: weight matches handler")))

  (testing "flag=true, select none"
    (let [trace (make-trace-via-generate m4-branch [true]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 73)
          gf-c (dyn/with-key m4-branch key)
          gf-h (dyn/with-key (force-handler m4-branch) key)
          result-c (p/regenerate gf-c trace sel/none)
          result-h (p/regenerate gf-h trace sel/none)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "m4, select none: score matches handler")
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-5)
          "m4, select none: weight = 0"))))

(deftest m3-partial-models-test
  (testing "Select :x (prefix site)"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 1.0) :y0 (mx/scalar 0.5) :y1 (mx/scalar -0.5)))
          key (rng/fresh-key 80)
          gf-c (dyn/with-key m3-loop key)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          result-c (p/regenerate gf-c trace (sel/select :x))
          result-h (p/regenerate gf-h trace (sel/select :x))]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "m3, select :x: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "m3, select :x: weight matches handler")))

  (testing "Select :y0 (dynamic site)"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 1.0) :y0 (mx/scalar 0.5) :y1 (mx/scalar -0.5)))
          key (rng/fresh-key 81)
          gf-c (dyn/with-key m3-loop key)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          result-c (p/regenerate gf-c trace (sel/select :y0))
          result-h (p/regenerate gf-h trace (sel/select :y0))]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "m3, select :y0: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "m3, select :y0: weight matches handler")))

  (testing "Select all"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 1.0) :y0 (mx/scalar 0.5) :y1 (mx/scalar -0.5)))
          key (rng/fresh-key 82)
          gf-c (dyn/with-key m3-loop key)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          result-c (p/regenerate gf-c trace sel/all)
          result-h (p/regenerate gf-h trace sel/all)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "m3, select all: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "m3, select all: weight matches handler")))

  (testing "Select none"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 1.0) :y0 (mx/scalar 0.5) :y1 (mx/scalar -0.5)))
          key (rng/fresh-key 83)
          gf-c (dyn/with-key m3-loop key)
          gf-h (dyn/with-key (force-handler m3-loop) key)
          result-c (p/regenerate gf-c trace sel/none)
          result-h (p/regenerate gf-h trace sel/none)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "m3, select none: score matches handler")
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-5)
          "m3, select none: weight = 0"))))

(deftest non-compilable-fallback-test
  (testing "Non-compilable model uses handler"
    (let [trace (p/simulate model-beta [])
          key (rng/fresh-key 90)
          gf-c (dyn/with-key model-beta key)
          gf-h (dyn/with-key (force-handler model-beta) key)
          result-c (p/regenerate gf-c trace sel/all)
          result-h (p/regenerate gf-h trace sel/all)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-5)
          "beta: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-5)
          "beta: weight matches handler")
      (is (nil? (:compiled-regenerate (:schema model-beta)))
          "beta: uses handler path (no compiled-regenerate)"))))

(deftest cross-operation-consistency-test
  (testing "regenerate(none) preserves trace"
    (let [trace (make-trace-via-generate model-indep [(mx/scalar 2.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          key (rng/fresh-key 100)
          result (p/regenerate (dyn/with-key model-indep key) trace sel/none)]
      (is (h/close? (mx/item (:score trace)) (mx/item (:score (:trace result))) 1e-5)
          "regen(none).score = old score")
      (is (h/close? 0.0 (mx/item (:weight result)) 1e-5)
          "regen(none).weight = 0")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result)) :x)))
                    1e-6)
          "regen(none).choices :x = old :x")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result)) :y)))
                    1e-6)
          "regen(none).choices :y = old :y")))

  (testing "regenerate(all) produces valid trace"
    (let [result (p/regenerate (dyn/with-key model-indep (rng/fresh-key 101))
                   (make-trace-via-generate model-indep [(mx/scalar 2.0)]
                     (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
                   sel/all)]
      (is (js/isFinite (mx/item (:score (:trace result))))
          "regen(all): finite score")))

  (testing "project(all) = new trace score after regenerate"
    (let [trace (make-trace-via-generate model-dep [(mx/scalar 2.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          result (p/regenerate (dyn/with-key model-dep (rng/fresh-key 102)) trace (sel/select :x))
          new-trace (:trace result)
          proj (p/project (dyn/with-key model-dep (rng/fresh-key 103)) new-trace sel/all)]
      (is (h/close? (mx/item (:score new-trace)) (mx/item proj) 1e-5)
          "project(all) = new trace score"))))

(cljs.test/run-tests)
