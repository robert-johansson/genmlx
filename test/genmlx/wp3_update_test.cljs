(ns genmlx.wp3-update-test
  "WP-3 tests: compiled update for static DynamicGF models."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled-ops :as compiled]))

;; ---------------------------------------------------------------------------
;; Non-assert helpers (kept)
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-update
                       :compiled-prefix :compiled-prefix-addrs)]
    (assoc gf :schema schema)))

(defn make-trace-via-generate
  "Create a trace by calling generate with all sites constrained via handler path."
  [gf args constraints]
  (let [gf-h (force-handler gf)
        {:keys [trace]} (p/generate (dyn/with-key gf-h (rng/fresh-key 1)) args constraints)]
    trace))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-model
  (dyn/auto-key
    (gen [mu]
      (let [x (trace :x (dist/gaussian mu (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 2.0)))]
        y))))

(def chain-model
  (dyn/auto-key
    (gen [a]
      (let [x (trace :x (dist/gaussian a (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 0.5)))
            z (trace :z (dist/gaussian (mx/add x y) (mx/scalar 1.0)))]
        z))))

(def multi-dist-model
  (dyn/auto-key
    (gen []
      (let [g (trace :g (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            u (trace :u (dist/uniform (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/bernoulli (mx/scalar 0.5)))
            e (trace :e (dist/exponential (mx/scalar 2.0)))]
        g))))

(def retval-model
  (dyn/auto-key
    (gen [s]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) s))
            y (trace :y (dist/gaussian (mx/scalar 0.0) s))]
        (mx/add x y)))))

(def delta-model
  (dyn/auto-key
    (gen [v]
      (let [x (trace :x (dist/delta v))
            y (trace :y (dist/gaussian x (mx/scalar 1.0)))]
        y))))

(def single-site-model
  (dyn/auto-key
    (gen [mu sigma]
      (let [x (trace :x (dist/gaussian mu sigma))]
        x))))

(def dynamic-addr-model
  (dyn/auto-key
    (gen [n]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
        (doseq [i (range (mx/item n))]
          (trace (keyword (str "y" i)) (dist/gaussian x (mx/scalar 1.0))))
        x))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest compilation-check-test
  (testing "Compilation check"
    (is (some? (:compiled-update (:schema simple-model)))
        "static model has :compiled-update")
    (is (some? (:compiled-update (:schema chain-model)))
        "chain model has :compiled-update")
    (is (some? (:compiled-update (:schema multi-dist-model)))
        "multi-dist model has :compiled-update")
    (is (nil? (:compiled-update (:schema dynamic-addr-model)))
        "dynamic-addr model has no :compiled-update")))

(deftest no-constraints-test
  (testing "No constraints (trace unchanged)"
    (let [trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          gf simple-model
          gf-h (force-handler simple-model)
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace cm/EMPTY)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace cm/EMPTY)]
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-6)
          "no-constraint: weight = 0")
      (is (h/close? (mx/item (:score trace)) (mx/item (:score (:trace result-c))) 1e-6)
          "no-constraint: score unchanged")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)))
                    1e-10)
          "no-constraint: x unchanged")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)))
                    1e-10)
          "no-constraint: y unchanged")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "no-constraint: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "no-constraint: weight matches handler"))))

(deftest single-site-constrained-test
  (testing "Single site constrained"
    (let [trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          gf simple-model
          gf-h (force-handler simple-model)
          constraints (cm/choicemap :x (mx/scalar 1.5))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 1.5 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-10)
          "single-constrained: new x = 1.5")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)))
                    1e-10)
          "single-constrained: y kept from old trace")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "single-constrained: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "single-constrained: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:discard result-c) :x)))
                    1e-10)
          "single-constrained: discard has old x")
      (is (not (cm/has-value? (cm/get-submap (:discard result-c) :y)))
          "single-constrained: discard does not have y"))))

(deftest all-sites-constrained-test
  (testing "All sites constrained"
    (let [trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          gf simple-model
          gf-h (force-handler simple-model)
          constraints (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-10)
          "all-constrained: x = 2.0")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))) 1e-10)
          "all-constrained: y = 3.0")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "all-constrained: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "all-constrained: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:discard result-c) :x)))
                    1e-10)
          "all-constrained: discard has old x"))))

(deftest dependency-chain-test
  (testing "Dependency chain (upstream change affects downstream score)"
    (let [trace (make-trace-via-generate chain-model [(mx/scalar 0.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.8) :z (mx/scalar 2.0)))
          gf chain-model
          gf-h (force-handler chain-model)
          constraints (cm/choicemap :x (mx/scalar 5.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 5.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-10)
          "chain: constrained x = 5.0")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)))
                    1e-10)
          "chain: y kept from old trace")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :z)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z)))
                    1e-10)
          "chain: z kept from old trace")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "chain: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "chain: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:discard result-c) :x)))
                    1e-10)
          "chain: discard only has x"))))

(deftest multi-distribution-test
  (testing "Multi-distribution model"
    (let [trace (make-trace-via-generate multi-dist-model []
                  (cm/choicemap :g (mx/scalar 0.3) :u (mx/scalar 0.5)
                                :b (mx/scalar 1.0) :e (mx/scalar 0.2)))
          gf multi-dist-model
          gf-h (force-handler multi-dist-model)
          constraints (cm/choicemap :g (mx/scalar 0.5))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 0.5 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :g))) 1e-10)
          "multi-dist: constrained g = 0.5")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "multi-dist: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "multi-dist: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :g)))
                    (mx/item (cm/get-value (cm/get-submap (:discard result-c) :g)))
                    1e-10)
          "multi-dist: discard correct"))))

(deftest idempotent-update-test
  (testing "Idempotent update (constrain to same values)"
    (let [trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          old-x (cm/get-value (cm/get-submap (:choices trace) :x))
          old-y (cm/get-value (cm/get-submap (:choices trace) :y))
          gf simple-model
          constraints (cm/choicemap :x old-x :y old-y)
          result (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 0.0 (mx/item (:weight result)) 1e-6)
          "idempotent: weight = 0")
      (is (h/close? (mx/item (:score trace)) (mx/item (:score (:trace result))) 1e-6)
          "idempotent: score unchanged")
      (is (h/close? (mx/item old-x) (mx/item (cm/get-value (cm/get-submap (:discard result) :x))) 1e-10)
          "idempotent: discard x = old x")
      (is (h/close? (mx/item old-y) (mx/item (cm/get-value (cm/get-submap (:discard result) :y))) 1e-10)
          "idempotent: discard y = old y"))))

(deftest delta-distribution-test
  (testing "Delta distribution"
    (let [trace (make-trace-via-generate delta-model [(mx/scalar 3.0)]
                  (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 2.5)))
          gf delta-model
          gf-h (force-handler delta-model)
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace cm/EMPTY)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace cm/EMPTY)]
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-10)
          "delta: x = 3.0 (deterministic)")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "delta: score matches handler")
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-6)
          "delta: weight = 0"))))

(deftest round-trip-chained-updates-test
  (testing "Round-trip: generate -> update -> update"
    (let [trace (make-trace-via-generate simple-model [(mx/scalar 0.0)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          gf simple-model
          gf-h (force-handler simple-model)
          c1 (cm/choicemap :x (mx/scalar 1.0))
          r1-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace c1)
          r1-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace c1)
          c2 (cm/choicemap :y (mx/scalar -1.0))
          r2-c (p/update (dyn/with-key gf (rng/fresh-key 88)) (:trace r1-c) c2)
          r2-h (p/update (dyn/with-key gf-h (rng/fresh-key 88)) (:trace r1-h) c2)]
      (is (h/close? (mx/item (:score (:trace r1-h))) (mx/item (:score (:trace r1-c))) 1e-6)
          "chain-update-1: score matches handler")
      (is (h/close? (mx/item (:weight r1-h)) (mx/item (:weight r1-c)) 1e-6)
          "chain-update-1: weight matches handler")
      (is (h/close? (mx/item (:score (:trace r2-h))) (mx/item (:score (:trace r2-c))) 1e-6)
          "chain-update-2: score matches handler")
      (is (h/close? (mx/item (:weight r2-h)) (mx/item (:weight r2-c)) 1e-6)
          "chain-update-2: weight matches handler")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace r2-c)) :x))) 1e-10)
          "chain-update-2: x = 1.0 (from step 2)")
      (is (h/close? -1.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace r2-c)) :y))) 1e-10)
          "chain-update-2: y = -1.0 (from step 3)"))))

(deftest mathematical-correctness-test
  (testing "weight = log-prob(new) - log-prob(old)"
    (let [mu (mx/scalar 0.0) sigma (mx/scalar 1.0)
          trace (make-trace-via-generate single-site-model [mu sigma]
                  (cm/choicemap :x (mx/scalar 0.3)))
          old-x (cm/get-value (cm/get-submap (:choices trace) :x))
          new-x (mx/scalar 2.0)
          constraints (cm/choicemap :x new-x)
          result (p/update (dyn/with-key single-site-model (rng/fresh-key 99)) trace constraints)
          log2pi (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
          lp-new (- (- log2pi) (* 0.5 (* 2.0 2.0)))
          lp-old (- (- log2pi) (* 0.5 (* 0.3 0.3)))]
      (is (h/close? (- lp-new lp-old) (mx/item (:weight result)) 1e-5)
          "analytical: weight = new_lp - old_lp")))

  (testing "no-op update: weight exactly 0"
    (let [trace (make-trace-via-generate single-site-model [(mx/scalar 0.0) (mx/scalar 1.0)]
                  (cm/choicemap :x (mx/scalar 0.5)))
          result (p/update (dyn/with-key single-site-model (rng/fresh-key 99)) trace cm/EMPTY)]
      (is (h/close? 0.0 (mx/item (:weight result)) 1e-10)
          "no-op: weight = 0")))

  (testing "extreme constraint: weight is large negative"
    (let [trace (make-trace-via-generate single-site-model [(mx/scalar 0.0) (mx/scalar 1.0)]
                  (cm/choicemap :x (mx/scalar 0.0)))
          constraints (cm/choicemap :x (mx/scalar 100.0))
          result (p/update (dyn/with-key single-site-model (rng/fresh-key 99)) trace constraints)]
      (is (< (mx/item (:weight result)) -100.0)
          "extreme: weight is very negative")))

  (testing "constrain to mode: weight is non-negative"
    (let [trace (make-trace-via-generate single-site-model [(mx/scalar 0.0) (mx/scalar 1.0)]
                  (cm/choicemap :x (mx/scalar 3.0)))
          constraints (cm/choicemap :x (mx/scalar 0.0))
          result (p/update (dyn/with-key single-site-model (rng/fresh-key 99)) trace constraints)]
      (is (>= (mx/item (:weight result)) (- 1e-10))
          "mode: weight >= 0 when constraining to mode"))))

(deftest return-value-test
  (testing "Return value"
    (let [trace (make-trace-via-generate retval-model [(mx/scalar 1.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0)))
          gf retval-model
          gf-h (force-handler retval-model)
          constraints (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 5.0 (mx/item (:retval (:trace result-c))) 1e-10)
          "retval: x+y = 5.0")
      (is (h/close? (mx/item (:retval (:trace result-h))) (mx/item (:retval (:trace result-c))) 1e-10)
          "retval: matches handler"))))

(deftest edge-cases-test
  (testing "Single-site model"
    (let [trace (make-trace-via-generate single-site-model [(mx/scalar 0.0) (mx/scalar 1.0)]
                  (cm/choicemap :x (mx/scalar 0.5)))
          gf single-site-model
          gf-h (force-handler single-site-model)
          constraints (cm/choicemap :x (mx/scalar 1.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "single-site: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "single-site: weight matches handler")))

  (testing "Empty constraints ChoiceMap"
    (let [trace (make-trace-via-generate chain-model [(mx/scalar 0.0)]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.8) :z (mx/scalar 2.0)))
          result (p/update (dyn/with-key chain-model (rng/fresh-key 99)) trace cm/EMPTY)]
      (is (h/close? 0.0 (mx/item (:weight result)) 1e-6)
          "empty-cm: weight = 0")
      (is (= cm/EMPTY (:discard result))
          "empty-cm: discard is empty"))))

(deftest accessor-test
  (testing "get-compiled-update accessor"
    (is (fn? (compiled/get-compiled-update simple-model))
        "get-compiled-update returns fn for static model")
    (is (nil? (compiled/get-compiled-update dynamic-addr-model))
        "get-compiled-update returns nil for dynamic model")))

(cljs.test/run-tests)
