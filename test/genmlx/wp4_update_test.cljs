(ns genmlx.wp4-update-test
  "WP-4 tests: compiled update for M4 (branch-rewritten) and M3 (partial prefix) models."
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
            [genmlx.compiled :as compiled]))

;; ---------------------------------------------------------------------------
;; Non-assert helpers (kept)
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip all compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-update
                       :compiled-prefix :compiled-prefix-addrs
                       :compiled-prefix-generate :compiled-prefix-update)]
    (assoc gf :schema schema)))

(defn make-trace-via-generate
  "Create a trace by calling generate with all sites constrained via handler path."
  [gf args constraints]
  (let [gf-h (force-handler gf)
        {:keys [trace]} (p/generate (dyn/with-key gf-h (rng/fresh-key 1)) args constraints)]
    trace))

;; ---------------------------------------------------------------------------
;; M4 test models (branch-rewritten)
;; ---------------------------------------------------------------------------

(def m4-simple
  (dyn/auto-key
    (gen [flag]
      (let [x (if flag
                (trace :x (dist/gaussian (mx/scalar 1.0) (mx/scalar 0.5)))
                (trace :x (dist/gaussian (mx/scalar -1.0) (mx/scalar 2.0))))
            y (trace :y (dist/gaussian x (mx/scalar 1.0)))]
        y))))

(def m4-multi-site
  (dyn/auto-key
    (gen [flag]
      (let [a (if flag
                (trace :a (dist/gaussian (mx/scalar 5.0) (mx/scalar 1.0)))
                (trace :a (dist/gaussian (mx/scalar 0.0) (mx/scalar 3.0))))
            b (if flag
                (trace :b (dist/gaussian a (mx/scalar 0.5)))
                (trace :b (dist/gaussian (mx/multiply a (mx/scalar 2.0)) (mx/scalar 1.0))))]
        (mx/add a b)))))

;; ---------------------------------------------------------------------------
;; M3 test models (partial prefix with dynamic tail)
;; ---------------------------------------------------------------------------

(def m3-loop
  (dyn/auto-key
    (gen [n]
      (let [x (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            y (trace :y (dist/gaussian x (mx/scalar 2.0)))]
        (doseq [i (range (mx/item n))]
          (trace (keyword (str "z" i)) (dist/gaussian y (mx/scalar 0.5))))
        y))))

(def m3-dep-prefix
  (dyn/auto-key
    (gen []
      (let [a (trace :a (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
            b (trace :b (dist/gaussian a (mx/scalar 0.5)))]
        (doseq [i (range 2)]
          (trace (keyword (str "c" i)) (dist/gaussian (mx/add a b) (mx/scalar 1.0))))
        (mx/add a b)))))

;; ===========================================================================
;; PART A: M4 Branch-Rewritten
;; ===========================================================================

(deftest m4-compilation-check-test
  (testing "M4 compilation check"
    (is (some? (:compiled-update (:schema m4-simple)))
        "m4-simple has :compiled-update")
    (is (some? (:compiled-update (:schema m4-multi-site)))
        "m4-multi-site has :compiled-update")
    (is (and (some? (:compiled-prefix-update (:schema m3-loop)))
             (nil? (:compiled-update (:schema m3-loop))))
        "m3-loop has :compiled-prefix-update (not :compiled-update)")))

(deftest m4-no-constraints-test
  (testing "M4 no constraints"
    (let [trace (make-trace-via-generate m4-simple [true]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          gf m4-simple
          gf-h (force-handler m4-simple)
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace cm/EMPTY)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace cm/EMPTY)]
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-6)
          "m4 no-cst: weight = 0")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m4 no-cst: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m4 no-cst: weight matches handler"))))

(deftest m4-single-site-constrained-test
  (testing "M4 single site constrained"
    (let [trace (make-trace-via-generate m4-simple [true]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          gf m4-simple
          gf-h (force-handler m4-simple)
          constraints (cm/choicemap :x (mx/scalar 2.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-10)
          "m4 single: x = 2.0")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y)))
                    1e-10)
          "m4 single: y kept")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m4 single: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m4 single: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:discard result-c) :x)))
                    1e-10)
          "m4 single: discard has old x"))))

(deftest m4-all-sites-constrained-test
  (testing "M4 all sites constrained"
    (let [trace (make-trace-via-generate m4-simple [true]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          gf m4-simple
          gf-h (force-handler m4-simple)
          constraints (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 4.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-10)
          "m4 all: x = 3.0")
      (is (h/close? 4.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :y))) 1e-10)
          "m4 all: y = 4.0")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m4 all: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m4 all: weight matches handler"))))

(deftest m4-branch-conditions-test
  (testing "M4 true branch"
    (let [trace (make-trace-via-generate m4-simple [true]
                  (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 0.5)))
          gf m4-simple
          gf-h (force-handler m4-simple)
          constraints (cm/choicemap :x (mx/scalar 0.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m4 true-branch: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m4 true-branch: weight matches handler")))

  (testing "M4 false branch"
    (let [trace (make-trace-via-generate m4-simple [false]
                  (cm/choicemap :x (mx/scalar -0.5) :y (mx/scalar 0.0)))
          gf m4-simple
          gf-h (force-handler m4-simple)
          constraints (cm/choicemap :x (mx/scalar 1.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m4 false-branch: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m4 false-branch: weight matches handler"))))

(deftest m4-retval-test
  (testing "M4 retval"
    (let [trace (make-trace-via-generate m4-multi-site [true]
                  (cm/choicemap :a (mx/scalar 2.0) :b (mx/scalar 3.0)))
          gf m4-multi-site
          gf-h (force-handler m4-multi-site)
          constraints (cm/choicemap :a (mx/scalar 4.0) :b (mx/scalar 5.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 9.0 (mx/item (:retval (:trace result-c))) 1e-10)
          "m4 retval: a+b = 9.0")
      (is (h/close? (mx/item (:retval (:trace result-h))) (mx/item (:retval (:trace result-c))) 1e-10)
          "m4 retval: matches handler"))))

(deftest m4-multi-site-dependency-test
  (testing "M4 multi-site with dependency chain"
    (let [trace (make-trace-via-generate m4-multi-site [true]
                  (cm/choicemap :a (mx/scalar 5.0) :b (mx/scalar 6.0)))
          gf m4-multi-site
          gf-h (force-handler m4-multi-site)
          constraints (cm/choicemap :a (mx/scalar 10.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 10.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :a))) 1e-10)
          "m4 dep: a = 10.0")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :b)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :b)))
                    1e-10)
          "m4 dep: b kept")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m4 dep: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m4 dep: weight matches handler"))))

(deftest m4-idempotent-test
  (testing "M4 idempotent"
    (let [trace (make-trace-via-generate m4-simple [true]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)))
          gf m4-simple
          constraints (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0))
          result (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 0.0 (mx/item (:weight result)) 1e-6)
          "m4 idempotent: weight = 0")
      (is (h/close? (mx/item (:score trace)) (mx/item (:score (:trace result))) 1e-6)
          "m4 idempotent: score unchanged"))))

;; ===========================================================================
;; PART B: M3 Partial Prefix
;; ===========================================================================

(deftest m3-compilation-check-test
  (testing "M3 compilation check"
    (is (some? (:compiled-prefix-update (:schema m3-loop)))
        "m3-loop has :compiled-prefix-update")
    (is (some? (:compiled-prefix-update (:schema m3-dep-prefix)))
        "m3-dep-prefix has :compiled-prefix-update")))

(deftest m3-no-constraints-test
  (testing "M3 no constraints"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                                :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
          gf m3-loop
          gf-h (force-handler m3-loop)
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace cm/EMPTY)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace cm/EMPTY)]
      (is (h/close? 0.0 (mx/item (:weight result-c)) 1e-6)
          "m3 no-cst: weight = 0")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m3 no-cst: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m3 no-cst: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)))
                    1e-10)
          "m3 no-cst: x unchanged"))))

(deftest m3-prefix-site-constrained-test
  (testing "M3 prefix site constrained"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                                :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
          gf m3-loop
          gf-h (force-handler m3-loop)
          constraints (cm/choicemap :x (mx/scalar 3.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-10)
          "m3 prefix-cst: x = 3.0")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :z0)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z0)))
                    1e-10)
          "m3 prefix-cst: z0 kept")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m3 prefix-cst: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m3 prefix-cst: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:discard result-c) :x)))
                    1e-10)
          "m3 prefix-cst: discard has old x"))))

(deftest m3-dynamic-site-constrained-test
  (testing "M3 dynamic site constrained"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                                :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
          gf m3-loop
          gf-h (force-handler m3-loop)
          constraints (cm/choicemap :z0 (mx/scalar 5.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 5.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z0))) 1e-10)
          "m3 dyn-cst: z0 = 5.0")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x)))
                    1e-10)
          "m3 dyn-cst: x kept")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m3 dyn-cst: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m3 dyn-cst: weight matches handler"))))

(deftest m3-both-constrained-test
  (testing "M3 both prefix + dynamic constrained"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                                :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
          gf m3-loop
          gf-h (force-handler m3-loop)
          constraints (cm/choicemap :x (mx/scalar 2.0) :z1 (mx/scalar 3.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 2.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :x))) 1e-10)
          "m3 both: x = 2.0")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :z1))) 1e-10)
          "m3 both: z1 = 3.0")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m3 both: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m3 both: weight matches handler")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :x)))
                    (mx/item (cm/get-value (cm/get-submap (:discard result-c) :x)))
                    1e-10)
          "m3 both: discard has old x")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :z1)))
                    (mx/item (cm/get-value (cm/get-submap (:discard result-c) :z1)))
                    1e-10)
          "m3 both: discard has old z1"))))

(deftest m3-idempotent-test
  (testing "M3 idempotent"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                                :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
          gf m3-loop
          constraints (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                                    :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2))
          result (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 0.0 (mx/item (:weight result)) 1e-6)
          "m3 idempotent: weight = 0")
      (is (h/close? (mx/item (:score trace)) (mx/item (:score (:trace result))) 1e-6)
          "m3 idempotent: score unchanged"))))

(deftest m3-chained-updates-test
  (testing "M3 chained updates"
    (let [trace (make-trace-via-generate m3-loop [(mx/scalar 2)]
                  (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0)
                                :z0 (mx/scalar 0.8) :z1 (mx/scalar 1.2)))
          gf m3-loop
          gf-h (force-handler m3-loop)
          c1 (cm/choicemap :x (mx/scalar 2.0))
          r1-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace c1)
          r1-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace c1)
          c2 (cm/choicemap :z0 (mx/scalar 4.0))
          r2-c (p/update (dyn/with-key gf (rng/fresh-key 88)) (:trace r1-c) c2)
          r2-h (p/update (dyn/with-key gf-h (rng/fresh-key 88)) (:trace r1-h) c2)]
      (is (h/close? (mx/item (:score (:trace r1-h))) (mx/item (:score (:trace r1-c))) 1e-6)
          "m3 chain-1: score matches handler")
      (is (h/close? (mx/item (:weight r1-h)) (mx/item (:weight r1-c)) 1e-6)
          "m3 chain-1: weight matches handler")
      (is (h/close? (mx/item (:score (:trace r2-h))) (mx/item (:score (:trace r2-c))) 1e-6)
          "m3 chain-2: score matches handler")
      (is (h/close? (mx/item (:weight r2-h)) (mx/item (:weight r2-c)) 1e-6)
          "m3 chain-2: weight matches handler"))))

(deftest m3-retval-test
  (testing "M3 retval"
    (let [trace (make-trace-via-generate m3-dep-prefix []
                  (cm/choicemap :a (mx/scalar 1.0) :b (mx/scalar 2.0)
                                :c0 (mx/scalar 3.0) :c1 (mx/scalar 4.0)))
          gf m3-dep-prefix
          gf-h (force-handler m3-dep-prefix)
          constraints (cm/choicemap :a (mx/scalar 2.0) :b (mx/scalar 3.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 5.0 (mx/item (:retval (:trace result-c))) 1e-10)
          "m3 retval: a+b = 5.0")
      (is (h/close? (mx/item (:retval (:trace result-h))) (mx/item (:retval (:trace result-c))) 1e-10)
          "m3 retval: matches handler"))))

(deftest m3-dep-prefix-upstream-test
  (testing "M3 dep-prefix upstream change"
    (let [trace (make-trace-via-generate m3-dep-prefix []
                  (cm/choicemap :a (mx/scalar 1.0) :b (mx/scalar 2.0)
                                :c0 (mx/scalar 3.0) :c1 (mx/scalar 4.0)))
          gf m3-dep-prefix
          gf-h (force-handler m3-dep-prefix)
          constraints (cm/choicemap :a (mx/scalar 5.0))
          result-c (p/update (dyn/with-key gf (rng/fresh-key 99)) trace constraints)
          result-h (p/update (dyn/with-key gf-h (rng/fresh-key 99)) trace constraints)]
      (is (h/close? 5.0 (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :a))) 1e-10)
          "m3 dep: a = 5.0")
      (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices trace) :b)))
                    (mx/item (cm/get-value (cm/get-submap (:choices (:trace result-c)) :b)))
                    1e-10)
          "m3 dep: b kept")
      (is (h/close? (mx/item (:score (:trace result-h))) (mx/item (:score (:trace result-c))) 1e-6)
          "m3 dep: score matches handler")
      (is (h/close? (mx/item (:weight result-h)) (mx/item (:weight result-c)) 1e-6)
          "m3 dep: weight matches handler"))))

(cljs.test/run-tests)
