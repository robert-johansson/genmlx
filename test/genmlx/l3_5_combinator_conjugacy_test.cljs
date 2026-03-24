(ns genmlx.l3-5-combinator-conjugacy-test
  "Level 3.5 WP-2: Combinator-aware conjugacy test suite.
   Verifies that kernel-internal conjugate pairs are detected and
   auto-handlers fire correctly inside unfold, map, scan, switch, and mix."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.combinators :as comb]
            [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- make-handler-only-kernel
  "Create a DynamicGF kernel with auto-handlers but no compiled paths."
  [gf]
  (let [s (-> (:schema gf)
              (dissoc :compiled-generate :compiled-simulate :compiled-update))]
    (dyn/auto-key (dyn/->DynamicGF (:body-fn gf) (:source gf) s))))

(defn- make-vanilla-kernel
  "Create a DynamicGF kernel with NO auto-handlers and NO compiled paths."
  [gf]
  (let [s (-> (:schema gf)
              (dissoc :compiled-generate :compiled-simulate :compiled-update :auto-handlers)
              (assoc :conjugate-pairs []))]
    (dyn/auto-key (dyn/->DynamicGF (:body-fn gf) (:source gf) s))))

(defn- deterministic-weight? [weights]
  (apply = weights))

(defn- weight-variance [weights]
  (let [n (count weights)
        mean (/ (reduce + weights) n)]
    (/ (reduce + (map #(* (- % mean) (- % mean)) weights)) n)))

;; ---------------------------------------------------------------------------
;; Standard kernels for testing
;; ---------------------------------------------------------------------------

(def nn-base-kernel
  (gen [t state]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          z  (trace :z (dist/gaussian mu (mx/scalar 1.0)))]
      (mx/add state z))))

(def bb-base-kernel
  (gen [t state]
    (let [p (trace :p (dist/beta-dist (mx/scalar 2.0) (mx/scalar 2.0)))
          x (trace :x (dist/bernoulli p))]
      (mx/add state x))))

(def nn-map-base-kernel
  (gen [x]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          z  (trace :z (dist/gaussian mu (mx/scalar 1.0)))]
      z)))

(def nn-scan-base-kernel
  (gen [carry input]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          z  (trace :z (dist/gaussian mu (mx/scalar 1.0)))]
      [(mx/add carry z) z])))

(def nn-branch-base
  (gen []
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          z  (trace :z (dist/gaussian mu (mx/scalar 1.0)))]
      z)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest schema-detection-test
  (testing "Schema detection in combinator kernels"
    (let [nn-k (make-handler-only-kernel nn-base-kernel)
          bb-k (make-handler-only-kernel bb-base-kernel)]
      (is (some? (:auto-handlers (:schema nn-k))) "NN kernel has auto-handlers")
      (is (= #{:mu :z} (set (keys (:auto-handlers (:schema nn-k)))))
          "NN kernel auto-handlers include :mu and :z")
      (is (= 1 (count (:conjugate-pairs (:schema nn-k)))) "NN kernel has conjugate-pairs")
      (is (= :normal-normal (:family (first (:conjugate-pairs (:schema nn-k)))))
          "NN kernel conjugate family is :normal-normal")
      (is (some? (:auto-handlers (:schema bb-k))) "BB kernel has auto-handlers")
      (is (= #{:p :x} (set (keys (:auto-handlers (:schema bb-k)))))
          "BB kernel auto-handlers include :p and :x")
      (is (= :beta-bernoulli (:family (first (:conjugate-pairs (:schema bb-k)))))
          "BB kernel conjugate family is :beta-bernoulli"))))

(deftest unfold-nn-kernel-test
  (testing "Unfold with Normal-Normal kernel"
    (let [nn-k (make-handler-only-kernel nn-base-kernel)
          unfold-nn (comb/unfold-combinator nn-k)
          obs-1step (cm/set-choice cm/EMPTY [0 :z] (mx/scalar 5.0))
          weights-1 (mapv (fn [_]
                            (mx/item (:weight (p/generate unfold-nn [3 (mx/scalar 0.0)] obs-1step))))
                          (range 10))]
      (is (deterministic-weight? weights-1)
          "Unfold NN: deterministic weight (1 constrained step)")
      (is (h/close? -3.35 (first weights-1) 0.05) "Unfold NN: marginal LL value")

      (let [result (p/generate unfold-nn [3 (mx/scalar 0.0)] obs-1step)
            step0 (cm/get-submap (:choices (:trace result)) 0)
            mu-val (mx/item (cm/get-value (cm/get-submap step0 :mu)))
            z-val (mx/item (cm/get-value (cm/get-submap step0 :z)))]
        (is (h/close? 5.0 z-val 0.001) "Unfold NN: constrained :z equals observation")
        (is (> mu-val 4.0) "Unfold NN: :mu posterior mean pulled toward obs (>4.0)"))

      (let [obs-multi (-> cm/EMPTY
                          (cm/set-choice [0 :z] (mx/scalar 5.0))
                          (cm/set-choice [1 :z] (mx/scalar 3.0))
                          (cm/set-choice [2 :z] (mx/scalar 7.0)))
            weights-m (mapv (fn [_]
                              (mx/item (:weight (p/generate unfold-nn [3 (mx/scalar 0.0)] obs-multi))))
                            (range 10))]
        (is (deterministic-weight? weights-m)
            "Unfold NN: deterministic weight (3 constrained steps)")))))

(deftest map-nn-kernel-test
  (testing "Map with Normal-Normal kernel"
    (let [nn-k (make-handler-only-kernel nn-map-base-kernel)
          map-nn (comb/map-combinator nn-k)
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :z] (mx/scalar 5.0))
                  (cm/set-choice [1 :z] (mx/scalar 3.0)))
          args [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
          weights (mapv (fn [_]
                          (mx/item (:weight (p/generate map-nn args obs))))
                        (range 10))]
      (is (deterministic-weight? weights) "Map NN: deterministic weight")
      (is (< (first weights) -6.0) "Map NN: weight is sum of 2 marginal LLs")

      (let [result (p/generate map-nn args obs)
            elem0 (cm/get-submap (:choices (:trace result)) 0)
            mu0 (mx/item (cm/get-value (cm/get-submap elem0 :mu)))]
        (is (> mu0 4.0) "Map NN: element 0 :mu pulled toward obs=5")))))

(deftest scan-nn-kernel-test
  (testing "Scan with Normal-Normal kernel"
    (let [nn-k (make-handler-only-kernel nn-scan-base-kernel)
          scan-nn (comb/scan-combinator nn-k)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          obs (cm/set-choice cm/EMPTY [0 :z] (mx/scalar 5.0))
          weights (mapv (fn [_]
                          (mx/item (:weight (p/generate scan-nn [(mx/scalar 0.0) inputs] obs))))
                        (range 10))]
      (is (deterministic-weight? weights) "Scan NN: deterministic weight")
      (is (h/close? -3.35 (first weights) 0.05) "Scan NN: marginal LL same as unfold"))))

(deftest switch-nn-branch-test
  (testing "Switch with Normal-Normal branch"
    (let [nn-b (make-handler-only-kernel nn-branch-base)
          other-b (dyn/auto-key (gen [] (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))))
          switch-nn (comb/switch-combinator nn-b other-b)
          obs (cm/set-value cm/EMPTY :z (mx/scalar 5.0))
          weights (mapv (fn [_]
                          (mx/item (:weight (p/generate switch-nn [0] obs))))
                        (range 10))]
      (is (deterministic-weight? weights) "Switch NN: deterministic weight (branch 0)")
      (is (h/close? -3.35 (first weights) 0.05) "Switch NN: marginal LL")

      (let [obs-b1 (cm/set-value cm/EMPTY :x (mx/scalar 2.0))
            result (p/generate switch-nn [1] obs-b1)
            w (mx/item (:weight result))
            expected-w (* -0.5 (+ (js/Math.log (* 2 js/Math.PI)) 4.0))]
        (is (h/close? expected-w w 0.01) "Switch: branch 1 standard Gaussian weight")))))

(deftest unfold-bb-kernel-test
  (testing "Unfold with Beta-Bernoulli kernel"
    (let [bb-k (make-handler-only-kernel bb-base-kernel)
          unfold-bb (comb/unfold-combinator bb-k)
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :x] (mx/scalar 1.0))
                  (cm/set-choice [1 :x] (mx/scalar 0.0)))
          weights (mapv (fn [_]
                          (mx/item (:weight (p/generate unfold-bb [3 (mx/scalar 0.0)] obs))))
                        (range 10))]
      (is (deterministic-weight? weights) "Unfold BB: deterministic weight")
      (is (h/close? -0.693 (first weights) 0.7) "Unfold BB: step0 marginal LL approx log(0.5)")

      (let [result (p/generate unfold-bb [3 (mx/scalar 0.0)] obs)
            step0 (cm/get-submap (:choices (:trace result)) 0)
            x0 (mx/item (cm/get-value (cm/get-submap step0 :x)))]
        (is (h/close? 1.0 x0 0.001) "Unfold BB: :x at step 0 = 1.0")))))

(deftest edge-cases-test
  (testing "Non-conjugate kernel: no auto-handlers"
    (let [non-conj-kernel (dyn/auto-key
                            (gen [t state]
                              (let [z (trace :z (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
                                (mx/add state z))))
          s (-> (:schema non-conj-kernel)
                (dissoc :compiled-generate :compiled-simulate :compiled-update))
          k (dyn/auto-key (dyn/->DynamicGF (:body-fn non-conj-kernel) (:source non-conj-kernel) s))
          unfold-nc (comb/unfold-combinator k)
          obs (cm/set-choice cm/EMPTY [0 :z] (mx/scalar 5.0))
          weights (mapv (fn [_]
                          (mx/item (:weight (p/generate unfold-nc [3 (mx/scalar 0.0)] obs))))
                        (range 5))]
      (is (nil? (:auto-handlers (:schema k))) "Non-conjugate kernel: no auto-handlers")
      (is (every? #(< % 0.0) weights) "Non-conjugate kernel: weight is valid log-prob")))

  (testing "Empty constraints: weight=0"
    (let [nn-k (make-handler-only-kernel nn-base-kernel)
          unfold-nn (comb/unfold-combinator nn-k)
          result (p/generate unfold-nn [3 (mx/scalar 0.0)] cm/EMPTY)]
      (is (h/close? 0.0 (mx/item (:weight result)) 0.001)
          "Unfold NN: no constraints -> weight=0"))))

(deftest variance-reduction-benchmark-test
  (testing "Variance reduction benchmark"
    (let [nn-k (make-handler-only-kernel nn-base-kernel)
          nn-v (make-vanilla-kernel nn-base-kernel)
          unfold-auto (comb/unfold-combinator nn-k)
          unfold-vanilla (comb/unfold-combinator nn-v)
          T 20
          obs (reduce (fn [cm t]
                        (cm/set-choice cm [t :z] (mx/scalar (+ 3.0 (* 0.1 t)))))
                      cm/EMPTY
                      (range T))
          n-trials 15
          auto-weights (mapv (fn [_]
                               (mx/item (:weight (p/generate unfold-auto [T (mx/scalar 0.0)] obs))))
                             (range n-trials))
          vanilla-weights (mapv (fn [_]
                                  (mx/item (:weight (p/generate unfold-vanilla [T (mx/scalar 0.0)] obs))))
                                (range n-trials))
          auto-var (weight-variance auto-weights)
          vanilla-var (weight-variance vanilla-weights)]
      (is (< auto-var 0.001) "Benchmark: auto-handler has zero variance")
      (is (> vanilla-var 100.0) "Benchmark: vanilla has high variance")
      (is (or (zero? auto-var) (> (/ vanilla-var (max auto-var 1e-10)) 1000.0))
          "Benchmark: variance reduction is dramatic"))))

(cljs.test/run-tests)
