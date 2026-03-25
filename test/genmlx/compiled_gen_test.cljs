(ns genmlx.compiled-gen-test
  "Tests for compiled gen functions.
   Verifies correctness and measures speedup of mx/compile-fn
   applied to the full vgenerate + gradient pipeline."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled-gen :as cg]
            [genmlx.inference.differentiable :as diff]
            [genmlx.inference.fisher :as fisher])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Model setup (same as fisher_test.cljs)
;; ---------------------------------------------------------------------------

(def K-obs 10)
(def true-mu 3.0)
(def obs-data (mapv (fn [i] (+ true-mu (* 0.5 (- i 4.5)))) (range K-obs)))

(def model-1
  (gen []
    (let [mu (param :mu 0.0)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu 1.0)))
      mu)))

(def obs-1
  (apply cm/choicemap
    (mapcat (fn [i] [(keyword (str "y" i)) (mx/scalar (nth obs-data i))])
            (range K-obs))))

;; ---------------------------------------------------------------------------
;; 2D model (mu + log-sigma)
;; ---------------------------------------------------------------------------

(def model-2
  (gen []
    (let [mu (param :mu 0.0)
          log-sigma (param :log-sigma 0.0)
          sigma (mx/exp log-sigma)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu sigma)))
      mu)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest compiled-log-ml-correctness-test
  (testing "Compiled log-ML matches uncompiled"
    (let [key (rng/fresh-key 42)
          params (mx/array [true-mu])
          {:keys [log-ml]} (diff/log-ml-gradient
                              {:n-particles 5000 :key key}
                              model-1 [] obs-1 [:mu] params)
          _ (mx/materialize! log-ml)
          uncompiled-val (mx/item log-ml)
          compiled-loss (cg/compile-log-ml
                          {:n-particles 5000 :key key}
                          model-1 [] obs-1 [:mu])
          compiled-val (- (mx/item (compiled-loss params)))]
      (is (h/close? uncompiled-val compiled-val 0.01) "Compiled ~ uncompiled"))))

(deftest compiled-gradient-correctness-test
  (testing "Compiled gradient matches uncompiled"
    (let [key (rng/fresh-key 42)
          params (mx/array [true-mu])
          {:keys [grad]} (diff/log-ml-gradient
                            {:n-particles 5000 :key key}
                            model-1 [] obs-1 [:mu] params)
          _ (mx/materialize! grad)
          uncompiled-grad (mx/item (mx/index grad 0))
          compiled-vg (cg/compile-log-ml-gradient
                        {:n-particles 5000 :key key}
                        model-1 [] obs-1 [:mu])
          [neg-lml grad-c] (compiled-vg params)
          _ (mx/materialize! neg-lml grad-c)
          compiled-grad (mx/item (mx/index grad-c 0))]
      (is (h/close? uncompiled-grad compiled-grad 0.01) "Compiled grad ~ uncompiled"))))

(deftest compiled-determinism-test
  (testing "Compiled gradient determinism (same key -> same result)"
    (let [key (rng/fresh-key 42)
          compiled-vg (cg/compile-log-ml-gradient
                        {:n-particles 5000 :key key}
                        model-1 [] obs-1 [:mu])
          params (mx/array [true-mu])
          [v1 g1] (compiled-vg params)
          [v2 g2] (compiled-vg params)]
      (mx/materialize! v1 v2 g1 g2)
      (is (h/close? (mx/item v1) (mx/item v2) 1e-6) "Deterministic"))))

(deftest compiled-different-params-test
  (testing "Different params (same shape -> cache hit)"
    (let [key (rng/fresh-key 42)
          compiled-vg (cg/compile-log-ml-gradient
                        {:n-particles 5000 :key key}
                        model-1 [] obs-1 [:mu])
          [v1 _] (compiled-vg (mx/array [0.0]))
          [v2 _] (compiled-vg (mx/array [3.0]))
          [v3 _] (compiled-vg (mx/array [6.0]))]
      (mx/materialize! v1 v2 v3)
      (let [l1 (- (mx/item v1))
            l2 (- (mx/item v2))
            l3 (- (mx/item v3))]
        (is (> l2 l1) "log-ML(3) > log-ML(0)")
        (is (> l2 l3) "log-ML(3) > log-ML(6)")))))

(deftest compiled-speedup-test
  (testing "Speedup benchmark"
    (let [key (rng/fresh-key 42)
          params (mx/array [true-mu])
          n-calls 20
          t0 (js/Date.now)
          _ (dotimes [_ n-calls]
              (let [{:keys [grad log-ml]} (diff/log-ml-gradient
                                            {:n-particles 5000 :key key}
                                            model-1 [] obs-1 [:mu] params)]
                (mx/materialize! grad log-ml)))
          t-uncompiled (- (js/Date.now) t0)
          compiled-vg (cg/compile-log-ml-gradient
                        {:n-particles 5000 :key key}
                        model-1 [] obs-1 [:mu])
          _ (dotimes [_ 2]
              (let [[v g] (compiled-vg params)] (mx/materialize! v g)))
          t1 (js/Date.now)
          _ (dotimes [_ n-calls]
              (let [[v g] (compiled-vg params)]
                (mx/materialize! v g)))
          t-compiled (- (js/Date.now) t1)
          speedup (/ t-uncompiled (max t-compiled 1))]
      (is (> speedup 1.0) "Compiled is faster"))))

(deftest compiled-2d-gradient-test
  (testing "2D compiled gradient"
    (let [key (rng/fresh-key 77)
          params (mx/array [true-mu 0.0])
          compiled-vg (cg/compile-log-ml-gradient
                        {:n-particles 5000 :key key}
                        model-2 [] obs-1 [:mu :log-sigma])
          [neg-lml grad] (compiled-vg params)]
      (mx/materialize! neg-lml grad)
      (let [g0 (mx/item (mx/index grad 0))
            g1 (mx/item (mx/index grad 1))]
        (is (h/close? 0.0 g0 1.0) "grad[mu] ~ 0 at MLE")
        (is (js/isFinite g1) "grad[log-sigma] is finite")))))

(deftest compiled-fisher-test
  (testing "Compiled Fisher correctness and speedup"
    (let [key (rng/fresh-key 42)
          params (mx/array [true-mu])
          t0 (js/Date.now)
          {:keys [fisher]} (fisher/observed-fisher
                             {:n-particles 5000 :key key}
                             model-1 [] obs-1 [:mu] params)
          _ (mx/materialize! fisher)
          t-uncompiled (- (js/Date.now) t0)
          f-uncompiled (mx/item (mx/mat-get fisher 0 0))
          t1 (js/Date.now)
          {:keys [fisher]} (fisher/observed-fisher
                             {:n-particles 5000 :key key :compiled? true}
                             model-1 [] obs-1 [:mu] params)
          _ (mx/materialize! fisher)
          t-compiled (- (js/Date.now) t1)
          f-compiled (mx/item (mx/mat-get fisher 0 0))]
      (is (h/close? f-uncompiled f-compiled 0.1) "Compiled Fisher ~ uncompiled")
      (is (< (js/Math.abs (- f-compiled 10.0)) 2.0) "Both ~ analytical (K=10)"))))

(cljs.test/run-tests)
