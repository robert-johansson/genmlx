;; @tier fast
(ns genmlx.vgenerate-compiled-test
  "genmlx-vjnn (Phase 2b of genmlx-z2gt): vgenerate-compiled — the batched
   handler run traced once through the persistent CompiledFn, replayed per
   call. Equivalence vs the handler path is the 819v-mandated safety net."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vect])
  (:require-macros [genmlx.gen :refer [gen]]))

(def xs [0.0 1.0 2.0 3.0 4.0])
(def ys [-1.0 1.0 3.0 5.0 7.0])

(def body-runs (atom 0))

(def model
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (swap! body-runs inc)
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))
        slope))))

(def hmodel (dyn/strip-analytical-path model))

(def obs
  (apply cm/choicemap
         (mapcat (fn [[j y]] [(keyword (str "y" j)) (mx/scalar y)])
                 (map-indexed vector ys))))

(if (mx/metal-is-available?)
  (println "SKIP: vgenerate-compiled is CUDA-only until measured on Metal (genmlx-vjnn)")
  (do

(deftest equivalence-same-key-test
  (testing "compiled :call matches handler vgenerate for the SAME key"
    (let [n   64
          cf  (dyn/vgenerate-compiled hmodel [xs] obs n)
          _   (is (= :key-traced (:mode cf))
                  "single traced mode after the genmlx-agcp factoring verdict")
          key (rng/fresh-key 42)
          vt-h (dyn/vgenerate hmodel [xs] obs n key)
          vt-c ((:call cf) key)]
      (try
        (is (h/close? (mx/item (vect/vtrace-log-ml-estimate vt-h))
                      (mx/item (vect/vtrace-log-ml-estimate vt-c)) 1e-3)
            "log-ML estimates match (same key, same noise)")
        (let [wh (vec (mx/->clj (:weight vt-h)))
              wc (vec (mx/->clj (:weight vt-c)))]
          (is (= n (count wc)) "weight has [N] shape")
          ;; RELATIVE tolerance: prior-drawn slopes give |w| ~ 1e4, where f32
          ;; rounding + compile fusion reordering exceed any absolute epsilon.
          (is (every? true? (map #(h/close? %1 %2 (max 1e-2 (* 1e-4 (js/Math.abs %1))))
                                 wh wc))
              "per-particle weights match to f32/fusion RELATIVE tolerance"))
        (let [sh (vec (mx/->clj (cm/get-value (cm/get-submap (:choices vt-h) :slope))))
              sc (vec (mx/->clj (cm/get-value (cm/get-submap (:choices vt-c) :slope))))]
          (is (every? true? (map #(h/close? %1 %2 1e-3) sh sc))
              "latent :slope samples match (same key -> same noise)"))
        (is (cm/has-value? (cm/get-submap (:choices vt-c) :y0))
            "constrained sites present in the rebuilt choicemap")
        (finally ((:free! cf)))))))

(deftest replay-respects-key-and-skips-body-test
  (testing "replays use the NEW key and never re-run the model body"
    (let [n  32
          cf (dyn/vgenerate-compiled hmodel [xs] obs n)]
      (try
        (let [runs-before-trace @body-runs
              w1 ((comp mx/->clj :weight) ((:call cf) (rng/fresh-key 1)))
              runs-after-trace @body-runs
              w2 ((comp mx/->clj :weight) ((:call cf) (rng/fresh-key 2)))
              w1b ((comp mx/->clj :weight) ((:call cf) (rng/fresh-key 1)))]
          (is (pos? (- runs-after-trace runs-before-trace))
              "first :call traced (body ran)")
          (is (= runs-after-trace @body-runs)
              "REPLAYS never re-ran the model body")
          (is (not= (vec w1) (vec w2)) "different keys give different weights")
          (is (= (vec w1) (vec w1b)) "same key replays identically"))
        (finally ((:free! cf)))))))

(deftest untraceable-sampler-degradation-test
  (testing "a rejection-sampler latent (gamma: item per draw) degrades loudly to the handler"
    (let [m (dyn/auto-key
              (gen [_]
                (let [th (trace :theta (dist/gamma-dist 2 2))]
                  (trace :y (dist/gaussian th 1))
                  th)))
          hm (dyn/strip-analytical-path m)
          ob (cm/choicemap :y (mx/scalar 0.7))
          cf (dyn/vgenerate-compiled hm [nil] ob 16)]
      (try
        (let [key (rng/fresh-key 9)
              vt-c ((:call cf) key)   ;; first call: trace fails -> degrade
              vt-c2 ((:call cf) (rng/fresh-key 10))]
          (is (some? (:weight vt-c)) "degraded first call still returns a trace")
          (is (some? (:weight vt-c2)) "later calls keep working on the handler path")
          (is (h/close? (mx/item (vect/vtrace-log-ml-estimate
                                  (dyn/vgenerate hm [nil] ob 16 key)))
                        (mx/item (vect/vtrace-log-ml-estimate vt-c)) 1e-3)
              "degraded call matches the handler (it IS the handler per call)"))
        (finally ((:free! cf)))))))

(deftest lifecycle-test
  (testing "free! then call errors cleanly"
    (let [cf (dyn/vgenerate-compiled hmodel [xs] obs 8)]
      ((:call cf) (rng/fresh-key 5))
      ((:free! cf))
      (is (thrown? js/Error ((:call cf) (rng/fresh-key 6)))
          "call after free throws, no crash"))))

))

(cljs.test/run-tests)
