;; @tier fast core
(ns genmlx.device-test
  "Positive oracles for the process-wide MLX device knob (genmlx-okeu, over the
   native pair added in genmlx-sko3).

   These MUST fail against the stub they replaced — a `default-device` that
   returned a hardcoded \"gpu\" and a `set-default-device!` that did nothing.
   Reinstate either and the round-trip assertions below go red. That is the
   whole point of this file: the health-audit rule says an assertion which
   would still pass if the function returned a plausible constant is not
   coverage, and \"gpu\" is the most plausible constant there is.

   Every test restores the device it found, so ordering is irrelevant and a
   failure cannot strand the process on the CPU path."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]))

(defn- abs' [x] (js/Math.abs x))

(deftest device-getter-test
  (testing "default-device reports a real device name"
    (is (contains? #{mx/cpu mx/gpu} (mx/default-device)))))

(deftest setter-moves-the-default-test
  (let [initial (mx/default-device)]
    (try
      (testing "setting cpu is observable"
        (mx/set-default-device! mx/cpu)
        (is (= mx/cpu (mx/default-device))
            "a hardcoded-\"gpu\" getter or a no-op setter fails exactly here"))
      (testing "setting gpu is observable"
        (mx/set-default-device! mx/gpu)
        (is (= mx/gpu (mx/default-device))))
      (testing "cpu again — a setter latched on one value cannot pass"
        (mx/set-default-device! mx/cpu)
        (is (= mx/cpu (mx/default-device))))
      (finally (mx/set-default-device! initial)))))

(deftest unknown-device-throws-test
  (let [initial (mx/default-device)]
    (try
      (testing "an unknown device name throws rather than silently no-opping"
        (is (thrown? js/Error (mx/set-default-device! "tpu"))))
      (testing "and leaves the default untouched"
        (is (= initial (mx/default-device))))
      (finally (mx/set-default-device! initial)))))

(deftest cross-device-agreement-test
  (let [initial (mx/default-device)
        ;; sum(x + x) for x = [1 2 3 4] is 20 — known independently of MLX,
        ;; so this cannot pass by agreeing with itself on both devices.
        compute #(mx/item (mx/sum (mx/add (mx/array [1.0 2.0 3.0 4.0])
                                          (mx/array [1.0 2.0 3.0 4.0]))))]
    (try
      (mx/set-default-device! mx/cpu)
      (let [on-cpu (compute)]
        (mx/set-default-device! mx/gpu)
        (let [on-gpu (compute)]
          (testing "the same graph agrees across devices"
            (is (< (abs' (- on-cpu on-gpu)) 1e-5)))
          (testing "and matches the closed form on cpu"
            (is (< (abs' (- on-cpu 20.0)) 1e-5)))
          (testing "and matches the closed form on gpu"
            (is (< (abs' (- on-gpu 20.0)) 1e-5)))))
      (finally (mx/set-default-device! initial)))))

(deftest with-default-device-scope-test
  (let [initial (mx/default-device)]
    (try
      (mx/set-default-device! mx/gpu)
      (testing "the scope applies the device to its body"
        (is (= mx/cpu (mx/with-default-device* mx/cpu #(mx/default-device)))))
      (testing "and restores the previous device after"
        (is (= mx/gpu (mx/default-device))))
      (testing "the body's value is returned"
        (is (= 42 (mx/with-default-device* mx/cpu (fn [] 42)))))
      (testing "restores even when the body throws"
        (is (thrown? js/Error
                     (mx/with-default-device* mx/cpu
                       (fn [] (throw (js/Error. "boom"))))))
        (is (= mx/gpu (mx/default-device))
            "a finally-less scope would strand the process on cpu here"))
      (finally (mx/set-default-device! initial)))))

(cljs.test/run-tests)
