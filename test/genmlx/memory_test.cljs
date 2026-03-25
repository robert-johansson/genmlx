(ns genmlx.memory-test
  "Tests for MLX memory management APIs."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]))

(deftest memory-monitoring
  (testing "memory monitoring returns numbers"
    (let [active (mx/get-active-memory)
          cache  (mx/get-cache-memory)
          peak   (mx/get-peak-memory)
          wraps  (mx/get-wrappers-count)]
      (is (number? active) "get-active-memory returns number")
      (is (>= active 0) "get-active-memory >= 0")
      (is (number? cache) "get-cache-memory returns number")
      (is (>= cache 0) "get-cache-memory >= 0")
      (is (number? peak) "get-peak-memory returns number")
      (is (>= peak 0) "get-peak-memory >= 0")
      (is (number? wraps) "get-wrappers-count returns number")
      (is (> wraps 0) "get-wrappers-count > 0"))))

(deftest metal-availability
  (testing "metal device"
    (is (mx/metal-is-available?) "metal-is-available? returns true")))

(deftest metal-device-info
  (testing "metal-device-info returns expected fields"
    (let [info (mx/metal-device-info)]
      (is (map? info) "device-info is a map")
      (is (contains? info :resource-limit) "has :resource-limit")
      (is (= 499000 (:resource-limit info)) "resource-limit is 499000")
      (is (contains? info :device-name) "has :device-name")
      (is (string? (:device-name info)) "device-name is string")
      (is (contains? info :memory-size) "has :memory-size")
      (is (> (:memory-size info) 0) "memory-size > 0")
      (is (contains? info :max-buffer-length) "has :max-buffer-length")
      (is (contains? info :max-recommended-working-set-size) "has :max-recommended-working-set-size")
      (is (contains? info :architecture) "has :architecture"))))

(deftest set-cache-limit-roundtrip
  (testing "set-cache-limit! roundtrip"
    (let [original (mx/set-cache-limit! 1024)
          restored (mx/set-cache-limit! original)]
      (is (= 1024 restored) "set-cache-limit! returns previous value"))))

(deftest set-memory-limit-roundtrip
  (testing "set-memory-limit! roundtrip"
    (let [original (mx/set-memory-limit! 2048)
          restored (mx/set-memory-limit! original)]
      (is (= 2048 restored) "set-memory-limit! returns previous value"))))

(deftest clear-cache-test
  (testing "clear-cache! doesn't throw"
    (mx/clear-cache!)
    (is true "clear-cache! completed without error")))

(deftest reset-peak-memory-test
  (testing "reset-peak-memory!"
    (mx/reset-peak-memory!)
    (is true "reset-peak-memory! completed without error")))

(deftest memory-report-test
  (testing "memory-report returns expected fields"
    (let [report (mx/memory-report)]
      (is (map? report) "memory-report is a map")
      (is (contains? report :active-bytes) "has :active-bytes")
      (is (contains? report :cache-bytes) "has :cache-bytes")
      (is (contains? report :peak-bytes) "has :peak-bytes")
      (is (contains? report :wrappers) "has :wrappers")
      (is (contains? report :resource-limit) "has :resource-limit")
      (is (= 499000 (:resource-limit report)) "resource-limit matches"))))

(deftest allocation-tracking
  (testing "allocating arrays increases active memory"
    (mx/clear-cache!)
    (mx/reset-peak-memory!)
    (let [before (mx/get-active-memory)
          arrs   (vec (repeatedly 10 #(mx/ones [1000 1000])))
          _      (apply mx/eval! arrs)
          after  (mx/get-active-memory)]
      (is (> after before) "active memory increased after allocation")
      (is (> (mx/get-peak-memory) 0)
          (str "peak memory reflects allocation (" (mx/get-peak-memory) " bytes)")))))

(cljs.test/run-tests)
