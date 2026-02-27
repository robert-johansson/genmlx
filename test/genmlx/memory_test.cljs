(ns genmlx.memory-test
  (:require [genmlx.mlx :as mx]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" expected)
        (println "    actual:  " actual))))

(println "\n=== Memory Management Tests ===")

;; -- Memory monitoring returns numbers --
(println "\n-- memory monitoring --")
(let [active (mx/get-active-memory)
      cache  (mx/get-cache-memory)
      peak   (mx/get-peak-memory)
      wraps  (mx/get-wrappers-count)]
  (assert-true "get-active-memory returns number" (number? active))
  (assert-true "get-active-memory >= 0" (>= active 0))
  (assert-true "get-cache-memory returns number" (number? cache))
  (assert-true "get-cache-memory >= 0" (>= cache 0))
  (assert-true "get-peak-memory returns number" (number? peak))
  (assert-true "get-peak-memory >= 0" (>= peak 0))
  (assert-true "get-wrappers-count returns number" (number? wraps))
  (assert-true "get-wrappers-count > 0" (> wraps 0)))

;; -- Metal availability --
(println "\n-- metal device --")
(assert-true "metal-is-available? returns true" (mx/metal-is-available?))

;; -- Metal device info --
(println "\n-- metal-device-info --")
(let [info (mx/metal-device-info)]
  (assert-true "device-info is a map" (map? info))
  (assert-true "has :resource-limit" (contains? info :resource-limit))
  (assert-equal "resource-limit is 499000" 499000 (:resource-limit info))
  (assert-true "has :device-name" (contains? info :device-name))
  (assert-true "device-name is string" (string? (:device-name info)))
  (assert-true "has :memory-size" (contains? info :memory-size))
  (assert-true "memory-size > 0" (> (:memory-size info) 0))
  (assert-true "has :max-buffer-length" (contains? info :max-buffer-length))
  (assert-true "has :max-recommended-working-set-size"
               (contains? info :max-recommended-working-set-size))
  (assert-true "has :architecture" (contains? info :architecture)))

;; -- set-cache-limit! roundtrip --
(println "\n-- set-cache-limit! roundtrip --")
(let [original (mx/set-cache-limit! 1024)
      restored (mx/set-cache-limit! original)]
  (assert-equal "set-cache-limit! returns previous value" 1024 restored))

;; -- set-memory-limit! roundtrip --
(println "\n-- set-memory-limit! roundtrip --")
(let [original (mx/set-memory-limit! 2048)
      restored (mx/set-memory-limit! original)]
  (assert-equal "set-memory-limit! returns previous value" 2048 restored))

;; -- clear-cache! doesn't throw --
(println "\n-- clear-cache! --")
(mx/clear-cache!)
(assert-true "clear-cache! completed without error" true)

;; -- reset-peak-memory! --
(println "\n-- reset-peak-memory! --")
(mx/reset-peak-memory!)
(assert-true "reset-peak-memory! completed without error" true)

;; -- memory-report --
(println "\n-- memory-report --")
(let [report (mx/memory-report)]
  (assert-true "memory-report is a map" (map? report))
  (assert-true "has :active-bytes" (contains? report :active-bytes))
  (assert-true "has :cache-bytes" (contains? report :cache-bytes))
  (assert-true "has :peak-bytes" (contains? report :peak-bytes))
  (assert-true "has :wrappers" (contains? report :wrappers))
  (assert-true "has :resource-limit" (contains? report :resource-limit))
  (assert-equal "resource-limit matches" 499000 (:resource-limit report)))

;; -- Allocating arrays increases active memory --
(println "\n-- allocation tracking --")
(mx/clear-cache!)
(mx/reset-peak-memory!)
(let [before (mx/get-active-memory)
      arrs   (vec (repeatedly 10 #(mx/ones [1000 1000])))
      _      (apply mx/eval! arrs)
      after  (mx/get-active-memory)]
  (assert-true "active memory increased after allocation" (> after before))
  (assert-true (str "peak memory reflects allocation (" (mx/get-peak-memory) " bytes)")
               (> (mx/get-peak-memory) 0)))

(println "\n=== All memory tests complete ===")
