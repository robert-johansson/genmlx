;; @tier fast
(ns genmlx.membrane-coverage-test
  "genmlx-0vwn: compute-membrane coverage. Verifies the newly-wired pure ops,
   guards the deliberately-OMITTED broken native broadcastTo, and pins the
   @mlx-node/core export count as an upstream-drift signal (a count change means
   re-run the coverage audit: a new export may need wiring or an omission entry)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]))

(def ^:private core (js/require "@mlx-node/core"))
(def ^:private fs (js/require "fs"))

(defn- i32 [a] (mx/->clj (mx/astype a mx/int32)))

(deftest new-ops-correctness-test
  (testing "the newly-wired pure ops produce correct values (closed-form / Math oracles)"
    ;; log-softmax([1,2,3]) = [1,2,3] - logsumexp = [-2.4076, -1.4076, -0.4076]
    (is (h/close? -2.40760 (first (mx/->clj (mx/log-softmax (mx/array [1.0 2.0 3.0])))) 1e-4))
    ;; logical truth tables
    (is (= [1 0 0 0] (i32 (mx/logical-and (mx/array [1 1 0 0]) (mx/array [1 0 1 0])))))
    (is (= [1 1 1 0] (i32 (mx/logical-or (mx/array [1 1 0 0]) (mx/array [1 0 1 0])))))
    (is (= [1 0] (i32 (mx/logical-not (mx/array [0 1])))))
    ;; isfinite: 1 for finite, 0 for inf
    (is (= [1 0] (i32 (mx/isfinite (mx/array [1.0 ##Inf])))))
    ;; cumprod / roll
    (is (= [1 2 6 24] (mx/->clj (mx/cumprod (mx/array [1.0 2.0 3.0 4.0]) 0))))
    (is (= [3 0 1 2] (mx/->clj (mx/roll (mx/array [0.0 1.0 2.0 3.0]) 1))))
    ;; trig / hyperbolic vs Math
    (is (h/close? 0.0 (mx/item (mx/sinh (mx/scalar 0.0))) 1e-5))
    (is (h/close? 1.0 (mx/item (mx/cosh (mx/scalar 0.0))) 1e-5))
    (is (h/close? (/ js/Math.PI 6) (mx/item (mx/arcsin (mx/scalar 0.5))) 1e-5))
    (is (h/close? (/ js/Math.PI 4) (mx/item (mx/arctan (mx/scalar 1.0))) 1e-5))))

(deftest broadcast-to-omission-test
  (testing "native broadcastTo is OMITTED (broken: mis-fills size-1 dims); the custom broadcast-to stays"
    (let [src (.readFileSync fs "src/genmlx/mlx.cljs" "utf8")]
      (is (nil? (re-find #"\.-broadcastTo c" src)) "mlx.cljs does NOT wrap native (.-broadcastTo c)")
      (is (nil? (re-find #"\.broadcastTo " src)) "mlx.cljs does NOT call native (.broadcastTo ...)")
      (is (some? (re-find #"defn broadcast-to" src)) "the custom broadcast-to reconstruction is present"))))

(deftest export-surface-drift-test
  (testing "the @mlx-node/core function-export count is pinned as a drift signal"
    (let [fns (filter #(fn? (aget core %)) (js-keys core))]
      ;; If this fails the upstream surface CHANGED — re-run the genmlx-0vwn
      ;; coverage audit: a newly-added export may need wiring or an omission entry,
      ;; or a deletion is the f6ov/dbce rebase tax surfacing.
      (is (= 212 (count fns))
          (str "@mlx-node/core now exports " (count fns)
               " functions (pinned at 212). Re-audit membrane coverage (genmlx-0vwn).")))))

(cljs.test/run-tests)
