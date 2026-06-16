(ns agentmodels.harness
  "Shared self-check harness for the runnable agentmodels example scripts.

   Each example under examples/agentmodels/ is runnable AND self-asserting. This is
   the single copy of the assert helpers (previously duplicated in every file),
   mirroring the test/genmlx assert-close / assert-true convention: print PASS/FAIL
   per check, then `(report!)` prints a tally and exits non-zero on any failure.

   Usage:
     (require '[agentmodels.harness :as h])
     (h/check-close \"slope\" 2.0 slope 1e-3)
     (h/check-true  \"reached goal\" reached?)
     (h/report!)"
  (:require [genmlx.mlx :as mx]))

(def ^:private passed (volatile! 0))
(def ^:private failed (volatile! 0))

(defn- num* [x] (if (mx/array? x) (mx/item x) x))

(defn check-true
  "Pass iff `c` is truthy."
  [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))

(defn check-close
  "Pass iff |expected - actual| <= tol. `actual` may be a JS number or an MLX scalar."
  [msg expected actual tol]
  (let [a (num* actual)]
    (if (<= (Math/abs (- expected a)) tol)
      (do (vswap! passed inc) (println " PASS" msg "  ~=" a))
      (do (vswap! failed inc) (println " FAIL" msg "  expected ~" expected "  got" a)))))

(defn check-equal
  "Pass iff (= expected actual)."
  [msg expected actual]
  (if (= expected actual)
    (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
    (do (vswap! failed inc) (println " FAIL" msg "  expected" (pr-str expected) "got" (pr-str actual)))))

(defn report!
  "Print the tally and exit non-zero if any check failed."
  []
  (println (str "\n== " @passed " passed, " @failed " failed =="))
  (when (pos? @failed) (js/process.exit 1)))
