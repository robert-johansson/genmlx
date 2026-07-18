;; @tier fast core
(ns genmlx.constants-test
  "Known-value pin for genmlx.mlx.constants (genmlx-66z9).

   These constants feed the log-density arithmetic of 18+ modules (dist,
   mcmc, kalman, vi, fisher, conjugate, translator, amortized, ekf-nd,
   combinators, ...). A typo'd constant silently corrupts every score and
   gradient downstream, and nothing else pins the values themselves — so this
   suite asserts each def against an independently written NUMERIC LITERAL
   (not a re-derivation via the same js/Math expression, which would only
   test js/Math against itself).

   MLX scalars are float32, so the mx-backed constants get 1e-6 absolute
   tolerance; the host-side LOG-2PI is float64 and pinned to 1e-12. The
   surface test pins the exact set of public defs, membrane_coverage-style:
   adding or removing a constant must update this file."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.constants :as c]))

(deftest host-side-log-2pi
  (testing "LOG-2PI (float64 host number)"
    (is (number? c/LOG-2PI) "LOG-2PI is a plain JS number")
    (is (h/close? 1.8378770664093453 c/LOG-2PI 1e-12) "LOG-2PI = log(2*pi)")))

(deftest mlx-scalar-values
  (testing "cached MLX scalar constants match independent reference literals"
    (is (h/close? 0.0 (mx/item c/ZERO) 0.0) "ZERO")
    (is (h/close? 1.0 (mx/item c/ONE) 0.0) "ONE")
    (is (h/close? 2.0 (mx/item c/TWO) 0.0) "TWO")
    (is (h/close? 3.0 (mx/item c/THREE) 0.0) "THREE")
    (is (h/close? 0.5 (mx/item c/HALF) 0.0) "HALF")
    (is (h/close? 0.6931471805599453 (mx/item c/LOG-2) 1e-6) "LOG-2 = log(2)")
    (is (h/close? 0.9189385332046727 (mx/item c/LOG-2PI-HALF) 1e-6)
        "LOG-2PI-HALF = 0.5*log(2*pi)")
    (is (h/close? 1.1447298858494002 (mx/item c/LOG-PI) 1e-6) "LOG-PI = log(pi)")
    (is (h/close? 3.141592653589793 (mx/item c/MLX-PI) 1e-6) "MLX-PI = pi")
    (is (h/close? 1.4142135623730951 (mx/item c/SQRT-TWO) 1e-6) "SQRT-TWO = sqrt(2)")
    (is (h/close? 1e-30 (mx/item c/TINY) 1e-33) "TINY = 1e-30 (log-guard floor)")))

(deftest neg-inf-is-negative-infinity
  (testing "NEG-INF"
    (let [v (mx/item c/NEG-INF)]
      (is (and (not (js/isFinite v)) (neg? v)) "NEG-INF is -Infinity"))))

(deftest mlx-constants-are-scalars
  (testing "every cached MLX constant is shape [] (broadcast-safe scalar)"
    (doseq [[nm arr] [["ZERO" c/ZERO] ["ONE" c/ONE] ["TWO" c/TWO]
                      ["THREE" c/THREE] ["HALF" c/HALF] ["NEG-INF" c/NEG-INF]
                      ["LOG-2" c/LOG-2] ["LOG-2PI-HALF" c/LOG-2PI-HALF]
                      ["LOG-PI" c/LOG-PI] ["MLX-PI" c/MLX-PI]
                      ["SQRT-TWO" c/SQRT-TWO] ["TINY" c/TINY]]]
      (is (= [] (mx/shape arr)) (str nm " is a scalar")))))

(deftest constants-surface-pin
  (testing "the exact public surface (add/remove a constant -> update this file)"
    (is (= #{"LOG-2PI" "ZERO" "ONE" "TWO" "THREE" "HALF" "NEG-INF" "LOG-2"
             "LOG-2PI-HALF" "LOG-PI" "MLX-PI" "SQRT-TWO" "TINY"}
           (set (map name (keys (ns-publics 'genmlx.mlx.constants)))))
        "13 public defs, no silent additions or removals")))

(cljs.test/run-tests)
