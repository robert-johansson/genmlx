;; @tier fast core
(ns genmlx.det-contract-test
  "genmlx-rqp9: determinant membrane contract.

   Two families in mlx.cljs:
   - spd-det / spd-logdet : Cholesky-based, lazy, DIFFERENTIABLE, valid ONLY for
     symmetric positive-definite matrices (covariances — MVN/Wishart).
   - logabsdet (QR) + slogdet/det/logdet (host-LU) : correct for ANY square
     matrix, including non-symmetric / indefinite ones.

   The original bug (genmlx-rqp9): a single cholesky-based `det`/`logdet` was used
   for general matrices, silently returning garbage on non-SPD inputs
   ([[1,2],[3,4]] -> 25 instead of -2; [[1,-1],[1,1]] -> 0 instead of 2). This
   guard pins both families against hand-computed oracles so a regression (or a
   stale/drifted MLX binary changing cholesky/qr behaviour) fails loudly."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]))

(defn- it [a] (mx/item a))
(defn- close? [a b] (< (js/Math.abs (- a b)) 1e-4))

;; ===========================================================================
;; SPD family (cholesky) — valid for symmetric positive-definite matrices
;; ===========================================================================

(deftest spd-determinants
  (testing "spd-det / spd-logdet on SPD matrices == hand oracle"
    ;; diagonal: det = product
    (is (close? (it (mx/spd-det (mx/array [[2.0 0.0] [0.0 3.0]]))) 6.0))
    (is (close? (it (mx/spd-logdet (mx/array [[2.0 0.0] [0.0 3.0]]))) (js/Math.log 6.0)))
    ;; dense SPD [[4,1],[1,3]] : det = 12 - 1 = 11
    (is (close? (it (mx/spd-det (mx/array [[4.0 1.0] [1.0 3.0]]))) 11.0))
    (is (close? (it (mx/spd-logdet (mx/array [[4.0 1.0] [1.0 3.0]]))) (js/Math.log 11.0)))))

;; ===========================================================================
;; General family — correct for ANY square matrix (the regression cases)
;; ===========================================================================

(deftest general-determinants
  (testing "det: the previously-WRONG non-SPD cases are now correct"
    (is (close? (it (mx/det (mx/array [[1.0 2.0] [3.0 4.0]]))) -2.0)
        "[[1,2],[3,4]] det = -2 (was 25)")
    (is (close? (it (mx/det (mx/array [[1.0 -1.0] [1.0 1.0]]))) 2.0)
        "[[1,-1],[1,1]] det = 2 (was 0)"))
  (testing "det: diagonal, 3x3, and singular"
    (is (close? (it (mx/det (mx/array [[2.0 0.0] [0.0 3.0]]))) 6.0))
    (is (close? (it (mx/det (mx/array [[1.0 2.0 3.0] [0.0 1.0 4.0] [5.0 6.0 0.0]]))) 1.0))
    (is (close? (it (mx/det (mx/array [[1.0 2.0] [2.0 4.0]]))) 0.0) "singular det = 0"))
  (testing "logabsdet (QR): log|det| for non-SPD matrices"
    (is (close? (it (mx/logabsdet (mx/array [[1.0 2.0] [3.0 4.0]]))) (js/Math.log 2.0)))
    (is (close? (it (mx/logabsdet (mx/array [[1.0 -1.0] [1.0 1.0]]))) (js/Math.log 2.0)))
    (is (close? (it (mx/logabsdet (mx/array [[1.0 2.0 3.0] [0.0 1.0 4.0] [5.0 6.0 0.0]]))) 0.0)
        "3x3 |det|=1 => log|det|=0"))
  (testing "slogdet: sign and log|det|"
    (let [[s la] (mx/slogdet (mx/array [[1.0 2.0] [3.0 4.0]]))]
      (is (= -1.0 (it s)) "negative-det sign")
      (is (close? (it la) (js/Math.log 2.0)) "log|det|"))
    (let [[s la] (mx/slogdet (mx/array [[1.0 -1.0] [1.0 1.0]]))]
      (is (= 1.0 (it s)) "positive-det sign")
      (is (close? (it la) (js/Math.log 2.0)))))
  (testing "logdet: log(det) for det>0, NaN for det<=0"
    (is (close? (it (mx/logdet (mx/array [[2.0 0.0] [0.0 3.0]]))) (js/Math.log 6.0)))
    (is (js/Number.isNaN (it (mx/logdet (mx/array [[1.0 2.0] [3.0 4.0]]))))
        "det = -2 < 0 => logdet NaN")))

;; ===========================================================================
;; Cross-family consistency
;; ===========================================================================

(deftest determinant-consistency
  (testing "det == sign * exp(logabsdet) for general matrices"
    (doseq [m [[[1.0 2.0] [3.0 4.0]] [[1.0 -1.0] [1.0 1.0]]
               [[1.0 2.0 3.0] [0.0 1.0 4.0] [5.0 6.0 0.0]]]]
      (let [a (mx/array m)
            [s la] (mx/slogdet a)]
        (is (close? (it (mx/det a)) (* (it s) (js/Math.exp (it la))))))))
  (testing "on SPD matrices, the general and SPD families agree"
    (doseq [m [[[2.0 0.0] [0.0 3.0]] [[4.0 1.0] [1.0 3.0]]]]
      (let [a (mx/array m)]
        (is (close? (it (mx/det a)) (it (mx/spd-det a))) "det == spd-det")
        (is (close? (it (mx/logabsdet a)) (it (mx/spd-logdet a))) "logabsdet == spd-logdet")
        (is (close? (it (mx/logdet a)) (it (mx/spd-logdet a))) "logdet == spd-logdet (det>0)")))))

(cljs.test/run-tests)
