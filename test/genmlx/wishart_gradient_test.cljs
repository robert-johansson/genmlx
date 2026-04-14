(ns genmlx.wishart-gradient-test
  (:require [clojure.test :refer [deftest is testing run-tests]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]))

;; --- Helpers ---

(defn- symmetric-fd-gradient
  "Compute finite-difference gradient of f w.r.t. a symmetric matrix.
   Perturbs both X[i,j] and X[j,i] together to stay on the SPD manifold.
   Returns the upper-triangle entries as a flat vector of [i j fd-value]."
  [f X-data k eps]
  (for [i (range k) j (range i k)]
    (let [X+ (cond-> X-data
               true (update-in [i j] + eps)
               (not= i j) (update-in [j i] + eps))
          X- (cond-> X-data
               true (update-in [i j] - eps)
               (not= i j) (update-in [j i] - eps))
          fd (/ (- (f (mx/array X+)) (f (mx/array X-))) (* 2 eps))]
      [i j fd])))

(defn- close?
  "Check if two numbers are within tolerance."
  [a b tol]
  (< (js/Math.abs (- a b)) tol))

;; --- Wishart Tests ---

(deftest wishart-gradient-wrt-x
  (testing "Wishart log-prob gradient w.r.t. X matches finite differences"
    (let [d (dist/wishart 5 (mx/eye 2))
          X (mx/array [[2.0 0.5] [0.5 1.0]])
          X-data [[2.0 0.5] [0.5 1.0]]
          f-ad (fn [x] (dc/dist-log-prob d x))
          vg (mx/value-and-grad f-ad)
          [v g] (vg X)
          _ (mx/eval! v g)
          g-clj (mx/->clj g)
          eps 1e-3
          f-fd (fn [x] (mx/item (dc/dist-log-prob d x)))
          fd-entries (symmetric-fd-gradient f-fd X-data 2 eps)]
      ;; For SPD matrices, autodiff symmetrizes: g[i,j] = g[j,i]
      ;; So for off-diagonal: autodiff_value * 2 should match symmetric FD
      (doseq [[i j fd] fd-entries]
        (let [ad-val (get-in g-clj [i j])
              ;; For off-diagonal, FD perturbs both elements, so compare with 2*autodiff
              expected-fd (if (= i j) ad-val (* 2 ad-val))]
          (is (close? expected-fd fd 0.02)
              (str "Wishart dX[" i "," j "]: autodiff=" ad-val
                   " (x2=" expected-fd ") vs FD=" fd)))))))

(deftest wishart-gradient-wrt-v
  (testing "Wishart log-prob gradient w.r.t. scale matrix V matches finite differences"
    (let [X (mx/array [[2.0 0.5] [0.5 1.0]])
          f-ad (fn [v] (dc/dist-log-prob (dist/wishart 5 v) X))
          vg (mx/value-and-grad f-ad)
          V (mx/eye 2)
          [val grad] (vg V)
          _ (mx/eval! val grad)
          g-clj (mx/->clj grad)
          eps 1e-3
          V-data [[1.0 0.0] [0.0 1.0]]
          f-fd (fn [v] (mx/item (dc/dist-log-prob (dist/wishart 5 v) X)))
          fd-entries (symmetric-fd-gradient f-fd V-data 2 eps)]
      (doseq [[i j fd] fd-entries]
        (let [ad-val (get-in g-clj [i j])
              expected-fd (if (= i j) ad-val (* 2 ad-val))]
          (is (close? expected-fd fd 0.02)
              (str "Wishart dV[" i "," j "]: autodiff=" ad-val
                   " (x2=" expected-fd ") vs FD=" fd)))))))

(deftest wishart-gradient-wrt-v-nonidentity
  (testing "Wishart log-prob gradient w.r.t. non-identity V matches finite differences"
    (let [X (mx/array [[4.0 1.5] [1.5 3.0]])
          V-data [[3.0 1.0] [1.0 2.0]]
          f-ad (fn [v] (dc/dist-log-prob (dist/wishart 7 v) X))
          vg (mx/value-and-grad f-ad)
          V (mx/array V-data)
          [val grad] (vg V)
          _ (mx/eval! val grad)
          g-clj (mx/->clj grad)
          eps 1e-3
          f-fd (fn [v] (mx/item (dc/dist-log-prob (dist/wishart 7 v) X)))
          fd-entries (symmetric-fd-gradient f-fd V-data 2 eps)]
      (doseq [[i j fd] fd-entries]
        (let [ad-val (get-in g-clj [i j])
              expected-fd (if (= i j) ad-val (* 2 ad-val))]
          (is (close? expected-fd fd 0.02)
              (str "Wishart dV[" i "," j "] (V=[[3,1],[1,2]]): autodiff=" ad-val
                   " (x2=" expected-fd ") vs FD=" fd)))))))

;; --- Inverse Wishart Tests ---

(deftest inv-wishart-gradient-wrt-x
  (testing "inv-Wishart log-prob gradient w.r.t. X matches finite differences"
    (let [d (dist/inv-wishart 5 (mx/eye 2))
          X (mx/array [[2.0 0.5] [0.5 1.0]])
          X-data [[2.0 0.5] [0.5 1.0]]
          f-ad (fn [x] (dc/dist-log-prob d x))
          vg (mx/value-and-grad f-ad)
          [v g] (vg X)
          _ (mx/eval! v g)
          g-clj (mx/->clj g)
          eps 1e-3
          f-fd (fn [x] (mx/item (dc/dist-log-prob d x)))
          fd-entries (symmetric-fd-gradient f-fd X-data 2 eps)]
      (doseq [[i j fd] fd-entries]
        (let [ad-val (get-in g-clj [i j])
              expected-fd (if (= i j) ad-val (* 2 ad-val))]
          (is (close? expected-fd fd 0.02)
              (str "inv-Wishart dX[" i "," j "]: autodiff=" ad-val
                   " (x2=" expected-fd ") vs FD=" fd)))))))

(deftest inv-wishart-gradient-wrt-psi
  (testing "inv-Wishart log-prob gradient w.r.t. Psi matches finite differences"
    (let [X (mx/array [[2.0 0.5] [0.5 1.0]])
          f-ad (fn [psi] (dc/dist-log-prob (dist/inv-wishart 5 psi) X))
          vg (mx/value-and-grad f-ad)
          Psi (mx/eye 2)
          [val grad] (vg Psi)
          _ (mx/eval! val grad)
          g-clj (mx/->clj grad)
          eps 1e-3
          Psi-data [[1.0 0.0] [0.0 1.0]]
          f-fd (fn [psi] (mx/item (dc/dist-log-prob (dist/inv-wishart 5 psi) X)))
          fd-entries (symmetric-fd-gradient f-fd Psi-data 2 eps)]
      (doseq [[i j fd] fd-entries]
        (let [ad-val (get-in g-clj [i j])
              expected-fd (if (= i j) ad-val (* 2 ad-val))]
          (is (close? expected-fd fd 0.02)
              (str "inv-Wishart dPsi[" i "," j "]: autodiff=" ad-val
                   " (x2=" expected-fd ") vs FD=" fd)))))))

(deftest inv-wishart-gradient-wrt-psi-nonidentity
  (testing "inv-Wishart log-prob gradient w.r.t. non-identity Psi matches finite differences"
    (let [X (mx/array [[4.0 1.5] [1.5 3.0]])
          Psi-data [[3.0 1.0] [1.0 2.0]]
          f-ad (fn [psi] (dc/dist-log-prob (dist/inv-wishart 7 psi) X))
          vg (mx/value-and-grad f-ad)
          Psi (mx/array Psi-data)
          [val grad] (vg Psi)
          _ (mx/eval! val grad)
          g-clj (mx/->clj grad)
          eps 1e-3
          f-fd (fn [psi] (mx/item (dc/dist-log-prob (dist/inv-wishart 7 psi) X)))
          fd-entries (symmetric-fd-gradient f-fd Psi-data 2 eps)]
      (doseq [[i j fd] fd-entries]
        (let [ad-val (get-in g-clj [i j])
              expected-fd (if (= i j) ad-val (* 2 ad-val))]
          (is (close? expected-fd fd 0.02)
              (str "inv-Wishart dPsi[" i "," j "] (Psi=[[3,1],[1,2]]): autodiff=" ad-val
                   " (x2=" expected-fd ") vs FD=" fd)))))))

;; --- Log-prob correctness tests ---

(deftest wishart-logprob-unchanged
  (testing "Wishart log-prob values unchanged after refactor"
    (let [d (dist/wishart 5 (mx/eye 2))
          X (mx/array [[2.0 0.5] [0.5 1.0]])
          lp (mx/item (dc/dist-log-prob d X))]
      (is (close? lp -5.263 0.01)
          (str "Wishart log-prob: " lp " expected ~-5.263")))))

(deftest inv-wishart-logprob-unchanged
  (testing "inv-Wishart log-prob values unchanged after refactor"
    (let [d (dist/inv-wishart 5 (mx/eye 2))
          X (mx/array [[2.0 0.5] [0.5 1.0]])
          lp (mx/item (dc/dist-log-prob d X))]
      (is (close? lp -7.418 0.01)
          (str "inv-Wishart log-prob: " lp " expected ~-7.418")))))

(run-tests)
