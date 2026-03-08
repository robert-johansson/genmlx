(ns genmlx.differentiable-hard2-test
  "Follow-up: test with (a) more iterations, (b) fewer latent dims."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.differentiable :as diff])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-close [label expected actual tol]
  (let [ok (< (js/Math.abs (- expected actual)) tol)]
    (if ok
      (println (str "  PASS: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")"))
      (println (str "  FAIL: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")")))))

;; Same RNG for reproducible test data
(def rng-state (atom 42))
(defn next-gaussian [mu sigma]
  (swap! rng-state #(mod (+ (* 1103515245 %) 12345) 2147483648))
  (let [u1 (/ @rng-state 2147483648.0)]
    (swap! rng-state #(mod (+ (* 1103515245 %) 12345) 2147483648))
    (let [u2 (/ @rng-state 2147483648.0)
          z (* (js/Math.sqrt (* -2 (js/Math.log (max u1 1e-10))))
               (js/Math.cos (* 2 js/Math.PI u2)))]
      (+ mu (* sigma z)))))

;; ---------------------------------------------------------------------------
;; Test A: 3 groups (3 latent dims) — should be easy for IS
;; ---------------------------------------------------------------------------

(println "\n=== Test A: 3 groups, 10 obs each (3 latent dims) ===")

(def N-GROUPS-A 3)
(def OBS-PER-GROUP-A 10)
(def TRUE-MU0-A 3.0)
(def TRUE-SIGMA0-A 2.0)

(def group-means-a (mapv (fn [_] (next-gaussian TRUE-MU0-A TRUE-SIGMA0-A)) (range N-GROUPS-A)))
(println (str "Group means: " (mapv #(.toFixed % 2) group-means-a)))

(def model-a
  (gen []
    (let [mu0 (param :mu0 0.0)
          log-sigma0 (param :log-sigma0 0.0)
          sigma0 (mx/exp log-sigma0)]
      (doseq [j (range N-GROUPS-A)]
        (let [theta-j (trace (keyword (str "theta_" j))
                             (dist/gaussian mu0 sigma0))]
          (doseq [i (range OBS-PER-GROUP-A)]
            (trace (keyword (str "y_" j "_" i))
                   (dist/gaussian theta-j 1.0))))))))

(def obs-a
  (apply cm/choicemap
    (mapcat (fn [j]
              (mapcat (fn [i]
                        [(keyword (str "y_" j "_" i))
                         (mx/scalar (next-gaussian (nth group-means-a j) 1.0))])
                      (range OBS-PER-GROUP-A)))
            (range N-GROUPS-A))))

(let [result (diff/optimize-params
               {:iterations 300 :lr 0.02 :n-particles 3000
                :callback (fn [{:keys [iter log-ml params]}]
                            (when (zero? (mod iter 50))
                              (let [mu (mx/item (mx/index params 0))
                                    ls (mx/item (mx/index params 1))]
                                (println (str "  iter " iter
                                             ": log-ml=" (.toFixed log-ml 2)
                                             "  mu0=" (.toFixed mu 3)
                                             "  sigma0=" (.toFixed (js/Math.exp ls) 3))))))}
               model-a [] obs-a
               [:mu0 :log-sigma0]
               (mx/array [0.0 0.0]))
      final-mu0 (mx/item (mx/index (:params result) 0))
      final-sigma0 (js/Math.exp (mx/item (mx/index (:params result) 1)))]
  (println (str "\n  Final: mu0=" (.toFixed final-mu0 3)
               ", sigma0=" (.toFixed final-sigma0 3)))
  (println (str "  Truth: mu0=" TRUE-MU0-A ", sigma0=" TRUE-SIGMA0-A))
  (assert-close "mu0 recovered" TRUE-MU0-A final-mu0 0.75)
  (assert-close "sigma0 recovered" TRUE-SIGMA0-A final-sigma0 1.0))

;; ---------------------------------------------------------------------------
;; Test B: 8 groups but 500 iterations + 5000 particles
;; ---------------------------------------------------------------------------

(println "\n=== Test B: 8 groups, 5 obs each — 500 iters, 5000 particles ===")

(reset! rng-state 42)  ;; Same data as hard test

(def N-GROUPS-B 8)
(def OBS-PER-GROUP-B 5)
(def group-means-b (mapv (fn [_] (next-gaussian 3.0 2.0)) (range N-GROUPS-B)))
(println (str "Group means: " (mapv #(.toFixed % 2) group-means-b)))

(def model-b
  (gen []
    (let [mu0 (param :mu0 0.0)
          log-sigma0 (param :log-sigma0 0.0)
          sigma0 (mx/exp log-sigma0)]
      (doseq [j (range N-GROUPS-B)]
        (let [theta-j (trace (keyword (str "theta_" j))
                             (dist/gaussian mu0 sigma0))]
          (doseq [i (range OBS-PER-GROUP-B)]
            (trace (keyword (str "y_" j "_" i))
                   (dist/gaussian theta-j 1.0))))))))

(def obs-b
  (apply cm/choicemap
    (mapcat (fn [j]
              (mapcat (fn [i]
                        [(keyword (str "y_" j "_" i))
                         (mx/scalar (next-gaussian (nth group-means-b j) 1.0))])
                      (range OBS-PER-GROUP-B)))
            (range N-GROUPS-B))))

(let [result (diff/optimize-params
               {:iterations 500 :lr 0.02 :n-particles 5000
                :callback (fn [{:keys [iter log-ml params]}]
                            (when (zero? (mod iter 100))
                              (let [mu (mx/item (mx/index params 0))
                                    ls (mx/item (mx/index params 1))]
                                (println (str "  iter " iter
                                             ": log-ml=" (.toFixed log-ml 2)
                                             "  mu0=" (.toFixed mu 3)
                                             "  sigma0=" (.toFixed (js/Math.exp ls) 3))))))}
               model-b [] obs-b
               [:mu0 :log-sigma0]
               (mx/array [0.0 0.0]))
      final-mu0 (mx/item (mx/index (:params result) 0))
      final-sigma0 (js/Math.exp (mx/item (mx/index (:params result) 1)))]
  (println (str "\n  Final: mu0=" (.toFixed final-mu0 3)
               ", sigma0=" (.toFixed final-sigma0 3)))
  (println (str "  Truth: mu0=3.000, sigma0=2.000"))
  (assert-close "mu0 recovered" 3.0 final-mu0 0.75)
  (assert-close "sigma0 recovered" 2.0 final-sigma0 1.0))

(println "\nDone.")
