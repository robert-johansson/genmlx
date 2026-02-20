(ns genmlx.vimco-test
  "Tests for VIMCO (Variational Inference with Multi-sample Objectives).
   Tests convergence, comparison with IWELBO, and shape correctness."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.inference.vi :as vi]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== VIMCO Tests ===\n")

;; ---------------------------------------------------------------------------
;; 1. VIMCO converges on a simple Gaussian problem
;; ---------------------------------------------------------------------------

(println "-- VIMCO convergence: Gaussian target --")
;; Model: z ~ N(0, 1), obs = z + noise. Observe obs=3.
;; True posterior: z | obs=3 ~ N(1.5, sqrt(0.5))
;; Guide: N(mu, exp(log-sigma)) parameterized by [mu, log-sigma]
;; VIMCO's REINFORCE term adds variance on continuous problems,
;; so we use more iterations/samples than ELBO/IWELBO.
(let [log-p (fn [z]
              (let [z-scalar (mx/index z 0)]
                (mx/add (dc/dist-log-prob (dist/gaussian 0 1) z-scalar)
                        (dc/dist-log-prob (dist/gaussian z-scalar 1) (mx/scalar 3.0)))))
      log-q (fn [z params]
              (let [mu (mx/index params 0)
                    log-sigma (mx/index params 1)
                    sigma (mx/exp log-sigma)]
                (dc/dist-log-prob (dist/gaussian mu sigma) (mx/index z 0))))
      sample-fn (fn [params key n]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)
                        eps (rng/normal (rng/ensure-key key) [n 1])]
                    (mx/add mu (mx/multiply sigma eps))))
      init-params (mx/array [0.0 0.0])
      result (vi/vimco
               {:iterations 500 :learning-rate 0.01 :n-samples 20}
               log-p log-q sample-fn init-params)]
  (mx/eval! (:params result))
  (let [final-mu (mx/item (mx/index (:params result) 0))
        final-log-sigma (mx/item (mx/index (:params result) 1))
        final-sigma (js/Math.exp final-log-sigma)
        losses (:loss-history result)
        first-loss (first losses)
        last-loss (last losses)]
    (println "    final mu:" final-mu "sigma:" final-sigma)
    (assert-close "VIMCO: mu near 1.5" 1.5 final-mu 1.0)
    (assert-true "VIMCO: sigma reasonable" (and (> final-sigma 0.1) (< final-sigma 3.0)))
    (assert-true "VIMCO: loss decreased" (< last-loss first-loss))))

;; ---------------------------------------------------------------------------
;; 2. VIMCO via programmable-vi interface
;; ---------------------------------------------------------------------------

(println "\n-- VIMCO via programmable-vi --")
(let [log-p (fn [z]
              (let [z-scalar (mx/index z 0)]
                (mx/add (dc/dist-log-prob (dist/gaussian 0 1) z-scalar)
                        (dc/dist-log-prob (dist/gaussian z-scalar 1) (mx/scalar 3.0)))))
      log-q (fn [z params]
              (let [mu (mx/index params 0)
                    log-sigma (mx/index params 1)
                    sigma (mx/exp log-sigma)]
                (dc/dist-log-prob (dist/gaussian mu sigma) (mx/index z 0))))
      sample-fn (fn [params key n]
                  (let [mu (mx/index params 0)
                        log-sigma (mx/index params 1)
                        sigma (mx/exp log-sigma)
                        eps (rng/normal (rng/ensure-key key) [n 1])]
                    (mx/add mu (mx/multiply sigma eps))))
      init-params (mx/array [0.0 0.0])
      result (vi/programmable-vi
               {:iterations 500 :learning-rate 0.01 :n-samples 20
                :objective :vimco}
               log-p log-q sample-fn init-params)]
  (mx/eval! (:params result))
  (let [final-mu (mx/item (mx/index (:params result) 0))
        losses (:loss-history result)]
    (println "    final mu:" final-mu)
    (assert-true "programmable-vi VIMCO: mu moved toward 1.5" (> final-mu 0.3))
    (assert-true "programmable-vi VIMCO: has loss history" (pos? (count losses)))
    (assert-true "programmable-vi VIMCO: loss decreased"
      (< (last losses) (first losses)))))

;; ---------------------------------------------------------------------------
;; 3. Shape correctness: objective returns scalar
;; ---------------------------------------------------------------------------

(println "\n-- VIMCO shape correctness --")
(let [log-p (fn [z] (dc/dist-log-prob (dist/gaussian 0 1) (mx/index z 0)))
      log-q (fn [z] (dc/dist-log-prob (dist/gaussian 0 1) (mx/index z 0)))
      obj-fn (vi/vimco-objective log-p log-q)
      samples (mx/random-normal [5 1])
      result (obj-fn samples)]
  (mx/eval! result)
  (assert-true "VIMCO objective: returns scalar" (= 0 (mx/ndim result)))
  (assert-true "VIMCO objective: finite value" (js/isFinite (mx/item result))))

;; Test with different K values
(println "\n-- VIMCO different K values --")
(let [log-p (fn [z] (dc/dist-log-prob (dist/gaussian 0 1) (mx/index z 0)))
      log-q (fn [z] (dc/dist-log-prob (dist/gaussian 0 1) (mx/index z 0)))
      obj-fn (vi/vimco-objective log-p log-q)]
  (doseq [k [3 10 20]]
    (let [samples (mx/random-normal [k 1])
          result (obj-fn samples)]
      (mx/eval! result)
      (assert-true (str "VIMCO K=" k ": returns scalar") (= 0 (mx/ndim result)))
      (assert-true (str "VIMCO K=" k ": finite") (js/isFinite (mx/item result))))))

(println "\n=== VIMCO Tests Complete ===")
