(ns genmlx.validation-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-throws [msg f]
  (let [threw (try (f) false (catch :default _ true))]
    (if threw
      (println "  PASS:" msg)
      (println "  FAIL:" msg "- expected throw"))))

(defn assert-no-throw [msg f]
  (let [threw (try (f) false (catch :default e (str e)))]
    (if threw
      (println "  FAIL:" msg "- unexpected throw:" threw)
      (println "  PASS:" msg))))

(println "\n=== Parameter Validation Tests ===")

;; --- bernoulli ---
(println "\n-- bernoulli --")
(assert-no-throw "valid p=0.5" #(dist/bernoulli 0.5))
(assert-no-throw "valid p=0" #(dist/bernoulli 0))
(assert-no-throw "valid p=1" #(dist/bernoulli 1))
(assert-throws "invalid p=-0.1" #(dist/bernoulli -0.1))
(assert-throws "invalid p=1.5" #(dist/bernoulli 1.5))
(assert-no-throw "mlx array skips" #(dist/bernoulli (mx/scalar 0.5)))

;; --- poisson ---
(println "\n-- poisson --")
(assert-no-throw "valid rate=5" #(dist/poisson 5))
(assert-throws "invalid rate=0" #(dist/poisson 0))
(assert-throws "invalid rate=-1" #(dist/poisson -1))
(assert-no-throw "mlx array skips" #(dist/poisson (mx/scalar 5)))

;; --- laplace ---
(println "\n-- laplace --")
(assert-no-throw "valid scale=1" #(dist/laplace 0 1))
(assert-throws "invalid scale=0" #(dist/laplace 0 0))
(assert-throws "invalid scale=-1" #(dist/laplace 0 -1))
(assert-no-throw "mlx array skips" #(dist/laplace 0 (mx/scalar 1)))

;; --- student-t ---
(println "\n-- student-t --")
(assert-no-throw "valid df=3 scale=1" #(dist/student-t 3 0 1))
(assert-throws "invalid df=0" #(dist/student-t 0 0 1))
(assert-throws "invalid scale=-1" #(dist/student-t 3 0 -1))
(assert-no-throw "mlx array skips" #(dist/student-t (mx/scalar 3) 0 (mx/scalar 1)))

;; --- log-normal ---
(println "\n-- log-normal --")
(assert-no-throw "valid sigma=1" #(dist/log-normal 0 1))
(assert-throws "invalid sigma=0" #(dist/log-normal 0 0))
(assert-throws "invalid sigma=-1" #(dist/log-normal 0 -1))
(assert-no-throw "mlx array skips" #(dist/log-normal 0 (mx/scalar 1)))

;; --- cauchy ---
(println "\n-- cauchy --")
(assert-no-throw "valid scale=1" #(dist/cauchy 0 1))
(assert-throws "invalid scale=0" #(dist/cauchy 0 0))
(assert-throws "invalid scale=-1" #(dist/cauchy 0 -1))
(assert-no-throw "mlx array skips" #(dist/cauchy 0 (mx/scalar 1)))

;; --- inv-gamma ---
(println "\n-- inv-gamma --")
(assert-no-throw "valid shape=2 scale=1" #(dist/inv-gamma 2 1))
(assert-throws "invalid shape=0" #(dist/inv-gamma 0 1))
(assert-throws "invalid scale=-1" #(dist/inv-gamma 2 -1))
(assert-no-throw "mlx array skips" #(dist/inv-gamma (mx/scalar 2) (mx/scalar 1)))

;; --- geometric ---
(println "\n-- geometric --")
(assert-no-throw "valid p=0.5" #(dist/geometric 0.5))
(assert-throws "invalid p=-0.1" #(dist/geometric -0.1))
(assert-throws "invalid p=1.5" #(dist/geometric 1.5))
(assert-no-throw "mlx array skips" #(dist/geometric (mx/scalar 0.5)))

;; --- neg-binomial ---
(println "\n-- neg-binomial --")
(assert-no-throw "valid r=5 p=0.5" #(dist/neg-binomial 5 0.5))
(assert-throws "invalid r=0" #(dist/neg-binomial 0 0.5))
(assert-throws "invalid p=-0.1" #(dist/neg-binomial 5 -0.1))
(assert-throws "invalid p=1.5" #(dist/neg-binomial 5 1.5))
(assert-no-throw "mlx array skips" #(dist/neg-binomial (mx/scalar 5) (mx/scalar 0.5)))

;; --- binomial ---
(println "\n-- binomial --")
(assert-no-throw "valid n=10 p=0.5" #(dist/binomial 10 0.5))
(assert-throws "invalid p=-0.1" #(dist/binomial 10 -0.1))
(assert-throws "invalid p=1.5" #(dist/binomial 10 1.5))
(assert-no-throw "mlx array skips" #(dist/binomial 10 (mx/scalar 0.5)))

;; --- discrete-uniform ---
(println "\n-- discrete-uniform --")
(assert-no-throw "valid lo=0 hi=10" #(dist/discrete-uniform 0 10))
(assert-throws "invalid lo=10 hi=0" #(dist/discrete-uniform 10 0))
(assert-throws "invalid lo=hi=5" #(dist/discrete-uniform 5 5))
(assert-no-throw "mlx array skips" #(dist/discrete-uniform (mx/scalar 0) (mx/scalar 10)))

;; --- truncated-normal ---
(println "\n-- truncated-normal --")
(assert-no-throw "valid sigma=1 lo=-1 hi=1" #(dist/truncated-normal 0 1 -1 1))
(assert-throws "invalid sigma=0" #(dist/truncated-normal 0 0 -1 1))
(assert-throws "invalid lo>hi" #(dist/truncated-normal 0 1 1 -1))
(assert-no-throw "mlx array skips" #(dist/truncated-normal 0 (mx/scalar 1) -1 1))

(println "\n=== Validation Tests Complete ===")
