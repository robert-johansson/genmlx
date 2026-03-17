(ns genmlx.kernel-compile-check
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled-ops :as cops])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Test 1: simple branch-free kernel
(def simple-kernel
  (dyn/auto-key
    (gen [t state a b]
      (let [dx (mx/subtract (:x state) (mx/scalar 5.0))
            mean (mx/add (mx/scalar 5.0) (mx/add (mx/multiply a dx) b))
            x (trace :x (dist/gaussian mean 1.0))]
        {:x x}))))

(println "Simple kernel:")
(println (str "  static? " (:static? (:schema simple-kernel))))
(println (str "  compiled-generate? " (some? (cops/get-compiled-generate simple-kernel))))
(println (str "  compiled-simulate? " (some? (:compiled-simulate (:schema simple-kernel)))))

;; Test 2: kernel with ALL constants as args (no closed-over vars)
(def week-kernel
  (dyn/auto-key
    (gen [t state ar-dep c-da mu-dep-arg sig-dep-arg]
      (let [dd (mx/subtract (:dep state) mu-dep-arg)
            dep-mean (mx/add mu-dep-arg
                       (mx/add (mx/multiply ar-dep dd)
                         (mx/multiply c-da dd)))
            dep (trace :dep (dist/gaussian dep-mean sig-dep-arg))]
        {:dep dep}))))

(println "\nWeek kernel (simplified):")
(println (str "  static? " (:static? (:schema week-kernel))))
(println (str "  compiled-generate? " (some? (cops/get-compiled-generate week-kernel))))
(println (str "  compiled-simulate? " (some? (:compiled-simulate (:schema week-kernel)))))

;; Print schema details
(let [schema (:schema week-kernel)]
  (println (str "  trace-sites: " (mapv :addr (:trace-sites schema))))
  (println (str "  has-branches? " (:has-branches? schema)))
  (println (str "  has-loops? " (:has-loops? schema)))
  (println (str "  dynamic-addresses? " (:dynamic-addresses? schema)))
  (println (str "  splice-sites: " (count (:splice-sites schema))))
  (println (str "  param-sites: " (count (:param-sites schema))))
  ;; Check compiled-simulate and compiled-generate directly on schema
  (println (str "  :compiled-simulate on schema? " (some? (:compiled-simulate schema))))
  (println (str "  :compiled-generate on schema? " (some? (:compiled-generate schema))))
  ;; Check compilation stages
  (let [prep (genmlx.compiled/prepare-static-sites schema (:source week-kernel))]
    (println (str "  prepare-static-sites: " (some? prep)))
    (when prep (println (str "  site-specs: " (count (:site-specs prep))))))
  ;; Try make-compiled-generate directly
  (let [cg (cops/make-compiled-generate schema (:source week-kernel))]
    (println (str "  make-compiled-generate: " (some? cg))))
  ;; Try make-compiled-simulate
  (let [cs (genmlx.compiled/make-compiled-simulate schema (:source week-kernel))]
    (println (str "  make-compiled-simulate: " (some? cs)))))

(.exit js/process 0)
