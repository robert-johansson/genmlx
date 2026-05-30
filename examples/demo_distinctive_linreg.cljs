(ns demo-distinctive-linreg
  "Demonstrates the most distinctive GenMLX properties on ONE linear-regression
   model that is never rewritten: it is an ordinary ClojureScript function whose
   source is simultaneously executable, statically analyzable, conditionable, and
   vectorizable — all behind the same GFI."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inspect :as inspect])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ── 1. A model is JUST a function. Bayesian linear regression in 8 lines. ──────
(def linreg
  (gen [xs]
    (let [slope     (trace :slope     (dist/gaussian 0 2))
          intercept (trace :intercept (dist/gaussian 0 2))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x)) intercept)
                              1.0)))
      {:slope slope :intercept intercept})))

(def xs [0.0 1.0 2.0 3.0 4.0])

(defn- m [v] (mx/item v))                         ; force one scalar to a JS number
(defn- mean [arr] (m (mx/mean arr)))              ; mean of an [N] array

(println "\n=== 1. It's a function: p/simulate just runs the body ===")
(let [tr (p/simulate (dyn/auto-key linreg) [xs])]
  (println "retval slope/intercept:"
           (m (:slope (:retval tr))) "/" (m (:intercept (:retval tr))))
  (println "trace score (log p):" (m (:score tr)))
  (println "addresses traced:    "
           (sort (cm/addresses (:choices tr)))))

(println "\n=== 2. The SAME source was statically analyzed (no execution) ===")
(let [info (inspect/inspect linreg)]
  (println "compilation level:   " (:compilation info))
  (println "classification:      " (:classification info))
  (println "dispatch resolution: " (:dispatch info))
  (println "conjugacy detected:  " (:conjugacy info))
  (println "trace-site dist types:"
           (mapv (juxt :addr :dist-type) (:trace-sites info))))

(println "\n=== 3. The SAME model conditions on data and returns an importance weight ===")
(let [obs (cm/choicemap :y0 (mx/scalar 0.05) :y1 (mx/scalar 1.1)
                        :y2 (mx/scalar 1.9)  :y3 (mx/scalar 3.05)
                        :y4 (mx/scalar 4.0))            ; ~ slope 1, intercept 0
      {:keys [trace weight]} (p/generate (dyn/auto-key linreg) [xs] obs)]
  (println "generate log-weight: " (m weight))
  (println "constrained slope:   " (m (:slope (:retval trace)))))

(println "\n=== 4. The SAME model, vectorized over N particles by changing SHAPES ===")
(println "    (body runs ONCE; MLX broadcasting carries the [N] dimension)")
(let [n  1000
      vt (dyn/vsimulate (dyn/auto-key linreg) [xs] n (rng/fresh-key))
      slope-n     (cm/get-value (cm/get-submap (:choices vt) :slope))
      intercept-n (cm/get-value (cm/get-submap (:choices vt) :intercept))]
  (println "slope array shape:   " (mx/shape slope-n) "(one body run, not" n "runs)")
  (println "E[slope]   over" n "particles:" (mean slope-n)     "(prior mean 0)")
  (println "E[intercept] over" n "particles:" (mean intercept-n) "(prior mean 0)")
  (println "Var[slope] over" n "particles:" (m (mx/variance slope-n)) "(prior var 4)"))

;; ── 5. A STATIC model: the analyzer reads the source and finds conjugacy ──────
(def normal-mean
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y0 (dist/gaussian mu 1.0))
      (trace :y1 (dist/gaussian mu 1.0))
      (trace :y2 (dist/gaussian mu 1.0))
      mu)))

(println "\n=== 5. A STATIC source → static analysis finds conjugacy (no user hint) ===")
(let [info (inspect/inspect normal-mean)]
  (println "compilation level:   " (:compilation info))
  (println "classification:      " (:classification info))
  (println "conjugacy detected:  " (:conjugacy info))
  (println "dispatch resolution: " (:dispatch info)))

(println "\n=== done ===")
