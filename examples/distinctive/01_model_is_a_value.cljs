(ns demo-01-model-is-a-value
  "DISTINCTIVE FEATURE: homoiconicity as the formalism.

   A GenMLX model is an ordinary ClojureScript function. Because the `gen` macro
   captures the body's source form alongside the executable closure, that SINGLE
   source is simultaneously:
     (1) EXECUTED        — p/simulate just runs the function body
     (2) ANALYZED        — the captured form is walked statically (no execution)
                           and compiled (L1-M3 prefix / L1-M2 full)
     (3) CONDITIONED     — the same source conditions on data, yielding a weight
     (4) VECTORIZED      — the same source runs ONCE for N particles by shape
     (5) AUTO-OPTIMIZED  — conjugacy is detected from the source, no user hint

   ONE source. Never rewritten. The form is both the program and the data the
   compiler reads. That is the distinctive property this demo proves."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inspect :as inspect])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ── A model is JUST a function. Bayesian linear regression in 8 lines. ────────
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

;; ── 1. EXECUTED: p/simulate just runs the body as a function ──────────────────
(println "\n=== 1. It's a function: p/simulate just runs the body ===")
(let [tr (p/simulate (dyn/auto-key linreg) [xs])]
  (println "retval slope/intercept:"
           (m (:slope (:retval tr))) "/" (m (:intercept (:retval tr))))
  (println "trace score (log p):" (m (:score tr)))
  (println "addresses traced:    "
           (sort (cm/addresses (:choices tr)))))

;; ── 2. ANALYZED: the SAME source form is read statically (no execution) ───────
(println "\n=== 2. The SAME source was statically analyzed (no execution) ===")
(let [info (inspect/inspect linreg)]
  (println "compilation level:   " (:compilation info)
           "  <- static prefix compiled, dynamic suffix interpreted")
  (println "classification:      " (:classification info))
  (println "dispatch resolution: " (:dispatch info))
  (println "conjugacy detected:  " (:conjugacy info))
  (println "trace-site dist types:"
           (mapv (juxt :addr :dist-type) (:trace-sites info))))

;; ── 3. CONDITIONED: the SAME source conditions on data → log-weight ───────────
(println "\n=== 3. The SAME model conditions on data and returns an importance weight ===")
(let [obs (cm/choicemap :y0 (mx/scalar 0.05) :y1 (mx/scalar 1.1)
                        :y2 (mx/scalar 1.9)  :y3 (mx/scalar 3.05)
                        :y4 (mx/scalar 4.0))            ; ~ slope 1, intercept 0
      {:keys [trace weight]} (p/generate (dyn/auto-key linreg) [xs] obs)]
  (println "generate log-weight: " (m weight))
  (println "constrained slope:   " (m (:slope (:retval trace)))))

;; ── 4. VECTORIZED: the SAME source runs ONCE for N particles, by SHAPE ────────
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

;; ── 5. AUTO-OPTIMIZED: a STATIC source whose conjugacy is read from the form ──
(def normal-mean
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y0 (dist/gaussian mu 1.0))
      (trace :y1 (dist/gaussian mu 1.0))
      (trace :y2 (dist/gaussian mu 1.0))
      mu)))

(println "\n=== 5. A STATIC source → static analysis finds conjugacy (no user hint) ===")
(let [info (inspect/inspect normal-mean)]
  (println "compilation level:   " (:compilation info)
           "  <- fully compiled (every address is a keyword literal)")
  (println "classification:      " (:classification info))
  (println "conjugacy detected:  " (:conjugacy info))
  (println "dispatch resolution: " (:dispatch info)))

(println "\nONE source per model, never rewritten — executed, analyzed,")
(println "conditioned, vectorized, and auto-optimized all from the same form.")
(println "\n=== done ===")
