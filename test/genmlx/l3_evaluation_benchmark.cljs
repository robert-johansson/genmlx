(ns genmlx.l3-evaluation-benchmark
  "Level 3 evaluation: Auto-analytical elimination vs standard inference.

   A: Observation scaling — L2 IS degrades exponentially, L3 stays exact
   B: Multi-group Rao-Blackwellization — 3 conjugate groups + 2 non-conjugate
   C: Amortized cost — equivalent particle count to match L3 accuracy

   Run: bun run --bun nbb test/genmlx/l3_evaluation_benchmark.cljs"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- time-ms [f]
  (let [start (js/Date.now)] (f) (- (js/Date.now) start)))

(defn- mean [xs] (/ (reduce + xs) (count xs)))
(defn- std [xs]
  (let [m (mean xs)
        v (/ (reduce + (map #(let [d (- % m)] (* d d)) xs)) (count xs))]
    (js/Math.sqrt v)))

(defn- strip-analytical
  "Remove auto-handlers from schema, forcing L2 standard generate (prior proposal)."
  [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :conjugate-pairs
                            :has-conjugate? :analytical-plan)))

(defn- ess-from-log-weights [log-ws]
  (let [max-w (apply max log-ws)
        ws (map #(js/Math.exp (- % max-w)) log-ws)
        s (reduce + ws)
        nw (map #(/ % s) ws)]
    (/ 1.0 (reduce + (map #(* % %) nw)))))

(defn- log-ml-from-log-weights [log-ws]
  (let [n (count log-ws)
        max-w (apply max log-ws)
        lse (+ max-w (js/Math.log (reduce + (map #(js/Math.exp (- % max-w)) log-ws))))]
    (- lse (js/Math.log n))))

(defn- generate-weight [model args obs]
  (let [{:keys [weight]} (p/generate model args obs)]
    (mx/eval! weight)
    (mx/item weight)))

(defn- run-is-trial
  "Run one IS trial: generate n-particles weights, return log-ML and ESS."
  [model args obs n-particles seed]
  (let [keys (rng/split-n (rng/fresh-key seed) n-particles)
        log-ws (mapv (fn [ki] (generate-weight (dyn/with-key model ki) args obs)) keys)]
    {:log-ml (log-ml-from-log-weights log-ws)
     :ess (ess-from-log-weights log-ws)}))

;; =========================================================================
;; Model Definitions
;; =========================================================================

;; Wide prior (std=10) makes prior-posterior mismatch severe
;; Prior variance = 100, obs variance = 1
;; With n obs at y=1, posterior concentrates at ~1 with var~1/n
;; Prior proposal ESS degrades exponentially with n

(def nn-5
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1))
      mu)))

(def nn-10
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1))
      (trace :y6 (dist/gaussian mu 1))
      (trace :y7 (dist/gaussian mu 1))
      (trace :y8 (dist/gaussian mu 1))
      (trace :y9 (dist/gaussian mu 1))
      (trace :y10 (dist/gaussian mu 1))
      mu)))

(def nn-20
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1))
      (trace :y6 (dist/gaussian mu 1))
      (trace :y7 (dist/gaussian mu 1))
      (trace :y8 (dist/gaussian mu 1))
      (trace :y9 (dist/gaussian mu 1))
      (trace :y10 (dist/gaussian mu 1))
      (trace :y11 (dist/gaussian mu 1))
      (trace :y12 (dist/gaussian mu 1))
      (trace :y13 (dist/gaussian mu 1))
      (trace :y14 (dist/gaussian mu 1))
      (trace :y15 (dist/gaussian mu 1))
      (trace :y16 (dist/gaussian mu 1))
      (trace :y17 (dist/gaussian mu 1))
      (trace :y18 (dist/gaussian mu 1))
      (trace :y19 (dist/gaussian mu 1))
      (trace :y20 (dist/gaussian mu 1))
      mu)))

(def nn-50
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1)) (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1)) (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1)) (trace :y6 (dist/gaussian mu 1))
      (trace :y7 (dist/gaussian mu 1)) (trace :y8 (dist/gaussian mu 1))
      (trace :y9 (dist/gaussian mu 1)) (trace :y10 (dist/gaussian mu 1))
      (trace :y11 (dist/gaussian mu 1)) (trace :y12 (dist/gaussian mu 1))
      (trace :y13 (dist/gaussian mu 1)) (trace :y14 (dist/gaussian mu 1))
      (trace :y15 (dist/gaussian mu 1)) (trace :y16 (dist/gaussian mu 1))
      (trace :y17 (dist/gaussian mu 1)) (trace :y18 (dist/gaussian mu 1))
      (trace :y19 (dist/gaussian mu 1)) (trace :y20 (dist/gaussian mu 1))
      (trace :y21 (dist/gaussian mu 1)) (trace :y22 (dist/gaussian mu 1))
      (trace :y23 (dist/gaussian mu 1)) (trace :y24 (dist/gaussian mu 1))
      (trace :y25 (dist/gaussian mu 1)) (trace :y26 (dist/gaussian mu 1))
      (trace :y27 (dist/gaussian mu 1)) (trace :y28 (dist/gaussian mu 1))
      (trace :y29 (dist/gaussian mu 1)) (trace :y30 (dist/gaussian mu 1))
      (trace :y31 (dist/gaussian mu 1)) (trace :y32 (dist/gaussian mu 1))
      (trace :y33 (dist/gaussian mu 1)) (trace :y34 (dist/gaussian mu 1))
      (trace :y35 (dist/gaussian mu 1)) (trace :y36 (dist/gaussian mu 1))
      (trace :y37 (dist/gaussian mu 1)) (trace :y38 (dist/gaussian mu 1))
      (trace :y39 (dist/gaussian mu 1)) (trace :y40 (dist/gaussian mu 1))
      (trace :y41 (dist/gaussian mu 1)) (trace :y42 (dist/gaussian mu 1))
      (trace :y43 (dist/gaussian mu 1)) (trace :y44 (dist/gaussian mu 1))
      (trace :y45 (dist/gaussian mu 1)) (trace :y46 (dist/gaussian mu 1))
      (trace :y47 (dist/gaussian mu 1)) (trace :y48 (dist/gaussian mu 1))
      (trace :y49 (dist/gaussian mu 1)) (trace :y50 (dist/gaussian mu 1))
      mu)))

(defn- make-nn-obs [n]
  (reduce (fn [cm i]
            (cm/set-value cm (keyword (str "y" (inc i))) (mx/scalar 1.0)))
          cm/EMPTY (range n)))

;; Multi-group model: 3 conjugate NN groups + 2 non-conjugate params
(def multi-group-model
  (gen []
    ;; Group 1: mu1 ~ N(0, 10), 5 obs
    (let [mu1 (trace :mu1 (dist/gaussian 0 10))]
      (trace :g1y1 (dist/gaussian mu1 1))
      (trace :g1y2 (dist/gaussian mu1 1))
      (trace :g1y3 (dist/gaussian mu1 1))
      (trace :g1y4 (dist/gaussian mu1 1))
      (trace :g1y5 (dist/gaussian mu1 1))
      ;; Group 2: mu2 ~ N(0, 10), 5 obs
      (let [mu2 (trace :mu2 (dist/gaussian 0 10))]
        (trace :g2y1 (dist/gaussian mu2 1))
        (trace :g2y2 (dist/gaussian mu2 1))
        (trace :g2y3 (dist/gaussian mu2 1))
        (trace :g2y4 (dist/gaussian mu2 1))
        (trace :g2y5 (dist/gaussian mu2 1))
        ;; Group 3: mu3 ~ N(0, 10), 5 obs
        (let [mu3 (trace :mu3 (dist/gaussian 0 10))]
          (trace :g3y1 (dist/gaussian mu3 1))
          (trace :g3y2 (dist/gaussian mu3 1))
          (trace :g3y3 (dist/gaussian mu3 1))
          (trace :g3y4 (dist/gaussian mu3 1))
          (trace :g3y5 (dist/gaussian mu3 1))
          ;; Non-conjugate: quadratic and sinusoidal links
          (let [theta1 (trace :theta1 (dist/gaussian 0 5))
                theta2 (trace :theta2 (dist/gaussian 0 5))]
            (trace :z1 (dist/gaussian (mx/multiply (mx/scalar 0.3)
                                        (mx/multiply theta1 theta1)) 0.5))
            (trace :z2 (dist/gaussian (mx/multiply (mx/scalar 3.0)
                                        (mx/sin theta2)) 0.5))
            [mu1 mu2 mu3 theta1 theta2]))))))

(def multi-group-obs
  (-> cm/EMPTY
      ;; Group 1 obs (mean ~ 1.2)
      (cm/set-value :g1y1 (mx/scalar 1.0)) (cm/set-value :g1y2 (mx/scalar 1.5))
      (cm/set-value :g1y3 (mx/scalar 0.5)) (cm/set-value :g1y4 (mx/scalar 2.0))
      (cm/set-value :g1y5 (mx/scalar 1.2))
      ;; Group 2 obs (mean ~ -0.6)
      (cm/set-value :g2y1 (mx/scalar -0.5)) (cm/set-value :g2y2 (mx/scalar -1.0))
      (cm/set-value :g2y3 (mx/scalar -0.3)) (cm/set-value :g2y4 (mx/scalar -0.7))
      (cm/set-value :g2y5 (mx/scalar -0.5))
      ;; Group 3 obs (mean ~ 3.0)
      (cm/set-value :g3y1 (mx/scalar 3.0)) (cm/set-value :g3y2 (mx/scalar 2.5))
      (cm/set-value :g3y3 (mx/scalar 3.5)) (cm/set-value :g3y4 (mx/scalar 2.8))
      (cm/set-value :g3y5 (mx/scalar 3.2))
      ;; Non-conjugate obs
      (cm/set-value :z1 (mx/scalar 2.0))
      (cm/set-value :z2 (mx/scalar 1.5))))

;; =========================================================================
;; BENCHMARK A: Observation Scaling
;;
;; mu ~ N(0, 100), y_i ~ N(mu, 1). As obs count grows, prior proposal IS
;; collapses (ESS → 1). L3 stays exact.
;; =========================================================================

(println "\n" (apply str (repeat 70 "="))
         "\n  BENCHMARK A: Observation Scaling (prior std = 10)"
         "\n" (apply str (repeat 70 "="))
         "\n"
         "\n  mu ~ N(0, 100), y_i ~ N(mu, 1), all observations = 1.0"
         "\n  As n grows, prior proposal becomes exponentially worse."
         "\n  L3 computes exact marginal LL in every case.\n")

(let [models   [nn-5 nn-10 nn-20 nn-50]
      obs-ns   [5 10 20 50]
      n-particles 200
      n-trials 10]

  (println "  Obs | L3 log-ML (exact)  | L2 log-ML (200 IS)     | L2 ESS | ESS/N")
  (println "  " (apply str (repeat 68 "-")))

  (doseq [[model n-obs] (map vector models obs-ns)]
    (let [obs (make-nn-obs n-obs)

          ;; L3 exact
          l3-w (generate-weight (dyn/auto-key model) [] obs)

          ;; L2 IS trials
          l2-model (strip-analytical model)
          trials (mapv (fn [t] (run-is-trial l2-model [] obs n-particles (+ t 100)))
                       (range n-trials))
          l2-mls (mapv :log-ml trials)
          l2-esses (mapv :ess trials)
          avg-ess (mean l2-esses)]

      (println (str "  " (.padStart (str n-obs) 3) " | "
                    (.toFixed l3-w 6) "           | "
                    (.toFixed (mean l2-mls) 4) " +/- "
                    (.padEnd (.toFixed (std l2-mls) 4) 8) " | "
                    (.padStart (.toFixed avg-ess 1) 5) "  | "
                    (.toFixed (/ avg-ess n-particles) 4))))))

;; =========================================================================
;; BENCHMARK B: Multi-Group Rao-Blackwellization
;;
;; 3 conjugate NN groups (15 obs) + 2 non-conjugate params.
;; L3 eliminates 3 latent dims → samples only 2 remaining.
;; L2 samples all 5 → much worse weight variance.
;; =========================================================================

(println "\n" (apply str (repeat 70 "="))
         "\n  BENCHMARK B: Multi-Group Rao-Blackwellization"
         "\n" (apply str (repeat 70 "="))
         "\n"
         "\n  3 conjugate groups (mu1,mu2,mu3 ~ N(0,100), 5 obs each)"
         "\n  + 2 non-conjugate params (theta1, theta2 with nonlinear links)"
         "\n  L3 eliminates 3/5 latent dimensions analytically.\n")

;; Verify conjugacy detection
(let [schema (:schema multi-group-model)]
  (println (str "  Conjugacy detected: " (:has-conjugate? schema)))
  (println (str "  Pairs: " (count (:conjugate-pairs schema))))
  (when (:conjugate-pairs schema)
    (doseq [pair (:conjugate-pairs schema)]
      (println (str "    " (:prior-addr pair) " -> " (:obs-addr pair)
                    " (" (:family pair) ")"))))
  (println))

(let [n-particles 200
      n-trials 15]

  ;; L3
  (println (str "  L3 (analytical elimination), " n-particles " particles x " n-trials " trials:"))
  (let [trials (mapv (fn [t] (run-is-trial multi-group-model [] multi-group-obs
                                           n-particles (+ t 5000)))
                     (range n-trials))
        l3-mls (mapv :log-ml trials)
        l3-esses (mapv :ess trials)]
    (println (str "    log-ML: " (.toFixed (mean l3-mls) 4) " +/- " (.toFixed (std l3-mls) 4)))
    (println (str "    ESS:    " (.toFixed (mean l3-esses) 1) " / " n-particles
                  " (" (.toFixed (* 100 (/ (mean l3-esses) n-particles)) 1) "%)\n"))

    ;; L2
    (println (str "  L2 (standard, all 5 dims sampled), " n-particles " particles x " n-trials " trials:"))
    (let [l2-model (strip-analytical multi-group-model)
          trials (mapv (fn [t] (run-is-trial l2-model [] multi-group-obs
                                             n-particles (+ t 6000)))
                       (range n-trials))
          l2-mls (mapv :log-ml trials)
          l2-esses (mapv :ess trials)]
      (println (str "    log-ML: " (.toFixed (mean l2-mls) 4) " +/- " (.toFixed (std l2-mls) 4)))
      (println (str "    ESS:    " (.toFixed (mean l2-esses) 1) " / " n-particles
                    " (" (.toFixed (* 100 (/ (mean l2-esses) n-particles)) 1) "%)"))

      ;; ESS ratio
      (let [ratio (/ (mean l3-esses) (max (mean l2-esses) 0.01))]
        (println (str "\n  ESS improvement: " (.toFixed ratio 1) "x"
                      " (L3 " (.toFixed (mean l3-esses) 1)
                      " vs L2 " (.toFixed (mean l2-esses) 1) ")"))
        (println (str "  log-ML std improvement: "
                      (.toFixed (/ (std l2-mls) (max (std l3-mls) 0.0001)) 1) "x"
                      " (L2 " (.toFixed (std l2-mls) 4)
                      " vs L3 " (.toFixed (std l3-mls) 4) ")"))))))

;; =========================================================================
;; BENCHMARK C: Amortized Cost — Equivalent Particle Count
;;
;; For n=10 and n=50 obs, how many L2 IS particles are needed to achieve
;; the same log-ML accuracy that L3 gets in 1 call?
;; =========================================================================

(println "\n" (apply str (repeat 70 "="))
         "\n  BENCHMARK C: Equivalent Particle Count"
         "\n" (apply str (repeat 70 "="))
         "\n"
         "\n  L3 gets exact log-ML in 1 call."
         "\n  How many L2 IS particles to reach std < 0.1 nats?\n")

(doseq [[model n-obs] [[nn-10 10] [nn-50 50]]]
  (let [obs (make-nn-obs n-obs)
        l3-w (generate-weight (dyn/auto-key model) [] obs)
        l2-model (strip-analytical model)]

    (println (str "  n=" n-obs " observations (L3 exact log-ML = " (.toFixed l3-w 4) "):"))
    (println "    Particles | L2 log-ML std | L2 ESS (avg) | Within 0.1 of truth?")
    (println "    " (apply str (repeat 58 "-")))

    (doseq [n-particles [50 200 500]]
      (let [n-trials 10
            trials (mapv (fn [t] (run-is-trial l2-model [] obs n-particles (+ t 9000)))
                         (range n-trials))
            l2-mls (mapv :log-ml trials)
            l2-esses (mapv :ess trials)
            s (std l2-mls)
            within (count (filter #(< (js/Math.abs (- % l3-w)) 0.1) l2-mls))]
        (println (str "    " (.padStart (str n-particles) 9)
                      " | " (.padStart (.toFixed s 4) 10)
                      "    | " (.padStart (.toFixed (mean l2-esses) 1) 8)
                      "     | " within "/" n-trials))))

    (println (str "    L3:     1 call → exact (std = 0.0)\n"))))

;; =========================================================================
;; Summary
;; =========================================================================

(println (apply str (repeat 70 "="))
         "\n  SUMMARY"
         "\n" (apply str (repeat 70 "=")))

(println "
  L3 auto-analytical elimination provides:

  A. IMMUNITY TO OBSERVATION SCALING
     Prior-proposal IS ESS drops to ~1 at 50 obs. L3: always exact.
     This is the fundamental win — L3 breaks the exponential wall.

  B. RAO-BLACKWELLIZATION FOR MIXED MODELS
     Eliminating conjugate dims from the sampling space gives higher ESS
     and lower log-ML variance with the same number of particles.

  C. MASSIVE PARTICLE SAVINGS
     For conjugate substructure, L3's 1 call replaces hundreds or
     thousands of L2 IS particles.

  Note: L3 auto-handlers improve p/generate (IS, SMC initial step).
  MCMC methods using p/regenerate are not directly affected.
")
