(ns bench.l3-5-combinator-conjugacy
  "L3.5 Combinator Conjugacy Benchmark.

   Demonstrates conjugacy detection inside Map combinators -- the L3.5 extension
   where analytical elimination works on combinator sub-models. The kernel has a
   normal-normal conjugate pair (mu ~ N(0,10), y ~ N(mu,1)). The Map combinator
   applies this kernel independently to each element, so conjugacy elimination
   happens per-element inside the combinator's generate path.

   Key mechanism: the Map combinator's generate calls p/generate on the kernel
   for each element. When the kernel goes through the DynamicGF dispatcher,
   the analytical handler (L3) takes priority over the compiled path. To ensure
   the Map combinator reaches this path, we strip compiled paths from the kernel
   (forcing the Map's handler fallback), while preserving conjugacy info.

   Two conditions:
     L2-standard:    kernel stripped of both compiled paths AND conjugacy
     L3.5-analytical: kernel stripped of compiled paths, conjugacy preserved

   Varies Map size K=3, 5, 10 to see how elimination scales.

   Output: results/l3.5-combinator-conjugacy/data.json

   Usage: bun run --bun nbb bench/l3_5_combinator_conjugacy.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as combinators]
            [genmlx.gfi :as gfi])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Infrastructure
;; ---------------------------------------------------------------------------

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(defn perf-now [] (js/performance.now))

(def results-dir
  (or (aget (.-env js/process) "GENMLX_RESULTS_DIR")
      (.resolve path-mod (js/process.cwd) "results/l3.5-combinator-conjugacy")))

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn write-json [filename data]
  (ensure-dir results-dir)
  (let [filepath (str results-dir "/" filename)]
    (.writeFileSync fs filepath (js/JSON.stringify (clj->js data) nil 2))
    (println (str "  wrote: " filepath))))

;; ---------------------------------------------------------------------------
;; Statistics
;; ---------------------------------------------------------------------------

(defn mean [xs] (/ (reduce + xs) (count xs)))

(defn variance [xs]
  (let [m (mean xs) n (count xs)]
    (/ (reduce + (map #(let [d (- % m)] (* d d)) xs)) n)))

(defn std [xs] (js/Math.sqrt (variance xs)))

(defn ess-from-log-weights
  "Effective sample size from unnormalized log-weights."
  [log-ws]
  (let [max-w (apply max log-ws)
        ws (map #(js/Math.exp (- % max-w)) log-ws)
        s (reduce + ws)
        nw (map #(/ % s) ws)]
    (/ 1.0 (reduce + (map #(* % %) nw)))))

(defn log-ml-from-log-weights
  "Log marginal likelihood estimate via log-sum-exp."
  [log-ws]
  (let [n (count log-ws)
        max-w (apply max log-ws)]
    (- (+ max-w (js/Math.log (reduce + (map #(js/Math.exp (- % max-w)) log-ws))))
       (js/Math.log n))))

;; ---------------------------------------------------------------------------
;; Schema stripping helpers
;; ---------------------------------------------------------------------------

(defn strip-compiled-paths
  "Remove compiled execution paths from schema, keeping conjugacy info.
   Forces the Map combinator to use the handler fallback, which then
   lets the kernel's dispatcher check analytical before compiled."
  [gf]
  (gfi/strip-compiled gf))

(defn strip-analytical
  "Remove conjugacy and auto-handler info from schema."
  [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :conjugate-pairs
                            :has-conjugate? :analytical-plan
                            :auto-regenerate-handlers :auto-regenerate-transition)))

(defn strip-all
  "Remove both compiled paths and analytical info from a gen-fn."
  [gf]
  (-> gf strip-compiled-paths strip-analytical))

;; ---------------------------------------------------------------------------
;; Benchmark helper
;; ---------------------------------------------------------------------------

(defn benchmark [label f & {:keys [warmup-n outer-n inner-n]
                             :or {warmup-n 5 outer-n 3 inner-n 5}}]
  (println (str "\n  [" label "] warming up..."))
  (dotimes [_ warmup-n] (f) (mx/materialize!))
  (mx/clear-cache!)
  (let [outer-times
        (vec (for [_ (range outer-n)]
               (let [inner-times
                     (vec (for [_ (range inner-n)]
                            (let [t0 (perf-now)]
                              (f)
                              (mx/materialize!)
                              (- (perf-now) t0))))]
                 (mx/clear-cache!)
                 (apply min inner-times))))
        mean-ms (/ (reduce + outer-times) (count outer-times))
        std-ms  (js/Math.sqrt (/ (reduce + (map #(* (- % mean-ms) (- % mean-ms))
                                                 outer-times))
                                  (max 1 (dec (count outer-times)))))]
    (println (str "  [" label "] " (.toFixed mean-ms 3) " +/- "
                  (.toFixed std-ms 3) " ms"))
    {:label label :mean-ms mean-ms :std-ms std-ms
     :min-ms (apply min outer-times) :max-ms (apply max outer-times)
     :raw outer-times}))

;; ---------------------------------------------------------------------------
;; Kernel definition: normal-normal conjugate pair
;; ---------------------------------------------------------------------------
;; Each kernel application has:
;;   mu ~ N(0, 10)       (prior)
;;   y  ~ N(mu, 1)       (observation, conjugate to mu)
;; When y is constrained, L3 analytical elimination computes the posterior
;; for mu in closed form, yielding an exact log-ML contribution per element.

(def raw-kernel
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1))
      mu)))

;; L3.5 kernel: compiled paths stripped (forces Map handler fallback),
;; conjugacy preserved (analytical handler fires per-element)
(def kernel-l35
  (dyn/auto-key (strip-compiled-paths raw-kernel)))

;; L2 kernel: both compiled paths AND conjugacy stripped
;; (Map handler fallback, kernel uses basic handler -- prior proposal)
(def kernel-l2
  (dyn/auto-key (strip-all raw-kernel)))

;; ---------------------------------------------------------------------------
;; Report kernel schema conjugacy info
;; ---------------------------------------------------------------------------

(println "\n" (apply str (repeat 70 "=")))
(println "  L3.5 COMBINATOR CONJUGACY BENCHMARK")
(println (apply str (repeat 70 "=")))

(let [conj-pairs (get-in kernel-l35 [:schema :conjugate-pairs])]
  (println (str "\n  Kernel conjugate pairs: " (count conj-pairs)))
  (when (seq conj-pairs)
    (doseq [pair conj-pairs]
      (println (str "    " (:prior-addr pair) " -> " (:obs-addr pair)
                    " [" (:family pair) "]"))))
  (println (str "  Kernel has-conjugate?: " (get-in kernel-l35 [:schema :has-conjugate?])))
  (println (str "  Kernel auto-handlers: " (some? (get-in kernel-l35 [:schema :auto-handlers]))))
  (println (str "  Kernel compiled-generate: " (some? (get-in kernel-l35 [:schema :compiled-generate]))))
  (println (str "  L2 kernel has-conjugate?: " (get-in kernel-l2 [:schema :has-conjugate?])))
  (println (str "  L2 kernel compiled-generate: " (some? (get-in kernel-l2 [:schema :compiled-generate])))))

;; ---------------------------------------------------------------------------
;; IS trial runner
;; ---------------------------------------------------------------------------

(defn generate-weight
  "Run p/generate and extract the log-weight as a JS number."
  [model args obs]
  (let [{:keys [weight]} (p/generate model args obs)]
    (mx/eval! weight)
    (mx/item weight)))

(defn run-is-trial
  "Run one IS trial with n-particles, return log-ML and ESS.
   Clears cache every batch-size particles to prevent memory buildup."
  [model args obs n-particles]
  (let [batch-size 10
        log-ws (loop [i 0 acc []]
                 (if (>= i n-particles)
                   acc
                   (let [end (min (+ i batch-size) n-particles)
                         batch-ws (mapv (fn [_]
                                          (generate-weight model args obs))
                                        (range i end))]
                     (mx/clear-cache!)
                     (recur end (into acc batch-ws)))))]
    {:log-ml (log-ml-from-log-weights log-ws)
     :ess (ess-from-log-weights log-ws)}))

(defn run-trials
  "Run n-trials IS trials, return collected statistics."
  [model args obs n-particles n-trials]
  (let [trials (mapv (fn [_]
                       (run-is-trial model args obs n-particles))
                     (range n-trials))
        log-mls (mapv :log-ml trials)
        esses (mapv :ess trials)]
    {:log-mls log-mls :esses esses
     :log-ml-mean (mean log-mls) :log-ml-std (std log-mls)
     :log-ml-var (variance log-mls)
     :ess-mean (mean esses)}))

;; ---------------------------------------------------------------------------
;; Map combinator models
;; ---------------------------------------------------------------------------

(def n-trials 10)

(def map-sizes [3 5 10])

;; Particles per map size -- lower for larger K to prevent Bun SIGTRAP
(def particles-for-K {3 100 5 100 10 30})

(defn make-inputs
  "Generate K input values: [1.0, 2.0, ..., K]"
  [K]
  (mapv #(mx/scalar (inc %)) (range K)))

(defn make-obs
  "Generate observations constraining :y in each Map element.
   Observations: y_i = x_i + 0.5 (so the true mu is near x_i + 0.5)."
  [K]
  (reduce (fn [cm i]
            (cm/set-choice cm [i :y] (mx/scalar (+ (inc i) 0.5))))
          cm/EMPTY
          (range K)))

(defn run-map-experiment
  "Run L2 vs L3.5 comparison for a Map combinator of size K."
  [K]
  (println (str "\n" (apply str (repeat 60 "-"))))
  (println (str "  Map K=" K " (normal-normal conjugate kernel)"))
  (println (apply str (repeat 60 "-")))

  (let [n-particles (get particles-for-K K 100)
        inputs (make-inputs K)
        obs (make-obs K)

        ;; --- L3.5: Map with analytical kernel (compiled paths stripped) ---
        mapped-l35 (combinators/map-combinator kernel-l35)

        ;; --- L2: Map with stripped kernel (no compiled, no conjugacy) ---
        mapped-l2 (combinators/map-combinator kernel-l2)

        _ (println (str "  Inputs: " K " elements, " n-particles " particles"))
        _ (println (str "  Observations: :y constrained in all " K " elements"))

        ;; L3.5 timing
        l35-timing (benchmark (str "K=" K "-L3.5")
                              (fn [] (generate-weight mapped-l35 [inputs] obs))
                              :warmup-n 3 :outer-n 3 :inner-n 5)

        ;; L3.5 trials
        _ (println (str "\n  Running L3.5 (analytical), " n-particles
                        " particles x " n-trials " trials..."))
        l35-results (run-trials mapped-l35 [inputs] obs n-particles n-trials)
        _ (println (str "  L3.5 log-ML: " (.toFixed (:log-ml-mean l35-results) 6)
                        " +/- " (.toFixed (:log-ml-std l35-results) 6)))
        _ (println (str "  L3.5 ESS:    " (.toFixed (:ess-mean l35-results) 1)
                        " / " n-particles
                        " (" (.toFixed (* 100 (/ (:ess-mean l35-results) n-particles)) 1) "%)"))

        ;; L2 timing
        l2-timing (benchmark (str "K=" K "-L2")
                             (fn [] (generate-weight mapped-l2 [inputs] obs))
                             :warmup-n 3 :outer-n 3 :inner-n 5)

        ;; L2 trials
        _ (println (str "\n  Running L2 (prior-proposal IS), " n-particles
                        " particles x " n-trials " trials..."))
        l2-results (run-trials mapped-l2 [inputs] obs n-particles n-trials)
        _ (println (str "  L2 log-ML: " (.toFixed (:log-ml-mean l2-results) 6)
                        " +/- " (.toFixed (:log-ml-std l2-results) 6)))
        _ (println (str "  L2 ESS:    " (.toFixed (:ess-mean l2-results) 1)
                        " / " n-particles
                        " (" (.toFixed (* 100 (/ (:ess-mean l2-results) n-particles)) 1) "%)"))

        ;; Variance reduction ratio
        var-ratio (if (> (:log-ml-var l35-results) 1e-20)
                    (/ (:log-ml-var l2-results) (:log-ml-var l35-results))
                    js/Infinity)
        ess-ratio (/ (:ess-mean l35-results) (max (:ess-mean l2-results) 0.01))]

    (println (str "\n  --- Summary K=" K " ---"))
    (println (str "  Variance reduction:  "
                  (if (= var-ratio js/Infinity)
                    "Inf (L3.5 exact)"
                    (str (.toFixed var-ratio 1) "x"))))
    (println (str "  ESS improvement:     " (.toFixed ess-ratio 1) "x"))
    (println (str "  L3.5 time: " (.toFixed (:mean-ms l35-timing) 3) " ms"
                  ", L2 time: " (.toFixed (:mean-ms l2-timing) 3) " ms"))

    ;; Return result map
    {:K K
     :L2-standard {:log-ml-mean (:log-ml-mean l2-results)
                   :log-ml-std (:log-ml-std l2-results)
                   :log-ml-var (:log-ml-var l2-results)
                   :ess-mean (:ess-mean l2-results)
                   :timing-ms (:mean-ms l2-timing)
                   :timing-std-ms (:std-ms l2-timing)}
     :L3.5-analytical {:log-ml-mean (:log-ml-mean l35-results)
                       :log-ml-std (:log-ml-std l35-results)
                       :log-ml-var (:log-ml-var l35-results)
                       :ess-mean (:ess-mean l35-results)
                       :timing-ms (:mean-ms l35-timing)
                       :timing-std-ms (:std-ms l35-timing)}
     :n-particles n-particles
     :variance-reduction var-ratio
     :ess-improvement ess-ratio}))

;; ---------------------------------------------------------------------------
;; Run experiments across Map sizes
;; ---------------------------------------------------------------------------

(println (str "\n  Config: particles per K=" (pr-str particles-for-K) ", " n-trials " trials per condition"))
(println "  Kernel: mu ~ N(0,10), y ~ N(mu,1)  [normal-normal conjugate]")
(println (str "  Map sizes: " (vec map-sizes)))

(def results-by-size
  (mapv run-map-experiment map-sizes))

;; ---------------------------------------------------------------------------
;; Summary table
;; ---------------------------------------------------------------------------

(println "\n\n" (apply str (repeat 70 "=")))
(println "         L3.5 COMBINATOR CONJUGACY RESULTS")
(println (apply str (repeat 70 "=")))

(println "\n| K  | L2 var(log-ML) | L3.5 var(log-ML) | Var Reduction | ESS L2 | ESS L3.5 | ESS Ratio |")
(println "|----|----------------|------------------|---------------|--------|----------|-----------|")
(doseq [r results-by-size]
  (let [l2-var (get-in r [:L2-standard :log-ml-var])
        l35-var (get-in r [:L3.5-analytical :log-ml-var])
        vr (:variance-reduction r)]
    (println (str "| " (.padStart (str (:K r)) 2)
                  " | " (.padStart (if (> l2-var 0) (.toFixed l2-var 6) "N/A") 14)
                  " | " (.padStart (if (> l35-var 0) (.toFixed l35-var 6) "0 (exact)") 16)
                  " | " (.padStart (if (= vr js/Infinity) "Inf" (.toFixed vr 1)) 13)
                  " | " (.padStart (.toFixed (get-in r [:L2-standard :ess-mean]) 0) 6)
                  " | " (.padStart (.toFixed (get-in r [:L3.5-analytical :ess-mean]) 0) 8)
                  " | " (.padStart (.toFixed (:ess-improvement r) 1) 9)
                  " |"))))

(println)
(println "| K  | L3.5 time (ms) | L2 time (ms) |")
(println "|----|----------------|--------------|")
(doseq [r results-by-size]
  (println (str "| " (.padStart (str (:K r)) 2)
                " | " (.padStart (.toFixed (get-in r [:L3.5-analytical :timing-ms]) 3) 14)
                " | " (.padStart (.toFixed (get-in r [:L2-standard :timing-ms]) 3) 12)
                " |")))

;; ---------------------------------------------------------------------------
;; Write JSON results
;; ---------------------------------------------------------------------------

(write-json "data.json"
  {:experiment "l3.5-combinator-conjugacy"
   :combinator "Map"
   :kernel "normal-normal conjugate"
   :description "Conjugacy detection inside Map combinator kernels. Per-element analytical elimination via L3.5."
   :timestamp (.toISOString (js/Date.))
   :hardware {:platform "macOS" :chip "Apple Silicon" :gpu "Metal"}
   :config {:particles_per_K particles-for-K :n_trials n-trials :map_sizes map-sizes}
   :kernel_info {:prior "N(0, 10)" :likelihood "N(mu, 1)" :family "normal-normal"
                 :conjugate_pairs (mapv (fn [p] {:prior (name (:prior-addr p))
                                                  :obs (name (:obs-addr p))
                                                  :family (name (:family p))})
                                        (get-in kernel-l35 [:schema :conjugate-pairs]))}
   :results-by-size (mapv (fn [r]
                           (update r :variance-reduction
                                   #(if (= % js/Infinity) "Infinity" %)))
                         results-by-size)
   :summary
   {:mean_variance_reduction
    (let [finite (filter #(and (number? %) (js/isFinite %))
                         (map :variance-reduction results-by-size))]
      (if (seq finite) (mean finite) "Infinity"))
    :mean_ess_improvement (mean (map :ess-improvement results-by-size))
    :all_exact? (every? #(= js/Infinity (:variance-reduction %)) results-by-size)}})

(println "\nL3.5 combinator conjugacy benchmark complete.")
