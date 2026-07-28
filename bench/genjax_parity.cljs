;; @tier bench
(ns genjax-parity
  "GenJAX parity bench (bean genmlx-ecsi): the GenMLX side of the shared-spec
   comparison. Reads the SAME JSON spec the GenJAX runner consumes
   (genjax-parity/specs/*.json — source of truth for model params, literal
   data, sweep, measurement protocol), runs the algorithm per the fidelity
   rules (float32, warmup reported separately, device-synced timing), and
   writes one JSON result into the parity repo's results/.

   Currently supports: linear_regression + importance_sampling (prior
   proposal) == dyn/vgenerate, timed as graph build + weight eval.

   The benched path STRIPS the analytical dispatcher (as cost_per_particle
   does): this model is exactly the joint linear-Gaussian class L3
   eliminates, and the spec pins the ALGORITHM — importance sampling on the
   handler path — not the best trick for the model. The analytical exact
   logZ is reported separately (key-independence verified) as ground truth
   for both sides' logmeanexp sanity metric.

   Correctness guard (always on, mirrors parity/run_genjax.py): a scalar
   p/generate weight under the prior proposal must equal the hand-computed
   Gaussian log-likelihood of the constrained observations.

   Usage (bench tier: STRICTLY SERIAL, nothing else on the GPU):
     bun run --bun nbb bench/genjax_parity.cljs [path/to/spec.json]
   Default spec: ../genjax-parity/specs/linreg_is.json"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vect])
  (:require-macros [genmlx.gen :refer [gen]]))

(def fs (js/require "fs"))
(def node-path (js/require "path"))
(def cp (js/require "child_process"))

;; ---------------------------------------------------------------------------
;; Spec
;; ---------------------------------------------------------------------------

(def spec-path
  (or (first *command-line-args*) "../genjax-parity/specs/linreg_is.json"))

(def spec
  (js->clj (js/JSON.parse (.readFileSync fs spec-path "utf8")) :keywordize-keys true))

(assert (= "float32" (:float_dtype spec)))
(assert (= "linear_regression" (get-in spec [:model :type])))
(assert (= "importance_sampling" (get-in spec [:algorithm :type])))
(assert (= "prior" (get-in spec [:algorithm :proposal])))
(doseq [site [:slope :intercept]]
  (assert (= "gaussian" (get-in spec [:model :prior site :dist]))))
(assert (= "gaussian" (get-in spec [:model :likelihood :dist])))

(def xs (get-in spec [:data :xs]))
(def ys (get-in spec [:data :ys]))
(def lik-sigma (get-in spec [:model :likelihood :sigma]))

;; ---------------------------------------------------------------------------
;; Model + constraints (one scalar site per observation, same as GenJAX side)
;; ---------------------------------------------------------------------------

(def model
  (let [s-mu (get-in spec [:model :prior :slope :mu])
        s-sd (get-in spec [:model :prior :slope :sigma])
        i-mu (get-in spec [:model :prior :intercept :mu])
        i-sd (get-in spec [:model :prior :intercept :sigma])]
    (dyn/auto-key
      (gen [xs]
        (let [slope     (trace :slope (dist/gaussian s-mu s-sd))
              intercept (trace :intercept (dist/gaussian i-mu i-sd))]
          (doseq [[j x] (map-indexed vector xs)]
            (trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope x) intercept)
                                  lik-sigma)))
          slope)))))

(def hmodel (dyn/strip-analytical-path model))

(def obs
  (apply cm/choicemap
         (mapcat (fn [[j y]] [(keyword (str "y" j)) (mx/scalar y)])
                 (map-indexed vector ys))))

;; ---------------------------------------------------------------------------
;; Correctness guard
;; ---------------------------------------------------------------------------

(defn gaussian-logpdf [y m s]
  (- (* -0.5 (js/Math.pow (/ (- y m) s) 2))
     (js/Math.log s)
     (* 0.5 (js/Math.log (* 2 js/Math.PI)))))

(defn check-weight-convention []
  (let [{:keys [trace weight]}
        (p/generate (dyn/with-key hmodel (rng/fresh-key 7)) [xs] obs)
        ch        (:choices trace)
        slope     (mx/item (cm/get-value (cm/get-submap ch :slope)))
        intercept (mx/item (cm/get-value (cm/get-submap ch :intercept)))
        expected  (reduce + (map (fn [x y]
                                   (gaussian-logpdf y (+ (* slope x) intercept)
                                                    lik-sigma))
                                 xs ys))
        w    (mx/item weight)
        diff (js/Math.abs (- w expected))
        tol  (max 0.01 (* 1e-4 (js/Math.abs expected)))]  ;; f32 vs f64 host math
    (when-not (< diff tol)
      (throw (ex-info "weight convention mismatch"
                      {:weight w :expected expected :diff diff})))
    diff))

;; Bonus ground truth: unstripped generate hits L3 joint linear-Gaussian
;; elimination, whose weight is the EXACT marginal — verified key-independent
;; before trusting it. Returns nil (with a notice) if elimination didn't apply.
(defn analytical-exact-logz []
  (let [w-at (fn [seed]
               (mx/item (:weight (p/generate (dyn/with-key model (rng/fresh-key seed))
                                             [xs] obs))))
        a (w-at 1) b (w-at 2)]
    (if (< (js/Math.abs (- a b)) 1e-4)
      a
      (do (println "  (analytical path not key-independent — no exact logZ)")
          nil))))

;; ---------------------------------------------------------------------------
;; Timing (protocol from the spec's :measurement, mirroring run_genjax.py)
;; ---------------------------------------------------------------------------

(defn percentile [sorted q]
  (let [i    (* q (dec (count sorted)))
        lo   (js/Math.floor i)
        hi   (js/Math.ceil i)
        frac (- i lo)]
    (+ (* (nth sorted lo) (- 1 frac)) (* (nth sorted hi) frac))))

(defn round-to [x d]
  (let [f (js/Math.pow 10 d)] (/ (js/Math.round (* x f)) f)))

(defn sweep-n [n]
  (let [{:keys [warmup_runs timed_runs]} (:measurement spec)
        ;; Distinct pre-materialized keys per run: no key work in timed regions.
        run-keys (mapv (fn [k] (mx/eval! k) k)
                       (rng/split-n (rng/fresh-key 0) (+ warmup_runs timed_runs)))
        one-call (fn [k]
                   (let [vt (dyn/vgenerate hmodel [xs] obs n k)]
                     (mx/eval! (:weight vt))
                     vt))
        t0        (js/performance.now)
        _         (one-call (nth run-keys 0))
        warmup-ms (- (js/performance.now) t0)
        _         (mx/sweep-dead-arrays!)
        _         (doseq [r (range 1 warmup_runs)]
                    (one-call (nth run-keys r))
                    (mx/sweep-dead-arrays!))
        last-vt   (volatile! nil)
        timings   (mapv (fn [r]
                          (let [k  (nth run-keys (+ warmup_runs r))
                                t0 (js/performance.now)
                                vt (one-call k)
                                ms (- (js/performance.now) t0)]
                            (vreset! last-vt vt)
                            (mx/sweep-dead-arrays!)
                            ms))
                        (range timed_runs))
        sorted    (vec (sort timings))
        lme       (mx/item (vect/vtrace-log-ml-estimate @last-vt))]
    {:n_particles n
     :warmup_ms   (round-to warmup-ms 3)
     :median_ms   (round-to (percentile sorted 0.5) 4)
     :p10_ms      (round-to (percentile sorted 0.1) 4)
     :p90_ms      (round-to (percentile sorted 0.9) 4)
     :logmeanexp_weight (round-to lme 4)}))

;; ---------------------------------------------------------------------------
;; Run + report
;; ---------------------------------------------------------------------------

(defn sh [cmd] (.trim (.toString (.execSync cp cmd))))

(println (str "spec " (:id spec) " — GenMLX side (genmlx "
              (sh "git rev-parse --short HEAD") ")"))

(let [conv-diff (check-weight-convention)]
  (println (str "weight-convention check OK (|diff| = " (.toExponential conv-diff 2) ")"))
  (let [exact (analytical-exact-logz)]
    (when exact
      (println (str "analytical exact logZ (L3 elimination, ground truth): "
                    (round-to exact 4))))
    (let [rows (mapv (fn [n]
                       (let [r (sweep-n n)]
                         (println (str "  N=" (.padStart (str n) 6)
                                       "  warmup " (.padStart (.toFixed (:warmup_ms r) 1) 9) " ms"
                                       "   median " (.padStart (.toFixed (:median_ms r) 3) 8) " ms"
                                       "   p10 " (.padStart (.toFixed (:p10_ms r) 3) 8)
                                       "   p90 " (.padStart (.toFixed (:p90_ms r) 3) 8)
                                       "   logZ~ " (.toFixed (:logmeanexp_weight r) 3)))
                         r))
                     (get-in spec [:sweep :n_particles]))
          stamp  (-> (.toISOString (js/Date.)) (.replace (js/RegExp. "[-:]" "g") "")
                     (.replace (js/RegExp. "\\..*") "") (.replace "T" "-"))
          result {:spec_id (:id spec)
                  :side "genmlx"
                  :timestamp (.toISOString (js/Date.))
                  :versions {:genmlx (sh "git rev-parse HEAD")
                             :mlx_node (sh "git -C mlx-node rev-parse HEAD")}
                  :host {:nvidia_smi (try (sh "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
                                          (catch :default _ nil))}
                  :weight_convention_absdiff conv-diff
                  :analytical_exact_logz (some-> exact (round-to 4))
                  :results rows}
          out-dir  (.join node-path (.dirname node-path spec-path) ".." "results")
          out-path (.join node-path out-dir (str (:id spec) ".genmlx." stamp ".json"))]
      (.writeFileSync fs out-path (str (js/JSON.stringify (clj->js result) nil 2) "\n"))
      (println (str "wrote " out-path)))))
