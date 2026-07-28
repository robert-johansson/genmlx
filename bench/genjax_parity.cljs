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
            [genmlx.vectorized :as vect]
            [genmlx.inference.mcmc :as mcmc])
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
(def alg-type (get-in spec [:algorithm :type]))
(assert (contains? #{"importance_sampling" "mala"} alg-type))
(when (= alg-type "importance_sampling")
  (assert (= "prior" (get-in spec [:algorithm :proposal]))))
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

(defn time-ms [f]
  (let [t0 (js/performance.now)] (f) (- (js/performance.now) t0)))

(defn median [xs] (nth (vec (sort xs)) (quot (count xs) 2)))

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
;; MALA sweep (fused-mala: whole chain as one lazy graph — the scan analog).
;; Timed unit matches the GenJAX side: init (constrained generate, latents
;; from prior) + S-step chain, materialized. :chain-fn from the first call is
;; reused for the rest (documented API), mirroring jit's compiled-fn reuse.
;; ---------------------------------------------------------------------------

(defn mala-one-call [s device chain-fn-box k]
  (let [alg (:algorithm spec)
        r   (mcmc/fused-mala
             (cond-> {:samples s :burn (:burn_in alg) :thin (:thin alg)
                      :step-size (:step_size alg)
                      :addresses (mapv keyword (:selection alg))
                      :key k :device device}
               @chain-fn-box (assoc :chain-fn @chain-fn-box))
             hmodel [xs] obs)]
    (mx/eval! (:samples r))
    (vreset! chain-fn-box (:chain-fn r))
    r))

(defn probe-device [s]
  (into {}
        (for [d [:cpu :gpu]]
          (let [box (volatile! nil)
                ks  (mapv (fn [k] (mx/eval! k) k)
                          (rng/split-n (rng/fresh-key 99) 3))]
            (mala-one-call s d box (nth ks 0))
            (mx/sweep-dead-arrays!)
            [d (median (mapv (fn [i]
                               (let [t (time-ms #(mala-one-call s d box (nth ks i)))]
                                 (mx/sweep-dead-arrays!) t))
                             [1 2]))]))))

(defn slope-tail-mean [r s]
  ;; :samples is [S,D]; column order follows :addresses ([:slope :intercept]).
  ;; The spec's posterior reference (1.99 vs -1.10) screams if this is wrong.
  (let [rows (js->clj (mx/->clj (:samples r)))
        tail (drop (quot s 2) rows)]
    (/ (reduce + (map first tail)) (count tail))))

(defn sweep-mala [s device]
  (let [{:keys [warmup_runs timed_runs]} (:measurement spec)
        run-keys (mapv (fn [k] (mx/eval! k) k)
                       (rng/split-n (rng/fresh-key 0) (+ warmup_runs timed_runs)))
        box       (volatile! nil)
        t0        (js/performance.now)
        _         (mala-one-call s device box (nth run-keys 0))
        warmup-ms (- (js/performance.now) t0)
        _         (mx/sweep-dead-arrays!)
        _         (doseq [r (range 1 warmup_runs)]
                    (mala-one-call s device box (nth run-keys r))
                    (mx/sweep-dead-arrays!))
        last-r    (volatile! nil)
        accs      (volatile! [])
        timings   (mapv (fn [r]
                          (let [k  (nth run-keys (+ warmup_runs r))
                                t0 (js/performance.now)
                                res (mala-one-call s device box k)
                                ms (- (js/performance.now) t0)]
                            (vreset! last-r res)
                            (vswap! accs conj (:acceptance-rate res))
                            (mx/sweep-dead-arrays!)
                            ms))
                        (range timed_runs))
        sorted    (vec (sort timings))]
    {:n_steps s
     :warmup_ms   (round-to warmup-ms 3)
     :median_ms   (round-to (percentile sorted 0.5) 4)
     :p10_ms      (round-to (percentile sorted 0.1) 4)
     :p90_ms      (round-to (percentile sorted 0.9) 4)
     ;; The block-compiled fallback returns :acceptance-rate nil (it does not
     ;; track acceptance) — report null honestly instead of coercing nil to 0
     ;; (genmlx-d62h: CLJS + treats nil as 0, which forged a 0.000 rate).
     :acceptance_rate (let [as (remove nil? @accs)]
                        (when (seq as)
                          (round-to (/ (reduce + as) (count as)) 4)))
     :mean_slope_tail (round-to (slope-tail-mean @last-r s) 4)}))

;; ---------------------------------------------------------------------------
;; Run + report
;; ---------------------------------------------------------------------------

(defn sh [cmd] (.trim (.toString (.execSync cp cmd))))

(println (str "spec " (:id spec) " — GenMLX side (genmlx "
              (sh "git rev-parse --short HEAD") ")"))

(defn write-result! [conv-diff extra rows]
  (let [stamp  (-> (.toISOString (js/Date.)) (.replace (js/RegExp. "[-:]" "g") "")
                   (.replace (js/RegExp. "\\..*") "") (.replace "T" "-"))
        result (merge {:spec_id (:id spec)
                       :side "genmlx"
                       :timestamp (.toISOString (js/Date.))
                       :versions {:genmlx (sh "git rev-parse HEAD")
                                  :mlx_node (sh "git -C mlx-node rev-parse HEAD")}
                       :host {:nvidia_smi (try (sh "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
                                               (catch :default _ nil))}
                       :weight_convention_absdiff conv-diff
                       :results rows}
                      extra)
        out-dir  (.join node-path (.dirname node-path spec-path) ".." "results")
        out-path (.join node-path out-dir (str (:id spec) ".genmlx." stamp ".json"))]
    (.writeFileSync fs out-path (str (js/JSON.stringify (clj->js result) nil 2) "\n"))
    (println (str "wrote " out-path))))

(let [conv-diff (check-weight-convention)]
  (println (str "weight-convention check OK (|diff| = " (.toExponential conv-diff 2) ")"))
  (case alg-type
    "importance_sampling"
    (let [exact (analytical-exact-logz)]
      (when exact
        (println (str "analytical exact logZ (L3 elimination, ground truth): "
                      (round-to exact 4))))
      (write-result!
       conv-diff {:analytical_exact_logz (some-> exact (round-to 4))}
       (mapv (fn [n]
               (let [r (sweep-n n)]
                 (println (str "  N=" (.padStart (str n) 6)
                               "  warmup " (.padStart (.toFixed (:warmup_ms r) 1) 9) " ms"
                               "   median " (.padStart (.toFixed (:median_ms r) 3) 8) " ms"
                               "   p10 " (.padStart (.toFixed (:p10_ms r) 3) 8)
                               "   p90 " (.padStart (.toFixed (:p90_ms r) 3) 8)
                               "   logZ~ " (.toFixed (:logmeanexp_weight r) 3)))
                 r))
             (get-in spec [:sweep :n_particles]))))

    "mala"
    (let [probe  (probe-device (second (get-in spec [:sweep :n_steps])))
          device (key (apply min-key val probe))]
      (println (str "device probe (median ms at S="
                    (second (get-in spec [:sweep :n_steps])) "): "
                    (pr-str probe) " -> " device))
      (write-result!
       conv-diff {:device (name device)
                  :device_probe (into {} (map (fn [[k v]] [k (round-to v 3)]) probe))}
       (mapv (fn [s]
               (let [r (sweep-mala s device)]
                 (println (str "  S=" (.padStart (str s) 5)
                               "  warmup " (.padStart (.toFixed (:warmup_ms r) 1) 9) " ms"
                               "   median " (.padStart (.toFixed (:median_ms r) 3) 9) " ms"
                               "   accept " (if-some [a (:acceptance_rate r)]
                                              (.toFixed a 3) "nil(fallback)")
                               "   slope-tail " (.toFixed (:mean_slope_tail r) 3)))
                 r))
             (get-in spec [:sweep :n_steps]))))))
