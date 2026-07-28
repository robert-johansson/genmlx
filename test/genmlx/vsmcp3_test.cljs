;; @tier medium
;; genmlx-ke97: vectorized SMCP3 (standard-SMC path) — semantic equivalence
;; with scalar smcp3, the loud kernel decline, and the analytic log-ML anchor.
(ns genmlx.vsmcp3-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn- assert-true [desc ok?]
  (if ok?
    (do (swap! pass inc) (println "  PASS:" desc))
    (do (swap! fail inc) (println "  FAIL:" desc))))
(defn- assert-close [desc expected actual tol]
  (let [ok? (and (js/isFinite actual) (< (js/Math.abs (- expected actual)) tol))]
    (if ok?
      (do (swap! pass inc) (println "  PASS:" desc "(", actual, "≈", expected, ")"))
      (do (swap! fail inc) (println "  FAIL:" desc "expected" expected "got" actual "tol" tol)))))

;; --------------------------------------------------------------------------
;; Models: the inference_smc_test normal-normal (analytic log-ML known) and a
;; fully-observed model (log-ML exact — no sampling, both paths must agree to
;; float32 accumulation noise).
;; --------------------------------------------------------------------------

(def nn-model
  (dyn/auto-key
    (gen [ys]
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (doseq [[i _] (map-indexed vector ys)]
          (trace (keyword (str "y" i))
                 (dist/gaussian mu 1)))
        mu))))
(def ys [2.8 3.1 2.9 3.2 3.0])
(def obs-seq
  (mapv (fn [[i y]] (cm/choicemap (keyword (str "y" i)) (mx/scalar y)))
        (map-indexed vector ys)))
;; From inference_smc_test: log p(y) = -7.7979 (MVN integral)
(def analytic-log-ml -7.797905896206512)

(def observed-only-model
  (dyn/auto-key
    (gen []
      (trace :a (dist/gaussian 0 1))
      (trace :b (dist/gaussian 2 3))
      nil)))
(def observed-only-seq
  [(cm/choicemap :a (mx/scalar 0.5))
   (cm/choicemap :b (mx/scalar 1.5))])

(println "\n-- vsmcp3: analytic log-ML anchor (normal-normal, N=1000, 5 steps) --")
(let [{:keys [log-ml-estimate vtrace]}
      (smcp3/vsmcp3 {:particles 1000 :key (rng/fresh-key 42)}
                    nn-model [ys] obs-seq)]
  ;; Same tolerance rationale as inference_smc_test's SMC log-ML bound.
  (assert-close "vsmcp3 log-ML ~ analytic" analytic-log-ml
                (mx/item log-ml-estimate) 0.6)
  (assert-true "vtrace has N=1000 particles" (= 1000 (:n-particles vtrace))))

(println "\n-- scalar smcp3 on the same problem (statistical agreement) --")
(let [scalar-ml (mx/item (:log-ml-estimate
                          (smcp3/smcp3 {:particles 1000 :key (rng/fresh-key 43)}
                                       nn-model [ys] obs-seq)))
      v-ml (mx/item (:log-ml-estimate
                     (smcp3/vsmcp3 {:particles 1000 :key (rng/fresh-key 43)}
                                   nn-model [ys] obs-seq)))]
  (assert-close "scalar smcp3 log-ML ~ analytic" analytic-log-ml scalar-ml 0.6)
  ;; Different PRNG stream shapes -> agreement is statistical, both anchored
  ;; to the same analytic value; the two estimates must sit within the
  ;; combined SMC noise band of each other.
  (assert-close "vsmcp3 ~ scalar smcp3" scalar-ml v-ml 1.0))

(println "\n-- fully-observed model: log-ML is exact, no sampling noise --")
(let [scalar-ml (mx/item (:log-ml-estimate
                          (smcp3/smcp3 {:particles 8 :key (rng/fresh-key 7)}
                                       observed-only-model [] observed-only-seq)))
      v-ml (mx/item (:log-ml-estimate
                     (smcp3/vsmcp3 {:particles 8 :key (rng/fresh-key 7)}
                                   observed-only-model [] observed-only-seq)))]
  (assert-close "scalar == batched (deterministic, float32)" scalar-ml v-ml 1e-4))

(println "\n-- vsmcp3 == vsmc under identical opts+key (delegation pinned) --")
(let [opts {:particles 64 :key (rng/fresh-key 99)}
      a (mx/item (:log-ml-estimate (smcp3/vsmcp3 opts nn-model [ys] obs-seq)))
      b (mx/item (:log-ml-estimate (smc/vsmc opts nn-model [ys] obs-seq)))]
  (assert-close "vsmcp3 log-ML == vsmc log-ML (same key)" b a 1e-6))

(println "\n-- scalar-only opts decline LOUDLY --")
(let [dummy-kernel (dyn/auto-key (gen [_] (trace :z (dist/gaussian 0 1))))]
  (doseq [[k v] {:forward-kernel dummy-kernel
                 :backward-kernel dummy-kernel
                 :init-proposal dummy-kernel
                 :rejuvenation-fn (fn [tr _] tr)}]
    (let [threw? (try (smcp3/vsmcp3 {:particles 4 k v} nn-model [ys] obs-seq)
                      false
                      (catch :default e
                        (boolean (some #{k} (:scalar-only-opts (ex-data e))))))]
      (assert-true (str "vsmcp3 throws on " k " (named in ex-data)") threw?))))

(println "\n-- statistical sweep: 3 seeds, all within band --")
(doseq [seed [101 202 303]]
  (let [ml (mx/item (:log-ml-estimate
                     (smcp3/vsmcp3 {:particles 1000 :key (rng/fresh-key seed)}
                                   nn-model [ys] obs-seq)))]
    (assert-close (str "seed " seed " log-ML ~ analytic") analytic-log-ml ml 0.6)))

(println (str "\n== vsmcp3_test: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (js/process.exit 1))
