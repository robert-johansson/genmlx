;; @tier fast
(ns genmlx.capture-replay-test
  "Captured-replay equivalence + safety pins (genmlx-7prh).

   compiled-call-captured returns EVALUATED outputs and, after the first
   successful capture (CUDA only), replays as launch-only retained graph
   execs with inputs memcpy'd into staged buffers. These tests pin the
   contract that makes that safe:

   - value equivalence with direct execution, across DIFFERENT inputs
     (the staging memcpy actually changes the computation)
   - PRNG-key inputs steer in-graph randomness (same key = same draws,
     different key = different draws)
   - no aliasing: results from earlier calls stay stable after later calls
     (outputs are copies, never views of the retained buffers)
   - shape drift falls back per call to the trace-cache path, and the
     capture keeps working afterwards
   - foreign GPU work between calls does not corrupt replays
   - vgenerate-compiled and persist-chain ride the captured path on CUDA
     (compiled-captured? — the honesty probe — asserts WHICH path ran,
     so a silently-degraded capture fails here instead of just being slow)

   On Metal / CPU-only builds every equivalence still holds via the plain
   fallback; only the captured? assertions are conditional."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as th]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(def cuda? (not (mx/metal-is-available?)))

(defn- ->v [a] (js->clj (mx/->clj a)))

;; ---------------------------------------------------------------------------
;; Raw handle behavior
;; ---------------------------------------------------------------------------

(deftest captured-call-matches-direct-execution
  (let [f (fn [x] (to-array [(mx/sum (mx/square x))
                             (mx/add (mx/multiply x 2.0) 1.0)]))
        h (mx/compile-create f)
        x1 (mx/array [1.0 2.0 3.0])
        x2 (mx/array [-4.0 0.5 7.0])]
    (mx/eval! x1 x2)
    (let [r1 (mx/compiled-call-captured h #js [x1])
          direct1 (f x1)
          _ (mx/eval! (aget direct1 0) (aget direct1 1))
          r2 (mx/compiled-call-captured h #js [x2])
          direct2 (f x2)
          _ (mx/eval! (aget direct2 0) (aget direct2 1))]
      (is (= (->v (aget direct1 0)) (->v (aget r1 0))) "call 1 sum matches")
      (is (= (->v (aget direct1 1)) (->v (aget r1 1))) "call 1 affine matches")
      (is (= (->v (aget direct2 0)) (->v (aget r2 0)))
          "call 2 (replay with NEW input) sum matches")
      (is (= (->v (aget direct2 1)) (->v (aget r2 1)))
          "call 2 affine matches — the staging memcpy took effect")
      (when cuda?
        (is (mx/compiled-captured? h) "CUDA: the handle actually captured")))
    (mx/compiled-free! h)))

(deftest key-input-steers-in-graph-randomness
  ;; The IS/vgenerate factoring: the PRNG key is the graph INPUT; splits and
  ;; sampling stay in-graph. Replays must honor a new key. Direct (unfused)
  ;; vs compiled (fused) erfinv chains differ in rounding, so the
  ;; direct-vs-captured comparison uses the vgenerate_compiled_test relative
  ;; tolerance; captured-vs-captured stays bit-exact.
  (let [f (fn [k]
            (let [[k1 k2] (rng/split k)]
              (to-array [(rng/normal k1 [4]) (rng/uniform k2 [4])])))
        h (mx/compile-create f)
        ka (rng/fresh-key 11)
        kb (rng/fresh-key 22)]
    (mx/eval! ka kb)
    (let [ra (mx/compiled-call-captured h #js [ka])
          rb (mx/compiled-call-captured h #js [kb])
          ra2 (mx/compiled-call-captured h #js [ka])
          da (f ka)
          _ (mx/eval! (aget da 0) (aget da 1))]
      (is (every? true? (map #(th/close? %1 %2 1e-4)
                             (->v (aget da 0)) (->v (aget ra 0))))
          "captured draw matches the direct draw for the same key (fusion tol)")
      (is (not= (->v (aget ra 0)) (->v (aget rb 0)))
          "different key, different draws")
      (is (= (->v (aget ra 0)) (->v (aget ra2 0)))
          "same key replayed later reproduces bit-exactly"))
    (mx/compiled-free! h)))

(deftest no-aliasing-across-calls
  (let [f (fn [x] (to-array [(mx/multiply x 10.0)]))
        h (mx/compile-create f)
        x1 (mx/array [1.0 1.0])
        x2 (mx/array [2.0 2.0])
        x3 (mx/array [3.0 3.0])]
    (mx/eval! x1 x2 x3)
    (let [r1 (aget (mx/compiled-call-captured h #js [x1]) 0)
          v1 (->v r1)
          r2 (aget (mx/compiled-call-captured h #js [x2]) 0)
          v2 (->v r2)
          _  (mx/compiled-call-captured h #js [x3])]
      (is (= [10 10] v1))
      (is (= [20 20] v2))
      (is (= v1 (->v r1))
          "call-1 result unchanged after two later launches (no aliasing)")
      (is (= v2 (->v r2))
          "call-2 result unchanged after a later launch"))
    (mx/compiled-free! h)))

(deftest shape-drift-falls-back-and-capture-survives
  (let [f (fn [x] (to-array [(mx/sum x)]))
        h (mx/compile-create f)
        x3 (mx/array [1.0 2.0 3.0])
        x5 (mx/array [1.0 2.0 3.0 4.0 5.0])]
    (mx/eval! x3 x5)
    ;; sum -> []-shaped scalar; ->clj of a scalar is a NUMBER, not a vector.
    (is (= 6 (->v (aget (mx/compiled-call-captured h #js [x3]) 0))))
    (is (= 15 (->v (aget (mx/compiled-call-captured h #js [x5]) 0)))
        "drifted shape returns the right value via the trace-cache path")
    (is (= 6 (->v (aget (mx/compiled-call-captured h #js [x3]) 0)))
        "the original capture still replays correctly after the drifted call")
    (mx/compiled-free! h)))

(deftest foreign-gpu-work-between-calls-is-harmless
  (let [f (fn [x] (to-array [(mx/add x 100.0)]))
        h (mx/compile-create f)
        x1 (mx/array [1.0])
        x2 (mx/array [2.0])]
    (mx/eval! x1 x2)
    (is (= [101] (->v (aget (mx/compiled-call-captured h #js [x1]) 0))))
    ;; Unrelated eval on the normal path between captured calls.
    (mx/eval! (mx/sum (mx/multiply (mx/ones [64 64]) 3.0)))
    (is (= [102] (->v (aget (mx/compiled-call-captured h #js [x2]) 0)))
        "replay after interleaved foreign eval is correct")
    (mx/compiled-free! h)))

;; ---------------------------------------------------------------------------
;; Product paths: vgenerate-compiled + fused chains
;; ---------------------------------------------------------------------------

(def linreg
  (dyn/auto-key
   (gen []
     (let [slope     (trace :slope (dist/gaussian 0 2))
           intercept (trace :intercept (dist/gaussian 0 2))]
       (doseq [[j x] (map-indexed vector [0.0 1.0 2.0 3.0])]
         (trace (keyword (str "y" j))
                (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1)))
       slope))))

(def linreg-h (dyn/strip-analytical-path linreg))

(def linreg-obs
  (apply cm/choicemap
         (mapcat (fn [[j y]] [(keyword (str "y" j)) (mx/scalar y)])
                 (map-indexed vector [0.5 1.4 2.6 3.4]))))

(deftest vgenerate-compiled-rides-the-captured-path
  (when-not cuda?
    (println "  (Metal: vgenerate-compiled throws by design — skipping)"))
  (when cuda?
    (let [n 64
          cf (dyn/vgenerate-compiled linreg-h [] linreg-obs n)
          k1 (rng/fresh-key 5)
          k2 (rng/fresh-key 6)
          vt-plain (dyn/vgenerate linreg-h [] linreg-obs n k1)
          vt-cap1 ((:call cf) k1)
          vt-cap2 ((:call cf) k2)
          vt-cap1-again ((:call cf) k1)]
      ;; handler (unfused) vs compiled (fused) differ by kernel-fusion
      ;; rounding — the vgenerate_compiled_test relative-tolerance
      ;; convention applies; captured-vs-captured stays bit-exact.
      (is (every? true? (map #(th/close? %1 %2 (max 1e-2 (* 1e-4 (js/Math.abs %1))))
                             (->v (:weight vt-plain)) (->v (:weight vt-cap1))))
          "captured replay weight matches the handler path (fusion tol)")
      (is (every? true? (map #(th/close? %1 %2 1e-3)
                             (->v (cm/get-value (cm/get-submap (:choices vt-plain) :slope)))
                             (->v (cm/get-value (cm/get-submap (:choices vt-cap1) :slope)))))
          "captured replay latents match the handler path (fusion tol)")
      (is (not= (->v (:weight vt-cap1)) (->v (:weight vt-cap2)))
          "a different key gives different particles")
      (is (= (->v (:weight vt-cap1)) (->v (:weight vt-cap1-again)))
          "same key reproduces after other keys ran (no state bleed)")
      ((:free! cf)))))

(deftest fused-mala-compiled-matches-eager
  ;; The whole-call factory (init generate + val-grad + noise + chain in ONE
  ;; traced graph — the h5wg scaffolding lever) replicates the eager PRNG
  ;; stream layout ([init-key stream-key] = split(k); noise/uniforms split
  ;; off stream-key), so it follows the SAME chain — to kernel-fusion
  ;; rounding: fusing the init/chain boundary into one graph reorders
  ;; float32 contractions vs the eager factoring (~1e-6 relative, the same
  ;; class vgenerate_compiled_test tolerates). Factory-vs-factory stays
  ;; BIT-exact.
  (when-not cuda?
    (println "  (Metal: fused-mala-compiled throws by design — skipping)"))
  (when cuda?
    (let [opts {:samples 40 :burn 5 :thin 1 :step-size 0.15
                :addresses [:slope :intercept]}
          f (mcmc/fused-mala-compiled opts linreg-h [] linreg-obs)
          k1 (rng/fresh-key 31)
          k2 (rng/fresh-key 32)
          r1 ((:call f) k1)
          r2 ((:call f) k2)
          r1b ((:call f) k1)
          eager (mcmc/fused-mala (assoc opts :key k1 :device :gpu)
                                 linreg-h [] linreg-obs)
          flat (fn [r] (flatten (->v (:samples r))))]
      (is (every? true? (map #(th/close? %1 %2 (max 1e-3 (* 1e-4 (js/Math.abs %1))))
                             (flat eager) (flat r1)))
          "factory follows the eager chain (same stream, fusion tolerance)")
      (is (< (js/Math.abs (- (:acceptance-rate eager) (:acceptance-rate r1)))
             0.051)
          "acceptance agrees (a boundary step may flip under 1e-6 rounding)")
      (is (not= (->v (:samples r1)) (->v (:samples r2)))
          "different key, different chain")
      (is (= (->v (:samples r1)) (->v (:samples r1b)))
          "same key reproduces BIT-exactly after another key ran")
      ((:free! f)))))

(deftest fused-mala-compiled-degrades-on-untraceable-models
  ;; gamma latent: the init generate's sampler calls item per draw
  ;; (Marsaglia-Tsang) -> the trace fails -> LOUD permanent degrade to the
  ;; eager path per call (y3ls doctrine).
  (when cuda?
    (let [m (dyn/auto-key
             (gen []
               (let [th (trace :theta (dist/gamma-dist 2 2))]
                 (trace :y (dist/gaussian th 1))
                 th)))
          hm (dyn/strip-analytical-path m)
          ob (cm/choicemap :y (mx/scalar 0.7))
          f (mcmc/fused-mala-compiled {:samples 10 :burn 0 :thin 1
                                       :step-size 0.1 :addresses [:theta]}
                                      hm [] ob)
          r ((:call f) (rng/fresh-key 44))
          r2 ((:call f) (rng/fresh-key 45))]
      (is (= [10 1] (vec (mx/shape (:samples r)))) "degraded call returns a chain")
      (is (some? (:acceptance-rate r2)) "later calls keep working")
      ((:free! f)))))

(deftest fused-hmc-compiled-matches-eager
  ;; The HMC analog of the whole-call factory: same stream-layout
  ;; replication, same fusion-rounding tolerance vs eager, bit-exact
  ;; factory-vs-factory.
  (when cuda?
    (let [opts {:samples 30 :burn 5 :thin 1 :step-size 0.1
                :leapfrog-steps 8 :addresses [:slope :intercept]}
          f (mcmc/fused-hmc-compiled opts linreg-h [] linreg-obs)
          k1 (rng/fresh-key 51)
          k2 (rng/fresh-key 52)
          r1 ((:call f) k1)
          r2 ((:call f) k2)
          r1b ((:call f) k1)
          eager (mcmc/fused-hmc (assoc opts :key k1 :device :gpu)
                                linreg-h [] linreg-obs)
          flat (fn [r] (flatten (->v (:samples r))))]
      (is (every? true? (map #(th/close? %1 %2 (max 1e-3 (* 1e-4 (js/Math.abs %1))))
                             (flat eager) (flat r1)))
          "factory follows the eager HMC chain (same stream, fusion tolerance)")
      (is (< (js/Math.abs (- (:acceptance-rate eager) (:acceptance-rate r1)))
             0.051)
          "acceptance agrees")
      (is (not= (->v (:samples r1)) (->v (:samples r2)))
          "different key, different chain")
      (is (= (->v (:samples r1)) (->v (:samples r1b)))
          "same key reproduces BIT-exactly")
      ((:free! f)))))

(deftest captured-point-fn-value-and-grad
  ;; persist-point-fn now rides the captured path: a value-and-grad handle
  ;; must return correct (value, grad) pairs across DIFFERENT inputs.
  (let [f (fn [q] (mx/negative (mx/sum (mx/square q))))
        vg (mx/value-and-grad f)
        h (mx/compile-create (fn [q] (to-array (vg q))))
        q1 (mx/array [1.0 2.0])
        q2 (mx/array [3.0 -1.0])]
    (mx/eval! q1 q2)
    (let [o1 (mx/compiled-call-captured h #js [q1])
          o2 (mx/compiled-call-captured h #js [q2])]
      (is (= -5 (->v (aget o1 0))) "value at q1")
      (is (= [-2 -4] (->v (aget o1 1))) "grad at q1")
      (is (= -10 (->v (aget o2 0))) "value at q2 (replay, new input)")
      (is (= [-6 2] (->v (aget o2 1))) "grad at q2 (replay, new input)"))
    (mx/compiled-free! h)))

(deftest fused-mala-chain-rides-the-captured-path
  (let [opts {:samples 50 :burn 0 :thin 1 :step-size 0.15
              :addresses [:slope :intercept]
              :key (rng/fresh-key 9) :device (if cuda? :gpu :cpu)}
        r1 (mcmc/fused-mala opts linreg-h [] linreg-obs)
        ;; chain-fn reuse — the replay path the parity bench times
        r2 (mcmc/fused-mala (assoc opts :chain-fn (:chain-fn r1))
                            linreg-h [] linreg-obs)]
    (is (= (->v (:samples r1)) (->v (:samples r2)))
        "same key + reused chain-fn reproduces the chain bit-exactly")
    (is (= (:acceptance-rate r1) (:acceptance-rate r2))
        "acceptance identical across replay")
    (when cuda?
      (is (some? (:chain-fn r1)) "fused path engaged")
      (when-let [h (:genmlx/compiled-handle (meta (:chain-fn r1)))]
        (is (mx/compiled-captured? h)
            "CUDA: the chain handle actually captured (launch-only replays)")))))

(cljs.test/run-tests)
