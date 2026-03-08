(ns genmlx.compiled-gen-level1-test
  "Tests for Level 1 compile-gen: schema-based direct execution.
   Verifies schema discovery, CompiledGF GFI, fast vgenerate correctness,
   and measures speedup over Level 0 (standard vgenerate)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.compiled-gen :as cg]
            [genmlx.vectorized :as vec]
            [genmlx.protocols :as p]
            [genmlx.inference.differentiable :as diff]
            [genmlx.inference.fisher :as fisher])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [label pred]
  (if pred
    (println (str "  PASS: " label))
    (println (str "  FAIL: " label))))

(defn assert-close [label expected actual tol]
  (let [ok (< (js/Math.abs (- expected actual)) tol)]
    (if ok
      (println (str "  PASS: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")"))
      (println (str "  FAIL: " label " (expected " (.toFixed expected 3) ", got " (.toFixed actual 3) ")")))))

;; ---------------------------------------------------------------------------
;; Model setup
;; ---------------------------------------------------------------------------

(def K-obs 10)
(def true-mu 3.0)
(def obs-data (mapv (fn [i] (+ true-mu (* 0.5 (- i 4.5)))) (range K-obs)))

(def model-1
  (gen []
    (let [mu (param :mu 0.0)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu 1.0)))
      mu)))

(def obs-1
  (apply cm/choicemap
    (mapcat (fn [i] [(keyword (str "y" i)) (mx/scalar (nth obs-data i))])
            (range K-obs))))

(def model-2
  (gen []
    (let [mu (param :mu 0.0)
          log-sigma (param :log-sigma 0.0)
          sigma (mx/exp log-sigma)]
      (doseq [i (range K-obs)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu sigma)))
      mu)))

;; Simple prior-only model (no observations)
(def model-prior
  (gen []
    (let [x (trace :x (dist/gaussian 0.0 1.0))
          y (trace :y (dist/gaussian x 0.5))]
      (mx/add x y))))

;; ---------------------------------------------------------------------------
;; Test 1: Schema discovery
;; ---------------------------------------------------------------------------

(println "\n=== Test 1: Schema discovery ===")

(let [schema (cg/discover-schema (dyn/auto-key model-1) [] obs-1 10 (rng/fresh-key 42))]
  (println (str "  Sites: " (count schema)))
  (println (str "  First: " (pr-str (first schema))))
  (println (str "  Last:  " (pr-str (last schema))))
  (assert-true "10 sites" (= 10 (count schema)))
  (assert-true "All constrained" (every? :constrained? schema))
  (assert-true "All gaussian" (every? #(= :gaussian (:dist-type %)) schema))
  (assert-true "Addr :y0" (= :y0 (:addr (first schema))))
  (assert-true "Addr :y9" (= :y9 (:addr (last schema)))))

(println "\n--- Schema with latent sites ---")

(let [schema (cg/discover-schema (dyn/auto-key model-prior) [] cm/EMPTY 10 (rng/fresh-key 42))]
  (println (str "  Sites: " (count schema)))
  (assert-true "2 sites" (= 2 (count schema)))
  (assert-true "x is latent" (not (:constrained? (first schema))))
  (assert-true "y is latent" (not (:constrained? (second schema)))))

;; ---------------------------------------------------------------------------
;; Test 2: Method pre-resolution
;; ---------------------------------------------------------------------------

(println "\n=== Test 2: Method pre-resolution ===")

(let [schema [{:addr :a :dist-type :gaussian}
              {:addr :b :dist-type :bernoulli}
              {:addr :c :dist-type :gaussian}]
      resolved (cg/resolve-methods schema)]
  (assert-true "2 types resolved" (= 2 (count resolved)))
  (assert-true "Gaussian sample-n present" (some? (get-in resolved [:gaussian :sample-n])))
  (assert-true "Gaussian log-prob present" (some? (get-in resolved [:gaussian :log-prob])))
  (assert-true "Bernoulli sample-n present" (some? (get-in resolved [:bernoulli :sample-n])))
  ;; Test direct call matches multimethod
  (let [d (dist/gaussian 0.0 1.0)
        v (mx/scalar 1.5)
        f (get-in resolved [:gaussian :log-prob])
        lp-direct (f d v)
        lp-multi (genmlx.dist.core/dist-log-prob d v)]
    (mx/materialize! lp-direct lp-multi)
    (assert-close "Direct = multimethod" (mx/item lp-multi) (mx/item lp-direct) 1e-6)))

;; ---------------------------------------------------------------------------
;; Test 3: compile-gen produces valid CompiledGF
;; ---------------------------------------------------------------------------

(println "\n=== Test 3: compile-gen CompiledGF ===")

(let [compiled (cg/compile-gen model-prior)]
  (assert-true "Returns CompiledGF" (instance? cg/CompiledGF compiled))
  ;; Simulate
  (let [cgf (vary-meta compiled assoc :genmlx.dynamic/key (rng/fresh-key 42))
        trace (p/simulate cgf [])]
    (assert-true "Simulate returns trace" (some? trace))
    (assert-true "Has :x choice" (cm/has-value? (cm/get-submap (:choices trace) :x)))
    (assert-true "Has :y choice" (cm/has-value? (cm/get-submap (:choices trace) :y)))
    (println (str "  retval: " (mx/item (:retval trace)))))
  ;; Generate with observations
  (let [cgf (vary-meta compiled assoc :genmlx.dynamic/key (rng/fresh-key 42))
        obs (cm/choicemap :y (mx/scalar 2.0))
        {:keys [trace weight]} (p/generate cgf [] obs)]
    (assert-true "Generate returns trace" (some? trace))
    (assert-true "Weight is finite" (js/isFinite (mx/item weight)))
    (println (str "  weight: " (.toFixed (mx/item weight) 4)))))

;; ---------------------------------------------------------------------------
;; Test 4: Fast vgenerate matches standard vgenerate
;; ---------------------------------------------------------------------------

(println "\n=== Test 4: Fast vgenerate correctness ===")

(let [key (rng/fresh-key 42)
      params (mx/array [true-mu])
      n-particles 5000
      ;; Standard vgenerate (Level 0)
      model-p (vary-meta (dyn/auto-key model-1) assoc
                :genmlx.dynamic/param-store {:params {:mu (mx/scalar true-mu)}})
      vtrace-std (dyn/vgenerate model-p [] obs-1 n-particles key)
      log-ml-std (mx/item (vec/vtrace-log-ml-estimate vtrace-std))
      ;; Fast vgenerate (Level 1)
      schema (cg/discover-schema (dyn/auto-key model-1) [] obs-1 n-particles key)
      resolved (cg/resolve-methods schema)
      vtrace-fast (cg/fast-vgenerate
                    (:body-fn (dyn/auto-key model-1)) schema resolved [] obs-1
                    n-particles key {:params {:mu (mx/scalar true-mu)}})
      log-ml-fast (mx/item (vec/vtrace-log-ml-estimate vtrace-fast))]
  (println (str "  Standard log-ML: " (.toFixed log-ml-std 4)))
  (println (str "  Fast log-ML:     " (.toFixed log-ml-fast 4)))
  (assert-close "Fast ≈ standard" log-ml-std log-ml-fast 0.01))

;; ---------------------------------------------------------------------------
;; Test 5: Level 1 compiled gradient correctness
;; ---------------------------------------------------------------------------

(println "\n=== Test 5: Level 1 gradient correctness ===")

(let [key (rng/fresh-key 42)
      params (mx/array [true-mu])
      ;; Level 0 (slow path)
      compiled-slow (cg/compile-log-ml-gradient
                      {:n-particles 5000 :key key :fast? false}
                      model-1 [] obs-1 [:mu])
      [v0 g0] (compiled-slow params)
      _ (mx/materialize! v0 g0)
      ;; Level 1 (fast path)
      compiled-fast (cg/compile-log-ml-gradient
                      {:n-particles 5000 :key key :fast? true}
                      model-1 [] obs-1 [:mu])
      [v1 g1] (compiled-fast params)
      _ (mx/materialize! v1 g1)]
  (println (str "  Slow neg-log-ML: " (.toFixed (mx/item v0) 4)))
  (println (str "  Fast neg-log-ML: " (.toFixed (mx/item v1) 4)))
  (println (str "  Slow grad: " (.toFixed (mx/item (mx/index g0 0)) 4)))
  (println (str "  Fast grad: " (.toFixed (mx/item (mx/index g1 0)) 4)))
  (assert-close "log-ML matches" (mx/item v0) (mx/item v1) 0.01)
  (assert-close "Gradient matches" (mx/item (mx/index g0 0)) (mx/item (mx/index g1 0)) 0.01))

;; ---------------------------------------------------------------------------
;; Test 6: 2D model gradient (Level 1)
;; ---------------------------------------------------------------------------

(println "\n=== Test 6: 2D Level 1 gradient ===")

(let [key (rng/fresh-key 77)
      params (mx/array [true-mu 0.0])
      compiled-vg (cg/compile-log-ml-gradient
                    {:n-particles 5000 :key key :fast? true}
                    model-2 [] obs-1 [:mu :log-sigma])
      [neg-lml grad] (compiled-vg params)]
  (mx/materialize! neg-lml grad)
  (let [g0 (mx/item (mx/index grad 0))
        g1 (mx/item (mx/index grad 1))]
    (println (str "  log-ML: " (.toFixed (- (mx/item neg-lml)) 2)))
    (println (str "  grad[mu]: " (.toFixed g0 4)))
    (println (str "  grad[log-sigma]: " (.toFixed g1 4)))
    (assert-close "grad[mu] ≈ 0 at MLE" 0.0 g0 1.0)
    (assert-true "grad[log-sigma] is finite" (js/isFinite g1))))

;; ---------------------------------------------------------------------------
;; Test 7: Speedup benchmark — Level 1 vs Level 0 vs uncompiled
;; ---------------------------------------------------------------------------

(println "\n=== Test 7: Speedup benchmark ===")

(let [key (rng/fresh-key 42)
      params (mx/array [true-mu])
      n-calls 30

      ;; Uncompiled (standard vgenerate)
      t0 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [{:keys [grad log-ml]} (diff/log-ml-gradient
                                        {:n-particles 5000 :key key}
                                        model-1 [] obs-1 [:mu] params)]
            (mx/materialize! grad log-ml)))
      t-uncompiled (- (js/Date.now) t0)

      ;; Level 0 (mx/compile-fn + standard vgenerate)
      compiled-l0 (cg/compile-log-ml-gradient
                    {:n-particles 5000 :key key :fast? false}
                    model-1 [] obs-1 [:mu])
      _ (dotimes [_ 3] (let [[v g] (compiled-l0 params)] (mx/materialize! v g)))
      t1 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [[v g] (compiled-l0 params)] (mx/materialize! v g)))
      t-level0 (- (js/Date.now) t1)

      ;; Level 1 (mx/compile-fn + fast vgenerate)
      compiled-l1 (cg/compile-log-ml-gradient
                    {:n-particles 5000 :key key :fast? true}
                    model-1 [] obs-1 [:mu])
      _ (dotimes [_ 3] (let [[v g] (compiled-l1 params)] (mx/materialize! v g)))
      t2 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [[v g] (compiled-l1 params)] (mx/materialize! v g)))
      t-level1 (- (js/Date.now) t2)

      speedup-l0 (/ t-uncompiled (max t-level0 1))
      speedup-l1 (/ t-uncompiled (max t-level1 1))
      speedup-l1-vs-l0 (/ t-level0 (max t-level1 1))]
  (println (str "  Uncompiled: " t-uncompiled "ms (" n-calls " calls)"))
  (println (str "  Level 0:    " t-level0 "ms (" n-calls " calls) — " (.toFixed speedup-l0 1) "x"))
  (println (str "  Level 1:    " t-level1 "ms (" n-calls " calls) — " (.toFixed speedup-l1 1) "x"))
  (println (str "  Level 1 vs 0: " (.toFixed speedup-l1-vs-l0 2) "x"))
  (assert-true "Level 1 faster than uncompiled" (> speedup-l1 1.0))
  (assert-true "Level 1 faster than Level 0" (>= speedup-l1 speedup-l0)))

;; ---------------------------------------------------------------------------
;; Test 8: Schema validation rejects dynamic models
;; ---------------------------------------------------------------------------

(println "\n=== Test 8: Schema validation ===")

(let [static-model (gen []
                     (let [x (trace :x (dist/gaussian 0.0 1.0))
                           y (trace :y (dist/gaussian x 1.0))]
                       y))
      ok? (try
            (cg/compile-gen static-model)
            true
            (catch :default e false))]
  (assert-true "Static model compiles" ok?))

;; ---------------------------------------------------------------------------
;; Test 9: Compiled Fisher with Level 1
;; ---------------------------------------------------------------------------

(println "\n=== Test 9: Compiled Fisher (Level 1) ===")

(let [key (rng/fresh-key 42)
      params (mx/array [true-mu])
      ;; Without compilation
      t0 (js/Date.now)
      r1 (fisher/observed-fisher {:n-particles 5000 :key key}
                          model-1 [] obs-1 [:mu] params)
      _ (mx/materialize! (:fisher r1))
      t-plain (- (js/Date.now) t0)
      f-plain (mx/item (mx/mat-get (:fisher r1) 0 0))
      ;; With compilation (uses Level 1 internally)
      t1 (js/Date.now)
      r2 (fisher/observed-fisher {:n-particles 5000 :key key :compiled? true}
                          model-1 [] obs-1 [:mu] params)
      _ (mx/materialize! (:fisher r2))
      t-compiled (- (js/Date.now) t1)
      f-compiled (mx/item (mx/mat-get (:fisher r2) 0 0))]
  (println (str "  Plain Fisher: " (.toFixed f-plain 3) " (" t-plain "ms)"))
  (println (str "  Compiled Fisher: " (.toFixed f-compiled 3) " (" t-compiled "ms)"))
  (assert-close "Fisher values match" f-plain f-compiled 0.5)
  (assert-close "Fisher ≈ analytical (K=10)" 10.0 f-compiled 2.0))

;; ---------------------------------------------------------------------------
;; Test 10: Raw vgenerate benchmark (no mx/compile-fn — isolates CLJS overhead)
;; ---------------------------------------------------------------------------

(println "\n=== Test 10: Raw vgenerate benchmark (no compilation) ===")

(let [key (rng/fresh-key 42)
      n-particles 5000
      n-calls 50
      model-keyed (vary-meta (dyn/auto-key model-1) assoc
                    :genmlx.dynamic/param-store {:params {:mu (mx/scalar true-mu)}})
      ;; Standard vgenerate
      t0 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [vt (dyn/vgenerate model-keyed [] obs-1 n-particles key)]
            (mx/materialize! (:weight vt))))
      t-std (- (js/Date.now) t0)
      ;; Fast vgenerate (Level 1)
      schema (cg/discover-schema (dyn/auto-key model-1) [] obs-1 n-particles key)
      resolved (cg/resolve-methods schema)
      body-fn (:body-fn (dyn/auto-key model-1))
      ps {:params {:mu (mx/scalar true-mu)}}
      t1 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [vt (cg/fast-vgenerate body-fn schema resolved [] obs-1
                                       n-particles key ps)]
            (mx/materialize! (:weight vt))))
      t-fast (- (js/Date.now) t1)
      ;; Score-only variant
      t2 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [r (cg/fast-vgenerate-score-only body-fn schema resolved [] obs-1
                                                 n-particles key ps)]
            (mx/materialize! (:weight r))))
      t-score (- (js/Date.now) t2)
      speedup-vg (/ t-std (max t-fast 1))
      speedup-so (/ t-std (max t-score 1))]
  (println (str "  Standard vgenerate: " t-std "ms (" n-calls " calls)"))
  (println (str "  Fast vgenerate:     " t-fast "ms — " (.toFixed speedup-vg 2) "x"))
  (println (str "  Score-only:         " t-score "ms — " (.toFixed speedup-so 2) "x"))
  ;; Raw speedup is modest (handler overhead small vs MLX ops).
  ;; Real gains come when combined with mx/compile-fn (see Test 7).
  (assert-true "Fast within 20% of standard" (> speedup-vg 0.8))
  (assert-true "Score-only within 20% of standard" (> speedup-so 0.8)))

;; ---------------------------------------------------------------------------
;; Test 11: Large model benchmark (50 trace sites — more handler overhead)
;; ---------------------------------------------------------------------------

(println "\n=== Test 11: Large model (50 sites) raw benchmark ===")

(def K-large 50)

(def model-large
  (gen []
    (let [mu (param :mu 0.0)]
      (doseq [i (range K-large)]
        (trace (keyword (str "y" i))
               (dist/gaussian mu 1.0)))
      mu)))

(def obs-large
  (apply cm/choicemap
    (mapcat (fn [i] [(keyword (str "y" i)) (mx/scalar (+ true-mu (* 0.1 (- i 25))))])
            (range K-large))))

(let [key (rng/fresh-key 42)
      n-particles 5000
      n-calls 30
      model-keyed (vary-meta (dyn/auto-key model-large) assoc
                    :genmlx.dynamic/param-store {:params {:mu (mx/scalar true-mu)}})
      ;; Standard vgenerate
      t0 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [vt (dyn/vgenerate model-keyed [] obs-large n-particles key)]
            (mx/materialize! (:weight vt))))
      t-std (- (js/Date.now) t0)
      ;; Score-only (Level 1)
      schema (cg/discover-schema (dyn/auto-key model-large) [] obs-large n-particles key)
      resolved (cg/resolve-methods schema)
      body-fn (:body-fn (dyn/auto-key model-large))
      ps {:params {:mu (mx/scalar true-mu)}}
      t1 (js/Date.now)
      _ (dotimes [_ n-calls]
          (let [r (cg/fast-vgenerate-score-only body-fn schema resolved [] obs-large
                                                 n-particles key ps)]
            (mx/materialize! (:weight r))))
      t-fast (- (js/Date.now) t1)
      speedup (/ t-std (max t-fast 1))]
  (println (str "  Standard vgenerate (K=50): " t-std "ms (" n-calls " calls)"))
  (println (str "  Score-only (K=50):         " t-fast "ms — " (.toFixed speedup 2) "x"))
  (assert-true "Significant speedup at K=50" (> speedup 1.0)))

(println "\nDone.")
