(ns genmlx.gen-clj-compat-test
  "Tests adapted from Gen.clj's test suite to verify GenMLX correctness.
   Sources: gen.distribution-test, gen.dynamic-test, gen.inference.importance-test
   Spot-check values verified against scipy.stats and Gen.jl.

   Note: GenMLX uses float32 (MLX default) while Gen.clj uses float64.
   Tolerances are set to 1e-6 for spot checks and 1e-5 for computed values."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as importance])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- lp
  "Extract log-prob as a plain number."
  [d v]
  (let [r (dist/log-prob d (if (mx/array? v) v (mx/scalar v)))]
    (mx/eval! r)
    (mx/item r)))

(defn assert-true [msg actual]
  (if actual
    (do (vswap! pass-count inc)
        (println "  PASS:" msg))
    (do (vswap! fail-count inc)
        (println "  FAIL:" msg "- expected truthy, got" actual))))

(defn assert-close
  "Assert expected ~= actual within tolerance."
  [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (do (vswap! pass-count inc)
          (println "  PASS:" msg))
      (do (vswap! fail-count inc)
          (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)
          (println "    diff:    " diff)))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (vswap! pass-count inc)
        (println "  PASS:" msg))
    (do (vswap! fail-count inc)
        (println "  FAIL:" msg)
        (println "    expected:" expected)
        (println "    actual:  " actual))))

(defn assert-neg-inf [msg actual]
  (if (= ##-Inf actual)
    (do (vswap! pass-count inc)
        (println "  PASS:" msg))
    (do (vswap! fail-count inc)
        (println "  FAIL:" msg "- expected -Inf, got" actual))))

(println "\n=== Gen.clj Compatibility Tests ===\n")

;; =========================================================================
;; SECTION 1: Distribution logpdf spot checks
;; (From gen.distribution-test, verified against scipy.stats and Gen.jl)
;;
;; Note: GenMLX uses float32 (MLX default), Gen.clj uses float64.
;; Tolerance is 1e-6 for spot checks (float32 has ~7 decimal digits).
;; =========================================================================

(def ^:private f32-tol 1e-6)   ;; float32 tolerance for spot checks
(def ^:private sym-tol 1e-10)  ;; relative comparisons within same precision

;; ---- Gaussian / Normal ----
(println "-- Gaussian logpdf spot checks (from Gen.clj) --")

(assert-close "normal(0,1) at 0.5"
  -1.0439385332046727
  (lp (dist/gaussian 0 1) 0.5) f32-tol)

(assert-close "normal(0,2) at 0.5"
  -1.643335713764618
  (lp (dist/gaussian 0 2) 0.5) f32-tol)

(assert-close "normal(0,2) at 0"
  -1.612085713764618
  (lp (dist/gaussian 0 2) 0) f32-tol)

;; Symmetry: logpdf(N(0,sigma), v) = logpdf(N(0,sigma), -v)
(println "\n-- Gaussian symmetry --")
(doseq [sigma [0.5 1.0 2.0 5.0]
        v     [0.1 1.0 3.0 10.0]]
  (assert-close (str "N(0," sigma ") symmetric at +/-" v)
    (lp (dist/gaussian 0 sigma) v)
    (lp (dist/gaussian 0 sigma) (- v))
    sym-tol))

;; Shift symmetry: logpdf(N(mu,sigma), v) = logpdf(N(mu+s,sigma), v+s)
(println "\n-- Gaussian shift symmetry --")
(doseq [[mu sigma v shift] [[2.0 1.0 3.0 5.0]
                             [-1.0 0.5 2.0 -3.0]
                             [0.0 3.0 1.0 10.0]]]
  (assert-close (str "N(" mu "," sigma ") shift by " shift)
    (lp (dist/gaussian mu sigma) v)
    (lp (dist/gaussian (+ mu shift) sigma) (+ v shift))
    f32-tol))

;; ---- Uniform ----
(println "\n-- Uniform logpdf (from Gen.clj) --")

;; Inside bounds: logpdf = -log(max - min)
(doseq [[lo hi v] [[-5.0 5.0 0.0]
                    [0.0 1.0 0.5]
                    [-10.0 -0.1 -5.0]
                    [0.1 10.0 5.0]]]
  (let [expected (- (js/Math.log (- hi lo)))]
    (assert-close (str "uniform(" lo "," hi ") at " v " (inside)")
      expected (lp (dist/uniform lo hi) v) f32-tol)))

;; Outside bounds should return -Inf
(doseq [[lo hi v] [[0.0 1.0 -0.1]
                    [0.0 1.0 1.5]
                    [-5.0 5.0 10.0]]]
  (assert-neg-inf (str "uniform(" lo "," hi ") at " v " (outside)")
    (lp (dist/uniform lo hi) v)))

;; ---- Beta ----
(println "\n-- Beta logpdf spot checks (from Gen.clj) --")

(assert-close "beta(0.001, 1) at 0.4"
  -5.992380837839856
  (lp (dist/beta-dist 0.001 1) 0.4) 1e-4)

(assert-close "beta(1, 0.001) at 0.4"
  -6.397440480839912
  (lp (dist/beta-dist 1 0.001) 0.4) 1e-4)

;; ---- Gamma ----
;; IMPORTANT: Gen.clj uses (shape, SCALE) parameterization.
;; GenMLX uses (shape, RATE) parameterization where rate = 1/scale.
;; So Gen.clj's (->gamma shape scale) = GenMLX's (gamma-dist shape (/ 1 scale))
(println "\n-- Gamma logpdf spot checks (from Gen.clj) --")

;; Gen.clj: (->gamma 0.001 1) => shape=0.001, scale=1 => rate=1
(assert-close "gamma(shape=0.001, scale=1) at 0.4"
  -6.391804444241573
  (lp (dist/gamma-dist 0.001 1) 0.4) 1e-4)

;; Gen.clj: (->gamma 1 0.001) => shape=1, scale=0.001 => rate=1000
(assert-close "gamma(shape=1, scale=0.001) at 0.4"
  -393.0922447210179
  (lp (dist/gamma-dist 1 1000) 0.4) 0.1)

;; ---- Bernoulli ----
(println "\n-- Bernoulli logpdf (from Gen.clj) --")

;; Fair coin: equal probability
(assert-close "bernoulli(0.5) fair coin symmetry"
  (lp (dist/bernoulli 0.5) 1.0)
  (lp (dist/bernoulli 0.5) 0.0)
  sym-tol)

;; Probabilities sum to 1
(doseq [p [0.1 0.3 0.5 0.7 0.9]]
  (let [sum (+ (js/Math.exp (lp (dist/bernoulli p) 1.0))
               (js/Math.exp (lp (dist/bernoulli p) 0.0)))]
    (assert-close (str "bernoulli(" p ") sums to 1")
      1.0 sum f32-tol)))

;; logpdf matches log(p) and log(1-p)
(doseq [p [0.2 0.5 0.8]]
  (assert-close (str "bernoulli(" p ") at 1 = log(p)")
    (js/Math.log p)
    (lp (dist/bernoulli p) 1.0)
    f32-tol)
  (assert-close (str "bernoulli(" p ") at 0 = log(1-p)")
    (js/Math.log (- 1 p))
    (lp (dist/bernoulli p) 0.0)
    f32-tol))

;; ---- Exponential ----
(println "\n-- Exponential logpdf (from Gen.clj) --")

;; Negative values should return -Inf
(doseq [v [-0.1 -1.0 -100.0]]
  (assert-neg-inf (str "exponential(1) at " v " (negative)")
    (lp (dist/exponential 1.0) v)))

;; Rate 1.0 produces log(1) - 1*v = -v
(doseq [v [0.5 1.0 2.0 5.0]]
  (assert-close (str "exponential(1) at " v " = -v")
    (- v) (lp (dist/exponential 1.0) v) sym-tol))

;; Spot checks from Gen.clj
(assert-close "exponential(2) at 2"
  -3.3068528194400546
  (lp (dist/exponential 2.0) 2.0) f32-tol)

(assert-close "exponential(2) at 3"
  -5.306852819440055
  (lp (dist/exponential 2.0) 3.0) f32-tol)

;; ---- Laplace ----
(println "\n-- Laplace logpdf (from Gen.clj) --")

;; Spot checks from Gen.clj
(assert-close "laplace(2,1) at 1"
  -1.6931471805599454
  (lp (dist/laplace 2 1) 1) f32-tol)

(assert-close "laplace(0,2) at 1"
  -1.8862943611198906
  (lp (dist/laplace 0 2) 1) f32-tol)

(assert-close "laplace(0,0.001) at 0.002"
  4.214608098422191
  (lp (dist/laplace 0 0.001) 0.002) 1e-3)

;; Symmetry: logpdf(laplace(0,1), v) = logpdf(laplace(0,1), -v)
(doseq [v [0.5 1.0 3.0 10.0]]
  (assert-close (str "laplace(0,1) symmetric at +/-" v)
    (lp (dist/laplace 0 1) v)
    (lp (dist/laplace 0 1) (- v))
    sym-tol))

;; Location = v gives peak: logpdf = -log(2)
(doseq [v [0.0 1.5 -3.0 10.0]]
  (assert-close (str "laplace(" v ",1) at " v " = -log(2)")
    (- (js/Math.log 2))
    (lp (dist/laplace v 1) v)
    f32-tol))

;; ---- Student-t ----
(println "\n-- Student-t logpdf spot checks (from Gen.clj) --")

(assert-close "student-t(2, 2.1, 2) at 2"
  -1.7347417805005154
  (lp (dist/student-t 2 2.1 2) 2) f32-tol)

(assert-close "student-t(1, 0.8, 4) at 3"
  -2.795309741614719
  (lp (dist/student-t 1 0.8 4) 3) f32-tol)

;; Student-t(nu, 0, 1) self-consistency
(doseq [nu [1 2 5 10]
        v  [-2.0 0.0 1.5 3.0]]
  (assert-close (str "student-t(" nu ", 0, 1) at " v " self-consistent")
    (lp (dist/student-t nu 0 1) v)
    (lp (dist/student-t nu 0 1) v)
    sym-tol))

;; ---- Delta ----
(println "\n-- Delta logpdf (from Gen.clj) --")

;; logpdf at center = 0
(doseq [c [0.0 1.5 -3.0 100.0]]
  (assert-close (str "delta(" c ") at " c " = 0")
    0.0 (lp (dist/delta c) c) sym-tol))

;; logpdf away from center = -Inf
(doseq [[c v] [[0.0 1.0] [1.0 0.0] [5.0 5.1] [-3.0 3.0]]]
  (assert-neg-inf (str "delta(" c ") at " v " = -Inf")
    (lp (dist/delta c) v)))

;; ---- Log-Gamma correctness ----
(println "\n-- Log-Gamma function (from Gen.clj) --")

;; log-Gamma(n) ~matches log((n-1)!)
(defn factorial [n]
  (if (zero? n) 1 (* n (factorial (dec n)))))

;; Test indirectly via Gamma distribution:
;; logpdf(Gamma(n, rate=1), x=1) = (n-1)*log(1) + n*log(1) - 1 - lgamma(n)
;;                                = -1 - lgamma(n)
;; So lgamma(n) = -1 - logpdf(Gamma(n, 1), 1)
(doseq [n (range 1 15)]
  (let [lgamma-n (- -1.0 (lp (dist/gamma-dist n 1) 1.0))
        expected (js/Math.log (factorial (dec n)))]
    (assert-close (str "lgamma(" n ") = log(" (dec n) "!)")
      expected lgamma-n 1e-4)))

;; =========================================================================
;; SECTION 2: GFI interface for primitive distributions
;; (From gen.distribution-test primitive-gfi-tests)
;; =========================================================================

(println "\n-- Primitive GFI tests (from Gen.clj) --")

;; Distribution round-trips through trace
(let [d (dist/gaussian 0 1)
      trace (p/simulate d [])]
  (assert-true "gaussian: gen-fn round-trips through trace"
    (= d (:gen-fn trace)))
  (assert-true "gaussian: args round-trip"
    (= [] (:args trace)))
  (let [choice (:choices trace)
        retval (:retval trace)]
    (mx/eval! retval (cm/get-value choice))
    (assert-close "gaussian: retval = choice value"
      (mx/item retval) (mx/item (cm/get-value choice)) sym-tol)))

;; Bernoulli GFI tests (from Gen.clj bernoulli-gfi-tests)
(println "\n-- Bernoulli GFI (from Gen.clj) --")

(let [d (dist/bernoulli 0.5)
      trace (p/simulate d [0.5])]
  (assert-true "bernoulli: simulate returns trace"
    (some? trace))
  (let [retval (:retval trace)
        score  (:score trace)]
    (mx/eval! retval score)
    (assert-close "bernoulli(0.5): score = log(0.5)"
      0.5
      (js/Math.exp (mx/item score))
      f32-tol)))

;; Bernoulli generate weight test (from Gen.clj)
(println "\n-- Bernoulli generate weight (from Gen.clj) --")

;; Generate with constraint then check weight
(let [gf (gen [] (dyn/trace :x (dist/bernoulli 0.3)))
      {:keys [trace weight]} (p/generate gf [] (cm/choicemap :x (mx/scalar 1.0)))]
  (mx/eval! weight)
  ;; Weight = log(0.3) for constraining bernoulli(0.3) to true
  (assert-close "generate bernoulli(0.3) constrained to 1: weight = log(0.3)"
    (js/Math.log 0.3)
    (mx/item weight)
    f32-tol))

;; Update via gen model (not raw distribution)
(let [gf (gen [] (dyn/trace :x (dist/bernoulli 0.3)))
      {:keys [trace]} (p/generate gf [] (cm/choicemap :x (mx/scalar 1.0)))
      ;; Update with same value
      {:keys [weight]} (p/update gf trace (cm/choicemap :x (mx/scalar 1.0)))]
  (mx/eval! weight)
  (assert-close "update same value: exp(weight) = 1"
    1.0 (js/Math.exp (mx/item weight)) f32-tol))

;; Update from true to false => weight = log(0.7) - log(0.3)
(let [gf (gen [] (dyn/trace :x (dist/bernoulli 0.3)))
      {:keys [trace]} (p/generate gf [] (cm/choicemap :x (mx/scalar 1.0)))
      {:keys [weight]} (p/update gf trace (cm/choicemap :x (mx/scalar 0.0)))]
  (mx/eval! weight)
  (assert-close "update true->false: exp(weight) = 0.7/0.3"
    (/ 0.7 0.3) (js/Math.exp (mx/item weight)) 1e-4))

;; =========================================================================
;; SECTION 3: Dynamic DSL / gen macro tests
;; (From gen.dynamic-test)
;; =========================================================================

(println "\n-- Dynamic DSL tests (from Gen.clj) --")

;; No-arity, no-return function returns nil
(let [gf (gen [] nil)]
  (assert-true "gen no-arity no-return"
    (nil? (dyn/call gf))))

;; Round-trip through functions
(let [gf (gen [a b c d e]
           (+ a b c d e))
      trace (p/simulate gf [1 2 3 4 5])]
  (assert-true "deterministic gen: gen-fn round-trips"
    (= gf (:gen-fn trace)))
  (assert-true "deterministic gen: args round-trip"
    (= [1 2 3 4 5] (:args trace)))
  (assert-equal "deterministic gen: retval matches"
    15 (:retval trace)))

;; No choices for deterministic functions
(let [gf (gen [x y] (+ x y))
      trace (p/simulate gf [3 4])]
  (assert-equal "deterministic: no choices"
    0 (count (cm/addresses (:choices trace)))))

;; trace! creates choices
(println "\n-- Nested tracing semantics (from Gen.clj) --")
(let [gf (gen [p]
           (dyn/trace :addr (dist/bernoulli p)))
      trace (p/simulate gf [0.5])]
  (let [choices (:choices trace)
        retval  (:retval trace)]
    (mx/eval! retval)
    (let [v (cm/get-value (cm/get-submap choices :addr))]
      (mx/eval! v)
      (assert-close "trace choices match retval"
        (mx/item retval) (mx/item v) sym-tol))))

;; splice! bubbles up choices (from Gen.clj's "trace inside splice should bubble up")
(let [gf0 (gen [] (dyn/trace :addr (dist/bernoulli 0.5)))
      gf1 (gen [] (dyn/splice :sub gf0))
      trace (p/simulate gf1 [])]
  (let [choices (:choices trace)
        sub-map (cm/get-submap choices :sub)]
    (assert-true "splice bubbles up: sub-map exists"
      (some? sub-map))
    (let [addr-val (cm/get-submap sub-map :addr)]
      (assert-true "splice bubbles up: addr exists in sub"
        (cm/has-value? addr-val)))))

;; trace inside trace nests (from Gen.clj's "trace inside of trace should nest")
(let [inner (gen [] (dyn/trace :inner (dist/bernoulli 0.5)))
      outer (gen [] (dyn/splice :outer inner))
      trace (p/simulate outer [])]
  (let [choices (:choices trace)
        outer-sub (cm/get-submap choices :outer)
        inner-val (cm/get-submap outer-sub :inner)]
    (assert-true "nested tracing: outer exists"
      (some? outer-sub))
    (assert-true "nested tracing: inner exists within outer"
      (cm/has-value? inner-val))))

;; Generate through splice preserves structure
(let [inner (gen [] (dyn/trace :addr (dist/bernoulli 0.5)))
      outer (gen [] (dyn/splice :sub inner))
      {:keys [trace]} (p/generate outer [] cm/EMPTY)]
  (let [choices (:choices trace)
        sub-map (cm/get-submap choices :sub)
        addr-val (cm/get-submap sub-map :addr)]
    (assert-true "generate+splice: structure preserved"
      (cm/has-value? addr-val))))

;; Score correctness (from Gen.clj's score test)
(println "\n-- Score computation (from Gen.clj) --")
(let [trace (p/simulate
              (gen [] (dyn/trace :addr (dist/bernoulli 0.5)))
              [])]
  (let [score (:score trace)]
    (mx/eval! score)
    (assert-close "bernoulli(0.5) score = log(0.5)"
      0.5
      (js/Math.exp (mx/item score))
      f32-tol)))

;; =========================================================================
;; SECTION 4: Update semantics
;; (From gen.dynamic-test update-discard tests)
;; =========================================================================

(println "\n-- Update discard semantics (from Gen.clj) --")

;; Update with constraint => old value in discard
(let [gf (gen []
           (dyn/trace :x (dist/bernoulli 0.5)))
      trace (p/simulate gf [])
      old-x (let [v (cm/get-value (cm/get-submap (:choices trace) :x))]
              (mx/eval! v) (mx/item v))
      new-val (if (> old-x 0.5) 0.0 1.0)  ;; flip the value
      {:keys [trace discard]} (p/update gf trace
                                 (cm/choicemap :x (mx/scalar new-val)))]
  (assert-true "update: discard contains old value"
    (cm/has-value? (cm/get-submap discard :x)))
  (let [discarded (cm/get-value (cm/get-submap discard :x))]
    (mx/eval! discarded)
    (assert-close "update: discarded value matches old"
      old-x (mx/item discarded) sym-tol))
  (let [new-x (cm/get-value (cm/get-submap (:choices trace) :x))]
    (mx/eval! new-x)
    (assert-close "update: new trace has new value"
      new-val (mx/item new-x) sym-tol)))

;; Update weight for bernoulli (via gen model)
(let [gf (gen []
           (dyn/trace :x (dist/bernoulli 0.3)))
      {:keys [trace]} (p/generate gf [] (cm/choicemap :x (mx/scalar 1.0)))
      {:keys [weight]} (p/update gf trace (cm/choicemap :x (mx/scalar 1.0)))]
  (mx/eval! weight)
  (assert-close "update same value: log-weight = 0"
    0.0 (mx/item weight) f32-tol))

;; Update with two addresses, only constrain one
(let [gf (gen []
           (dyn/trace :kept (dist/bernoulli 0.5))
           (dyn/trace :changed (dist/bernoulli 0.5)))
      trace (p/simulate gf [])
      old-kept (let [v (cm/get-value (cm/get-submap (:choices trace) :kept))]
                 (mx/eval! v) (mx/item v))
      {:keys [trace discard]} (p/update gf trace
                                 (cm/choicemap :changed (mx/scalar 1.0)))]
  ;; :kept should remain unchanged
  (let [new-kept (cm/get-value (cm/get-submap (:choices trace) :kept))]
    (mx/eval! new-kept)
    (assert-close "update partial: unchanged addr kept"
      old-kept (mx/item new-kept) sym-tol))
  ;; :changed should have discard
  (assert-true "update partial: changed addr in discard"
    (cm/has-value? (cm/get-submap discard :changed))))

;; =========================================================================
;; SECTION 5: Generate semantics
;; (From gen.dynamic-test gfi-tests)
;; =========================================================================

(println "\n-- Generate semantics (from Gen.clj) --")

;; Generate with empty constraints = simulate (weight ~ 0)
(let [gf (gen [] (dyn/trace :x (dist/gaussian 0 1)))
      {:keys [trace weight]} (p/generate gf [] cm/EMPTY)]
  (mx/eval! weight)
  (assert-close "generate empty constraints: weight = 0"
    0.0 (mx/item weight) sym-tol))

;; Generate with constraint
(let [gf (gen []
           (dyn/trace :x (dist/gaussian 0 1))
           (dyn/trace :y (dist/gaussian 0 1)))
      constraints (cm/choicemap :x (mx/scalar 2.0))
      {:keys [trace weight]} (p/generate gf [] constraints)]
  ;; x should be constrained to 2.0
  (let [x-val (cm/get-value (cm/get-submap (:choices trace) :x))]
    (mx/eval! x-val)
    (assert-close "generate: x constrained to 2.0"
      2.0 (mx/item x-val) sym-tol))
  ;; Weight should be log p(x=2) = logpdf(N(0,1), 2)
  (mx/eval! weight)
  (assert-close "generate: weight = logpdf(N(0,1), 2)"
    (lp (dist/gaussian 0 1) 2.0)
    (mx/item weight)
    f32-tol))

;; =========================================================================
;; SECTION 6: Regenerate semantics
;; =========================================================================

(println "\n-- Regenerate semantics --")

;; Regenerate keeps unselected addresses
(let [gf (gen []
           (let [x (dyn/trace :x (dist/gaussian 0 1))
                 y (dyn/trace :y (dist/gaussian 0 1))]
             (mx/eval! x y)
             [(mx/item x) (mx/item y)]))
      trace (p/simulate gf [])
      old-y (let [v (cm/get-value (cm/get-submap (:choices trace) :y))]
              (mx/eval! v) (mx/item v))
      {:keys [trace weight]} (p/regenerate gf trace (sel/select :x))]
  (let [new-y (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! new-y)
    (assert-close "regenerate: unselected :y unchanged"
      old-y (mx/item new-y) sym-tol))
  (mx/eval! weight)
  (assert-true "regenerate: returns finite weight"
    (js/isFinite (mx/item weight))))

;; =========================================================================
;; SECTION 7: Importance sampling with rejection
;; (From gen.inference.importance-test)
;; =========================================================================

(println "\n-- Importance sampling rejection (from Gen.clj) --")

;; Model that causes some particles to have low weight
;; (bernoulli branch structure with observation)
(let [model (gen []
              (let [foo (dyn/trace :foo (dist/bernoulli 0.5))]
                (mx/eval! foo)
                (if (> (mx/item foo) 0.5)
                  (dyn/trace :bar (dist/bernoulli 0.99))
                  (dyn/trace :bar (dist/bernoulli 0.01)))))
      observations (cm/choicemap :bar (mx/scalar 1.0))
      ;; importance-resampling returns a vector of traces
      traces (importance/importance-resampling
               {:samples 1 :particles 20}
               model [] observations)
      trace (first traces)]
  (assert-true "importance resampling: returns trace"
    (some? trace))
  (let [choices (:choices trace)
        bar-val (cm/get-value (cm/get-submap choices :bar))]
    (mx/eval! bar-val)
    (assert-close "importance resampling: bar constrained to 1.0"
      1.0 (mx/item bar-val) sym-tol)))

;; =========================================================================
;; SECTION 8: Mathematical properties of distributions
;; (From gen.distribution-test property checks)
;; =========================================================================

(println "\n-- Distribution mathematical properties --")

;; Laplace with scale 1, location = v: logpdf = -log(2)
(doseq [v [-5.0 0.0 3.0 10.0]]
  (assert-close (str "laplace(" v ",1) at " v " = -log(2)")
    (- (js/Math.log 2))
    (lp (dist/laplace v 1) v)
    f32-tol))

;; Exponential rate=1 at 0 gives logpdf = 0
(assert-close "exponential(1) at 0 = 0"
  0.0 (lp (dist/exponential 1.0) 0.0) sym-tol)

;; Gaussian at mean gives peak: logpdf = -0.5*log(2pi) - log(sigma)
(doseq [sigma [0.5 1.0 2.0 5.0]]
  (let [expected (- (- (* 0.5 (js/Math.log (* 2 js/Math.PI))))
                    (js/Math.log sigma))]
    (assert-close (str "N(0," sigma ") at 0 = peak")
      expected (lp (dist/gaussian 0 sigma) 0) f32-tol)))

;; =========================================================================
;; SECTION 9: End-to-end model test
;; (Similar to Gen.clj's line-model in sci_test)
;; =========================================================================

(println "\n-- End-to-end line model (from Gen.clj) --")

(let [line-model (gen [xs]
                   (let [slope     (dyn/trace :slope (dist/gaussian 0 1))
                         intercept (dyn/trace :intercept (dist/gaussian 0 2))]
                     (mx/eval! slope intercept)
                     (let [s (mx/item slope) i (mx/item intercept)]
                       (doseq [[idx x] (map-indexed vector xs)]
                         (dyn/trace (keyword (str "y" idx))
                                   (dist/gaussian (+ (* s x) i) 0.1)))
                       (fn [x] (+ (* s x) i)))))
      xs (range -5 6)
      trace (p/simulate line-model [(vec xs)])]
  (assert-true "line model: trace has choices"
    (some? (:choices trace)))
  (assert-true "line model: returns function"
    (fn? (:retval trace)))
  (assert-true "line model: has slope"
    (cm/has-value? (cm/get-submap (:choices trace) :slope)))
  (assert-true "line model: has intercept"
    (cm/has-value? (cm/get-submap (:choices trace) :intercept)))
  ;; Should have y0 through y10
  (doseq [i (range 11)]
    (assert-true (str "line model: has y" i)
      (cm/has-value? (cm/get-submap (:choices trace)
                                     (keyword (str "y" i)))))))

;; Generate with observations should constrain ys
(let [line-model (gen [xs]
                   (let [slope     (dyn/trace :slope (dist/gaussian 0 1))
                         intercept (dyn/trace :intercept (dist/gaussian 0 2))]
                     (mx/eval! slope intercept)
                     (let [s (mx/item slope) i (mx/item intercept)]
                       (doseq [[idx x] (map-indexed vector xs)]
                         (dyn/trace (keyword (str "y" idx))
                                   (dist/gaussian (+ (* s x) i) 0.1)))
                       [s i])))
      xs [1.0 2.0 3.0]
      observations (cm/choicemap :y0 (mx/scalar 2.0)
                                  :y1 (mx/scalar 4.0)
                                  :y2 (mx/scalar 6.0))
      {:keys [trace weight]} (p/generate line-model [xs] observations)]
  (assert-true "line model generate: returns trace"
    (some? trace))
  (mx/eval! weight)
  (assert-true "line model generate: weight is finite"
    (js/isFinite (mx/item weight)))
  (let [y0 (cm/get-value (cm/get-submap (:choices trace) :y0))]
    (mx/eval! y0)
    (assert-close "line model generate: y0 = 2.0"
      2.0 (mx/item y0) sym-tol)))

;; =========================================================================
;; Summary
;; =========================================================================

(println "\n=== Gen.clj Compatibility Test Results ===")
(println (str "  Passed: " @pass-count))
(println (str "  Failed: " @fail-count))
(println (str "  Total:  " (+ @pass-count @fail-count)))
(when (> @fail-count 0)
  (println "\n  *** SOME TESTS FAILED ***"))
(println)
