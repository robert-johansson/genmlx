(ns genmlx.conjugacy-test
  "Tests for Level 3 conjugacy detection (WP-0).
   Gate 0: Address-based dispatch prototype (marginal LL matches conjugate.cljs)
   Gate 1: 15 test models accuracy (5 conjugate, 5 non-conjugate, 5 edge cases)"
  (:require [genmlx.conjugacy :as conj]
            [genmlx.schema :as schema]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.conjugate :as conjugate]
            [genmlx.inference.analytical :as ana]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- assert-true [desc pred]
  (if pred
    (do (vswap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (vswap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " desc " — expected " expected ", got " actual)))))

(defn- assert-close [desc expected actual tol]
  (let [e (if (number? expected) expected (mx/item expected))
        a (if (number? actual) actual (mx/item actual))
        diff (js/Math.abs (- e a))]
    (if (<= diff tol)
      (do (vswap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toExponential diff 2) ")")))
      (do (vswap! fail-count inc)
          (println (str "  FAIL: " desc " — expected " e ", got " a " (diff=" diff ")"))))))

;; ---------------------------------------------------------------------------
;; Section 1: Conjugacy Table Lookups
;; ---------------------------------------------------------------------------

(println "\n=== Section 1: Conjugacy Table Lookups ===")

(let [t conj/conjugacy-table]
  ;; 5 known conjugate families
  (assert-equal "NN family" :normal-normal (:family (get t [:gaussian :gaussian])))
  (assert-equal "BB family" :beta-bernoulli (:family (get t [:beta-dist :bernoulli])))
  (assert-equal "GP family" :gamma-poisson (:family (get t [:gamma-dist :poisson])))
  (assert-equal "GE family" :gamma-exponential (:family (get t [:gamma-dist :exponential])))
  (assert-equal "DC family" :dirichlet-categorical (:family (get t [:dirichlet :categorical])))

  ;; Param keys correct
  (assert-equal "NN prior-mean-key" :mu (:prior-mean-key (get t [:gaussian :gaussian])))
  (assert-equal "NN obs-noise-key" :sigma (:obs-noise-key (get t [:gaussian :gaussian])))
  (assert-equal "BB prior-alpha-key" :alpha (:prior-alpha-key (get t [:beta-dist :bernoulli])))
  (assert-equal "GP prior-shape-key" :shape-param (:prior-shape-key (get t [:gamma-dist :poisson])))

  ;; Natural param index
  (assert-equal "NN natural-param-idx" 0 (:natural-param-idx (get t [:gaussian :gaussian])))
  (assert-equal "BB natural-param-idx" 0 (:natural-param-idx (get t [:beta-dist :bernoulli])))

  ;; Explicitly not conjugate (nil entries)
  (assert-true "gaussian→bernoulli is nil" (nil? (get t [:gaussian :bernoulli])))
  (assert-true "beta→gaussian is nil" (nil? (get t [:beta-dist :gaussian])))
  (assert-true "gaussian→poisson is nil" (nil? (get t [:gaussian :poisson])))

  ;; Not in table at all
  (assert-true "uniform→gaussian not in table" (not (contains? t [:uniform :gaussian])))
  (assert-true "gaussian→uniform not in table" (not (contains? t [:gaussian :uniform]))))

;; ---------------------------------------------------------------------------
;; Section 2: Gate 1 — 15 Test Models
;; ---------------------------------------------------------------------------

(println "\n=== Section 2: Gate 1 — 5 Conjugate Models ===")

;; -- 5 Conjugate Models --

;; C1: Normal-Normal
(let [m (gen [sigma]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu sigma))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "C1-NN: 1 pair detected" 1 (count pairs))
  (assert-equal "C1-NN: family" :normal-normal (:family (first pairs)))
  (assert-equal "C1-NN: prior-addr" :mu (:prior-addr (first pairs)))
  (assert-equal "C1-NN: obs-addr" :y (:obs-addr (first pairs)))
  (assert-equal "C1-NN: direct dependency" :direct (get-in (first pairs) [:dependency-type :type])))

;; C2: Beta-Bernoulli
(let [m (gen []
          (let [p (trace :p (dist/beta-dist 2 5))]
            (trace :x (dist/bernoulli p))
            p))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "C2-BB: 1 pair detected" 1 (count pairs))
  (assert-equal "C2-BB: family" :beta-bernoulli (:family (first pairs)))
  (assert-equal "C2-BB: prior-addr" :p (:prior-addr (first pairs)))
  (assert-equal "C2-BB: obs-addr" :x (:obs-addr (first pairs))))

;; C3: Gamma-Poisson
(let [m (gen []
          (let [rate (trace :rate (dist/gamma-dist 2 1))]
            (trace :count (dist/poisson rate))
            rate))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "C3-GP: 1 pair detected" 1 (count pairs))
  (assert-equal "C3-GP: family" :gamma-poisson (:family (first pairs))))

;; C4: Multiple observations (NN, 3 obs)
(let [m (gen [sigma]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y1 (dist/gaussian mu sigma))
            (trace :y2 (dist/gaussian mu sigma))
            (trace :y3 (dist/gaussian mu sigma))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))
      grouped (conj/group-by-prior pairs)]
  (assert-equal "C4-multi-obs: 3 pairs detected" 3 (count pairs))
  (assert-true "C4-multi-obs: all same prior" (every? #(= :mu (:prior-addr %)) pairs))
  (assert-equal "C4-multi-obs: grouped correctly" [:y1 :y2 :y3]
    (sort (mapv :obs-addr (get grouped :mu)))))

;; C5: Gamma-Exponential
(let [m (gen []
          (let [rate (trace :rate (dist/gamma-dist 2 1))]
            (trace :x (dist/exponential rate))
            rate))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "C5-GE: 1 pair detected" 1 (count pairs))
  (assert-equal "C5-GE: family" :gamma-exponential (:family (first pairs))))

;; -- 5 Non-Conjugate Models --

(println "\n=== Section 2: Gate 1 — 5 Non-Conjugate Models ===")

;; NC1: Gaussian → Bernoulli (probit-like, explicitly not conjugate)
(let [m (gen []
          (let [z (trace :z (dist/gaussian 0 1))]
            (trace :x (dist/bernoulli (mx/sigmoid z)))
            z))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "NC1: gaussian→bernoulli = 0 pairs" 0 (count pairs)))

;; NC2: Wrong pairing (beta → gaussian)
(let [m (gen []
          (let [p (trace :p (dist/beta-dist 2 5))]
            (trace :y (dist/gaussian p 1))
            p))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "NC2: beta→gaussian = 0 pairs" 0 (count pairs)))

;; NC3: Nonlinear dependency (gaussian → gaussian via exp)
(let [m (gen []
          (let [mu (trace :mu (dist/gaussian 0 1))]
            (trace :y (dist/gaussian (mx/exp mu) 1))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "NC3: nonlinear dep = 0 pairs" 0 (count pairs)))

;; NC4: Indirect dependency (a → b → c, a NOT direct dep of c's dist)
(let [m (gen []
          (let [a (trace :a (dist/gaussian 0 1))
                b (trace :b (dist/gaussian a 1))
                c (trace :c (dist/gaussian b 1))]
            c))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  ;; Should detect :a→:b and :b→:c, NOT :a→:c
  (assert-equal "NC4: indirect = 2 pairs (a→b, b→c)" 2 (count pairs))
  (assert-true "NC4: no a→c pair"
    (not (some #(and (= :a (:prior-addr %)) (= :c (:obs-addr %))) pairs))))

;; NC5: No dependencies at all
(let [m (gen []
          (let [x (trace :x (dist/gaussian 0 1))
                y (trace :y (dist/gaussian 0 1))]
            (mx/add x y)))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "NC5: independent = 0 pairs" 0 (count pairs)))

;; -- 5 Edge Cases --

(println "\n=== Section 2: Gate 1 — 5 Edge Cases ===")

;; E1: Mixed — some conjugate, some not
(let [m (gen []
          (let [mu (trace :mu (dist/gaussian 0 10))
                z  (trace :z (dist/gaussian 0 1))]
            (trace :y (dist/gaussian mu 1))
            (trace :x (dist/bernoulli (mx/sigmoid z)))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "E1-mixed: 1 conjugate pair" 1 (count pairs))
  (assert-equal "E1-mixed: correct pair" :normal-normal (:family (first pairs))))

;; E2: Two independent conjugate priors
(let [m (gen []
          (let [mu (trace :mu (dist/gaussian 0 10))
                p  (trace :p (dist/beta-dist 2 5))]
            (trace :y (dist/gaussian mu 1))
            (trace :x (dist/bernoulli p))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))
      families (set (map :family pairs))]
  (assert-equal "E2-two-priors: 2 pairs" 2 (count pairs))
  (assert-true "E2-two-priors: NN + BB" (= #{:normal-normal :beta-bernoulli} families)))

;; E3: Obs depends on two traced values (only mean = conjugate)
(let [m (gen []
          (let [mu  (trace :mu (dist/gaussian 0 10))
                sig (trace :sig (dist/gamma-dist 2 1))]
            (trace :y (dist/gaussian mu sig))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "E3-two-deps: 1 pair (mu→y only)" 1 (count pairs))
  (assert-equal "E3-two-deps: prior is mu" :mu (:prior-addr (first pairs))))

;; E4: Shared prior with mixed obs (conjugate + non-conjugate)
(let [m (gen []
          (let [p (trace :p (dist/beta-dist 2 5))]
            (trace :x1 (dist/bernoulli p))    ;; conjugate
            (trace :x2 (dist/gaussian p 1))   ;; NOT conjugate (beta → gaussian)
            p))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "E4-shared-prior: 1 conjugate pair" 1 (count pairs))
  (assert-equal "E4-shared-prior: correct obs" :x1 (:obs-addr (first pairs))))

;; E5: Affine dependency → NOT detected as conjugate (deferred to WP-3)
(let [m (gen [slope intercept sigma]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian (mx/add (mx/multiply slope mu) intercept) sigma))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))]
  (assert-equal "E5-affine: 1 pair detected (WP-3 affine analysis)" 1 (count pairs))
  (assert-equal "E5-affine: dependency type is :affine"
    :affine (:type (:dependency-type (first pairs)))))

;; ---------------------------------------------------------------------------
;; Section 3: Schema Augmentation
;; ---------------------------------------------------------------------------

(println "\n=== Section 3: Schema Augmentation ===")

(let [m (gen [sigma]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu sigma))
            mu))
      s (:schema m)
      aug (conj/augment-schema-with-conjugacy s)]
  (assert-true "augmented has :conjugate-pairs" (contains? aug :conjugate-pairs))
  (assert-true "augmented has :has-conjugate?" (contains? aug :has-conjugate?))
  (assert-equal "has-conjugate? = true" true (:has-conjugate? aug))
  (assert-equal "conjugate-pairs count" 1 (count (:conjugate-pairs aug)))
  ;; Original schema keys still present
  (assert-true "still has :trace-sites" (contains? aug :trace-sites))
  (assert-true "still has :static?" (contains? aug :static?))
  (assert-true "still has :dep-order" (contains? aug :dep-order)))

(let [m (gen []
          (let [x (trace :x (dist/gaussian 0 1))
                y (trace :y (dist/gaussian 0 1))]
            (mx/add x y)))
      aug (conj/augment-schema-with-conjugacy (:schema m))]
  (assert-equal "non-conjugate: has-conjugate? = false" false (:has-conjugate? aug))
  (assert-equal "non-conjugate: empty pairs" 0 (count (:conjugate-pairs aug))))

;; ---------------------------------------------------------------------------
;; Section 4: group-by-prior
;; ---------------------------------------------------------------------------

(println "\n=== Section 4: group-by-prior ===")

(let [m (gen [sigma]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y1 (dist/gaussian mu sigma))
            (trace :y2 (dist/gaussian mu sigma))
            (trace :y3 (dist/gaussian mu sigma))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))
      grouped (conj/group-by-prior pairs)]
  (assert-equal "group-by-prior: 1 group" 1 (count grouped))
  (assert-true "group-by-prior: key is :mu" (contains? grouped :mu))
  (assert-equal "group-by-prior: 3 obs" 3 (count (get grouped :mu))))

(let [m (gen []
          (let [mu (trace :mu (dist/gaussian 0 10))
                p  (trace :p (dist/beta-dist 2 5))]
            (trace :y (dist/gaussian mu 1))
            (trace :x (dist/bernoulli p))
            mu))
      pairs (conj/detect-conjugate-pairs (:schema m))
      grouped (conj/group-by-prior pairs)]
  (assert-equal "group-by-prior: 2 groups" 2 (count grouped))
  (assert-equal "group-by-prior: mu has 1 obs" 1 (count (get grouped :mu)))
  (assert-equal "group-by-prior: p has 1 obs" 1 (count (get grouped :p))))

;; ---------------------------------------------------------------------------
;; Section 5: Gate 0 — Address-Based Dispatch Prototype
;; ---------------------------------------------------------------------------

(println "\n=== Section 5: Gate 0 — Address-Based Dispatch Prototype ===")

;; Build a minimal address-based handler for NN, verify marginal LL matches conjugate.cljs

(let [;; Model: mu ~ N(0, 10), y ~ N(mu, 1), observe y=3.0
      prior-mean (mx/scalar 0.0)
      prior-std (mx/scalar 10.0)
      prior-var (mx/multiply prior-std prior-std)  ;; 100
      obs-value (mx/scalar 3.0)
      obs-std (mx/scalar 1.0)
      obs-var (mx/multiply obs-std obs-std)  ;; 1

      ;; Reference: conjugate.cljs nn-update
      ref-result (conjugate/nn-update
                   {:mean prior-mean :var prior-var}
                   obs-value obs-var (mx/scalar 1.0))
      ref-ll (:ll ref-result)
      ref-posterior (:posterior ref-result)

      ;; Address-based prototype: compute marginal LL manually
      ;; Marginal: y ~ N(prior-mean, prior-var + obs-var)
      marginal-var (mx/add prior-var obs-var)  ;; 101
      diff (mx/subtract obs-value prior-mean)  ;; 3
      proto-ll (mx/multiply (mx/scalar -0.5)
                 (mx/add (mx/scalar 1.8378770664093453)  ;; log(2pi)
                   (mx/add (mx/log marginal-var)
                     (mx/divide (mx/multiply diff diff) marginal-var))))

      ;; Posterior
      inv-prior (mx/divide (mx/scalar 1.0) prior-var)
      inv-obs (mx/divide (mx/scalar 1.0) obs-var)
      post-var (mx/divide (mx/scalar 1.0) (mx/add inv-prior inv-obs))
      post-mean (mx/multiply post-var
                  (mx/add (mx/multiply inv-prior prior-mean)
                          (mx/multiply inv-obs obs-value)))]

  (mx/eval!)
  (assert-close "Gate0-NN: marginal LL matches conjugate.cljs"
    (mx/item ref-ll) (mx/item proto-ll) 1e-10)
  (assert-close "Gate0-NN: posterior mean matches"
    (mx/item (:mean ref-posterior)) (mx/item post-mean) 1e-10)
  (assert-close "Gate0-NN: posterior var matches"
    (mx/item (:var ref-posterior)) (mx/item post-var) 1e-10))

;; Gate 0 for Beta-Bernoulli
(let [;; p ~ Beta(2, 5), x ~ Bernoulli(p), observe x=1
      alpha (mx/scalar 2.0)
      beta-p (mx/scalar 5.0)
      obs-value (mx/scalar 1.0)

      ;; Reference
      ref-result (conjugate/bb-update
                   {:alpha alpha :beta beta-p}
                   obs-value (mx/scalar 1.0))
      ref-ll (:ll ref-result)

      ;; Prototype marginal LL: log(x*alpha + (1-x)*beta) - log(alpha+beta)
      sum-ab (mx/add alpha beta-p)
      proto-ll (mx/subtract
                 (mx/log (mx/add (mx/multiply obs-value alpha)
                                 (mx/multiply (mx/subtract (mx/scalar 1.0) obs-value) beta-p)))
                 (mx/log sum-ab))]
  (mx/eval!)
  (assert-close "Gate0-BB: marginal LL matches conjugate.cljs"
    (mx/item ref-ll) (mx/item proto-ll) 1e-10))

;; Gate 0 for Gamma-Poisson
(let [;; rate ~ Gamma(2, 1), count ~ Poisson(rate), observe count=3
      shape (mx/scalar 2.0)
      rate (mx/scalar 1.0)
      obs-value (mx/scalar 3.0)

      ;; Reference
      ref-result (conjugate/gp-update
                   {:shape shape :rate rate}
                   obs-value (mx/scalar 1.0))
      ref-ll (:ll ref-result)

      ;; Prototype: NegBin marginal LL
      bp1 (mx/add rate (mx/scalar 1.0))
      proto-ll (-> (mx/lgamma (mx/add shape obs-value))
                   (mx/subtract (mx/lgamma shape))
                   (mx/subtract (mx/lgamma (mx/add obs-value (mx/scalar 1.0))))
                   (mx/add (mx/multiply shape (mx/subtract (mx/log rate) (mx/log bp1))))
                   (mx/add (mx/multiply obs-value (mx/negative (mx/log bp1)))))]
  (mx/eval!)
  (assert-close "Gate0-GP: marginal LL matches conjugate.cljs"
    (mx/item ref-ll) (mx/item proto-ll) 1e-10))

;; Gate 0: Standard handler fallthrough test
(let [m (dyn/auto-key
           (gen [sigma]
             (let [mu (trace :mu (dist/gaussian 0 10))]
               (trace :y (dist/gaussian mu sigma))
               mu)))
      ;; Standard generate (no analytical handler)
      result (p/generate m [(mx/scalar 1.0)]
               (-> cm/EMPTY (cm/set-value :y (mx/scalar 3.0))))
      tr (:trace result)]
  (mx/eval!)
  (assert-true "Gate0-fallthrough: trace has :mu"
    (cm/has-value? (cm/get-submap (:choices tr) :mu)))
  (assert-true "Gate0-fallthrough: trace has :y"
    (cm/has-value? (cm/get-submap (:choices tr) :y)))
  (assert-close "Gate0-fallthrough: y value = 3.0"
    3.0 (mx/item (cm/get-value (cm/get-submap (:choices tr) :y))) 1e-10))

;; ---------------------------------------------------------------------------
;; Section 6: Performance
;; ---------------------------------------------------------------------------

(println "\n=== Section 6: Performance ===")

;; Detection on a 50-site model should be fast
(let [;; Build a model with 50 trace sites (1 prior + 49 obs)
      ;; Use schema directly since we can't programmatically build 50 traces in gen
      big-schema {:trace-sites
                  (into [{:addr :mu :dist-type :gaussian :dist-args [0 10]
                          :deps #{} :static? true}]
                    (for [i (range 49)]
                      {:addr (keyword (str "y" i))
                       :dist-type :gaussian
                       :dist-args ['mu 'sigma]
                       :deps #{:mu}
                       :static? true}))
                  :dep-order (into [:mu] (for [i (range 49)] (keyword (str "y" i))))
                  :static? true}
      start (js/Date.now)
      pairs (conj/detect-conjugate-pairs big-schema)
      elapsed (- (js/Date.now) start)]
  (assert-equal "50-site: 49 pairs detected" 49 (count pairs))
  (assert-true (str "50-site: detection < 10ms (took " elapsed "ms)") (< elapsed 10)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== RESULTS: " @pass-count "/" (+ @pass-count @fail-count)
              " passed, " @fail-count " failed ==="))
