;; @tier fast
(ns genmlx.dist-boundary2-test
  "Boundary and GFI-citizenship tests for the genmlx-yeam audit cluster:
   xlogy guards at p in {0,1}, student-t fractional df, delta batch shapes,
   map->dist registry protection, and Distribution update/regenerate/project.
   Oracles are closed-form densities and IEEE semantics, never the path
   under test. (dist_boundary_test.cljs covers general support boundaries;
   this file covers the audit's specific holes.)"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]))

(defn- lp [d v] (h/realize (dist/log-prob d (mx/scalar v))))

;; ---------------------------------------------------------------------------
;; (1) xlogy guards: binomial and geometric at p in {0,1}
;; ---------------------------------------------------------------------------

(deftest binomial-boundary-log-probs
  (testing "binomial log-prob at degenerate p (pre-fix: NaN)"
    (is (h/close? 0.0 (lp (dist/binomial 5 0.0) 0.0) 1e-6)
        "P(0 successes | p=0) = 1 -> lp 0")
    (is (h/close? 0.0 (lp (dist/binomial 5 1.0) 5.0) 1e-6)
        "P(5 successes in 5 | p=1) = 1 -> lp 0")
    (is (= ##-Inf (lp (dist/binomial 5 0.0) 3.0))
        "P(3 | p=0) = 0 -> lp -Inf")
    (is (= ##-Inf (lp (dist/binomial 5 1.0) 2.0))
        "P(2 | p=1) = 0 -> lp -Inf")))

(deftest geometric-boundary-log-probs
  (testing "geometric at p=1 (pre-fix: NaN) and p=0 (pre-fix: garbage samples)"
    (is (h/close? 0.0 (lp (dist/geometric 1.0) 0.0) 1e-6)
        "P(k=0 | p=1) = 1 -> lp 0")
    (is (= ##-Inf (lp (dist/geometric 1.0) 2.0))
        "P(k=2 | p=1) = 0 -> lp -Inf")
    (is (h/close? 0.0 (h/realize (dc/dist-sample (dist/geometric 1.0)
                                                 (rng/fresh-key 1)))
                  1e-6)
        "p=1 always samples k=0")
    (is (thrown? js/Error (dist/geometric 0.0))
        "p=0 rejected at construction (sampler divides by log(1)=0)")))

(deftest neg-binomial-open-interval
  (testing "neg-binomial p boundaries rejected (pre-fix: rate division by
            zero at p=1, non-termination at p=0)"
    (is (thrown? js/Error (dist/neg-binomial 3 1.0)) "p=1 rejected")
    (is (thrown? js/Error (dist/neg-binomial 3 0.0)) "p=0 rejected")
    (is (some? (dist/neg-binomial 3 0.5)) "interior p constructs")))

;; ---------------------------------------------------------------------------
;; (2) student-t: fractional df samples from the distribution it scores
;; ---------------------------------------------------------------------------

(deftest student-t-fractional-df
  (testing "df < 1 produces finite samples (pre-fix: int(0.5)=0 normals ->
            chi2=0 -> division -> Inf)"
    (let [d (dist/student-t 0.5 0.0 1.0)
          scalar-draws (mapv #(h/realize (dc/dist-sample d (rng/fresh-key %)))
                             (range 50))
          batch-draws (h/realize-vec (dc/dist-sample-n d (rng/fresh-key 99) 50))]
      (is (every? h/finite? scalar-draws) "50 scalar draws all finite")
      (is (every? h/finite? batch-draws) "50 batched draws all finite")))
  (testing "gamma-based chi2 reproduces the t variance df/(df-2)"
    ;; df=7: Var = 7/5 = 1.4, finite kurtosis -> stable sample variance.
    (let [draws (h/realize-vec (dc/dist-sample-n (dist/student-t 7.0 0.0 1.0)
                                                 (rng/fresh-key 7) 4000))]
      (is (h/close? 1.4 (h/sample-variance draws) 0.25)
          "sample variance ~ 1.4 over 4000 draws"))))

;; ---------------------------------------------------------------------------
;; (4) delta: batch sampling with non-scalar points
;; ---------------------------------------------------------------------------

(deftest delta-batch-shapes
  (testing "dist-sample-n prepends the particle axis (pre-fix: broadcast-to
            [n] is a shape error for any non-scalar point)"
    (let [vec-point (mx/array [1.0 2.0])
          batch (dc/dist-sample-n (dist/delta vec-point) (rng/fresh-key 3) 4)]
      (is (= [4 2] (h/realize-shape batch)) "[d] point -> [n d]")
      (is (every? #(= [1 2] (mapv int %)) (mx/->clj batch))
          "every row is the point"))
    (is (= [4] (h/realize-shape
                (dc/dist-sample-n (dist/delta (mx/scalar 5.0))
                                  (rng/fresh-key 4) 4)))
        "scalar point -> [n]")))

;; ---------------------------------------------------------------------------
;; (5) map->dist: registry protection
;; ---------------------------------------------------------------------------

(deftest map->dist-registry-protection
  (testing "colliding with a builtin type throws (pre-fix: silently
            redefined :gaussian process-wide)"
    (is (thrown? js/Error
                 (dc/map->dist {:type :gaussian
                                :sample (fn [_] (mx/scalar 0.0))
                                :log-prob (fn [_] (mx/scalar 0.0))}))
        "registering :gaussian rejected"))
  (testing "re-registering your own custom type is allowed (REPL flow)"
    (let [spec {:type ::my-dist
                :sample (fn [_] (mx/scalar 1.0))
                :log-prob (fn [_] (mx/scalar -1.0))}
          d1 (dc/map->dist spec)
          d2 (dc/map->dist spec)]
      (is (= ::my-dist (:type d1) (:type d2)) "both registrations succeed"))))

;; ---------------------------------------------------------------------------
;; (6) Distribution GFI citizenship: update / regenerate / project
;; ---------------------------------------------------------------------------

(deftest distribution-update
  (testing "raw distribution supports update (pre-fix: no IUpdate -> throw)"
    (let [d (dist/gaussian 0 1)
          trace (p/simulate (vary-meta d assoc :genmlx.dynamic/key
                                       (rng/fresh-key 5)) [])
          old-v (h/realize (:retval trace))
          {:keys [trace weight discard]} (p/update d trace (cm/->Value (mx/scalar 2.0)))]
      (is (h/close? 2.0 (h/realize (:retval trace)) 1e-6) "value replaced")
      (is (h/close? (- (h/gaussian-lp 2.0 0.0 1.0) (h/gaussian-lp old-v 0.0 1.0))
                    (h/realize weight) 1e-4)
          "update weight = lp(v') - lp(v), closed form")
      (is (h/close? old-v (h/realize (cm/get-value discard)) 1e-6)
          "discard holds the old value"))))

(deftest distribution-regenerate
  (testing "raw distribution supports regenerate (pre-fix: no IRegenerate)"
    (let [d (dist/gaussian 0 1)
          trace (p/simulate (vary-meta d assoc :genmlx.dynamic/key
                                       (rng/fresh-key 6)) [])
          old-v (h/realize (:retval trace))
          sel-res (p/regenerate d trace sel/all)
          none-res (p/regenerate d trace sel/none)]
      (is (h/close? 0.0 (h/realize (:weight sel-res)) 1e-6)
          "selected regenerate weight 0 (prior-proposal cancellation)")
      (is (not= old-v (h/realize (:retval (:trace sel-res))))
          "selected regenerate resamples the value")
      (is (h/close? old-v (h/realize (:retval (:trace none-res))) 1e-6)
          "unselected regenerate keeps the value")
      (is (h/close? 0.0 (h/realize (:weight none-res)) 1e-6)
          "unselected regenerate weight 0"))))

(deftest distribution-project
  (testing "project returns the score iff the root is selected (pre-fix:
            ANY selection other than `none` projected the full score)"
    (let [d (dist/gaussian 0 1)
          trace (p/simulate (vary-meta d assoc :genmlx.dynamic/key
                                       (rng/fresh-key 8)) [])
          score (h/realize (:score trace))]
      (is (h/close? score (h/realize (p/project d trace sel/all)) 1e-6)
          "sel/all -> score")
      (is (h/close? 0.0 (h/realize (p/project d trace sel/none)) 1e-6)
          "sel/none -> 0")
      (is (h/close? 0.0 (h/realize (p/project d trace (sel/select :unrelated))) 1e-6)
          "unrelated address selection -> 0, NOT the full score")
      (is (h/close? score (h/realize (p/project d trace
                                                (sel/complement-sel sel/none)))
                    1e-6)
          "complement of none -> score"))))

;; ---------------------------------------------------------------------------
;; (3) iid-gaussian: validation no longer dead
;; ---------------------------------------------------------------------------

(deftest iid-gaussian-validation
  (testing "negative sigma rejected (pre-fix: ensure-array ran first, so
            the number? guard in check-positive never fired)"
    (is (thrown? js/Error (dist/iid-gaussian 0.0 -1.0 5))
        "sigma must be positive")
    (is (some? (dist/iid-gaussian 0.0 1.0 5)) "valid params construct")))

(cljs.test/run-tests)
