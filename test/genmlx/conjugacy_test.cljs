(ns genmlx.conjugacy-test
  "Tests for Level 3 conjugacy detection.
   Address-based dispatch prototype (marginal LL matches conjugate.cljs)
   15 test models accuracy (5 conjugate, 5 non-conjugate, 5 edge cases)"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.conjugacy :as conj]
            [genmlx.schema :as schema]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as handler]
            [genmlx.runtime :as rt]
            [genmlx.inference.conjugate :as conjugate]
            [genmlx.inference.analytical :as ana]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]))

;; ---------------------------------------------------------------------------
;; Section 1: Conjugacy Table Lookups
;; ---------------------------------------------------------------------------

(deftest conjugacy-table-lookups
  (testing "5 known conjugate families"
    (let [t conj/conjugacy-table]
      (is (= :normal-normal (:family (get t [:gaussian :gaussian]))) "NN family")
      (is (= :beta-bernoulli (:family (get t [:beta-dist :bernoulli]))) "BB family")
      (is (= :gamma-poisson (:family (get t [:gamma-dist :poisson]))) "GP family")
      (is (= :gamma-exponential (:family (get t [:gamma-dist :exponential]))) "GE family")
      (is (= :dirichlet-categorical (:family (get t [:dirichlet :categorical]))) "DC family")))

  (testing "Param keys correct"
    (let [t conj/conjugacy-table]
      (is (= :mu (:prior-mean-key (get t [:gaussian :gaussian]))) "NN prior-mean-key")
      (is (= :sigma (:obs-noise-key (get t [:gaussian :gaussian]))) "NN obs-noise-key")
      (is (= :alpha (:prior-alpha-key (get t [:beta-dist :bernoulli]))) "BB prior-alpha-key")
      (is (= :shape-param (:prior-shape-key (get t [:gamma-dist :poisson]))) "GP prior-shape-key")))

  (testing "Natural param index"
    (let [t conj/conjugacy-table]
      (is (= 0 (:natural-param-idx (get t [:gaussian :gaussian]))) "NN natural-param-idx")
      (is (= 0 (:natural-param-idx (get t [:beta-dist :bernoulli]))) "BB natural-param-idx")))

  (testing "Explicitly not conjugate"
    (let [t conj/conjugacy-table]
      (is (nil? (get t [:gaussian :bernoulli])) "gaussian->bernoulli is nil")
      (is (nil? (get t [:beta-dist :gaussian])) "beta->gaussian is nil")
      (is (nil? (get t [:gaussian :poisson])) "gaussian->poisson is nil")))

  (testing "Not in table at all"
    (let [t conj/conjugacy-table]
      (is (not (contains? t [:uniform :gaussian])) "uniform->gaussian not in table")
      (is (not (contains? t [:gaussian :uniform])) "gaussian->uniform not in table"))))

;; ---------------------------------------------------------------------------
;; Section 2: Gate 1 -- 5 Conjugate Models
;; ---------------------------------------------------------------------------

(deftest conjugate-model-c1-normal-normal
  (testing "C1: Normal-Normal"
    (let [m (gen [sigma]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu sigma))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "C1-NN: 1 pair detected")
      (is (= :normal-normal (:family (first pairs))) "C1-NN: family")
      (is (= :mu (:prior-addr (first pairs))) "C1-NN: prior-addr")
      (is (= :y (:obs-addr (first pairs))) "C1-NN: obs-addr")
      (is (= :direct (get-in (first pairs) [:dependency-type :type])) "C1-NN: direct dependency"))))

(deftest conjugate-model-c2-beta-bernoulli
  (testing "C2: Beta-Bernoulli"
    (let [m (gen []
              (let [p (trace :p (dist/beta-dist 2 5))]
                (trace :x (dist/bernoulli p))
                p))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "C2-BB: 1 pair detected")
      (is (= :beta-bernoulli (:family (first pairs))) "C2-BB: family")
      (is (= :p (:prior-addr (first pairs))) "C2-BB: prior-addr")
      (is (= :x (:obs-addr (first pairs))) "C2-BB: obs-addr"))))

(deftest conjugate-model-c3-gamma-poisson
  (testing "C3: Gamma-Poisson"
    (let [m (gen []
              (let [rate (trace :rate (dist/gamma-dist 2 1))]
                (trace :count (dist/poisson rate))
                rate))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "C3-GP: 1 pair detected")
      (is (= :gamma-poisson (:family (first pairs))) "C3-GP: family"))))

(deftest conjugate-model-c4-multi-obs
  (testing "C4: Multiple observations (NN, 3 obs)"
    (let [m (gen [sigma]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y1 (dist/gaussian mu sigma))
                (trace :y2 (dist/gaussian mu sigma))
                (trace :y3 (dist/gaussian mu sigma))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))
          grouped (conj/group-by-prior pairs)]
      (is (= 3 (count pairs)) "C4-multi-obs: 3 pairs detected")
      (is (every? #(= :mu (:prior-addr %)) pairs) "C4-multi-obs: all same prior")
      (is (= [:y1 :y2 :y3] (sort (mapv :obs-addr (get grouped :mu)))) "C4-multi-obs: grouped correctly"))))

(deftest conjugate-model-c5-gamma-exponential
  (testing "C5: Gamma-Exponential"
    (let [m (gen []
              (let [rate (trace :rate (dist/gamma-dist 2 1))]
                (trace :x (dist/exponential rate))
                rate))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "C5-GE: 1 pair detected")
      (is (= :gamma-exponential (:family (first pairs))) "C5-GE: family"))))

;; ---------------------------------------------------------------------------
;; Section 2: Gate 1 -- 5 Non-Conjugate Models
;; ---------------------------------------------------------------------------

(deftest non-conjugate-nc1-gaussian-bernoulli
  (testing "NC1: Gaussian -> Bernoulli (probit-like)"
    (let [m (gen []
              (let [z (trace :z (dist/gaussian 0 1))]
                (trace :x (dist/bernoulli (mx/sigmoid z)))
                z))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 0 (count pairs)) "NC1: gaussian->bernoulli = 0 pairs"))))

(deftest non-conjugate-nc2-wrong-pairing
  (testing "NC2: Wrong pairing (beta -> gaussian)"
    (let [m (gen []
              (let [p (trace :p (dist/beta-dist 2 5))]
                (trace :y (dist/gaussian p 1))
                p))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 0 (count pairs)) "NC2: beta->gaussian = 0 pairs"))))

(deftest non-conjugate-nc3-nonlinear
  (testing "NC3: Nonlinear dependency (gaussian -> gaussian via exp)"
    (let [m (gen []
              (let [mu (trace :mu (dist/gaussian 0 1))]
                (trace :y (dist/gaussian (mx/exp mu) 1))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 0 (count pairs)) "NC3: nonlinear dep = 0 pairs"))))

(deftest non-conjugate-nc4-indirect
  (testing "NC4: Indirect dependency (a -> b -> c)"
    (let [m (gen []
              (let [a (trace :a (dist/gaussian 0 1))
                    b (trace :b (dist/gaussian a 1))
                    c (trace :c (dist/gaussian b 1))]
                c))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      ;; Should detect :a->:b and :b->:c, NOT :a->:c
      (is (= 2 (count pairs)) "NC4: indirect = 2 pairs (a->b, b->c)")
      (is (not (some #(and (= :a (:prior-addr %)) (= :c (:obs-addr %))) pairs))
          "NC4: no a->c pair"))))

(deftest non-conjugate-nc5-independent
  (testing "NC5: No dependencies at all"
    (let [m (gen []
              (let [x (trace :x (dist/gaussian 0 1))
                    y (trace :y (dist/gaussian 0 1))]
                (mx/add x y)))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 0 (count pairs)) "NC5: independent = 0 pairs"))))

;; ---------------------------------------------------------------------------
;; Section 2: Gate 1 -- 5 Edge Cases
;; ---------------------------------------------------------------------------

(deftest edge-case-e1-mixed
  (testing "E1: Mixed -- some conjugate, some not"
    (let [m (gen []
              (let [mu (trace :mu (dist/gaussian 0 10))
                    z  (trace :z (dist/gaussian 0 1))]
                (trace :y (dist/gaussian mu 1))
                (trace :x (dist/bernoulli (mx/sigmoid z)))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "E1-mixed: 1 conjugate pair")
      (is (= :normal-normal (:family (first pairs))) "E1-mixed: correct pair"))))

(deftest edge-case-e2-two-priors
  (testing "E2: Two independent conjugate priors"
    (let [m (gen []
              (let [mu (trace :mu (dist/gaussian 0 10))
                    p  (trace :p (dist/beta-dist 2 5))]
                (trace :y (dist/gaussian mu 1))
                (trace :x (dist/bernoulli p))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))
          families (set (map :family pairs))]
      (is (= 2 (count pairs)) "E2-two-priors: 2 pairs")
      (is (= #{:normal-normal :beta-bernoulli} families) "E2-two-priors: NN + BB"))))

(deftest edge-case-e3-two-deps
  (testing "E3: Obs depends on two traced values"
    (let [m (gen []
              (let [mu  (trace :mu (dist/gaussian 0 10))
                    sig (trace :sig (dist/gamma-dist 2 1))]
                (trace :y (dist/gaussian mu sig))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "E3-two-deps: 1 pair (mu->y only)")
      (is (= :mu (:prior-addr (first pairs))) "E3-two-deps: prior is mu"))))

(deftest edge-case-e4-shared-prior
  (testing "E4: Shared prior with mixed obs"
    (let [m (gen []
              (let [p (trace :p (dist/beta-dist 2 5))]
                (trace :x1 (dist/bernoulli p))    ;; conjugate
                (trace :x2 (dist/gaussian p 1))   ;; NOT conjugate (beta -> gaussian)
                p))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "E4-shared-prior: 1 conjugate pair")
      (is (= :x1 (:obs-addr (first pairs))) "E4-shared-prior: correct obs"))))

(deftest edge-case-e5-affine
  (testing "E5: Affine dependency"
    (let [m (gen [slope intercept sigma]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian (mx/add (mx/multiply slope mu) intercept) sigma))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "E5-affine: 1 pair detected (WP-3 affine analysis)")
      (is (= :affine (:type (:dependency-type (first pairs)))) "E5-affine: dependency type is :affine"))))

;; ---------------------------------------------------------------------------
;; Section 3: Schema Augmentation
;; ---------------------------------------------------------------------------

(deftest schema-augmentation
  (testing "Conjugate schema augmentation"
    (let [m (gen [sigma]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu sigma))
                mu))
          s (:schema m)
          aug (conj/augment-schema-with-conjugacy s)]
      (is (contains? aug :conjugate-pairs) "augmented has :conjugate-pairs")
      (is (contains? aug :has-conjugate?) "augmented has :has-conjugate?")
      (is (= true (:has-conjugate? aug)) "has-conjugate? = true")
      (is (= 1 (count (:conjugate-pairs aug))) "conjugate-pairs count")
      ;; Original schema keys still present
      (is (contains? aug :trace-sites) "still has :trace-sites")
      (is (contains? aug :static?) "still has :static?")
      (is (contains? aug :dep-order) "still has :dep-order")))

  (testing "Non-conjugate schema augmentation"
    (let [m (gen []
              (let [x (trace :x (dist/gaussian 0 1))
                    y (trace :y (dist/gaussian 0 1))]
                (mx/add x y)))
          aug (conj/augment-schema-with-conjugacy (:schema m))]
      (is (= false (:has-conjugate? aug)) "non-conjugate: has-conjugate? = false")
      (is (= 0 (count (:conjugate-pairs aug))) "non-conjugate: empty pairs"))))

;; ---------------------------------------------------------------------------
;; Section 4: group-by-prior
;; ---------------------------------------------------------------------------

(deftest group-by-prior-tests
  (testing "Single prior, multiple obs"
    (let [m (gen [sigma]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y1 (dist/gaussian mu sigma))
                (trace :y2 (dist/gaussian mu sigma))
                (trace :y3 (dist/gaussian mu sigma))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))
          grouped (conj/group-by-prior pairs)]
      (is (= 1 (count grouped)) "group-by-prior: 1 group")
      (is (contains? grouped :mu) "group-by-prior: key is :mu")
      (is (= 3 (count (get grouped :mu))) "group-by-prior: 3 obs")))

  (testing "Two priors"
    (let [m (gen []
              (let [mu (trace :mu (dist/gaussian 0 10))
                    p  (trace :p (dist/beta-dist 2 5))]
                (trace :y (dist/gaussian mu 1))
                (trace :x (dist/bernoulli p))
                mu))
          pairs (conj/detect-conjugate-pairs (:schema m))
          grouped (conj/group-by-prior pairs)]
      (is (= 2 (count grouped)) "group-by-prior: 2 groups")
      (is (= 1 (count (get grouped :mu))) "group-by-prior: mu has 1 obs")
      (is (= 1 (count (get grouped :p))) "group-by-prior: p has 1 obs"))))

;; ---------------------------------------------------------------------------
;; Section 5: Gate 0 -- Address-Based Dispatch Prototype
;; ---------------------------------------------------------------------------

(deftest gate0-nn-marginal-ll
  (testing "Gate 0 NN: marginal LL matches conjugate.cljs"
    (let [prior-mean (mx/scalar 0.0)
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
          marginal-var (mx/add prior-var obs-var)  ;; 101
          diff (mx/subtract obs-value prior-mean)  ;; 3
          proto-ll (mx/multiply (mx/scalar -0.5)
                     (mx/add (mx/scalar 1.8378770664093453)
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
      (is (h/close? (mx/item ref-ll) (mx/item proto-ll) 1e-10) "Gate0-NN: marginal LL matches conjugate.cljs")
      (is (h/close? (mx/item (:mean ref-posterior)) (mx/item post-mean) 1e-10) "Gate0-NN: posterior mean matches")
      (is (h/close? (mx/item (:var ref-posterior)) (mx/item post-var) 1e-10) "Gate0-NN: posterior var matches"))))

(deftest gate0-bb-marginal-ll
  (testing "Gate 0 BB: marginal LL matches conjugate.cljs"
    (let [alpha (mx/scalar 2.0)
          beta-p (mx/scalar 5.0)
          obs-value (mx/scalar 1.0)

          ref-result (conjugate/bb-update
                       {:alpha alpha :beta beta-p}
                       obs-value (mx/scalar 1.0))
          ref-ll (:ll ref-result)

          sum-ab (mx/add alpha beta-p)
          proto-ll (mx/subtract
                     (mx/log (mx/add (mx/multiply obs-value alpha)
                                     (mx/multiply (mx/subtract (mx/scalar 1.0) obs-value) beta-p)))
                     (mx/log sum-ab))]
      (mx/eval!)
      (is (h/close? (mx/item ref-ll) (mx/item proto-ll) 1e-10) "Gate0-BB: marginal LL matches conjugate.cljs"))))

(deftest gate0-gp-marginal-ll
  (testing "Gate 0 GP: marginal LL matches conjugate.cljs"
    (let [shape (mx/scalar 2.0)
          rate (mx/scalar 1.0)
          obs-value (mx/scalar 3.0)

          ref-result (conjugate/gp-update
                       {:shape shape :rate rate}
                       obs-value (mx/scalar 1.0))
          ref-ll (:ll ref-result)

          bp1 (mx/add rate (mx/scalar 1.0))
          proto-ll (-> (mx/lgamma (mx/add shape obs-value))
                       (mx/subtract (mx/lgamma shape))
                       (mx/subtract (mx/lgamma (mx/add obs-value (mx/scalar 1.0))))
                       (mx/add (mx/multiply shape (mx/subtract (mx/log rate) (mx/log bp1))))
                       (mx/add (mx/multiply obs-value (mx/negative (mx/log bp1)))))]
      (mx/eval!)
      (is (h/close? (mx/item ref-ll) (mx/item proto-ll) 1e-10) "Gate0-GP: marginal LL matches conjugate.cljs"))))

(deftest gate0-fallthrough
  (testing "Gate 0: Standard handler fallthrough test"
    (let [m (dyn/auto-key
               (gen [sigma]
                 (let [mu (trace :mu (dist/gaussian 0 10))]
                   (trace :y (dist/gaussian mu sigma))
                   mu)))
          result (p/generate m [(mx/scalar 1.0)]
                   (-> cm/EMPTY (cm/set-value :y (mx/scalar 3.0))))
          tr (:trace result)]
      (mx/eval!)
      (is (cm/has-value? (cm/get-submap (:choices tr) :mu)) "Gate0-fallthrough: trace has :mu")
      (is (cm/has-value? (cm/get-submap (:choices tr) :y)) "Gate0-fallthrough: trace has :y")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices tr) :y))) 1e-10)
          "Gate0-fallthrough: y value = 3.0"))))

;; ---------------------------------------------------------------------------
;; Section 6: Performance
;; ---------------------------------------------------------------------------

(deftest performance-50-site
  (testing "Detection on a 50-site model should be fast"
    (let [big-schema {:trace-sites
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
      (is (= 49 (count pairs)) "50-site: 49 pairs detected")
      (is (< elapsed 10) (str "50-site: detection < 10ms (took " elapsed "ms)")))))

(cljs.test/run-tests)
