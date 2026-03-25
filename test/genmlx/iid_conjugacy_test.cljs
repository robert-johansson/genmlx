(ns genmlx.iid-conjugacy-test
  "M2 Step 4: Conjugacy + auto-analytical for iid-gaussian."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.inference.auto-analytical :as aa]
            [genmlx.conjugacy :as conj]
            [genmlx.handler :as handler]
            [genmlx.runtime :as rt]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; 1. Conjugacy table entry
;; ---------------------------------------------------------------------------

(deftest conjugacy-table-entry
  (testing "Conjugacy table: :gaussian + :iid-gaussian"
    (let [entry (get conj/conjugacy-table [:gaussian :iid-gaussian])]
      (is (some? entry) "entry exists")
      (is (= :normal-iid-normal (:family entry)) "family")
      (is (= 0 (:natural-param-idx entry)) "natural-param-idx")
      (is (= :mu (:prior-mean-key entry)) "prior-mean-key")
      (is (= :sigma (:prior-std-key entry)) "prior-std-key")
      (is (= :mu (:obs-mean-key entry)) "obs-mean-key")
      (is (= :sigma (:obs-noise-key entry)) "obs-noise-key"))))

;; ---------------------------------------------------------------------------
;; 2. Conjugate pair detection on iid model
;; ---------------------------------------------------------------------------

(def iid-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
      mu)))

(deftest conjugate-pair-detection
  (testing "Conjugate pair detection"
    (let [pairs (conj/detect-conjugate-pairs (:schema iid-model))]
      (is (= 1 (count pairs)) "1 conjugate pair")
      (let [pair (first pairs)]
        (is (= :mu (:prior-addr pair)) "prior-addr")
        (is (= :ys (:obs-addr pair)) "obs-addr")
        (is (= :normal-iid-normal (:family pair)) "family")
        (is (= :direct (get-in pair [:dependency-type :type])) "dep-type direct"))))

  (testing "Augmented schema"
    (let [aug (conj/augment-schema-with-conjugacy (:schema iid-model))]
      (is (:has-conjugate? aug) "has-conjugate?")
      (is (= 1 (count (:conjugate-pairs aug))) "conjugate-pairs count"))))

;; ---------------------------------------------------------------------------
;; 3. nn-iid-update-step math correctness
;; ---------------------------------------------------------------------------

(deftest nn-iid-update-step-math
  (testing "nn-iid-update-step basic"
    (let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
          obs (mx/array [1.0 2.0 3.0 4.0 5.0])
          obs-var (mx/scalar 1.0)
          result (aa/nn-iid-update-step prior obs obs-var)]
      (mx/eval!)
      (is (h/close? 2.994 (mx/item (:mean result)) 0.01) "posterior mean ~ 2.994")
      (is (h/close? 0.1996 (mx/item (:var result)) 0.01) "posterior var ~ 0.1996")
      (is (js/isFinite (mx/item (:ll result))) "ll is finite")
      (is (neg? (mx/item (:ll result))) "ll is negative")))

  (testing "T=1 matches nn-update-step"
    (let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
          obs-single (mx/array [3.0])
          obs-var (mx/scalar 1.0)
          iid-result (aa/nn-iid-update-step prior obs-single obs-var)
          scalar-result (aa/nn-update-step prior (mx/scalar 3.0) obs-var)]
      (mx/eval!)
      (is (h/close? (mx/item (:mean scalar-result)) (mx/item (:mean iid-result)) 1e-6)
          "T=1: mean matches nn-update")
      (is (h/close? (mx/item (:var scalar-result)) (mx/item (:var iid-result)) 1e-6)
          "T=1: var matches nn-update")
      (is (h/close? (mx/item (:ll scalar-result)) (mx/item (:ll iid-result)) 1e-6)
          "T=1: ll matches nn-update")))

  (testing "Large T: posterior tight around sample mean"
    (let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
          obs (mx/array (vec (repeat 100 5.0)))
          obs-var (mx/scalar 1.0)
          result (aa/nn-iid-update-step prior obs obs-var)]
      (mx/eval!)
      (is (h/close? 5.0 (mx/item (:mean result)) 0.01) "T=100: posterior mean ~ 5.0")
      (is (h/close? 0.01 (mx/item (:var result)) 0.001) "T=100: posterior var ~ 0.01"))))

;; ---------------------------------------------------------------------------
;; 4. Handler integration: build-auto-handlers with iid pair
;; ---------------------------------------------------------------------------

(deftest handler-integration
  (testing "Handler integration"
    (let [pairs [{:prior-addr :mu :obs-addr :ys :family :normal-iid-normal}]
          handlers (aa/build-auto-handlers pairs)]
      (is (contains? handlers :mu) "has :mu handler")
      (is (contains? handlers :ys) "has :ys handler")
      (is (= 2 (count handlers)) "2 handlers total"))))

;; ---------------------------------------------------------------------------
;; 5. run-handler with auto-analytical transition (iid-gaussian obs)
;; ---------------------------------------------------------------------------

(deftest run-handler-iid-gaussian
  (testing "run-handler + iid-gaussian"
    (let [model (dyn/auto-key
                  (gen []
                    (let [mu (trace :mu (dist/gaussian 0 10))]
                      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
                      mu)))
          pairs (conj/detect-conjugate-pairs (:schema model))
          handlers (aa/build-auto-handlers pairs)
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          obs-data (mx/array [1.0 2.0 3.0 4.0 5.0])
          constraints (-> cm/EMPTY (cm/set-value :ys obs-data))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          result (rt/run-handler transition init
                   (fn [rt] (apply (:body-fn model) rt [])))]
      (mx/eval!)
      (is (js/isFinite (mx/item (:weight result))) "weight is finite")
      (is (neg? (mx/item (:weight result))) "weight is negative")
      (is (js/isFinite (mx/item (:score result))) "score is finite")
      (is (some? (cm/get-submap (:choices result) :mu)) "choices has :mu")
      (is (some? (cm/get-submap (:choices result) :ys)) "choices has :ys")
      (let [mu-val (mx/item (cm/get-value (cm/get-submap (:choices result) :mu)))
            ref (aa/nn-iid-update-step {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                        obs-data (mx/scalar 1.0))]
        (mx/eval!)
        (is (h/close? (mx/item (:mean ref)) mu-val 1e-6) "mu = posterior mean")
        (is (h/close? (mx/item (:ll ref)) (mx/item (:weight result)) 1e-6) "weight = marginal LL")
        (is (h/close? (mx/item (:ll ref)) (mx/item (:score result)) 1e-6) "score = marginal LL")))))

;; ---------------------------------------------------------------------------
;; 6. End-to-end: p/generate with auto-analytical elimination
;; ---------------------------------------------------------------------------

(def iid-model-e2e
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
      mu)))

(deftest generate-end-to-end
  (testing "p/generate end-to-end"
    (let [gf (dyn/auto-key iid-model-e2e)
          obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          result (p/generate gf [] obs)]
      (mx/eval!)
      (is (js/isFinite (mx/item (:weight result))) "e2e: weight is finite")
      (is (neg? (mx/item (:weight result))) "e2e: weight is negative")
      (let [ref (aa/nn-iid-update-step {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                        (mx/array [1.0 2.0 3.0 4.0 5.0])
                                        (mx/scalar 1.0))]
        (mx/eval!)
        (is (h/close? (mx/item (:ll ref)) (mx/item (:weight result)) 1e-4)
            "e2e: weight = marginal LL")))))

;; ---------------------------------------------------------------------------
;; 7. Multi-obs: iid-gaussian + scalar gaussian on same prior
;; ---------------------------------------------------------------------------

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 3))
      (trace :y-extra (dist/gaussian mu 1))
      mu)))

(deftest mixed-iid-scalar-obs
  (testing "Mixed iid + scalar obs"
    (let [pairs (conj/detect-conjugate-pairs (:schema mixed-model))]
      (is (= 2 (count pairs)) "mixed: 2 pairs detected")
      (let [families (set (map :family pairs))]
        (is (contains? families :normal-iid-normal) "mixed: has normal-iid-normal")
        (is (contains? families :normal-normal) "mixed: has normal-normal")))))

;; ---------------------------------------------------------------------------
;; 8. Regenerate handlers for iid-gaussian
;; ---------------------------------------------------------------------------

(deftest regenerate-handlers
  (testing "Regenerate handlers"
    (let [pairs [{:prior-addr :mu :obs-addr :ys :family :normal-iid-normal}]
          handlers (aa/build-regenerate-handlers pairs)]
      (is (contains? handlers :mu) "regen: has :mu handler")
      (is (contains? handlers :ys) "regen: has :ys handler")
      (is (= 2 (count handlers)) "regen: 2 handlers total"))))

;; ---------------------------------------------------------------------------
;; 9. Variance reduction: auto-analytical should reduce IS weight variance
;; ---------------------------------------------------------------------------

(def var-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 10))
      mu)))

(deftest variance-reduction
  (testing "Variance reduction"
    (let [gf (dyn/auto-key var-model)
          obs (cm/choicemap :ys (mx/array [2.0 2.1 1.9 2.0 2.1 1.9 2.0 2.1 1.9 2.0]))
          weights (vec (for [_ (range 50)]
                         (mx/item (:weight (p/generate gf [] obs)))))
          mean-w (/ (reduce + weights) (count weights))
          var-w (/ (reduce + (map #(* (- % mean-w) (- % mean-w)) weights)) (count weights))]
      (is (h/close? 0.0 var-w 1e-6) "variance reduction: weight variance ~ 0")
      (is (every? #(< (js/Math.abs (- % (first weights))) 1e-6) weights)
          "variance reduction: all weights equal"))))

;; ---------------------------------------------------------------------------
;; 10. Score accounting: score = weight for fully constrained model
;; ---------------------------------------------------------------------------

(deftest score-weight-accounting
  (testing "Score = weight accounting"
    (let [gf (dyn/auto-key iid-model-e2e)
          obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          results (for [_ (range 10)]
                    (let [r (p/generate gf [] obs)
                          tr (:trace r)]
                      {:weight (mx/item (:weight r))
                       :score (mx/item (:score tr))}))]
      (doseq [r results]
        (is (h/close? (:weight r) (:score r) 1e-6) "score ~ weight")))))

(cljs.test/run-tests)
