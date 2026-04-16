(ns genmlx.gfi-laws-test-p9
  "GFI law tests part 9: WELL-FORMEDNESS + VECTORIZED + GRADIENT laws"
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.diff :as diff]
            [genmlx.dynamic :as dyn]
            [genmlx.edit :as edit]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.gfi :as gfi]
            [genmlx.gradients :as grad]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]
            [genmlx.learning :as learn]
            [genmlx.test-helpers :as h]
            [genmlx.combinators :as comb]
            [genmlx.inference.smc :as smc]
            [genmlx.verify :as verify]
            [genmlx.gfi-laws-helpers :as glh
             :refer [ev close? gen-model gen-nonbranching gen-multisite gen-splice
                     gen-vectorizable gen-differentiable gen-with-args gen-compiled
                     gen-compiled-multisite model-pool non-branching-pool multi-site-pool
                     vectorized-pool differentiable-pool models-with-args compiled-pool
                     compiled-multisite-pool splice-pool
                     single-gaussian single-uniform single-exponential single-beta
                     two-independent three-independent gaussian-chain three-chain
                     mixed-disc-cont branching-model linear-regression single-arg-model
                     two-arg-model splice-inner splice-dependent splice-independent
                     splice-inner-inner splice-mid splice-nested five-site arg-branching
                     logsumexp N-moment-samples collect-samples sample-mean sample-var
                     sample-cov]])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; HAS-ARGUMENT-GRADS consistency [T] §2.3.1
;; ---------------------------------------------------------------------------

(t/deftest law:has-argument-grads-consistency
  (t/testing "DynamicGF returns nil for has-argument-grads"
    (t/is (nil? (p/has-argument-grads (:model single-gaussian)))
          "DynamicGF with no args")
    (t/is (nil? (p/has-argument-grads (:model single-arg-model)))
          "DynamicGF with args"))
  (t/testing "Gradient is correct (not just finite) for single-arg model"
    ;; x ~ N(mu, 1): d(score)/d(mu) = x - mu (analytical)
    ;; At mu=3, with known x, gradient should match analytical value
    (let [model (:model single-arg-model)
          mu-val 3.0
          t (p/simulate model [(mx/scalar mu-val)])
          choices (:choices t)
          x-val (ev (cm/get-value (cm/get-submap choices :x)))
          score-fn (fn [mu] (:weight (p/generate model [mu] choices)))
          grad-val (ev ((mx/grad score-fn) (mx/scalar mu-val)))
          ;; Analytical: d/d(mu) log N(x; mu, 1) = x - mu
          analytical (- x-val mu-val)]
      (t/is (h/finite? grad-val)
            (str "Gradient should be finite, got " grad-val))
      (t/is (close? grad-val analytical 1e-4)
            (str "Gradient=" grad-val " vs analytical=" analytical)))))

;; ---------------------------------------------------------------------------
;; WELL-FORMEDNESS laws [T] §2.2.1
;; ---------------------------------------------------------------------------

(defspec law:no-external-randomness 100
  ;; DML restriction 3: no external randomness in model source
  (prop/for-all [m gen-nonbranching]
                (let [source (:source (:model m))]
                  (if (nil? source)
                    true
                    (empty? (verify/check-no-external-randomness source))))))

(defspec law:no-mutation 100
  ;; DML restriction 4: no mutation in model source
  (prop/for-all [m gen-nonbranching]
                (let [source (:source (:model m))]
                  (if (nil? source)
                    true
                    (empty? (verify/check-no-mutation source))))))

(defspec law:no-hof-gen-fns 100
  ;; DML restriction 5: no gen fns passed to HOFs
  (prop/for-all [m gen-nonbranching]
                (let [source (:source (:model m))]
                  (if (nil? source)
                    true
                    (empty? (verify/check-no-hof-gen-fns source))))))

;; Negative tests: known-bad source forms MUST be detected
(t/deftest law:well-formedness-negative-tests
  (t/testing "External randomness detected"
    (let [bad-source '([x] (let [r (rand)] (trace :x (dist/gaussian r 1))))]
      (t/is (seq (verify/check-no-external-randomness bad-source))
            "rand should trigger external randomness violation")))
  (t/testing "Mutation detected"
    (let [bad-source '([x] (let [a (atom 0)] (swap! a inc) (trace :x (dist/gaussian 0 1))))]
      (t/is (seq (verify/check-no-mutation bad-source))
            "atom/swap! should trigger mutation violation")))
  (t/testing "HOF gen fn detected"
    (let [bad-source '([xs] (map (gen [x] (trace :y (dist/gaussian x 1))) xs))]
      (t/is (seq (verify/check-no-hof-gen-fns bad-source))
            "gen fn in map should trigger HOF violation"))))

;; ---------------------------------------------------------------------------
;; VECTORIZED GFI laws — shape and finiteness for batched execution
;; ---------------------------------------------------------------------------

(defspec law:vgenerate-shape-and-finiteness 50
  ;; vgenerate produces [N]-shaped scores, all finite
  (prop/for-all [m gen-vectorizable]
                (let [{:keys [model args]} m
                      n 10
                      t (p/simulate model args)
                      addrs (cm/addresses (:choices t))]
                  (if (< (count addrs) 2)
                    true ;; skip single-site: full constraints is degenerate for partial-obs test
                    (let [obs-addr (first (first addrs))
                          obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                          obs (cm/choicemap obs-addr obs-val)
                          vt (dyn/vgenerate model args obs n (rng/fresh-key 99))]
                      (and (= [n] (vec (mx/shape (:score vt))))
                           (every? h/finite? (mx/->clj (:score vt)))))))))

(defspec law:vupdate-shape-and-finiteness 50
  (prop/for-all [m gen-vectorizable]
                (let [{:keys [model args]} m
                      n 10
                      vt (dyn/vsimulate model args n (rng/fresh-key 88))
                      t-scalar (p/simulate model args)
                      addrs (cm/addresses (:choices t-scalar))]
                  (if (empty? addrs)
                    true
                    (let [obs-addr (first (first addrs))
                          obs-val (cm/get-value (cm/get-submap (:choices t-scalar) obs-addr))
                          obs (cm/choicemap obs-addr obs-val)
                          {:keys [weight]} (dyn/vupdate model vt obs (rng/fresh-key 77))]
                      (and (= [n] (vec (mx/shape weight)))
                           (every? h/finite? (mx/->clj weight))))))))

(defspec law:vregenerate-preserves-unselected 50
  (prop/for-all [m gen-vectorizable]
                (let [{:keys [model args]} m
                      n 10
                      vt (dyn/vsimulate model args n (rng/fresh-key 66))
                      addrs (cm/addresses (:choices vt))]
                  (if (< (count addrs) 2)
                    true
                    (let [selected (first (first addrs))
                          unselected-addrs (map first (rest addrs))
                          orig-vals (into {} (map (fn [a]
                                                    [a (mx/->clj (cm/get-value
                                                                   (cm/get-submap
                                                                    (:choices vt) a)))])
                                                  unselected-addrs))
                          {:keys [vtrace]} (dyn/vregenerate model vt
                                                             (sel/select selected)
                                                             (rng/fresh-key 55))]
                      (every? (fn [a]
                                (let [new-vals (mx/->clj (cm/get-value
                                                          (cm/get-submap
                                                           (:choices vtrace) a)))]
                                  (= (get orig-vals a) new-vals)))
                              unselected-addrs))))))

;; ---------------------------------------------------------------------------
;; GRADIENT ON CONSTRAINED ADDRESSES [T] Eq 2.12
;; ---------------------------------------------------------------------------

(t/deftest law:gradient-on-constrained-addresses
  (t/testing "d(score)/d(x) when y is constrained: x ~ N(0,1), y ~ N(x,1), y=3"
    ;; Analytical: score = -0.5*x^2 - 0.5*(3-x)^2 + const
    ;; d(score)/d(x) = -x + (3-x) = 3 - 2x
    (let [model (dyn/auto-key
                  (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                            (trace :y (dist/gaussian x 1)))))
          obs (cm/choicemap :y (mx/scalar 3.0))]
      (doseq [x-val [0.0 1.0 1.5 2.0 3.0]]
        (let [choices (cm/choicemap :x (mx/scalar x-val) :y (mx/scalar 3.0))
              {:keys [trace]} (p/generate model [] choices)
              _ (mx/materialize! (:score trace))
              grads (grad/choice-gradients model trace [:x])
              ad-grad (ev (get grads :x))
              analytical (- 3.0 (* 2.0 x-val))
              ;; Also verify via finite differences
              h 1e-3
              choices+ (cm/choicemap :x (mx/scalar (+ x-val h)) :y (mx/scalar 3.0))
              choices- (cm/choicemap :x (mx/scalar (- x-val h)) :y (mx/scalar 3.0))
              score+ (ev (:score (:trace (p/generate model [] choices+))))
              score- (ev (:score (:trace (p/generate model [] choices-))))
              fd-grad (/ (- score+ score-) (* 2.0 h))]
          (t/is (close? ad-grad analytical 1e-4)
                (str "AD grad at x=" x-val ": " ad-grad " vs analytical " analytical))
          (t/is (close? fd-grad analytical 0.01)
                (str "FD grad at x=" x-val ": " fd-grad " vs analytical " analytical)))))))

(t/run-tests)
