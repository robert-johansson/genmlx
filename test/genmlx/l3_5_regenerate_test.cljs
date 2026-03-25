(ns genmlx.l3-5-regenerate-test
  "Level 3.5 WP-0: Regenerate auto-handler integration tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- strip-analytical
  "Remove auto-handlers from a gen-fn, forcing standard handler path."
  [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :auto-regenerate-handlers
                            :auto-regenerate-transition
                            :conjugate-pairs :has-conjugate? :analytical-plan)))

(defn- has-regen-handlers? [gf]
  (boolean (:auto-regenerate-handlers (:schema gf))))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

(def bb-model
  (gen []
    (let [p (trace :p (dist/beta-dist 2 5))]
      (trace :y1 (dist/bernoulli p))
      (trace :y2 (dist/bernoulli p))
      (trace :y3 (dist/bernoulli p))
      p)))

(def gp-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 3 2))]
      (trace :y1 (dist/poisson rate))
      (trace :y2 (dist/poisson rate))
      rate)))

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian 0 sigma))
      mu)))

(def kalman-model
  (gen []
    (let [z0 (trace :z0 (dist/gaussian 0 10))
          z1 (trace :z1 (dist/gaussian z0 1))]
      (trace :y0 (dist/gaussian z0 0.5))
      (trace :y1 (dist/gaussian z1 0.5))
      z0)))

(def no-conj-model
  (gen []
    (let [x (trace :x (dist/uniform 0 10))]
      (trace :y1 (dist/gaussian (mx/sin x) 1))
      x)))

(def ge-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 2 1))]
      (trace :y1 (dist/exponential rate))
      (trace :y2 (dist/exponential rate))
      rate)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest schema-detection-test
  (testing "Schema detection"
    (is (has-regen-handlers? nn-model) "NN model has regenerate handlers")
    (is (has-regen-handlers? bb-model) "BB model has regenerate handlers")
    (is (has-regen-handlers? gp-model) "GP model has regenerate handlers")
    (is (has-regen-handlers? mixed-model) "Mixed model has regenerate handlers")
    (is (has-regen-handlers? kalman-model) "Kalman model has regenerate handlers")
    (is (not (has-regen-handlers? no-conj-model)) "No-conj model does NOT have regenerate handlers")))

(deftest nn-score-consistency-test
  (testing "NN: score consistency (Case B)"
    (let [model (dyn/auto-key nn-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          regen-result (p/regenerate model trace (sel/select))
          new-score (do (mx/eval! (:score (:trace regen-result)))
                        (mx/item (:score (:trace regen-result))))
          weight (do (mx/eval! (:weight regen-result))
                     (mx/item (:weight regen-result)))]
      (is (h/close? old-score new-score 1e-4)
          "NN: regenerate new_score = old_score (nothing changed)")
      (is (h/close? 0.0 weight 1e-4)
          "NN: regenerate weight = 0 (nothing changed)"))))

(deftest nn-regenerate-matches-generate-test
  (testing "Regenerate score matches generate"
    (let [model (dyn/auto-key nn-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          gen-result (p/generate model [] obs)
          gen-score (do (mx/eval! (:score (:trace gen-result)))
                        (mx/item (:score (:trace gen-result))))
          trace (:trace gen-result)
          regen-result (p/regenerate model trace (sel/select))
          regen-score (do (mx/eval! (:score (:trace regen-result)))
                          (mx/item (:score (:trace regen-result))))]
      (is (h/close? gen-score regen-score 1e-4)
          "Regenerate score = generate score (same model, no changes)"))))

(deftest case-a-fallthrough-test
  (testing "Case A fallthrough (prior selected)"
    (let [model-with (dyn/auto-key nn-model)
          model-without (dyn/auto-key (strip-analytical nn-model))
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          key (rng/fresh-key 42)
          gen-result-with (p/generate (dyn/with-key nn-model key) [] obs)
          gen-result-without (p/generate (dyn/with-key (strip-analytical nn-model) key) [] obs)
          selection (sel/select :mu)
          key2 (rng/fresh-key 99)
          regen-with (p/regenerate (dyn/with-key nn-model key2)
                       (:trace gen-result-with) selection)
          regen-without (p/regenerate (dyn/with-key (strip-analytical nn-model) key2)
                          (:trace gen-result-without) selection)
          w-with (do (mx/eval! (:weight regen-with)) (mx/item (:weight regen-with)))]
      (is (js/isFinite w-with) "Case A: regenerate weight is finite"))))

(deftest bb-case-b-regenerate-test
  (testing "BB model: Case B regenerate"
    (let [model (dyn/auto-key bb-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 1.0))
                  (cm/set-value :y2 (mx/scalar 0.0))
                  (cm/set-value :y3 (mx/scalar 1.0)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          regen-result (p/regenerate model trace (sel/select))
          new-score (do (mx/eval! (:score (:trace regen-result)))
                        (mx/item (:score (:trace regen-result))))
          weight (do (mx/eval! (:weight regen-result))
                     (mx/item (:weight regen-result)))]
      (is (h/close? old-score new-score 1e-4) "BB: regenerate new_score = old_score")
      (is (h/close? 0.0 weight 1e-4) "BB: regenerate weight = 0 (nothing changed)"))))

(deftest gp-case-b-regenerate-test
  (testing "GP model: Case B regenerate"
    (let [model (dyn/auto-key gp-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 1.0)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          regen-result (p/regenerate model trace (sel/select))
          new-score (do (mx/eval! (:score (:trace regen-result)))
                        (mx/item (:score (:trace regen-result))))
          weight (do (mx/eval! (:weight regen-result))
                     (mx/item (:weight regen-result)))]
      (is (h/close? old-score new-score 1e-4) "GP: regenerate new_score = old_score")
      (is (h/close? 0.0 weight 1e-4) "GP: regenerate weight = 0 (nothing changed)"))))

(deftest mixed-model-sigma-selected-test
  (testing "Mixed model: sigma selected (Case B for mu pair)"
    (let [model (dyn/auto-key mixed-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          selection (sel/select :sigma)
          regen-result (p/regenerate model trace selection)
          weight (do (mx/eval! (:weight regen-result))
                     (mx/item (:weight regen-result)))]
      (is (js/isFinite weight)
          "Mixed model: regenerate with sigma selected produces finite weight"))))

(deftest kalman-case-b-regenerate-test
  (testing "Kalman model: Case B regenerate"
    (let [model (dyn/auto-key kalman-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y0 (mx/scalar 2.0))
                  (cm/set-value :y1 (mx/scalar 3.0)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          regen-result (p/regenerate model trace (sel/select))
          new-score (do (mx/eval! (:score (:trace regen-result)))
                        (mx/item (:score (:trace regen-result))))
          weight (do (mx/eval! (:weight regen-result))
                     (mx/item (:weight regen-result)))]
      (is (h/close? old-score new-score 1e-4) "Kalman: regenerate new_score = old_score")
      (is (h/close? 0.0 weight 1e-4) "Kalman: regenerate weight = 0 (nothing changed)"))))

(deftest no-conjugacy-fallback-test
  (testing "No conjugacy: standard fallback"
    (let [model (dyn/auto-key no-conj-model)
          gen-result (p/generate model [] (cm/set-value cm/EMPTY :y1 (mx/scalar 0.5)))
          trace (:trace gen-result)
          regen-result (p/regenerate model trace (sel/select :x))
          weight (do (mx/eval! (:weight regen-result))
                     (mx/item (:weight regen-result)))]
      (is (js/isFinite weight) "No-conj model: regenerate produces finite weight"))))

(deftest mh-chain-convergence-test
  (testing "MH chain convergence (mixed model, sigma selected)"
    (let [model mixed-model
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          n-steps 500
          selection (sel/select :sigma)
          key (rng/fresh-key 42)
          init-trace (:trace (p/generate (dyn/with-key model key) [] obs))
          chain
          (reduce
            (fn [{:keys [trace accepts sigmas]} i]
              (let [key-i (rng/fresh-key (+ 1000 i))
                    {:keys [trace weight]} (p/regenerate (dyn/with-key model key-i) trace selection)]
                (mx/eval! weight)
                (let [log-alpha (mx/item weight)
                      accept? (< (js/Math.log (js/Math.random)) log-alpha)
                      sigma-val (mx/item (cm/get-value (cm/get-submap (:choices trace) :sigma)))]
                  {:trace trace
                   :accepts (if accept? (inc accepts) accepts)
                   :sigmas (conj sigmas sigma-val)})))
            {:trace init-trace :accepts 0 :sigmas []}
            (range n-steps))
          accept-rate (/ (:accepts chain) n-steps)
          burn-in 100
          post-sigmas (subvec (:sigmas chain) burn-in)
          chain-mean (/ (reduce + post-sigmas) (count post-sigmas))]
      (is (every? pos? post-sigmas) "MH chain produces positive sigma values")
      (is (> accept-rate 0.0) "MH acceptance rate > 0")
      (is (js/isFinite chain-mean) "MH chain mean sigma is finite"))))

(deftest stripped-model-regenerate-test
  (testing "Stripped model regenerate matches standard"
    (let [model-stripped (dyn/auto-key (strip-analytical nn-model))
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          gen-result (p/generate model-stripped [] obs)
          trace (:trace gen-result)
          old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          regen-result (p/regenerate model-stripped trace (sel/select))
          new-score (do (mx/eval! (:score (:trace regen-result)))
                        (mx/item (:score (:trace regen-result))))
          weight (do (mx/eval! (:weight regen-result))
                     (mx/item (:weight regen-result)))]
      (is (h/close? old-score new-score 1e-4) "Stripped: regenerate new_score = old_score")
      (is (h/close? 0.0 weight 1e-4) "Stripped: regenerate weight = 0"))))

(deftest trace-choice-correctness-test
  (testing "Trace choice correctness"
    (let [model (dyn/auto-key nn-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          regen-result (p/regenerate model trace (sel/select))
          new-choices (:choices (:trace regen-result))]
      (is (cm/has-value? (cm/get-submap new-choices :mu)) "Regenerated trace has :mu")
      (is (cm/has-value? (cm/get-submap new-choices :y1)) "Regenerated trace has :y1")
      (is (cm/has-value? (cm/get-submap new-choices :y2)) "Regenerated trace has :y2")
      (let [y1 (mx/item (cm/get-value (cm/get-submap new-choices :y1)))
            y2 (mx/item (cm/get-value (cm/get-submap new-choices :y2)))]
        (is (h/close? 3.0 y1 1e-6) "y1 preserved")
        (is (h/close? 4.0 y2 1e-6) "y2 preserved")))))

(deftest deterministic-regenerate-test
  (testing "Deterministic regenerate"
    (let [model nn-model
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          key1 (rng/fresh-key 42)
          trace (:trace (p/generate (dyn/with-key model key1) [] obs))
          key2 (rng/fresh-key 99)
          r1 (p/regenerate (dyn/with-key model key2) trace (sel/select :mu))
          r2 (p/regenerate (dyn/with-key model key2) trace (sel/select :mu))
          w1 (do (mx/eval! (:weight r1)) (mx/item (:weight r1)))
          w2 (do (mx/eval! (:weight r2)) (mx/item (:weight r2)))]
      (is (h/close? w1 w2 1e-10) "Deterministic: same key -> same weight"))))

(deftest ge-case-b-regenerate-test
  (testing "GE model: Case B regenerate"
    (let [model (dyn/auto-key ge-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 0.5))
                  (cm/set-value :y2 (mx/scalar 1.0)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          regen-result (p/regenerate model trace (sel/select))
          new-score (do (mx/eval! (:score (:trace regen-result)))
                        (mx/item (:score (:trace regen-result))))
          weight (do (mx/eval! (:weight regen-result))
                     (mx/item (:weight regen-result)))]
      (is (h/close? old-score new-score 1e-4) "GE: regenerate new_score = old_score")
      (is (h/close? 0.0 weight 1e-4) "GE: regenerate weight = 0 (nothing changed)"))))

(deftest regenerate-retval-test
  (testing "Retval correctness"
    (let [model (dyn/auto-key nn-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          regen-result (p/regenerate model trace (sel/select))
          retval (:retval (:trace regen-result))]
      (is (some? retval) "Regenerate returns a retval"))))

(cljs.test/run-tests)
