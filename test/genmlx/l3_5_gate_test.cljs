(ns genmlx.l3-5-gate-test
  "Level 3.5 Gate 0 + Gate 1: Regenerate correctness and performance."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- strip-analytical [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :auto-regenerate-handlers
                            :auto-regenerate-transition
                            :conjugate-pairs :has-conjugate? :analytical-plan)))

(defn- time-ms [f]
  (let [start (js/Date.now)]
    (f)
    (- (js/Date.now) start)))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian 0 sigma))
      mu)))

(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest gate0a-score-consistency-test
  (testing "Gate 0a: Score consistency"
    (let [model (dyn/auto-key nn-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          gen-result (p/generate model [] obs)
          trace (:trace gen-result)
          gen-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
          gen-weight (do (mx/eval! (:weight gen-result)) (mx/item (:weight gen-result)))
          regen (p/regenerate model trace (sel/select))
          regen-score (do (mx/eval! (:score (:trace regen)))
                          (mx/item (:score (:trace regen))))
          regen-weight (do (mx/eval! (:weight regen))
                           (mx/item (:weight regen)))]
      (is (h/close? gen-score regen-score 1e-6) "regenerate score = generate score")
      (is (h/close? 0.0 regen-weight 1e-6) "regenerate weight = 0 (nothing selected)"))))

(deftest gate0b-mh-chain-convergence-test
  (testing "Gate 0b: MH chain convergence (mixed model, Case B)"
    (let [model mixed-model
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          n-steps 2000
          selection (sel/select :sigma)
          key (rng/fresh-key 42)
          init-trace (:trace (p/generate (dyn/with-key model key) [] obs))
          chain
          (reduce
            (fn [{:keys [trace accepts sigmas scores]} i]
              (let [key-i (rng/fresh-key (+ 1000 i))
                    {:keys [trace weight]} (p/regenerate (dyn/with-key model key-i) trace selection)]
                (mx/eval! weight)
                (mx/eval! (:score trace))
                (let [log-alpha (mx/item weight)
                      accept? (< (js/Math.log (js/Math.random)) log-alpha)
                      sigma-val (mx/item (cm/get-value (cm/get-submap (:choices trace) :sigma)))
                      score-val (mx/item (:score trace))]
                  {:trace trace
                   :accepts (if accept? (inc accepts) accepts)
                   :sigmas (conj sigmas sigma-val)
                   :scores (conj scores score-val)})))
            {:trace init-trace :accepts 0 :sigmas [] :scores []}
            (range n-steps))
          accept-rate (/ (:accepts chain) n-steps)
          burn-in 400
          post-sigmas (subvec (:sigmas chain) burn-in)
          chain-mean (/ (reduce + post-sigmas) (count post-sigmas))]
      (is (every? pos? post-sigmas) "All sigma values positive")
      (is (> accept-rate 0.1) "Acceptance rate > 0.1")
      (is (< accept-rate 0.99) "Acceptance rate < 0.99 (not trivially accepting)")
      (is (and (js/isFinite chain-mean) (pos? chain-mean))
          "Chain mean sigma is finite and positive")
      (let [post-scores (subvec (:scores chain) burn-in)]
        (is (every? js/isFinite post-scores) "All scores finite")))))

(deftest gate0c-trace-validity-test
  (testing "Gate 0c: Trace validity across regenerate cycles"
    (let [model (dyn/auto-key nn-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          trace (:trace (p/generate model [] obs))
          traces
          (reduce
            (fn [traces _]
              (let [regen (p/regenerate model (:trace (last traces)) (sel/select))]
                (conj traces regen)))
            [{:trace trace :weight (mx/scalar 0.0)}]
            (range 20))
          scores (mapv (fn [r] (do (mx/eval! (:score (:trace r)))
                                   (mx/item (:score (:trace r))))) traces)]
      (is (every? #(< (js/Math.abs (- % (first scores))) 1e-4) scores)
          "All 20 regenerate cycles produce same score")
      (is (every? #(< (js/Math.abs (- 3.0 (mx/item (cm/get-value
                    (cm/get-submap (:choices (:trace %)) :y1))))) 1e-6) traces)
          "All regenerate cycles have :y1=3.0"))))

(deftest gate1-performance-test
  (testing "Gate 1a: Wall-clock comparison"
    (let [model-with mixed-model
          model-without (strip-analytical mixed-model)
          obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          selection (sel/select :sigma)
          n-steps 1000

          run-chain
          (fn [model]
            (let [key (rng/fresh-key 42)
                  init-trace (:trace (p/generate (dyn/with-key model key) [] obs))]
              (reduce
                (fn [trace i]
                  (let [key-i (rng/fresh-key (+ 5000 i))
                        {:keys [trace weight]} (p/regenerate (dyn/with-key model key-i) trace selection)]
                    (mx/eval! weight)
                    trace))
                init-trace
                (range n-steps))))

          _ (run-chain model-with)
          _ (run-chain model-without)

          time-with (time-ms #(run-chain model-with))
          time-without (time-ms #(run-chain model-without))

          slowdown (if (pos? time-without) (/ (- time-with time-without) time-without) 0)]
      (is (< slowdown 0.30) "Slowdown < 30% (optimized dispatch)"))))

(cljs.test/run-tests)
