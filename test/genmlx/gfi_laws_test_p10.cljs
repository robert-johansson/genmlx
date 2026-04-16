(ns genmlx.gfi-laws-test-p10
  "GFI law tests part 10: VERIFY INTEGRATION tests"
  (:require [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.gfi :as gfi]
            [genmlx.gfi-laws-helpers :as glh
             :refer [gaussian-chain branching-model]])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Integration tests — gfi/verify runs many laws internally, heavy on memory
;; ---------------------------------------------------------------------------

(t/deftest gfi-verify-integration
  (t/testing "gfi/verify runs algebraic laws on gaussian-chain"
    ;; Excludes statistical/training laws (tested by their own deftests):
    ;; :mixture-kernel-stationarity, :hmc-acceptance-correctness,
    ;; :proposal-training-objective. The integration test verifies the
    ;; gfi/verify API on algebraic laws, not MCMC/training convergence.
    (let [slow-laws #{:mixture-kernel-stationarity
                      :hmc-acceptance-correctness
                      :proposal-training-objective
                      :involutive-mh-convergence}
          algebraic-laws (->> gfi/laws
                              (remove #(slow-laws (:name %)))
                              (mapv :name))
          report (gfi/verify (:model gaussian-chain) (:args gaussian-chain)
                             :law-names algebraic-laws
                             ;; n-trials=2: API integration test, not statistical.
                             ;; Statistical rigor is in individual law tests (p1-p9).
                             ;; Higher values exhaust Bun's ~1.8GB RSS limit.
                             :n-trials 2)]
      (t/is (:all-pass? report)
            (str "GFI laws failed: "
                 (pr-str (filterv #(not (:pass? %)) (:results report))))))))

(t/deftest gfi-verify-branching
  (t/testing "gfi/verify core laws on branching model"
    (let [report (gfi/verify (:model branching-model) (:args branching-model)
                             :tags #{:simulate :core}
                             :n-trials 3)]
      ;; Some laws (update-round-trip, decomposition) may not apply
      ;; to branching models — just verify simulate/generate/assess
      (t/is (>= (:total-pass report) (* 3 3))
            (str "Too few core law passes on branching model: "
                 (:total-pass report))))))

(t/run-tests)
