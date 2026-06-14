;; @tier fast
(ns genmlx.llm-msa-scoring-test
  "Model-free regression for MSA scoring (genmlx-sndo): an opaque-fallback model
   (empty trace-sites schema) must NOT be scored :exact (a single joint draw);
   it routes to importance sampling labeled :handler-is. Independent oracle: the
   gaussian-gaussian marginal in closed form."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.llm.msa :as msa]))

;; x ~ N(0,1), y ~ N(x,1)  =>  marginal y ~ N(0, 2).
;; log p(y=1) = -0.5*log(2*pi*2) - 1^2/(2*2)
(def code "(fn [trace] (let [x (trace :x (dist/gaussian 0 1))] (trace :y (dist/gaussian x 1)) x))")
(def true-logml (- (* -0.5 (js/Math.log (* 2 js/Math.PI 2))) (/ 1.0 4.0)))

(deftest opaque-fallback-not-mislabeled-exact
  (testing "opaque model (no faithful source) has empty trace-sites"
    (let [opaque (msa/wrap-model (msa/eval-model-fn code))]   ; one-arg = opaque wrapper
      (is (empty? (:trace-sites (:schema opaque))) "opaque schema sees no trace-sites")
      (testing "score-model* routes it to IS, not :exact (genmlx-sndo)"
        (let [{:keys [log-ml method]} (msa/score-model* opaque {:y 1.0} {:n-particles 300})]
          (is (not= :exact method) (str "must not be :exact (single-draw), got " method))
          (is (= :handler-is method) "opaque/no-site escape -> :handler-is")
          (is (js/isFinite log-ml) "log-ml finite")
          (is (< (js/Math.abs (- log-ml true-logml)) 0.4)
              (str "IS log-ml " (.toFixed log-ml 4) " ~ closed-form marginal "
                   (.toFixed true-logml 4))))))))

(deftest faithful-source-stays-exact
  (testing "a faithful source form (real keyword trace-sites) IS :exact"
    (let [gf (msa/wrap-model (msa/eval-model-fn code) (msa/code->source-form code))]
      (is (pos? (count (:trace-sites (:schema gf)))) "faithful schema has trace-sites")
      (let [{:keys [log-ml method]} (msa/score-model* gf {:y 1.0})]
        (is (= :exact method) "conjugate gaussian-gaussian eliminates exactly")
        (is (< (js/Math.abs (- log-ml true-logml)) 1e-3)
            (str "exact log-ml " (.toFixed log-ml 6) " == closed form "
                 (.toFixed true-logml 6)))))))

(cljs.test/run-tests)
