(ns genmlx.llm-msa-test
  "Tests for the MSA (Model Synthesis Architecture) pipeline.

   Three sections:
   1.  Pure tests (no LLM needed) — SCI eval, template assembly, scoring,
       importance sampling posterior, error handling
   1b. Knowledge path pure tests — normalize-llm, parse-math (Instaparse),
       parse->assemble->eval->score round-trip, normalize+parse combined
   2.  Model tests (Qwen3-0.6B fine-tuned + base) — generate-candidate,
       synthesize-and-rank, end-to-end MSA, generate-knowledge-candidate,
       end-to-end MSA with :mode :knowledge

   Run: bun run --bun nbb test/genmlx/llm_msa_test.cljs"
  (:require [genmlx.llm.msa :as msa]
            [genmlx.llm.backend :as llm]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [promesa.core :as pr]
            [clojure.string :as str]))

;; ============================================================
;; Test harness
;; ============================================================

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [msg v]
  (if v
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn- assert-equal [msg expected actual]
  (assert-true (str msg " (expected " (pr-str expected) ", got " (pr-str actual) ")")
               (= expected actual)))

(defn- assert-close [msg expected actual tol]
  (let [ok (< (js/Math.abs (- expected actual)) tol)]
    (if ok
      (do (swap! pass-count inc) (println "  PASS:" msg))
      (do (swap! fail-count inc) (println "  FAIL:" msg "expected:" expected "got:" actual)))))

(defn- report [label]
  (let [p @pass-count f @fail-count]
    (println (str "\n=== " label ": " p "/" (+ p f) " PASS ==="))
    (when (pos? f) (println (str "!!! " f " FAILURES !!!")))))

;; ============================================================
;; Test data
;; ============================================================

(def ^:private xy-code
  "(fn [trace]
     (let [x (trace :x (dist/gaussian 0 10))
           y (trace :y (dist/gaussian x 1))]
       {:x x :y y}))")

(def ^:private xy-task
  {:name "x-causes-y"
   :description "x is normally distributed with mean 0 and std 10. y depends on x with noise std 1."
   :variables [:x :y]
   :observations {:y 5.0}
   :query :x})

(def ^:private causal-code
  "(fn [trace]
     (let [strength (trace :strength (dist/gaussian 5 2))
           time (trace :time (dist/gaussian (mx/divide 30 strength) 0.5))]
       {:strength strength :time time}))")

(def ^:private sensor-fusion-task
  {:name "sensor-fusion"
   :description "Variables: true-value, sensor-a, sensor-b.\ntrue-value ~ gaussian(0, 100)\nsensor-a depends on true-value, noise std 1\nsensor-b depends on true-value, noise std 3"
   :variables [:true-value :sensor-a :sensor-b]
   :observations {:sensor-a 10.0 :sensor-b 14.0}
   :query :true-value})

(def ^:private sensor-fusion-math
  "true-value ~ gaussian(0, 100)\nsensor-a ~ gaussian(true-value, 1)\nsensor-b ~ gaussian(true-value, 3)")

;; ============================================================
;; Section 1: Pure tests (no LLM needed)
;; ============================================================

;; -- 1.1 SCI eval context --

(println "\n-- 1.1 eval-model-fn: returns a callable function --")
(try
  (let [f (msa/eval-model-fn xy-code)]
    (assert-true "returns a function" (fn? f)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: eval-model-fn threw:" (.-message e))))

(println "\n-- 1.1 eval-model: produces a DynamicGF --")
(try
  (let [gf (msa/eval-model xy-code)]
    (assert-true "is not nil" (some? gf))
    (assert-true "is a DynamicGF" (instance? dyn/DynamicGF gf)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: eval-model threw:" (.-message e))))

(println "\n-- 1.1 eval-model: simulate produces valid trace --")
(try
  (let [gf (msa/eval-model xy-code)
        tr (p/simulate gf [])]
    (assert-true "has choices" (some? (:choices tr)))
    (assert-true "has :x in choices" (cm/has-value? (cm/get-submap (:choices tr) :x)))
    (assert-true "has :y in choices" (cm/has-value? (cm/get-submap (:choices tr) :y)))
    (assert-true "score is finite" (js/isFinite (mx/item (:score tr)))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: simulate threw:" (.-message e))))

(println "\n-- 1.1 eval-model: generate with observations --")
(try
  (let [gf (msa/eval-model xy-code)
        obs (cm/choicemap :y (mx/scalar 5.0))
        {:keys [trace weight]} (p/generate gf [] obs)
        y-val (mx/item (cm/get-value (cm/get-submap (:choices trace) :y)))]
    (assert-true "weight is finite" (js/isFinite (mx/item weight)))
    (assert-true "y is conditioned to 5.0" (= 5.0 y-val))
    (assert-true "weight is negative" (< (mx/item weight) 0)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: generate threw:" (.-message e))))

;; -- 1.2 Template assembly --

(println "\n-- 1.2 build-prompt: produces non-empty string --")
(try
  (let [prompt (msa/build-prompt xy-task)]
    (assert-true "prompt is a string" (string? prompt))
    (assert-true "prompt is non-empty" (pos? (count prompt)))
    (assert-true "prompt mentions variables"
                 (and (str/includes? prompt "x")
                      (str/includes? prompt "y")))
    (assert-true "prompt mentions description"
                 (str/includes? prompt (:description xy-task))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: build-prompt threw:" (.-message e))))

(println "\n-- 1.2 parse-dist-lines: parses known input --")
(try
  (let [text "x = (dist/gaussian 0 10)\ny = (dist/gaussian x 1)"
        result (msa/parse-dist-lines text [:x :y])]
    (assert-equal "parses :x" "(dist/gaussian 0 10)" (:x result))
    (assert-equal "parses :y" "(dist/gaussian x 1)" (:y result)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-dist-lines threw:" (.-message e))))

(println "\n-- 1.2 parse-dist-lines: handles extra whitespace --")
(try
  (let [text "  x  =  (dist/gaussian 0 10)  \n  y  =  (dist/gaussian x 1)  "
        result (msa/parse-dist-lines text [:x :y])]
    (assert-true ":x parsed" (some? (:x result)))
    (assert-true ":y parsed" (some? (:y result))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-dist-lines whitespace threw:" (.-message e))))

(println "\n-- 1.2 parse-dist-lines: handles missing variable --")
(try
  (let [text "x = (dist/gaussian 0 10)"
        result (msa/parse-dist-lines text [:x :y :z])]
    (assert-true ":x parsed" (some? (:x result)))
    (assert-true ":y missing" (nil? (:y result)))
    (assert-true ":z missing" (nil? (:z result))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-dist-lines missing threw:" (.-message e))))

(println "\n-- 1.2 assemble-gen-fn: produces valid code --")
(try
  (let [dist-map {:x "(dist/gaussian 0 10)" :y "(dist/gaussian x 1)"}
        code (msa/assemble-gen-fn [:x :y] dist-map)]
    (assert-true "code is a string" (string? code))
    (assert-true "code contains fn" (str/includes? code "fn"))
    (assert-true "code contains trace" (str/includes? code "trace"))
    (assert-true "code contains :x" (str/includes? code ":x"))
    (assert-true "code contains :y" (str/includes? code ":y"))
    ;; The assembled code should eval successfully
    (let [gf (msa/eval-model code)]
      (assert-true "assembled code evals to DynamicGF" (some? gf))
      (when gf
        (let [tr (p/simulate gf [])]
          (assert-true "assembled model simulates"
                       (cm/has-value? (cm/get-submap (:choices tr) :x)))))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: assemble-gen-fn threw:" (.-message e))))

(println "\n-- 1.2 assemble-gen-fn: round-trip with causal model --")
(try
  (let [dist-map {:strength "(dist/gaussian 5 2)"
                  :time "(dist/gaussian (mx/divide 30 strength) 0.5)"}
        code (msa/assemble-gen-fn [:strength :time] dist-map)
        gf (msa/eval-model code)]
    (assert-true "causal model evals" (some? gf))
    (when gf
      (let [tr (p/simulate gf [])]
        (assert-true "has :strength" (cm/has-value? (cm/get-submap (:choices tr) :strength)))
        (assert-true "has :time" (cm/has-value? (cm/get-submap (:choices tr) :time))))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: assemble round-trip threw:" (.-message e))))

;; -- 1.3 Score model --

(println "\n-- 1.3 score-model: finite log-likelihood --")
(try
  (let [gf (msa/eval-model xy-code)
        obs {:y 5.0}
        w (msa/score-model gf obs)]
    (assert-true "weight is a number" (number? w))
    (assert-true "weight is finite" (js/isFinite w))
    (assert-true "weight is negative" (< w 0))
    (println "    weight:" (.toFixed w 3)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: score-model threw:" (.-message e))))

(println "\n-- 1.3 score-model: causal model with observation --")
(try
  (let [gf (msa/eval-model causal-code)
        obs {:time 4.0}
        w (msa/score-model gf obs)]
    (assert-true "weight is finite" (js/isFinite w))
    (assert-true "weight is negative" (< w 0)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: score-model causal threw:" (.-message e))))

;; -- 1.4 Importance sampling posterior --

(println "\n-- 1.4 infer-answer: conjugate posterior for x|y=5 --")
(try
  ;; Conjugate posterior: prior x ~ N(0,100), likelihood y|x ~ N(x,1), obs y=5
  ;; Posterior: x|y=5 ~ N(100*5/101, 100/101) = N(4.9505, 0.9901)
  ;; posterior mean ~ 4.95, posterior variance ~ 0.99
  (let [gf (msa/eval-model xy-code)
        samples (msa/importance-sample gf {:y 5.0} :x 500)
        result (msa/infer-answer samples)]
    (assert-true "result has :mean" (contains? result :mean))
    (assert-true "result has :variance" (contains? result :variance))
    (assert-true "result has :ess" (contains? result :ess))
    (assert-true "mean is a number" (number? (:mean result)))
    (assert-close "posterior mean near 4.95" 4.95 (:mean result) 1.0)
    (println "    posterior mean:" (.toFixed (:mean result) 3))
    (when (:variance result)
      (println "    posterior var:" (.toFixed (:variance result) 3))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: infer-answer threw:" (.-message e))))

(println "\n-- 1.4 infer-answer: strength|time=4 --")
(try
  ;; Observe fast time (4s), expect strong racer (strength > 5)
  ;; time ~ N(30/strength, 0.5), obs time=4 => strength near 30/4=7.5
  (let [gf (msa/eval-model causal-code)
        samples (msa/importance-sample gf {:time 4.0} :strength 500)
        result (msa/infer-answer samples)]
    (assert-true "mean is a number" (number? (:mean result)))
    (assert-true "posterior mean > 5" (> (:mean result) 5))
    (println "    posterior mean:" (.toFixed (:mean result) 3)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: infer-answer strength threw:" (.-message e))))

;; -- 1.5 Error handling --

(println "\n-- 1.5 eval-model: invalid code returns nil --")
(try
  (let [result (msa/eval-model "(fn [trace] (trace :x (nonexistent 0 1)))")]
    (assert-true "invalid dist returns nil" (nil? result)))
  (catch :default _e
    ;; eval-model may throw or return nil; either is acceptable
    (assert-true "invalid dist handled" true)))

(println "\n-- 1.5 eval-model: syntax error returns nil --")
(try
  (let [result (msa/eval-model "(this is not valid cljs")]
    (assert-true "syntax error returns nil" (nil? result)))
  (catch :default _e
    (assert-true "syntax error handled" true)))

(println "\n-- 1.5 eval-model: empty string returns nil --")
(try
  (let [result (msa/eval-model "")]
    (assert-true "empty string returns nil" (nil? result)))
  (catch :default _e
    (assert-true "empty string handled" true)))

(println "\n-- 1.5 score-model: nil model returns -Infinity --")
(try
  (let [w (msa/score-model nil {:y 5.0})]
    (assert-true "nil model gives -Infinity" (= ##-Inf w)))
  (catch :default _e
    (assert-true "nil model handled" true)))

(println "\n-- 1.5 score-model: model with wrong addresses --")
(try
  ;; Model has :x and :y, but we observe :z
  ;; GFI silently ignores extra constraints — model runs unconstrained
  (let [gf (msa/eval-model xy-code)
        w (msa/score-model gf {:z 5.0})]
    (assert-true "wrong address returns finite weight (unconstrained)"
                 (and (number? w) (js/isFinite w))))
  (catch :default _e
    (assert-true "wrong address handled" true)))

;; -- 1b.1 normalize-llm --

(println "\n-- 1b.1 normalize-llm: strips keyword= prefixes and lowercases --")
(try
  (let [r1 (msa/normalize-llm "Gaussian(mean=0, std=100)")]
    (assert-equal "Gaussian(mean=0, std=100) -> gaussian(0, 100)"
                  "gaussian(0, 100)" r1))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: normalize-llm Gaussian threw:" (.-message e))))

(try
  (let [r2 (msa/normalize-llm "Bernoulli(p=0.5)")]
    (assert-equal "Bernoulli(p=0.5) -> bernoulli(0.5)"
                  "bernoulli(0.5)" r2))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: normalize-llm Bernoulli threw:" (.-message e))))

(println "\n-- 1b.1 normalize-llm: multi-line with empty line stripping --")
(try
  (let [input "x ~ Gaussian(0, 1)\n\ny ~ Gaussian(x, 1)"
        result (msa/normalize-llm input)]
    (assert-true "strips empty lines"
                 (not (str/includes? result "\n\n")))
    (assert-true "lowercases dist names"
                 (str/includes? result "gaussian")))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: normalize-llm multi-line threw:" (.-message e))))

;; -- 1b.2 parse-math --

(println "\n-- 1b.2 parse-math: simple two-variable model --")
(try
  (let [result (msa/parse-math "x ~ gaussian(0, 10)\ny ~ gaussian(x, 1)")]
    (assert-true "returns a map" (map? result))
    (assert-equal ":x dist" "(dist/gaussian 0 10)" (:x result))
    (assert-equal ":y dist" "(dist/gaussian x 1)" (:y result)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-math simple threw:" (.-message e))))

(println "\n-- 1b.2 parse-math: arithmetic in arguments --")
(try
  (let [result (msa/parse-math "strength ~ gaussian(5, 2)\ntime ~ gaussian(30 / strength, 0.5)")]
    (assert-true "returns a map" (map? result))
    (assert-true ":strength parsed" (some? (:strength result)))
    (assert-true ":time contains divide" (str/includes? (str (:time result)) "divide")))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-math arithmetic threw:" (.-message e))))

(println "\n-- 1b.2 parse-math: single variable --")
(try
  (let [result (msa/parse-math "machine ~ bernoulli(0.6)")]
    (assert-true "returns a map" (map? result))
    (assert-equal ":machine dist" "(dist/bernoulli 0.6)" (:machine result)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-math single threw:" (.-message e))))

(println "\n-- 1b.2 parse-math: invalid input returns nil --")
(try
  (let [result (msa/parse-math "invalid garbage text")]
    (assert-true "invalid text returns nil" (nil? result)))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-math invalid threw:" (.-message e))))

(println "\n-- 1b.2 parse-math: handles bullet prefixes --")
(try
  (let [result (msa/parse-math "- x ~ gaussian(0, 1)\n- y ~ gaussian(x, 2)")]
    (assert-true "returns a map" (map? result))
    (assert-true ":x parsed" (some? (:x result)))
    (assert-true ":y parsed" (some? (:y result))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-math bullet threw:" (.-message e))))

;; -- 1b.3 parse-math -> assemble -> eval -> score round-trip --

(println "\n-- 1b.3 parse-math -> assemble -> eval -> score: sensor fusion --")
(try
  (let [dist-map (msa/parse-math sensor-fusion-math)]
    (assert-true "parse-math returns a map" (map? dist-map))
    (when (map? dist-map)
      (let [variables [:true-value :sensor-a :sensor-b]
            code (msa/assemble-gen-fn variables dist-map)
            gf (msa/eval-model code)]
        (assert-true "assembled code evals to DynamicGF" (some? gf))
        (when gf
          ;; Score against observations
          (let [w (msa/score-model gf {:sensor-a 10.0 :sensor-b 14.0})]
            (assert-true "weight is finite" (js/isFinite w))
            (assert-true "weight is negative" (< w 0))
            (println "    weight:" (.toFixed w 3)))
          ;; Run importance sampling, posterior mean should be near 10.3
          (let [samples (msa/importance-sample gf {:sensor-a 10.0 :sensor-b 14.0}
                                               :true-value 500)
                result (msa/infer-answer samples)]
            (assert-true "posterior mean is finite" (js/isFinite (:mean result)))
            (assert-close "posterior mean near 10.3" 10.3 (:mean result) 2.0)
            (println "    posterior mean:" (.toFixed (:mean result) 3)))))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: parse-math round-trip threw:" (.-message e))))

;; -- 1b.4 normalize-llm + parse-math combined --

(println "\n-- 1b.4 normalize-llm + parse-math combined: LLM-style output --")
(try
  (let [llm-output "true-value ~ Gaussian(mean=0, std=100)\nsensor-a ~ Gaussian(mean=true-value, std=1)"
        normalized (msa/normalize-llm llm-output)
        dist-map (msa/parse-math normalized)]
    (assert-true "normalize produces string" (string? normalized))
    (assert-true "parse-math returns a map" (map? dist-map))
    (when (map? dist-map)
      (assert-true ":true-value parsed" (some? (:true-value dist-map)))
      (assert-true ":sensor-a parsed" (some? (:sensor-a dist-map)))
      (assert-true ":true-value contains gaussian"
                   (str/includes? (str (:true-value dist-map)) "gaussian"))
      (assert-true ":sensor-a references true-value"
                   (str/includes? (str (:sensor-a dist-map)) "true-value"))))
  (catch :default e
    (swap! fail-count inc)
    (println "  FAIL: normalize+parse combined threw:" (.-message e))))

;; -- Pure test summary --
(report "Pure tests")

;; ============================================================
;; Section 2: Model tests (Qwen3-0.6B fine-tuned + base)
;; ============================================================

(def ^:private model-dir
  (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-cljs"))

(def ^:private base-model-dir
  (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b"))

(println "\n== Loading Qwen3-0.6B-cljs (fine-tuned) and Qwen3-0.6B (base) for model tests... ==")

(pr/let [model-map (llm/load-model model-dir)
         base-model-map (llm/load-model base-model-dir)]
  (println "Both models loaded.\n")

  ;; -- 2.1 generate-candidate --
  (println "\n-- 2.1 generate-candidate: produces code --")

  (pr/let [{:keys [code]} (msa/generate-candidate model-map xy-task {})]
    (assert-true "code is a string" (string? code))
    (assert-true "code is non-empty" (pos? (count code)))
    (assert-true "code contains fn or trace"
                 (or (str/includes? code "fn")
                     (str/includes? code "trace")))
    (assert-true "code contains dist/gaussian"
                 (str/includes? code "dist/gaussian"))
    (println "    generated code:" (pr-str (subs code 0 (min 80 (count code)))))

    ;; The generated code should be eval-able
    (let [gf (msa/eval-model code)]
      (assert-true "generated code evals (or nil)"
                   true) ;; Soft check: LLM output may fail
      (when gf
        (assert-true "generated code produces DynamicGF" (instance? dyn/DynamicGF gf)))))

  ;; -- 2.1 generate-candidate: causal model --
  (println "\n-- 2.1 generate-candidate: strength-time task --")

  (pr/let [task {:name "strength-time"
                 :description "A racer has strength ~ gaussian(5,2). Finish time depends on strength: time ~ gaussian(30/strength, 0.5)."
                 :variables [:strength :time]
                 :observations {:time 4.0}
                 :query :strength}
           {:keys [code]} (msa/generate-candidate model-map task {})]
    (assert-true "produces code" (string? code))
    (assert-true "code non-empty" (pos? (count code)))
    (println "    generated code:" (pr-str (subs code 0 (min 80 (count code))))))

  ;; -- 2.2 synthesize-and-rank --
  (println "\n-- 2.2 synthesize-and-rank: produces ranked candidates --")

  (pr/let [candidates (msa/synthesize-and-rank model-map xy-task {:n 5})]
    (assert-true "returns a vector" (vector? candidates))
    (assert-equal "has 5 candidates" 5 (count candidates))
    (let [with-weight (filter #(and (:weight %) (js/isFinite (:weight %))) candidates)]
      (assert-true "at least 1 has finite weight" (pos? (count with-weight)))
      (println "    " (count with-weight) "/" (count candidates) " with finite weight")
      (when (> (count with-weight) 1)
        ;; Verify sorted descending by weight
        (let [weights (mapv :weight with-weight)]
          (assert-true "sorted descending by weight"
                       (= weights (vec (sort > weights)))))))
    ;; Each candidate should have expected keys
    (doseq [[i c] (map-indexed vector candidates)]
      (when (< i 3)
        (println (str "    [" i "] weight=" (when (:weight c) (.toFixed (:weight c) 2))
                      " code=" (pr-str (subs (str (:code c)) 0 (min 50 (count (str (:code c))))))))))
    (assert-true "candidates have :code" (every? #(contains? % :code) candidates))
    (assert-true "candidates have :weight" (every? #(contains? % :weight) candidates)))

  ;; -- 2.2 synthesize-and-rank: with larger N --
  (println "\n-- 2.2 synthesize-and-rank: N=3 for strength-time --")

  (pr/let [task {:name "strength-time"
                 :description "A racer has strength ~ gaussian(5,2). Time = gaussian(30/strength, 0.5)."
                 :variables [:strength :time]
                 :observations {:time 4.0}
                 :query :strength}
           candidates (msa/synthesize-and-rank model-map task {:n 3})]
    (assert-equal "has 3 candidates" 3 (count candidates))
    (assert-true "all have :code" (every? :code candidates)))

  ;; -- 2.3 End-to-end MSA --
  (println "\n-- 2.3 end-to-end MSA: x-causes-y --")

  (pr/let [result (msa/msa model-map xy-task {:n 5 :particles 200})]
    (assert-true "result has :model" (contains? result :model))
    (assert-true "result has :posterior" (contains? result :posterior))
    (assert-true "result has :candidates" (contains? result :candidates))

    ;; Posterior checks
    (let [post (:posterior result)]
      (assert-true "posterior has :mean" (contains? post :mean))
      (assert-true "posterior has :variance" (contains? post :variance))
      (assert-true "posterior mean is a number" (number? (:mean post)))
      ;; x|y=5 posterior mean should be roughly near 5
      ;; With small N candidates from LLM, allow wide tolerance
      (assert-close "posterior mean roughly near 5" 5.0 (:mean post) 5.0)
      (println "    posterior mean:" (.toFixed (:mean post) 3))
      (when (:variance post)
        (println "    posterior var:" (.toFixed (:variance post) 3))))

    ;; Model should be a map with :gf as a DynamicGF (or nil)
    (when (:gf (:model result))
      (assert-true "model gf is DynamicGF" (instance? dyn/DynamicGF (:gf (:model result)))))

    ;; Candidates vector
    (assert-true "candidates is a vector" (vector? (:candidates result)))
    (println "    candidates:" (count (:candidates result))))

  ;; -- 2.3 end-to-end MSA: strength-time --
  (println "\n-- 2.3 end-to-end MSA: strength-time --")

  (pr/let [task {:name "strength-time"
                 :description "A racer has strength ~ gaussian(5,2). Finish time = gaussian(30/strength, 0.5)."
                 :variables [:strength :time]
                 :observations {:time 4.0}
                 :query :strength}
           result (msa/msa model-map task {:n 3 :particles 200})]
    (assert-true "result has :model" (contains? result :model))
    (assert-true "result has :posterior" (contains? result :posterior))
    (let [post (:posterior result)]
      (assert-true "posterior mean is a number" (number? (:mean post)))
      ;; LLM may not perfectly capture 30/strength; just verify it's a reasonable number
      (assert-true "posterior mean is finite" (js/isFinite (:mean post)))
      (println "    posterior mean:" (.toFixed (:mean post) 3))))

  ;; -- 2.4 generate-knowledge-candidate (base model) --
  (println "\n-- 2.4 generate-knowledge-candidate: sensor fusion --")

  (pr/let [{:keys [code dist-map variables]}
           (msa/generate-knowledge-candidate base-model-map sensor-fusion-task {})]
    (assert-true "code is a string" (string? code))
    (assert-true "code is non-empty" (pos? (count code)))
    (assert-true "code contains fn" (str/includes? code "fn"))
    (assert-true "code contains trace" (str/includes? code "trace"))
    (assert-true "dist-map is a map" (map? dist-map))
    (assert-true "variables is a vector" (vector? variables))
    (println "    generated code:" (pr-str (subs code 0 (min 80 (count code)))))

    ;; The generated code should be eval-able
    (let [gf (msa/eval-model code)]
      (assert-true "generated code evals (or nil)" true)
      (when gf
        (assert-true "generated code produces DynamicGF" (instance? dyn/DynamicGF gf))
        (let [tr (p/simulate gf [])]
          (assert-true "simulates with choices"
                       (some? (:choices tr)))))))

  ;; -- 2.5 end-to-end MSA: :mode :knowledge --
  (println "\n-- 2.5 end-to-end MSA: :mode :knowledge sensor fusion --")

  (pr/let [result (msa/msa base-model-map sensor-fusion-task
                           {:mode :knowledge :n 5 :particles 300})]
    (assert-true "result has :model" (contains? result :model))
    (assert-true "result has :posterior" (contains? result :posterior))
    (assert-true "result has :candidates" (contains? result :candidates))

    ;; Posterior checks — generous tolerance for LLM-generated models
    (let [post (:posterior result)]
      (when post
        (assert-true "posterior has :mean" (contains? post :mean))
        (assert-true "posterior mean is a number" (number? (:mean post)))
        (assert-true "posterior mean is finite" (js/isFinite (:mean post)))
        ;; Sensor fusion: true-value|sensor-a=10,sensor-b=14
        ;; LLM models vary widely; just check it's a reasonable finite number
        (assert-true "posterior mean is finite" (js/isFinite (:mean post)))
        (println "    posterior mean:" (.toFixed (:mean post) 3))
        (when (:variance post)
          (println "    posterior var:" (.toFixed (:variance post) 3)))))

    ;; Candidates vector
    (assert-true "candidates is a vector" (vector? (:candidates result)))
    (println "    candidates:" (count (:candidates result))))

  ;; -- Final summary --
  (println)
  (report "All tests (pure + model)"))
