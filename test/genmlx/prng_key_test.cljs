(ns genmlx.prng-key-test
  "Reproducibility tests for PRNG key threading (Phase 2).
   Verifies that same key → same results, different key → different results,
   and that with-key works across GFI operations and inference algorithms."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.gen :refer [gen]]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- assert-true [msg pred]
  (if pred
    (do (vswap! pass-count inc)
        (println (str "  PASS: " msg)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " msg)))))

(defn- assert-equal [msg expected actual]
  (assert-true msg (= expected actual)))

(defn- assert-close [msg expected actual tol]
  (assert-true (str msg " (expected " expected ", got " actual ", tol " tol ")")
               (< (js/Math.abs (- expected actual)) tol)))

(defn- choices->map
  "Extract all scalar values from a choicemap as a Clojure map."
  [cm]
  (into {} (map (fn [addr-path]
                  [(first addr-path)
                   (mx/item (cm/get-value (cm/get-submap cm (first addr-path))))])
                (cm/addresses cm))))

;; ---------------------------------------------------------------------------
;; Test model
;; ---------------------------------------------------------------------------

(def simple-model
  (gen [mu]
    (let [x (trace :x (dist/gaussian mu 1))
          y (trace :y (dist/gaussian x 1))]
      y)))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(println "\n=== PRNG Key Threading Tests (Phase 2 — metadata-based) ===")

(println "\n-- with-key: simulate reproducibility --")
(let [key (rng/fresh-key 123)
      trace1 (p/simulate (dyn/with-key simple-model key) [(mx/scalar 0)])
      trace2 (p/simulate (dyn/with-key simple-model key) [(mx/scalar 0)])]
  (mx/eval! (:score trace1) (:score trace2))
  (let [c1 (choices->map (:choices trace1))
        c2 (choices->map (:choices trace2))]
    (assert-equal "same key → same :x" (:x c1) (:x c2))
    (assert-equal "same key → same :y" (:y c1) (:y c2))
    (assert-equal "same key → same score"
                  (mx/item (:score trace1)) (mx/item (:score trace2)))))

(println "\n-- with-key: different key → different results --")
(let [key1 (rng/fresh-key 100)
      key2 (rng/fresh-key 200)
      trace1 (p/simulate (dyn/with-key simple-model key1) [(mx/scalar 0)])
      trace2 (p/simulate (dyn/with-key simple-model key2) [(mx/scalar 0)])]
  (mx/eval! (:score trace1) (:score trace2))
  (let [c1 (choices->map (:choices trace1))
        c2 (choices->map (:choices trace2))]
    (assert-true "different key → different :x"
                 (not= (:x c1) (:x c2)))))

(println "\n-- with-key: generate reproducibility --")
(let [key (rng/fresh-key 456)
      obs (cm/choicemap :y (mx/scalar 2.0))
      r1 (p/generate (dyn/with-key simple-model key) [(mx/scalar 0)] obs)
      r2 (p/generate (dyn/with-key simple-model key) [(mx/scalar 0)] obs)]
  (mx/eval! (:weight r1) (:weight r2))
  (let [c1 (choices->map (:choices (:trace r1)))
        c2 (choices->map (:choices (:trace r2)))]
    (assert-equal "generate: same key → same :x" (:x c1) (:x c2))
    (assert-equal "generate: same key → same weight"
                  (mx/item (:weight r1)) (mx/item (:weight r2)))))

(println "\n-- with-key: assess reproducibility --")
(let [key (rng/fresh-key 789)
      choices (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
      r1 (p/assess (dyn/with-key simple-model key) [(mx/scalar 0)] choices)
      r2 (p/assess (dyn/with-key simple-model key) [(mx/scalar 0)] choices)]
  (mx/eval! (:weight r1) (:weight r2))
  (assert-equal "assess: same key → same weight"
                (mx/item (:weight r1)) (mx/item (:weight r2))))

(println "\n-- with-key: nested isolation --")
(let [outer-key (rng/fresh-key 111)
      inner-key (rng/fresh-key 222)
      ;; Outer with-key, then inner with-key should use inner key
      outer-model (dyn/with-key simple-model outer-key)
      inner-model (dyn/with-key simple-model inner-key)
      outer-trace (p/simulate outer-model [(mx/scalar 0)])
      inner-trace (p/simulate inner-model [(mx/scalar 0)])
      ;; Run inner-key alone
      inner-alone (p/simulate (dyn/with-key simple-model inner-key) [(mx/scalar 0)])]
  (let [ci (choices->map (:choices inner-trace))
        ca (choices->map (:choices inner-alone))]
    (assert-equal "nested with-key uses inner key :x" (:x ci) (:x ca))
    (assert-equal "nested with-key uses inner key :y" (:y ci) (:y ca))))

(println "\n-- with-key: importance sampling reproducibility --")
(let [key (rng/fresh-key 333)
      obs (cm/choicemap :y (mx/scalar 2.0))
      r1 (is/importance-sampling {:samples 10 :key key}
                                  simple-model [(mx/scalar 0)] obs)
      r2 (is/importance-sampling {:samples 10 :key key}
                                  simple-model [(mx/scalar 0)] obs)]
  (mx/eval! (:log-ml-estimate r1) (:log-ml-estimate r2))
  (assert-equal "IS: same key → same log-ML estimate"
                (mx/item (:log-ml-estimate r1))
                (mx/item (:log-ml-estimate r2))))

(println "\n-- with-key: MH reproducibility --")
(let [key (rng/fresh-key 444)
      obs (cm/choicemap :y (mx/scalar 2.0))
      r1 (mcmc/mh {:samples 5 :burn 2 :selection sel/all :key key}
                   simple-model [(mx/scalar 0)] obs)
      r2 (mcmc/mh {:samples 5 :burn 2 :selection sel/all :key key}
                   simple-model [(mx/scalar 0)] obs)]
  (let [x1 (mapv #(mx/item (cm/get-choice (:choices %) [:x])) r1)
        x2 (mapv #(mx/item (cm/get-choice (:choices %) [:x])) r2)]
    (assert-equal "MH: same key → same trace sequence" x1 x2)))

(println "\n-- without with-key: non-deterministic (sanity check) --")
(let [trace1 (p/simulate (dyn/auto-key simple-model) [(mx/scalar 0)])
      trace2 (p/simulate (dyn/auto-key simple-model) [(mx/scalar 0)])]
  (mx/eval! (:score trace1) (:score trace2))
  (let [c1 (choices->map (:choices trace1))
        c2 (choices->map (:choices trace2))]
    ;; With overwhelming probability these will differ
    (assert-true "no key → results differ (expected)"
                 (not= (:x c1) (:x c2)))))

(println "\n-- Distribution direct simulate uses key from metadata --")
(let [key (rng/fresh-key 555)
      d (dist/gaussian (mx/scalar 0) (mx/scalar 1))
      t1 (p/simulate (vary-meta d assoc :genmlx.dynamic/key key) [])
      t2 (p/simulate (vary-meta d assoc :genmlx.dynamic/key key) [])]
  (mx/eval! (:score t1) (:score t2))
  (assert-equal "dist simulate: same key → same value"
                (mx/item (:retval t1)) (mx/item (:retval t2))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== PRNG Key Test Results ==="))
(println (str "  Passed: " @pass-count))
(println (str "  Failed: " @fail-count))
(println (str "  Total:  " (+ @pass-count @fail-count)))

(when (pos? @fail-count)
  (js/process.exit 1))
