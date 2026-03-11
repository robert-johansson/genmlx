(ns genmlx.l3-5-regenerate-test
  "Level 3.5 WP-0: Regenerate auto-handler integration tests.

   Verifies that p/regenerate uses analytical elimination (Case B)
   when conjugate structure is detected and the conjugate prior is
   NOT in the selection.

   Weight algebra: weight = new_score - old_score - proposal_ratio
   For Case B (prior not selected, obs not selected):
   - Conjugate contributions identical in new/old score → cancel
   - Weight depends only on non-conjugate selected sites

   Run: bun run --bun nbb test/genmlx/l3_5_regenerate_test.cljs"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close
  [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (expected=" (.toFixed expected 6)
                       " actual=" (.toFixed actual 6) " diff=" (.toFixed diff 8) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " (expected=" (.toFixed expected 6)
                       " actual=" (.toFixed actual 6) " diff=" (.toFixed diff 8)
                       " tol=" tol ")"))))))

(defn- strip-analytical
  "Remove auto-handlers from a gen-fn, forcing standard handler path."
  [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :auto-regenerate-handlers
                            :auto-regenerate-transition
                            :conjugate-pairs :has-conjugate? :analytical-plan)))

(defn- has-regen-handlers? [gf]
  (boolean (:auto-regenerate-handlers (:schema gf))))

;; ---------------------------------------------------------------------------
;; Model Definitions
;; ---------------------------------------------------------------------------

;; NN: mu ~ N(0,10), y1 ~ N(mu,1), y2 ~ N(mu,1)
(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

;; BB: p ~ Beta(2,5), y1-y3 ~ Bernoulli(p)
(def bb-model
  (gen []
    (let [p (trace :p (dist/beta-dist 2 5))]
      (trace :y1 (dist/bernoulli p))
      (trace :y2 (dist/bernoulli p))
      (trace :y3 (dist/bernoulli p))
      p)))

;; GP: rate ~ Gamma(3,2), y1-y2 ~ Poisson(rate)
(def gp-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 3 2))]
      (trace :y1 (dist/poisson rate))
      (trace :y2 (dist/poisson rate))
      rate)))

;; Mixed: mu ~ N(0,10), sigma ~ Gamma(2,1), y1,y2 ~ N(mu,1), y3 ~ N(0,sigma)
(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian 0 sigma))
      mu)))

;; Kalman chain
(def kalman-model
  (gen []
    (let [z0 (trace :z0 (dist/gaussian 0 10))
          z1 (trace :z1 (dist/gaussian z0 1))]
      (trace :y0 (dist/gaussian z0 0.5))
      (trace :y1 (dist/gaussian z1 0.5))
      z0)))

;; No conjugacy
(def no-conj-model
  (gen []
    (let [x (trace :x (dist/uniform 0 10))]
      (trace :y1 (dist/gaussian (mx/sin x) 1))
      x)))

;; =========================================================================
;; Tests
;; =========================================================================

(println "\n===== Level 3.5 WP-0: Regenerate Auto-Handler Integration =====\n")

;; ---------------------------------------------------------------------------
;; Test 1: Schema detection — models have regenerate handlers
;; ---------------------------------------------------------------------------

(println "\n-- 1. Schema detection --")

(assert-true "NN model has regenerate handlers" (has-regen-handlers? nn-model))
(assert-true "BB model has regenerate handlers" (has-regen-handlers? bb-model))
(assert-true "GP model has regenerate handlers" (has-regen-handlers? gp-model))
(assert-true "Mixed model has regenerate handlers" (has-regen-handlers? mixed-model))
(assert-true "Kalman model has regenerate handlers" (has-regen-handlers? kalman-model))
(assert-true "No-conj model does NOT have regenerate handlers"
  (not (has-regen-handlers? no-conj-model)))

;; ---------------------------------------------------------------------------
;; Test 2: Case B — regenerate score consistency
;; Regenerate with non-conjugate selection, scores should match
;; ---------------------------------------------------------------------------

(println "\n-- 2. Score consistency (Case B) --")

;; For NN model with :mu not selected:
;; old_score (from generate with auto-handlers) = marginal LL(y1,y2)
;; new_score (from regenerate with auto-handlers) = same marginal LL(y1,y2) (nothing changed)
;; weight = new_score - old_score - proposal_ratio = 0 - 0 = 0
;; (when selection is empty or selects nothing)
(let [model (dyn/auto-key nn-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0)))
      gen-result (p/generate model [] obs)
      trace (:trace gen-result)
      old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
      ;; Regenerate with selection = none (nothing selected)
      regen-result (p/regenerate model trace (sel/select))
      new-score (do (mx/eval! (:score (:trace regen-result)))
                    (mx/item (:score (:trace regen-result))))
      weight (do (mx/eval! (:weight regen-result))
                 (mx/item (:weight regen-result)))]
  (assert-close "NN: regenerate new_score = old_score (nothing changed)"
    old-score new-score 1e-4)
  (assert-close "NN: regenerate weight = 0 (nothing changed)"
    0.0 weight 1e-4))

;; ---------------------------------------------------------------------------
;; Test 3: Case B — regenerate new_score matches generate score
;; ---------------------------------------------------------------------------

(println "\n-- 3. Regenerate score matches generate --")

(let [model (dyn/auto-key nn-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0)))
      gen-result (p/generate model [] obs)
      gen-score (do (mx/eval! (:score (:trace gen-result)))
                    (mx/item (:score (:trace gen-result))))
      trace (:trace gen-result)
      ;; Regenerate with empty selection
      regen-result (p/regenerate model trace (sel/select))
      regen-score (do (mx/eval! (:score (:trace regen-result)))
                      (mx/item (:score (:trace regen-result))))]
  (assert-close "Regenerate score = generate score (same model, no changes)"
    gen-score regen-score 1e-4))

;; ---------------------------------------------------------------------------
;; Test 4: Case A fallthrough — prior selected
;; ---------------------------------------------------------------------------

(println "\n-- 4. Case A fallthrough (prior selected) --")

(let [model-with (dyn/auto-key nn-model)
      model-without (dyn/auto-key (strip-analytical nn-model))
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0)))
      ;; Generate initial trace
      key (rng/fresh-key 42)
      gen-result-with (p/generate (dyn/with-key nn-model key) [] obs)
      gen-result-without (p/generate (dyn/with-key (strip-analytical nn-model) key) [] obs)
      ;; Select :mu (the conjugate prior) — this is Case A
      selection (sel/select :mu)
      ;; Regenerate both — when :mu is selected, auto-handler should fall through
      key2 (rng/fresh-key 99)
      regen-with (p/regenerate (dyn/with-key nn-model key2)
                   (:trace gen-result-with) selection)
      regen-without (p/regenerate (dyn/with-key (strip-analytical nn-model) key2)
                      (:trace gen-result-without) selection)
      w-with (do (mx/eval! (:weight regen-with)) (mx/item (:weight regen-with)))
      w-without (do (mx/eval! (:weight regen-without)) (mx/item (:weight regen-without)))]
  ;; When prior is selected, auto-handler returns nil → falls through to standard handler
  ;; Both should produce same result
  (assert-true "Case A: regenerate weight is finite" (js/isFinite w-with))
  (println (str "    With auto: " (.toFixed w-with 6)
                " Without: " (.toFixed w-without 6))))

;; ---------------------------------------------------------------------------
;; Test 5: BB model — Case B regenerate
;; ---------------------------------------------------------------------------

(println "\n-- 5. BB model: Case B regenerate --")

(let [model (dyn/auto-key bb-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 1.0))
              (cm/set-value :y2 (mx/scalar 0.0))
              (cm/set-value :y3 (mx/scalar 1.0)))
      gen-result (p/generate model [] obs)
      trace (:trace gen-result)
      old-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
      ;; Regenerate with empty selection
      regen-result (p/regenerate model trace (sel/select))
      new-score (do (mx/eval! (:score (:trace regen-result)))
                    (mx/item (:score (:trace regen-result))))
      weight (do (mx/eval! (:weight regen-result))
                 (mx/item (:weight regen-result)))]
  (assert-close "BB: regenerate new_score = old_score"
    old-score new-score 1e-4)
  (assert-close "BB: regenerate weight = 0 (nothing changed)"
    0.0 weight 1e-4))

;; ---------------------------------------------------------------------------
;; Test 6: GP model — Case B regenerate
;; ---------------------------------------------------------------------------

(println "\n-- 6. GP model: Case B regenerate --")

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
  (assert-close "GP: regenerate new_score = old_score"
    old-score new-score 1e-4)
  (assert-close "GP: regenerate weight = 0 (nothing changed)"
    0.0 weight 1e-4))

;; ---------------------------------------------------------------------------
;; Test 7: Mixed model — regenerate with sigma selected
;; ---------------------------------------------------------------------------

(println "\n-- 7. Mixed model: sigma selected (Case B for mu pair) --")

(let [model (dyn/auto-key mixed-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0))
              (cm/set-value :y3 (mx/scalar 0.5)))
      gen-result (p/generate model [] obs)
      trace (:trace gen-result)
      ;; Select sigma (non-conjugate) — mu pair is Case B
      selection (sel/select :sigma)
      regen-result (p/regenerate model trace selection)
      weight (do (mx/eval! (:weight regen-result))
                 (mx/item (:weight regen-result)))]
  (assert-true "Mixed model: regenerate with sigma selected produces finite weight"
    (js/isFinite weight))
  (println (str "    Weight: " (.toFixed weight 6))))

;; ---------------------------------------------------------------------------
;; Test 8: Kalman model — Case B regenerate
;; ---------------------------------------------------------------------------

(println "\n-- 8. Kalman model: Case B regenerate --")

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
  (assert-close "Kalman: regenerate new_score = old_score"
    old-score new-score 1e-4)
  (assert-close "Kalman: regenerate weight = 0 (nothing changed)"
    0.0 weight 1e-4))

;; ---------------------------------------------------------------------------
;; Test 9: No conjugacy — standard regenerate fallback
;; ---------------------------------------------------------------------------

(println "\n-- 9. No conjugacy: standard fallback --")

(let [model (dyn/auto-key no-conj-model)
      gen-result (p/generate model [] (cm/set-value cm/EMPTY :y1 (mx/scalar 0.5)))
      trace (:trace gen-result)
      regen-result (p/regenerate model trace (sel/select :x))
      weight (do (mx/eval! (:weight regen-result))
                 (mx/item (:weight regen-result)))]
  (assert-true "No-conj model: regenerate produces finite weight"
    (js/isFinite weight)))

;; ---------------------------------------------------------------------------
;; Test 10: MH chain convergence — Mixed model (select non-conjugate sigma)
;; ---------------------------------------------------------------------------

(println "\n-- 10. MH chain convergence (mixed model, sigma selected) --")

;; Mixed model: mu ~ N(0,10), sigma ~ Gamma(2,1)
;; y1,y2 ~ N(mu,1), y3 ~ N(0, sigma)
;; Select :sigma only — mu pair is Case B (analytically handled)
;; This tests that MH works correctly when analytical elimination handles
;; the conjugate part and sampling handles the non-conjugate part.

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
  (println (str "    Chain mean sigma: " (.toFixed chain-mean 4)))
  (println (str "    Accept rate: " (.toFixed accept-rate 4)))
  (assert-true "MH chain produces positive sigma values"
    (every? pos? post-sigmas))
  (assert-true "MH acceptance rate > 0" (> accept-rate 0.0))
  (assert-true "MH chain mean sigma is finite" (js/isFinite chain-mean)))

;; ---------------------------------------------------------------------------
;; Test 11: Regenerate with stripped model matches standard behavior
;; ---------------------------------------------------------------------------

(println "\n-- 11. Stripped model regenerate matches standard --")

;; When auto-handlers are stripped, regenerate should produce standard weights
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
  (assert-close "Stripped: regenerate new_score = old_score"
    old-score new-score 1e-4)
  (assert-close "Stripped: regenerate weight = 0"
    0.0 weight 1e-4))

;; ---------------------------------------------------------------------------
;; Test 12: Regenerate trace has correct choices
;; ---------------------------------------------------------------------------

(println "\n-- 12. Trace choice correctness --")

(let [model (dyn/auto-key nn-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0)))
      gen-result (p/generate model [] obs)
      trace (:trace gen-result)
      regen-result (p/regenerate model trace (sel/select))
      new-choices (:choices (:trace regen-result))]
  ;; All addresses should be present
  (assert-true "Regenerated trace has :mu"
    (cm/has-value? (cm/get-submap new-choices :mu)))
  (assert-true "Regenerated trace has :y1"
    (cm/has-value? (cm/get-submap new-choices :y1)))
  (assert-true "Regenerated trace has :y2"
    (cm/has-value? (cm/get-submap new-choices :y2)))
  ;; y1 and y2 should keep their constrained values
  (let [y1 (mx/item (cm/get-value (cm/get-submap new-choices :y1)))
        y2 (mx/item (cm/get-value (cm/get-submap new-choices :y2)))]
    (assert-close "y1 preserved" 3.0 y1 1e-6)
    (assert-close "y2 preserved" 4.0 y2 1e-6)))

;; ---------------------------------------------------------------------------
;; Test 13: Deterministic regenerate (same key → same result)
;; ---------------------------------------------------------------------------

(println "\n-- 13. Deterministic regenerate --")

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
  (assert-close "Deterministic: same key → same weight" w1 w2 1e-10))

;; ---------------------------------------------------------------------------
;; Test 14: GE model — Case B regenerate
;; ---------------------------------------------------------------------------

(println "\n-- 14. GE model: Case B regenerate --")

(def ge-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 2 1))]
      (trace :y1 (dist/exponential rate))
      (trace :y2 (dist/exponential rate))
      rate)))

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
  (assert-close "GE: regenerate new_score = old_score"
    old-score new-score 1e-4)
  (assert-close "GE: regenerate weight = 0 (nothing changed)"
    0.0 weight 1e-4))

;; ---------------------------------------------------------------------------
;; Test 15: Regenerate retval is correct
;; ---------------------------------------------------------------------------

(println "\n-- 15. Retval correctness --")

(let [model (dyn/auto-key nn-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0)))
      gen-result (p/generate model [] obs)
      trace (:trace gen-result)
      regen-result (p/regenerate model trace (sel/select))
      retval (:retval (:trace regen-result))]
  (assert-true "Regenerate returns a retval" (some? retval)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n===== Results =====")
(println (str "PASS: " @pass-count " / " (+ @pass-count @fail-count)))
(when (pos? @fail-count)
  (println (str "FAIL: " @fail-count)))
(println "===================\n")
