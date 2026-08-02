;; @tier slow
(ns genmlx.llm.composition-test
  "Phase 2 flagship payoff (genmlx-t3z5 + genmlx-ct1r): the schema-typed LLM GF
   used compositionally.

   t3z5 — LLM structured output as a SCORED likelihood inside Bayesian
          inference: a latent is inferred from an observed structured value by
          exact enumeration, using st/score as the per-hypothesis likelihood.
   ct1r — schema-as-program-grammar: a malli schema describes a tiny DSL;
          structured generation yields only valid 'programs', which we evaluate."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.bytes :as bytes]
            [genmlx.llm.structured :as st]
            [genmlx.llm.schema-grammar :as sg]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [promesa.core :as pr]
            ["fs" :as fs]))

;; Every sampling site is SEEDED (gen-structured :key), the convention
;; structured_test.cljs already established and this file was missed by
;; (genmlx-bytb). Rationale, unchanged from there: the :int leaf is unbounded
;; (schema_grammar's int-regex ignores :max), so an unseeded draw can run away
;; emitting digits and truncate mid-structure against :max-bytes. MLX's RNG is
;; bit-reproducible, so a pinned key makes each draw a fixed input rather than
;; a coin flip. st/score teacher-forces a given value and needs no key.

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

(defn- logsumexp [xs]
  (let [m (apply max xs)]
    (+ m (js/Math.log (reduce + (map #(js/Math.exp (- % m)) xs))))))

(defn- normalize-logweights [m]
  (let [z (logsumexp (vals m))]
    (into {} (map (fn [[k v]] [k (js/Math.exp (- v z))]) m))))

(def model-dir (str (.-HOME js/process.env) "/.cache/models/qwen3.5-0.8b-mlx-bf16"))

(if-not (.existsSync fs (str model-dir "/model.safetensors"))
  (println "SKIP llm-composition-test: qwen3.5-0.8b checkpoint absent at" model-dir)
  (pr/let
   [m    (llm/load-model model-dir)
    tok  (:tokenizer m)
    prep (bytes/prepare tok)
    opts {:trie (:trie prep) :max-bytes 64}]

    ;; =========================================================
    ;; t3z5 — LLM structured output as a scored likelihood
    ;; =========================================================
    (println "\n== t3z5: infer latent sentiment from a structured review ==")
    (pr/let
     [review-schema [:map [:rating [:int {:min 0}]] [:verdict [:enum :good :bad]]]
      ;; prompts conditioning the LLM on each hypothesis
      pos-raw (llm/encode tok "The movie was wonderful, a masterpiece. Review as EDN: ")
      neg-raw (llm/encode tok "The movie was terrible, a total waste. Review as EDN: ")
      hyp->ids {:positive (vec pos-raw) :negative (vec neg-raw)}
      ;; observed structured value (a clearly-positive review)
      observed {:rating 9 :verdict :good}
      prior {:positive (js/Math.log 0.5) :negative (js/Math.log 0.5)}
      ;; exact enumeration: log joint = log prior + structured-LLM log-likelihood
      joint (pr/let [lp (st/score m review-schema (:positive hyp->ids) observed opts)
                     ln (st/score m review-schema (:negative hyp->ids) observed opts)]
              {:positive (+ (:positive prior) (:logp lp))
               :negative (+ (:negative prior) (:logp ln))})
      posterior (normalize-logweights joint)]
      (println "  observed:" (pr-str observed))
      (println "  log-likelihoods:" (pr-str (zipmap (keys joint) (map #(.toFixed % 3) (vals joint)))))
      (println "  posterior:" (pr-str posterior))
      (assert-true "both likelihoods finite" (every? js/isFinite (vals joint)))
      (assert-true "posterior normalizes to 1"
                   (< (js/Math.abs (- 1.0 (reduce + (vals posterior)))) 1e-6))
      (assert-true "posterior is a valid distribution" (every? #(<= 0 % 1) (vals posterior)))
      ;; soft semantic check (informational — 0.8b): positive should be favored
      (println (str "  [info] argmax sentiment = "
                    (key (apply max-key val posterior))
                    (if (= :positive (key (apply max-key val posterior))) "  ✓ matches observed" "")))

      ;; conditioning composition: fix the verdict, let the model fill the rating
      ;;
      ;; This block used to be a single UNSEEDED semantic claim -- "the model's
      ;; free-form rating happens to conform" -- which is a coin flip, not a
      ;; test (genmlx-bytb: measured flaky in BOTH arms of a powered A/B, 2/17
      ;; and 3/14 failures). Seeding alone would not have fixed it either: it
      ;; would just have frozen one lucky draw. So the per-draw assertions below
      ;; are the ones that hold for EVERY seed by construction, and the semantic
      ;; conformance rate is pinned separately as a measured number.
      ;;
      ;; Measured 2026-08-02 (qwen3.5-0.8b, sm_120), seeds 1..20: 15/20 conform.
      ;; The 5 misses are NOT grammar violations: seeds 2,3,10,18 fill the byte
      ;; budget with a digit run, and seed 11 emits the well-formed but
      ;; JS-out-of-range {:rating 316166762481384775667719 :verdict :good}.
      ;; Raising :max-bytes does not help -- 64 and 128 gave identical outcomes
      ;; on all 20 seeds, because the int regex is unbounded.
      (println "\n  -- generate with verdict fixed to :good (seeded) --")
      (let [seeds [1 2 3 4 5 6]
            gs (mapv (fn [s]
                       (st/generate m review-schema (:positive hyp->ids) {:verdict :good}
                                    (assoc opts :key (rng/fresh-key s))))
                     seeds)
            n-ok (count (filter :ok? gs))
            capped? (fn [g] (>= (count (:text g)) (:max-bytes opts)))
            ;; the full conditioned language: rating digits, verdict pinned
            well-formed? (fn [g] (some? (re-matches #"\{:rating \d+ :verdict :good\}" (:text g))))]
        (doseq [[s g] (map vector seeds gs)]
          (println (str "     seed " s "  ok=" (boolean (:ok? g))
                        "  weight=" (.toFixed (:weight g) 3) "  " (pr-str (:text g)))))
        (println (str "     conformance " n-ok "/" (count seeds)
                      "  (measured 15/20 over seeds 1..20 on 2026-08-02)"))

        ;; (1) determinism -- the direct anti-regression for the unseeded defect
        (let [again (st/generate m review-schema (:positive hyp->ids) {:verdict :good}
                                 (assoc opts :key (rng/fresh-key 1)))]
          (assert-true "seeded generation is reproducible (same key -> same text)"
                       (= (:text (first gs)) (:text again))))

        ;; (2) conditioning is honored on EVERY draw: the fixed field can never
        ;; take the other enum value, whether or not the draw completed.
        (assert-true "no draw ever emits the excluded :verdict :bad"
                     (not-any? #(re-find #":verdict :bad" (:text %)) gs))

        ;; (3) every draw is ON-GRAMMAR: a non-conforming draw may only be a
        ;; byte-budget truncation, never an illegal string. This is what the
        ;; constrained decoder actually promises.
        (assert-true "every draw is on-grammar (conforming, well-formed, or capped)"
                     (every? #(or (:ok? %) (well-formed? %) (capped? %)) gs))

        ;; (4) the importance weight is always a real log-evidence
        (assert-true "conditioning weight finite on every draw"
                     (every? #(js/isFinite (:weight %)) gs))
        (assert-true "conditioning weight <= 0 on every draw (log-evidence)"
                     (every? #(<= (:weight %) 1e-4) gs))

        ;; (5) the semantic rate, pinned as a band rather than a coin flip. A red
        ;; here is not automatically a regression -- it means the model's sampled
        ;; outputs moved (different arch, different checkpoint) and the number
        ;; above needs re-measuring. It is deterministic on a fixed arch.
        (assert-true (str "conformance rate >= 3/6 on the pinned seeds (got " n-ok "/6)")
                     (>= n-ok 3))

        ;; (6) the original semantic claim, now scoped to the draws that parsed
        (assert-true "every conforming draw honors the fixed field"
                     (every? #(= :good (:verdict (:value %))) (filter :ok? gs)))))

    ;; =========================================================
    ;; ct1r — schema-as-program-grammar synthesis
    ;; =========================================================
    (println "\n== ct1r: synthesize a valid mini-DSL program ==")
    (pr/let
     [expr-schema [:map [:op [:enum :+ :- :*]] [:a [:int {:min 0}]] [:b [:int {:min 0}]]]
      pr-raw (llm/encode tok "Emit one arithmetic op as EDN, e.g. {:op :+ :a 2 :b 3}\nEDN: ")
      prompt-ids (vec pr-raw)
      ;; seeded for the same reason as t3z5 above -- two unbounded :int leaves
      ;; here, so an unseeded draw carries the same digit-runaway risk. Measured
      ;; 2026-08-02: 12/12 conformance over seeds 1..12, deterministic per seed.
      r (st/sample m expr-schema prompt-ids (assoc opts :key (rng/fresh-key 1)))]
      (println "  synthesized program:" (pr-str (:value r)) "text:" (pr-str (:text r)))
      (assert-true "program parses+validates against the DSL schema" (:ok? r))
      (assert-true "program is well-formed (op + two int operands)"
                   (and (contains? #{:+ :- :*} (:op (:value r)))
                        (int? (:a (:value r))) (int? (:b (:value r)))))
      ;; evaluate the synthesized program — synthesis -> execution
      (when (:ok? r)
        (let [{:keys [op a b]} (:value r)
              f ({:+ + :- - :* *} op)
              result (f a b)]
          (println "  evaluated:" a (name op) b "=" result)
          (assert-true "synthesized program is executable" (number? result))))
      ;; every sample is a valid program by construction
      (pr/let [r2 (st/sample m expr-schema prompt-ids (assoc opts :key (rng/fresh-key 2)))
               r3 (st/sample m expr-schema prompt-ids (assoc opts :key (rng/fresh-key 3)))]
        (assert-true "all 3 syntheses are valid programs by construction"
                     (every? :ok? [r r2 r3]))))

    (println (str "\n=== composition: " @pass " PASS, " @fail " FAIL ==="))
    (when (pos? @fail) (set! (.-exitCode js/process) 1))))
