;; @tier slow
(ns genmlx.llm-composition-test
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
            [promesa.core :as pr]))

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
    (println "\n  -- generate with verdict fixed to :good --")
    (pr/let [g (st/generate m review-schema (:positive hyp->ids) {:verdict :good} opts)]
      (println "    " (pr-str (:value g)) " weight:" (.toFixed (:weight g) 3))
      (assert-true "conditioned value conforms" (:ok? g))
      (assert-true "fixed field honored" (= :good (:verdict (:value g))))
      (assert-true "conditioning weight finite" (js/isFinite (:weight g)))))

  ;; =========================================================
  ;; ct1r — schema-as-program-grammar synthesis
  ;; =========================================================
  (println "\n== ct1r: synthesize a valid mini-DSL program ==")
  (pr/let
   [expr-schema [:map [:op [:enum :+ :- :*]] [:a [:int {:min 0}]] [:b [:int {:min 0}]]]
    pr-raw (llm/encode tok "Emit one arithmetic op as EDN, e.g. {:op :+ :a 2 :b 3}\nEDN: ")
    prompt-ids (vec pr-raw)
    r (st/sample m expr-schema prompt-ids opts)]
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
    (pr/let [r2 (st/sample m expr-schema prompt-ids opts)
             r3 (st/sample m expr-schema prompt-ids opts)]
      (assert-true "all 3 syntheses are valid programs by construction"
                   (every? :ok? [r r2 r3]))))

  (println (str "\n=== composition: " @pass " PASS, " @fail " FAIL ===")))
