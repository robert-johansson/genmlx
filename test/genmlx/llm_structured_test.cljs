;; @tier slow
(ns genmlx.llm-structured-test
  "Phase 2 flagship (genmlx-xi71): schema-typed structured generation as a GF.
   Validates sample (conforms), score (finite + matches the handler trace
   density), and generate (conditioning weight), on the fast 0.8b model."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.bytes :as bytes]
            [genmlx.llm.structured :as st]
            [genmlx.llm.schema-grammar :as sg]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [promesa.core :as pr]))

;; Seed every sampling site with a fixed PRNG key (gen-structured :key) so the
;; gate is DETERMINISTIC. The :int leaf is unbounded (schema_grammar int-regex
;; ignores :max), so an unseeded draw can emit a budget-filling digit run and
;; truncate the value mid-structure against :max-bytes — a flaky failure
;; unrelated to the forward. Pinned seeds draw clean, conforming values
;; (MLX RNG is bit-reproducible). score teacher-forces a given value, so it
;; needs no key.

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))
(defn assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol) (do (swap! pass inc) (println (str "  PASS: " label " (diff=" (.toFixed d 5) ")")))
        (do (swap! fail inc) (println (str "  FAIL: " label " expected=" expected " actual=" actual))))))

(def model-dir (str (.-HOME js/process.env) "/.cache/models/qwen3.5-0.8b-mlx-bf16"))

(pr/let
 [m         (llm/load-model model-dir)
  prep      (bytes/prepare (:tokenizer m))
  opts      {:trie (:trie prep) :max-bytes 64}
  ids-raw   (llm/encode (:tokenizer m)
                        "Reply with EDN only. Question: is the sky blue?\nAnswer: ")
  ids       (vec ids-raw)]

  ;; ---------------------------------------------------------
  (println "\n== enum schema ==")
  (pr/let [enum-s [:enum :yes :no :maybe]
           r (st/sample m enum-s ids (assoc opts :key (rng/fresh-key 5)))]
    (println "  sampled:" (pr-str (:value r)) "text:" (pr-str (:text r)))
    (assert-true "enum sample parses+validates" (:ok? r))
    (assert-true "enum value conforms to schema" (sg/validate enum-s (:value r)))
    (assert-true "enum value is one of the members" (contains? #{:yes :no :maybe} (:value r)))
    ;; score == trace density (handler-equivalence oracle)
    (pr/let [sc (st/score m enum-s ids (:value r) opts)]
      (println "  score logp:" (:logp sc) "trace density:" (mx/item (:score (:trace r))))
      (assert-true "enum score finite & negative" (and (js/isFinite (:logp sc)) (neg? (:logp sc))))
      (assert-close "enum score == trace density" (mx/item (:score (:trace r))) (:logp sc) 0.05)))

  ;; ---------------------------------------------------------
  (println "\n== map schema {:answer enum :score int} ==")
  (pr/let [map-s [:map [:answer [:enum :yes :no]] [:score :int]]
           ids2-raw (llm/encode (:tokenizer m)
                                "Reply with EDN only. Rate the sky's blueness.\nEDN: ")
           ids2 (vec ids2-raw)
           r (st/sample m map-s ids2 (assoc opts :max-bytes 96 :key (rng/fresh-key 7)))]
    (println "  sampled:" (pr-str (:value r)) "text:" (pr-str (:text r)))
    (assert-true "map sample parses+validates" (:ok? r))
    (assert-true "map has :answer key" (contains? (:value r) :answer))
    (assert-true "map has :score int" (int? (:score (:value r))))
    (assert-true "map :answer is enum member" (contains? #{:yes :no} (:answer (:value r))))
    (pr/let [sc (st/score m map-s ids2 (:value r) (assoc opts :max-bytes 96))]
      (assert-true "map score finite" (js/isFinite (:logp sc)))
      (assert-close "map score == trace density" (mx/item (:score (:trace r))) (:logp sc) 0.1))

    ;; conditioning: fix :answer, sample :score
    (println "\n== generate conditioning (fix :answer :yes) ==")
    (pr/let [g (st/generate m map-s ids2 {:answer :yes} (assoc opts :max-bytes 96 :key (rng/fresh-key 11)))]
      (println "  conditioned:" (pr-str (:value g)) "weight:" (:weight g)
               "base:" (:base-logp g) "cond:" (:cond-logp g))
      (assert-true "conditioned parses+validates" (:ok? g))
      (assert-true "fixed field honored (:answer = :yes)" (= :yes (:answer (:value g))))
      (assert-true "conditioning weight finite" (js/isFinite (:weight g)))
      (assert-true "weight <= 0 (log-evidence of fixed field)" (<= (:weight g) 1e-4))))

  ;; ---------------------------------------------------------
  (println "\n== vector schema [:vector :int] ==")
  (pr/let [vec-s [:vector {:max 4} :int]
           ids3-raw (llm/encode (:tokenizer m)
                                "Reply with EDN only. List up to 4 small numbers.\nEDN: ")
           ids3 (vec ids3-raw)
           r (st/sample m vec-s ids3 (assoc opts :max-bytes 64 :key (rng/fresh-key 3)))]
    (println "  sampled:" (pr-str (:value r)) "text:" (pr-str (:text r)))
    (assert-true "vector sample parses+validates" (:ok? r))
    (assert-true "vector of ints" (every? int? (:value r)))
    (assert-true "respects :max 4" (<= (count (:value r)) 4)))

  (println (str "\n=== structured: " @pass " PASS, " @fail " FAIL ===")))
