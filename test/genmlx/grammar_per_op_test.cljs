;; @tier fast
(ns genmlx.grammar-per-op-test
  "Model-free regressions for the genmlx-xwxh grammar bundle: constrain
   runs each GFI op under its OWN transition (update/regenerate previously
   ran generate semantics), apply-mask guards (vocab>logits truncation,
   :dead state throw), and EOS-then-continue no longer NaNs.

   Uses a toy 4-'token' vocab [a b c <eos>] and hand-built constraint maps —
   no LLM, no tokenizer; constrain works on any GF using dist/categorical."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.llm.grammar :as gram]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist])
  (:require-macros [genmlx.gen :refer [gen]]))

(def token-index ["a" "b" "c" "<eos>"])
(def eos-id 3)

(defn make-constraint [regex]
  {:dfa (gram/compile-regex regex)
   :token-index token-index
   :eos-id eos-id
   :masks nil})

(def uniform-logits (mx/array [0.0 0.0 0.0 0.0]))

;; 2 token sites, regex a[bc]: t0 must be a(0); t1 is b(1) or c(2)
(def gf2
  (dyn/auto-key
    (gen []
      (trace :t0 (dist/categorical uniform-logits))
      (trace :t1 (dist/categorical uniform-logits)))))

(def abc-constraint (make-constraint "a[bc]"))
(def gf2c (gram/constrain gf2 abc-constraint))

(defn choice [trace addr] (int (mx/item (cm/get-choice (:choices trace) [addr]))))

(deftest simulate-respects-grammar
  (testing "simulate masks every site by the DFA"
    (dotimes [_ 5]
      (let [tr (p/simulate gf2c [])]
        (is (= 0 (choice tr :t0)) "t0 forced to a")
        (is (contains? #{1 2} (choice tr :t1)) "t1 in [bc]")))))

(deftest update-runs-update-semantics
  (testing "p/update on a constrained GF preserves old choices and discards (genmlx-xwxh)"
    (let [tr (p/simulate gf2c [])
          old-t1 (choice tr :t1)
          new-t1 (if (= old-t1 1) 2 1)
          {:keys [trace discard weight]} (p/update gf2c tr (cm/choicemap :t1 (mx/scalar new-t1 mx/int32)))]
      (is (= 0 (choice trace :t0)) "unconstrained t0 replayed (not regenerated)")
      (is (= new-t1 (choice trace :t1)) "constrained t1 updated")
      ;; pre-fix: generate semantics -> no discard of the old value
      (is (= old-t1 (int (mx/item (cm/get-choice discard [:t1]))))
          "discard holds the OLD t1 (update semantics, not generate)")
      (is (js/isFinite (mx/item weight)) "finite update weight"))))

(deftest regenerate-runs-regenerate-semantics
  (testing "p/regenerate resamples only the selection (genmlx-xwxh)"
    (let [tr (p/simulate gf2c [])
          t0 (choice tr :t0)
          {:keys [trace weight]} (p/regenerate gf2c tr (sel/select :t1))]
      (is (= t0 (choice trace :t0)) "unselected t0 kept")
      (is (contains? #{1 2} (choice trace :t1)) "resampled t1 stays in grammar")
      (is (js/isFinite (mx/item weight)) "finite regenerate weight"))))

(deftest assess-and-generate-agree
  (testing "p/assess scores grammar-masked choices finitely"
    (let [choices (cm/choicemap :t0 (mx/scalar 0 mx/int32)
                                :t1 (mx/scalar 1 mx/int32))
          {:keys [weight]} (p/assess gf2c [] choices)]
      (is (js/isFinite (mx/item weight)) "finite assess weight")
      ;; both sites masked: t0 from {a}, t1 from {b,c} -> log(1) + log(1/2)
      (is (< (Math/abs (- (Math/log 0.5) (mx/item weight))) 1e-4)
          "assess weight = grammar-conditioned log-prob"))))

(deftest eos-then-continue-no-nan
  (testing "EOS does not advance the DFA into :dead; later sites stay finite (genmlx-xwxh)"
    ;; regex ab: t0=a, t1=b, then accept -> t2 must be EOS; pre-fix the DFA
    ;; advanced through '<eos>' text to :dead and t3's all--inf mask sampled NaN
    (let [gf4 (dyn/auto-key
                (gen []
                  (trace :t0 (dist/categorical uniform-logits))
                  (trace :t1 (dist/categorical uniform-logits))
                  (trace :t2 (dist/categorical uniform-logits))
                  (trace :t3 (dist/categorical uniform-logits))))
          gf4c (gram/constrain gf4 (make-constraint "ab"))
          tr (p/simulate gf4c [])]
      (is (= [0 1 eos-id eos-id] (mapv #(choice tr %) [:t0 :t1 :t2 :t3]))
          "a, b, then EOS repeats — never NaN")
      (is (js/isFinite (mx/item (:score tr))) "finite score"))))

(deftest apply-mask-guards
  (testing "vocab > logits dim truncates instead of RangeError (genmlx-xwxh)"
    (let [big-constraint {:dfa (gram/compile-regex "[abcdef]")
                          :token-index ["a" "b" "c" "d" "e" "f"]
                          :eos-id 5
                          :masks nil}
          ;; logits dim 4 < token-index 6
          masked (gram/apply-mask big-constraint
                                  (get-in big-constraint [:dfa :start])
                                  uniform-logits)]
      (is (= [4] (vec (mx/shape masked))) "masked logits keep logits dim")
      (is (every? js/isFinite (vec (mx/->clj (mx/exp masked))))
          "probabilities well-defined")))
  (testing ":dead DFA state throws actionably instead of NaN sampling (genmlx-xwxh)"
    (is (thrown-with-msg? js/Error #"dead"
                          (gram/apply-mask abc-constraint :dead uniform-logits)))))

(cljs.test/run-tests)
