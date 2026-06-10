;; @tier fast
(ns genmlx.llm-pure-helpers-test
  "Model-free regressions for the pure llm helpers fixed in genmlx-xwxh:
   deterministic byte-trie token commit and vision label matching."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.llm.bytes :as bytes]
            [genmlx.llm.vision :as vision]))

(deftest commit-token-id-deterministic
  (testing "duplicate-text tokens resolve to the smallest id (genmlx-xwxh)"
    ;; pre-fix: (first set) — arbitrary id conditioned the KV cache, making
    ;; logits nondeterministic for identical byte choices
    (is (= 2 (bytes/commit-token-id {:token-ids #{9 5 2}})) "min of set")
    (is (= 7 (bytes/commit-token-id {:token-ids #{7}})) "singleton")
    (is (= 2 (bytes/commit-token-id {:token-ids (conj (sorted-set 9 5) 2)}))
        "independent of set ordering")))

(deftest label-matching-multi-word
  (testing "multi-word labels match (genmlx-xwxh)"
    (let [cell-types ["fire truck" "tree" "house"]]
      (is (= 0 (vision/label->index cell-types "fire truck")) "exact multi-word")
      (is (= 0 (vision/label->index cell-types "Fire Truck!")) "case+punct insensitive")
      (is (= 0 (vision/label->index cell-types "fire truck is here")) "answer-prefix match")
      (is (= 0 (vision/label->index cell-types "fire")) "terse answer prefix-matches label")
      (is (= 1 (vision/label->index cell-types "a tree")) "leading article: no exact match, but label is suffix"
          )
      (is (nil? (vision/label->index cell-types "boat")) "unknown stays nil")
      (is (nil? (vision/label->index cell-types "")) "empty answer stays nil"))))

(cljs.test/run-tests)
