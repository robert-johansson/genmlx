;; @tier fast
(ns genmlx.llm-batched-mask-test
  "genmlx-9uyg fast contracts: the masked-EOS algebra that lets K LLM lanes
   run through vsimulate with ZERO handler changes, and the batched-DFA
   grammar tables — all checkpoint-free.

   The load-bearing identity (spec D2): an inactive lane's logits are
   replaced by a one-hot pad row (0 at pad, -1e9 elsewhere), so
     log_softmax(pad-row)[pad] = -log(1 + (V-1)·e^{-1e9}) = 0 exactly in f32,
   the Gumbel-max sample is deterministically pad, and the lane's score
   freezes — the handler keeps adding site lps unconditionally, they're just
   all zero after death.

   Also pins the L1 law on a TOY model (fixed categorical logits, no LLM):
   per-lane vsimulate score == scalar assess of the lane's tokens truncated
   at EOS. This is the same law the slow-tier real-model test asserts."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.llm.core :as llmc]
            [genmlx.llm.grammar :as gr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private V 4)
(def ^:private EOS 2)
(def ^:private PAD 3)
(def ^:private step-logits (mx/array [0.5 1.0 0.3 -2.0] [V]))
(def ^:private pad-row (llmc/pad-onehot-row V PAD))

(defn- mask->vec [a] (vec (mx/->clj (mx/astype a mx/float32))))

;; ---------------------------------------------------------------------------
;; pad-onehot-row / mask-inactive-logits / advance-active
;; ---------------------------------------------------------------------------

(deftest pad-row-freezes-logprob-exactly
  (let [lp (dc/dist-log-prob (dist/categorical pad-row) (mx/scalar PAD mx/int32))]
    (is (zero? (mx/item lp)) "lp(pad | pad-row) == 0 exactly in f32"))
  (let [lp0 (dc/dist-log-prob (dist/categorical pad-row) (mx/scalar 0 mx/int32))]
    (is (< (mx/item lp0) -1e8) "non-pad tokens are unsampleable")))

(deftest mask-inactive-selects-per-lane
  ;; active [K], shared logits [V] -> [K V]: active lanes keep the real row,
  ;; inactive lanes get the pad row.
  (let [active (mx/equal (mx/array [1 0 1] [3]) 1)
        out    (llmc/mask-inactive-logits step-logits active pad-row)]
    (is (= [3 V] (mx/shape out)) "broadcasts [V] against active [K] to [K V]")
    (is (zero? (mx/item (mx/amax (mx/abs (mx/subtract (mx/index out 0) step-logits)))))
        "active lane row unchanged")
    (is (zero? (mx/item (mx/amax (mx/abs (mx/subtract (mx/index out 1) pad-row)))))
        "inactive lane row is the pad row"))
  ;; per-lane logits [K V]
  (let [active (mx/equal (mx/array [0 1] [2]) 1)
        lg     (mx/array [1.0 2.0 3.0 4.0, 5.0 6.0 7.0 8.0] [2 V])
        out    (llmc/mask-inactive-logits lg active pad-row)]
    (is (zero? (mx/item (mx/amax (mx/abs (mx/subtract (mx/index out 0) pad-row)))))
        "inactive lane of [K V] logits replaced")
    (is (zero? (mx/item (mx/amax (mx/abs (mx/subtract (mx/index out 1) (mx/index lg 1))))))
        "active lane of [K V] logits kept")))

(deftest inactive-lane-samples-are-deterministic-pad
  (let [active (mx/equal (mx/array [1 0 1 0] [4]) 1)
        masked (llmc/mask-inactive-logits step-logits active pad-row)]
    (doseq [seed [1 2 3]]
      (let [s (mask->vec (dc/dist-sample-n (dist/categorical masked)
                                           (h/deterministic-key seed) 4))]
        (is (= (double PAD) (nth s 1)) (str "seed " seed ": dead lane 1 samples pad"))
        (is (= (double PAD) (nth s 3)) (str "seed " seed ": dead lane 3 samples pad"))))))

(deftest advance-active-is-monotone
  (let [tok1 (mx/array [EOS 0 1] [3])
        a1   (llmc/advance-active nil tok1 EOS)
        tok2 (mx/array [0 EOS 1] [3])
        a2   (llmc/advance-active a1 tok2 EOS)
        tok3 (mx/array [0 0 1] [3])
        a3   (llmc/advance-active a2 tok3 EOS)]
    (is (= [0.0 1.0 1.0] (mask->vec a1)) "nil init: lane dies on its first eos")
    (is (= [0.0 0.0 1.0] (mask->vec a2)) "second eos kills lane 1; lane 0 stays dead")
    (is (= [0.0 0.0 1.0] (mask->vec a3)) "death is permanent (monotone)")))

;; ---------------------------------------------------------------------------
;; L1 law on a toy model (no LLM): vsimulate lane score == scalar assess
;; ---------------------------------------------------------------------------

(def ^:private t-addr #(keyword (str "t" %)))
(def ^:private N-SITES 3)

(defn- toy-batched
  "The masked-EOS decode pattern over fixed logits — the same loop shape as
   make-llm-gf-batched minus the transformer."
  [logits]
  (gen []
    (loop [i 0 active nil toks []]
      (if (= i N-SITES)
        toks
        (let [lg  (if active (llmc/mask-inactive-logits logits active pad-row) logits)
              tok (trace (t-addr i) (dist/categorical lg))]
          (recur (inc i) (llmc/advance-active active tok EOS) (conj toks tok)))))))

(defn- toy-scalar
  "Scalar early-exit equivalent (make-llm-gf semantics)."
  [logits]
  (gen []
    (loop [i 0 toks []]
      (if (= i N-SITES)
        toks
        (let [tok (trace (t-addr i) (dist/categorical logits))]
          (if (= EOS (mx/item tok))
            (conj toks tok)
            (recur (inc i) (conj toks tok))))))))

(defn- lane-tokens
  "Host [K][site] token matrix from a vtrace's :t0..:tN sites."
  [vtrace k]
  (let [cols (mapv (fn [i]
                     (mapv int (mx/->clj (mx/astype
                                          (cm/get-value (cm/get-submap (:choices vtrace)
                                                                       (t-addr i)))
                                          mx/int32))))
                   (range N-SITES))]
    (mapv (fn [l] (mapv #(nth % l) cols)) (range k))))

(defn- truncate-at-eos [toks]
  (let [idx (.indexOf (clj->js toks) EOS)]
    (if (neg? idx) toks (subvec toks 0 (inc idx)))))

(defn- assess-weight [gf toks]
  (let [choices (reduce (fn [c [i t]]
                          (cm/set-value c (t-addr i) (mx/scalar t mx/int32)))
                        cm/EMPTY (map-indexed vector toks))]
    (mx/item (:weight (p/assess (dyn/auto-key gf) [] choices)))))

(deftest toy-l1-law-lane-score-equals-scalar-assess
  (doseq [k [1 16]]
    (testing (str "K=" k)
      (let [vt     (dyn/vsimulate (toy-batched step-logits) [] k
                                  (h/deterministic-key (+ 100 k)))
            scores (vec (mx/->clj (mx/astype (:score vt) mx/float32)))
            lanes  (lane-tokens vt k)]
        (is (= k (count scores)) "score is [K]")
        (doseq [l (range k)]
          (let [toks (truncate-at-eos (nth lanes l))
                w    (assess-weight (toy-scalar step-logits) toks)]
            (is (h/close? w (nth scores l) 1e-5)
                (str "lane " l ": vsimulate score == scalar assess of " toks))))
        ;; post-eos sites are pad
        (doseq [l (range k)]
          (let [toks (nth lanes l)
                cut  (count (truncate-at-eos toks))]
            (is (every? #(= PAD %) (drop cut toks))
                (str "lane " l ": every post-eos site is pad"))))))))

(deftest toy-all-eos-at-t0
  ;; logits force EOS at the first site: every lane dies immediately; score
  ;; is lp(eos) ~= 0 (one-hot) and every later site is pad with lp 0.
  (let [eos-row (llmc/pad-onehot-row V EOS)
        vt      (dyn/vsimulate (toy-batched eos-row) [] 8 (h/deterministic-key 7))
        scores  (vec (mx/->clj (mx/astype (:score vt) mx/float32)))
        lanes   (lane-tokens vt 8)]
    (is (every? #(< (js/Math.abs %) 1e-6) scores) "all-eos-at-t0 lanes freeze at score 0")
    (is (every? #(= [EOS PAD PAD] %) lanes) "sites are [eos pad pad] in every lane")))

;; ---------------------------------------------------------------------------
;; Batched-DFA grammar tables (synthetic token-index, no tokenizer)
;; ---------------------------------------------------------------------------

(def ^:private token-index ["a" "b" "c" "ab" "<eos>"])
(def ^:private GV (count token-index))
(def ^:private G-EOS 4)
(def ^:private fake-tokenizer #js {:getEosTokenId (fn [] G-EOS)})

(def ^:private constraint
  (gr/compile-constraint fake-tokenizer "a(b|c)" {:token-index token-index}))

(def ^:private vt (gr/build-vtables constraint GV))

(defn- row->vec [table r] (vec (mx/->clj (mx/astype (mx/index table r) mx/float32))))

(defn- state-row
  "Dense row index of a DFA state reached by string s from start."
  [s]
  (let [{:keys [dfa]} constraint
        st (gr/dfa-advance-string dfa (:start dfa) s)]
    (get (zipmap (:states vt) (range)) st)))

(deftest vtables-mask-rows-match-dfa
  (let [start-row (:start-row vt)
        mrow (row->vec (:mask-table vt) start-row)]
    (is (zero? (nth mrow 0)) "start: token \"a\" valid")
    (is (zero? (nth mrow 3)) "start: token \"ab\" valid (reaches accept)")
    (is (neg? (nth mrow 1)) "start: token \"b\" invalid")
    (is (neg? (nth mrow G-EOS)) "start: eos invalid (not an accept state)"))
  (let [arow (row->vec (:mask-table vt) (state-row "a"))]
    (is (zero? (nth arow 1)) "after-a: \"b\" valid")
    (is (zero? (nth arow 2)) "after-a: \"c\" valid")
    (is (neg? (nth arow 0)) "after-a: \"a\" invalid"))
  (let [acc (row->vec (:mask-table vt) (state-row "ab"))]
    (is (zero? (nth acc G-EOS)) "accept: eos valid")
    (is (every? neg? (map #(nth acc %) [0 1 2 3])) "accept: no continuation tokens")))

(deftest vtables-transitions-match-dfa
  (let [start-row (:start-row vt)
        trow (mapv int (row->vec (:trans-table vt) start-row))]
    (is (= (state-row "a") (nth trow 0)) "start --\"a\"--> after-a row")
    (is (= (state-row "ab") (nth trow 3)) "start --\"ab\"--> accept row")
    (is (= (:dead-row vt) (nth trow 1)) "invalid token routes to the dead row")
    (is (= start-row (nth trow G-EOS)) "eos self-loops (wrap-grammar semantics)"))
  (let [drow (mapv int (row->vec (:trans-table vt) (:dead-row vt)))]
    (is (every? #(= (:dead-row vt) %) drow) "dead row absorbs")))

(deftest vtables-dead-row-is-nan-safe
  (let [mrow (mx/index (:mask-table vt) (:dead-row vt))
        lp   (dc/dist-log-prob (dist/categorical mrow) (mx/scalar G-EOS mx/int32))]
    (is (h/finite? (mx/item lp)) "a lane parked on the dead row cannot NaN the score")))

(deftest vectorized-hook-drives-lanes-independently
  (let [hook (gr/vectorized-hook vt)
        st0  ((:init hook))
        ;; step 1: lanes sample "a" / "ab" from start
        st1  ((:advance hook) st0 (mx/array [0 3] [2]))
        ;; step 2: lane 0 samples "b" (-> accept); lane 1 holds via eos self-loop
        st2  ((:advance hook) st1 (mx/array [1 G-EOS] [2]))
        rows (fn [st] (mapv int (mx/->clj (mx/astype st mx/int32))))]
    (is (= [(state-row "a") (state-row "ab")] (rows st1)) "per-lane states after step 1")
    (is (= [(state-row "ab") (state-row "ab")] (rows st2))
        "lane 0 advanced to accept; lane 1 held by eos self-loop")
    ;; masks gather per-lane rows
    (let [masked ((:mask hook) st1 (mx/zeros [2 GV]) 1)]
      (is (zero? (mx/item (mx/index (mx/index masked 0) 1))) "lane 0 (after-a) allows \"b\"")
      (is (neg? (mx/item (mx/index (mx/index masked 1) 0))) "lane 1 (accept) blocks \"a\"")
      (is (zero? (mx/item (mx/index (mx/index masked 1) G-EOS))) "lane 1 (accept) allows eos"))))

(cljs.test/run-tests)
