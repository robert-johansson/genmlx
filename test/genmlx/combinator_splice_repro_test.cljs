;; @tier fast core
(ns genmlx.combinator-splice-repro-test
  "genmlx-9ssv: a non-DynamicGF combinator (Map/Unfold/Scan/Switch/Mix) spliced
   under vsimulate/vgenerate/vregenerate routes through
   handler/combinator-batched-fallback, which ran each per-particle combinator
   GFI op with NO key — so the combinator self-seeded entropy via
   rng/fresh-key (js/Math.random), and TWO runs under the same fixed top-level
   key diverged. (The split produced [k1 k2] but k2 was dead.)

   Fix: the fallback now derives per-particle keys (rng/split-n) and attaches
   each to the spliced combinator via :genmlx.dynamic/key metadata, which the
   combinator GFI ops read through combinators/splice-key (falling back to a
   fresh key when no real key is threaded — so direct, non-spliced use is
   byte-for-byte unchanged). The dead k2 is replaced by a real carry key.

   These tests pin the reproducibility (done-means): same fixed key -> bit
   identical batched choices; different key -> different; particles stay
   mutually independent within a run."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- elem-leaf
  "[N]-vector at <splice-addr>/<i>/<leaf> in a VectorizedTrace's choices."
  [vt splice-addr i leaf]
  (mx/->clj (cm/get-value
              (cm/get-submap (cm/get-submap (cm/get-submap (:choices vt) splice-addr) i) leaf))))

(defn- all-elems [vt splice-addr leaf n]
  (vec (for [i (range n)] (elem-leaf vt splice-addr i leaf))))

;; --- Map combinator spliced (the done-means case) ----------------------------
(def map-kernel (gen [x] (trace :v (dist/gaussian x 1))))
(def map-model (gen [] (splice :m (comb/map-combinator map-kernel) [0.0 1.0 2.0])))

(deftest map-splice-vsimulate-reproducible
  (let [n 4
        k (rng/fresh-key 42)
        a (dyn/vsimulate map-model [] n k)
        b (dyn/vsimulate map-model [] n k)
        c (dyn/vsimulate map-model [] n (rng/fresh-key 99))]
    (testing "same fixed key -> bit-identical :m choices (3 elements)"
      (is (= (all-elems a :m :v 3) (all-elems b :m :v 3))
          "two vsimulate runs under the same key are byte-for-byte identical"))
    (testing "different key -> different choices (entropy actually descends from the key)"
      (is (not= (all-elems a :m :v 3) (all-elems c :m :v 3))
          "a different top-level key produces different draws"))
    (testing "particles stay mutually independent within a run"
      (is (apply distinct? (elem-leaf a :m 0 :v))
          "the N particles of element 0 are not all identical"))
    (testing "score is [N]-shaped and identical across the two same-key runs"
      (is (= [n] (mx/shape (:score a))) "score is [N]-shaped")
      (is (= (mx/->clj (:score a)) (mx/->clj (:score b))) "same-key scores identical"))))

;; --- Unfold combinator spliced (state-threading kernel) ----------------------
(def unfold-kernel
  (gen [t state a b]
    (let [mean (if state (mx/add (mx/multiply a (:x state)) b) (mx/scalar 0.0))
          x (trace :x (dist/gaussian mean 1.0))]
      {:x x})))
(def unfold-model
  (gen [] (splice :seq (comb/unfold-combinator unfold-kernel) 3 nil (mx/scalar 0.5) (mx/scalar 1.0))))

(deftest unfold-splice-vsimulate-reproducible
  (let [n 4
        k (rng/fresh-key 7)
        a (dyn/vsimulate unfold-model [] n k)
        b (dyn/vsimulate unfold-model [] n k)
        c (dyn/vsimulate unfold-model [] n (rng/fresh-key 8))]
    (testing "same fixed key -> bit-identical :seq choices (3 steps)"
      (is (= (all-elems a :seq :x 3) (all-elems b :seq :x 3))
          "two vsimulate runs under the same key are byte-for-byte identical"))
    (testing "different key -> different choices"
      (is (not= (all-elems a :seq :x 3) (all-elems c :seq :x 3))
          "a different top-level key produces different draws"))
    (testing "score identical across same-key runs"
      (is (= (mx/->clj (:score a)) (mx/->clj (:score b))) "same-key scores identical"))))

;; --- direct (non-spliced) use is unaffected ----------------------------------
(deftest direct-combinator-use-unchanged
  (testing "a directly-simulated combinator still self-seeds (no metadata key) and works"
    (let [tr (p/simulate (comb/map-combinator map-kernel) [[0.0 1.0 2.0]])]
      (is (= 3 (count (:retval tr))) "direct Map simulate returns 3 elements")
      (is (mx/array? (:score tr)) "direct Map simulate has a score"))))

(cljs.test/run-tests)
