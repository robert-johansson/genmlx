;; @tier fast
(ns genmlx.cost-test
  "genmlx-i0s4: the compute/cost meter (CostMeter value + cost+ + measure +
   structural-cost) and the monotonic mlx.cljs membrane counters."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.inference.cost :as cost]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic]
            [genmlx.gen :refer [gen]]))

(deftest counter-monotonicity-test
  (testing "membrane counters are monotonic and reset cleanly; the three chokepoints are distinct"
    (mx/reset-cost-counters!)
    (is (= {:forced-evals 0 :items 0 :clj-reads 0} (mx/read-cost-counters)) "reset zeroes all")
    (let [a (mx/add (mx/scalar 1.0) (mx/scalar 2.0))]
      (mx/eval! a)
      (is (= 1 (:forced-evals (mx/read-cost-counters))) "eval! -> forced-evals")
      (is (= 0 (:items (mx/read-cost-counters))) "eval! does NOT bump items")
      (mx/item a)
      (is (= 1 (:items (mx/read-cost-counters))) "item -> items")
      (is (= 1 (:forced-evals (mx/read-cost-counters))) "item does NOT bump forced-evals (bypasses eval!)")
      (mx/->clj a)
      (is (= 1 (:clj-reads (mx/read-cost-counters))) "->clj -> clj-reads")
      (mx/materialize! a)
      (is (= 2 (:forced-evals (mx/read-cost-counters))) "materialize! routes through eval! (no double-count elsewhere)"))))

(deftest counter-non-reset-test
  (testing "cost counters do NOT reset on the GC-heuristic interval (they are a true compute meter)"
    (mx/reset-cost-counters!)
    (let [a (mx/scalar 1.0)]
      ;; >50 forced evals crosses the membrane's auto-cleanup check interval;
      ;; a heuristic counter would have reset, the cost counter must not.
      (dotimes [_ 60] (mx/eval! a)))
    (is (= 60 (:forced-evals (mx/read-cost-counters)))
        "60 evals accumulate monotonically past the heuristic reset interval")))

(deftest cost-plus-additivity-test
  (testing "cost+ is associative and commutative, matching a direct merge-with + (independent oracle)"
    (let [a {:forced-evals 3 :items 1}
          b {:steps 2 :particles 10}
          c {:forced-evals 5 :clj-reads 4 :structural 7}]
      (is (= (cost/cost+ a (cost/cost+ b c)) (cost/cost+ (cost/cost+ a b) c)) "associative")
      (is (= (cost/cost+ a b) (cost/cost+ b a)) "commutative")
      (is (= (cost/cost+ a b)
             (merge-with + (merge cost/zero a) (merge cost/zero b)))
          "matches a direct merge-with + over zero-padded meters")
      (is (= (set (keys (cost/cost+ a))) (set (keys cost/zero))) "result has the full key set"))))

(deftest structural-cost-test
  (testing "structural-cost = sum of descendant counts; a 3-site chain a->b->c == 3"
    (let [chain (gen []
                  (let [a (trace :a (dist/gaussian 0 1))
                        b (trace :b (dist/gaussian a 1))
                        c (trace :c (dist/gaussian b 1))]
                    c))]
      ;; descendants: a->{b,c}=2, b->{c}=1, c->{}=0  => 3
      (is (= 3 (cost/structural-cost chain)) "hand-computed blast-radius sum for a 3-chain"))
    (testing "two independent sites have zero descendants each"
      (let [indep (gen []
                    (let [a (trace :a (dist/gaussian 0 1))
                          b (trace :b (dist/gaussian 0 1))]
                      [a b]))]
        (is (= 0 (cost/structural-cost indep)) "no dependencies -> structural cost 0")))))

(deftest measure-accuracy-test
  (testing "measure reports the exact item/clj-read deltas a thunk incurs (independent oracle = the counts in the thunk)"
    (let [a (mx/scalar 1.0)
          {:keys [result cost]} (cost/measure
                                  (fn []
                                    (dotimes [_ 3] (mx/item a))
                                    (dotimes [_ 2] (mx/->clj a))
                                    :done))]
      (is (= :done result) "measure returns the thunk result")
      (is (= 3 (:items cost)) "exactly 3 item reads counted")
      (is (= 2 (:clj-reads cost)) "exactly 2 ->clj reads counted")
      (is (= 0 (:forced-evals cost)) "item/->clj do not count as forced-evals"))
    (testing "measure-step adds :steps 1 and :particles"
      (let [{:keys [cost]} (cost/measure-step (fn [] (mx/eval! (mx/scalar 1.0))) :particles 64)]
        (is (= 1 (:steps cost)) "one step")
        (is (= 64 (:particles cost)) "particles recorded")
        (is (= 1 (:forced-evals cost)) "the eval! inside the step is counted")))))

(deftest wall-clock-opt-in-test
  (testing "wall-clock is off by default and opt-in via *wall-clock?*"
    (let [{:keys [cost]} (cost/measure (fn [] (mx/eval! (mx/scalar 1.0)) :x))]
      (is (= 0 (:wall-ns cost)) "wall-ns is 0 when *wall-clock?* is false (default)"))
    (binding [cost/*wall-clock?* true]
      (let [{:keys [cost]} (cost/measure (fn [] (dotimes [_ 50] (mx/eval! (mx/scalar 1.0))) :x))]
        (is (>= (:wall-ns cost) 0) "wall-ns is measured (>=0) when enabled")))))

(cljs.test/run-tests)
