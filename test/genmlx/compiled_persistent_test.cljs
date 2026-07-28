;; @tier fast
(ns genmlx.compiled-persistent-test
  "genmlx-cqgx (Phase 1 of genmlx-z2gt): the persistent-compile trio.

   The equivalence suite is the SAFETY NET the 819v verdict demands before
   any compiled-path call site opts in: compiled values AND GRADIENTS must
   match direct execution (the failure mode being guarded is silent zero
   gradients), replay must actually replay (builder invoked once, not per
   call), shape changes must retrace, and the handle lifecycle must be safe."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; Value equivalence + replay-without-rebuild
;; ---------------------------------------------------------------------------

(deftest value-equivalence-and-replay-test
  (testing "compiled values match direct; builder traces ONCE across calls"
    (let [calls (atom 0)
          f     (fn [x y]
                  (swap! calls inc)
                  (mx/add (mx/multiply x y) 2))
          cf    (mx/compile-create f)]
      (try
        (let [x1 (mx/array [1.0 2.0 3.0])
              y1 (mx/array [4.0 5.0 6.0])
              [r1] (mx/compiled-call cf x1 y1)
              direct (f x1 y1)]
          (is (= (mx/->clj r1) (mx/->clj direct))
              "first call (trace) matches direct execution")
          ;; direct call above bumped the counter too: trace=1, direct=1
          (is (= 2 @calls) "builder ran once for trace, once directly"))
        (let [x2 (mx/array [10.0 20.0 30.0])
              y2 (mx/array [0.5 0.25 0.1])
              [r2] (mx/compiled-call cf x2 y2)]
          (is (h/close? 7.0 (mx/item (mx/index r2 0)) 1e-5)
              "replay computes with NEW inputs (10*0.5+2)")
          (is (= 2 @calls)
              "REPLAY did not invoke the builder — the graph was cached"))
        (let [x3 (mx/array [1.0 2.0 3.0 4.0 5.0])
              y3 (mx/array [1.0 1.0 1.0 1.0 1.0])
              [r3] (mx/compiled-call cf x3 y3)]
          (is (= [3.0 4.0 5.0 6.0 7.0] (vec (mx/->clj r3)))
              "shape change produces correct values")
          (is (= 3 @calls) "shape change RETRACED (builder invoked again)"))
        (finally (mx/compiled-free! cf))))))

;; ---------------------------------------------------------------------------
;; Multi-output (builder must return a JS array)
;; ---------------------------------------------------------------------------

(deftest multi-output-test
  (testing "multi-output builder round-trips through replay"
    (let [f  (fn [x] (to-array [(mx/sum x) (mx/multiply x x)]))
          cf (mx/compile-create f)]
      (try
        (let [[s sq] (mx/compiled-call cf (mx/array [1.0 2.0 3.0]))]
          (is (h/close? 6.0 (mx/item s) 1e-5) "output 0 (sum)")
          (is (= [1.0 4.0 9.0] (vec (mx/->clj sq))) "output 1 (square)"))
        (let [[s sq] (mx/compiled-call cf (mx/array [2.0 2.0 2.0]))]
          (is (h/close? 6.0 (mx/item s) 1e-5) "replay output 0")
          (is (= [4.0 4.0 4.0] (vec (mx/->clj sq))) "replay output 1"))
        (finally (mx/compiled-free! cf))))))

;; ---------------------------------------------------------------------------
;; GRADIENT equivalence — the silent-zero tripwire
;; ---------------------------------------------------------------------------

(deftest gradient-equivalence-test
  (testing "value_and_grad INSIDE the traced builder: gradients match direct, not zero"
    (let [score  (fn [q] (mx/sum (mx/square q)))
          vg     (mx/value-and-grad score)
          f      (fn [q] (let [[v g] (vg q)] (to-array [v g])))
          cf     (mx/compile-create f)]
      (try
        (let [q       (mx/array [1.5 -2.0 0.5])
              [v g]   (mx/compiled-call cf q)
              [dv dg] (vg q)]
          (is (h/close? (mx/item dv) (mx/item v) 1e-5)
              "compiled value matches direct value")
          (is (= (mx/->clj dg) (mx/->clj g))
              "compiled gradient matches direct gradient")
          (is (h/close? 3.0 (mx/item (mx/index g 0)) 1e-5)
              "gradient is 2q, NOT silently zero"))
        ;; replay with different q — gradients must track the new input
        (let [q2    (mx/array [10.0 0.0 -1.0])
              [_ g2] (mx/compiled-call cf q2)]
          (is (h/close? 20.0 (mx/item (mx/index g2 0)) 1e-4)
              "replayed gradient tracks the NEW input (2*10), not the traced one"))
        (finally (mx/compiled-free! cf))))))

;; ---------------------------------------------------------------------------
;; Chain-shaped equivalence (mini fused-MALA body, deterministic given inputs)
;; ---------------------------------------------------------------------------

(deftest chain-equivalence-test
  (testing "a 20-step gradient chain replays with results equal to direct build"
    (let [n-steps 20
          score   (fn [q] (mx/negative (mx/sum (mx/square (mx/subtract q 2.0)))))
          vg      (mx/value-and-grad score)
          eps     0.1
          chain   (fn [q0 noise]
                    (loop [q q0, i 0]
                      (if (>= i n-steps)
                        q
                        (let [[_ g] (vg q)
                              q'    (mx/add q (mx/multiply (* 0.5 eps eps) g)
                                            (mx/multiply eps (mx/index noise i)))]
                          (recur q' (inc i))))))
          cf      (mx/compile-create chain)
          q0      (mx/array [0.0 0.0])
          noise   (mx/reshape (mx/arange 0 (* n-steps 2)) [n-steps 2])
          noise   (mx/multiply noise 0.01)]
      (try
        (let [[qc]   (mx/compiled-call cf q0 noise)
              qd     (chain q0 noise)
              cc     (vec (mx/->clj qc))
              dd     (vec (mx/->clj qd))]
          (is (h/close? (first dd) (first cc) 1e-4) "chain endpoint [0] matches direct")
          (is (h/close? (second dd) (second cc) 1e-4) "chain endpoint [1] matches direct"))
        (finally (mx/compiled-free! cf))))))

;; ---------------------------------------------------------------------------
;; Lifecycle safety
;; ---------------------------------------------------------------------------

(deftest lifecycle-test
  (testing "free is idempotent; call-after-free errors cleanly; with-compiled fences"
    (let [f  (fn [x] (mx/add x 1))
          cf (mx/compile-create f)]
      (is (true? (mx/compiled-free! cf)) "first free returns true")
      (is (false? (mx/compiled-free! cf)) "second free returns false (idempotent)")
      (is (thrown? js/Error (mx/compiled-call cf (mx/scalar 1.0)))
          "call after free throws cleanly (no crash)"))
    (let [result (mx/with-compiled (fn [x] (mx/multiply x 3))
                   (fn [call]
                     (let [[r] (call (mx/array [1.0 2.0]))]
                       (vec (mx/->clj r)))))]
      (is (= [3.0 6.0] result) "with-compiled runs the body and returns its value"))
    (let [threw (atom false)]
      (try
        (mx/with-compiled (fn [x] x)
          (fn [_] (throw (ex-info "boom" {}))))
        (catch :default _ (reset! threw true)))
      (is @threw "with-compiled propagates the throw (after freeing)"))))

(cljs.test/run-tests)
