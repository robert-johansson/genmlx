;; @tier medium
(ns genmlx.world-proc-test
  "genmlx-5zmv: the synchronous anytime scheduler (control's eval!). Deadline
   logic is tested deterministically with an injected fake clock + fake
   substrate; an integration test drives the real rfal steppable."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.world.proc :as proc]
            [genmlx.inference.steppable :as sp]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.gen :refer [gen]]))

;; Deterministic fake substrate: a counter to 1000. No MLX.
(def fake {:init (fn [] 0)
           :step inc
           :done? (fn [s] (>= s 1000))
           :best identity})

;; Fake clock: returns 0, 1ms, 2ms, ... ns per call (read-then-advance).
(defn- fake-clock []
  (let [clk (atom 0)]
    (fn [] (let [v @clk] (swap! clk + 1000000) v))))

(deftest available-test
  (testing "available? returns a boolean"
    (is (boolean? (proc/available?)) "available? is a boolean")))

(deftest runs-to-done-test
  (testing "with a generous budget the scheduler runs the substrate to completion"
    (let [r (proc/with-deadline (:init fake) (:step fake) (:done? fake) (:best fake)
              {:budget-ms 1000 :chunk 256 :gc-every 0 :now-fn (fake-clock)})]
      (is (= :done (:stopped-by r)) "stops because the substrate is done")
      (is (= 1000 (:steps r)) "advanced exactly to the done threshold")
      (is (= 1000 (:best r)) "best is the terminal counter"))))

(deftest deadline-path-test
  (testing "a tight budget stops at the deadline (deterministic fake clock)"
    (let [r (proc/with-deadline (:init fake) (:step fake) (:done? fake) (:best fake)
              {:budget-ms 3 :chunk 10 :min-steps 1 :gc-every 0 :now-fn (fake-clock)})]
      ;; start=0; advance@iter1 (steps 0<min, no clock read) -> 10; clock reads
      ;; 1,2,3ms at iters 2,3,4; 3ms>=3ms deadline at iter4 -> stop. steps=30.
      (is (= :deadline (:stopped-by r)) "stops at the deadline, not done")
      (is (= 30 (:steps r)) "deterministic step count under the fake clock")
      (is (< (:steps r) 1000) "did not finish the substrate"))))

(deftest anytime-min-steps-test
  (testing "budget 0 still runs at least one chunk (the anytime guarantee)"
    (let [r (proc/with-deadline (:init fake) (:step fake) (:done? fake) (:best fake)
              {:budget-ms 0 :chunk 10 :min-steps 1 :gc-every 0 :now-fn (fake-clock)})]
      (is (= :deadline (:stopped-by r)) "deadline at budget 0")
      (is (>= (:steps r) 10) "at least one full chunk ran (min-steps honored)"))))

(deftest monotonicity-test
  (testing "a larger budget yields no fewer steps (paired, deterministic clock)"
    (let [run (fn [b] (:steps (proc/with-deadline (:init fake) (:step fake) (:done? fake) (:best fake)
                                 {:budget-ms b :chunk 10 :min-steps 1 :gc-every 0 :now-fn (fake-clock)})))]
      (is (<= (run 3) (run 5)) "steps non-decreasing in budget")
      (is (< (run 3) (run 7)) "a much larger budget runs strictly more"))))

(deftest gc-sweep-ownership-test
  (testing "proc sweeps dead MLX arrays at each gc-every chunk boundary (re-homing smcp3.cljs:218)"
    (let [sweeps (atom 0)]
      (with-redefs [mx/sweep-dead-arrays! (fn [] (swap! sweeps inc))
                    mx/clear-cache! (fn [] nil)]
        (proc/with-deadline (:init fake) (:step fake) (:done? fake) (:best fake)
          {:budget-ms 1000 :chunk 256 :gc-every 1 :now-fn (fake-clock)}))
      ;; 1000 / 256 -> 4 chunks -> 4 sweeps
      (is (= 4 @sweeps) "one sweep per chunk boundary"))))

(deftest synchronous-result-test
  (testing "with-deadline returns a plain map synchronously (never a promise)"
    (let [r (proc/with-deadline (:init fake) (:step fake) (:done? fake) (:best fake)
              {:budget-ms 1000 :chunk 256 :gc-every 0 :now-fn (fake-clock)})]
      (is (map? r) "result is a plain map, not a promise")
      (is (not (instance? js/Promise r)) "not a native Promise"))))

(deftest worker-fenced-test
  (testing "the spike-gated worker scheduler throws (cannot silently ship)"
    (is (thrown? :default (proc/worker-pool-UNPROVEN)))))

;; ---------------------------------------------------------------------------
;; Integration: drive the real rfal steppable substrate to completion
;; ---------------------------------------------------------------------------
(def model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 3))]
      (doseq [i (range 3)]
        (trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(def obs-seq
  (mapv (fn [i] (cm/choicemap (keyword (str "y" i)) (mx/scalar (nth [0.4 1.1 0.7] i)))) (range 3)))

(deftest steppable-integration-test
  (testing "with-deadline drives the rfal steppable to :done and recovers the driver log-ML"
    (let [o {:particles 2000 :ess-threshold 0.5 :key (rng/fresh-key 314)}
          steppable {:init (fn [] (sp/init-state model [] obs-seq o))
                     :step sp/step
                     :done? sp/done?
                     :best (fn [s] (:log-ml-estimate (sp/peek s)))}
          r (proc/with-deadline (:init steppable) (:step steppable) (:done? steppable) (:best steppable)
              {:budget-ms 60000 :chunk 1 :gc-every 1})
          driver (mx/realize (:log-ml-estimate (smcp3/smcp3 o model [] obs-seq)))]
      (is (= :done (:stopped-by r)) "the SMC run completed within budget")
      (is (= 3 (:steps r)) "advanced one step per observation")
      (is (h/close? driver (:best r) 1e-4)
          (str "scheduler-driven log-ML " (:best r) " == driver " driver)))))

(cljs.test/run-tests)
