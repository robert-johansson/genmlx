;; @tier fast
(ns genmlx.synth-steppable-test
  "Self-tests for genmlx.control.synth-steppable (genmlx-yd7c): the {:propose :deepen
   :stop} synthesis substrate. Trivial injected scorer (no GPU) so the test is pure +
   fast; the RRPS bench exercises the real exact/IS scorer."
  (:require [genmlx.control.synth-steppable :as ss]
            [genmlx.inference.cost :as cost]
            [genmlx.world.proc :as proc]))

(def ^:private fails (atom 0))
(defn assert-true [msg p] (if p (println (str "  [PASS] " msg))
                              (do (swap! fails inc) (println (str "  [FAIL] " msg)))))
(defn close? [a b tol] (< (js/Math.abs (- a b)) tol))

;; A trivial scorer: each candidate carries a :true-ml and (for IS candidates) a
;; depth-dependent downward bias that vanishes as depth grows — mirroring real IS.
(defn make-scorer []
  (fn [cand depth]
    (if (:conjugate? cand)
      {:log-ml (:true-ml cand) :method :exact :conjugate? true}
      {:log-ml (- (:true-ml cand) (/ 4.0 (js/Math.sqrt depth)))  ;; bias -> 0 as depth grows
       :method :handler-is :conjugate? false})))

;; A (conjugate) is the shallow-depth winner; B (IS, the true best) is UNDER-rated at
;; depth 64 (-8.0 - 4/8 = -8.5 < A=-8.2) so it LOSES until deepened, then recovers
;; (-8.0 - 4/sqrt(512) ~= -8.18 > A) and wins — the directional-margin deepen scenario.
(def stream
  [{:id :A :true-ml -8.2  :conjugate? true}
   {:id :B :true-ml -8.0  :conjugate? false}    ;; IS candidate (the late, true-best one, under-rated shallow)
   {:id :C :true-ml -12.0 :conjugate? true}])

(println "\n== synth_steppable: proc drives default policy to a committed stop ==")
(let [base (ss/synth-steppable {:stream stream :score (make-scorer) :init-depth 64})
      out  (proc/with-deadline (:init base) (:step base) (:done? base) (:best base)
                               {:budget-ms 60000 :chunk 1 :gc-every 0})]
  (assert-true "proc reaches a committed stop (:done)" (= :done (:stopped-by out)))
  (assert-true "all 3 candidates revealed" (= 3 (count (:pool (:state out)))))
  ;; The default greedy policy only PROPOSES (never deepens), so B stays under-rated at
  ;; depth 64 (-8.5) and the shallow winner A (-8.2) is selected — exactly the mis-ranking
  ;; the controller's :deepen action exists to correct (demonstrated below).
  (assert-true ":best = shallow winner A (default propose-only policy never deepens B)"
               (= :A (:best out))))

(println "\n== synth_steppable: action interface (propose / deepen / stop) + cost ==")
(let [base (ss/synth-steppable {:stream stream :score (make-scorer)
                                :init-depth 64 :deepen-factor 8 :max-depth 4096})
      s0 ((:init base))]
  (assert-true "fresh state: only :stop and :propose available (no pool to deepen)"
               (= #{:stop :propose} (set ((:actions base) s0))))
  ;; propose A (conjugate, exact)
  (let [{s1 :state c1 :cost} ((:apply-action base) s0 :propose)]
    (assert-true "propose A charges host :llm-tokens (120) + :sci-evals (1), no particles"
                 (and (= 120 (:llm-tokens c1)) (= 1 (:sci-evals c1)) (= 0 (:particles c1))))
    (assert-true "after 1 propose, no IS candidate yet -> can't deepen"
                 (not (some #{:deepen} ((:actions base) s1))))
    ;; propose B (IS) -> deepen becomes available
    (let [{s2 :state c2 :cost} ((:apply-action base) s1 :propose)]
      (assert-true "propose B (IS) charges :particles = init-depth (64)" (= 64 (:particles c2)))
      (assert-true "after proposing IS candidate B, :deepen available" (some #{:deepen} ((:actions base) s2)))
      (assert-true "before deepen, shallow winner is A (B@64 under-rated, losing)"
                   (= :A (:id ((:best-entry base) s2))))
      (let [b-shallow (:log-ml (first (filter #(= :B (:id %)) (:pool s2))))
            {s3 :state c3 :cost} ((:apply-action base) s2 :deepen)
            b-deep (:log-ml (first (filter #(= :B (:id %)) (:pool s3))))]
        (assert-true "deepen B raises its log-ml toward true (-8) — IS bias shrinks"
                     (> b-deep b-shallow))
        (assert-true "deepen FLIPS the selection: B now beats A (the depth knob paying off)"
                     (= :B (:id ((:best-entry base) s3))))
        (assert-true "deepen charges extra :particles (512-64=448)" (= 448 (:particles c3)))
        (assert-true "deepen B reaches depth 512" (= 512 (:depth (first (filter #(= :B (:id %)) (:pool s3))))))
        ;; stop
        (let [{s4 :state c4 :cost} ((:apply-action base) s3 :stop)]
          (assert-true "stop sets :stopped? and is free" (and (:stopped? s4) (= cost/zero c4)))
          (assert-true "done? true after stop" ((:done? base) s4))
          (assert-true "synth-compute folds host+particles" (> (cost/synth-compute c3) 400)))))))

(println (str "\n== synth_steppable_test: " (if (zero? @fails) "ALL PASS" (str @fails " FAIL")) " =="))
(when (pos? @fails) (js/process.exit 1))
