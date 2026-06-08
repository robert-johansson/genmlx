;; @tier slow
(ns genmlx.resource-recovery-test
  "Regression for genmlx-5ucd + Layer 2 (genmlx-x7cl): a Metal buffer-count
   exhaustion must be SURVIVABLE — and, with Layer 2, PRE-EMPTED so the hot path
   never hits the limit in the first place.

   Layer 1 (REACTIVE, merged): mlx-node's C++ shim catches MLX allocation throws
   and returns a sentinel; Rust turns it into a catchable error; mlx.cljs's
   `with-alloc-retry` catches a resource error, runs force-gc! + clear-cache!,
   and retries once. Net: the ~499000-buffer crash self-heals instead of aborting.
   `alloc-retry-count` counts how many times that REACTIVE catch fired.

   Layer 2 (PROACTIVE, this test): the membrane now reads the live Metal buffer
   COUNT (mx/get-num-resources, exposed from MLX's get_num_resources) and sweeps
   dead buffers when the count crosses ~80% of the limit — BEFORE the wall. So on
   a tiny-array loop the reactive path should NEVER trigger.
   `proactive-sweep-count` counts how many times the proactive sweep fired.

   The proof here is a CONTROLLED experiment on the SAME 700k-scalar loop:
     - OFF block (proactive disabled): the loop reaches the wall, so the reactive
       catch fires (alloc-retry-count > 0). Layer 1 keeps the process alive.
     - ON  block (proactive enabled):  the proactive sweep keeps the count under
       the limit, so the reactive catch NEVER fires (alloc-retry-count == 0) while
       the proactive sweep DID fire (proactive-sweep-count > 0).
   If natural GC alone kept the loop safe, the OFF block would show 0 reactive
   catches and the experiment would (correctly) fail to demonstrate the wall —
   so the OFF block doubles as a guard that the regression is real."
  (:require [genmlx.mlx :as mx]))

(def pass (atom 0))
(def fail (atom 0))
(defn assert-true [desc x]
  (if x
    (do (swap! pass inc) (println "  PASS:" desc))
    (do (swap! fail inc) (println "  FAIL:" desc))))

(defn resource-error? [e]
  (boolean (re-find #"Resource limit|metal::malloc|out of memory|MLX error"
                    (str (.-message e)))))

(def N 700000)

(defn run-alloc-loop!
  "Allocate N scalar arrays in one process, surviving (Layer 1) any resource
   error. Returns :completed (self-healed / never hit the wall) or :caught-
   catchable (a resource error surfaced catchably). A non-resource error is a
   real failure and rethrows."
  []
  (try
    (dotimes [i N]
      (mx/scalar (double i))
      (when (and (pos? i) (zero? (mod i 200000)))
        (println (str "    ... " i " allocations, still alive"))))
    :completed
    (catch :default e
      (if (resource-error? e)
        (do (println "    caught a CATCHABLE resource error (not an abort):"
                     (.-message e))
            :caught-catchable)
        (throw e)))))

(println "\n-- resource recovery / proactive buffer-count sweep (genmlx-5ucd + genmlx-x7cl) --")

;; 1. Sanity: the guard wrappers did not break ordinary construction + read.
(assert-true "scalar->item roundtrips"
             (< (js/Math.abs (- (mx/item (mx/scalar 3.5)) 3.5)) 1e-5))
(assert-true "from-vec->item roundtrips"
             (< (js/Math.abs (- (mx/item (mx/sum (mx/array [1.0 2.0 3.0]))) 6.0)) 1e-5))

;; 2. Buffer-count query bindings (Layer 2 plumbing: MLX -> FFI -> Rust -> CLJS).
(let [limit (mx/get-resource-limit)]
  (assert-true (str "get-resource-limit reports a sane wall (" limit ")")
               (> limit 100000))
  (let [before (mx/get-num-resources)
        keep   (mapv #(mx/scalar (double %)) (range 5000))   ; hold refs so they can't be freed
        after  (mx/get-num-resources)]
    (assert-true (str "get-num-resources non-negative (" before ")") (>= before 0))
    (assert-true (str "get-num-resources rises with live allocations (" before " -> " after ")")
                 (> after before))
    (count keep)))                                            ; force `keep` to stay live
(mx/force-gc!) (mx/clear-cache!)

;; 3. CONTROLLED experiment on the SAME 700k loop.
(def default-threshold @mx/buffer-count-threshold)

;; 3a. OFF: disable the proactive sweep -> the loop must reach the wall and the
;;     REACTIVE catch must fire (proving the regression is real, not masked by GC).
(reset! mx/alloc-retry-count 0)
(reset! mx/proactive-sweep-count 0)
(reset! mx/buffer-count-threshold 1e15)              ; effectively infinite -> never sweep proactively
(println (str "  [OFF] proactive sweep DISABLED; allocating " N " scalars"
              " (Layer 1 reactive catch must self-heal)..."))
(def t-off-start (js/Date.now))
(def off-outcome (run-alloc-loop!))
(def t-off (- (js/Date.now) t-off-start))
(def retries-off @mx/alloc-retry-count)
(def proactive-off @mx/proactive-sweep-count)
(assert-true "[OFF] loop did NOT abort the process" (some? off-outcome))
(assert-true (str "[OFF] proactive sweep stayed silent (count=" proactive-off ")")
             (zero? proactive-off))
(assert-true (str "[OFF] reactive retry FIRED — the wall is genuinely reached (retries="
                  retries-off ")")
             (pos? retries-off))
(println (str "    [OFF] outcome=" off-outcome " retries=" retries-off
              " proactive=" proactive-off " time=" t-off "ms"))

(mx/force-gc!) (mx/clear-cache!)

;; 3b. ON: restore the default threshold -> the proactive sweep must keep the
;;     count under the wall, so the reactive catch NEVER fires.
(reset! mx/alloc-retry-count 0)
(reset! mx/proactive-sweep-count 0)
(reset! mx/buffer-count-threshold default-threshold)
(println (str "  [ON]  proactive sweep ENABLED (threshold=" default-threshold
              "); allocating " N " scalars..."))
(def t-on-start (js/Date.now))
(def on-outcome (run-alloc-loop!))
(def t-on (- (js/Date.now) t-on-start))
(def retries-on @mx/alloc-retry-count)
(def proactive-on @mx/proactive-sweep-count)
(assert-true "[ON] loop completed" (some? on-outcome))
(assert-true (str "[ON] ZERO reactive retry catches — proactive sweep pre-empted the wall (retries="
                  retries-on ")")
             (zero? retries-on))
(assert-true (str "[ON] proactive sweep DID fire (count=" proactive-on ")")
             (pos? proactive-on))
(println (str "    [ON]  outcome=" on-outcome " retries=" retries-on
              " proactive=" proactive-on " time=" t-on "ms"))

;; 3c. Measure: proactive must not be meaningfully slower than reactive (it avoids
;;     C++ exception unwinding, so it should be <=). Loose guard to avoid flake.
(println (str "  [MEASURE] reactive(OFF)=" t-off "ms  proactive(ON)=" t-on "ms"
              "  ratio=" (if (pos? t-off) (/ (Math/round (* 100.0 (/ t-on (max 1 t-off)))) 100.0) "n/a")))
(assert-true "[MEASURE] proactive sweep is not >2x slower than reactive recovery"
             (<= t-on (* 2.0 (max 1 t-off))))

;; 3d. Hysteresis: when a high buffer count is due to LIVE buffers a sweep cannot
;;     reclaim, the proactive sweep must fire AT MOST ONCE (disarm) — not on every
;;     interval check. Without hysteresis the loop below crosses ~12 interval
;;     checks all above threshold and would force-gc! ~12 times (pure waste).
(mx/force-gc!) (mx/clear-cache!)
(reset! mx/proactive-sweep-count 0)
(let [n0  (mx/get-num-resources)
      hi  (+ n0 3000)
      _   (reset! mx/buffer-count-threshold hi)
      ;; hold > hi LIVE buffers so a sweep can't drop the count below the low
      ;; watermark (0.75*hi) — the sweep stays disarmed after the first fire.
      held (mapv #(mx/scalar (double %)) (range 8000))]
  (dotimes [_ (* 6 4096)] (mx/item (mx/scalar 1.0)))
  (assert-true (str "[HYST] live-buffer high count fires the sweep AT MOST ONCE (count="
                    @mx/proactive-sweep-count ", would be ~12 without hysteresis)")
               (<= @mx/proactive-sweep-count 1))
  (count held))                                     ; keep `held` live to here
(reset! mx/buffer-count-threshold default-threshold)

;; 4. The membrane is still usable after all the pressure.
(mx/force-gc!) (mx/clear-cache!)
(assert-true "membrane usable after exhaustion + recovery"
             (try (< (js/Math.abs (- (mx/item (mx/scalar 9.0)) 9.0)) 1e-5)
                  (catch :default _ false)))

(println (str "\n== resource-recovery: " @pass " pass, " @fail " fail =="))
(when (pos? @fail) (js/process.exit 1))
