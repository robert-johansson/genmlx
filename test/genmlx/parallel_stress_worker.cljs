;; @tier exclude
(ns genmlx.parallel-stress-worker
  "WORKER for parallel_stress_test.cljs (genmlx-7yam) — one process of the
   N-way parallel Metal buffer-count fleet. Driven entirely by STRESS_* env
   vars; prints one machine-readable 'WORKER-RESULT {json}' line and exits.
   NOT a test file (@tier exclude): it asserts nothing itself — the driver
   asserts over the fleet's exit codes and result lines.

   Modes (STRESS_MODE):
     churn — allocate STRESS_N tiny scalars (the genmlx-5ucd wall workload,
             the same loop as resource_recovery_test) with the proactive
             Layer-2 sweep at its default threshold. Deterministic (a pure
             counting loop, no entropy) — reproducible by construction.
     hold  — allocate STRESS_HOLD scalars and keep them LIVE (a sweep cannot
             reclaim them), report the process-local buffer count, then churn
             STRESS_CHURN_AFTER more. With N workers together holding more
             than the ~499000 wall, per-process walls mean every worker
             succeeds; a device-global wall would fail the late holders.

   Barrier: when STRESS_BARRIER_DIR is set, write ready-<id> and busy-wait
   (Bun.sleepSync) for the driver's 'go' file, so the fleet's hot loops
   overlap maximally. Exit 3 on barrier timeout; exit 1 on a catchable
   resource failure; exit 0 on success. Any exit >= 128 is the uncatchable
   crash class the driver hunts."
  (:require [genmlx.mlx :as mx]))

(def ^:private fs (js/require "fs"))
(def ^:private node-path (js/require "path"))

(defn- env [k d] (or (aget (.-env js/process) k) d))
(defn- env-int [k d] (js/parseInt (env k (str d)) 10))

(def ^:private id (env "STRESS_ID" "0"))

(defn- resource-error? [e]
  (boolean (re-find #"Resource limit|metal::malloc|out of memory|MLX error"
                    (str (.-message e)))))

;; Peak trackers, sampled every 50k allocations (cheap FFI reads).
(def ^:private peak-count (atom 0))
(def ^:private peak-active (atom 0))
(defn- sample! []
  (swap! peak-count max (mx/get-num-resources))
  (swap! peak-active max (mx/get-active-memory)))

(defn- churn!
  "The genmlx-5ucd wall workload: n tiny-scalar allocations. Returns
   :completed or :caught-catchable; a non-resource error rethrows (a real
   bug, surfaced as a non-zero exit with a stack trace)."
  [n]
  (try
    (dotimes [i n]
      (mx/scalar (double i))
      (when (and (pos? i) (zero? (mod i 50000))) (sample!)))
    (sample!)
    :completed
    (catch :default e
      (if (resource-error? e) :caught-catchable (throw e)))))

(defn- barrier!
  "Signal ready-<id>, then busy-wait for the driver's 'go' file (2 min cap)."
  [bdir]
  (when (seq bdir)
    (.writeFileSync fs (.join node-path bdir (str "ready-" id)) "1")
    (let [t0 (js/Date.now)]
      (loop []
        (cond
          (.existsSync fs (.join node-path bdir "go")) :go
          (> (- (js/Date.now) t0) 120000)
          (do (println "WORKER-RESULT"
                       (js/JSON.stringify #js {:mode (env "STRESS_MODE" "churn")
                                               :id id :outcome "barrier-timeout"}))
              (js/process.exit 3))
          :else (do (js/Bun.sleepSync 100) (recur)))))))

;; Live-hold anchor: a global ref guarantees the held wrappers stay reachable
;; (and their Metal buffers pinned) for the whole process lifetime.
(def ^:private held-ref (atom nil))

(defn- finish! [result-js outcome]
  (println "WORKER-RESULT" (js/JSON.stringify result-js))
  (js/process.exit (if (= outcome :completed) 0 1)))

(let [mode (env "STRESS_MODE" "churn")
      bdir (env "STRESS_BARRIER_DIR" "")]
  (case mode
    "churn"
    (let [n (env-int "STRESS_N" 700000)
          _ (barrier! bdir)
          t0 (js/Date.now)
          outcome (churn! n)
          ms (- (js/Date.now) t0)]
      (finish!
       #js {:mode "churn" :id id :outcome (name outcome) :n n :ms ms
            :retries @mx/alloc-retry-count
            :proactive @mx/proactive-sweep-count
            :peak-count @peak-count
            :peak-active-bytes @peak-active
            :threshold @mx/buffer-count-threshold
            :limit (mx/get-resource-limit)}
       outcome))

    "hold"
    (let [k (env-int "STRESS_HOLD" 100000)
          churn-n (env-int "STRESS_CHURN_AFTER" 50000)
          t0 (js/Date.now)
          _ (reset! held-ref (mapv #(mx/scalar (double %)) (range k)))
          count-after-hold (mx/get-num-resources)
          _ (sample!)
          _ (barrier! bdir)
          outcome (churn! churn-n)
          count-after-churn (mx/get-num-resources)
          ms (- (js/Date.now) t0)]
      (finish!
       #js {:mode "hold" :id id :outcome (name outcome) :held k :ms ms
            :count-after-hold count-after-hold
            :count-after-churn count-after-churn
            :retries @mx/alloc-retry-count
            :proactive @mx/proactive-sweep-count
            :peak-count @peak-count
            :peak-active-bytes @peak-active
            :limit (mx/get-resource-limit)}
       outcome))

    (do (println "WORKER-RESULT"
                 (js/JSON.stringify #js {:id id :outcome (str "unknown-mode-" mode)}))
        (js/process.exit 2))))
