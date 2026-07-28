;; @tier slow
(ns genmlx.parallel-stress-test
  "genmlx-7yam: N-way PARALLEL Metal buffer-count stress on ONE device.

   The ~499000 live-MTLBuffer wall (genmlx-5ucd) is handled two ways in a
   SINGLE process: Layer 1 makes a wall hit a catchable, self-healing error;
   Layer 2's proactive count sweep pre-empts it entirely (genmlx-x7cl,
   resource_recovery_test). This test proves both hold when N processes share
   one Metal device — the unvalidated case that keeps the Mac slow tier
   serial (test/run.sh JOBS_SLOW default).

   Phase A — parallel churn: N workers run the SAME 700k-tiny-scalar wall
   workload concurrently (barrier-started for maximal overlap, proactive
   sweep at default threshold). Every worker must complete with ZERO reactive
   catches fleet-wide and zero crash exits (no SIGTRAP/SIGSEGV/133/144).

   Phase B — per-process wall confirmation: N workers each HOLD ~100k LIVE
   buffers (a sweep cannot reclaim them), so the fleet holds well past the
   ~499k wall COMBINED, then each churns more. The wall is per
   MetalAllocator, i.e. per process: every worker's own count stays far under
   its own wall and every allocation succeeds. A device-global wall would
   fail the late holders — that failure (catchable, Layer 1) would surface
   here as retries/caught outcomes, never a crash.

   Workload is deterministic (pure counting loops, no entropy) — reproducible
   by construction. Unified-memory high-water is recorded (worker peaks +
   driver os.freemem min), not asserted.

   Off-Metal (CUDA) or degraded-Metal: the count wall does not exist
   (get-resource-limit = 0) — assert the negative contract and exit, like
   resource_recovery_test. Parallel-tier validation on CUDA was done with
   real batteries (genmlx-ehni)."
  (:require [genmlx.mlx :as mx]
            [promesa.core :as p]
            [clojure.string :as str]))

(def ^:private fs (js/require "fs"))
(def ^:private os (js/require "os"))
(def ^:private node-path (js/require "path"))

(def pass (atom 0))
(def fail (atom 0))
(defn assert-true [desc x]
  (if x
    (do (swap! pass inc) (println "  PASS:" desc))
    (do (swap! fail inc) (println "  FAIL:" desc))))

(defn- finish! []
  (println (str "\n== parallel-stress: " @pass " pass, " @fail " fail =="))
  (js/process.exit (if (pos? @fail) 1 0)))

;; ---------------------------------------------------------------------------
;; Platform gate (the resource_recovery_test negative-contract pattern)
;; ---------------------------------------------------------------------------
(when (or (not (mx/metal-is-available?))
          (zero? (mx/get-resource-limit)))
  (assert-true (str "off-Metal/degraded: no buffer-count wall (limit="
                    (mx/get-resource-limit) ")")
               (zero? (mx/get-resource-limit)))
  (assert-true (str "off-Metal/degraded: no count (" (mx/get-num-resources) ")")
               (zero? (mx/get-num-resources)))
  (println "  [gate] Metal buffer-count wall absent — parallel wall stress is"
           "meaningless here (CUDA parallel tiers validated via genmlx-ehni).")
  (finish!))

(def ^:private LIMIT (mx/get-resource-limit))
(def ^:private N (js/parseInt (or (aget (.-env js/process) "PARALLEL_STRESS_N") "8") 10))
(def ^:private WORKER "test/genmlx/parallel_stress_worker.cljs")

(println (str "\n-- parallel buffer-count stress (genmlx-7yam): N=" N
              " workers, wall=" LIMIT " --"))

;; ---------------------------------------------------------------------------
;; Fleet plumbing: spawn, barrier, collect
;; ---------------------------------------------------------------------------
(def ^:private min-freemem (atom js/Number.MAX_SAFE_INTEGER))
(defn- sample-freemem! [] (swap! min-freemem min (.freemem os)))

(defn- spawn-worker [env-map]
  ;; Stagger-free spawn; the launcher race (parallel bunx state, genmlx-lr9c)
  ;; is handled by the no-result retry in run-fleet below.
  (js/Bun.spawn
   #js {:cmd #js ["bun" "run" "--bun" "nbb" WORKER]
        :env (js/Object.assign #js {} (.-env js/process) (clj->js env-map))
        :stdout "pipe" :stderr "pipe"}))

(defn- collect
  "Resolve to {:code :out :err} once the worker exits and streams drain."
  [proc]
  (p/let [code (.-exited proc)
          out  (.text (js/Response. (.-stdout proc)))
          err  (.text (js/Response. (.-stderr proc)))]
    {:code code :out out :err err}))

(defn- parse-result [out]
  (let [line (->> (str/split-lines (or out ""))
                  (filter #(str/starts-with? % "WORKER-RESULT "))
                  last)]
    (when line
      (js->clj (js/JSON.parse (subs line 14)) :keywordize-keys true))))

(defn- ready-count [bdir]
  (->> (.readdirSync fs bdir)
       (filter #(str/starts-with? % "ready-"))
       count))

(defn- launcher-noise?
  "The parallel-bunx launcher race (genmlx-lr9c): the process died before any
   test code ran — no WORKER-RESULT line. A real worker failure always leaves
   one (or a stack trace much larger than the launcher's one-liner)."
  [{:keys [code out]}]
  (and (not= 0 code) (nil? (parse-result out))))

(defn- run-fleet
  "Spawn n workers with (env-fn i), barrier-start them, and resolve to a
   vector of {:code :out :err :result :id} maps. Retries a worker once on
   launcher noise (before the barrier opens; a retried worker joins late but
   the 'go' file is only written once all n are READY)."
  [n env-fn]
  (let [bdir (.mkdtempSync fs (.join node-path (.tmpdir os) "genmlx-pstress-"))
        spawn1 (fn [i] (spawn-worker (assoc (env-fn i)
                                            :STRESS_ID (str i)
                                            :STRESS_BARRIER_DIR bdir)))
        procs (atom (mapv (fn [i] {:i i :proc (spawn1 i) :retried? false})
                          (range n)))
        t0 (js/Date.now)]
    (p/loop []
      (sample-freemem!)
      (let [nready (ready-count bdir)]
        (cond
          (>= nready n)
          (do (.writeFileSync fs (.join node-path bdir "go") "1")
              ;; collect everything, then clean the barrier dir
              (p/let [results (p/all (map (fn [{:keys [i proc]}]
                                            (p/let [r (collect proc)]
                                              (assoc r :id i :result (parse-result (:out r)))))
                                          @procs))]
                (sample-freemem!)
                (.rmSync fs bdir #js {:recursive true :force true})
                (vec results)))

          (> (- (js/Date.now) t0) 180000)
          (do (doseq [{:keys [proc]} @procs] (.kill proc))
              (.rmSync fs bdir #js {:recursive true :force true})
              (throw (ex-info (str "fleet barrier timeout: " nready "/" n " ready")
                              {:ready nready})))

          :else
          ;; A worker that died before signalling READY: retry once if it is a
          ;; launcher-race casualty (no output at all, genmlx-lr9c); fail the
          ;; fleet FAST on a real pre-barrier death (load error, second death)
          ;; instead of hanging to the barrier timeout.
          (p/let [_ (p/all (map (fn [{:keys [i proc retried?] :as w}]
                                  (if (and (some? (.-exitCode proc))
                                           (not (.existsSync fs (.join node-path bdir (str "ready-" i)))))
                                    (p/let [r (collect proc)]
                                      (if (and (launcher-noise? r) (not retried?))
                                        (do (println (str "    [fleet] worker " i
                                                          " launcher-race casualty (exit " (:code r) ") — respawning"))
                                            (swap! procs assoc i (assoc w :proc (spawn1 i) :retried? true)))
                                        (throw (ex-info (str "worker " i " died pre-barrier (exit " (:code r)
                                                             "): " (first (str/split-lines (str (:err r) (:out r)))))
                                                        {:worker i :code (:code r)}))))
                                    nil))
                                @procs))
                  _ (p/delay 200)]
            (p/recur)))))))

(defn- crash-exit? [code] (or (>= code 128) (= code 124)))

(defn- fmt-mb [b] (str (js/Math.round (/ b 1048576.0)) "MB"))

;; ---------------------------------------------------------------------------
;; Phase A — N-way parallel churn (the wall workload, proactive ON)
;; ---------------------------------------------------------------------------
(defn- phase-a []
  (println (str "\n  [A] " N " workers x 700k-scalar churn, barrier-started..."))
  (p/let [rs (run-fleet N (fn [_] {:STRESS_MODE "churn" :STRESS_N "700000"}))]
    (let [codes (mapv :code rs)
          results (mapv :result rs)
          crashes (filterv crash-exit? codes)
          retries (reduce + 0 (map #(or (:retries %) 0) results))]
      (doseq [{:keys [id code result err]} rs]
        (println (str "    [A] worker " id ": exit=" code
                      " outcome=" (:outcome result)
                      " ms=" (:ms result)
                      " retries=" (:retries result)
                      " proactive=" (:proactive result)
                      " peak-count=" (:peak-count result)
                      " peak-active=" (fmt-mb (or (:peak-active-bytes result) 0))))
        (when (and (not= 0 code) (seq (or err "")))
          (println (str "      stderr| " (first (str/split-lines err))))))
      (assert-true (str "[A] ZERO crash exits across the fleet (codes=" codes ")")
                   (empty? crashes))
      (assert-true "[A] every worker exited 0"
                   (every? zero? codes))
      (assert-true "[A] every worker completed its loop"
                   (every? #(= "completed" (:outcome %)) results))
      (assert-true (str "[A] ZERO reactive catches fleet-wide (sum=" retries ")"
                        " — proactive sweep pre-empts the wall under " N "-way load")
                   (zero? retries))
      (assert-true "[A] proactive sweep fired in every worker (per-process pressure seen)"
                   (every? #(pos? (or (:proactive %) 0)) results))
      (assert-true (str "[A] every sampled peak stayed under the wall ("
                        (str/join "," (map :peak-count results)) " < " LIMIT ")")
                   (every? #(< (or (:peak-count %) 0) LIMIT) results))
      rs)))

;; ---------------------------------------------------------------------------
;; Phase B — per-process wall confirmation (fleet holds past the wall LIVE)
;; ---------------------------------------------------------------------------
;; Per-worker LIVE hold, sized so the FLEET crosses the wall (N*HOLD > limit)
;; while each process stays far under its OWN wall and under the proactive
;; threshold (0.8*limit): capped at 350k so a small-N run cannot push a single
;; process into its own wall (live buffers are unsweepable — that would be a
;; genuine per-process exhaustion, not the parallel question).
(def ^:private HOLD
  (js/Math.min 350000
               (js/Math.max 100000 (js/Math.ceil (/ (* LIMIT 1.25) N)))))

(defn- phase-b []
  (println (str "\n  [B] " N " workers x hold " HOLD " LIVE buffers (fleet "
                (* N HOLD) " vs wall " LIMIT "), then churn..."))
  (p/let [rs (run-fleet N (fn [_] {:STRESS_MODE "hold"
                                   :STRESS_HOLD (str HOLD)
                                   :STRESS_CHURN_AFTER "50000"}))]
    (let [codes (mapv :code rs)
          results (mapv :result rs)
          crashes (filterv crash-exit? codes)
          fleet-live (reduce + 0 (map #(or (:count-after-hold %) 0) results))
          retries (reduce + 0 (map #(or (:retries %) 0) results))]
      (doseq [{:keys [id code result]} rs]
        (println (str "    [B] worker " id ": exit=" code
                      " outcome=" (:outcome result)
                      " count-after-hold=" (:count-after-hold result)
                      " count-after-churn=" (:count-after-churn result)
                      " retries=" (:retries result))))
      (assert-true (str "[B] ZERO crash exits across the fleet (codes=" codes ")")
                   (empty? crashes))
      (assert-true "[B] every worker exited 0" (every? zero? codes))
      (assert-true "[B] every worker completed (held past the wall fleet-wide, kept allocating)"
                   (every? #(= "completed" (:outcome %)) results))
      (if (> (* N HOLD) LIMIT)
        (assert-true (str "[B] fleet-wide LIVE count exceeded the wall (" fleet-live
                          " > " LIMIT ") — the experiment genuinely crossed it")
                     (> fleet-live LIMIT))
        (println (str "  [B] note: N=" N " too small to cross the wall at the "
                      HOLD " per-process cap — cross-wall assert skipped"
                      " (per-process accounting still asserted below)")))
      (assert-true (str "[B] each worker saw only ITS OWN count (max "
                        (apply max 0 (map #(or (:count-after-hold %) 0) results))
                        " ≈ " HOLD " + baseline, not the fleet total)"
                        " — the count is per-process")
                   (every? #(and (>= (or (:count-after-hold %) 0) HOLD)
                                 (< (or (:count-after-hold %) 0) (+ HOLD 50000)))
                           results))
      (assert-true (str "[B] ZERO resource errors while the fleet held past the wall"
                        " (retries sum=" retries ") — the WALL is per-process")
                   (zero? retries))
      rs)))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------
(-> (p/let [_ (phase-a)
            _ (phase-b)]
      (println (str "\n  [MEM] driver min os.freemem during run: "
                    (fmt-mb @min-freemem)))
      ;; The membrane in THIS process is untouched by the workers — sanity.
      (assert-true "driver membrane usable after the fleet"
                   (< (js/Math.abs (- (mx/item (mx/scalar 9.0)) 9.0)) 1e-5))
      (finish!))
    (p/catch (fn [e]
               (println "  FATAL:" (str e))
               (swap! fail inc)
               (finish!))))
