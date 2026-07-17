;; @tier slow
(ns genmlx.membrane-churn-test
  "M2 verification for the membrane buffer-COUNT fix (genmlx-5ucd Layer 1+2,
   genmlx-x7cl) under REAL GFI churn — the counterpart to resource_recovery_test
   (which uses a synthetic 700k pure-scalar loop).

   genmlx-g8vs repro: a p/generate host loop with NO manual force-gc!/materialize!
   (each call samples a tail of unconstrained Bernoullis) used to climb RSS until
   an uncatchable SIGTRAP at the ~499000 Metal-buffer wall. With the fix the
   membrane's automatic gfi-cleanup! proactive count-aware sweep (+ reactive
   self-heal) must keep the live buffer COUNT bounded and let the loop COMPLETE.

   genmlx-py4a repro (IS retains all N traces): exercised here only at a MODEST
   scale on a light model — enough to confirm the trace-retaining path runs and
   stays bounded under the new membrane. The FULL stress (the mct nested-
   combinator plate, escalating to :samples 3000) was re-run against the
   ORIGINAL mct-genmlx repro on Thor 2026-07-12 (genmlx-plfx): 3/3 clean
   completions, bounded memory — the plfx-controlled-loop-plate test below
   pins that shape in-tree."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.importance :as is]
            [genmlx.protocols :as p])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- assert-count-ceiling!
  "The buffer-COUNT ceiling is a Metal-only gauge: get-num-resources /
   get-resource-limit live in MLX's Metal allocator and return 0 on CUDA,
   where the ~499000 count wall does not exist (see count-tracking-available?
   in mlx.cljs). Assert the ceiling only where it is measurable; elsewhere a
   visible SKIP — the completion/no-abort asserts above it still run, and the
   CUDA-side bounded-memory evidence is the guarded MemAvailable log of the
   genmlx-plfx repro runs."
  [c limit msg]
  (if (pos? limit)
    (is (< c (long (* 0.9 limit))) msg)
    (println (str "  [SKIP] " msg " — no Metal buffer-count gauge on this "
                  "backend (limit=0, CUDA)"))))

(def churn-model
  (dyn/auto-key
    (gen [n]
      (let [pp (trace :p (dist/uniform 0 1))]
        (dotimes [t n] (trace (keyword (str "x" t)) (dist/bernoulli pp)))
        :done))))

;; partial constraints: pin a few sites, leave the rest SAMPLED every call
(def partial-cons
  (cm/choicemap :x0 (mx/scalar 1.0) :x1 (mx/scalar 0.0)
                :x2 (mx/scalar 1.0) :x3 (mx/scalar 0.0)))

(deftest g8vs-host-loop-bounded
  (testing "p/generate host loop with NO manual cleanup stays bounded + completes"
    (reset! mx/alloc-retry-count 0)
    (reset! mx/proactive-sweep-count 0)
    (let [iters 2500
          limit (mx/get-resource-limit)
          maxc (atom 0)
          completed (atom false)]
      (dotimes [i iters]
        ;; intentionally NO force-gc!/materialize! — rely entirely on the
        ;; membrane's automatic gfi-cleanup! proactive sweep + reactive self-heal.
        (mx/item (:weight (p/generate churn-model [40] partial-cons)))
        (let [c (mx/get-num-resources)]
          (when (> c @maxc) (reset! maxc c))
          (when (zero? (mod (inc i) 500))
            (println (str "  [" (inc i) "/" iters "] live-buffers=" c
                          " peak=" @maxc " sweeps=" @mx/proactive-sweep-count
                          " retries=" @mx/alloc-retry-count)))))
      (reset! completed true)
      (println (str "  DONE " iters " generates | peak live-buffers=" @maxc "/" limit
                    " | proactive-sweeps=" @mx/proactive-sweep-count
                    " | reactive-retries=" @mx/alloc-retry-count))
      ;; reaching here at all means no SIGTRAP/exit-144 (a crash would kill the
      ;; process before the asserts) — the uncatchable-abort wedge is gone.
      (is @completed "completed all generates without SIGTRAP/exit-144")
      (assert-count-ceiling! @maxc limit
        (str "peak live-buffer count " @maxc " stays well under the ~" limit " wall"))
      (is (= 0 @mx/alloc-retry-count)
          "automatic cleanup kept the loop off the wall (reactive retry never needed)"))))

(deftest py4a-importance-sampling-bounded
  (testing "importance-sampling (retains all N traces) runs + stays bounded at modest scale"
    (reset! mx/alloc-retry-count 0)
    (reset! mx/proactive-sweep-count 0)
    (let [limit (mx/get-resource-limit)
          obs (cm/choicemap :x0 (mx/scalar 1.0) :x1 (mx/scalar 0.0) :x2 (mx/scalar 1.0))
          result (is/importance-sampling {:samples 400 :key (rng/fresh-key 7)}
                                         churn-model [40] obs)
          c (mx/get-num-resources)]
      (println (str "  IS 400 samples | live-buffers=" c "/" limit
                    " | proactive-sweeps=" @mx/proactive-sweep-count
                    " | reactive-retries=" @mx/alloc-retry-count))
      (is (some? result) "importance-sampling completed without abort")
      (assert-count-ceiling! c limit "IS live-buffer count bounded under the wall"))))

;; --- genmlx-py4a: nested PLATE (shared latent + N sessions x cycles tracing
;;     sites) at MODERATE scale. Before the deep-trace materialize fix
;;     (importance.cljs only materialized weight+score), each of the N retained
;;     traces pinned its whole per-sample subgraph (hundreds of native buffers)
;;     alive — the dead-buffer sweep cannot free buffers a live retained leaf
;;     references. After the fix (u/materialize-state per particle) each retained
;;     trace holds only bounded leaf buffers and the intermediates become
;;     dead+sweepable. This is a bounded-completion regression guard exercising
;;     the retain-many-leaves path; the EXTREME regime (~264 arrays/sample at
;;     very high :samples) stays deferred to the wedge-tolerant host.
(def plate-model
  (dyn/auto-key
    (gen [n-sessions n-cycles]
      (let [mu (trace :mu (dist/gaussian 0 5))]
        (dotimes [s n-sessions]
          (let [seed (trace (keyword (str "s" s "_seed")) (dist/gaussian mu 1.0))]
            (dotimes [c n-cycles]
              (trace (keyword (str "s" s "_c" c)) (dist/gaussian seed 0.5)))))
        :done))))

(deftest py4a-plate-importance-sampling-bounded
  (testing "IS on a nested plate deep-materializes each retained trace + stays bounded"
    (reset! mx/alloc-retry-count 0)
    (reset! mx/proactive-sweep-count 0)
    (let [sessions 8 cycles 12 samples 250
          sites-per-sample (+ 1 (* sessions (+ 1 cycles)))
          limit (mx/get-resource-limit)
          obs (cm/choicemap (keyword "s0_c0") (mx/scalar 0.0)
                            (keyword "s1_c0") (mx/scalar 0.5))
          {:keys [traces log-ml-estimate]}
          (is/importance-sampling {:samples samples :gc-every 25 :key (rng/fresh-key 11)}
                                  plate-model [sessions cycles] obs)
          c (mx/get-num-resources)
          ml (mx/item log-ml-estimate)]
      (println (str "  PLATE IS " samples " samples x " sites-per-sample
                    " sites/sample | live-buffers=" c "/" limit
                    " | proactive-sweeps=" @mx/proactive-sweep-count
                    " | reactive-retries=" @mx/alloc-retry-count " | log-ml=" ml))
      (is (= samples (count traces)) "all N traces returned (usable weighted traces)")
      (is (js/isFinite ml) "log-ml estimate is finite")
      (is (js/isFinite (mx/item (cm/get-value (cm/get-submap (:choices (first traces)) :mu))))
          "retained trace leaf is materialized + usable (deep-materialize worked)")
      (assert-count-ceiling! c limit
        (str "plate IS live-buffer count " c " stays under the ~" limit " wall")))))

;; --- genmlx-plfx: the mct repro shape PROPER — shared latents + N
;;     controlled-loop blocks with traced bernoulli stop sites and HOST
;;     (mx/item) branching per cycle. This is the nested-combinator plate the
;;     original exit-144 report ran (docs/genmlx-bugs/ in mct-genmlx): loop
;;     length is data-dependent, every cycle syncs the GPU, and each sample
;;     traces O(sessions x cycles) sites. Verified against the ORIGINAL repro
;;     on this host 2026-07-12 (3 escalating runs to :samples 3000, zero
;;     aborts, bounded memory); this test pins the shape in-tree at a
;;     tier-appropriate scale.
(defn- run-stop-loop
  "Inline mct-style controlled loop (their run-controlled-loop): per cycle,
   a traced monitor site + a traced Bernoulli stop site branched on with
   mx/item."
  [trace addr-prefix stop-p mon-p max-iters]
  (loop [t 0]
    (let [_ (trace (keyword (str addr-prefix "_mon" t)) (dist/bernoulli mon-p))
          s (trace (keyword (str addr-prefix "_stop" t)) (dist/bernoulli stop-p))]
      (if (or (>= (mx/item s) 1.0) (>= (inc t) max-iters))
        t
        (recur (inc t))))))

(def controlled-loop-plate
  (dyn/auto-key
    (gen [n-sessions]
      (let [cb (trace :cb (dist/uniform 0 1))
            ca (trace :ca (dist/uniform 0 1))
            sm (trace :sm (dist/uniform 0 1))
            ps (mx/multiply (mx/scalar 0.45)
                            (mx/multiply ca (mx/subtract (mx/scalar 1.0) cb)))]
        (dotimes [s n-sessions]
          (run-stop-loop trace (str "s" s) ps sm 22))
        :done))))

(deftest plfx-controlled-loop-plate-is-bounded
  (testing "library IS survives the controlled-loop plate (the mct exit-144 shape)"
    (reset! mx/alloc-retry-count 0)
    (reset! mx/proactive-sweep-count 0)
    (let [sessions 10 samples 300
          limit (mx/get-resource-limit)
          obs (cm/choicemap :ca (mx/scalar 0.85) :sm (mx/scalar 0.6))
          {:keys [traces log-ml-estimate]}
          (is/importance-sampling {:samples samples :key (rng/fresh-key 7)}
                                  controlled-loop-plate [sessions] obs)
          c (mx/get-num-resources)
          ml (mx/item log-ml-estimate)]
      (println (str "  CONTROLLED-LOOP PLATE IS " samples " samples x " sessions
                    " sessions (<=22 cycles, mx/item branching)"
                    " | live-buffers=" c "/" limit
                    " | proactive-sweeps=" @mx/proactive-sweep-count
                    " | reactive-retries=" @mx/alloc-retry-count
                    " | log-ml=" ml))
      (is (= samples (count traces)) "all N traces returned — completion, no abort")
      (is (js/isFinite ml) "log-ml estimate is finite")
      (assert-count-ceiling! c limit
        (str "controlled-loop plate live-buffer count " c
             " stays under the ~" limit " wall")))))

(cljs.test/run-tests)
