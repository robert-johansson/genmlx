(ns genmlx.resource-recovery-test
  "Regression for genmlx-5ucd: a Metal buffer-count exhaustion must be SURVIVABLE
   — self-healing or, at worst, a CATCHABLE error — never an uncatchable process
   abort (libc++abi terminate -> SIGTRAP).

   Mechanism under test (the catchable-boundary fix):
     - mlx-node's C++ shim catches MLX allocation throws and returns a sentinel
       (null/false) instead of letting the exception unwind out and abort.
     - the Rust napi layer turns that into a thrown, catchable error carrying the
       real MLX message.
     - mlx.cljs's `with-alloc-retry` catches a resource error on the construction/
       read boundary, runs force-gc! (finalizes dead MxArray wrappers -> frees
       their Metal buffers) + clear-cache!, and retries once.

   On the PRE-FIX binary the loop below SIGTRAPs around the ~499000-buffer limit
   (each scalar eagerly allocates a Metal buffer; the dead-but-unfinalized wrappers
   pin them). Post-fix it completes (self-heal) or throws catchably — either way the
   PROCESS SURVIVES and we reach the assertions."
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

(println "\n-- resource recovery / catchable OOM (genmlx-5ucd) --")

;; 1. Sanity: the guard wrappers did not break ordinary construction + read.
(assert-true "scalar->item roundtrips"
             (< (js/Math.abs (- (mx/item (mx/scalar 3.5)) 3.5)) 1e-5))
(assert-true "from-vec->item roundtrips"
             (< (js/Math.abs (- (mx/item (mx/sum (mx/array [1.0 2.0 3.0]))) 6.0)) 1e-5))

;; 2. THE regression: allocate well past the ~499000 live-buffer limit in ONE
;;    process. Pre-fix => SIGTRAP (process dies, we never get here). Post-fix =>
;;    the loop self-heals (or throws a catchable error), and we continue.
(def N 700000)
(println (str "  allocating " N " scalar arrays in one process"
              " (pre-fix this aborts at ~499000)..."))
(let [outcome (try
                (dotimes [i N]
                  (mx/scalar (double i))
                  (when (and (pos? i) (zero? (mod i 100000)))
                    (println (str "    ... " i " allocations, still alive"))))
                :completed
                (catch :default e
                  (if (resource-error? e)
                    (do (println "    caught a CATCHABLE resource error (not an abort):"
                                 (.-message e))
                        :caught-catchable)
                    (throw e))))]  ; a non-resource error is a real failure
  (assert-true "buffer-exhaustion loop did NOT abort the process" (some? outcome))
  (println "    outcome:" outcome))

;; 3. The membrane is still usable after the pressure (explicit reclaim + a fresh op).
(mx/force-gc!)
(mx/clear-cache!)
(assert-true "membrane usable after exhaustion + recovery"
             (try (< (js/Math.abs (- (mx/item (mx/scalar 9.0)) 9.0)) 1e-5)
                  (catch :default _ false)))

(println (str "\n== resource-recovery: " @pass " pass, " @fail " fail =="))
(when (pos? @fail) (js/process.exit 1))
