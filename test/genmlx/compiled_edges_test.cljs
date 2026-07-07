;; @tier fast
(ns genmlx.compiled-edges-test
  "genmlx-8mih: two compiled-path edges.

   1. Destructured gen params ({:keys [a]}) are legal fn syntax the handler
      executes fine, but build-binding-env crashed model DEFINITION (the M2
      builder runs eagerly at make-gen-fn). Now: non-symbol params decline
      compilation; the model constructs and runs on the handler path.
   2. An arg-derived retval had a JS-number type from handler + compiled
      simulate but an MLX-scalar type from compiled generate/update/assess/
      regenerate (mlx-args vs raw args-vec fed to retval-fn). Now uniform.

   Run: bunx --bun nbb@1.4.208 test/genmlx/compiled_edges_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

;; ===========================================================================
(println "\n-- edge 1: destructured params construct + run (handler path) --")

(let [m (try (dyn/auto-key
              (gen [{:keys [a]}]
                (trace :x (dist/gaussian a 1))))
             (catch :default e e))]
  (assert-true "destructured-param model CONSTRUCTS (no throw at make-gen-fn)"
               (not (instance? js/Error m)))
  (when-not (instance? js/Error m)
    (assert-true "compilation declined (no compiled-simulate — handler path)"
                 (nil? (get-in m [:schema :compiled-simulate])))
    (let [t (p/simulate (dyn/with-key m (rng/fresh-key 3)) [{:a 2.0}])]
      (assert-true "model simulates via the handler"
                   (some? (cm/get-value (cm/get-submap (:choices t) :x)))))))

;; ===========================================================================
(println "\n-- edge 2: retval type identical across all ops (arg-returning model) --")

(def m2 (dyn/auto-key (gen [a] (trace :x (dist/gaussian 0 1)) a)))

(let [_ (assert-true "arg-returning model IS compiled (sanity)"
                     (some? (get-in m2 [:schema :compiled-simulate])))
      arg 3.5
      t (p/simulate (dyn/with-key m2 (rng/fresh-key 5)) [arg])
      sim-type (mx/array? (:retval t))
      g (p/generate (dyn/with-key m2 (rng/fresh-key 6)) [arg]
                    (cm/set-choice cm/EMPTY [:x] (mx/scalar 0.2)))
      gen-type (mx/array? (:retval (:trace g)))
      u (p/update (dyn/with-key m2 (rng/fresh-key 7)) t
                  (cm/set-choice cm/EMPTY [:x] (mx/scalar 0.4)))
      upd-type (mx/array? (:retval (:trace u)))
      a (p/assess (dyn/with-key m2 (rng/fresh-key 8)) [arg]
                  (cm/set-choice cm/EMPTY [:x] (mx/scalar 0.1)))
      ass-type (mx/array? (:retval a))
      r (p/regenerate (dyn/with-key m2 (rng/fresh-key 9)) t (sel/select :x))
      reg-type (mx/array? (:retval (:trace r)))
      ;; handler oracle
      h (dyn/auto-key (dyn/strip-alternate-paths m2))
      ht (p/simulate (dyn/with-key h (rng/fresh-key 5)) [arg])
      handler-type (mx/array? (:retval ht))]
  (println "    retval mx-array?: handler" handler-type "| sim" sim-type
           "| gen" gen-type "| upd" upd-type "| assess" ass-type "| regen" reg-type)
  (assert-true "all six ops agree with the handler's retval TYPE"
               (= handler-type sim-type gen-type upd-type ass-type reg-type))
  (assert-true "and the value is the arg"
               (let [rv (:retval (:trace g))]
                 (= 3.5 (if (mx/array? rv) (mx/realize rv) rv)))))

;; ===========================================================================
(println (str "\n== compiled-edges: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
