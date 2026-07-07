;; @tier fast
(ns genmlx.node-constraint-test
  "genmlx-dp60: a NODE-shaped constraint at a primitive site must THROW
   (Gen.jl semantics), not silently sample fresh with weight 0.

   Audit genmlx-ansg (agent-VERIFIED): (p/generate m [] (choicemap :x {:oops 1.0}))
   on x ~ N(0,10) produced weight 0 with :x freshly sampled and
   :unused-constraints nil — the user believed they conditioned; the
   importance weight was silently wrong.

   Run: bunx --bun nbb@1.4.208 test/genmlx/node_constraint_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(defn- error-of [f]
  (try (f) nil (catch :default e e)))

(def model (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 10))] x))))

(def bad-constraint
  ;; a nested choicemap where a VALUE belongs
  (cm/set-choice cm/EMPTY [:x :oops] (mx/scalar 1.0)))

;; ===========================================================================
(println "\n-- generate: node-at-leaf throws with the address --")

(let [e (error-of #(p/generate model [] bad-constraint))]
  (assert-true "generate throws on a node-shaped constraint at a leaf site" (some? e))
  (assert-true "the error names the address"
               (and e (re-find #":x" (.-message e))))
  (assert-true "the error carries :genmlx/error :node-constraint-at-leaf"
               (= :node-constraint-at-leaf (:genmlx/error (ex-data e)))))

(println "\n-- update: node-at-leaf throws --")

(let [t (p/simulate model [])
      e (error-of #(p/update model t bad-constraint))]
  (assert-true "update throws on a node-shaped constraint at a leaf site" (some? e))
  (assert-true "update error names the address" (and e (re-find #":x" (.-message e)))))

(println "\n-- batched: vgenerate + vupdate throw --")

(let [e (error-of #(dyn/vgenerate model [] bad-constraint 4 (rng/fresh-key 1)))]
  (assert-true "vgenerate throws on a node-shaped constraint" (some? e)))

(let [vt (dyn/vsimulate model [] 4 (rng/fresh-key 2))
      e (error-of #(dyn/vupdate model vt bad-constraint (rng/fresh-key 3)))]
  (assert-true "vupdate throws on a node-shaped constraint" (some? e)))

;; ===========================================================================
(println "\n-- sanity: legitimate shapes still work --")

(let [{:keys [trace weight]} (p/generate model [] (cm/set-choice cm/EMPTY [:x] (mx/scalar 1.5)))]
  (assert-true "a proper leaf constraint still conditions (weight nonzero)"
               (not= 0.0 (mx/realize weight)))
  (assert-true "the constrained value lands in the trace"
               (< (js/Math.abs (- (mx/realize (cm/get-value (cm/get-submap (:choices trace) :x))) 1.5))
                  1e-6)))

(assert-true "empty constraints still mean unconstrained (no throw)"
             (some? (p/generate model [] cm/EMPTY)))

;; hierarchical constraints at SPLICE sites must keep working (nodes are the
;; correct shape there)
(let [sub (dyn/auto-key (gen [] (trace :y (dist/gaussian 0 1))))
      outer (dyn/auto-key
             (dyn/make-gen-fn
              (fn [rt] ((.-splice rt) :s sub))
              '([] (splice :s pefsub/sub))))
      obs (cm/set-choice cm/EMPTY [:s :y] (mx/scalar 0.5))
      {:keys [trace]} (p/generate outer [] obs)]
  (assert-true "node constraints at SPLICE addresses still condition sub-sites"
               (< (js/Math.abs (- (mx/realize (cm/get-choice (:choices trace) [:s :y])) 0.5))
                  1e-6)))

;; prefix-compiled models (M3): node-at-leaf on a PREFIX site throws too
(println "\n-- prefix-compiled model --")
(let [pm (dyn/auto-key
          (dyn/make-gen-fn
           (fn [rt a0]
             (let [t (.-trace rt)
                   x (t :x0 (dist/gaussian 0 1))]
               (if (pos? a0)
                 (t :b (dist/gaussian 0 1))
                 (t :bf (dist/gaussian 1 1)))))
           '([a0] (let [x (trace :x0 (dist/gaussian 0 1))]
                    (if (pos? a0)
                      (trace :b (dist/gaussian 0 1))
                      (trace :bf (dist/gaussian 1 1)))))))
      bad (cm/set-choice cm/EMPTY [:x0 :oops] (mx/scalar 1.0))
      e (error-of #(p/generate pm [1.0] bad))]
  (assert-true "model took the prefix path (sanity)"
               (some? (get-in pm [:schema :compiled-prefix-generate])))
  (assert-true "prefix generate throws on node-at-leaf at a prefix site" (some? e)))

;; ===========================================================================
(println (str "\n== node-constraint: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
