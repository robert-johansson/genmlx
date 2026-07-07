;; @tier fast core
(ns genmlx.regen-gate-test
  "Regenerate fast-path gate regression (genmlx-njzu, genmlx-dv66).

   The fast per-site regenerate path is only equivalent to the general
   retained-only path when the SELECTED objects are mutually independent.
   Two dependency-edge classes used to be invisible to the eligibility check:

     - splice edges (both directions: splice retval -> selected site params,
       and selected site -> splice args), via a let-bound symbol whose splice
       provenance the schema walker never recorded (njzu);
     - a literal (trace :mu ...) nested inside another site's dist-args,
       which contributes no symbol for compute-deps to resolve (dv66).

   Both made jointly-selected interacting pairs 'fast eligible', producing
   MH weights that are wrong by the un-cancelled dependent-site terms —
   sel/all on such models violated the :regenerate-select-all-zero law
   (weights like -114 where exactly 0 is required).

   This file pins, per case: (1) fast(default) == general(forced) weight with
   the same key — nonzero where a retained dependent site legitimately
   rescores; (2) the eligibility routing itself, so a future gate change that
   silently falls back to general-everywhere (losing the fast path) or
   fast-everywhere (losing correctness) fails by name."
  (:require [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private fails (atom 0))
(def ^:private passes (atom 0))

(defn- assert-true [desc x]
  (if x (swap! passes inc)
        (do (swap! fails inc) (println "  FAIL" desc))))

(def sub  (gen [] (trace :z (dist/gaussian 0 10))))
(def subx (gen [x] (trace :z (dist/gaussian x 1))))

;; A: splice retval feeds a site (dependence: :sub -> :y)
(def mA (gen [] (let [a (splice :sub sub)]
                  (trace :y (dist/gaussian a 1))
                  a)))
;; B: site feeds splice args (dependence: :x -> :sub)
(def mB (gen [] (let [x (trace :x (dist/gaussian 0 10))]
                  (splice :sub subx x)
                  x)))
;; E: splice and sites mutually independent
(def mE (gen [] (let [x (trace :x (dist/gaussian 0 10))]
                  (splice :sub sub)
                  (trace :y (dist/gaussian 0 1))
                  x)))
;; F: literal trace nested in another site's dist-args (:mu -> :y)
(def mF (gen [] (trace :y (dist/gaussian (trace :mu (dist/gaussian 0 1)) 1))))
;; G: site feeds a dependent site THROUGH an immediately-invoked fn literal's
;; parameter (genmlx-7qdz: the dropped arg->param dep edge made the coupled
;; {:a :y} selection wrongly fast-eligible)
(def mG (gen [] (let [a (trace :a (dist/gaussian 0 10))]
                  ((fn [z] (trace :y (dist/gaussian z 1))) a)
                  a)))
;; H: same coupling via letfn — an opaque escape, so the gate must decline
;; fast for EVERY selection (genmlx-7qdz pin)
(def mH (gen [] (letfn [(f [z] (trace :y (dist/gaussian z 1)))]
                  (let [a (trace :a (dist/gaussian 0 10))]
                    (f a)
                    a))))

(defn- fast-vs-general
  "Regenerate with the same key on the default path and the forced-general
   path; return [w-fast w-general]."
  [model selection seed]
  (let [tr   (p/simulate (dyn/with-key model (rng/fresh-key (* seed 7))) [])
        mk   #(dyn/with-key model (rng/fresh-key seed))
        wf   (mx/item (:weight (p/regenerate (mk) tr selection)))
        wg   (mx/item (:weight (binding [dyn/*force-general-regen* true]
                                 (p/regenerate (mk) tr selection))))]
    [wf wg]))

(defn- check-case [desc model selection & {:keys [zero?]}]
  (doseq [seed [1 2 3]]
    (let [[wf wg] (fast-vs-general model selection seed)]
      (assert-true (str desc " seed " seed ": fast==general (fast " wf " general " wg ")")
                   (< (js/Math.abs (- wf wg)) 1e-3))
      (when zero?
        (assert-true (str desc " seed " seed ": weight is exactly 0 (got " wf ")")
                     (< (js/Math.abs wf) 1e-6))))))

(println "\n-- jointly-selected interacting pairs (must route general, W = 0) --")
;; Full-prior resamples: every site/splice selected => all terms cancel; the
;; :regenerate-select-all-zero law requires exactly 0.
(check-case "A sel/all"          mA sel/all              :zero? true)
(check-case "A (select :sub :y)" mA (sel/select :sub :y) :zero? true)
(check-case "B sel/all"          mB sel/all              :zero? true)
(check-case "F sel/all"          mF sel/all              :zero? true)

(println "\n-- selected -> retained dependence (fast path legal, W nonzero) --")
;; The retained dependent site's (lp-new - lp-old) must appear identically on
;; both paths.
(check-case "C (select :sub), :y retained" mA (sel/select :sub))
(check-case "D (select :x), :sub retained" mB (sel/select :x))
(check-case "F (select :mu), :y retained"  mF (sel/select :mu))

(println "\n-- independent selections (fast path legal) --")
(check-case "E sel/all"        mE sel/all        :zero? true)
(check-case "E (select :x :y)" mE (sel/select :x :y) :zero? true)

(println "\n-- IIFE/letfn coupling (genmlx-7qdz) --")
(check-case "G sel/all"                  mG sel/all            :zero? true)
(check-case "G (select :a :y)"           mG (sel/select :a :y) :zero? true)
(check-case "G (select :a), :y retained" mG (sel/select :a))
(check-case "H sel/all"                  mH sel/all            :zero? true)
(check-case "H (select :a), :y retained" mH (sel/select :a))

(println "\n-- eligibility routing (gate must not degrade either direction) --")
(let [elig? @#'dyn/regen-fast-eligible?]
  (assert-true "A sel/all NOT fast-eligible"          (false? (elig? mA sel/all)))
  (assert-true "A (select :sub :y) NOT fast-eligible" (false? (elig? mA (sel/select :sub :y))))
  (assert-true "B sel/all NOT fast-eligible"          (false? (elig? mB sel/all)))
  (assert-true "F sel/all NOT fast-eligible"          (false? (elig? mF sel/all)))
  (assert-true "C (select :sub) fast-eligible"        (true?  (elig? mA (sel/select :sub))))
  (assert-true "D (select :x) fast-eligible"          (true?  (elig? mB (sel/select :x))))
  (assert-true "E sel/all fast-eligible"              (true?  (elig? mE sel/all)))
  (assert-true "F (select :mu) fast-eligible"         (true?  (elig? mF (sel/select :mu))))
  ;; genmlx-7qdz: the IIFE arg->param edge must gate the coupled selection...
  (assert-true "G sel/all NOT fast-eligible"          (false? (elig? mG sel/all)))
  (assert-true "G (select :a :y) NOT fast-eligible"   (false? (elig? mG (sel/select :a :y))))
  ;; ...without losing the fast path for the legal selected->retained case
  (assert-true "G (select :a) fast-eligible"          (true?  (elig? mG (sel/select :a))))
  ;; letfn tracer is an opaque escape: never fast, for any selection
  (assert-true "H (select :a) NOT fast-eligible"      (false? (elig? mH (sel/select :a))))
  (assert-true "H sel/all NOT fast-eligible"          (false? (elig? mH sel/all))))

(println "\n-- batched decline surface on spliced models (genmlx-89jo) --")
;; The njzu shape (site -> splice arg -> site) on the BATCHED path. A
;; non-fast-eligible selection on a spliced model must decline with the
;; documented ex-info — never a raw null deref (before the 0901fc6 gate fix,
;; coupled selections were wrongly fast-eligible and fell through the fast
;; path into one). Fast-eligible spliced selections keep working (8xia/20p7).
(def mI (gen [] (let [x (trace :x0 (dist/gaussian 0 1))
                      v (splice :sp0 subx x)]
                  (trace :x2 (dist/gaussian v 0.7))
                  v)))

(let [n  4
      vt (dyn/vsimulate mI [] n (rng/fresh-key 11))
      run (fn [selection]
            (try (do (dyn/vregenerate mI vt selection (rng/fresh-key 12)) :ok)
                 (catch :default e
                   (if (= :batched-regenerate-unsupported
                          (:genmlx/error (ex-data e)))
                     :loud-decline
                     :raw-crash))))]
  (assert-true "I coupled [[:x0] [:sp0 :z]]: loud decline, not a null deref"
               (= :loud-decline (run (sel/from-paths [[:x0] [:sp0 :z]]))))
  (assert-true "I sel/all: loud decline, not a null deref"
               (= :loud-decline (run sel/all)))
  (assert-true "I (select :x2): fast-eligible spliced vregenerate still works"
               (= :ok (run (sel/select :x2))))
  (assert-true "I [:sp0 :z]: sub-descending vregenerate still works"
               (= :ok (run (sel/select [:sp0 :z])))))

(println (str "\n== regen-gate: " @passes " passed, " @fails " failed =="))
(when (pos? @fails) (set! (.-exitCode js/process) 1))
