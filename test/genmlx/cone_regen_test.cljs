;; @tier fast
(ns genmlx.cone-regen-test
  "Cone-restricted incremental regenerate (genmlx-ltx2, SPEC v1).

   A single-site regenerate on a flat static model only changes the log-probs
   of the selected site and its DIRECT children (sites whose dist params read
   its value through deterministic code) — everything else cancels exactly.
   The cone path reuses the compiled per-site step-fns over just that cone,
   with the incremental score algebra score' = old - Σ lp_old(cone) + Σ
   lp_new(cone) and W = Σ_children (lp_new - lp_old).

   Contract under test (SPEC v1 invariants):
     I1 resampled values BIT-IDENTICAL to handler/compiled under the same key
        (retained sites never split keys, so the selected site's sample key is
        position-independent for single-site selections)
     I2 scores/weights within 1e-4 of handler + compiled (summation order
        differs — incremental vs full re-sum — so not bit-exact)
     I3 sweep drift bounded: after a 200-move MH sweep the incremental trace
        score stays within 1e-3 of a fresh full assess
     I4 every decline (multi-site, sel/all, branchy, non-schema addr) falls
        through to today's paths with unchanged results

   :direct-deps extraction (SPEC §2) is the load-bearing analysis: a
   trace-bound symbol contributes ONLY its own address (not its transitive
   closure), and a nested (trace :addr ...) call inside a form contributes
   only {addr} — its dist-arg reads belong to that SITE, not to the value."
  (:require [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.pef :as pef])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn- assert-true [label v]
  (if v (swap! pass inc)
        (do (swap! fail inc) (println "  FAIL" label))))

(defn- assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (< d tol) (swap! pass inc)
        (do (swap! fail inc)
            (println "  FAIL" label "expected" expected "actual" actual "d" d)))))

;; =========================================================================
;; Models
;; =========================================================================

(def chain3 (gen [] (let [a (trace :a (dist/gaussian 0 1))
                          b (trace :b (dist/gaussian a 1))
                          c (trace :c (dist/gaussian b 1))]
                      c)))
;; skip edge: c reads BOTH a (directly) and b — the case dep_graph's lossy
;; direct-parent reconstruction drops (its docstring admits skip-edge loss).
(def skip (gen [] (let [a (trace :a (dist/gaussian 0 1))
                        b (trace :b (dist/gaussian a 1))
                        c (trace :c (dist/gaussian (mx/add a b) 1))]
                    c)))
(def diamond (gen [] (let [a (trace :a (dist/gaussian 0 1))
                           b (trace :b (dist/gaussian a 1))
                           c (trace :c (dist/gaussian a 2))
                           d (trace :d (dist/gaussian (mx/add b c) 1))]
                       d)))
(def hub (gen [] (let [a (trace :a (dist/gaussian 0 1))
                       y1 (trace :y1 (dist/gaussian a 1))
                       y2 (trace :y2 (dist/gaussian a 1))
                       y3 (trace :y3 (dist/gaussian a 1))]
                   a)))
(def det-mid (gen [] (let [a (trace :a (dist/gaussian 0 1))
                           k (mx/exp a)
                           c (trace :c (dist/gaussian k 1))]
                       c)))
(def nested-lit (gen [] (trace :y (dist/gaussian (trace :mu (dist/gaussian 0 1)) 1))))
(def iife-m (gen [] (let [a (trace :a (dist/gaussian 0 1))]
                      ((fn [z] (trace :y (dist/gaussian z 1))) a))))
(def shadow-m (gen [] (let [a (trace :a (dist/gaussian 0 1))]
                        ((fn [a] (trace :y (dist/gaussian a 1))) (mx/scalar 3)))))
(def branchy (gen [x] (if x
                        (trace :a (dist/gaussian 0 1))
                        (trace :b (dist/gaussian 5 1)))))

(defn- chain-source
  "([] (let [x0 (trace :x0 (dist/gaussian 0 1)) x1 (trace :x1 (dist/gaussian x0 1)) ...] xN))"
  [t]
  (let [syms (mapv #(symbol (str "x" %)) (range t))
        addrs (mapv #(keyword (str "x" %)) (range t))
        bindings (vec (mapcat (fn [i]
                                [(syms i)
                                 (list 'trace (addrs i)
                                       (if (zero? i)
                                         (list 'dist/gaussian 0 1)
                                         (list 'dist/gaussian (syms (dec i)) 1)))])
                              (range t)))]
    (list [] (list 'let bindings (peek syms)))))

;; =========================================================================
;; 1. :direct-deps extraction (SPEC §2)
;; =========================================================================
(println "\n-- 1. :direct-deps extraction --")

(defn- dd [model addr]
  (:direct-deps (first (filter #(= addr (:addr %)) (:trace-sites (:schema model))))))
(defn- dchildren [model] (:direct-children (:schema model)))

(assert-true "chain: direct(c) = #{:b} (NOT transitive #{:a :b})"
             (= #{:b} (dd chain3 :c)))
(assert-true "chain: direct(b) = #{:a}" (= #{:a} (dd chain3 :b)))
(assert-true "chain: direct(a) = #{}" (= #{} (dd chain3 :a)))
(assert-true "chain: direct-children {:a #{:b} :b #{:c}}"
             (= {:a #{:b} :b #{:c}} (dchildren chain3)))
(assert-true "skip: direct(c) = #{:a :b} (skip edge kept)"
             (= #{:a :b} (dd skip :c)))
(assert-true "diamond: direct(d) = #{:b :c}" (= #{:b :c} (dd diamond :d)))
(assert-true "diamond: direct-children(a) = #{:b :c}"
             (= #{:b :c} (get (dchildren diamond) :a)))
(assert-true "hub: direct-children(a) = #{:y1 :y2 :y3}"
             (= #{:y1 :y2 :y3} (get (dchildren hub) :a)))
(assert-true "det-mid: direct(c) = #{:a} through (mx/exp a)"
             (= #{:a} (dd det-mid :c)))
(assert-true "nested literal: direct(y) = #{:mu}; :mu's dist-args do not leak"
             (= #{:mu} (dd nested-lit :y)))
(assert-true "IIFE: direct(y) = #{:a} through the param (7qdz analog)"
             (= #{:a} (dd iife-m :y)))
(assert-true "shadowing: direct(y) = #{} (param bound to constant)"
             (= #{} (dd shadow-m :y)))

;; =========================================================================
;; 2. :cone-regenerate attachment (SPEC §1 gates)
;; =========================================================================
(println "\n-- 2. attachment gates --")

(doseq [[nm m] [["chain3" chain3] ["skip" skip] ["diamond" diamond]
                ["hub" hub] ["det-mid" det-mid]]]
  (assert-true (str nm " has :cone-regenerate")
               (fn? (:cone-regenerate (:schema m)))))
(assert-true "branchy model has NO :cone-regenerate"
             (nil? (:cone-regenerate (:schema branchy))))

;; =========================================================================
;; 3. Four-way equivalence (SPEC §5 I1/I2): cone vs compiled-only vs
;;    handler-fast vs handler-general, same key
;; =========================================================================
(println "\n-- 3. single-site equivalence (cone / compiled / handler / general) --")

(defn- leaf [tr addr] (mx/item (cm/get-value (cm/get-submap (:choices tr) addr))))

(defn- regen-4way
  "Regenerate addr on model m from the same base trace under the same key via
   all four paths. Returns [{:w :score :val :tr} x4] in order
   [cone compiled handler general]."
  [m addr seed]
  (let [tr (p/simulate (dyn/with-key m (rng/fresh-key (* seed 7))) [])
        k  #(rng/fresh-key seed)
        run (fn [gf forced-general?]
              (let [r (if forced-general?
                        (binding [dyn/*force-general-regen* true]
                          (p/regenerate (dyn/with-key gf (k)) tr (sel/select addr)))
                        (p/regenerate (dyn/with-key gf (k)) tr (sel/select addr)))]
                {:w (mx/item (:weight r))
                 :score (mx/item (:score (:trace r)))
                 :val (leaf (:trace r) addr)
                 :tr (:trace r)}))]
    {:base tr
     :cone     (run m false)
     :compiled (run (update m :schema dissoc :cone-regenerate) false)
     :handler  (run (dyn/strip-alternate-paths m) false)
     :general  (run (dyn/strip-alternate-paths m) true)}))

(doseq [[nm m addrs] [["chain3" chain3 [:a :b :c]]
                      ["skip" skip [:a :b :c]]
                      ["diamond" diamond [:a :b :c :d]]
                      ["hub" hub [:a :y1 :y2 :y3]]
                      ["det-mid" det-mid [:a :c]]]
        addr addrs
        seed [11 22 33]]
  (let [{:keys [base cone compiled handler general]} (regen-4way m addr seed)]
    ;; I1: bit-identical resampled value across all four paths
    (assert-true (str nm " " addr " s" seed ": value cone==compiled==handler==general")
                 (and (= (:val cone) (:val compiled))
                      (= (:val cone) (:val handler))
                      (= (:val cone) (:val general))))
    ;; I2: weight and score within 1e-4
    (assert-close (str nm " " addr " s" seed ": weight cone~compiled")
                  (:w compiled) (:w cone) 1e-4)
    (assert-close (str nm " " addr " s" seed ": weight cone~handler")
                  (:w handler) (:w cone) 1e-4)
    (assert-close (str nm " " addr " s" seed ": weight cone~general")
                  (:w general) (:w cone) 1e-4)
    (assert-close (str nm " " addr " s" seed ": score cone~handler")
                  (:score handler) (:score cone) 1e-4)
    ;; retained sites unchanged
    (doseq [other (remove #{addr} (map :addr (:trace-sites (:schema m))))]
      (assert-true (str nm " " addr " s" seed ": retained " other " unchanged")
                   (= (leaf base other) (leaf (:tr cone) other))))))

;; =========================================================================
;; 4. Decline paths (SPEC §5 I4)
;; =========================================================================
(println "\n-- 4. decline paths fall through unchanged --")

(doseq [seed [5 6]]
  (let [tr (p/simulate (dyn/with-key chain3 (rng/fresh-key (* seed 7))) [])
        k #(rng/fresh-key seed)
        multi-cone (p/regenerate (dyn/with-key chain3 (k)) tr (sel/select :a :c))
        multi-hand (p/regenerate (dyn/with-key (dyn/strip-alternate-paths chain3) (k))
                                 tr (sel/select :a :c))]
    (assert-close (str "multi-site s" seed ": declined cone == handler weight")
                  (mx/item (:weight multi-hand)) (mx/item (:weight multi-cone)) 1e-4)
    (assert-true (str "multi-site s" seed ": values match handler")
                 (and (= (leaf (:trace multi-cone) :a) (leaf (:trace multi-hand) :a))
                      (= (leaf (:trace multi-cone) :c) (leaf (:trace multi-hand) :c))))))

(let [tr (p/simulate (dyn/with-key chain3 (rng/fresh-key 77)) [])
      r (p/regenerate (dyn/with-key chain3 (rng/fresh-key 78)) tr sel/all)]
  (assert-close "sel/all: full-prior resample weight is exactly 0"
                0.0 (mx/item (:weight r)) 1e-6))

(let [tr (p/simulate (dyn/with-key branchy (rng/fresh-key 79)) [true])
      r (p/regenerate (dyn/with-key branchy (rng/fresh-key 80)) tr (sel/select :a))]
  (assert-true "branchy model regenerate still works (no cone interference)"
               (js/isFinite (mx/item (:weight r)))))

;; nonexistent address: cone declines; result matches handler
(let [tr (p/simulate (dyn/with-key chain3 (rng/fresh-key 81)) [])
      rc (p/regenerate (dyn/with-key chain3 (rng/fresh-key 82)) tr (sel/select :nope))
      rh (p/regenerate (dyn/with-key (dyn/strip-alternate-paths chain3) (rng/fresh-key 82))
                       tr (sel/select :nope))]
  (assert-close "nonexistent addr: weight matches handler (0)"
                (mx/item (:weight rh)) (mx/item (:weight rc)) 1e-6))

;; =========================================================================
;; 5. MH sweep drift (SPEC §5 I3) on a generated T=50 chain
;; =========================================================================
(println "\n-- 5. MH sweep drift (T=50, 200 moves) --")

(let [t 50
      model (pef/source->model (chain-source t))]
  (assert-true "generated T=50 chain has :cone-regenerate"
               (fn? (:cone-regenerate (:schema model))))
  (let [final
        (loop [i 0
               tr (p/simulate (dyn/with-key model (rng/fresh-key 500)) [])]
          (if (= i 200)
            tr
            (let [addr (keyword (str "x" (mod i t)))
                  r (p/regenerate (dyn/with-key model (rng/fresh-key (+ 1000 i)))
                                  tr (sel/select addr))
                  w (mx/item (:weight r))
                  tr' (if (u/accept-mh? w (rng/fresh-key (+ 9000 i))) (:trace r) tr)]
              (mx/materialize! (:score tr'))
              (recur (inc i) tr'))))
        final-score (mx/item (:score final))
        fresh (mx/item (:weight (p/assess model [] (:choices final))))]
    (assert-true "final score finite" (js/isFinite final-score))
    (assert-close "I3: incremental score == fresh assess after 200 moves"
                  fresh final-score 1e-3)))

(println (str "\n== cone-regen: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
