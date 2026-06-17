;; @tier fast core
(ns genmlx.combinator-splice-general-regen-test
  "genmlx-1a23: a Map/Unfold/Scan combinator spliced inside a parent gen-fn (or
   serialize-round-tripped) loses its ::element-scores / ::step-scores metadata
   at the splice boundary (execute-sub reconstructs the child trace with only
   ::splice-scores). Before the fix the combinator regenerate fell into a
   metadata-loss branch that returned `Σ_i W_i - (:score trace)`. The
   -(:score trace) re-base ONLY undoes the FAST per-site path's ZERO-old
   inflation; for a GENERAL (project-based, retained-only) kernel each W_i is
   already correct, so the subtraction made the combinator (and the parent it
   feeds) wrong by exactly the old combinator score.

   The fix rehydrates the lost metadata by re-generating the trace fully
   constrained from its own choices, so regenerate takes its correct
   metadata-present path for both fast and general kernels.

   ORACLE (independent of the code under test): a prior-proposal regenerate
   weight equals Δ over the RETAINED leaf densities, i.e.
   project(new, retained) - project(old, retained). Validated below against a
   bare DynamicGF kernel whose weight has an independent hand-computed value.

   NOTE: per-element/per-step sites live UNDER an integer index, so the proper
   selection is `(from-paths [[i :addr] ...])`, NOT a flat `(select :addr)`
   (which descends to nothing per element — the reason the prior
   combinator_nested_regen_test asserted 0==0 vacuously)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.gfi :as gfi]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- num [x] (mx/eval! x) (mx/item x))
(defn- close? [a b] (< (js/Math.abs (- a b)) 1e-3))
(defn- idx-paths [n & addrs] (sel/from-paths (mapcat (fn [i] (map (fn [a] [i a]) addrs)) (range n))))
(defn- strip [k trace] (with-meta trace (dissoc (meta trace) k)))

(def ELEM ::comb/element-scores)
(def STEP ::comb/step-scores)

;; ── Oracle sanity: bare general-path kernel ────────────────────────────────
;; a→b→c chain; select {:a :b} (interdependent ⇒ general path), retain {:c}.
;; Hand value: lp(c0; b') - lp(c0; b0). Confirms delta-project is the right oracle.

(def chain
  (dyn/auto-key (gen [m] (let [a (trace :a (dist/gaussian m 1))
                               b (trace :b (dist/gaussian a 1))
                               c (trace :c (dist/gaussian b 1))]
                           c))))

(deftest bare-kernel-oracle-is-valid
  (is (not (#'dyn/regen-fast-eligible? chain (sel/select :a :b)))
      "chain + {:a :b} must force the general path (anti-vacuity)")
  (let [tr  (p/simulate (dyn/with-key chain (rng/fresh-key 7)) [0.0])
        cv  (fn [t a] (num (cm/get-value (cm/get-submap (:choices t) a))))
        r   (p/regenerate chain tr (sel/select :a :b))
        tr2 (:trace r)
        hand (- (num (dist/log-prob (dist/gaussian (mx/scalar (cv tr2 :b)) 1) (mx/scalar (cv tr :c))))
                (num (dist/log-prob (dist/gaussian (mx/scalar (cv tr :b)) 1)  (mx/scalar (cv tr :c)))))
        dproj (mx/subtract (p/project chain tr2 (sel/select :c))
                           (p/project chain tr  (sel/select :c)))]
    (is (not= (cv tr :b) (cv tr2 :b)) ":b resampled")
    (is (= (cv tr :c) (cv tr2 :c)) ":c retained")
    (is (close? (num (:weight r)) hand) "bare weight = hand density ratio")
    (is (close? (num (:weight r)) (num dproj)) "bare weight = delta-project")))

;; ── Generic checker: combinator regenerate weight = delta-project, both with
;;    metadata intact AND with it stripped (splice/serialize loss). ──────────

(defn- check-meta-loss [label comb meta-key sel-sites ret-sites args resampled-path]
  (let [comb (dyn/with-key comb (rng/fresh-key 7))
        dp   (fn [old new] (mx/subtract (p/project comb new ret-sites)
                                        (p/project comb old ret-sites)))
        tr   (p/simulate comb args)
        cval (fn [t path] (num (cm/get-value (reduce cm/get-submap (:choices t) path))))
        ;; metadata intact (control)
        r1   (p/regenerate comb tr sel-sites)
        ;; metadata stripped (the spliced / serialize-loss condition)
        tr0  (strip meta-key tr)
        r2   (p/regenerate comb tr0 sel-sites)]
    (testing (str label " — metadata intact")
      (is (not= (cval tr resampled-path) (cval (:trace r1) resampled-path))
          (str label " resampled a selected site (anti-vacuity)"))
      (is (close? (num (:weight r1)) (num (dp tr (:trace r1))))
          (str label " intact: weight = delta-project")))
    (testing (str label " — metadata stripped (splice / serialize loss)")
      (is (close? (num (:weight r2)) (num (dp tr0 (:trace r2))))
          (str label " stripped: weight = delta-project, got " (num (:weight r2))
               " vs oracle " (num (dp tr0 (:trace r2))))))))

;; general-path kernels (a→b chain forces general; the combinator state/carry is
;; decoupled from the resampled sites so each element/step is independent).
(def map-kernel    (dyn/auto-key (gfi/strip-compiled chain)))
(def unfold-kernel (dyn/auto-key (gfi/strip-compiled
                                   (gen [_t _prev]
                                        (let [a (trace :a (dist/gaussian 0 1))
                                              b (trace :b (dist/gaussian a 1))]
                                          (trace :c (dist/gaussian b 1))
                                          (mx/scalar 0.0))))))
(def scan-kernel   (dyn/auto-key (gfi/strip-compiled
                                   (gen [carry _input]
                                        (let [a (trace :a (dist/gaussian 0 1))
                                              b (trace :b (dist/gaussian a 1))]
                                          (trace :c (dist/gaussian b 1))
                                          [carry (mx/scalar 0.0)])))))

(deftest map-spliced-general-regen-weight
  (check-meta-loss "Map" (comb/map-combinator map-kernel) ELEM
                   (idx-paths 3 :a :b) (idx-paths 3 :c) [[0.0 1.0 2.0]] [0 :a]))

(deftest unfold-spliced-general-regen-weight
  (check-meta-loss "Unfold" (comb/unfold-combinator unfold-kernel) STEP
                   (idx-paths 3 :a :b) (idx-paths 3 :c) [3 (mx/scalar 0.0)] [0 :a]))

(deftest scan-spliced-general-regen-weight
  (check-meta-loss "Scan" (comb/scan-combinator scan-kernel) STEP
                   (idx-paths 3 :a :b) (idx-paths 3 :c)
                   [(mx/scalar 0.0) [(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]] [0 :a]))

;; ── End-to-end THROUGH A REAL SPLICE (the actual reported condition) ───────
;; A parent gen-fn that only splices the combinator; regenerate (hierarchical
;; :m <per-index>). execute-sub reconstructs the child without ::element-scores,
;; so the rehydration path is exercised for real (not a manual strip).

;; splice SPREADS its trailing args, so the Map's single arg (the input vector)
;; is passed directly — (splice :m mcomb [0 1 2]) gives the Map args [[0 1 2]].
(def parent
  (dyn/auto-key (gen [] (splice :m (comb/map-combinator map-kernel) [0.0 1.0 2.0]))))

(deftest map-through-real-splice-regen-weight
  ;; Exercises the rehydration through a REAL splice (execute-sub drops
  ;; ::element-scores). The genmlx-1a23 fix is verified end-to-end: the
  ;; through-splice regen weight is scalar AND equals the project oracle.
  (let [sel-ab (sel/hierarchical :m (idx-paths 3 :a :b))
        ret    (sel/hierarchical :m (idx-paths 3 :c))
        tr     (p/simulate parent [])
        {tr2 :trace w :weight} (p/regenerate parent tr sel-ab)
        oracle (mx/subtract (p/project parent tr2 ret) (p/project parent tr ret))]
    (is (= [] (mx/shape w)) "parent regen weight is scalar")
    (is (close? (num w) (num oracle))
        (str "through-splice weight = delta-project, got " (num w)
             " vs oracle " (num oracle)))))

(cljs.test/run-tests)
